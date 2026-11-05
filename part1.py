
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 42

LABELS = [
    "Safety", "Funding", "Efficacy",
    "Recruitment", "Other"
]

FEATURES = [f"emb_{i}" for i in range(150)]

train = pd.read_csv("part1-data/part1_train.csv")
val = pd.read_csv("part1-data/part1_val.csv")

# -100 means all five labels are missing.
labeled_mask = (train[LABELS] != -100).all(axis=1)
unlabeled_mask = (train[LABELS] == -100).all(axis=1)

assert (labeled_mask | unlabeled_mask).all()

print("Labeled trials:", labeled_mask.sum())
print("Unlabeled trials:", unlabeled_mask.sum())

# Fit preprocessing on training data only.
scaler = StandardScaler()
X_train = scaler.fit_transform(train[FEATURES])
X_val = scaler.transform(val[FEATURES])

X_labeled = X_train[labeled_mask.to_numpy()]
X_unlabeled = X_train[unlabeled_mask.to_numpy()]

y_labeled = train.loc[labeled_mask, LABELS].to_numpy(dtype=int)
y_val = val[LABELS].to_numpy(dtype=int)

def train_ovr(X, y, sample_weight=None):
    """Train one binary Logistic Regression per reason."""
    models = []

    for j in range(y.shape[1]):
        model = LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        model.fit(
            X, y[:, j],
            sample_weight=sample_weight
        )
        models.append(model)

    return models


def predict_probabilities(models, X):
    """Return five probabilities per trial."""
    return np.column_stack([
        model.predict_proba(X)[:, 1]
        for model in models
    ])


def evaluate(name, models, X, y):
    probabilities = predict_probabilities(models, X)
    predictions = (probabilities >= 0.5).astype(int)

    result = {
        "Experiment": name,
        "Sample F1": f1_score(
            y, predictions, average="samples",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y, predictions, average="macro",
            zero_division=0
        )
    }

    for j, reason in enumerate(LABELS):
        result[reason] = f1_score(
            y[:, j], predictions[:, j],
            zero_division=0
        )

    return result

# Experiment A : simply train the model on the existing labeled data, and predict
# the missing labels.
baseline_model = train_ovr(
    X_labeled,
    y_labeled
)

results = []

results.append(
    evaluate(
        "A: Supervised baseline",
        baseline_model,
        X_val,
        y_val
    )
)

print()
for key, val in results[0].items():
    print(key + " " + str(val))

print()
print("Experiment B: ")
############
# Start of experiment B
K = 8  # Starting hypothesis, not necessarily optimal

kmeans = KMeans(
    n_clusters=K,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_ids = kmeans.fit_predict(X_train)

labeled_clusters = cluster_ids[labeled_mask.to_numpy()]
unlabeled_clusters = cluster_ids[unlabeled_mask.to_numpy()]


def estimate_cluster_labels(
    labeled_clusters, y_labeled, K, smoothing=5.0
):
    # Overall prevalence of each reason
    prior = y_labeled.mean(axis=0)

    cluster_probs = np.zeros((K, len(LABELS)))
    cluster_counts = np.zeros(K, dtype=int)

    for k in range(K):
        mask = labeled_clusters == k
        count = mask.sum()
        cluster_counts[k] = count

        positive_counts = y_labeled[mask].sum(axis=0)

        # Shrink small clusters toward global prevalence
        cluster_probs[k] = (
            positive_counts + smoothing * prior
        ) / (count + smoothing)

    return cluster_probs, cluster_counts


cluster_probs, cluster_counts = estimate_cluster_labels(
    labeled_clusters, y_labeled, K
)

print(pd.DataFrame(cluster_probs, columns=LABELS))
print("Labeled examples per cluster:", cluster_counts)


kmeans_probs = cluster_probs[unlabeled_clusters]

confident = (
    (kmeans_probs <= 0.2) |
    (kmeans_probs >= 0.8)
).all(axis=1)

# Require enough labeled trials in the cluster
confident &= cluster_counts[unlabeled_clusters] >= 5

pseudo_X = X_unlabeled[confident]
pseudo_y = (
    kmeans_probs[confident] >= 0.8
).astype(int)

print("K-Means pseudo-labeled trials:", len(pseudo_y))


X_aug_kmeans = np.vstack([
    X_labeled, pseudo_X
])

y_aug_kmeans = np.vstack([
    y_labeled, pseudo_y
])

weights_kmeans = np.concatenate([
    np.ones(len(y_labeled)),
    np.full(len(pseudo_y), 0.3)
])

kmeans_model = train_ovr(
    X_aug_kmeans,
    y_aug_kmeans,
    sample_weight=weights_kmeans
)

results.append(
    evaluate(
        "B: K-Means + LR",
        kmeans_model,
        X_val,
        y_val
    )
)

print()
print("Experiment C: ")
#############
# Start of experiment C

gmm = GaussianMixture(
    n_components=8,
    covariance_type="diag",
    reg_covar=1e-4,
    random_state=RANDOM_STATE,
    n_init=3
)

gmm.fit(X_train)

# Each row contains probabilities across 8 clusters
memberships = gmm.predict_proba(X_train)

labeled_memberships = memberships[
    labeled_mask.to_numpy()
]

unlabeled_memberships = memberships[
    unlabeled_mask.to_numpy()
]


def estimate_soft_cluster_labels(
    memberships, y, smoothing=5.0
):
    prior = y.mean(axis=0)

    # Effective labeled sample count per cluster
    counts = memberships.sum(axis=0)

    # Weighted positive counts for each reason
    positives = memberships.T @ y

    probabilities = (
        positives + smoothing * prior
    ) / (counts[:, None] + smoothing)

    return probabilities, counts


gmm_cluster_probs, gmm_counts = (
    estimate_soft_cluster_labels(
        labeled_memberships, y_labeled
    )
)

# Weighted average of cluster reason probabilities
gmm_probs = (
    unlabeled_memberships @ gmm_cluster_probs
)


gmm_confident = (
    (gmm_probs <= 0.2) |
    (gmm_probs >= 0.8)
).all(axis=1)

# Require sufficient effective labeled support
support = unlabeled_memberships @ gmm_counts
gmm_confident &= support >= 5

gmm_pseudo_X = X_unlabeled[gmm_confident]
gmm_pseudo_y = (
    gmm_probs[gmm_confident] >= 0.8
).astype(int)

print("GMM pseudo-labeled trials:", len(gmm_pseudo_y))

X_aug_gmm = np.vstack([
    X_labeled, gmm_pseudo_X
])

y_aug_gmm = np.vstack([
    y_labeled, gmm_pseudo_y
])

weights_gmm = np.concatenate([
    np.ones(len(y_labeled)),
    np.full(len(gmm_pseudo_y), 0.3)
])

gmm_model = train_ovr(
    X_aug_gmm,
    y_aug_gmm,
    sample_weight=weights_gmm
)

results.append(
    evaluate(
        "C: GMM + LR",
        gmm_model,
        X_val,
        y_val
    )
)


comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    "Sample F1",
    ascending=False
)

print()
print("Final Results: ")
print(comparison.round(3).to_string(index=False))

print("Pseudo-label counts:")
print(pd.DataFrame(pseudo_y, columns=LABELS).sum())

print()
print("\nPseudo-label proportions:")
print(pd.DataFrame(pseudo_y, columns=LABELS).mean())


# Probabilities from our existing K-Means experiment
kmeans_probs = cluster_probs[unlabeled_clusters]

# Only trust clusters with sufficient labeled examples
valid_cluster = (
    cluster_counts[unlabeled_clusters] >= 5
)

per_label_models = []
pseudo_counts = {}

for j, reason in enumerate(LABELS):

    probabilities = kmeans_probs[:, j]

    # Select confident examples for this reason only
    confident = (
        (probabilities <= 0.2) |
        (probabilities >= 0.8)
    ) & valid_cluster

    X_pseudo = X_unlabeled[confident]

    y_pseudo = (
        probabilities[confident] >= 0.8
    ).astype(int)

    pseudo_counts[reason] = {
        "Total": len(y_pseudo),
        "Positive": int(y_pseudo.sum()),
        "Negative": int((y_pseudo == 0).sum())
    }

    # Combine real and pseudo-labeled examples
    X_aug = np.vstack([
        X_labeled,
        X_pseudo
    ])

    y_aug = np.concatenate([
        y_labeled[:, j],
        y_pseudo
    ])

    # Give pseudo-labels less influence
    weights = np.concatenate([
        np.ones(len(y_labeled)),
        np.full(len(y_pseudo), 0.3)
    ])

    model = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    model.fit(
        X_aug,
        y_aug,
        sample_weight=weights
    )

    per_label_models.append(model)

print(pd.DataFrame(pseudo_counts).T)


new_result = evaluate(
    "D: Per-label K-Means + LR",
    per_label_models,
    X_val,
    y_val
)

results.append(new_result)

comparison = pd.DataFrame(results)

print(
    comparison.sort_values(
        "Sample F1",
        ascending=False
    ).round(3).to_string(index=False)
)


models = {
    "Baseline": baseline_model,
    "K-Means": kmeans_model,
    "GMM": gmm_model,
    "Per-label": per_label_models
}

for name, model in models.items():
    probs = predict_probabilities(model, X_val)
    preds = (probs >= 0.5).astype(int)

    print(f"\n{name}")
    print(pd.Series(
        preds.sum(axis=0),
        index=LABELS
    ))

print("\nActual positive labels in validation:")
print(pd.Series(
    y_val.sum(axis=0),
    index=LABELS
))


# Experiment E:
# Use distances to K-Means cluster centers
# as additional supervised features.

from sklearn.preprocessing import StandardScaler

# kmeans was fitted on X_train in Experiment B

# Distances from each trial to each cluster center
dist_train = kmeans.transform(X_labeled)
dist_val = kmeans.transform(X_val)

# Standardize the new distance features
distance_scaler = StandardScaler()

dist_train_scaled = distance_scaler.fit_transform(
    dist_train
)

dist_val_scaled = distance_scaler.transform(
    dist_val
)

# Combine original embeddings with cluster distances
X_labeled_extended = np.hstack([
    X_labeled,
    dist_train_scaled
])

X_val_extended = np.hstack([
    X_val,
    dist_val_scaled
])

print(
    "Original features:",
    X_labeled.shape[1]
)

print(
    "Extended features:",
    X_labeled_extended.shape[1]
)

# Train five supervised classifiers
kmeans_feature_models = train_ovr(
    X_labeled_extended,
    y_labeled
)

# Evaluate
result_E = evaluate(
    "E: K-Means Features + LR",
    kmeans_feature_models,
    X_val_extended,
    y_val
)

results.append(result_E)

print(
    pd.DataFrame(results)
    .sort_values("Sample F1", ascending=False)
    .round(3)
    .to_string(index=False)
)


# Experiment F: Random-selection control
# Use the same number and type of pseudo-labels
# as Experiment B, but select random unlabeled trials.

rng = np.random.default_rng(RANDOM_STATE)

n_pseudo = 285

random_indices = rng.choice(
    len(X_unlabeled),
    size=n_pseudo,
    replace=False
)

X_random = X_unlabeled[random_indices]

# Same pseudo-label pattern as Experiment B
y_random = np.tile(
    np.array([0, 0, 0, 1, 0]),
    (n_pseudo, 1)
)

X_aug = np.vstack([
    X_labeled,
    X_random
])

y_aug = np.vstack([
    y_labeled,
    y_random
])

weights = np.concatenate([
    np.ones(len(X_labeled)),
    np.full(n_pseudo, 0.3)
])

random_models = train_ovr(
    X_aug,
    y_aug,
    sample_weight=weights
)

result_F = evaluate(
    "F: Random Pseudo-label Control",
    random_models,
    X_val,
    y_val
)

results.append(result_F)

print(
    pd.DataFrame(results)
    .sort_values("Sample F1", ascending=False)
    .round(3)
    .to_string(index=False)
)


from sklearn.metrics import f1_score
import numpy as np

# Use our current best K-Means model
val_probs = predict_probabilities(
    kmeans_model, X_val
)

# Tune each label's threshold separately
candidate_thresholds = np.arange(0.20, 0.81, 0.05)

best_thresholds = []

for j, reason in enumerate(LABELS):
    scores = []

    for threshold in candidate_thresholds:
        predictions = (
            val_probs[:, j] >= threshold
        ).astype(int)

        score = f1_score(
            y_val[:, j],
            predictions,
            zero_division=0
        )

        scores.append(score)

    best_idx = np.argmax(scores)
    best_thresholds.append(
        candidate_thresholds[best_idx]
    )

    print(
        reason,
        "best threshold:",
        round(candidate_thresholds[best_idx], 2),
        "F1:",
        round(scores[best_idx], 3)
    )

# Evaluate all five thresholds together
best_thresholds = np.array(best_thresholds)

tuned_predictions = (
    val_probs >= best_thresholds
).astype(int)

print(
    "\nSample F1:",
    f1_score(
        y_val,
        tuned_predictions,
        average="samples",
        zero_division=0
    )
)

print(
    "Macro F1:",
    f1_score(
        y_val,
        tuned_predictions,
        average="macro",
        zero_division=0
    )
)
