"""K-Means and GMM pseudo-label estimation, selection, and augmentation."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from config import RANDOM_STATE, LABELS, SMOOTHING, LOW_CONFIDENCE, HIGH_CONFIDENCE, MIN_LABELED_CLUSTER, PSEUDO_WEIGHT
from models import train_ovr


def estimate_cluster_labels(labeled_clusters, y_labeled, K, smoothing=SMOOTHING):
    prior = y_labeled.mean(axis=0)
    cluster_probs = np.zeros((K, len(LABELS)))
    cluster_counts = np.zeros(K, dtype=int)
    for k in range(K):
        mask = labeled_clusters == k
        count = int(mask.sum())
        cluster_counts[k] = count
        positive_counts = y_labeled[mask].sum(axis=0)
        cluster_probs[k] = (positive_counts + smoothing * prior) / (count + smoothing)
    return cluster_probs, cluster_counts


def estimate_soft_cluster_labels(memberships, y, smoothing=SMOOTHING):
    prior = y.mean(axis=0)
    counts = memberships.sum(axis=0)
    positives = memberships.T @ y
    probabilities = (positives + smoothing * prior) / (counts[:, None] + smoothing)
    return probabilities, counts


def augment_and_train(X_labeled, y_labeled, pseudo_X, pseudo_y, weight=PSEUDO_WEIGHT):
    X_aug = np.vstack([X_labeled, pseudo_X])
    y_aug = np.vstack([y_labeled, pseudo_y])
    weights = np.concatenate([np.ones(len(y_labeled)), np.full(len(pseudo_y), weight)])
    return train_ovr(X_aug, y_aug, sample_weight=weights)


def fit_kmeans_pseudo(data, K=8):
    km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
    cluster_ids = km.fit_predict(data.X_train)
    labeled_clusters = cluster_ids[data.labeled_mask]
    unlabeled_clusters = cluster_ids[data.unlabeled_mask]
    cluster_probs, cluster_counts = estimate_cluster_labels(labeled_clusters, data.y_labeled, K)
    pseudo_probs = cluster_probs[unlabeled_clusters]
    confident = ((pseudo_probs <= LOW_CONFIDENCE) | (pseudo_probs >= HIGH_CONFIDENCE)).all(axis=1)
    confident &= cluster_counts[unlabeled_clusters] >= MIN_LABELED_CLUSTER
    pseudo_y = (pseudo_probs[confident] >= HIGH_CONFIDENCE).astype(int)
    models = augment_and_train(data.X_labeled, data.y_labeled, data.X_unlabeled[confident], pseudo_y)
    return {
        "models": models, "clusterer": km, "cluster_probs": cluster_probs,
        "cluster_counts": cluster_counts, "unlabeled_clusters": unlabeled_clusters,
        "pseudo_probs": pseudo_probs, "confident": confident, "pseudo_y": pseudo_y,
    }


def fit_gmm_pseudo(data):
    gmm = GaussianMixture(n_components=8, covariance_type="diag", reg_covar=1e-4, random_state=RANDOM_STATE, n_init=3)
    gmm.fit(data.X_train)
    memberships = gmm.predict_proba(data.X_train)
    labeled_memberships = memberships[data.labeled_mask]
    unlabeled_memberships = memberships[data.unlabeled_mask]
    cluster_probs, counts = estimate_soft_cluster_labels(labeled_memberships, data.y_labeled)
    pseudo_probs = unlabeled_memberships @ cluster_probs
    confident = ((pseudo_probs <= LOW_CONFIDENCE) | (pseudo_probs >= HIGH_CONFIDENCE)).all(axis=1)
    support = unlabeled_memberships @ counts
    confident &= support >= MIN_LABELED_CLUSTER
    pseudo_y = (pseudo_probs[confident] >= HIGH_CONFIDENCE).astype(int)
    models = augment_and_train(data.X_labeled, data.y_labeled, data.X_unlabeled[confident], pseudo_y)
    return {"models": models, "clusterer": gmm, "confident": confident, "pseudo_y": pseudo_y}


def fit_per_label_kmeans(data, kmeans_result):
    from sklearn.linear_model import LogisticRegression
    pseudo_probs = kmeans_result["pseudo_probs"]
    valid = kmeans_result["cluster_counts"][kmeans_result["unlabeled_clusters"]] >= MIN_LABELED_CLUSTER
    models, counts = [], {}
    for j, reason in enumerate(LABELS):
        probs = pseudo_probs[:, j]
        confident = ((probs <= LOW_CONFIDENCE) | (probs >= HIGH_CONFIDENCE)) & valid
        pseudo_y = (probs[confident] >= HIGH_CONFIDENCE).astype(int)
        X_aug = np.vstack([data.X_labeled, data.X_unlabeled[confident]])
        y_aug = np.concatenate([data.y_labeled[:, j], pseudo_y])
        weights = np.concatenate([np.ones(len(data.y_labeled)), np.full(len(pseudo_y), PSEUDO_WEIGHT)])
        model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_aug, y_aug, sample_weight=weights)
        models.append(model)
        counts[reason] = {"Total": len(pseudo_y), "Positive": int(pseudo_y.sum()), "Negative": int((pseudo_y == 0).sum())}
    return models, counts
