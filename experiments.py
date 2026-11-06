"""Reproduce experiments A-H and optionally compare K-Means cluster counts."""
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import LABELS, K, RANDOM_STATE, SUBMISSION_THRESHOLDS, OUTPUT_DIR
from data_utils import load_data
from models import train_ovr, train_rbf_svm, predict_probabilities, predict_labels, evaluate, evaluate_predictions, tune_thresholds
from pseudo_labeling import fit_kmeans_pseudo, fit_gmm_pseudo, fit_per_label_kmeans, augment_and_train


def run_experiments(data, include_svm=True):
    results = []
    baseline = train_ovr(data.X_labeled, data.y_labeled)
    results.append(evaluate("A: Supervised baseline", baseline, data.X_val, data.y_val))

    km = fit_kmeans_pseudo(data, K=K)
    print("K-Means cluster probabilities:\n", pd.DataFrame(km["cluster_probs"], columns=LABELS))
    print("Labeled examples per cluster:", km["cluster_counts"])
    print("K-Means pseudo-labeled trials:", len(km["pseudo_y"]))
    results.append(evaluate("B: K-Means + LR", km["models"], data.X_val, data.y_val))

    gmm = fit_gmm_pseudo(data)
    print("GMM pseudo-labeled trials:", len(gmm["pseudo_y"]))
    results.append(evaluate("C: GMM + LR", gmm["models"], data.X_val, data.y_val))
    print("K-Means pseudo-label counts:\n", pd.DataFrame(km["pseudo_y"], columns=LABELS).sum())

    per_label_models, counts = fit_per_label_kmeans(data, km)
    print("Per-label pseudo-label counts:\n", pd.DataFrame(counts).T)
    results.append(evaluate("D: Per-label K-Means + LR", per_label_models, data.X_val, data.y_val))

    print("\nPredicted positive counts at threshold 0.5:")
    for name, models in [("Baseline", baseline), ("K-Means", km["models"]), ("GMM", gmm["models"]), ("Per-label", per_label_models)]:
        preds = predict_labels(predict_probabilities(models, data.X_val))
        print(name, dict(zip(LABELS, preds.sum(axis=0))))
    print("Actual:", dict(zip(LABELS, data.y_val.sum(axis=0))))

    dist_train = km["clusterer"].transform(data.X_labeled)
    dist_val = km["clusterer"].transform(data.X_val)
    distance_scaler = StandardScaler()
    X_labeled_ext = np.hstack([data.X_labeled, distance_scaler.fit_transform(dist_train)])
    X_val_ext = np.hstack([data.X_val, distance_scaler.transform(dist_val)])
    feature_models = train_ovr(X_labeled_ext, data.y_labeled)
    results.append(evaluate("E: K-Means Features + LR", feature_models, X_val_ext, data.y_val))

    rng = np.random.default_rng(RANDOM_STATE)
    n_pseudo = len(km["pseudo_y"])
    indices = rng.choice(len(data.X_unlabeled), size=n_pseudo, replace=False)
    # Preserve original Experiment F's Recruitment-only pseudo-label pattern.
    random_y = np.tile(np.array([0, 0, 0, 1, 0]), (n_pseudo, 1))
    random_models = augment_and_train(data.X_labeled, data.y_labeled, data.X_unlabeled[indices], random_y)
    results.append(evaluate("F: Random Pseudo-label Control", random_models, data.X_val, data.y_val))

    val_probs = predict_probabilities(km["models"], data.X_val)
    best_thresholds, per_label_f1 = tune_thresholds(val_probs, data.y_val)
    print("\nTuned thresholds:", dict(zip(LABELS, np.round(best_thresholds, 2))))
    print("Per-label tuned F1:", dict(zip(LABELS, np.round(per_label_f1, 3))))
    results.append(evaluate_predictions("G: K-Means + Tuned Thresholds", data.y_val, predict_labels(val_probs, best_thresholds)))
    # Kaggle submission uses a fallback for empty predictions; report it separately.
    results.append(evaluate_predictions("G: Tuned + Nonempty Fallback", data.y_val, predict_labels(val_probs, SUBMISSION_THRESHOLDS, ensure_one=True)))

    if include_svm:
        svm = train_rbf_svm(data.X_labeled, data.y_labeled)
        results.append(evaluate("H: RBF SVM", svm, data.X_val, data.y_val))

    comparison = pd.DataFrame(results).sort_values("Sample F1", ascending=False)
    OUTPUT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(OUTPUT_DIR / "experiment_results.csv", index=False)
    print("\nExperiment results:\n", comparison.round(4).to_string(index=False))
    return {"baseline": baseline, "kmeans": km, "gmm": gmm, "results": comparison, "thresholds": best_thresholds}


def compare_cluster_counts(data, ks=(4, 6, 8, 10, 12, 16)):
    rows = []
    for k in ks:
        result = fit_kmeans_pseudo(data, K=k)
        models = result["models"]
        for threshold_name, thresholds in [("default", 0.5), ("previously tuned", SUBMISSION_THRESHOLDS)]:
            metrics = evaluate(f"K={k}", models, data.X_val, data.y_val, thresholds=thresholds)
            rows.append({"K": k, "Thresholds": threshold_name, "Pseudo-labeled": len(result["pseudo_y"]), "Sample F1": metrics["Sample F1"], "Macro F1": metrics["Macro F1"]})
    table = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "cluster_count_comparison.csv", index=False)
    print("\nK-Means cluster count comparison:\n", table.round(4).to_string(index=False))
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune-k", action="store_true", help="Also compare K = 4, 6, 8, 10, 12, 16")
    parser.add_argument("--skip-svm", action="store_true", help="Skip the optional SVM experiment")
    args = parser.parse_args()
    data = load_data()
    print("Labeled trials:", len(data.X_labeled), "Unlabeled trials:", len(data.X_unlabeled))
    run_experiments(data, include_svm=not args.skip_svm)
    if args.tune_k:
        compare_cluster_counts(data)


if __name__ == "__main__":
    main()
