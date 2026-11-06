"""Create Kaggle Part 1 CSV with no blank Predicted values."""
import argparse
import numpy as np
import pandas as pd
from config import LABELS, K, SUBMISSION_THRESHOLDS, OUTPUT_DIR
from data_utils import load_data, load_test
from models import predict_probabilities, predict_labels, evaluate_predictions
from pseudo_labeling import fit_kmeans_pseudo


def make_submission(test, probabilities, thresholds=SUBMISSION_THRESHOLDS, ensure_one=True):
    preds = predict_labels(probabilities, thresholds, ensure_one=ensure_one)
    predicted_labels = [" ".join(label for label, positive in zip(LABELS, row) if positive) for row in preds]
    submission = pd.DataFrame({"ID": test["ID"], "Predicted": predicted_labels})
    assert submission["ID"].notna().all() and submission["ID"].is_unique
    assert submission["Predicted"].notna().all()
    assert (submission["Predicted"].str.len() > 0).all(), "Kaggle rejects empty predictions."
    assert all(set(s.split()).issubset(set(LABELS)) for s in submission["Predicted"])
    return submission, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K, help="K-Means cluster count (default: 8)")
    args = parser.parse_args()
    data = load_data()
    result = fit_kmeans_pseudo(data, K=args.k)
    models = result["models"]
    thresholds = np.asarray(SUBMISSION_THRESHOLDS)
    val_probs = predict_probabilities(models, data.X_val)
    val_preds = predict_labels(val_probs, thresholds, ensure_one=True)
    print("Validation with Kaggle fallback:", evaluate_predictions("Final", data.y_val, val_preds))
    print("Validation rows needing fallback:", int((predict_labels(val_probs, thresholds).sum(axis=1) == 0).sum()))
    test, X_test = load_test(data.scaler)
    test_probs = predict_probabilities(models, X_test)
    submission, preds = make_submission(test, test_probs, thresholds)
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / "part1_submission.csv"
    submission.to_csv(path, index=False)
    print("Saved:", path)
    print("Rows:", len(submission), "Test rows needing fallback:", int((predict_labels(test_probs, thresholds).sum(axis=1) == 0).sum()))
    print("Positive counts:", dict(zip(LABELS, preds.sum(axis=0))))
    print(submission.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
