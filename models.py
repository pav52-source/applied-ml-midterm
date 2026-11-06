"""Reusable classifiers, prediction helpers, and validation metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from config import LABELS, RANDOM_STATE


def train_ovr(X, y, sample_weight=None):
    models = []
    for j in range(y.shape[1]):
        model = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X, y[:, j], sample_weight=sample_weight)
        models.append(model)
    return models


def train_rbf_svm(X, y):
    models = []
    for j in range(y.shape[1]):
        model = SVC(kernel="rbf", C=1, gamma="scale", class_weight="balanced", probability=True, random_state=RANDOM_STATE)
        model.fit(X, y[:, j])
        models.append(model)
    return models


def predict_probabilities(models, X):
    return np.column_stack([model.predict_proba(X)[:, 1] for model in models])


def predict_labels(probabilities, thresholds=0.5, ensure_one=False):
    predictions = (probabilities >= np.asarray(thresholds)).astype(int)
    if ensure_one:
        empty = np.flatnonzero(predictions.sum(axis=1) == 0)
        predictions[empty, np.argmax(probabilities[empty], axis=1)] = 1
    return predictions


def evaluate_predictions(name, y, predictions):
    result = {
        "Experiment": name,
        "Sample F1": f1_score(y, predictions, average="samples", zero_division=0),
        "Macro F1": f1_score(y, predictions, average="macro", zero_division=0),
    }
    for j, reason in enumerate(LABELS):
        result[reason] = f1_score(y[:, j], predictions[:, j], zero_division=0)
    return result


def evaluate(name, models, X, y, thresholds=0.5, ensure_one=False):
    probs = predict_probabilities(models, X)
    return evaluate_predictions(name, y, predict_labels(probs, thresholds, ensure_one))


def tune_thresholds(probs, y, candidates=None):
    if candidates is None:
        candidates = np.arange(0.20, 0.81, 0.05)
    thresholds = []
    per_label_scores = []
    for j in range(y.shape[1]):
        scores = [f1_score(y[:, j], probs[:, j] >= t, zero_division=0) for t in candidates]
        idx = int(np.argmax(scores))
        thresholds.append(float(candidates[idx]))
        per_label_scores.append(float(scores[idx]))
    return np.asarray(thresholds), per_label_scores
