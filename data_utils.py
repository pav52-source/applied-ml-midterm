"""Read data and fit the embedding scaler using training trials only."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TRAIN_PATH, VAL_PATH, TEST_PATH, LABELS, FEATURES

@dataclass
class Dataset:
    train: pd.DataFrame
    val: pd.DataFrame
    scaler: StandardScaler
    X_train: np.ndarray
    X_val: np.ndarray
    X_labeled: np.ndarray
    X_unlabeled: np.ndarray
    y_labeled: np.ndarray
    y_val: np.ndarray
    labeled_mask: np.ndarray
    unlabeled_mask: np.ndarray


def load_data():
    train = pd.read_csv(TRAIN_PATH)
    val = pd.read_csv(VAL_PATH)
    labeled_mask = (train[LABELS] != -100).all(axis=1).to_numpy()
    unlabeled_mask = (train[LABELS] == -100).all(axis=1).to_numpy()
    if not np.all(labeled_mask | unlabeled_mask):
        raise ValueError("Some training rows have partially missing labels.")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[FEATURES])
    X_val = scaler.transform(val[FEATURES])
    return Dataset(
        train, val, scaler, X_train, X_val,
        X_train[labeled_mask], X_train[unlabeled_mask],
        train.loc[labeled_mask, LABELS].to_numpy(dtype=int),
        val[LABELS].to_numpy(dtype=int), labeled_mask, unlabeled_mask,
    )


def load_test(scaler):
    test = pd.read_csv(TEST_PATH)
    return test, scaler.transform(test[FEATURES])
