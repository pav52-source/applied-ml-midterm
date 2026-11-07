"""Generate and independently validate an ordered Kaggle probability CSV."""
import pickle
import numpy as np
import pandas as pd
from .config import OUTPUT
from .data import load_data, prepare
from .training import predict


def validate_submission(submission, test, sample):
    assert list(submission.columns) == ['ID', 'label']
    assert len(submission) == len(test) == 1500
    assert submission.ID.equals(test.ID) and submission.ID.equals(sample.ID)
    assert submission.ID.is_unique and not submission.isna().any().any()
    assert np.isfinite(submission.label).all() and submission.label.between(0, 1).all()


def generate(bundle, test, sample):
    submission = pd.DataFrame({'ID': test.ID, 'label': predict(bundle, prepare(test))})
    validate_submission(submission, test, sample)
    path = OUTPUT / 'part2_submission.csv'
    submission.to_csv(path, index=False)
    validate_submission(pd.read_csv(path), test, sample)
    return path


if __name__ == '__main__':
    _, _, test, sample = load_data()
    # Load only this locally generated, trusted pickle.
    with (OUTPUT / 'best_model.pkl').open('rb') as f:
        bundle = pickle.load(f)
    print(generate(bundle, test, sample))
