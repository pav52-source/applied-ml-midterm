"""Small behavioral checks for metrics, parsing, leakage boundaries, and submission."""
import unittest
import numpy as np
import pandas as pd
from .data import flatten, load_data, prepare
from .features import build_features
from .evaluation import evaluate
from .submission import validate_submission


class BaselineChecks(unittest.TestCase):
    def test_metrics(self):
        frame = pd.DataFrame({'phase': ['I', 'I', 'II', 'II', 'III', 'III'], 'label': [0, 1] * 3})
        p = np.array([0.2, 0.8] * 3)
        metrics = evaluate(frame, p)
        self.assertAlmostEqual(metrics['overall_log_loss'], -np.log(0.8))
        self.assertAlmostEqual(metrics['phase_macro_log_loss'], -np.log(0.8))
        for phase in ('I', 'II', 'III'):
            for metric in ('f1', 'roc_auc', 'average_precision'):
                self.assertEqual(metrics[f'{phase}_{metric}'], 1.0)
        metrics = evaluate(frame, np.full(6, 0.5))
        self.assertAlmostEqual(metrics['I_f1'], 2 / 3)  # >= 0.5 means positive
        self.assertAlmostEqual(metrics['I_roc_auc'], 0.5)
        self.assertAlmostEqual(metrics['I_average_precision'], 0.5)
        with self.assertRaises(ValueError):
            evaluate(frame, np.full(6, np.nan))

    def test_single_class(self):
        metrics = evaluate(pd.DataFrame({'phase': ['I'], 'label': [0]}), [0.1])
        self.assertAlmostEqual(metrics['I_log_loss'], -np.log(0.9))
        self.assertTrue(np.isnan(metrics['I_roc_auc']))

    def test_nested_lists(self):
        self.assertEqual(flatten('["[\'E11.65\', \'E11.9\']"]'), 'E11.65 E11.9')
        self.assertEqual(flatten(np.nan), '')
        self.assertEqual(flatten('[]'), '')

    def test_preprocessing_boundary(self):
        train = pd.DataFrame({'criteria': ['common trainonly'] * 4, 'phase': ['I'] * 4,
                              'start_year': [2000., 2002., np.nan, 2004.]})
        val = pd.DataFrame({'criteria': ['common validationonly'], 'phase': ['II'], 'start_year': [2020.]})
        features = build_features(['criteria'], True, True)
        features.fit(train)
        features.transform(val)
        self.assertNotIn('validationonly', features.named_transformers_['criteria'].vocabulary_)
        self.assertEqual(features.named_transformers_['year']['impute'].statistics_[0], 2002.)
        self.assertEqual(list(features.named_transformers_['phase'].categories_[0]), ['I'])

    def test_submission_and_allowlist(self):
        train, _, test, sample = load_data()
        x = prepare(train.assign(Safety=1, emb_0=2))
        self.assertNotIn('Safety', x)
        self.assertNotIn('emb_0', x)
        self.assertNotIn('label', x)
        self.assertNotIn('trial_id', x)
        validate_submission(sample, test, sample)
        with self.assertRaises(AssertionError):
            validate_submission(sample.iloc[::-1], test, sample)
        with self.assertRaises(AssertionError):
            validate_submission(sample.assign(label=1.1), test, sample)


if __name__ == '__main__':
    unittest.main()
