"""Calibration behavior, group isolation, metrics and saved candidate checks."""
import unittest
import numpy as np
import pandas as pd
from scipy.special import expit
from .calibration import (diagnostics, grouped_folds, fit_sigmoids, apply_sigmoids,
                          calibrated_predict, CAL_OUTPUT)
from .data import load_data
from .submission import validate_submission


class CalibrationChecks(unittest.TestCase):
    def test_bins_and_brier(self):
        frame = pd.DataFrame({'phase': ['I', 'I', 'II', 'II', 'III', 'III'], 'label': [0, 1]*3})
        p = np.array([0., 1., .1, .9, .5, .5])
        metrics, bins = diagnostics(frame, p, 'test')
        overall = metrics.set_index('phase').loc['overall']
        self.assertAlmostEqual(overall.brier_score, (0+0+.01+.01+.25+.25)/6)
        self.assertEqual(bins.query("phase == 'overall'").n.sum(), 6)
        self.assertEqual(bins.query("phase == 'overall' and bin == 9").n.iloc[0], 2)
        self.assertEqual(bins.query("phase == 'overall' and bin == 1").n.iloc[0], 1)
        empty = bins.query("n == 0")
        self.assertTrue(empty.mean_probability.isna().all())
        self.assertTrue(empty.observed_success_rate.isna().all())

    def test_group_isolation(self):
        train, _, _, _ = load_data()
        for fit, held in grouped_folds(train):
            self.assertFalse(set(train.iloc[fit].trial_id) & set(train.iloc[held].trial_id))
            for phase in ('I', 'II', 'III'):
                self.assertEqual(set(train.iloc[fit].loc[train.iloc[fit].phase.eq(phase), 'label']), {0, 1})

    def test_known_sigmoid(self):
        # Replicated deterministic frequencies follow expit(0.5*s - 0.3).
        grid = np.linspace(-3, 3, 21)
        scores = np.repeat(grid, 1000)
        labels = np.concatenate([np.r_[np.ones(round(expit(.5*s-.3)*1000)),
                                              np.zeros(1000-round(expit(.5*s-.3)*1000))] for s in grid])
        phases = np.full(len(scores), 'I')
        models = fit_sigmoids(scores, labels, phases)
        self.assertAlmostEqual(models['pooled'].coef_[0, 0], .5, places=2)
        self.assertAlmostEqual(models['pooled'].intercept_[0], -.3, places=2)
        p = apply_sigmoids(models, grid, np.full(len(grid), 'I'))
        self.assertTrue((np.diff(p) > 0).all())
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_saved_artifacts(self):
        if not (CAL_OUTPUT / 'metadata.json').exists():
            self.skipTest('Run python -m part2.calibration first')
        import json
        import pickle
        import hashlib
        from .config import ROOT
        meta = json.loads((CAL_OUTPUT / 'metadata.json').read_text())
        for path, digest in meta['protected_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)
        train, _, test, sample = load_data()
        scores = pd.read_csv(CAL_OUTPUT / 'training_oof_scores.csv')
        self.assertTrue(scores.ID.equals(train.ID))
        self.assertEqual(scores.groupby('trial_id').fold.nunique().max(), 1)
        self.assertTrue(np.isfinite(scores.decision_score).all())
        metrics = pd.read_csv(CAL_OUTPUT / 'validation_metrics.csv')
        for phase in ('I', 'II', 'III'):
            selected = metrics[metrics.phase.eq(phase)]
            for metric in ('roc_auc', 'average_precision'):
                self.assertTrue(np.allclose(selected[metric], selected[metric].iloc[0], atol=1e-12))
        for name in meta['candidate_submissions']:
            submission = pd.read_csv(CAL_OUTPUT / name)
            validate_submission(submission, test, sample)
            method = name.removesuffix('_candidate_submission.csv')
            with (CAL_OUTPUT / f'{method}_model.pkl').open('rb') as f:
                model = pickle.load(f)
            self.assertTrue(np.allclose(calibrated_predict(model, test), submission.label, atol=1e-14, rtol=0))


if __name__ == '__main__':
    unittest.main()
