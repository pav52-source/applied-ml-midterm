import unittest
import numpy as np
import pandas as pd
from .error_analysis import (REASONS, NO_REASON, build_reason_table,
                             match_unsuccessful, summarize_groups)


class ReasonAnalysisChecks(unittest.TestCase):
    def test_provenance_unknowns_multilabel_threshold_and_empty_groups(self):
        def frame(ids, labels=None):
            result = pd.DataFrame({'ID': ids})
            if labels is not None:
                result[list(REASONS)] = labels
            return result
        parts = {
            'train': frame(['TR00001', 'TR00002'], [[1, 0, 0, 1, 0], [-100] * 5]),
            'val': frame(['TR00003'], [[0, 1, 0, 0, 0]]),
            'test': frame(['TR00004']),
        }
        reasons = build_reason_table(parts, pd.DataFrame({'ID': ['TR00004'], 'Predicted': ['Safety Other']}))
        self.assertEqual(reasons.set_index('trial_id').loc['TR00002', 'reason_source'], 'unavailable')
        self.assertTrue(reasons.set_index('trial_id').loc['TR00002', list(REASONS)].isna().all())
        validation = pd.DataFrame({
            'trial_id': ['TR00001', 'TR00002', 'TR00004', 'TR00005', 'TR00001', 'TR00003'],
            'phase': ['I', 'I', 'II', 'II', 'III', 'III'],
            'label': [0, 0, 0, 0, 0, 1], 'probability': [.5, .9, .3, .8, .2, .7],
        })
        validation['ID'] = validation.trial_id + '_' + validation.phase
        validation['predicted_label'] = (validation.probability >= .5).astype(int)
        joined = match_unsuccessful(validation, validation[['ID', 'trial_id', 'phase', 'label']], reasons)
        groups, coverage, unresolved, sources = summarize_groups(joined)
        groups = groups.set_index(['phase', 'reason_group'])
        self.assertEqual(len(joined), 5)  # successful outcome excluded
        self.assertEqual(groups.loc[('I', 'Safety'), 'n_predicted_success'], 1)  # threshold inclusive
        self.assertEqual(groups.loc[('I', 'Recruitment'), 'n_trials'], 1)  # multilabel membership
        self.assertEqual(groups.loc[('III', 'Safety'), 'n_trials'], 1)  # combined phase counted once each
        self.assertEqual(groups.loc[('I', NO_REASON), 'n_trials'], 0)  # masked is not no record
        self.assertEqual(groups.loc[('II', NO_REASON), 'n_trials'], 1)
        self.assertTrue(np.isnan(groups.loc[('I', 'Efficacy'), 'false_success_rate']))
        self.assertTrue(np.isnan(groups.loc[('I', 'Efficacy'), 'mean_predicted_success_probability']))
        self.assertEqual(groups.loc[('II', 'Safety'), 'n_predictions'], 1)
        self.assertEqual(unresolved.n_trials.sum(), 1)
        self.assertEqual(coverage.n_trials.sum(), 5)
        self.assertAlmostEqual(groups.loc[('II', NO_REASON), 'mean_predicted_success_probability'], .8)


if __name__ == '__main__':
    unittest.main()
