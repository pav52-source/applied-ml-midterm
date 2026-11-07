"""Lightweight integrated-project checks: no classifier training or calibration fitting."""
import hashlib
import json
import pickle
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from part2.data import load_data, prepare
from part2.training import predict
from part2.calibration import calibrated_predict, diagnostics, grouped_folds
from part2.evaluation import evaluate
from part2.submission import validate_submission
from integration.reconcile_reasons import authoritative_reasons

ROOT = Path(__file__).resolve().parents[1]


def verify():
    provenance = json.loads((ROOT / 'integration/provenance.json').read_text())
    for path, entry in provenance['files'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == entry['sha256'], path
    train, val, test, sample = load_data()
    lookup = authoritative_reasons(ROOT)
    assert lookup.reason_source.eq('ground_truth').sum() == 235
    reasons = ['Safety','Funding','Efficacy','Recruitment','Other']
    truth = pd.read_csv(ROOT / 'part1-data/part1_val.csv')
    p1 = pd.read_csv(ROOT / 'part1-results/knn-experiment/validation_kmeans_knn_hybrid_k3_uniform.csv')
    assert p1.ID.equals(truth.ID)
    p1_metrics = dict(sample_f1=f1_score(truth[reasons], p1[reasons], average='samples'),
                      macro_f1=f1_score(truth[reasons], p1[reasons], average='macro'))
    assert np.isclose(p1_metrics['sample_f1'], .8867521367521366)
    assert np.isclose(p1_metrics['macro_f1'], .7792343604108309)
    submission1 = pd.read_csv(ROOT / 'part1-results/knn-experiment/best_augmented_candidate_submission.csv')
    test1 = pd.read_csv(ROOT / 'part1-data/part1_test.csv')
    assert len(submission1) == 312 and submission1.ID.equals(test1.ID)
    assert list(submission1) == ['ID','Predicted']
    assert submission1.Predicted.notna().all() and submission1.Predicted.str.strip().ne('').all()
    assert all(set(s.split()) <= set(reasons) for s in submission1.Predicted)
    # Saved full pipeline uses only criteria and phase.
    with (ROOT / 'part2/outputs/best_model.pkl').open('rb') as handle:
        baseline = pickle.load(handle)
    inputs = baseline['models']['pooled']['features'].transformers
    assert [row[0] for row in inputs] == ['criteria','phase']
    assert set(prepare(train).columns).isdisjoint(set(reasons) | {'trial_id','ID','label'} | {f'emb_{i}' for i in range(150)})
    original = pd.read_csv(ROOT / 'part2/outputs/best_validation_predictions.csv')
    assert np.allclose(predict(baseline, prepare(val)), original.probability, atol=1e-14,rtol=0)
    expected = {'baseline':.5737533226117716, 'sigmoid_pooled':.5722087687860149,'sigmoid_by_phase':.5726097524662837}
    scored = []
    combined = pd.read_csv(ROOT / 'part2/calibration_outputs/validation_predictions.csv')
    for name, target in expected.items():
        if name == 'baseline':
            p = original.probability.to_numpy()
            path = ROOT / 'part2/outputs/part2_submission.csv'
        else:
            with (ROOT / 'part2/calibration_outputs' / f'{name}_model.pkl').open('rb') as handle:
                model = pickle.load(handle)
            p = calibrated_predict(model, val)
            path = ROOT / 'part2/calibration_outputs' / f'{name}_candidate_submission.csv'
            saved = combined.loc[combined.candidate.eq(name)]
            assert np.allclose(p,saved.probability,atol=1e-14,rtol=0)
        metric = evaluate(val,p)
        assert abs(metric['overall_log_loss']-target) < 1e-12
        submitted = pd.read_csv(path)
        validate_submission(submitted,test,sample)
        if name != 'baseline':
            assert np.allclose(calibrated_predict(model,test),submitted.label,atol=1e-14,rtol=0)
        scored.append(dict(candidate=name,log_loss=metric['overall_log_loss'],phase_macro_log_loss=metric['phase_macro_log_loss']))
    oof = pd.read_csv(ROOT / 'part2/calibration_outputs/training_oof_scores.csv')
    assert oof.ID.equals(train.ID) and oof.groupby('trial_id').fold.nunique().max() == 1
    for fold, (fit, held) in enumerate(grouped_folds(train)):
        assert not set(train.iloc[fit].trial_id) & set(train.iloc[held].trial_id)
        assert oof.iloc[held].fold.eq(fold).all()
    all_preds = pd.read_csv(ROOT / 'part2/outputs/validation_predictions.csv')
    all_metrics = pd.read_csv(ROOT / 'part2/outputs/experiment_results.csv').set_index('experiment')
    for name, group in all_preds.groupby('experiment'):
        m=evaluate(group,group.probability)
        assert np.isclose(m['overall_log_loss'],all_metrics.loc[name,'val_overall_log_loss'])
    assert len(all_metrics) == 37
    coverage = pd.read_csv(ROOT / 'integration/outputs/coverage.csv')
    assert coverage.n_reasons_unavailable.eq(0).all()
    resolved = pd.read_csv(ROOT / 'integration/outputs/resolved_previous_unknowns.csv')
    assert len(resolved)==131 and resolved.trial_id.nunique()==113
    assert resolved.reason_source.eq('prediction').all()
    # Original behavior tests, except the intentionally historical full-tree fingerprint test.
    from part2.test_baselines import BaselineChecks
    from part2.test_error_analysis import ReasonAnalysisChecks
    from part2.test_calibration import CalibrationChecks
    suite=unittest.TestSuite()
    for cls in (BaselineChecks,ReasonAnalysisChecks,CalibrationChecks):
        for name in unittest.defaultTestLoader.getTestCaseNames(cls):
            if name != 'test_saved_artifacts': suite.addTest(cls(name))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful()
    report=dict(source_files_verified=len(provenance['files']),part1_validation=p1_metrics,
        part2_validation=scored,original_behavior_tests_run=result.testsRun,
        tests_passed=result.wasSuccessful(),
        historical_test_not_run='part2.test_calibration.CalibrationChecks.test_saved_artifacts: asserts Michael-only root tree hashes; deliberately inapplicable to integration. Saved predictions, models, submissions and folds independently verified here.',
        original_37_experiments_verified=True,no_model_retraining=True,
        resolved_rows=131,resolved_unique_trials=113,remaining_unresolved=0,
        part1_submission_rows=312,part2_submission_rows=1500)
    (ROOT/'integration/outputs/verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    verify()
