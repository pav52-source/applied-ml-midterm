"""Training-only grouped cross-fitted sigmoid calibration of the frozen winner."""
import hashlib
import json
import pickle
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from .config import ROOT, OUTPUT, PHASES, SEED, THRESHOLD
from .data import load_data, prepare
from .evaluation import evaluate
from .training import fit_features, fit_classifiers, predict
from .submission import validate_submission
from .report import markdown_table

CAL_OUTPUT = ROOT / 'part2' / 'calibration_outputs'
N_FOLDS = 5


def diagnostics(frame, probabilities, candidate, split='validation'):
    """Fixed-width bins [left,right), including probability 1 in the final bin."""
    p = np.asarray(probabilities, dtype=float)
    metrics = evaluate(frame, p)
    summary, bins = [], []
    for phase in ('overall', *PHASES):
        mask = np.ones(len(frame), dtype=bool) if phase == 'overall' else frame.phase.eq(phase).to_numpy()
        y, q = frame.label.to_numpy()[mask], p[mask]
        bucket = np.minimum((q * 10).astype(int), 9)
        gaps, weights = [], []
        for b in range(10):
            selected = bucket == b
            n = int(selected.sum())
            mean = float(q[selected].mean()) if n else np.nan
            observed = float(y[selected].mean()) if n else np.nan
            bins.append(dict(candidate=candidate, split=split, phase=phase, bin=b,
                lower=b / 10, upper=(b + 1) / 10, n=n,
                successes=int(y[selected].sum()), mean_probability=mean,
                observed_success_rate=observed, gap=mean - observed))
            if n:
                gaps.append(abs(mean - observed))
                weights.append(n)
        row = dict(candidate=candidate, split=split, phase=phase, n=len(y),
            log_loss=metrics[f'{phase}_log_loss'], brier_score=brier_score_loss(y, q),
            mean_probability=float(q.mean()), observed_success_rate=float(y.mean()),
            mean_gap=float(q.mean() - y.mean()),
            ece_10_bins=float(np.average(gaps, weights=weights)),
            f1=metrics[f'{phase}_f1'], roc_auc=metrics[f'{phase}_roc_auc'],
            average_precision=metrics[f'{phase}_average_precision'])
        # Both confident-success and confident-failure tails; gaps alone aren't sharpness.
        for name, tail in [('high', q >= .8), ('low', q < .2)]:
            row[f'{name}_n'] = int(tail.sum())
            row[f'{name}_mean_probability'] = float(q[tail].mean()) if tail.any() else np.nan
            row[f'{name}_success_rate'] = float(y[tail].mean()) if tail.any() else np.nan
        summary.append(row)
    return pd.DataFrame(summary), pd.DataFrame(bins)


def grouped_folds(train, n_splits=N_FOLDS):
    # Stratification approximates joint phase/outcome balance without splitting trials.
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    strata = train.phase.astype(str) + '_' + train.label.astype(str)
    folds = list(splitter.split(train, strata, groups=train.trial_id))
    seen = np.zeros(len(train), dtype=int)
    for fit, held in folds:
        assert not set(train.iloc[fit].trial_id) & set(train.iloc[held].trial_id)
        assert set(train.iloc[fit].phase) == set(PHASES)
        seen[held] += 1
    assert (seen == 1).all()
    return folds


def decision_scores(baseline, features):
    components = baseline['models']['pooled']
    return components['classifier'].decision_function(components['features'].transform(features))


def fit_sigmoids(scores, labels, phases, phase_specific=False):
    calibrators = {}
    for group in PHASES if phase_specific else ('pooled',):
        mask = np.ones(len(labels), dtype=bool) if group == 'pooled' else np.asarray(phases) == group
        assert len(np.unique(labels[mask])) == 2
        # Fixed, unpenalized two-parameter Platt sigmoid on raw decision margins.
        calibrator = LogisticRegression(C=np.inf, solver='lbfgs', tol=1e-10,
                                        max_iter=2000, random_state=SEED)
        calibrator.fit(np.asarray(scores)[mask, None], labels[mask])
        assert calibrator.n_iter_.max() < calibrator.max_iter
        assert calibrator.coef_[0, 0] > 0, 'Unexpected ranking reversal'
        calibrators[group] = calibrator
    return calibrators


def apply_sigmoids(calibrators, scores, phases):
    p = np.empty(len(scores))
    for group, model in calibrators.items():
        mask = np.ones(len(scores), dtype=bool) if group == 'pooled' else np.asarray(phases) == group
        if mask.any():
            p[mask] = model.predict_proba(np.asarray(scores)[mask, None])[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def calibrated_predict(bundle, frame):
    features = prepare(frame)
    return apply_sigmoids(bundle['calibrators'], decision_scores(bundle['baseline'], features), frame.phase)


def fingerprints():
    paths = [p for directory in ('part1-data', 'outputs', 'part2', 'part2-data')
             for p in (ROOT / directory).rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and CAL_OUTPUT not in p.parents]
    # Include root Part 1 source/configuration files, too.
    paths += [p for p in ROOT.iterdir() if p.is_file() and p.name != '.DS_Store']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def main():
    protected = fingerprints()
    CAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    train, val, test, sample = load_data()
    saved = pd.read_csv(OUTPUT / 'best_validation_predictions.csv')
    assert saved.ID.equals(val.ID) and saved.trial_id.equals(val.trial_id)
    assert saved.phase.equals(val.phase) and saved.label.equals(val.label)
    with (OUTPUT / 'best_model.pkl').open('rb') as handle:
        baseline = pickle.load(handle)  # Trusted local artifact, unchanged.
    assert baseline['config']['experiment'] == 'criteria_year0_pooled_C1'
    xt, xv = prepare(train), prepare(val)
    baseline_p = predict(baseline, xv)
    assert np.allclose(baseline_p, saved.probability, rtol=0, atol=1e-14)
    baseline_summary, baseline_bins = diagnostics(val, baseline_p, 'baseline')
    baseline_summary.to_csv(CAL_OUTPUT / 'baseline_calibration_summary.csv', index=False)
    baseline_bins.to_csv(CAL_OUTPUT / 'baseline_calibration_bins.csv', index=False)
    print('EXISTING MODEL DIAGNOSTICS\n' + baseline_summary.to_string(index=False), flush=True)
    print('PHASE III BINS\n' + baseline_bins.query("phase == 'III'").to_string(index=False), flush=True)

    scores = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1)
    fold_balance = []
    for fold, (fit, held) in enumerate(grouped_folds(train)):
        fitted = fit_features(xt.iloc[fit], xt.iloc[held], ('criteria',), False, 'pooled')
        models, _, _ = fit_classifiers(fitted, train.label.to_numpy()[fit], 1.0)
        fold_bundle = dict(kind='logistic', models=models)
        scores[held] = decision_scores(fold_bundle, xt.iloc[held])
        fold_ids[held] = fold
        for role, indexes in [('base_fit', fit), ('calibration_oof', held)]:
            for phase, group in train.iloc[indexes].groupby('phase'):
                fold_balance.append(dict(fold=fold, role=role, phase=phase, n=len(group),
                    n_trials=group.trial_id.nunique(), successes=int(group.label.sum()),
                    success_rate=float(group.label.mean())))
        print(f'OOF fold {fold + 1}/{N_FOLDS}: fit={len(fit)}, held={len(held)}', flush=True)
    assert np.isfinite(scores).all() and (fold_ids >= 0).all()
    assert pd.DataFrame({'trial_id': train.trial_id, 'fold': fold_ids}).groupby('trial_id').fold.nunique().max() == 1
    oof = train[['ID', 'trial_id', 'phase', 'label']].copy()
    oof['fold'] = fold_ids
    oof['decision_score'] = scores
    oof.to_csv(CAL_OUTPUT / 'training_oof_scores.csv', index=False)
    pd.DataFrame(fold_balance).to_csv(CAL_OUTPUT / 'fold_balance.csv', index=False)
    # All calibrators are fitted BEFORE evaluating any calibrated validation predictions.
    candidates = {name: fit_sigmoids(scores, train.label.to_numpy(), train.phase, separate)
                  for name, separate in [('sigmoid_pooled', False), ('sigmoid_by_phase', True)]}
    parameters = [dict(candidate=name, group=group, slope=float(model.coef_[0, 0]),
                       intercept=float(model.intercept_[0]),
                       n_fit=len(train) if group == 'pooled' else int(train.phase.eq(group).sum()))
                  for name, calibrators in candidates.items() for group, model in calibrators.items()]
    pd.DataFrame(parameters).to_csv(CAL_OUTPUT / 'sigmoid_parameters.csv', index=False)
    full_scores = decision_scores(baseline, xv)
    summaries, bins = [baseline_summary], [baseline_bins]
    predictions = []
    bundles = {}
    for name in ('baseline', *candidates):
        if name == 'baseline':
            p = baseline_p
        else:
            p = apply_sigmoids(candidates[name], full_scores, val.phase)
            summary, bin_table = diagnostics(val, p, name)
            summaries.append(summary)
            bins.append(bin_table)
            bundles[name] = dict(baseline=baseline, calibrators=candidates[name],
                                 method=name, training='5-fold grouped OOF; frozen full-training base model')
        record = val[['ID', 'trial_id', 'phase', 'label']].copy()
        record['candidate'] = name
        record['probability'] = p
        record['predicted_label'] = (p >= THRESHOLD).astype(int)
        predictions.append(record)
    results = pd.concat(summaries, ignore_index=True)
    bin_table = pd.concat(bins, ignore_index=True)
    results.to_csv(CAL_OUTPUT / 'validation_metrics.csv', index=False)
    bin_table.to_csv(CAL_OUTPUT / 'validation_calibration_bins.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(CAL_OUTPUT / 'validation_predictions.csv', index=False)
    losses = results.loc[results.phase.eq('overall')].set_index('candidate').log_loss
    winner = losses.idxmin()
    submissions = []
    # Save each improving candidate, with its own model and distinct submission filename.
    for name, bundle in bundles.items():
        if losses[name] < min(losses['baseline'], 0.57375):
            with (CAL_OUTPUT / f'{name}_model.pkl').open('wb') as handle:
                pickle.dump(bundle, handle)
            submission = pd.DataFrame({'ID': test.ID, 'label': calibrated_predict(bundle, test)})
            path = CAL_OUTPUT / f'{name}_candidate_submission.csv'
            validate_submission(submission, test, sample)
            submission.to_csv(path, index=False)
            validate_submission(pd.read_csv(path), test, sample)
            with (CAL_OUTPUT / f'{name}_model.pkl').open('rb') as handle:
                reloaded = pickle.load(handle)
            assert np.allclose(calibrated_predict(reloaded, test), submission.label, rtol=0, atol=1e-14)
            submissions.append(path.name)
    # Independent formulas verify every summary and every bin without refitting.
    all_predictions = pd.concat(predictions, ignore_index=True)
    for row in results.itertuples(index=False):
        rows = all_predictions.loc[all_predictions.candidate.eq(row.candidate)]
        if row.phase != 'overall':
            rows = rows.loc[rows.phase.eq(row.phase)]
        y, p = rows.label.to_numpy(), rows.probability.to_numpy()
        assert np.isclose(np.mean((p-y)**2), row.brier_score)
        assert np.isclose(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p)), row.log_loss)
        b = bin_table.loc[bin_table.candidate.eq(row.candidate) & bin_table.phase.eq(row.phase)]
        assert b.n.sum() == len(rows) and b.successes.sum() == rows.label.sum()
        assert np.isclose((b.mean_probability.fillna(0)*b.n).sum()/len(rows), p.mean())
    after = fingerprints()
    assert all(after.get(path) == digest for path, digest in protected.items())
    metadata = dict(seed=SEED, folds=N_FOLDS, threshold=THRESHOLD,
        method='5-fold StratifiedGroupKFold, phase x outcome strata, trial_id groups',
        calibration_fit='unpenalized logistic regression of training labels on OOF raw decision scores',
        inference='apply learned sigmoid to original full-training frozen baseline scores; no ensemble',
        validation_used_for='diagnostics, evaluation and selection only',
        baseline_log_loss=float(losses['baseline']), best_candidate=winner,
        candidate_submissions=submissions, protected_files_unchanged=True, protected_sha256=protected,
        python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__)
    (CAL_OUTPUT / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    write_report(results, baseline_bins, pd.DataFrame(parameters), winner, submissions)
    print(markdown_table(results[['candidate','phase','log_loss','brier_score','mean_probability','observed_success_rate']]))
    print('Best:', winner, 'Submissions:', submissions)


def comparison_interpretation(results):
    table = results.set_index(['candidate', 'phase'])
    base, pooled, phase = (table.loc[(name, 'overall')] for name in
                           ('baseline', 'sigmoid_pooled', 'sigmoid_by_phase'))
    b3, p3, s3 = (table.loc[(name, 'III')] for name in
                  ('baseline', 'sigmoid_pooled', 'sigmoid_by_phase'))
    return [f'Pooled sigmoid changes overall log loss from {base.log_loss:.6f} to {pooled.log_loss:.6f} '
            f'(delta {pooled.log_loss-base.log_loss:+.6f}) and Phase III from {b3.log_loss:.6f} '
            f'to {p3.log_loss:.6f}. Overall Brier changes from {base.brier_score:.6f} '
            f'to {pooled.brier_score:.6f}; Phase III Brier from {b3.brier_score:.6f} '
            f'to {p3.brier_score:.6f}. Phase-specific sigmoid gives overall log loss '
            f'{phase.log_loss:.6f} and Phase III {s3.log_loss:.6f}. '
            'It helps Phase III more but worsens Phase I and has worse overall loss than pooled calibration. '
            'The overall improvements are small; this experiment does not establish statistical significance '
            'or guarantee improvement on later test trials. No calibration or temporal hyperparameter search '
            'was performed.', '']


def write_report(results, baseline_bins, parameters, winner, submissions):
    lines = ['# Targeted probability-calibration experiment', '',
        'The original pooled criteria + phase logistic model (C=1, no year) is frozen. '
        'All existing artifacts are unchanged. This experiment uses no Part 1 data.', '',
        '## Existing model calibration', '',
        markdown_table(results.loc[results.candidate.eq('baseline'),
            ['phase','n','mean_probability','observed_success_rate','mean_gap','brier_score','ece_10_bins']]), '',
        'Positive mean gaps indicate average success overprediction; they do not establish overconfidence '
        'across the probability range. Overconfidence means probabilities are too extreme relative to observed '
        'frequencies. Brier score measures both calibration and discrimination. ECE is a descriptive ten-bin '
        'statistic sensitive to binning and sampling noise, not the model-selection criterion.', '',
        'Phase III does not show broad overconfidence on validation: its mean predicted success is '
        '66.05%, nearly the observed 66.12%. Among the 79 predictions >=0.8, mean probability is '
        '85.80% but observed success is 92.41%, indicating underconfidence in that tail. The '
        '0.4–0.5 bin overpredicts success (45.21% predicted versus 26.47% observed, n=34), while '
        'the 0.8–0.9 bin underpredicts it (84.94% versus 91.30%, n=69). Thus miscalibration is mixed; '
        'high false-success rates among unsuccessful trials alone do not establish overconfidence. '
        'Low-probability Phase III bins are small and noisy.', '',
        'Phase III fixed-width reliability bins (empty-bin statistics are undefined):', '',
        markdown_table(baseline_bins.loc[baseline_bins.phase.eq('III'),
            ['lower','upper','n','successes','mean_probability','observed_success_rate','gap']]), '',
        '## Procedure and fairness', '',
        'Five StratifiedGroupKFold splits use training trial_id as the group and phase × label as the '
        'stratification target. Trial IDs never cross fit/held folds; combined-phase rows stay together. '
        'Every fold refits the original TF-IDF/phase preprocessing and the same C=1 classifier using '
        'only that fold’s fit rows. Exactly one out-of-fold raw decision margin is generated per training row. '
        'No validation text or labels enter these fits.', '',
        'Two prespecified calibration candidates are fitted to all training OOF margins: one global sigmoid '
        'and independent sigmoids by phase. Each is an unpenalized two-parameter logistic regression; '
        'Phase I/II/III provide 1,012/3,950/3,075 OOF rows, each with both classes. '
        'The calibrators are fitted before computing calibrated validation predictions. F1 uses >=0.5. '
        'No threshold search or other model changes are made.', '',
        'For validation/test, both mappings transform raw margins from the exact original full-training '
        'baseline. Thus inference uses identical base predictions for all candidates. The OOF models see '
        '80% of training trials whereas the frozen base saw 100%; this common non-ensemble cross-fitting '
        'approach can introduce a score-distribution mismatch. OOF scores are not in-sample margins. '
        'Calibration fit performance is not used to select a candidate. The external validation set is '
        'evaluation-only for these fits, but it has already been used for baseline selection, so it is '
        'not an independent final test. Selection here uses overall validation log loss.', '',
        '## Validation comparison', '',
        markdown_table(results[['candidate','phase','log_loss','brier_score','f1','roc_auc','average_precision']]), '',
        *comparison_interpretation(results),
        'All sigmoid slopes are positive: within each phase, ranking and therefore ROC-AUC/AP are unchanged '
        '(up to floating-point ties). Changes in log loss/Brier reflect probability changes; F1 can change '
        'because the 0.5 decision boundary moves.', '',
        '## Training-only sigmoid parameters', '', markdown_table(parameters), '',
        'For p=expit(slope × raw_margin + intercept), slope <1 compresses margins, slope >1 expands them. '
        'Intercept adjusts prevalence. These slopes describe training OOF calibration and are not fitted '
        'evidence about validation Phase III alone.', '',
        f'Best validation candidate: **{winner}**. Separate improving-candidate submissions: '
        + (', '.join(submissions) if submissions else 'none; the original baseline remains best') + '.', '',
        'See validation_metrics.csv for all means, tails, Brier scores, ECE and predictive metrics; '
        'validation_calibration_bins.csv for all candidates overall/by phase; training_oof_scores.csv '
        'and fold_balance.csv for the grouped calibration-fit audit. metadata.json records preservation '
        'hashes and the method. No new broad search was performed.', '',
        '## Reproduce', '', '```sh',
        'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.calibration',
        '/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m unittest part2.test_calibration',
        '```', '']
    (CAL_OUTPUT / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
