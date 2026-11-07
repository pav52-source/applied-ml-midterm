"""Recompute only reason-dependent analyses; never train or alter a classifier."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from part2.error_analysis import REASONS, match_unsuccessful, summarize_groups
from part2.evaluation import evaluate
from part2.report import markdown_table

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'integration' / 'outputs'


def authoritative_reasons(root):
    parts = {s: pd.read_csv(root / 'part1-data' / f'part1_{s}.csv') for s in ('train','val','test')}
    train_predictions = pd.read_csv(root / 'part1-results/knn-experiment/best_augmented_training_labels.csv')
    test_predictions = pd.read_csv(root / 'part1-results/knn-experiment/best_augmented_candidate_submission.csv')
    assert train_predictions.ID.is_unique and set(train_predictions.ID) == set(parts['train'].ID)
    assert test_predictions.ID.is_unique and set(test_predictions.ID) == set(parts['test'].ID)
    assert train_predictions[list(REASONS)].isin([0, 1]).all().all()
    assert set(train_predictions.label_source) == {'observed_train', 'pseudo_train'}
    estimated = train_predictions.set_index('ID')
    test_estimated = test_predictions.set_index('ID')
    records = []
    for split, frame in parts.items():
        assert frame.ID.is_unique and frame.ID.str.fullmatch(r'TR\d+').all()
        for _, row in frame.iterrows():
            if split in ('train', 'val'):
                valid = row[list(REASONS)].isin([0, 1]).all()
                assert valid or row[list(REASONS)].eq(-100).all()
            else:
                valid = False
            if valid:
                values = row[list(REASONS)].astype(int).to_dict()
                source = f'observed_{split}'
                if split == 'train':
                    assert estimated.loc[row.ID, 'label_source'] == 'observed_train'
                    assert all(estimated.loc[row.ID, r] == values[r] for r in REASONS)
            elif split == 'train':
                assert estimated.loc[row.ID, 'label_source'] == 'pseudo_train'
                values = estimated.loc[row.ID, list(REASONS)].astype(int).to_dict()
                source = 'pseudo_train'
            else:
                text = test_estimated.loc[row.ID, 'Predicted']
                assert isinstance(text, str) and text.strip()
                labels = text.split()
                assert set(labels) <= set(REASONS) and len(labels) == len(set(labels))
                values = {reason: int(reason in labels) for reason in REASONS}
                source = 'predicted_part1_test'
            records.append(dict(trial_id=row.ID, part1_split=split,
                reason_source='ground_truth' if valid else 'prediction',
                label_source=source, **values))
    lookup = pd.DataFrame(records)
    assert lookup.trial_id.is_unique and len(lookup) == 1561
    return lookup


def reconcile(root=ROOT, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    lookup = authoritative_reasons(root)
    lookup.to_csv(out / 'authoritative_reason_lookup.csv', index=False)
    outcome = pd.read_csv(root / 'part2-data/outcome_val.csv')
    original = pd.read_csv(root / 'part2/outputs/best_validation_predictions.csv')
    calibrated = pd.read_csv(root / 'part2/calibration_outputs/validation_predictions.csv')
    candidates = {'baseline': original}
    for name in ('sigmoid_pooled', 'sigmoid_by_phase'):
        frame = calibrated.loc[calibrated.candidate.eq(name)].reset_index(drop=True).copy()
        assert frame.ID.equals(outcome.ID)
        candidates[name] = frame
    old_audit = pd.read_csv(root / 'part2/outputs/stopping_reason_trial_audit.csv')
    old_unknown = old_audit.loc[old_audit.reason_source.eq('unavailable')]
    assert len(old_unknown) == 131 and old_unknown.trial_id.nunique() == 113
    groups_all, coverage_all, audit_all, metric_rows = [], [], [], []
    sources_all = []
    for name, predictions in candidates.items():
        joined = match_unsuccessful(predictions, outcome, lookup)
        joined['label_source'] = joined.label_source.fillna('no_recorded_stopping_reason')
        groups, coverage, unresolved, sources = summarize_groups(joined)
        assert unresolved.n_trials.sum() == 0
        assert len(joined) == 452 and joined.trial_id.nunique() == 398
        resolved = joined.merge(old_unknown[['ID']], on='ID', validate='one_to_one')
        assert len(resolved) == 131 and resolved.reason_source.eq('prediction').all()
        assert resolved.label_source.eq('pseudo_train').all()
        assert coverage.n_ground_truth.sum() == 27
        assert coverage.n_predictions.sum() == 171 and coverage.n_no_record.sum() == 254
        # Independent row-level verification, separate from the group summary helper.
        for row in groups.itertuples(index=False):
            phase = joined.loc[joined.phase.eq(row.phase)]
            subset = phase.loc[~phase.part1_match] if row.reason_group == 'no recorded stopping reason' else phase.loc[phase[row.reason_group].eq(1)]
            assert len(subset) == row.n_trials
            count = int((subset.probability >= .5).sum())
            assert count == row.n_predicted_success
            if len(subset):
                assert np.isclose(row.false_success_rate, count / len(subset))
                assert np.isclose(row.mean_predicted_success_probability, subset.probability.mean())
            else:
                assert pd.isna(row.false_success_rate)
        for table, collection in [(groups, groups_all), (coverage, coverage_all), (joined, audit_all), (sources, sources_all)]:
            table['candidate'] = name
            collection.append(table)
        metrics = evaluate(outcome, predictions.probability.to_numpy())
        metric_rows.append(dict(candidate=name, **metrics))
    groups = pd.concat(groups_all, ignore_index=True)
    coverage = pd.concat(coverage_all, ignore_index=True)
    audits = pd.concat(audit_all, ignore_index=True)
    pd.concat(sources_all, ignore_index=True).to_csv(out / 'reason_groups_by_source.csv', index=False)
    groups.to_csv(out / 'reason_groups.csv', index=False)
    coverage.to_csv(out / 'coverage.csv', index=False)
    audits.to_csv(out / 'unsuccessful_trial_audit.csv', index=False)
    baseline_audit = audits.loc[audits.candidate.eq('baseline')]
    baseline_audit.loc[baseline_audit.ID.isin(old_unknown.ID)].to_csv(out / 'resolved_previous_unknowns.csv', index=False)
    old_groups = pd.read_csv(root / 'part2/outputs/stopping_reason_groups.csv')
    comparison = old_groups.merge(groups.loc[groups.candidate.eq('baseline')], on=['phase','reason_group'], suffixes=('_old','_integrated'), validate='one_to_one')
    for col in ('n_trials','n_predicted_success','false_success_rate','mean_predicted_success_probability'):
        comparison[col + '_change'] = comparison[col + '_integrated'] - comparison[col + '_old']
    comparison.to_csv(out / 'old_vs_integrated_baseline.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(out / 'unchanged_outcome_metrics.csv', index=False)
    best_comparison = groups.loc[groups.candidate.eq('baseline')].merge(
        groups.loc[groups.candidate.eq('sigmoid_pooled')], on=['phase','reason_group'], suffixes=('_baseline','_calibrated'),validate='one_to_one')
    best_comparison.to_csv(out / 'baseline_vs_pooled_calibration.csv', index=False)
    input_files = ['part1-results/knn-experiment/best_augmented_training_labels.csv',
        'part1-results/knn-experiment/best_augmented_candidate_submission.csv',
        'part2/outputs/best_validation_predictions.csv','part2/calibration_outputs/validation_predictions.csv',
        *[f'part1-data/part1_{s}.csv' for s in ('train','val','test')], 'part2-data/outcome_val.csv']
    metadata = dict(part1_source='origin/main final kmeans_knn_hybrid (3 uniform neighbors)',
        no_model_training=True, threshold=.5, previously_unresolved_phase_rows=131,
        previously_unresolved_unique_trials=113, resolved_as_predictions=131,
        remaining_unresolved_phase_rows=0,
        ground_truth_rows=27,predicted_rows=171,no_record_rows=254,
        hashes={p:hashlib.sha256((root / p).read_bytes()).hexdigest() for p in input_files})
    (out / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2))
    report = ['# Integrated stopping-reason error analysis', '',
        'Part 1 is the final origin/main hybrid. Part 2 probabilities are exactly the frozen origin/michael artifacts. '
        'Only true outcome label=0 rows are analyzed; probability >=0.5 is a false success. '
        'Visible training/validation reasons take precedence over predictions. Masked training reasons now use '
        'the teammate’s saved hybrid pseudo-labels; test reasons use the hybrid’s submitted predictions. '
        'Root part1_submission.csv and part1-results/reason_lookup.csv describe earlier models and are not used.', '',
        '**All 131 previously unresolved phase rows (113 trials) are now covered by predictions, not truth.** '
        'There remain 27 rows with true reasons, 171 with predicted reasons and 254 without a Part 1 record. '
        'No record is not a claim of successful completion. Multi-label categories overlap; each trial appears '
        'at most once per phase/group. Empty groups have undefined rates.', '',
        '## Coverage (baseline; membership identical for calibrated candidates)', '',
        markdown_table(coverage.loc[coverage.candidate.eq('baseline')]), '',
        '## All candidate group metrics', '', markdown_table(groups), '',
        '## Changes from Michael’s original baseline analysis', '',
        markdown_table(comparison[['phase','reason_group','n_trials_old','n_trials_integrated','false_success_rate_old','false_success_rate_integrated']]), '',
        'Reason-specific changes arise from expanded coverage and replacement of older test-reason predictions, '
        'not changed outcome probabilities. No-record group membership is unchanged. '
        'Pseudo-labels and test predictions have errors; larger groups improve descriptive denominators but '
        'do not turn inferred reasons into ground truth.', '',
        '## Baseline versus the retained pooled calibration', '',
        markdown_table(best_comparison[['phase','reason_group','n_trials_baseline','false_success_rate_baseline','false_success_rate_calibrated']]), '',
        'Calibration optimizes probability quality, not every thresholded subgroup rate. '
        'The same integrated group membership is used before/after calibration. Full subgroup provenance is '
        'retained in unsuccessful_trial_audit.csv and reason_groups_by_source.csv. No reason label or embedding '
        'enters the outcome classifier or calibrator.', '']
    (out / 'analysis_report.md').write_text('\n'.join(report))
    return groups, coverage, comparison


def figures(groups, out=OUT):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['Safety','Funding','Efficacy','Recruitment','Other','no recorded stopping reason']
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, phase in zip(axes, ('I','II','III')):
        for offset, name in [(-.18,'baseline'),(.18,'sigmoid_pooled')]:
            rows = groups.loc[groups.phase.eq(phase) & groups.candidate.eq(name)].set_index('reason_group').reindex(names)
            ax.bar(np.arange(6)+offset, rows.false_success_rate, width=.36, label=name)
        ax.set_xticks(range(6), ['Safety','Funding','Efficacy','Recruitment','Other','No record'], rotation=55, ha='right')
        ax.set(title=f'Phase {phase}', ylim=(0,1))
    axes[0].set_ylabel('False-success fraction among unsuccessful trials')
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'integrated_reason_errors.png', dpi=150)
    fig.savefig(out / 'integrated_reason_errors.pdf')
    plt.close(fig)


if __name__ == '__main__':
    result, _, _ = reconcile()
    figures(result)
    print(result[['candidate','phase','reason_group','n_trials','n_predicted_success','false_success_rate']].to_string(index=False))
