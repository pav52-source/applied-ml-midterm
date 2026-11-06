"""Analysis-only stopping-reason joins; never fits or changes any model."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ROOT, OUTPUT, PHASES, THRESHOLD
from .report import markdown_table

REASONS = ('Safety', 'Funding', 'Efficacy', 'Recruitment', 'Other')
NO_REASON = 'no recorded stopping reason'
UNAVAILABLE = 'matched Part 1 record; reasons unavailable'


def build_reason_table(parts, predictions):
    """Prefer complete true labels, then saved predictions, else retain unknowns."""
    records = []
    for split, frame in parts.items():
        assert frame.ID.str.fullmatch(r'TR\d+').all() and frame.ID.is_unique
        for _, row in frame.iterrows():
            values = row[list(REASONS)] if set(REASONS) <= set(frame.columns) else pd.Series([-100] * 5)
            assert values.isin([0, 1, -100]).all()
            valid = values.isin([0, 1]).all()
            assert valid or values.eq(-100).all(), 'Unexpected partial reason labels'
            records.append(dict(trial_id=row.ID, part1_split=split,
                                reason_source='ground_truth' if valid else 'unavailable',
                                **{r: int(values.iloc[i]) if valid else pd.NA for i, r in enumerate(REASONS)}))
    reasons = pd.DataFrame(records).set_index('trial_id')
    assert reasons.index.is_unique, 'Part 1 split IDs must be disjoint'
    assert predictions.ID.is_unique and predictions.ID.str.fullmatch(r'TR\d+').all()
    assert set(predictions.ID) == set(parts['test'].ID), 'Saved submission must cover actual Part 1 test IDs'
    for row in predictions.itertuples(index=False):
        assert isinstance(row.Predicted, str) and row.Predicted.strip()
        labels = row.Predicted.split()
        assert set(labels) <= set(REASONS) and len(labels) == len(set(labels))
        if reasons.loc[row.ID, 'reason_source'] != 'ground_truth':
            reasons.loc[row.ID, 'reason_source'] = 'prediction'
            for reason in REASONS:
                reasons.loc[row.ID, reason] = int(reason in labels)
    for reason in REASONS:
        reasons[reason] = reasons[reason].astype('Int64')
    return reasons.reset_index()


def match_unsuccessful(validation, outcome, reasons):
    assert validation.ID.is_unique and not validation.duplicated(['trial_id', 'phase']).any()
    assert validation.trial_id.str.fullmatch(r'TR\d+').all()
    assert validation.ID.eq(validation.trial_id + '_' + validation.phase).all()
    assert set(validation.phase) == set(PHASES)
    assert validation.label.isin([0, 1]).all()
    assert np.isfinite(validation.probability).all() and validation.probability.between(0, 1).all()
    assert validation.predicted_label.eq((validation.probability >= THRESHOLD).astype(int)).all()
    # Verify saved predictions against the actual outcome validation split, not ID format alone.
    check = validation.merge(outcome[['ID', 'trial_id', 'phase', 'label']], on='ID',
                             suffixes=('', '_actual'), validate='one_to_one', how='outer', indicator=True)
    assert check._merge.eq('both').all()
    for col in ('trial_id', 'phase', 'label'):
        assert check[col].eq(check[col + '_actual']).all()
    unsuccessful = validation.loc[validation.label.eq(0)].copy()
    joined = unsuccessful.merge(reasons, on='trial_id', how='left', validate='many_to_one', indicator=True)
    joined['part1_match'] = joined.pop('_merge').eq('both')
    joined['reason_source'] = joined.reason_source.fillna('no_record')
    joined['part1_split'] = joined.part1_split.fillna('none')
    assert not joined.duplicated(['trial_id', 'phase']).any()
    return joined


def summarize(group):
    n = len(group)
    fp = int(group.predicted_label.sum())
    return dict(n_trials=n, n_predicted_success=fp,
                false_success_rate=fp / n if n else np.nan,
                mean_predicted_success_probability=float(group.probability.mean()) if n else np.nan,
                n_ground_truth=int(group.reason_source.eq('ground_truth').sum()),
                n_predictions=int(group.reason_source.eq('prediction').sum()),
                n_no_record=int(group.reason_source.eq('no_record').sum()),
                n_reasons_unavailable=int(group.reason_source.eq('unavailable').sum()))


def summarize_groups(joined):
    rows, coverage, unresolved, source_rows = [], [], [], []
    for phase in PHASES:
        phase_rows = joined.loc[joined.phase.eq(phase)]
        coverage.append(dict(phase=phase, **summarize(phase_rows)))
        unresolved.append(dict(phase=phase, reason_group=UNAVAILABLE,
                               **summarize(phase_rows.loc[phase_rows.reason_source.eq('unavailable')])))
        for reason in (*REASONS, NO_REASON):
            mask = ~phase_rows.part1_match if reason == NO_REASON else phase_rows[reason].eq(1).fillna(False)
            group = phase_rows.loc[mask]
            rows.append(dict(phase=phase, reason_group=reason, **summarize(group)))
            for source in ('ground_truth', 'prediction', 'no_record'):
                source_rows.append(dict(phase=phase, reason_group=reason, reason_source=source,
                                        **summarize(group.loc[group.reason_source.eq(source)])))
    return tuple(pd.DataFrame(r) for r in (rows, coverage, unresolved, source_rows))


def main():
    # Fingerprint every existing Part 1 input/output and Part 2 baseline artifact.
    # No imports from Part 1 training modules and no model loading are required.
    protected = sorted([*ROOT.joinpath('part1-data').glob('*'), *ROOT.joinpath('outputs').glob('*'),
                        *OUTPUT.glob('*')])
    protected = [p for p in protected if p.is_file() and not p.name.startswith('stopping_reason_')]
    before = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    parts = {s: pd.read_csv(ROOT / 'part1-data' / f'part1_{s}.csv') for s in ('train', 'val', 'test')}
    predictions = pd.read_csv(ROOT / 'outputs' / 'part1_submission.csv')
    validation = pd.read_csv(OUTPUT / 'best_validation_predictions.csv')
    outcome = pd.read_csv(ROOT / 'part2-data' / 'outcome_val.csv')
    reasons = build_reason_table(parts, predictions)
    joined = match_unsuccessful(validation, outcome, reasons)
    groups, coverage, unresolved, sources = summarize_groups(joined)
    for name, frame in dict(groups=groups, coverage=coverage, unresolved=unresolved,
                            by_source=sources, trial_audit=joined, reason_lookup=reasons).items():
        frame.to_csv(OUTPUT / f'stopping_reason_{name}.csv', index=False)
    # Independent row-by-row membership/count/sum verification (no summarize reuse).
    for row in groups.itertuples(index=False):
        members = [r for r in joined.to_dict('records') if r['phase'] == row.phase and
                   ((not r['part1_match']) if row.reason_group == NO_REASON else
                    (not pd.isna(r[row.reason_group]) and r[row.reason_group] == 1))]
        assert len(members) == row.n_trials
        assert sum(r['probability'] >= THRESHOLD for r in members) == row.n_predicted_success
        if members:
            assert np.isclose(sum(r['probability'] for r in members) / len(members), row.mean_predicted_success_probability)
            assert np.isclose(row.n_predicted_success / len(members), row.false_success_rate)
        else:
            assert pd.isna(row.false_success_rate) and pd.isna(row.mean_predicted_success_probability)
    assert coverage.n_trials.sum() == validation.label.eq(0).sum()
    assert (coverage.n_trials == coverage[['n_ground_truth', 'n_predictions', 'n_no_record', 'n_reasons_unavailable']].sum(axis=1)).all()
    report = ['# Stopping-reason error analysis of the fixed Part 2 winner', '',
        'Uses only true outcome label=0 validation rows. A false success is probability >= 0.5. '
        'The winner remains criteria_year0_pooled_C1 (validation log loss 0.57375). '
        'No training, model loading, or prediction changes were performed.', '',
        '## Matching and reason provenance', '',
        f'{len(joined)} unsuccessful phase rows represent {joined.trial_id.nunique()} unique trials. '
        'Part 1 ID has format TR followed by digits and equals Part 2 trial_id. '
        'The phase-suffixed Part 2 ID is checked against trial_id + phase. Saved outcome labels, '
        'phases and trial IDs are checked against outcome_val.csv with a one-to-one join. '
        'Part 1 split IDs are disjoint and unique; saved reason predictions cover exactly Part 1 test IDs. '
        'The reason join is many-to-one and each phase has at most one row per trial.', '',
        'Available binary ground-truth labels from Part 1 train/val take precedence. '
        'Predicted reasons come only from the existing outputs/part1_submission.csv for Part 1 test. '
        'Part 1 train labels masked as -100 remain unknown: no saved predictions or fitted Part 1 model '
        'were found for them, and Part 1 training is not rerun. '
        'Only trials absent from all three Part 1 splits enter “no recorded stopping reason”; '
        'this is an assignment-defined absence-of-record group, not proof that the trial completed normally.', '',
        markdown_table(coverage), '',
        '## Required six groups', '',
        'The five category rows include only known positive reasons, either ground truth or saved predictions. '
        'Their counts are incomplete wherever matched reasons are unavailable; unknown values are not negatives. '
        'Multi-label trials appear in each applicable category, so category counts overlap. '
        'Combined-phase trials can contribute once in each phase. Undefined empty-group rates and means '
        'are NaN (blank in CSV), not zero. Provenance counts are phase-row counts.', '',
        markdown_table(groups), '',
        '## Matched records with unavailable reasons (outside the six groups)', '',
        markdown_table(unresolved), '', '## Interpretation', '']
    for phase in PHASES:
        available = groups.loc[groups.phase.eq(phase) & groups.n_trials.gt(0)].sort_values('false_success_rate', ascending=False)
        maximum = available.false_success_rate.max()
        top = available.loc[available.false_success_rate.eq(maximum)]
        detail = '; '.join(f'{r.reason_group}: {r.n_predicted_success}/{r.n_trials} ({r.false_success_rate:.1%})'
                           for r in top.itertuples())
        report.append(f'- Phase {phase}: highest observed fraction — {detail}.')
    report += ['',
        'Among larger groups, Phase III no recorded stopping reason has 53/82 false successes (64.6%), '
        'versus 13/28 (46.4%) in Phase I and 52/144 (36.1%) in Phase II. This is the clearest '
        'descriptive concentration of errors, although phases have different outcome definitions and trial mixes. '
        'Phase II Recruitment has 6/31 false successes (19.4%). Phase I Other (2/3), Phase II Safety (3/5), '
        'and Phase III Efficacy (1/1) lead their phases but are all too small for strong conclusions. '
        'Phase I Other and Phase III Efficacy are entirely based on predicted reasons. '
        f'There are {int(unresolved.n_trials.sum())} unresolved unsuccessful phase rows, so the reason-specific '
        'results describe the available subset rather than all stopped trials.', '',
        'Cells with fewer than 10 trials are especially fragile: one trial changes a fraction by at least '
        '10 percentage points. Larger groups offer more stable descriptive evidence, but these comparisons '
        'are not significance tests or causal effects. Missing labels may be systematically different '
        'from available labels, and predicted Part 1 reasons introduce classification error. '
        'The source-stratified CSV permits inspection without conflating truth and predictions. '
        'Do not rank complete stopping-reason populations from these incomplete category counts.', '',
        '## Verification and reproduction', '',
        'Independent row-wise membership checks verify all 18 cells, integer false-success counts, '
        'rates, mean probabilities, and empty-group behavior. Coverage counts partition all unsuccessful '
        'validation rows into truth, predictions, no record, and unavailable reasons. '
        'SHA-256 checks confirm existing input/output artifacts did not change.', '',
        'Run from the repository root:', '', '```sh',
        '/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.error_analysis', '```', '',
        'Outputs: stopping_reason_groups.csv (18 requested cells), stopping_reason_coverage.csv, '
        'stopping_reason_unresolved.csv, stopping_reason_by_source.csv, stopping_reason_trial_audit.csv, '
        'stopping_reason_reason_lookup.csv and stopping_reason_verification.json. '
        'The last two audit tables retain unknown reasons as missing values. No reasons are predictive inputs. '
        'Analysis stops here; no model improvements are implemented.', '']
    (OUTPUT / 'stopping_reason_report.md').write_text('\n'.join(report))
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in before.items())
    (OUTPUT / 'stopping_reason_verification.json').write_text(json.dumps(dict(
        threshold=THRESHOLD, n_group_cells_verified=len(groups), n_unsuccessful_rows=len(joined),
        n_unique_unsuccessful_trials=joined.trial_id.nunique(),
        protected_files_unchanged=True, protected_sha256=before), indent=2))
    print(markdown_table(groups))
    print('\nCOVERAGE\n' + markdown_table(coverage))
    print('\nUNRESOLVED\n' + markdown_table(unresolved))


if __name__ == '__main__':
    main()
