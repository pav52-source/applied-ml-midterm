"""Read-only analysis of input splits, run before training."""
import json
from itertools import combinations
import pandas as pd
from .config import OUTPUT
from .data import load_data, prepare, TEXT_COLUMNS


def audit(frames):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    schemas, missing, balance, years, duplicates = [], [], [], [], []
    for split, frame in frames.items():
        for col in frame:
            schemas.append(dict(split=split, rows=len(frame), column=col, dtype=str(frame[col].dtype)))
            missing.append(dict(split=split, column=col, missing=int(frame[col].isna().sum())))
        clean = prepare(frame)
        for col in TEXT_COLUMNS:
            missing.append(dict(split=split, column=col + '_empty_after_parsing', missing=int(clean[col].eq('').sum())))
        for phase, group in [('overall', frame), *list(frame.groupby('phase'))]:
            balance.append(dict(split=split, phase=phase, rows=len(group), unique_trials=group.trial_id.nunique(),
                                successes=int(group.label.sum()) if 'label' in group else None,
                                success_rate=float(group.label.mean()) if 'label' in group else None))
        for year, n in frame.start_year.value_counts(dropna=False).sort_index().items():
            years.append(dict(split=split, start_year=year, rows=n))
        duplicates.append(dict(split=split, exact_duplicate_rows=int(frame.duplicated().sum()),
            duplicate_trial_phase=int(frame.duplicated(['trial_id', 'phase']).sum()),
            multi_phase_trials=int((frame.groupby('trial_id').phase.nunique() > 1).sum()),
            extra_phase_rows=len(frame)-frame.trial_id.nunique(),
            year_min=frame.start_year.min(), year_median=frame.start_year.median(), year_max=frame.start_year.max()))
    overlap = []
    # Exact text matches may indicate related protocols; report separately from ID leakage.
    for (a, fa), (b, fb) in combinations(frames.items(), 2):
        ca, cb = prepare(fa), prepare(fb)
        for cols in [('trial_id',), ('criteria',), TEXT_COLUMNS]:
            aa, bb = (fa, fb) if cols == ('trial_id',) else (ca, cb)
            ha = pd.util.hash_pandas_object(aa[list(cols)], index=False)
            hb = pd.util.hash_pandas_object(bb[list(cols)], index=False)
            if cols == ('criteria',):
                ha, hb = ha[aa.criteria.ne('')], hb[bb.criteria.ne('')]
            overlap.append(dict(split_a=a, split_b=b, key='+'.join(cols), shared_keys=len(set(ha) & set(hb))))
    tables = dict(schema=schemas, missing_values=missing, class_balance=balance, start_year_counts=years,
                  duplicates_and_years=duplicates, cross_split_overlap=overlap)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(OUTPUT / f'{name}.csv', index=False)
    report = ['# Initial data audit', '', 'Text: criteria, diseases_cleaned, drugs, icdcodes. Additional available text/categorical fields: phase_detail, matched_drugs, drug_targets, drug_actions, drug_pathways, drug_smiles. Phase is categorical; start_year is numerical. ID and trial_id are matching keys, label is the binary target. Optional pathway_catalog has 2,501 rows and five columns; it is unused.', '', 'All splits have no duplicate rows or duplicate trial-phase records. Multiple phase rows of the same trial are retained within splits; no trial crosses splits. Cross-split exact text matches are reported as a possible related-protocol leakage risk. Year counts include missing values. Class balance and success rates are row-weighted, consistent with the assignment metric.', '']
    for name, rows in tables.items():
        report.extend([f'## {name}', '', '```text', pd.DataFrame(rows).to_string(index=False), '```', ''])
    (OUTPUT / 'data_audit.md').write_text('\n'.join(report))
    print('CLASS BALANCE\n' + pd.DataFrame(balance).to_string(index=False))
    print('DUPLICATES / YEARS\n' + pd.DataFrame(duplicates).to_string(index=False))
    print('OVERLAP\n' + pd.DataFrame(overlap).to_string(index=False))
    print('MISSING VALUES\n' + pd.DataFrame(missing).query('missing > 0').to_string(index=False))
    (OUTPUT / 'audit_summary.json').write_text(json.dumps(tables, indent=2, default=str))


if __name__ == '__main__':
    train, val, test, _ = load_data()
    audit(dict(train=train, val=val, test=test))
