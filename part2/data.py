import ast
import re
import pandas as pd
from .config import DATA, PHASES

TEXT_COLUMNS = ('criteria', 'diseases_cleaned', 'drugs', 'icdcodes')


def load_data():
    names = ('outcome_train', 'outcome_val', 'outcome_test', 'part2_sample_submission')
    missing = [str(DATA / f'{n}.csv') for n in names if not (DATA / f'{n}.csv').exists()]
    if missing:
        raise FileNotFoundError(f'Missing files: {missing}')
    train, val, test, sample = [pd.read_csv(DATA / f'{n}.csv') for n in names]
    for frame, expected in zip((train, val, test), (8037, 1013, 1500)):
        assert len(frame) == expected
        assert frame.ID.is_unique and frame.ID.notna().all()
        assert frame.trial_id.notna().all()
        assert set(frame.phase) == set(PHASES)
        assert not frame.duplicated(['trial_id', 'phase']).any()
        assert frame.ID.eq(frame.trial_id + '_' + frame.phase).all()
        assert set(TEXT_COLUMNS + ('start_year',)).issubset(frame.columns)
    for frame in (train, val):
        assert set(frame.label) == {0, 1}
        assert frame.groupby('trial_id').label.nunique().max() == 1
    for a, b in ((train, val), (train, test), (val, test)):
        assert not set(a.trial_id).intersection(b.trial_id), 'Cross-split trial leakage'
    assert sample.ID.equals(test.ID)
    assert list(sample.columns) == ['ID', 'label']
    return train, val, test, sample


def flatten(value):
    """Decode nested Python-list strings safely (including nested ICD lists)."""
    if isinstance(value, (list, tuple)):
        return ' '.join(flatten(v) for v in value)
    if pd.isna(value):
        return ''
    value = str(value).strip()
    if value.startswith('['):
        try:
            return flatten(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            pass
    return re.sub(r'\s+', ' ', value).strip()


def prepare(frame):
    # Explicit allowlist: IDs, labels, reason labels/embeddings never enter features.
    result = frame[list(TEXT_COLUMNS) + ['phase', 'start_year']].copy()
    for col in TEXT_COLUMNS:
        result[col] = result[col].map(flatten)
    return result
