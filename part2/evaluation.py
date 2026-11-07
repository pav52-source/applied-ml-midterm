import numpy as np
from sklearn.metrics import log_loss, f1_score, roc_auc_score, average_precision_score
from .config import PHASES, THRESHOLD


def evaluate(frame, probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(frame),) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Expected one finite probability per row in [0, 1]')
    result = {}
    for phase in ('overall', *PHASES):
        mask = np.ones(len(frame), dtype=bool) if phase == 'overall' else frame.phase.eq(phase).to_numpy()
        y, pred = frame.label.to_numpy()[mask], p[mask]
        result[f'{phase}_n'] = len(y)
        if not len(y):
            for metric in ('log_loss', 'f1', 'roc_auc', 'average_precision'):
                result[f'{phase}_{metric}'] = np.nan
            continue
        result[f'{phase}_log_loss'] = log_loss(y, pred, labels=[0, 1])
        result[f'{phase}_f1'] = f1_score(y, pred >= THRESHOLD, zero_division=0)
        both = len(np.unique(y)) == 2
        result[f'{phase}_roc_auc'] = roc_auc_score(y, pred) if both else np.nan
        result[f'{phase}_average_precision'] = average_precision_score(y, pred) if both else np.nan
    result['phase_macro_log_loss'] = np.mean([result[f'{p}_log_loss'] for p in PHASES])
    return result
