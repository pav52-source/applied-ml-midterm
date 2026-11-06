import numpy as np
from sklearn.linear_model import LogisticRegression
from .config import PHASES, SEED
from .features import build_features


def fit_features(train, val, text_columns, use_year, regime):
    groups = ('pooled',) if regime == 'pooled' else PHASES
    prepared = {}
    for group in groups:
        tm = np.ones(len(train), dtype=bool) if group == 'pooled' else train.phase.eq(group).to_numpy()
        vm = np.ones(len(val), dtype=bool) if group == 'pooled' else val.phase.eq(group).to_numpy()
        features = build_features(text_columns, use_year, regime == 'pooled')
        xt = features.fit_transform(train.loc[tm])
        xv = features.transform(val.loc[vm])
        prepared[group] = (features, xt, xv, tm, vm)
    return prepared


def fit_classifiers(prepared, labels, c):
    train_p = np.empty(len(labels))
    val_p = np.empty(sum(int(v[4].sum()) for v in prepared.values()))
    models = {}
    for group, (features, xt, xv, tm, vm) in prepared.items():
        model = LogisticRegression(C=c, solver='liblinear', max_iter=2000, tol=1e-5, random_state=SEED)
        model.fit(xt, labels[tm])
        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError(f'Nonconvergence for {group}, C={c}')
        train_p[tm] = model.predict_proba(xt)[:, 1]
        val_p[vm] = model.predict_proba(xv)[:, 1]
        models[group] = dict(features=features, classifier=model)
    return models, train_p, val_p


def predict(bundle, frame):
    if bundle['kind'] == 'prior':
        return frame.phase.map(bundle['priors']).to_numpy(dtype=float)
    p = np.empty(len(frame))
    for group, components in bundle['models'].items():
        mask = np.ones(len(frame), dtype=bool) if group == 'pooled' else frame.phase.eq(group).to_numpy()
        if mask.any():
            x = components['features'].transform(frame.loc[mask])
            p[mask] = components['classifier'].predict_proba(x)[:, 1]
    return p
