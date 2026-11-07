"""Run the predeclared small factorial baseline grid; select by validation log loss."""
import hashlib
import json
import pickle
import platform
import time
from itertools import product
import numpy as np
import pandas as pd
import scipy
import sklearn
from .audit import audit
from .config import DATA, OUTPUT, TEXT_SETS, CS, SEED, THRESHOLD
from .data import load_data, prepare
from .evaluation import evaluate
from .training import fit_features, fit_classifiers, predict
from .submission import generate


def main():
    train, val, test, sample = load_data()
    audit(dict(train=train, val=val, test=test))
    xt, xv = prepare(train), prepare(val)
    records, predictions = [], []
    best_loss, best_id = float('inf'), None

    def record(config, bundle, pt, pv, seconds):
        nonlocal best_loss, best_id
        row = {**config, 'threshold': THRESHOLD, 'fit_seconds': seconds}
        row.update({f'train_{k}': v for k, v in evaluate(train, pt).items()})
        row.update({f'val_{k}': v for k, v in evaluate(val, pv).items()})
        records.append(row)
        pred = val[['ID', 'trial_id', 'phase', 'label']].copy()
        pred['experiment'] = config['experiment']
        pred['probability'] = pv
        predictions.append(pred)
        loss = row['val_overall_log_loss']
        print(f"{config['experiment']:35s} train={row['train_overall_log_loss']:.5f} val={loss:.5f}", flush=True)
        if loss < best_loss:
            best_loss, best_id = loss, config['experiment']
            bundle['config'] = config
            with (OUTPUT / 'best_model.pkl').open('wb') as f:
                pickle.dump(bundle, f)
        pd.DataFrame(records).to_csv(OUTPUT / 'experiment_results.csv', index=False)

    prior = dict(kind='prior', priors=train.groupby('phase').label.mean().to_dict())
    record(dict(experiment='phase_prior', kind='prior', text_set='none', use_year=False, regime='phase_specific', C=None),
           prior, predict(prior, xt), predict(prior, xv), 0.0)
    for text_set, use_year, regime in product(TEXT_SETS, (False, True), ('pooled', 'phase_specific')):
        start = time.perf_counter()
        prepared = fit_features(xt, xv, TEXT_SETS[text_set], use_year, regime)
        feature_seconds = time.perf_counter() - start
        for c in CS:
            start = time.perf_counter()
            models, pt, pv = fit_classifiers(prepared, train.label.to_numpy(), c)
            config = dict(experiment=f'{text_set}_year{int(use_year)}_{regime}_C{c:g}', kind='logistic',
                          text_set=text_set, use_year=use_year, regime=regime, C=c)
            record(config, dict(kind='logistic', models=models), pt, pv,
                   feature_seconds + time.perf_counter() - start)
    pd.concat(predictions, ignore_index=True).to_csv(OUTPUT / 'validation_predictions.csv', index=False)
    with (OUTPUT / 'best_model.pkl').open('rb') as f:
        best = pickle.load(f)
    pv = predict(best, xv)
    assert np.allclose(evaluate(val, pv)['overall_log_loss'], best_loss)
    selected = val[['ID', 'trial_id', 'phase', 'label']].copy()
    selected['probability'] = pv
    selected['predicted_label'] = (pv >= THRESHOLD).astype(int)
    selected.to_csv(OUTPUT / 'best_validation_predictions.csv', index=False)
    path = generate(best, test, sample)
    metadata = dict(seed=SEED, best_experiment=best_id, validation_log_loss=best_loss,
        selection='minimum overall validation log loss; no validation refit',
        threshold=THRESHOLD, python=platform.python_version(), pandas=pd.__version__,
        numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        inputs_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DATA.glob('*.csv'))},
        feature_settings=dict(min_df=3, ngram_range=[1, 2], sublinear_tf=True,
                              criteria_max_features=15000, other_max_features=5000),
        logistic_settings=dict(solver='liblinear', max_iter=2000, tol=1e-5),
        submission=str(path))
    (OUTPUT / 'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    table = pd.DataFrame(records).sort_values('val_overall_log_loss')
    print(table[['experiment', 'train_overall_log_loss', 'val_overall_log_loss', 'val_phase_macro_log_loss']].to_string(index=False))
    print(f'BEST: {best_id}; submission: {path}')


if __name__ == '__main__':
    main()
