# Part 1 — Stopping-reason classification

Refactored from the original `part1.py` with the same core model settings and experiments.

## Layout

- `config.py` — paths, labels, hyperparameters, fixed Kaggle thresholds.
- `data_utils.py` — train/validation/test loading and training-only scaling.
- `models.py` — Logistic Regression, SVM, prediction and evaluation helpers.
- `pseudo_labeling.py` — K-Means, GMM, and per-label pseudo-label methods.
- `experiments.py` — experiments A–H, validation results, optional K search.
- `submission.py` — retrains the selected K-Means configuration and exports Kaggle CSV.
- `outputs/` — generated result tables and submission files.
- `part1-data/` — put the original three data CSV files here (not included in ZIP).

## Run

From this directory:

```bash
python -m pip install -r requirements.txt
python experiments.py
python experiments.py --tune-k
python submission.py
# For a different chosen cluster count:
python submission.py --k 12
```

The default submission is written to `outputs/part1_submission.csv`. It uses K=8, thresholds `[0.2, 0.2, 0.2, 0.4, 0.4]`, and assigns the highest-probability reason to rows otherwise predicted with no labels. This matches the **corrected** nonempty Kaggle workflow.

## Notes

- Original data files are **not included**. Place `part1_train.csv`, `part1_val.csv`, and `part1_test.csv` in `part1-data/`.
- The original script evaluated thresholds on the same validation set used to tune them. Treat the tuned validation score as optimistic.
- `experiments.py` reports both tuned scores *without* fallback and the validation score *with* the fallback used for Kaggle.
- Experiment F preserves the original fixed Recruitment-only random control. Its pseudo-label count follows the selected K-Means experiment (285 at K=8 on the original data).
- Experiment H uses the RBF SVM settings previously discussed; it was not present in the uploaded `part1.py`.
- If you change K, the submission still uses the fixed thresholds unless you edit `config.py`.
- The Kaggle `sample_submission.csv` is authoritative for column names and ID ordering; check it before submitting.
