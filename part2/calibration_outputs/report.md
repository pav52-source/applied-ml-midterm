# Targeted probability-calibration experiment

The original pooled criteria + phase logistic model (C=1, no year) is frozen. All existing artifacts are unchanged. This experiment uses no Part 1 data.

## Existing model calibration

| phase | n | mean_probability | observed_success_rate | mean_gap | brier_score | ece_10_bins |
| --- | --- | --- | --- | --- | --- | --- |
| overall | 1013 | 0.55857 | 0.55380 | 0.00477 | 0.19547 | 0.03488 |
| I | 149 | 0.56860 | 0.57047 | -0.00187 | 0.19103 | 0.07117 |
| II | 501 | 0.48171 | 0.47106 | 0.01065 | 0.20735 | 0.04048 |
| III | 363 | 0.66055 | 0.66116 | -0.00061 | 0.18089 | 0.05071 |

Positive mean gaps indicate average success overprediction; they do not establish overconfidence across the probability range. Overconfidence means probabilities are too extreme relative to observed frequencies. Brier score measures both calibration and discrimination. ECE is a descriptive ten-bin statistic sensitive to binning and sampling noise, not the model-selection criterion.

Phase III does not show broad overconfidence on validation: its mean predicted success is 66.05%, nearly the observed 66.12%. Among the 79 predictions >=0.8, mean probability is 85.80% but observed success is 92.41%, indicating underconfidence in that tail. The 0.4–0.5 bin overpredicts success (45.21% predicted versus 26.47% observed, n=34), while the 0.8–0.9 bin underpredicts it (84.94% versus 91.30%, n=69). Thus miscalibration is mixed; high false-success rates among unsuccessful trials alone do not establish overconfidence. Low-probability Phase III bins are small and noisy.

Phase III fixed-width reliability bins (empty-bin statistics are undefined):

| lower | upper | n | successes | mean_probability | observed_success_rate | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00000 | 0.10000 | 0 | 0 | nan | nan | nan |
| 0.10000 | 0.20000 | 3 | 1 | 0.15187 | 0.33333 | -0.18146 |
| 0.20000 | 0.30000 | 14 | 3 | 0.25566 | 0.21429 | 0.04137 |
| 0.30000 | 0.40000 | 10 | 3 | 0.35527 | 0.30000 | 0.05527 |
| 0.40000 | 0.50000 | 34 | 9 | 0.45214 | 0.26471 | 0.18743 |
| 0.50000 | 0.60000 | 57 | 35 | 0.55387 | 0.61404 | -0.06016 |
| 0.60000 | 0.70000 | 72 | 47 | 0.65104 | 0.65278 | -0.00174 |
| 0.70000 | 0.80000 | 94 | 69 | 0.75094 | 0.73404 | 0.01690 |
| 0.80000 | 0.90000 | 69 | 63 | 0.84939 | 0.91304 | -0.06366 |
| 0.90000 | 1.00000 | 10 | 10 | 0.91760 | 1.00000 | -0.08240 |

## Procedure and fairness

Five StratifiedGroupKFold splits use training trial_id as the group and phase × label as the stratification target. Trial IDs never cross fit/held folds; combined-phase rows stay together. Every fold refits the original TF-IDF/phase preprocessing and the same C=1 classifier using only that fold’s fit rows. Exactly one out-of-fold raw decision margin is generated per training row. No validation text or labels enter these fits.

Two prespecified calibration candidates are fitted to all training OOF margins: one global sigmoid and independent sigmoids by phase. Each is an unpenalized two-parameter logistic regression; Phase I/II/III provide 1,012/3,950/3,075 OOF rows, each with both classes. The calibrators are fitted before computing calibrated validation predictions. F1 uses >=0.5. No threshold search or other model changes are made.

For validation/test, both mappings transform raw margins from the exact original full-training baseline. Thus inference uses identical base predictions for all candidates. The OOF models see 80% of training trials whereas the frozen base saw 100%; this common non-ensemble cross-fitting approach can introduce a score-distribution mismatch. OOF scores are not in-sample margins. Calibration fit performance is not used to select a candidate. The external validation set is evaluation-only for these fits, but it has already been used for baseline selection, so it is not an independent final test. Selection here uses overall validation log loss.

## Validation comparison

| candidate | phase | log_loss | brier_score | f1 | roc_auc | average_precision |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | overall | 0.57375 | 0.19547 | 0.74790 | 0.76930 | 0.80508 |
| baseline | I | 0.56316 | 0.19103 | 0.77174 | 0.78327 | 0.83880 |
| baseline | II | 0.60146 | 0.20735 | 0.64655 | 0.73556 | 0.71390 |
| baseline | III | 0.53986 | 0.18089 | 0.82657 | 0.76707 | 0.86074 |
| sigmoid_pooled | overall | 0.57221 | 0.19493 | 0.74894 | 0.76930 | 0.80508 |
| sigmoid_pooled | I | 0.56027 | 0.19008 | 0.77174 | 0.78327 | 0.83880 |
| sigmoid_pooled | II | 0.60080 | 0.20705 | 0.64629 | 0.73556 | 0.71390 |
| sigmoid_pooled | III | 0.53765 | 0.18019 | 0.82810 | 0.76707 | 0.86074 |
| sigmoid_by_phase | overall | 0.57261 | 0.19507 | 0.74596 | 0.76848 | 0.80304 |
| sigmoid_by_phase | I | 0.57058 | 0.19365 | 0.77838 | 0.78327 | 0.83880 |
| sigmoid_by_phase | II | 0.60119 | 0.20725 | 0.64516 | 0.73556 | 0.71390 |
| sigmoid_by_phase | III | 0.53400 | 0.17884 | 0.82353 | 0.76707 | 0.86074 |

Pooled sigmoid changes overall log loss from 0.573753 to 0.572209 (delta -0.001545) and Phase III from 0.539862 to 0.537649. Overall Brier changes from 0.195465 to 0.194930; Phase III Brier from 0.180890 to 0.180189. Phase-specific sigmoid gives overall log loss 0.572610 and Phase III 0.533999. It helps Phase III more but worsens Phase I and has worse overall loss than pooled calibration. The overall improvements are small; this experiment does not establish statistical significance or guarantee improvement on later test trials. No calibration or temporal hyperparameter search was performed.

All sigmoid slopes are positive: within each phase, ranking and therefore ROC-AUC/AP are unchanged (up to floating-point ties). Changes in log loss/Brier reflect probability changes; F1 can change because the 0.5 decision boundary moves.

## Training-only sigmoid parameters

| candidate | group | slope | intercept | n_fit |
| --- | --- | --- | --- | --- |
| sigmoid_pooled | pooled | 1.07020 | -0.02013 | 8037 |
| sigmoid_by_phase | I | 0.86285 | 0.03638 | 1012 |
| sigmoid_by_phase | II | 1.03269 | 0.00386 | 3950 |
| sigmoid_by_phase | III | 1.26611 | -0.17715 | 3075 |

For p=expit(slope × raw_margin + intercept), slope <1 compresses margins, slope >1 expands them. Intercept adjusts prevalence. These slopes describe training OOF calibration and are not fitted evidence about validation Phase III alone.

Best validation candidate: **sigmoid_pooled**. Separate improving-candidate submissions: sigmoid_pooled_candidate_submission.csv, sigmoid_by_phase_candidate_submission.csv.

See validation_metrics.csv for all means, tails, Brier scores, ECE and predictive metrics; validation_calibration_bins.csv for all candidates overall/by phase; training_oof_scores.csv and fold_balance.csv for the grouped calibration-fit audit. metadata.json records preservation hashes and the method. No new broad search was performed.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.calibration
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m unittest part2.test_calibration
```
