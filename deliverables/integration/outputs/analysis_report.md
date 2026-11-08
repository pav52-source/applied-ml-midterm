# Integrated stopping-reason error analysis

Part 1 is the final origin/main hybrid. Part 2 probabilities are exactly the frozen origin/michael artifacts. Only true outcome label=0 rows are analyzed; probability >=0.5 is a false success. Visible training/validation reasons take precedence over predictions. Masked training reasons now use the teammate’s saved hybrid pseudo-labels; test reasons use the hybrid’s submitted predictions. Root part1_submission.csv and part1-results/reason_lookup.csv describe earlier models and are not used.

**All 131 previously unresolved phase rows (113 trials) are now covered by predictions, not truth.** There remain 27 rows with true reasons, 171 with predicted reasons and 254 without a Part 1 record. No record is not a claim of successful completion. Multi-label categories overlap; each trial appears at most once per phase/group. Empty groups have undefined rates.

## Coverage (baseline; membership identical for calibrated candidates)

| phase | n_trials | n_predicted_success | false_success_rate | mean_predicted_success_probability | n_ground_truth | n_predictions | n_no_record | n_reasons_unavailable | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | 64 | 28 | 0.43750 | 0.46540 | 4 | 32 | 28 | 0 | baseline |
| II | 265 | 78 | 0.29434 | 0.40702 | 16 | 105 | 144 | 0 | baseline |
| III | 123 | 78 | 0.63415 | 0.55506 | 7 | 34 | 82 | 0 | baseline |

## All candidate group metrics

| phase | reason_group | n_trials | n_predicted_success | false_success_rate | mean_predicted_success_probability | n_ground_truth | n_predictions | n_no_record | n_reasons_unavailable | candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | Safety | 8 | 4 | 0.50000 | 0.49100 | 1 | 7 | 0 | 0 | baseline |
| I | Funding | 3 | 2 | 0.66667 | 0.50748 | 1 | 2 | 0 | 0 | baseline |
| I | Efficacy | 4 | 3 | 0.75000 | 0.56704 | 0 | 4 | 0 | 0 | baseline |
| I | Recruitment | 24 | 9 | 0.37500 | 0.43650 | 2 | 22 | 0 | 0 | baseline |
| I | Other | 3 | 1 | 0.33333 | 0.41318 | 0 | 3 | 0 | 0 | baseline |
| I | no recorded stopping reason | 28 | 13 | 0.46429 | 0.47444 | 0 | 0 | 28 | 0 | baseline |
| II | Safety | 11 | 3 | 0.27273 | 0.40669 | 1 | 10 | 0 | 0 | baseline |
| II | Funding | 4 | 0 | 0.00000 | 0.34167 | 1 | 3 | 0 | 0 | baseline |
| II | Efficacy | 15 | 6 | 0.40000 | 0.44421 | 0 | 15 | 0 | 0 | baseline |
| II | Recruitment | 93 | 18 | 0.19355 | 0.37512 | 12 | 81 | 0 | 0 | baseline |
| II | Other | 15 | 4 | 0.26667 | 0.42107 | 3 | 12 | 0 | 0 | baseline |
| II | no recorded stopping reason | 144 | 52 | 0.36111 | 0.42112 | 0 | 0 | 144 | 0 | baseline |
| III | Safety | 5 | 2 | 0.40000 | 0.48321 | 1 | 4 | 0 | 0 | baseline |
| III | Funding | 0 | 0 | nan | nan | 0 | 0 | 0 | 0 | baseline |
| III | Efficacy | 8 | 7 | 0.87500 | 0.69438 | 0 | 8 | 0 | 0 | baseline |
| III | Recruitment | 25 | 14 | 0.56000 | 0.51204 | 5 | 20 | 0 | 0 | baseline |
| III | Other | 9 | 6 | 0.66667 | 0.58854 | 2 | 7 | 0 | 0 | baseline |
| III | no recorded stopping reason | 82 | 53 | 0.64634 | 0.55953 | 0 | 0 | 82 | 0 | baseline |
| I | Safety | 8 | 4 | 0.50000 | 0.48717 | 1 | 7 | 0 | 0 | sigmoid_pooled |
| I | Funding | 3 | 2 | 0.66667 | 0.50301 | 1 | 2 | 0 | 0 | sigmoid_pooled |
| I | Efficacy | 4 | 3 | 0.75000 | 0.56661 | 0 | 4 | 0 | 0 | sigmoid_pooled |
| I | Recruitment | 24 | 9 | 0.37500 | 0.42832 | 2 | 22 | 0 | 0 | sigmoid_pooled |
| I | Other | 3 | 1 | 0.33333 | 0.40423 | 0 | 3 | 0 | 0 | sigmoid_pooled |
| I | no recorded stopping reason | 28 | 13 | 0.46429 | 0.46874 | 0 | 0 | 28 | 0 | sigmoid_pooled |
| II | Safety | 11 | 3 | 0.27273 | 0.39769 | 1 | 10 | 0 | 0 | sigmoid_pooled |
| II | Funding | 4 | 0 | 0.00000 | 0.32946 | 1 | 3 | 0 | 0 | sigmoid_pooled |
| II | Efficacy | 15 | 6 | 0.40000 | 0.43822 | 0 | 15 | 0 | 0 | sigmoid_pooled |
| II | Recruitment | 93 | 15 | 0.16129 | 0.36413 | 12 | 81 | 0 | 0 | sigmoid_pooled |
| II | Other | 15 | 4 | 0.26667 | 0.41450 | 3 | 12 | 0 | 0 | sigmoid_pooled |
| II | no recorded stopping reason | 144 | 51 | 0.35417 | 0.41313 | 0 | 0 | 144 | 0 | sigmoid_pooled |
| III | Safety | 5 | 2 | 0.40000 | 0.47756 | 1 | 4 | 0 | 0 | sigmoid_pooled |
| III | Funding | 0 | 0 | nan | nan | 0 | 0 | 0 | 0 | sigmoid_pooled |
| III | Efficacy | 8 | 7 | 0.87500 | 0.70079 | 0 | 8 | 0 | 0 | sigmoid_pooled |
| III | Recruitment | 25 | 14 | 0.56000 | 0.50910 | 5 | 20 | 0 | 0 | sigmoid_pooled |
| III | Other | 9 | 6 | 0.66667 | 0.58912 | 2 | 7 | 0 | 0 | sigmoid_pooled |
| III | no recorded stopping reason | 82 | 52 | 0.63415 | 0.55850 | 0 | 0 | 82 | 0 | sigmoid_pooled |
| I | Safety | 8 | 4 | 0.50000 | 0.49867 | 1 | 7 | 0 | 0 | sigmoid_by_phase |
| I | Funding | 3 | 2 | 0.66667 | 0.51550 | 1 | 2 | 0 | 0 | sigmoid_by_phase |
| I | Efficacy | 4 | 3 | 0.75000 | 0.56715 | 0 | 4 | 0 | 0 | sigmoid_by_phase |
| I | Recruitment | 24 | 9 | 0.37500 | 0.45248 | 2 | 22 | 0 | 0 | sigmoid_by_phase |
| I | Other | 3 | 1 | 0.33333 | 0.43122 | 0 | 3 | 0 | 0 | sigmoid_by_phase |
| I | no recorded stopping reason | 28 | 13 | 0.46429 | 0.48551 | 0 | 0 | 28 | 0 | sigmoid_by_phase |
| II | Safety | 11 | 3 | 0.27273 | 0.40508 | 1 | 10 | 0 | 0 | sigmoid_by_phase |
| II | Funding | 4 | 0 | 0.00000 | 0.33861 | 1 | 3 | 0 | 0 | sigmoid_by_phase |
| II | Efficacy | 15 | 6 | 0.40000 | 0.44409 | 0 | 15 | 0 | 0 | sigmoid_by_phase |
| II | Recruitment | 93 | 18 | 0.19355 | 0.37272 | 12 | 81 | 0 | 0 | sigmoid_by_phase |
| II | Other | 15 | 4 | 0.26667 | 0.42053 | 3 | 12 | 0 | 0 | sigmoid_by_phase |
| II | no recorded stopping reason | 144 | 53 | 0.36806 | 0.42014 | 0 | 0 | 144 | 0 | sigmoid_by_phase |
| III | Safety | 5 | 2 | 0.40000 | 0.44129 | 1 | 4 | 0 | 0 | sigmoid_by_phase |
| III | Funding | 0 | 0 | nan | nan | 0 | 0 | 0 | 0 | sigmoid_by_phase |
| III | Efficacy | 8 | 7 | 0.87500 | 0.69850 | 0 | 8 | 0 | 0 | sigmoid_by_phase |
| III | Recruitment | 25 | 13 | 0.52000 | 0.48173 | 5 | 20 | 0 | 0 | sigmoid_by_phase |
| III | Other | 9 | 5 | 0.55556 | 0.56882 | 2 | 7 | 0 | 0 | sigmoid_by_phase |
| III | no recorded stopping reason | 82 | 47 | 0.57317 | 0.53457 | 0 | 0 | 82 | 0 | sigmoid_by_phase |

## Changes from Michael’s original baseline analysis

| phase | reason_group | n_trials_old | n_trials_integrated | false_success_rate_old | false_success_rate_integrated |
| --- | --- | --- | --- | --- | --- |
| I | Safety | 2 | 8 | 0.50000 | 0.50000 |
| I | Funding | 1 | 3 | 0.00000 | 0.66667 |
| I | Efficacy | 0 | 4 | nan | 0.75000 |
| I | Recruitment | 7 | 24 | 0.28571 | 0.37500 |
| I | Other | 3 | 3 | 0.66667 | 0.33333 |
| I | no recorded stopping reason | 28 | 28 | 0.46429 | 0.46429 |
| II | Safety | 5 | 11 | 0.60000 | 0.27273 |
| II | Funding | 2 | 4 | 0.00000 | 0.00000 |
| II | Efficacy | 3 | 15 | 0.33333 | 0.40000 |
| II | Recruitment | 31 | 93 | 0.19355 | 0.19355 |
| II | Other | 8 | 15 | 0.37500 | 0.26667 |
| II | no recorded stopping reason | 144 | 144 | 0.36111 | 0.36111 |
| III | Safety | 1 | 5 | 0.00000 | 0.40000 |
| III | Funding | 0 | 0 | nan | nan |
| III | Efficacy | 1 | 8 | 1.00000 | 0.87500 |
| III | Recruitment | 9 | 25 | 0.44444 | 0.56000 |
| III | Other | 4 | 9 | 0.75000 | 0.66667 |
| III | no recorded stopping reason | 82 | 82 | 0.64634 | 0.64634 |

Reason-specific changes arise from expanded coverage and replacement of older test-reason predictions, not changed outcome probabilities. No-record group membership is unchanged. Pseudo-labels and test predictions have errors; larger groups improve descriptive denominators but do not turn inferred reasons into ground truth.

## Baseline versus the retained pooled calibration

| phase | reason_group | n_trials_baseline | false_success_rate_baseline | false_success_rate_calibrated |
| --- | --- | --- | --- | --- |
| I | Safety | 8 | 0.50000 | 0.50000 |
| I | Funding | 3 | 0.66667 | 0.66667 |
| I | Efficacy | 4 | 0.75000 | 0.75000 |
| I | Recruitment | 24 | 0.37500 | 0.37500 |
| I | Other | 3 | 0.33333 | 0.33333 |
| I | no recorded stopping reason | 28 | 0.46429 | 0.46429 |
| II | Safety | 11 | 0.27273 | 0.27273 |
| II | Funding | 4 | 0.00000 | 0.00000 |
| II | Efficacy | 15 | 0.40000 | 0.40000 |
| II | Recruitment | 93 | 0.19355 | 0.16129 |
| II | Other | 15 | 0.26667 | 0.26667 |
| II | no recorded stopping reason | 144 | 0.36111 | 0.35417 |
| III | Safety | 5 | 0.40000 | 0.40000 |
| III | Funding | 0 | nan | nan |
| III | Efficacy | 8 | 0.87500 | 0.87500 |
| III | Recruitment | 25 | 0.56000 | 0.56000 |
| III | Other | 9 | 0.66667 | 0.66667 |
| III | no recorded stopping reason | 82 | 0.64634 | 0.63415 |

Calibration optimizes probability quality, not every thresholded subgroup rate. The same integrated group membership is used before/after calibration. Full subgroup provenance is retained in unsuccessful_trial_audit.csv and reason_groups_by_source.csv. No reason label or embedding enters the outcome classifier or calibrator.
