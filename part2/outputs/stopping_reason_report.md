# Stopping-reason error analysis of the fixed Part 2 winner

Uses only true outcome label=0 validation rows. A false success is probability >= 0.5. The winner remains criteria_year0_pooled_C1 (validation log loss 0.57375). No training, model loading, or prediction changes were performed.

## Matching and reason provenance

452 unsuccessful phase rows represent 398 unique trials. Part 1 ID has format TR followed by digits and equals Part 2 trial_id. The phase-suffixed Part 2 ID is checked against trial_id + phase. Saved outcome labels, phases and trial IDs are checked against outcome_val.csv with a one-to-one join. Part 1 split IDs are disjoint and unique; saved reason predictions cover exactly Part 1 test IDs. The reason join is many-to-one and each phase has at most one row per trial.

Available binary ground-truth labels from Part 1 train/val take precedence. Predicted reasons come only from the existing outputs/part1_submission.csv for Part 1 test. Part 1 train labels masked as -100 remain unknown: no saved predictions or fitted Part 1 model were found for them, and Part 1 training is not rerun. Only trials absent from all three Part 1 splits enter “no recorded stopping reason”; this is an assignment-defined absence-of-record group, not proof that the trial completed normally.

| phase | n_trials | n_predicted_success | false_success_rate | mean_predicted_success_probability | n_ground_truth | n_predictions | n_no_record | n_reasons_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | 64 | 28 | 0.43750 | 0.46540 | 4 | 7 | 28 | 25 |
| II | 265 | 78 | 0.29434 | 0.40702 | 16 | 27 | 144 | 78 |
| III | 123 | 78 | 0.63415 | 0.55506 | 7 | 6 | 82 | 28 |

## Required six groups

The five category rows include only known positive reasons, either ground truth or saved predictions. Their counts are incomplete wherever matched reasons are unavailable; unknown values are not negatives. Multi-label trials appear in each applicable category, so category counts overlap. Combined-phase trials can contribute once in each phase. Undefined empty-group rates and means are NaN (blank in CSV), not zero. Provenance counts are phase-row counts.

| phase | reason_group | n_trials | n_predicted_success | false_success_rate | mean_predicted_success_probability | n_ground_truth | n_predictions | n_no_record | n_reasons_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | Safety | 2 | 1 | 0.50000 | 0.52110 | 1 | 1 | 0 | 0 |
| I | Funding | 1 | 0 | 0.00000 | 0.45221 | 1 | 0 | 0 | 0 |
| I | Efficacy | 0 | 0 | nan | nan | 0 | 0 | 0 | 0 |
| I | Recruitment | 7 | 2 | 0.28571 | 0.36636 | 2 | 5 | 0 | 0 |
| I | Other | 3 | 2 | 0.66667 | 0.55746 | 0 | 3 | 0 | 0 |
| I | no recorded stopping reason | 28 | 13 | 0.46429 | 0.47444 | 0 | 0 | 28 | 0 |
| II | Safety | 5 | 3 | 0.60000 | 0.52256 | 1 | 4 | 0 | 0 |
| II | Funding | 2 | 0 | 0.00000 | 0.36941 | 1 | 1 | 0 | 0 |
| II | Efficacy | 3 | 1 | 0.33333 | 0.42094 | 0 | 3 | 0 | 0 |
| II | Recruitment | 31 | 6 | 0.19355 | 0.34515 | 12 | 19 | 0 | 0 |
| II | Other | 8 | 3 | 0.37500 | 0.51224 | 3 | 5 | 0 | 0 |
| II | no recorded stopping reason | 144 | 52 | 0.36111 | 0.42112 | 0 | 0 | 144 | 0 |
| III | Safety | 1 | 0 | 0.00000 | 0.42827 | 1 | 0 | 0 | 0 |
| III | Funding | 0 | 0 | nan | nan | 0 | 0 | 0 | 0 |
| III | Efficacy | 1 | 1 | 1.00000 | 0.74317 | 0 | 1 | 0 | 0 |
| III | Recruitment | 9 | 4 | 0.44444 | 0.45256 | 5 | 4 | 0 | 0 |
| III | Other | 4 | 3 | 0.75000 | 0.61828 | 2 | 2 | 0 | 0 |
| III | no recorded stopping reason | 82 | 53 | 0.64634 | 0.55953 | 0 | 0 | 82 | 0 |

## Matched records with unavailable reasons (outside the six groups)

| phase | reason_group | n_trials | n_predicted_success | false_success_rate | mean_predicted_success_probability | n_ground_truth | n_predictions | n_no_record | n_reasons_unavailable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | matched Part 1 record; reasons unavailable | 25 | 11 | 0.44000 | 0.47033 | 0 | 0 | 0 | 25 |
| II | matched Part 1 record; reasons unavailable | 78 | 15 | 0.19231 | 0.39072 | 0 | 0 | 0 | 78 |
| III | matched Part 1 record; reasons unavailable | 28 | 18 | 0.64286 | 0.56791 | 0 | 0 | 0 | 28 |

## Interpretation

- Phase I: highest observed fraction — Other: 2/3 (66.7%).
- Phase II: highest observed fraction — Safety: 3/5 (60.0%).
- Phase III: highest observed fraction — Efficacy: 1/1 (100.0%).

Among larger groups, Phase III no recorded stopping reason has 53/82 false successes (64.6%), versus 13/28 (46.4%) in Phase I and 52/144 (36.1%) in Phase II. This is the clearest descriptive concentration of errors, although phases have different outcome definitions and trial mixes. Phase II Recruitment has 6/31 false successes (19.4%). Phase I Other (2/3), Phase II Safety (3/5), and Phase III Efficacy (1/1) lead their phases but are all too small for strong conclusions. Phase I Other and Phase III Efficacy are entirely based on predicted reasons. There are 131 unresolved unsuccessful phase rows, so the reason-specific results describe the available subset rather than all stopped trials.

Cells with fewer than 10 trials are especially fragile: one trial changes a fraction by at least 10 percentage points. Larger groups offer more stable descriptive evidence, but these comparisons are not significance tests or causal effects. Missing labels may be systematically different from available labels, and predicted Part 1 reasons introduce classification error. The source-stratified CSV permits inspection without conflating truth and predictions. Do not rank complete stopping-reason populations from these incomplete category counts.

## Verification and reproduction

Independent row-wise membership checks verify all 18 cells, integer false-success counts, rates, mean probabilities, and empty-group behavior. Coverage counts partition all unsuccessful validation rows into truth, predictions, no record, and unavailable reasons. SHA-256 checks confirm existing input/output artifacts did not change.

Run from the repository root:

```sh
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.error_analysis
```

Outputs: stopping_reason_groups.csv (18 requested cells), stopping_reason_coverage.csv, stopping_reason_unresolved.csv, stopping_reason_by_source.csv, stopping_reason_trial_audit.csv, stopping_reason_reason_lookup.csv and stopping_reason_verification.json. The last two audit tables retain unknown reasons as missing values. No reasons are predictive inputs. Analysis stops here; no model improvements are implemented.
