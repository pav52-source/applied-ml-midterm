# Part 1 test prediction discrepancies

Compared the latest personal all-five-C=0.1 K21 submission with the teammate selected K-means/KNN hybrid. Both files contain the same 312 unique test IDs, valid nonempty reason sets, and no test ground truth. Files were joined by `ID`; label order was ignored when checking exact set agreement.

- Exact reason-set agreement: **231/312 (74.0%)**; disagreement: **81/312 (26.0%)**.
- Total individual reason-bit disagreements across all trials: **111**.
- Earlier personal K21 C1 submission versus teammate: **232/312** exact agreement (80 disagreements). This is context only; it is a different model and fallback policy.

## Per-reason assignments

| Reason | Mine positive | Teammate positive | Both positive | Mine only | Teammate only | Jaccard |
|---|---:|---:|---:|---:|---:|---:|
| Safety | 43 | 49 | 43 | 0 | 6 | 0.878 |
| Funding | 6 | 6 | 6 | 0 | 0 | 1.000 |
| Efficacy | 76 | 47 | 46 | 30 | 1 | 0.597 |
| Recruitment | 168 | 191 | 168 | 0 | 23 | 0.880 |
| Other | 89 | 44 | 41 | 48 | 3 | 0.446 |

Funding assignments agree exactly. Every personal Safety and Recruitment positive is also a teammate positive. The largest differences are personal-only Efficacy (30 trials) and Other (48), plus teammate-only Recruitment (23). These are prediction differences, not measured errors.

## Most frequent exact-set transitions (mine → teammate)

| Mine | Teammate | Trials |
|---|---|---:|
| Recruitment Other | Recruitment | 15 |
| Safety Other | Safety | 11 |
| Efficacy Other | Efficacy | 7 |
| Safety Efficacy | Safety | 7 |
| Efficacy | Recruitment | 6 |
| Other | Recruitment Other | 5 |
| Efficacy Other | Other | 5 |
| Other | Recruitment | 5 |
| Efficacy Other | Recruitment | 3 |
| Efficacy | Safety | 3 |

## Example disagreement IDs

TR05267, TR09553, TR06231, TR03479, TR10699, TR09077, TR06460, TR01153, TR03193, TR05009, TR07047, TR04082

Full row-level comparison: [`test_prediction_discrepancies.csv`](test_prediction_discrepancies.csv). The `Mine_only` and `Teammate_only` columns identify labels unique to each prediction. Without test labels, these comparisons cannot show which predictions are correct.

Sources:

- Personal latest: `prith_attempts/part_1/part1_kmeans_blend_sfo_k21_all_five_c01_submission.csv`
- Teammate selected: `part1-results/knn-experiment/best_augmented_candidate_submission.csv`
- Personal earlier K21 C1: `prith_attempts/part_1/part1_kmeans_blend_sfo_k21_weight025_threshold30_submission.csv`
