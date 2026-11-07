# Cross-part dependency audit

Authoritative commits and per-file SHA-256 values are in `provenance.json`.
The two branches have identical supplied datasets and the same root starter
notebook. Main's final Part 1 is the K-Means/K-NN hybrid in that notebook,
`PART1_NOTES.md`, and `part1-results/knn-experiment/`. Its 3-uniform-neighbor
candidate is authoritative; root `part1_submission.csv`,
`part1-results/reason_lookup.csv`, and the initial `selected_model.json` describe
an earlier K-Means model and are not final hybrid inputs.

| Component | Existing input/dependency | Origin and integration decision |
|---|---|---|
| `part2/data.py`, `features.py`, `training.py`, `experiments.py`, `evaluation.py`, `submission.py` | `part2-data/` only; criteria/diseases/drugs/ICD/year/phase allowlist | Michael's independent classifier. No Part 1 features; preserve byte-for-byte. No training rerun. |
| `part2/audit.py` | Part 2 schemas, text, IDs and years | No Part 1 dependency; preserve. |
| `part2/calibration.py` calibration fitting/inference | Training-only Part 2 OOF margins/labels; frozen base model | No Part 1 labels, embeddings or predictions. Preserve models and submissions; no recalibration needed. |
| `part2/calibration.py` preservation hashes | Traverses `part1-data/`, root `outputs/`, and source files | Administrative fingerprints only, not predictive inputs. Historical hashes refer to Michael's old tree; retain as historical provenance. |
| `part2/error_analysis.py` | True `part1-data/` labels; **root `outputs/part1_submission.csv`**; Part 2 saved probabilities | Raw truth is identical on both branches. Old predictions were Michael's root-module Part 1. Original script/artifacts preserved; final analysis uses new `integration/reconcile_reasons.py`. |
| `part2/outputs/stopping_reason_*` | Old lookup: true labels + old test predictions; masked training rows unavailable | Preserve original results for comparison. Recompute lookup, membership, 18 phase/group cells, coverage, means, sources and figures for baseline and both calibrations. |
| `part2/notebook_reproduction/part2_error_analysis/` | Same old reason source | Historical verification outputs, not authoritative final deliverables. Preserve; use integration outputs instead. |
| `part2/part2_error_analysis.ipynb` | Embeds old Part 1 K=8 implementation and old `outputs/` prediction path | Preserve as historical; exclude from final notebook set. New final notebook embeds the authoritative join and uses main's hybrid artifacts. |
| `part2/part1_reproduction.ipynb` | Michael's obsolete root `config.py`, models and K=8 pipeline copied inline | Preserve archive only; do not submit as final Part 1. Final Part 1 notebook copies main's Part 1 cells. |
| `part2/historical_workflow_reproduction.ipynb` | Complete main-like workflow, including a separate per-phase Part 2 | Preserve archive only. Final notebook set splits authoritative Part 1 from Michael's Part 2; avoids competing final classifiers. |
| `part2/part2_modeling.ipynb`, `part2/part2_calibration.ipynb` | Part 2 datasets and historical Part 2 metrics; generic preservation scans | Copy to final deliverables with path/output adapters only. No predictive dependency on Part 1. |
| `part2/test_error_analysis.py` | Synthetic labels/provenance fixtures | Tests original pure join/summary behavior; no old predictor needed. |
| `part2/test_calibration.py::test_saved_artifacts` | Historical full-tree hashes, including Michael-only root Part 1 files | Cannot pass on a deliberately different integration tree. Preserve original test and report this incompatibility. Run all other original tests; new integration verification replaces this historical-tree assertion and checks models/predictions directly. |
| Root `CS5785_Midterm_Starter.ipynb` and `part2-results/` | Teammate's separate Part 2 models/analysis | Preserve main's files unchanged, but do not use as the selected Part 2 approach or attribute its Kaggle score to Michael's candidates. |

## Authoritative integration inputs

1. Visible 0/1 labels from `part1-data/part1_train.csv` and `part1_val.csv`:
   always use truth in preference to estimates.
2. `part1-results/knn-experiment/best_augmented_training_labels.csv`:
   use `pseudo_train` values only where all original training labels are -100;
   verify observed rows remain exactly equal to truth.
3. `part1-results/knn-experiment/best_augmented_candidate_submission.csv`:
   parse the five exact reason names; its IDs must equal the Part 1 test IDs.
4. Match Part 1 ID to Part 2 `trial_id`, not the phase-suffixed row ID.
   Keep combined-phase rows together at trial level and count once per phase/group.
5. Absence from all three Part 1 splits means no recorded stopping reason.
   Present-but-unavailable labels remain unresolved if estimates are missing;
   they must never become negative labels or no-record rows.

The saved hybrid covers all 113 previously unresolved trials (131 phase rows).
They are resolved as **predictions**, not ground truth. No outcome training,
features, coefficients, calibration maps, or probabilities are changed.

## Recomputed outputs

`integration/outputs/` contains authoritative lookup, trial/source audit, coverage,
all three candidates' group tables, original-vs-integrated baseline comparison,
baseline-vs-pooled-calibration comparison, figures, and input fingerprints.
The original Michael artifacts remain in `part2/outputs/` and
`part2/calibration_outputs/`. Old root Part 1 modules and root `outputs/` are not
copied from Michael; they remain available in Git history at the recorded commit.

A text-reference inventory is recorded in `part1_reference_inventory.csv`. A
string match is a dependency lead, not proof of predictive use; the table above
classifies the actual uses.
