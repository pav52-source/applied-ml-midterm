# Reproducible notebook code submission

The notebooks contain the actual source implementations. They do not import the
project's Python modules or shell out to run them. Existing source, notebooks,
models, results, submissions and reports are preserved. No `midterm/` directory
is created.

## Contents and recommended order

| Notebook | Source incorporated | Scope |
|---|---|---|
| `part1_reproduction.ipynb` | root `config.py`, `data_utils.py`, `models.py`, `pseudo_labeling.py`, `experiments.py`, `submission.py` | Standardization; K-Means/GMM; confidence/support filters; weighted LR; supervised, cluster-feature, per-label and random controls; threshold tuning/fallback; RBF SVM; K sweep; F1 tables and submission |
| `part2_modeling.ipynb` | `part2/config.py`, `data.py`, `audit.py`, `features.py`, `training.py`, `evaluation.py`, `experiments.py`, `submission.py`, `report.py` | Data/leakage audit, phase prior, all 36 LR settings, phase metrics, original winner, probability submission |
| `part2_error_analysis.ipynb` | common Part 2 data/features/training/evaluation/submission; root Part 1 preparation, K-Means and prediction code; `part2/error_analysis.py` and table helper from `report.py` | Truth/predicted/no-record/unavailable provenance; label=0 matching; 18 group cells; coverage; unresolved labels; rate and coverage figures |
| `part2_calibration.ipynb` | common Part 2 data/features/training/evaluation/submission; `part2/calibration.py`; table helper from `report.py` | Full-training fixed baseline; five grouped OOF fits; global/per-phase Platt calibration; Brier/reliability bins, predictive metrics, candidate models/submissions |
| `historical_workflow_reproduction.ipynb` | **All cells** of existing root `CS5785_Midterm_Starter.ipynb` | Distinct notebook-only research: 70-setting Part 1 suite, PCA/cluster/reason figures, Funding controls, K-NN/hybrid, separate per-phase Part 2 search, error comparison and LaTeX tables |

The first four notebooks represent the Python-module workflow discussed in this
session. Run them in the listed order for readability, but each is independently
executable: full error-analysis mode includes the exact Part 1 predictor needed
for its joins; full calibration mode reconstructs the fixed base model directly
from CSVs. Neither needs a previous notebook's kernel or a project `.py` file.
Do not run notebooks in one shared kernel: their independently copied helpers
intentionally reuse names.

The fifth notebook preserves a different existing workflow rather than omitting
its substantive research. Its K-NN settings and per-phase Part 2 selection are
**not** the pooled 0.57221 calibration experiment. Existing Markdown describes
historical findings; it is not a claim that verification mode reran those studies.
The historical workflow does not use the five-fold calibration method of the
primary calibration notebook; its original method is copied unchanged.

## Paths, datasets and safe outputs

Start each notebook from the repository root **or `part2/`**. The setup cell
recognizes both working directories without machine-specific absolute paths.
Preserve the relative layout:

- `part1-data/part1_train.csv`, `part1_val.csv`, `part1_test.csv`;
- `part2-data/outcome_train.csv`, `outcome_val.csv`, `outcome_test.csv`,
  `part2_sample_submission.csv`;
- `part1-data/pathway_catalog.csv` and `part2-data/pathway_catalog.csv`
  (optional for the module-based model; the historical workflow reads the Part 1 catalog).

**All newly computed files go under `part2/notebook_reproduction/`**, in separate
`part1/`, `part2_modeling/`, `part2_error_analysis/`, `part2_calibration/`, and
`historical_workflow/` subdirectories. These notebooks never write to original
`outputs/`, `part1-results/`, `part2-results/`, `part2/outputs/`,
`part2/calibration_outputs/`, or `writeup/`. Re-execution can overwrite files only
inside the new reproduction directory. Original-file SHA-256 checks run at the end.

## Modes

The first setup code cell has `RUN_FULL = False`.

- **Verification mode (default):** fits no classifiers or clustering algorithms.
  It loads datasets, defines the copied implementations, and explicitly reads
  historical result/prediction CSVs for comparison. Part 1 fits its inexpensive
  original standardizer only. Error analysis and calibration diagnostics are
  freshly recomputed from those fixed probabilities. Reliability and error plots
  are generated. These outputs are labeled as verification, not fresh training.
  The historical notebook-only workflow's execution blocks are skipped.
- **Complete reproduction:** set `RUN_FULL = True`, restart the kernel, and run
  all cells. No cached model or cached OOF scores substitute for fresh computation.
  The first four notebooks regenerate results from the dataset CSVs alone;
  historical artifact comparisons are optional when their files are absent.
  Full error analysis regenerates only the established Part 1 **test** predictions:
  it does not newly infer missing training reasons, preserving the 131-row gap.
  Every original grid remains executable. No new models or hyperparameters were added.

Default verification expects the corresponding original CSV artifacts under
`outputs/`, `part2/outputs/`, and `part2/calibration_outputs/`. Full mode does not
require those artifact directories or original `.py` files. The historical
workflow regenerates and reads its own intermediate files in its isolated folder.

## Environment and execution

The module-based Part 2 artifacts record:

```text
Python 3.14.8
numpy 2.5.3
scipy 1.18.1
pandas 3.0.6
scikit-learn 1.9.1
```

These numerical dependencies are pinned in the existing `part2/requirements.txt`.
The current environment also supplies matplotlib and IPython/ipykernel. For a
separate notebook environment, install those requirements and Jupyter tooling:

```sh
python -m pip install -r part2/requirements.txt matplotlib ipykernel jupyterlab nbformat nbclient
python -m jupyterlab
```

Choose that environment's Python kernel, open the desired notebook, set its mode,
and use **Restart Kernel and Run All Cells**. To execute with nbconvert, first
install `nbconvert` in your notebook environment, then run from the repository root:

```sh
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 part2/part2_modeling.ipynb
```

This uses the mode already saved in the notebook. Run each notebook separately;
no internet, API credentials, Kaggle CLI, or automated submission is involved.

The historical workflow's recorded environment differs slightly: Python 3.14.7,
numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, scikit-learn 1.9.1,
matplotlib 3.11.2, nbclient 0.11.0 (see original `part1-results/environment.json`
and `part2-results/environment.json`). Use those versions to reproduce that
workflow most closely. Library/platform differences can affect optimization,
threshold ties, and exact floating-point matches. Recorded timestamps and fit
seconds are not required to reproduce identically.

## Main results and candidate files

Read-only artifact checks verify these recorded module-based validation results:

- `criteria_year0_pooled_C1`: overall log loss **0.5737533226**;
- pooled sigmoid: **0.5722087688**;
- phase-specific sigmoid: **0.5726097525**;
- unresolved reasons: **131 phase rows / 113 unique trials**.

Full modeling mode executes all 37 baseline configurations and creates
`notebook_reproduction/part2_modeling/part2_submission.csv`. Full calibration
creates the two calibrated model pickles and separate candidate submissions,
including the recommended
`notebook_reproduction/part2_calibration/sigmoid_pooled_candidate_submission.csv`.
Full Part 1 creates `notebook_reproduction/part1/part1_submission.csv`.
All candidate submissions are checked for columns, IDs, row order and probabilities.
These are validation findings, not Kaggle leaderboard scores. Historical public
scores mentioned in the copied legacy Markdown belong only to that workflow.

Phase F1 uses threshold >=0.5. Overall validation loss is weighted by row count;
phase-macro loss uses equal phase weights, as does the equal-size phase test set.
Reason labels, embeddings and IDs never enter the Part 2 predictive features.
Reasons are used exclusively for post-hoc error analysis. Multi-label counts can
overlap, absent records are not masked records, and empty group rates are undefined.

## Notebook-only adaptations

- Replace `__file__`/absolute paths with root detection and isolated output paths.
- Remove project-specific imports because their actual definitions are embedded.
- Replace CLI parsers/`__main__` dispatch with notebook execution cells.
- Retain the algorithms and parameters; reuse the already fitted K=8 model within
  Part 1 full mode rather than deterministically fitting it twice.
- Recreate the identical baseline in full calibration mode instead of loading the
  original pickle. Each of its five OOF pipelines is still freshly fitted.
- Add explicit artifact verification, informative tables/figures, and full-mode guards.
- Rename colliding helpers only where required (`load_reason_data`, experiment and
  report entry points). Do not import the original modules at runtime.
- Omit test runners, CLI boilerplate and redundant report dispatch; all substantive
  algorithms, experiments, evaluation and prediction generation are embedded.

`notebook_source_manifest.json` records source hashes and copied definitions.
`notebook_validation.json` records what was actually checked/executed. Packaging
validation does not claim fresh full-training reproduction. The current environment
has neither nbformat nor nbclient; standard JSON/spec checks and in-process
IPython execution validate verification cells, without claiming a Jupyter-kernel
or official nbformat-schema run. Full training and the historical research suites
are deliberately not rerun during this source-conversion task.

## Validation performed and expected cost

All five new notebooks passed JSON/notebook-structure checks and Python compilation
for every code cell. All 160 code cells were run in verification mode in separate
in-process IPython sessions; their outputs and five figures are saved in the
notebooks. A guard rejected any attempted classifier or clustering `.fit`, so
this validation did not accidentally rerun research. The full-mode guards executed,
but their training bodies did not. The historical notebook's experiment and PCA
bodies were deliberately skipped. No full fresh-training notebook run is claimed.

Source checks compared 73 copied function/class occurrences exactly at the Python
AST level. Five renamed/adapted entry points are explicitly documented. Every
original historical code cell matches after only the documented path/guard changes.
Read-only checks confirmed the three key log losses, 131 unresolved phase rows /
113 trials, five-fold trial grouping, and candidate CSV format. SHA-256 comparison
confirmed all 174 pre-existing non-cache project files remained byte-for-byte
unchanged. Packaging validation also checked paths from both root and `part2/`.

No fresh training timings were benchmarked. Relative costs are: verification
(seconds, apart from first-time plotting/font setup); Part 1 full mode (A–H plus
six K-Means controls); Part 2 modeling (37 configurations, reusing features across
C); calibration (one fixed base fit plus five OOF base fits and four small sigmoid
fits); error analysis full mode (one outcome-model fit and one Part 1 K=8 fit).
The historical 70-setting Part 1 suite, K-NN controls and separate Part 2 suite are
the largest optional reproduction run. Exact time depends on hardware and numerical
libraries; increase the notebook execution timeout if needed.
