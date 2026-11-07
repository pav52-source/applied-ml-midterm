# Final integrated notebook deliverables

The four authoritative notebooks in `deliverables/notebooks/` (or at the root of
an assembled `midterm/` folder) are:

1. `part1_final.ipynb`: authoritative main-branch Part 1, including all 99 existing
   configurations, feature ablations, PCA, Funding controls, K-NN/hybrid, and final
   validation/test predictions. This does not use Michael's obsolete K=8 code.
2. `part2_modeling_final.ipynb`: Michael's phase prior and 36 controlled LR settings.
3. `part2_error_analysis_final.ipynb`: updated joins using the final Part 1 hybrid;
   full coverage, source labels, old/new comparisons, and before/after-calibration
   error analysis. It recomputes the analysis without fitting any model.
4. `part2_calibration_final.ipynb`: Michael's five trial-grouped OOF folds, pooled
   and phase-specific sigmoids, metrics, reliability plots and candidate generation.

Two additional Part 1 notebooks document experiments; neither changes the selected
submission or replaces `part1_final.ipynb`:

- `part1_kmeans_experiments.ipynb`: Prith's cluster-count, selective label-recovery,
  weighting, and regularization comparisons.
- `part1_hybrid_clustering_experiments.ipynb`: an exact modular reproduction of the
  selected Part 1 hybrid, followed by KernelPCA and kernel K-means comparisons.

Each contains the substantive implementation directly. No project-module import
is needed to execute a notebook. Use a separate fresh kernel per notebook.
Original notebooks in `part2/` are preserved archives; several include Michael's
older Part 1 or main's competing Part 2 approach and are not final deliverables.
Root `CS5785_Midterm_Starter.ipynb` is preserved as main's original notebook; only
its Part 1 is authoritative for this integration.

## Execution

Run from inside the repository or assembled `midterm/` folder. Notebooks find the
nearest ancestor containing `part1-data/` and `part2-data/`. Install the supplied
requirements and open JupyterLab. Select the correct Python environment/kernel.

For the four authoritative notebooks, default mode is `RUN_FULL=False`: read-only
artifact verification plus freshly computed audits/figures. It does not fit
project classifiers. Full mode on the final Part 1, modeling, and calibration
notebooks reruns the existing research using dataset CSVs, with unchanged
algorithms and parameters. It may take minutes or longer; full-suite timings
were not re-benchmarked during integration. Error analysis uses existing
authoritative predictions and always recomputes the joins/metrics. The two
supplementary notebooks are experiment records and do not use `RUN_FULL`; running
their model cells fits the corresponding experiments.

New notebook outputs stay under `deliverables/reproduced/`; the supplementary
Part 1 notebooks write to `deliverables/reproduced/part1_supplementary/`.
Some saved supplementary notebook output cells show paths from their original
runs; rerunning the copied notebooks uses the supplementary output directory.
Original source and model artifacts are never overwritten. To reproduce calibrated
submissions in full mode, run the calibration notebook; it recreates the same
fixed full-training baseline plus five OOF fits. Do not rerun a broad experiment
suite merely to view or recompute tables. Model pickles must only be loaded from
this trusted project.

The final analysis notebook uses preserved baseline/calibration probability CSVs
and authoritative Part 1 hybrid artifacts. After a deliberate new model experiment,
update its input paths explicitly; do not silently mix old and new predictions.
The existing outcome model does not use reason labels or embeddings as features.

## Correct submission artifacts

- Part 1: `part1-results/knn-experiment/best_augmented_candidate_submission.csv`
  (312 rows, `ID,Predicted`). Root `part1_submission.csv` is the older K-Means file.
- Part 2: `part2/calibration_outputs/sigmoid_pooled_candidate_submission.csv`
  (1,500 rows, `ID,label`). This is Michael's selected classifier, not main's
  root-level Part 2 submission.

Nothing is uploaded by the notebooks or assembly script. Validation log losses
are 0.573753 (raw), 0.572209 (pooled sigmoid), 0.572610 (phase-specific sigmoid).
Pooled calibration phase-macro loss is 0.566240. The Part 1 recorded public F1 is
0.87927; a public score for Michael's final Part 2 candidate is not recorded here.
The unrelated main-branch Part 2 public score must not be assigned to it.

## Build the Gradescope folder

From the integration repository root, choose a new destination outside the repo:

```sh
python -m integration.assemble_gradescope /tmp/cs5785-final/midterm
```

The command refuses an existing destination. It copies the four final notebooks,
the two supplementary Part 1 notebooks, datasets, final Part 1 results, original
Part 2 artifacts and modules, integrated analyses, report sources/PDF, dependencies
and provenance. It excludes the obsolete
Michael-only Part 1 modules/notebooks and the alternative main-branch Part 2
research. Inspect the assembled `midterm/` folder, then upload manually according
to Gradescope. Do not upload `.git/`, virtual environments or transient caches.

All dataset files needed by notebooks are present: three Part 1 splits, three
outcome splits, outcome sample submission, and pathway catalogs. For reproducible
full retraining, also consult `part1-results/environment.json` and
`part2/outputs/run_metadata.json`: contributors recorded slightly different
numpy/pandas versions. No full training rerun under a combined environment is
claimed; saved-model predictions and validation metrics were verified exactly.

## Verification and remaining report tasks

The four authoritative final notebooks were executed in verification mode through
in-process IPython. JSON structure, imports and code compilation passed; the error
notebook recomputed all 54 candidate/phase/group cells. The environment lacked nbformat/
nbclient, so official kernel-driven execution was not performed. Full training
bodies remain unexecuted during integration. The two supplementary notebooks were
added afterward; their code cells parse, but their full experiments were not
rerun as part of this deliverable packaging. See `integration/outputs/` for the
authoritative notebook checks.

Original Michael `part2/error_analysis.py` is historical and expects the older
root `outputs/` input. Use the new final error notebook or
`python -m integration.reconcile_reasons` for the authoritative join. The original
calibration test's historical whole-tree fingerprint assertion is inapplicable
after integration; nine other behavior tests pass, and saved inference/artifacts
were independently checked. No original test/source was rewritten to hide this.

The report draft is `writeup/main.pdf` and `writeup/main.tex`.
Fill full team names, confirm each contribution and the AI-disclosure wording,
and add a verified Part 2 leaderboard result only if available. The assignment
PDF is absent; requirements were read from `Midterm_Project.tex`. The draft uses
the CVPR style corresponding to the assignment's current Overleaf template link
and is four pages including references. Review before submission.
