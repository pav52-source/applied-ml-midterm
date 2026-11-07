# Integration summary

## Branches and provenance

- Integration branch: `midterm-integration`, created exactly at fetched `origin/main`
  commit `0233983c491ae36f96516085e82d15b8a47c7eaf`. Its inherited upstream was removed; it has no push target.
- Part 2 source: `origin/michael` commit `6078a3f0a390500d9518424dac925f38269b3459`.
- Only the `part2/` subtree was restored from Michael. Main's files remain intact.
- Per-file source commits and SHA-256 values are in `provenance.json`:
  110 files from main and 85 files from Michael.
- Existing local `main` and `michael` branches were not reset, rewritten or changed.
  No push, Kaggle submission, or Gradescope upload occurred.
- Integration additions are left **uncommitted** for review on the new branch;
  HEAD still identifies the starting main commit. This is intentional.

## Authoritative methods

Part 1 is main's final hybrid: training-standardized embeddings, 10-cluster
K-Means smoothed reason scores equally blended with three uniform labeled
neighbors, recovery threshold 0.3 plus nonempty fallback, five LR classifiers
(C=0.1, pseudo-row weight 0.25). Validation sample/macro F1 are
0.886752/0.779234. Recorded public F1 is 0.87927 in main's notes (not re-queried).
The source includes the 70 initial configurations, four Funding controls and
25 local recovery controls. Final prediction/label files are under
`part1-results/knn-experiment/`. No fitted Part 1 model pickle is available or
needed for this integration.

Part 2 is Michael's pooled eligibility TF-IDF LR plus phase, C=1, no year;
row-weighted validation log loss is 0.573753. Five-fold training-only grouped OOF
sigmoids give 0.572209 pooled and 0.572610 phase-specific. Pooled calibration is
retained; its equal-phase validation loss is 0.566240. All original pipelines,
model pickles, probabilities, experiments and submissions are copied unchanged.
No Part 1 reason information enters predictive features.

Michael's older root Part 1 modules and `outputs/` were deliberately not copied.
His Part 1/historical notebooks remain archived within the preserved `part2/`
subtree but are excluded from final deliverables. Main's competing Part 2 notebook,
`part2-results/`, report, and root submission remain preserved as historical work;
their different methods and public score are not assigned to Michael's model.

## Recomputed analysis

`DEPENDENCY_AUDIT.md` identifies all cross-part uses. Only reason-dependent
post-hoc analyses required recomputation. `reconcile_reasons.py` builds a fresh
lookup from observed train/validation labels, final hybrid train pseudo-labels,
and final hybrid test predictions. It verifies unique IDs and once-per-phase
counting, and preserves overlapping multi-label groups and empty-group rates.

All **131 formerly unresolved phase rows / 113 trials** are now covered as
predictions, not truth. Of 452 unsuccessful phase rows (398 trials), 27 have true
reasons, 171 predicted reasons and 254 no record; no matched rows remain unresolved.
`outputs/` includes all 54 candidate/phase/group cells, source-stratified means,
coverage, per-trial audit, exact old/new baseline comparisons and figures.

Examples with authoritative membership:

- Phase II Recruitment: raw 18/93 false successes; pooled calibration 15/93.
- Phase III no record: raw 53/82; pooled calibration 52/82.
- Phase III Efficacy: 7/8 for both; all eight reasons are predicted.
- Phase I Efficacy: 3/4 for both; small groups remain fragile.

Group counts and group-conditional means/rates changed because predictions were
replaced and coverage expanded. **Outcome probabilities and all aggregate/phase
outcome validation metrics did not change.** Neither Part 2 training nor calibration
was rerun.

## Verification

- `python -m integration.verify`: nine applicable original behavior tests passed;
  final Part 1 metrics recomputed from its saved validation predictions; all 37
  original outcome experiment losses verified; exact saved baseline/calibrated
  inference reproduced; five-fold trial isolation and submission schemas checked.
- 195 source/artifact files verified byte-for-byte against the recorded Git blobs.
- One original test (`test_saved_artifacts`) asserts Michael's entire old tree,
  including root Part 1 files deliberately absent here. It is preserved and not
  applicable; the new verifier independently checks its substantive saved-model,
  submission, score and fold invariants. This is documented, not silently skipped.
- Four final notebooks ran all 93 code cells in verification mode via separate
  in-process IPython sessions. Full training bodies were skipped. The updated
  analysis notebook freshly recomputed all 54 cells. No official nbformat/nbclient
  kernel execution or full model-training notebook run is claimed; those packages
  are absent in the available environment.
- A temporary standalone `midterm/` staging folder was assembled. All four notebooks
  again passed verification there, without project-module imports or retraining.
  Location/checks are in `outputs/package_validation.json`.
- The new LaTeX report compiled twice with resolved references to four pages total,
  within the five-page main-text limit. First-page layout was visually inspected.

## Final deliverables and reproduction

- Final notebooks: `deliverables/notebooks/` (four files).
- Final notebook/environment/assembly guide: `deliverables/README.md`.
- Unified draft: `writeup/main.pdf` and `writeup/main.tex`.
- Measured tables/figure source: `writeup/build_tables.py`; Part 1 PCA source is
  copied from main in the final Part 1 notebook.
- New authoritative error analysis: `integration/outputs/analysis_report.md`.
- All originals remain under their original paths.

From the repository root, using the available Python environment:

```sh
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m integration.reconcile_reasons
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m integration.verify
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python writeup/build_tables.py
cd reports
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error final_report.tex
```

To assemble a new Gradescope folder outside the repo:
`python -m integration.assemble_gradescope /tmp/cs5785-final/midterm`.
The script refuses to overwrite an existing folder and uploads nothing.

## Remaining review items and limitations

Fill full team names, confirm responsibilities/any additional contributors, and
review the AI-disclosure wording under course policy. The assignment PDF is absent;
its local transcription was read and the assignment-linked CVPR template checked.
Cross-check the original PDF if available. Michael's final Part 2 public score is
not recorded; do not substitute main's different Part 2 score 0.61046. Contributors
recorded slightly different numpy/pandas versions, so exact full retraining under
one combined environment remains untested. Historical error scripts expecting
Michael's old root `outputs/` are archival; the new integration entry point and
final notebook are authoritative. Review the draft, stage/commit the integration
if desired, and upload only after the team approves. No push is authorized.
