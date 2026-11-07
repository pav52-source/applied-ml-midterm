# Part 2: clinical-trial outcome baselines

All new code and generated artifacts live here. Part 1 is untouched. Inputs are
read from `../part2-data/`; all required inputs are present, including the optional
pathway catalog (not used). The requested PDF is absent; the local assignment
transcription `../Midterm_Project.tex` was read instead.

From the repository root, using the existing environment:

```sh
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m unittest part2.test_baselines
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.experiments
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.report
```

The experiment command runs the audit **before** fitting, all 37 experiments,
selection by minimum overall validation log loss, and submission verification.
It overwrites only generated files in `part2/outputs/`. It retains the winner
trained on training alone: validation labels are never used for fitting features,
classifier coefficients, or a final refit. Seeds are fixed and thread counts are
limited for repeatability. See `outputs/run_metadata.json` for installed versions
and input hashes; `requirements.txt` pins the observed environment.

Optional independent commands:

```sh
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.audit
/Users/michaelherbstman/GradSchool/CS_5785/appliedML_venv/bin/python -m part2.submission
```

`baseline_workflow.ipynb` is a notebook entry point for this same code. Its commands
must run from the repository root (the notebook handles either root or part2 cwd).
It does not duplicate the implementation.

Modules: `config.py` (paths/grid), `data.py` (schemas/list decoding/feature allowlist),
`audit.py` (data analysis), `features.py` (train-fitted transformations),
`training.py` (prior and logistic fits/inference), `evaluation.py` (probability
metrics and thresholded F1), `experiments.py` (grid/selection/artifacts),
`report.py` (readable tables), `submission.py` (ordered CSV checks).

The grid compares clinical metadata text, eligibility criteria, and all four text
fields; C=0.1/1/10; year on/off; pooled with phase one-hot versus three independent
phase models. This is 36 logistic settings plus the phase-prior baseline.
Lists and nested ICD lists are decoded with `ast.literal_eval`, never `eval`.
Text missingness becomes empty text; year uses training-median imputation,
a missingness indicator, and training-fitted standardization. TF-IDF vocabularies
and IDF weights are fitted on training only (independently per phase when needed).
Available unused features include categorical phase_detail, drug names, targets,
actions, pathways and SMILES strings; the only measured numeric feature is year.
IDs are matching keys only. Labels are targets only. Reasons/embeddings are excluded.

Read `outputs/experiment_report.md` for all results, observations and limits;
audit CSVs record dtypes, missingness, class balance, year distributions and leakage
checks. `outputs/part2_submission.csv` has exactly 1,500 probabilities in the
original test ID order. Nothing is submitted to Kaggle.

See `ERROR_ANALYSIS.md` for the deferred Part 1 reason integration. The full
assignment still needs that analysis, its follow-up improvement, and the final
write-up; this directory completes only the requested initial baseline stage.
