# Midterm report

`main.tex` is the report. Its Part 1 tables are in this directory;
the Part 2 tables are in `tables/`, and the two included figures are in `figures/`.
`main.pdf` is the compiled draft. Fill in the author placeholders and review the
responsibilities and AI assistance disclosure before submission.

From the repository root, compile with `make report-pdf`. Regenerate the Part 2 tables and reliability figure with
`python writeup/build_tables.py` after updating the underlying analysis outputs.
The report uses the pooled Part 2 model described there, not the separate
per-phase model in the starter notebook.
