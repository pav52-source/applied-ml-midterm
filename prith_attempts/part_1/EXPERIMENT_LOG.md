# Experiment log

This file records decisions, settings, observed results, and next steps as we work through the project. Results copied from an existing notebook are distinguished from experiments executed during this work.

## Project requirements and boundaries

Source: `Midterm_Project.tex`.

- Part 1 predicts five independent binary stopping reasons: Safety, Funding, Efficacy, Recruitment, and Other. One trial can have several positive reasons.
- The training set has 1,171 trials: 157 with visible labels and 1,014 with all five labels masked as `-100`. Validation has 78 labeled trials. Test has 312 trials with hidden labels.
- Use at least one unsupervised method to recover missing training labels. Select models and features using validation performance. Report sample-wise F1, macro F1, and each reason's F1.
- The 150 embedding features represent stopping-reason text, produced with a pretrained sentence embedding model, reduced with PCA, and randomly rotated. The assignment does not name the embedding model. Individual coordinates have no semantic interpretation; standardize before use.
- Fit preprocessing and clustering on training features. Use visible training labels to map clusters to reason rates. Keep validation labels for evaluation and selection; never use hidden test labels.
- IDs identify and match trials; they are not predictive features. Do not search for fully labeled source data or reverse engineer test labels. Deep learning is disallowed.
- Labeled trials are mostly earlier than unlabeled/test trials. Label frequencies can change with time, so cluster reason rates may inherit bias from the labeled subset.
- Part 2 predicts outcome probabilities by phase. Reason embeddings and reason labels are allowed only for error analysis, not as outcome-model inputs.

## 2026-11-05 — Understanding and choosing the first experiment

### Questions clarified

The user asked how embeddings were generated, whether to add other features, which splits contain correct labels, and how clustering can support multiple reasons per trial.

The final target is an embedding-to-five-label mapping. Clusters are intermediate groups of similar embeddings, not the five reason categories. A cluster can have a high observed rate for both Safety and Efficacy. Estimating each reason independently therefore permits multiple labels for one trial.

A separate binary classifier per reason is appropriate for supervised prediction. Running ordinary two-cluster K-means five times on identical embeddings does not make each run reason-specific, because labels do not guide ordinary K-means. GMM is a probabilistic clustering/density model commonly fitted with EM; EM is its fitting procedure. A separate label-guided EM approach would require a semi-supervised model.

The user requested a separate teaching notebook and a sub-agent to maintain this process log as experiments evolve.

### Existing starter results — saved outputs, not rerun here

Source: `CS5785_Midterm_Starter.ipynb`. Its supervised baseline uses standardized embeddings and one-vs-rest logistic regression trained on the 157 labeled trials. The starter has placeholders for unsupervised label recovery, Part 2 reason-group error analysis, and outcome-model improvement.

| Metric | Saved validation result |
|---|---:|
| Sample-wise F1 | 0.8654 |
| Macro F1 | 0.7067 |
| Safety F1 | 0.8571 |
| Funding F1 | 0.2857 |
| Efficacy F1 | 0.8276 |
| Recruitment F1 | 0.9796 |
| Other F1 | 0.5833 |

These numbers are historical notebook outputs. They are not evidence of a new execution or improvement.

### Teaching notebook created — code cleared for incremental implementation

Notebook: `Part1_Clustering_Walkthrough.ipynb`.

1. Load Part 1 data, show visible/masked labels, and inspect examples of multiple positive reasons.
2. Fit standardization on training embeddings and transform other splits with the same scaler.
3. Reproduce an embeddings-only supervised baseline using the visible training labels.
4. Fit K-means on all training embeddings without supplying labels.
5. Show cluster sizes, labeled support, known positive counts, and reason rates. Smooth rates toward the overall labeled-training rates so sparsely labeled clusters do not produce unsupported extreme estimates; clusters with no labeled members require a documented fallback.
6. Inspect actual trials and display how their cluster leads to five estimated reason probabilities and pseudo-labels. Explain that confidence summaries are heuristic scores, not demonstrated calibration or certainty.
7. Preserve the original visible labels and augment with cluster-derived pseudo-labels. Compare supervised models that give pseudo-labeled examples lower training weight with the original baseline.
8. Compare validation sample-wise F1, macro F1, and per-reason F1. Interpret gains or losses rather than assuming pseudo-labeling helps.

Initial settings proposed for later implementation:

| Setting | Initial value |
|---|---|
| Features | 150 reason embeddings |
| Preprocessing | StandardScaler fit on all training embeddings |
| K-means clusters | 8 |
| K-means initializations (`n_init`) | 20 |
| Random seed | 42 |
| Cluster-rate smoothing strength | 2, toward visible-training reason rates |
| Independent reason threshold | 0.5 |
| Known-label training weight | 1.0 |
| Pseudo-label training weight | 0.25 |

For each cluster and reason, the smoothed estimate is `(known positive count + 2 * global labeled rate) / (known member count + 2)`. With no labeled cluster members, this falls back to the global visible-training rate. These global rates can themselves reflect the selection bias in the labeled subset.

The proposed comparison includes (1) supervised logistic regression trained only on known labels, (2) direct predictions from cluster reason rates, and (3) logistic regression trained on known labels plus lower-weight pseudo-labels. The latter would use five binary models to preserve multiple positive reasons per trial.

The teaching plan calls for cluster support and rates, one concrete unlabeled-trial assignment, counts of all-zero and multiple-positive pseudo-label vectors, and a comparison of label prevalence. Any confidence summary should be described as a heuristic, not calibrated confidence. An earlier, unexecuted implementation included automatic result logging to `experiments/part1_clustering_runs.csv`; that code has now been cleared, so no runtime or automatic CSV logging behavior remains.

### User-directed change: implement together incrementally

After the initial notebook was created, the user requested removing all code so that we can implement each step together. All eight code cells were cleared, including sources, outputs, and execution counts. Markdown explanations and empty code cells remain. The proposed settings above are a teaching plan, not a currently implemented or executed experiment. No model was executed before the code was cleared.

### Status and next steps

- Current notebook state: incremental implementation has resumed, as recorded below. Remaining code cells are left blank for subsequent work with the user. No new model metrics have been recorded.
- Execution environment: no local `.venv` was found, and system Python lacks NumPy, pandas, and scikit-learn.
- Verification before clearing: valid notebook JSON, eight code cells parsed with Python AST, unique cell IDs, and no execution counts or outputs. A standard-library CSV inspection confirmed expected columns, 157 known training rows, and 1,014 masked training rows. These checks did not execute models or establish model performance; the AST result describes the implementation that was subsequently cleared.
- The Markdown outline ends at validation comparison. GMM/EM experiments and test-prediction export remain future steps.
- Start with embeddings to make cluster behavior interpretable. Later comparisons can vary the cluster count, pseudo-label weighting, GMM/EM, and additional feature groups while recording what changed.
- Reserve validation for selection; repeated experimentation on only 78 validation trials can make small gains unreliable.
- This log and notebook are working artifacts. The final report and both project parts still require completion.

### Incremental step: clustering and a two-dimensional view

The user asked to begin label recovery and visualize the embeddings while acknowledging that a two-dimensional view loses information. The notebook now imports PCA and matplotlib; `requirements.txt` includes `matplotlib>=3.9,<4`.

- Cell 4 fits K-means on all training embeddings in the full 150 standardized dimensions. The exploratory settings are 8 clusters, `n_init=20`, and random seed 42. It identifies visible-label trials and prepares total, labeled, and unlabeled member counts for each cluster.
- Cell 5 fits PCA with two components and `svd_solver="full"` on training embeddings for display. The plot colors trials by their full-dimensional cluster IDs, adds black rings around known-label trials, and marks projected centroids with an X. PCA is a visualization step; K-means uses all 150 dimensions.
- Upon execution, the visualization cell will print retained variance, explain that projection overlap does not imply overlap in the full embedding space, and save `experiments/figures/part1_kmeans_pca.png`.
- Reason assignment is not implemented yet. Remaining code cells stay blank for incremental work.

Execution status: syntax checks pass. No model fitting was run here because the local environment and required dependencies remain unavailable. No retained-variance value or actual cluster composition has been observed, and no rendered figure exists yet. The intended figure path describes future output only.

## Entry format for subsequent experiments

For each run, record: question/hypothesis; notebook and data split; exact features and preprocessing; model parameters and random seed; pseudo-label rules and weight; execution status; validation metrics; cluster support and failure observations; decision and next change. Record failed or skipped runs explicitly and never fill in unobserved metrics.

### Walkthrough reset

At the user’s request, all Markdown cells were also removed from `Part1_Clustering_Walkthrough.ipynb`. It now contains only empty code cells for incremental work together.

### Incremental implementation — imports and data loading

Added imports in the first code cell and Part 1 train/validation/test plus pathway-catalog loading in the second. The loading cell displays shapes and a training-data preview. Remaining cells are empty. Both added cells pass Python syntax checks; they have not been executed.

### Data-path correction

A user execution failed because the kernel working directory did not match the original path assumptions. Updated cell 2 to search the current directory, ancestors, and project subfolders, with the verified local project path as fallback. Directory discovery and CSV existence were checked from the project, data folder, and an unrelated directory; data loading itself was not executed.

### Incremental implementation — embedding scaling

Added scaling in the third code cell: select emb_0 through emb_149, fit StandardScaler on all training embeddings without labels, and transform validation/test using the training statistics. The cell prints matrix shapes. Syntax checked; not executed.

### Clustering and visualization cells cleared

At the user’s request, code and any outputs were removed from cells 4 and 5. Imports, data loading, and scaling remain. No cluster count has been chosen for the next implementation step.

### Incremental implementation — reusable embedding-scaling function

Refactored cell 3 into `create_scaled_embeddings(train_df, val_df, test_df)`. The function selects `emb_0` through `emb_149`, fits a scaler on all training embeddings, transforms validation and test with those training statistics, and returns `X_train, X_val, X_test, scaler`. The call below the function uses the Part 1 DataFrames and retains the three shape prints. Other notebook cells are unchanged. Notebook JSON and Python syntax were checked; no scaling or model code was executed during this edit.

### Incremental implementation — K-means training function

Added `train_kmeans(X_train, n_clusters, n_init=20, random_state=42)` in cell 4. It fits K-means on the supplied scaled training embeddings and returns the fitted model. No cluster count was selected and no training call was added. The existing KMeans import was confirmed. Notebook JSON and function syntax were checked; the function has not been executed.

### Incremental implementation — inspect cluster reason percentages

Added `inspect_cluster_percentages(model, X_train, train_df)` in cell 5. It predicts cluster assignments, prints each cluster's total/labeled/unlabeled counts, and prints all five reason percentages using only members with five visible binary labels. The denominator is the labeled member count; reasons can overlap. It selects source rows by position, checks the input row counts, and explicitly reports unavailable percentages when a cluster has no labeled members. All configured K-means or GMM clusters are included, including empty predicted clusters. No pseudo-label assignment, smoothing, thresholds, plotting, or invocation was added. Notebook JSON and Python syntax checks passed; the function was not executed.

### Executed K-means pipeline — 8 clusters

Execution timestamp: 2026-11-06T03:01:43.718181+00:00. Executed the six code cells in order using existing user environment `/Users/prith/.venv/bin/python` (Python 3.14.7, NumPy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0). No dependencies were installed. nbclient was unavailable, so cells were executed with Python AST execution and stdout/stderr plus final-expression results captured into notebook outputs. No existing interactive kernel state was used.

The new sixth cell calls `create_scaled_embeddings(part1_train, part1_val, part1_test)`, then `train_kmeans(X_train, n_clusters=8)`, then `inspect_cluster_percentages(kmeans_model, X_train, part1_train)`. Settings: 150 training-standardized reason embeddings; K=8; n_init=20; random_state=42; remaining KMeans settings use library defaults. Validation/test embeddings were transformed with training statistics but not used in clustering.

The row order was preserved. Checks confirmed all 1,171 training rows were accounted for, training predictions matched fitted assignments, and support totaled 157 labeled plus 1,014 unlabeled rows. Reason percentages use only visible training labels, independently for each reason. No pseudo-labels, validation scores, plots, or outcome models were generated.

| Cluster | Total | Labeled | Unlabeled | Safety % | Funding % | Efficacy % | Recruitment % | Other % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 226 | 46 | 180 | 0.0 | 0.0 | 0.0 | 100.0 | 2.2 |
| 1 | 78 | 4 | 74 | 0.0 | 0.0 | 0.0 | 100.0 | 25.0 |
| 2 | 248 | 36 | 212 | 36.1 | 5.6 | 25.0 | 5.6 | 27.8 |
| 3 | 28 | 2 | 26 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 |
| 4 | 110 | 11 | 99 | 0.0 | 9.1 | 0.0 | 81.8 | 9.1 |
| 5 | 156 | 23 | 133 | 0.0 | 17.4 | 0.0 | 91.3 | 21.7 |
| 6 | 115 | 10 | 105 | 0.0 | 0.0 | 0.0 | 100.0 | 10.0 |
| 7 | 210 | 25 | 185 | 12.0 | 4.0 | 32.0 | 36.0 | 16.0 |

These are descriptive rates among labeled cluster members, not measured prediction accuracy. All eight clusters had at least one labeled member in this run. Cluster IDs are arbitrary.

### Cluster printing disabled

At the user’s request, print statements in `inspect_cluster_percentages()` were commented out, along with its final print-only reason loop. Computations and the pipeline invocation remain unchanged. Cleared saved output and execution counts in the function and invocation cells; earlier eight-cluster results remain historical. Notebook JSON and Python syntax checks passed. No model was rerun.

### Cluster-to-reason mapping with a percentage threshold

Updated `inspect_cluster_percentages(model, X_train, train_df, percentage_threshold)` to return an integer cluster-ID-to-reason-list mapping instead of printing. Thresholds use 0–100 percentages (50 means 50%); reasons at or above the threshold are included independently, using only labeled training members. Clusters with no labeled members or no qualifying reasons return an empty list. The function validates threshold bounds and row counts; input rows must remain in corresponding order. The pipeline cell now stores and displays `cluster_reason_mapping` with threshold 50. Changed-cell outputs were cleared. Notebook JSON and code syntax were checked; the pipeline was not rerun and no labels were assigned to individual trials.

In-memory synthetic checks passed for inclusive thresholds, multiple assigned reasons, exclusion of masked labels, clusters with no qualifying reasons or no labeled members (including empty clusters), invalid threshold bounds, and mismatched row counts. These checks used a mocked cluster model; no actual training or pipeline execution occurred.

### Assign cluster-derived labels to unlabeled training points

Added `label_unlabeled_points(model, X_train, train_df, cluster_reason_mapping)` in cell 7. It returns a copy of the training DataFrame, preserving all existing known labels and other features. For a fully masked row with a nonempty cluster mapping, it sets each of the five reason columns to 1 or 0 according to membership in that reason list. Empty or missing mappings leave all five labels at -100, because no qualifying reason is not evidence that every reason is absent. The new `label_source` column records `observed`, `cluster_prediction`, or `unresolved`. Row counts must match and rows must be either fully binary-labeled or fully masked; positional assignment preserves alignment even with duplicate DataFrame indices.

Cell 8 now calls the function to create `part1_train_augmented` and displays counts by label source. No CSV is saved, no model is fitted, and the pipeline has not been rerun. Modified cells have empty outputs and execution counts. Notebook JSON and Python syntax checks passed. Small in-memory checks with a mocked model verified original-data preservation, known-label preservation, multiple predicted reasons, unresolved empty/missing mappings, duplicate-index alignment, and rejection of mismatched row counts and partially labeled rows. No observed augmentation counts or validation scores are claimed.

### Incremental implementation — logistic regression on augmented labels

Added `train_logistic_regression(X_train, train_df, C=1.0, max_iter=500, random_state=42)` and a separate call using `part1_train_augmented`. The function fits `OneVsRestClassifier` around `LogisticRegression` with the `liblinear` solver, producing five binary classifiers in Safety, Funding, Efficacy, Recruitment, Other order. Observed and cluster-derived labels have equal training weight. Only rows with all five labels in {0, 1} are included; unresolved -100 rows are excluded. The function checks row counts and raises an error if no trainable rows exist; embeddings and rows must remain positionally aligned. It returns the fitted classifier. No reason/source columns are used as input features. Notebook JSON and Python syntax checks passed. Neither new cell was executed; no new model fitting or validation result is claimed.

### Incremental implementation — validation F1 evaluation

Added `evaluate_reason_model(model, X_val, val_df)` in cell 11 and its call in cell 12. The function predicts using the fitted model and already-scaled validation embeddings, checks matching row counts and five binary validation labels, and returns a pandas Series named `Validation F1`. Metrics are sample-wise F1, macro F1, and per-reason F1 in Safety, Funding, Efficacy, Recruitment, Other order, with `zero_division=0`. The call evaluates `reason_model` against `part1_val` and displays four-decimal results. No model or scaler is fitted, thresholds are unchanged, and validation labels are used only to score predictions.

Notebook JSON and all code-cell syntax checks passed. A small in-memory fixed-prediction check verified expected metric values, column order, and errors for invalid row counts or labels. No real classifier fitting or pipeline/validation execution was performed; both new cells remain unexecuted and no new validation metrics are claimed.

### Label-source column refactor

At the user’s request, `label_unlabeled_points` now creates and updates `label_source` directly in the copied DataFrame instead of maintaining a separate array. Observed labels and unresolved rows retain their prior behavior; positional updates support arbitrary row indices. Notebook JSON and Python syntax checks passed. No model fitting or pipeline rerun was performed.

### End-to-end pipeline cell

Appended one unexecuted cell that calls the existing scaling, K-means training, cluster-reason mapping, label recovery, logistic regression, and validation evaluation functions in sequence. Settings are 8 clusters, a 50% reason threshold, seed 42, 20 K-means initializations, logistic C=1.0, and max_iter=500. The cell prints label-source counts and displays all required Part 1 validation F1 metrics. Imports, data loading, and function definitions must be executed first. Existing cells were preserved. Notebook JSON, Python syntax, and argument/signature compatibility were checked without fitting models or running the new pipeline cell.

### Cluster-count sweep prepared

Replaced the final single-run pipeline cell with a loop over cluster counts 2 through 30 inclusive. Scaling is fitted once on training embeddings and reused. Every run uses a 50% cluster-reason threshold, K-means n_init=20, random_state=42, and logistic regression C=1.0/max_iter=500. `pipeline_results[n_clusters]` retains the fitted K-means model, cluster-to-reason mapping, augmented training DataFrame, fitted logistic regression model, validation metrics, and label-source counts. A separate cell displays macro F1 for all cluster counts and identifies all exact ties for the best raw score. Selection uses unrounded metrics. Choosing among 29 settings on only 78 validation trials may overfit the validation set; compare with the supervised baseline. Notebook JSON and code syntax were checked. The modified pipeline and new summary cell are unexecuted, so no best cluster count or new scores have been observed.

### Sweep selection metric switched to sample-wise F1

Updated the final cluster-count comparison cell to display validation sample-wise F1 for every run and identify the highest unrounded score, including tied cluster counts. This matches the Part 1 Kaggle metric. Macro and per-reason F1 remain computed and stored. Cleared the display cell’s old output; the sweep was not rerun. Notebook JSON and Python syntax checks passed.

### Cluster-reason threshold raised to 60%

The saved 50% threshold sweep selected 2 clusters with validation sample-wise F1 of 0.8419 (historical result). At the user’s request, the current sweep configuration now uses `PERCENTAGE_THRESHOLD = 60`, keeping cluster counts 2–30 and all other settings unchanged. Cleared the sweep and comparison cells’ saved outputs/execution counts to avoid displaying the old results as a 60% run. The 60% sweep has not been executed; rerun the sweep cell followed by its comparison cell. Notebook JSON and code syntax checks passed.

### Cluster-reason threshold raised to 70%

Changed the sweep configuration from 60% to `PERCENTAGE_THRESHOLD = 70`, preserving cluster counts 2–30 and all other settings. Cleared the sweep and sample-wise F1 comparison cells’ saved outputs/execution counts. The 70% sweep is pending and has not been executed. Prior observed results remain historical. Notebook JSON and Python syntax checks passed.

### Sweep table expanded to all Part 1 metrics

The comparison cell now displays sample-wise F1, macro F1, and Safety, Funding, Efficacy, Recruitment, and Other F1 for each cluster count in ascending order. It still selects the best cluster counts by unrounded sample-wise F1 and includes ties. Only the display cell was changed and its saved output/count cleared; the 70% sweep configuration and all functions are unchanged. Notebook JSON and Python syntax checks passed. No sweep was rerun.

### Sweep table — remaining unlabeled percentage

Added a `Percent unlabeled` column to the final comparison table. For each cluster count it is 100 times the fraction of all augmented training trials whose five reason labels remain `-100` after label recovery. The denominator is all training rows, not only the originally masked subset. All seven F1 columns and selection by unrounded sample-wise F1 remain unchanged. Display-cell output was cleared. JSON and Python syntax checks passed; the sweep was not rerun.

### Cluster-reason threshold lowered to 40%

Changed the sweep configuration to `PERCENTAGE_THRESHOLD = 40`, preserving cluster counts 2–30 and all other settings and metrics-table columns. Cleared the sweep and dependent comparison cells’ saved outputs/execution counts so prior results are not shown as a 40% run. The 40% sweep is pending and has not been executed. Notebook JSON and Python syntax checks passed.

### Optimization — downweight recovered labels and compare a fresh baseline

Retained K-means, the 40% cluster-to-reason threshold, K=2 through 30, n_init=20, seed=42, and logistic C=1/max_iter=500. Added `pseudo_weight=0.25` to `train_logistic_regression`: observed rows retain weight 1; recovered rows receive weight 0.25; unresolved rows are excluded as before. Original training frames without `label_source` use weight 1 for every eligible row. A scoped sklearn metadata-routing context forwards weights to the five binary logistic regressions and restores the caller's routing configuration. The fitted OneVsRestClassifier API is preserved. Unknown `-100` labels have not been replaced by zeros.

The sweep computes an observed-label-only baseline once, stores each run's pseudo weight, and displays `Delta vs baseline` using sample-wise F1. No tree, extra feature, or extra hyperparameter search was introduced.

Verification: notebook JSON and Python AST checks passed. A synthetic check confirmed zero-weight recovered rows reproduce observed-only coefficients, weight 0.25 supports fitting/prediction, unresolved rows are omitted, invalid weights are rejected, and global metadata-routing settings are unchanged.

Executed the updated notebook and all 29 cluster-count settings using the existing `/Users/prith/.venv/bin/python`; no installations. Real outputs and execution counts are saved. Elapsed time: 34.5 seconds. Validation labels are used only for evaluation.

| Method | Sample-wise F1 | Macro F1 |
|---|---:|---:|
| Fresh observed-label-only baseline | 0.8653846154 | 0.7066737040 |
| Earlier equal-weight recovered labels, K=2, threshold 40% | 0.8419 (historical rounded output) | 0.6255 (historical rounded output) |
| Recovered-label weight 0.25, K=2, threshold 40% | 0.8572649573 | 0.6619865320 |

The equal-weight values are historical rounded notebook output, not rerun here. The weighted run improves sample-wise F1 by approximately 0.0154 over the historical equal-weight K=2 run, but remains 0.0081196581 below the observed-only baseline. K=2 is the best weighted setting; every weighted setting remains below baseline. At K=2, 53.4586% of all training trials remain unresolved. Downweighting reduces harm but has not shown an improvement over supervision alone. More experiments must be described as pending until executed.

### Weighted sweep threshold raised to 50%

Changed `PERCENTAGE_THRESHOLD` from 40 to 50 while retaining recovered-label weight 0.25, K-means, K=2 through 30, the observed-label-only baseline, and all comparison columns. Cleared saved outputs/counts in the sweep and dependent display cells so the historical 40% results are not presented as 50%. No models were run or refitted; the 50% weighted sweep is pending. Notebook JSON and Python syntax checks passed.

### Recovered-label weighting removed at user request

The current logistic-regression function now trains on observed and recovered binary labels equally. Removed the `pseudo_weight` parameter, sample-weight construction, metadata routing, weighting import, sweep configuration, and stored weighting key. Rows still masked as `-100` remain excluded from training. The current K-means sweep remains 2–30 clusters at a 50% label threshold, retaining the observed-label-only baseline and all validation metrics. Earlier weighted experiment results remain historical; outputs for the affected function/import, sweep, and comparison cells were cleared. JSON, Python syntax, call signatures, and removal of weighting identifiers were checked. No models were retrained and the equal-weight 50% sweep has not been rerun.

### 2026-11-06 — Minimum original label support per cluster

Added `min_labeled_members=5` to `inspect_cluster_percentages` and `MIN_LABELED_MEMBERS = 5` to the sweep configuration. A cluster with fewer than five originally labeled trials maps to an empty reason list, so its unlabeled members remain unresolved. The independent reason-percentage threshold stays at 50%; recovered and observed examples retain equal training weight. Cleared the modified function and downstream sweep/comparison outputs. JSON and syntax checks passed; the updated real sweep has not been run.

A synthetic mock-model check confirmed that four known members leave a cluster unresolved at the default minimum, five permit reason assignment, an explicit minimum of four permits four-member clusters, and a zero minimum is rejected. No real models were fitted.

### 2026-11-06 — Removed minimum labeled support rule

At the user’s request, removed the minimum-five-labeled-members parameter, configuration, and sweep argument. Clusters with any originally labeled members can again map reasons using the percentage threshold; clusters with zero labeled members retain empty mappings and unresolved trials remain masked. The threshold remains 50%, labels retain equal training weight, and baseline/sweep metric columns are unchanged. Cleared affected function, sweep, and display outputs. JSON and Python syntax checks passed, and no minimum-support code remains. No models were rerun.

### 2026-11-06 — Modular ordinary PCA for clustering

Added `create_pca_embeddings(X_train, n_components=50)` in its own function cell before the combined sweep. It fits ordinary PCA with deterministic full SVD on scaled training embeddings only, returning the reduced embeddings and fitted PCA object. The sweep config uses `PCA_COMPONENTS = 50` as an exploratory starting point; this has not been selected as optimal. PCA is fitted once outside the cluster-count loop.

K-means, cluster reason inspection, and label recovery now use `X_train_cluster`; baseline and augmented logistic regression still use the original 150-dimensional scaled training embeddings, and validation evaluation still uses original scaled validation embeddings. The shared fitted PCA is retained as `pca_model` and in each run dictionary for reproducible future clustering transforms. On execution, the pipeline reports component count and retained training variance. No reconstruction, whitening, validation fitting, or test fitting is introduced. The 50% reason threshold, equal label weighting, cluster-count sweep, baseline comparison, and metric columns remain.

Cleared the modified sweep and dependent display outputs to avoid showing no-PCA results as PCA results. JSON, Python syntax, unique cell IDs, and representation-routing checks passed. This real PCA clustering sweep has not been executed; no variance retention or improvement is claimed. Kernel PCA remains a later experiment.

A small in-memory synthetic PCA check passed: a 12×6 input produces 12×3 reduced embeddings; the fitted transformer reproduces those coordinates, whitening remains disabled, and the input array is unchanged. This check used synthetic data only and did not execute the real sweep.

### 2026-11-06 — Modular RBF kernel PCA for clustering

Added `create_kernel_pca_embeddings(X_train, n_components=50, kernel="rbf", gamma=None, random_state=42)` as a separate function, keeping ordinary `create_pca_embeddings` unchanged. The helper fits only the scaled training embeddings, uses the ARPACK eigensolver, and returns the transformed training embeddings and fitted transformer. `gamma=None` explicitly resolves to 1 divided by the input dimension (1/150 for this dataset); this is a configurable starting value, not an optimized setting.

The current sweep now uses `KERNEL_PCA_COMPONENTS=50`, `KERNEL="rbf"`, and `KERNEL_GAMMA=None`. Kernel PCA is fitted once before K-means; clustering, reason mapping, and label recovery use its transformed coordinates. Both observed-only and augmented logistic regressions still use the original scaled embeddings. Threshold 50%, equal training weights, K=2–30, and all comparison columns remain unchanged. Each result retains `kernel_pca_model` for transforming future points before cluster prediction.

Replaced ordinary PCA variance diagnostics with output dimension, kernel, and effective gamma. No original-space variance percentage is claimed for kernel PCA. Cleared modified import, sweep, and comparison outputs/counts to avoid presenting ordinary-PCA results as kernel-PCA results. The prior saved ordinary PCA output reported 86.2% variance retained and a best sample-wise F1 of 0.8419 at two clusters versus a 0.8654 supervised baseline; these are historical notebook outputs, not newly executed results.

Notebook JSON, unique cell IDs, Python syntax, and representation routing were checked. The real kernel PCA sweep has not been run; no performance improvement is claimed.

A synthetic 12×6 kernel PCA check passed: output shape 12×3, explicit default gamma 1/6, consistent training transform and transformation of new rows, no inverse fitting, and unchanged input. This used synthetic data only; the real clustering sweep remains unexecuted.


### 2026-11-06 — Observed RBF kernel PCA sweep results

Read the user-executed, saved outputs in `Part1_Clustering_Walkthrough.ipynb`; this entry does not represent a fresh execution by the logging agent. The saved run used 50 RBF kernel PCA components, gamma=1/150 (printed as 0.00666667), a 50% reason-assignment threshold, equal observed/recovered-label training weights, and K-means cluster counts 2–30. Clustering and label recovery used kernel PCA coordinates; logistic regression continued to use the original standardized 150-dimensional embeddings.

The best saved sample-wise validation F1 was **0.8731 at 10 clusters**, compared with the freshly reported observed-label-only baseline of **0.8654** (displayed delta **+0.0077**). The prior ordinary-PCA sweep's best sample-wise F1 was 0.8419. The selected kernel-PCA run reported:

| Metric | Saved result |
|---|---:|
| Sample-wise F1 | 0.8731 |
| Macro F1 | 0.7046 |
| Safety F1 | 0.8182 |
| Funding F1 | 0.2500 |
| Efficacy F1 | 0.8667 |
| Recruitment F1 | 0.9796 |
| Other F1 | 0.6087 |
| Percent of all training trials still unresolved | 44.2357% |
| Delta versus observed-only sample-wise F1 | +0.0077 |

This is an observed validation improvement for the selected setting, not evidence of improved test performance. The setting was chosen among 29 cluster counts using only 78 validation trials, after earlier experiments on the same split. The small gain should be interpreted with that selection uncertainty in mind; it does not establish that nonlinear clustering alone caused the gain.

### 2026-11-06 — Planned gamma comparisons for user execution

The user will execute the next experiments and ask a sub-agent to analyze the saved outputs after each iteration. No notebook code was changed and no models were executed for this entry. The reference below was checked against the current saved notebook outputs; displayed metrics have four decimal places, not recovered full precision.

Hold the following settings fixed to isolate gamma: RBF kernel, 50 kernel PCA components, **10 clusters**, reason threshold 50%, K-means `n_init=20`, random seed 42, logistic regression `C=1.0`, `max_iter=500`, and equal observed/recovered-label weights. Classifiers continue to use the original scaled embeddings.

| Run ID | Gamma | Status | Sample-wise F1 | Macro F1 | Percent unresolved |
|---|---:|---|---:|---:|---:|
| `gamma_reference` | 1/150 = 0.006666666666666667 | Observed saved output, K=10 | 0.8731 | 0.7046 | 44.2357% |
| `gamma_low` | 1/300 = 0.0033333333333333335 | Observed saved output, K=10 (executions 203/204) | 0.8184 | 0.6396 | 28.7788% |
| `gamma_high` | 1/75 = 0.013333333333333334 | Observed saved output, K=10 (executions 251/252) | 0.8769 | 0.6904 | 49.6157% |

Test `gamma_low` first, then `gamma_high`. If the user retains the existing K=2–30 sweep, compare the **K=10 row** across gamma settings to isolate the gamma change. Separately record each sweep's best K and score; the best cluster count is not assumed to stay at 10.

For each completed iteration, record gamma, cluster count, kernel PCA components, reason threshold, random seed, sample-wise F1, macro F1, the five per-reason F1 values, percent unresolved, and deltas against both the observed-only baseline and `gamma_reference`. The currently saved observed-only baseline sample-wise F1 is 0.8654. Reference per-reason F1 values are Safety 0.8182, Funding 0.2500, Efficacy 0.8667, Recruitment 0.9796, and Other 0.6087. Record execution status and any changes to supposedly fixed settings before comparing results. Planned rows must not receive inferred or invented metrics.

### 2026-11-06 — First gamma-test report: saved evidence remains the reference setting

The user reported running the first test. Inspection of the saved notebook found execution counts 167 (sweep) and 168 (comparison). The sweep source still has `KERNEL_GAMMA = None`, and its actual printed fitted-transformer diagnostic is `Kernel: rbf; gamma: 0.00666667`, corresponding to **1/150**, not the planned lower gamma 1/300. Other visible settings are 50 components, 50% reason threshold, seed 42, K-means `n_init=20`, and logistic `C=1.0`, `max_iter=500`; the sweep covers K=2–30.

Therefore this inspection verifies a saved run of the reference configuration, not a completed low-gamma comparison. The user may have run an unsaved version or changed a different setting; the saved artifact does not establish that. Keep `gamma_low` pending verification rather than infer its performance or conclude that changing gamma had no effect.

| K=10 metric | Prior reference gamma 1/150 | Latest saved gamma 1/150 |
|---|---:|---:|
| Sample-wise F1 | 0.8731 | 0.8731 |
| Macro F1 | 0.7046 | 0.7046 |
| Safety F1 | 0.8182 | 0.8182 |
| Funding F1 | 0.2500 | 0.2500 |
| Efficacy F1 | 0.8667 | 0.8667 |
| Recruitment F1 | 0.9796 | 0.9796 |
| Other F1 | 0.6087 | 0.6087 |
| Percent unresolved | 44.2357% | 44.2357% |
| Delta versus observed-only baseline | +0.0077 | +0.0077 |

The current saved comparison reports **best cluster count(s): [10]** and highest sample-wise F1 **0.8731**, with observed-only baseline **0.8654**. The selection cell uses unrounded in-memory values, but this inspection has only rounded saved metric values; no unrounded deltas are claimed. There is no reported best-score tie. The K=10 sample-wise delta versus the prior reference is zero at displayed precision.

Next: explicitly set `KERNEL_GAMMA = 1 / 300`, rerun the sweep and comparison cells, and save the notebook before analysis. The planned `1 / 75` test remains useful as the higher-gamma comparison afterward, but its usefulness is prospective; neither proposed new gamma has verified saved results yet. This logging step did not edit the notebook or execute any model.

### 2026-11-06 — Follow-up gamma inspection after correction report

Re-read the newly saved notebook after the user requested another look. Execution counts advanced to **179** (sweep) and **180** (comparison), but the saved sweep still sets `KERNEL_GAMMA = None` and prints **`gamma: 0.00666667`**. The saved helper still resolves `None` to `1 / X_train.shape[1]`, i.e. 1/150. Consequently, the saved evidence again describes the reference gamma, not the planned 1/300 test. No low-gamma performance conclusion is supported yet; `gamma_low` remains pending verification and `gamma_high` remains planned.

All other saved settings remain RBF kernel, 50 components, 50% reason threshold, seed 42, K-means `n_init=20`, logistic `C=1.0`/`max_iter=500`, and equal weights. At K=10 the saved values remain sample-wise F1 0.8731, macro F1 0.7046, Safety 0.8182, Funding 0.2500, Efficacy 0.8667, Recruitment 0.9796, Other 0.6087, and percent unresolved 44.2357%. The reported overall winner is still **[10]**, with sample-wise F1 **0.8731**, no reported tie, and baseline **0.8654** (displayed delta +0.0077). Relative to the reference, all displayed K=10 metrics have zero change. The saved metrics are rounded; raw precision is unavailable to this inspection. Only this log was updated; no notebook code or models were changed or executed.

### 2026-11-06 — Third fresh saved-output verification

Read the notebook from disk again after the next user request. The sweep/comparison execution counts are now **191/192**, establishing that a newer saved execution was inspected. Nevertheless, the current source still reads `KERNEL_GAMMA = None`, and the actual run diagnostic still prints `Kernel: rbf; gamma: 0.00666667` (1/150). The components/threshold/seed remain 50/50%/42. Thus no saved 1/300 result is available; this is another reference-configuration run, not evidence that the lower gamma performs identically.

The latest K=10 row remains sample-wise F1 **0.8731**, macro **0.7046**, Safety **0.8182**, Funding **0.2500**, Efficacy **0.8667**, Recruitment **0.9796**, Other **0.6087**, percent unresolved **44.2357%**, and displayed baseline delta **+0.0077**. The observed-only baseline remains **0.8654**. The printed overall winner remains **[10]** with score **0.8731** and no reported tie. Differences from the original K=10 reference are zero at displayed precision. `gamma_low` remains pending verification and `gamma_high=1/75` remains planned; no notebook code or models were changed or executed during this inspection.

### 2026-11-06 — Explicitly configured the pending lower-gamma test

To resolve the repeated reference-setting executions, changed the sweep configuration directly to `KERNEL_GAMMA = 1 / 300` in notebook cell 12. All other source and settings were preserved. Cleared the sweep and dependent comparison cells' saved outputs/execution counts (cells 12 and 13) so reference-gamma results are not presented as the new test. The earlier 1/150 outputs and comparisons remain recorded above as historical observations.

The lower-gamma experiment is now explicitly configured but **unrun**. The user will execute and save it; no performance metrics have been inferred. Notebook JSON and Python syntax checks passed. No model was fitted or executed.

### 2026-11-06 — Verified lower-gamma results: 1/300

Read the user-executed saved outputs with sweep/comparison execution counts **203/204**. The sweep explicitly sets `KERNEL_GAMMA = 1 / 300` and prints **`gamma: 0.00333333`**, verifying the intended lower-gamma experiment. The other settings remain RBF kernel, 50 kernel PCA components, 50% reason threshold, seed 42, K-means `n_init=20`, logistic `C=1.0`/`max_iter=500`, and equal observed/recovered-label weights. The observed-only baseline remains **0.8654**. This inspection did not edit notebook code or execute models.

| K=10 metric | Reference gamma 1/150 | Lower gamma 1/300 |
|---|---:|---:|
| Sample-wise F1 | 0.8731 | 0.8184 |
| Macro F1 | 0.7046 | 0.6396 |
| Safety F1 | 0.8182 | 0.7368 |
| Funding F1 | 0.2500 | 0.3333 |
| Efficacy F1 | 0.8667 | 0.6957 |
| Recruitment F1 | 0.9796 | 0.9615 |
| Other F1 | 0.6087 | 0.4706 |
| Percent unresolved | 44.2357% | 28.7788% |
| Delta versus observed-only baseline | +0.0077 | -0.0470 |

At fixed K=10, the displayed sample-wise F1 declines by **0.0547** and macro F1 by **0.0650**. The unresolved fraction decreases by **15.4569 percentage points**, so this run assigns labels to more training trials but predicts validation reasons less accurately. This association alone does not establish that the greater recovery coverage caused the decline.

The lower-gamma sweep's reported overall winner is **K=4**, with no reported tie: sample-wise F1 **0.8641**, macro **0.6706**, Safety **0.6667**, Funding **0.2857**, Efficacy **0.8276**, Recruitment **0.9899**, Other **0.5833**, and unresolved **38.0017%**. Its displayed sample-wise delta versus baseline is **-0.0013**. Relative to the prior best 1/150, K=10 result, the lower-gamma sweep's best is **0.0090** lower at displayed precision. All metric values and differences here come from four-decimal saved output; unrounded comparisons beyond the notebook's reported winner are unavailable.

Decision for this iteration: retain **gamma=1/150, K=10** as the best observed configuration. The planned **gamma=1/75** remains a useful next test in the opposite direction, with the same components, threshold, seed, and classifier settings. Compare both K=10 and the sweep winner; do not assume the optimal K is unchanged. `gamma_high` remains pending and has no observed metrics yet.

### 2026-11-06 — Higher-gamma test explicitly configured

At the user's request, changed only the sweep gamma assignment to `KERNEL_GAMMA = 1 / 75` in cell 12. Other parameters and functions are unchanged. Cleared outputs and execution counts in the sweep and comparison cells (12 and 13) so observed lower-gamma results are not shown as higher-gamma results. The verified lower-gamma observations remain recorded above. The new higher-gamma sweep is configured but has not been executed; no new metrics are claimed. Notebook JSON and Python syntax checks passed.

### 2026-11-06 — Verified higher-gamma results: 1/75

Read the user-executed saved notebook with sweep/comparison execution counts **251/252**. Source explicitly sets `KERNEL_GAMMA = 1 / 75`, and the fitted-transformer diagnostic prints **`gamma: 0.0133333`**. Other settings remain RBF kernel, 50 kernel PCA components, 50% reason threshold, random seed 42, K-means `n_init=20`, logistic `C=1.0`/`max_iter=500`, and equal weights. The observed-only baseline remains **0.8654**. This entry is analysis of saved results, not a new execution by the logging agent.

| Best reported run per gamma | Cluster count | Sample-wise F1 | Macro F1 | Percent unresolved |
|---|---:|---:|---:|---:|
| 1/300 | 4 | 0.8641 | 0.6706 | 38.0017% |
| 1/150 | 10 | 0.8731 | 0.7046 | 44.2357% |
| **1/75** | **21** | **0.8880** | **0.7351** | **46.7122%** |

The higher-gamma sweep reports **best cluster count(s): [21]**, with no reported tie. Its displayed sample-wise improvement is **+0.0226** versus baseline and **+0.0149** versus the previous best gamma=1/150, K=10. Selected-run per-reason scores are:

| Metric | Gamma 1/150, K=10 | Gamma 1/75, K=21 |
|---|---:|---:|
| Safety F1 | 0.8182 | 0.8571 |
| Funding F1 | 0.2500 | 0.3333 |
| Efficacy F1 | 0.8667 | 0.8966 |
| Recruitment F1 | 0.9796 | 0.9800 |
| Other F1 | 0.6087 | 0.6087 |

For a comparison holding K fixed, the **K=10** rows are:

| K=10 metric | Gamma 1/150 | Gamma 1/75 |
|---|---:|---:|
| Sample-wise F1 | 0.8731 | 0.8769 |
| Macro F1 | 0.7046 | 0.6904 |
| Safety F1 | 0.8182 | 0.8182 |
| Funding F1 | 0.2500 | 0.2222 |
| Efficacy F1 | 0.8667 | 0.9032 |
| Recruitment F1 | 0.9796 | 0.9899 |
| Other F1 | 0.6087 | 0.5185 |
| Percent unresolved | 44.2357% | 49.6157% |
| Delta versus baseline | +0.0077 | +0.0115 |

At fixed K=10, gamma 1/75 improves sample-wise F1 by **0.0038**, but macro F1 falls by **0.0142**: Efficacy and Recruitment improve while Funding and Other decline. Selecting K=21 yields a better balance in this validation sweep, including a higher macro F1 than the prior selected run. Changing gamma alone therefore has mixed category effects; the gamma/K interaction matters.

High gamma is not uniformly better across K: K=3, 6, and 12 have sample-wise scores 0.6496, 0.6581, and 0.6688 respectively, substantially below their neighboring configurations. This sensitivity should be retained in the interpretation rather than assuming a monotonic improvement. All quoted metrics and differences are based on four-decimal saved output. Selection after multiple sweeps on only 78 validation trials remains uncertain and does not establish better test performance.

Current best observed configuration: **RBF gamma=1/75, 50 components, K=21, threshold 50%, seed 42**. Keep this result as the reference for subsequent discussion. A possible local gamma=1/50 comparison has not been configured or run; no further experiment was started in this logging step. Only this log was modified.

### 2026-11-06 — Gamma 1/50 explicitly configured

At the user's request, changed only the sweep gamma assignment to `KERNEL_GAMMA = 1 / 50` (0.02) in cell 12. All other settings and functions are unchanged. Cleared outputs and execution counts in cells 12 and 13 so prior 1/75 results are not presented as 1/50 results. The user will execute the new sweep; it is pending and no metrics are claimed.

The best observed reference remains gamma=1/75, K=21, 50 components, reason threshold 50%, seed 42: sample-wise F1 0.8880, macro F1 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, and 46.7122% unresolved. For the new run, compare the fixed K=21 row against that reference and separately record the full sweep winner. The observed-only baseline remains 0.8654. Notebook JSON and Python syntax checks passed; no models were executed.

### 2026-11-06 — Verified gamma 1/50 results

The user executed and saved the gamma=1/50 experiment. Verified sweep/comparison execution counts **263/264**, source `KERNEL_GAMMA = 1 / 50`, and actual diagnostic **`gamma: 0.02`**. Other settings remain RBF kernel, 50 kernel PCA components, 50% reason threshold, seed 42, K-means `n_init=20`, and equal-weight logistic regression with `C=1.0`, `max_iter=500`. Gamma 1/50 is now **observed**, superseding its earlier pending status. This logging step did not modify the notebook or run models.

The full K=2–30 sweep again reports **best cluster count(s): [21]**, with no reported tie. Because both experiments select K=21, the following comparison also holds cluster count fixed:

| K=21 metric | Previous gamma 1/75 | Current gamma 1/50 |
|---|---:|---:|
| Sample-wise F1 | 0.8880 | **0.8902** |
| Macro F1 | 0.7351 | 0.7126 |
| Safety F1 | 0.8571 | 0.8182 |
| Funding F1 | 0.3333 | 0.3333 |
| Efficacy F1 | 0.8966 | 0.9032 |
| Recruitment F1 | 0.9800 | 0.9899 |
| Other F1 | 0.6087 | 0.5185 |
| Percent unresolved | 46.7122% | 57.7284% |
| Delta versus observed-only sample-wise baseline | +0.0226 | +0.0248 |

The observed-only baseline remains **0.8654**. Gamma 1/50 improves sample-wise F1 over the previous selected run by approximately **0.0022** using displayed rounded values, while macro F1 declines by **0.0225**. Efficacy and Recruitment improve, Funding is unchanged, and Safety and Other decline. The unresolved share rises by **11.0162 percentage points**, so this selected run uses fewer recovered labels. The small competition-metric gain should be weighed against those category tradeoffs and repeated selection on 78 validation trials; no claim of statistical or test-set improvement is made.

Several low-cluster-count runs are substantially worse: K=2 through 6 all display sample-wise F1 0.6496, and K=7 through 14 (except 10) are around 0.6325–0.6453. Thus the larger gamma improves the selected K=21 setting, not every clustering configuration.

Best observed competition-metric configuration is now **RBF gamma=1/50 (0.02), 50 kernel PCA components, K=21, 50% reason threshold, seed 42**, sample-wise F1 **0.8902**. Gamma=1/75 remains the stronger of these two selected settings on macro F1. No new hyperparameters were configured in this analysis.

### 2026-11-06 — User selected gamma 1/75 for per-reason balance

The user chose gamma=1/75 as the working configuration because its selected K=21 run has better macro/per-reason balance than gamma=1/50, despite slightly lower sample-wise F1. Changed only the notebook sweep assignment back to `KERNEL_GAMMA = 1 / 75`; 50 components, 50% reason threshold, other settings, and functions remain unchanged. Cleared sweep/comparison outputs and execution counts in cells 12 and 13 so prior 1/50 results are not attributed to 1/75. The sweep still evaluates K=2–30; K=21 is the previously observed selected run, not a new hard-coded cluster count.

The chosen prior observed result is sample-wise F1 0.8880, macro F1 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, and 46.7122% unresolved at K=21. Gamma=1/50, K=21, sample-wise F1 0.8902 remains recorded above as the higher-sample-F1 historical option. This selection reflects the user's preference for category balance; it does not establish superior generalization. JSON and Python syntax checks passed. No models were rerun; cells 12 and 13 need rerunning for updated outputs.

### 2026-11-06 — Pending 30-component kernel PCA experiment

At the user's request, changed only `KERNEL_PCA_COMPONENTS` from 50 to **30** in the sweep configuration (cell 12). Gamma remains **1/75**, the reason threshold remains **50%**, and the other settings/functions are unchanged. Cleared the sweep and comparison outputs/counts in cells 12 and 13 to avoid attributing 50-component outputs to this pending 30-component test. JSON and Python syntax checks passed; no models were run.

The comparison reference is the user-chosen 50-component, gamma=1/75, K=21 run: sample-wise F1 **0.8880**, macro F1 **0.7351**, Safety **0.8571**, Funding **0.3333**, Efficacy **0.8966**, Recruitment **0.9800**, Other **0.6087**, and percent unresolved **46.7122%**. The observed-only baseline sample-wise F1 is 0.8654.

The user explicitly requests holistic analysis on future iterations: assess **every per-reason F1, macro F1, sample-wise F1, and percent unresolved**, rather than judging only the competition metric. Compare the fixed **K=21** row to isolate the component change and separately assess the full sweep's selected cluster count and its category tradeoffs. The new experiment has no observed metrics until the user runs and saves it. Rerun cells 12 and 13.

### 2026-11-06 — Observed 30-component kernel PCA results

Read the fresh user-executed saved notebook with sweep/comparison execution counts **287/288**. Both configuration and actual diagnostic verify **30 kernel PCA components**, RBF gamma **1/75** (printed 0.0133333), 50% reason threshold, and seed 42. K-means still uses `n_init=20` and K=2–30. Verified that logistic regression still trains observed/recovered labels equally, uses the original scaled embeddings, and that no minimum-five-members restriction has returned. The observed-only sample-wise baseline remains **0.8654**. The 30-component experiment is now **observed**, superseding its earlier pending status; this analysis did not change the notebook or execute models.

The sample-wise sweep winner is **K=22**, with no reported tie. K=22 also has the highest displayed macro F1 in this sweep. Compare both it and the fixed K=21 row against the chosen reference:

| Metric | Chosen reference: 50 components, K=21 | 30 components, K=21 | 30 components, K=22 |
|---|---:|---:|---:|
| Sample-wise F1 | **0.8880** | 0.8778 | 0.8812 |
| Macro F1 | **0.7351** | 0.7162 | 0.7184 |
| Safety F1 | 0.8571 | 0.8000 | **0.9000** |
| Funding F1 | **0.3333** | 0.2857 | 0.2222 |
| Efficacy F1 | 0.8966 | 0.8966 | 0.8966 |
| Recruitment F1 | 0.9800 | **0.9899** | **0.9899** |
| Other F1 | **0.6087** | **0.6087** | 0.5833 |
| Percent unresolved | 46.7122% | 45.4313% | 45.6020% |
| Delta versus observed-only sample-wise baseline | +0.0226 | +0.0124 | +0.0158 |

At fixed K=21, reducing components lowers displayed sample-wise F1 by **0.0102** and macro by **0.0189**. Safety and Funding decline, Efficacy and Other stay unchanged, and Recruitment improves. The unresolved share decreases by only **1.2809 percentage points**. After reselecting K, the 30-component winner still trails the chosen reference by **0.0068** sample-wise and **0.0167** macro; it improves Safety and Recruitment while weakening Funding and Other. The tiny gain in recovery coverage is not evidence that the additional labels are correct.

Other rows illustrate category tradeoffs, without establishing superior generalization:

- **K=21** retains Other at 0.6087 and Funding at 0.2857, both better than K=22, but sacrifices Safety and some sample-wise F1. This is a reasonable alternative within the 30-component sweep if those categories matter more.
- **K=9** has the sweep's highest displayed Funding F1, **0.4000**, but sample-wise **0.8560** is below the 0.8654 baseline; macro is 0.7057, Safety 0.7500, Efficacy 0.8387, Recruitment 0.9796, Other 0.5600, unresolved 60.7173%. It does not improve overall balance relative to the chosen 50-component run.
- **K=18** has the sweep's highest displayed Other F1, **0.6364**, but sample-wise **0.7902** and macro **0.6663** are substantially worse; Safety 0.8000, Funding 0.3333, Efficacy 0.6486, Recruitment 0.9130, unresolved 45.4313%. Optimizing one category in isolation would hide these losses.

Holistic decision: retain **50 components, gamma=1/75, K=21** as the preferred observed reference. A **75-component** experiment remains informative because 30 underperformed 50 and the opposite direction has not yet been tested; this is a proposed next comparison, not an assumption that retaining more components must help. No new settings were changed. Values and differences are based on rounded saved outputs; repeated selection on 78 validation trials and unknown per-category support limit the strength of conclusions.

### 2026-11-06 — Pending 75-component experiment

At the user's request, changed only `KERNEL_PCA_COMPONENTS` from 30 to **75** in cell 12. Gamma remains **1/75**, reason threshold **50%**, seed **42**, equal observed/recovered-label weighting, and the K=2–30 sweep. Cleared sweep/comparison outputs and execution counts in cells 12 and 13 so historical 30-component outputs are not attributed to 75. JSON and Python syntax checks passed; no models were run.

The 75-component experiment is pending user execution. Retain the chosen 50-component, gamma=1/75, K=21 reference: sample-wise F1 0.8880, macro F1 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, and unresolved 46.7122%. The observed 30-component results remain historical above. Future analysis should compare fixed K=21 and the sweep winner across every reason, macro/sample F1, and unresolved percentage. Rerun cells 12 and 13.

### 2026-11-06 — Observed 75-component kernel PCA results

Verified fresh saved sweep/comparison executions **299/300**: source and diagnostic show **75 components**, RBF gamma **1/75** (0.0133333), reason threshold **50%**, and seed **42**. Other sweep/classifier settings are unchanged. The observed-only baseline remains **0.8654**. The 75-component experiment is now **observed**, superseding the pending entry. Only this log was changed; no notebook edits or model execution occurred during analysis.

The reported sample-wise winner is **K=11**, with no reported tie. **K=15** has the highest displayed macro F1 in this 75-component sweep and is the strongest alternative for a different category tradeoff:

| Metric | Reference 50 components/K21 | 75 components/K21 | 75 components/K11 winner | 75 components/K15 alternative |
|---|---:|---:|---:|---:|
| Sample-wise F1 | 0.8880 | 0.8449 | **0.8906** | 0.8838 |
| Macro F1 | **0.7351** | 0.6181 | 0.7149 | 0.7189 |
| Safety F1 | **0.8571** | 0.7778 | **0.8571** | 0.8182 |
| Funding F1 | **0.3333** | 0.0000 | 0.2857 | 0.2500 |
| Efficacy F1 | 0.8966 | 0.7778 | **0.9032** | 0.8966 |
| Recruitment F1 | 0.9800 | 0.9796 | **0.9899** | **0.9899** |
| Other F1 | 0.6087 | 0.5556 | 0.5385 | **0.6400** |
| Percent unresolved | 46.7122% | 41.0760% | 49.4449% | 47.9078% |
| Delta versus baseline sample-wise F1 | +0.0226 | -0.0205 | +0.0252 | +0.0184 |

At fixed K=21, 75 components lower sample-wise F1 by **0.0431** and macro by **0.1170**, including Funding falling to zero. After selecting a different K, the new winner improves sample-wise F1 by only **0.0026** over the chosen reference, while macro falls by **0.0202**, Funding falls by **0.0476**, and Other by **0.0702**. Safety is unchanged; Efficacy and Recruitment improve. Its unresolved share rises by **2.7327 percentage points**, so the recovery coverage is slightly lower. This is a tradeoff, not an across-category improvement.

K=15 recovers stronger Other F1 (0.6400), but has lower Funding and Safety and sample-wise F1 than the 50-component reference. It does not replace that reference under the user's stated preference for broader category balance. The earlier 30-component winner (K=22) scored sample-wise 0.8812 and macro 0.7184; neither component-count experiment improves the chosen reference's macro F1.

Current interpretation: keep **50 components/gamma=1/75/K=21** as the preferred balanced configuration, while retaining **75 components/K=11** as the higher sample-wise option. The kernel/component/K interaction is substantial; these results do not imply that increasing component count monotonically helps. All differences use rounded saved values, and repeated selection on 78 validation trials does not establish better test performance. No next settings were configured.

### 2026-11-06 — Pending 40% threshold test on the balanced representation

At the user's request, set reason threshold **40%** and restored **50 kernel PCA components**, keeping gamma **1/75**, seed 42, equal label weights, K=2–30, and all other settings/functions unchanged. Cleared sweep/comparison outputs and execution counts in cells 12 and 13; no models were run. JSON and syntax checks passed.

Reference: 50 components, gamma=1/75, threshold 50%, K=21; sample F1 0.8880, macro 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, unresolved 46.7122%. The new threshold test is pending user execution. Future analysis will compare fixed K=21 and the sweep winner across all five reason F1 scores, macro/sample F1, and unresolved percentage. Rerun cells 12 and 13.

### 2026-11-06 — Observed 40% threshold results

Verified fresh saved executions **311/312**: reason threshold **40%**, **50 kernel PCA components**, RBF gamma **1/75** (actual diagnostic 0.0133333), seed 42, and otherwise unchanged pipeline settings. The observed-only baseline remains **0.8654**. This threshold experiment is now **observed**; the logging agent did not edit notebook code or execute models.

The reported sample-wise winner is **K=24**, with no reported tie. It also has the highest displayed macro F1 in this 40% sweep; no other row offers a clearly stronger broad-category compromise.

| Metric | Reference threshold50%, K21 | Threshold40%, K21 | Threshold40%, K24 winner |
|---|---:|---:|---:|
| Sample-wise F1 | **0.8880** | 0.6752 | 0.8859 |
| Macro F1 | **0.7351** | 0.2986 | 0.7342 |
| Safety F1 | 0.8571 | 0.2000 | **0.9000** |
| Funding F1 | **0.3333** | 0.0000 | 0.2857 |
| Efficacy F1 | **0.8966** | 0.1333 | **0.8966** |
| Recruitment F1 | **0.9800** | 0.8264 | **0.9800** |
| Other F1 | **0.6087** | 0.3333 | **0.6087** |
| Percent unresolved | 46.7122% | 4.0137% | 45.6874% |
| Delta versus baseline sample-wise F1 | +0.0226 | -0.1902 | +0.0205 |

At fixed K=21, lowering the threshold decreases sample-wise F1 by **0.2128** and macro by **0.4365**. Every reason worsens, with Funding at zero. The unresolved share decreases by **42.6985 percentage points**: substantially more training rows receive recovered labels, but this is not evidence that those labels are accurate. The pattern is consistent with less selective label recovery introducing systematic mistakes; the exact affected cluster mappings would need inspection to establish the mechanism.

After reselecting K, the 40% winner comes close to the reference but remains **0.0021** lower in sample-wise F1 and **0.0009** lower in macro. It improves Safety by **0.0429**, lowers Funding by **0.0476**, and leaves Efficacy, Recruitment, and Other unchanged at displayed precision. Its unresolved share is only **1.0248 percentage points** lower. This is a Safety-versus-Funding tradeoff rather than a clear overall improvement.

Many other K values also degrade severely at 40% (K=2–16 have sample-wise F1 approximately 0.65–0.68); simply recovering nearly all rows is not a useful optimization target. Current balanced reference remains **threshold50%, 50 components, gamma1/75, K21**, with threshold40%/K24 retained as a close Safety-favoring alternative. All differences use rounded saved metrics, and repeated selection on 78 validation trials limits confidence. No further parameter changes or runs were started.

### 2026-11-06 — Pending 60% threshold experiment

At the user's request, changed only `PERCENTAGE_THRESHOLD` from 40 to **60** in cell 12. Kernel PCA remains **50 components**, gamma **1/75**, seed **42**; equal weighting and K=2–30 remain unchanged. Cleared outputs and execution counts in sweep/comparison cells 12 and 13 so historical 40% results are not presented as 60%. JSON and syntax checks passed; no models were run.

Compare the pending test against the chosen threshold50%/K21 reference: sample-wise F1 0.8880, macro F1 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, unresolved 46.7122%. The observed-only baseline sample-wise F1 remains 0.8654. The observed threshold40% results above remain historical. Analyze both fixed K21 and the sweep winner across all five reasons, macro/sample F1, and unresolved percentage once the user executes and saves the run. Rerun cells 12 and 13.

### 2026-11-06 — Observed 60% threshold results

Verified fresh saved sweep/comparison executions **323/324**: source threshold **60%**, **50 components**, RBF gamma **1/75** (actual output 0.0133333), and seed **42**. The baseline remains **0.8654**. The 60% experiment is now **observed**. This analysis modified only the log and did not execute models.

The sample-wise winner is **K=21**, with no reported tie; it also has the highest displayed macro F1. Its metrics and recovery coverage are unchanged from the chosen 50%/K21 reference at displayed precision:

| Metric | Reference 50%/K21 | 60%/K21 (winner) | 60%/K24 alternative |
|---|---:|---:|---:|
| Sample-wise F1 | **0.8880** | **0.8880** | 0.8859 |
| Macro F1 | **0.7351** | **0.7351** | 0.7342 |
| Safety F1 | 0.8571 | 0.8571 | **0.9000** |
| Funding F1 | **0.3333** | **0.3333** | 0.2857 |
| Efficacy F1 | 0.8966 | 0.8966 | 0.8966 |
| Recruitment F1 | 0.9800 | 0.9800 | 0.9800 |
| Other F1 | 0.6087 | 0.6087 | 0.6087 |
| Percent unresolved | 46.7122% | 46.7122% | 45.6874% |
| Delta versus baseline sample-wise F1 | +0.0226 | +0.0226 | +0.0205 |

All displayed fixed-K21 differences versus the reference are zero. This establishes no observed validation/coverage change for this setting, not proof that its exact recovered label arrays are identical; cluster-rate mappings would need inspection to verify that. Raising the threshold changes some other K settings, but does not improve the selected score or category balance here.

K=24 remains the closest balanced alternate, trading better Safety for worse Funding and slightly lower sample/macro F1; Efficacy, Recruitment, and Other stay the same. K=4 has higher Funding F1 (0.4000), but sample-wise 0.8496 is below baseline and macro is 0.7002, so it is not a stronger broad-category choice. No candidate clearly improves on K21 holistically. The previous 40% winner K24 scored 0.8859/0.7342, also below the current selected sample/macro scores.

Conclusion: threshold60% ties the chosen threshold50% at K21 on every displayed metric; it does not justify claiming better generalization or a new best model. Keep both settings' observed history without an automatic parameter change. Recovery coverage alone remains insufficient evidence of label accuracy. Values are rounded saved outputs, and selection across repeated sweeps on 78 validation trials remains uncertain.

### 2026-11-06 — Restored 50% threshold; proposed clustering stability check

At the user's request, restored only `PERCENTAGE_THRESHOLD = 50` in cell 12. The 50-component RBF representation, gamma1/75, seed42, equal weighting, and K=2–30 sweep remain unchanged. Cleared sweep/comparison outputs and execution counts in cells 12 and 13. JSON and syntax checks passed; no models were run.

Suggested next experiment, **not implemented or executed**: hold K=21, gamma1/75, 50 components, and threshold50% fixed; compare K-means seeds **0, 21, 42** while keeping kernel PCA and logistic regression seeds fixed at **42**. This isolates clustering stochasticity rather than changing every randomized stage. Analyze all five reason F1 values, macro/sample F1, and unresolved percentage against the chosen seed42 reference (sample0.8880, macro0.7351, unresolved46.7122%). The user has not approved a seed-loop implementation yet.

### 2026-11-06 — Pending K-means seed0 stability experiment

At the user's request, introduced `KMEANS_RANDOM_STATE = 0` for `train_kmeans` only, retaining `RANDOM_STATE = 42` for kernel PCA and both logistic-regression fits. Changed the cluster loop to `[21]` for a fixed-configuration stability test rather than selecting a new K for this seed. Gamma1/75, 50 components, threshold50%, `n_init=20`, equal label weights, and logistic `C=1.0`/`max_iter=500` remain unchanged. Each result records `kmeans_random_state` for provenance; a comment explains how to restore the full K sweep later.

The test is configured but pending user execution. Reference K-means seed42 at the same fixed configuration: sample-wise F1 0.8880, macro F1 0.7351, Safety 0.8571, Funding 0.3333, Efficacy 0.8966, Recruitment 0.9800, Other 0.6087, unresolved46.7122%; baseline sample-wise F1 0.8654. Compare every metric after the user runs and saves it. Cleared sweep/comparison outputs and execution counts in cells12–13; JSON and syntax checks passed. No models were run. Rerun cells12 and13.

### 2026-11-06 — Observed seed0 stability result at fixed K21

Verified fresh saved executions **335/336**: `KMEANS_RANDOM_STATE=0`, loop **[21]**, and kernel PCA/logistic seeds held at **42**. Gamma remains **1/75**, kernel PCA components **50**, reason threshold **50%**, and K-means `n_init=20`; other settings are unchanged. The observed-only baseline remains **0.8654**. Seed0 is now **observed**. Only this log was modified; the logging agent did not run models.

| Metric at fixed K21 | Reference K-means seed42 | K-means seed0 |
|---|---:|---:|
| Sample-wise F1 | 0.8880 | 0.8457 |
| Macro F1 | 0.7351 | 0.6748 |
| Safety F1 | 0.8571 | 0.7273 |
| Funding F1 | 0.3333 | 0.3333 |
| Efficacy F1 | 0.8966 | 0.8571 |
| Recruitment F1 | 0.9800 | 0.9800 |
| Other F1 | 0.6087 | 0.4762 |
| Percent unresolved | 46.7122% | 34.0734% |
| Delta versus observed-only sample-wise baseline | +0.0226 | -0.0197 |

Changing only the clustering seed lowers displayed sample-wise F1 by **0.0423** and macro by **0.0603**. Safety and Other decline most, Efficacy also declines, while Funding and Recruitment remain unchanged. Unresolved coverage decreases by **12.6388 percentage points**, showing that recovering more labels does not ensure better validation predictions. The result is consistent with the clustering/cluster-label mapping being sensitive to initialization; exact cluster/pseudo-label differences were not measured here.

This exposes meaningful seed sensitivity rather than validating a stable improvement over the supervised baseline: one observed seed beats baseline and one falls below it. With two seeds, no stable mean or generalization claim is justified. The output's "best cluster count [21]" merely reflects the single fixed candidate, not a fresh K search. Proposed seed21 remains **planned and unconfigured**; its purpose is to assess variation under the same fixed setup, not to select whichever random seed maximizes validation F1. All quoted differences use rounded saved metrics.

### 2026-11-06 — Pending K-means seed21 stability check

At the user's request, changed only `KMEANS_RANDOM_STATE` from 0 to **21** in cell12. Cluster count stays fixed at **21**; kernel PCA/logistic seeds remain42, with50components, gamma1/75, threshold50%, n_init20, and equal weights unchanged. Cleared outputs/counts in cells12–13 to avoid showing seed0 results as seed21. JSON and syntax checks passed; no models were executed.

The seed21 run is pending user execution. Historical seed42 and seed0 observations remain above. Once saved, compare all per-reason F1 values, sample/macro F1, and unresolved percentage across seeds0/21/42, summarizing means and ranges without choosing a lucky seed from validation. Rerun cells12 and13.

### 2026-11-06 — Observed seed21 result and three-seed stability summary

Verified fresh saved executions **347/348**: K-means seed **21**, fixed **K=21**, kernel PCA/logistic seeds **42**, RBF gamma **1/75**, **50 components**, threshold **50%**, `n_init=20`, and equal label weights. The baseline sample-wise F1 remains **0.8654**. Seed21 is now **observed**. Only this log was modified; no notebook code or models were changed or executed during analysis.

| Metric | Seed42 | Seed0 | Seed21 | Approximate mean | Observed range |
|---|---:|---:|---:|---:|---|
| Sample-wise F1 | 0.8880 | 0.8457 | 0.8799 | 0.8712 | 0.8457–0.8880 |
| Macro F1 | 0.7351 | 0.6748 | 0.7186 | 0.7095 | 0.6748–0.7351 |
| Safety F1 | 0.8571 | 0.7273 | 0.8000 | 0.7948 | 0.7273–0.8571 |
| Funding F1 | 0.3333 | 0.3333 | 0.3333 | 0.3333 | 0.3333–0.3333 |
| Efficacy F1 | 0.8966 | 0.8571 | 0.8966 | 0.8834 | 0.8571–0.8966 |
| Recruitment F1 | 0.9800 | 0.9800 | 0.9800 | 0.9800 | 0.9800–0.9800 |
| Other F1 | 0.6087 | 0.4762 | 0.5833 | 0.5561 | 0.4762–0.6087 |
| Percent unresolved | 46.7122% | 34.0734% | 49.1033% | 43.2963% | 34.0734%–49.1033% |
| Sample-wise delta versus baseline | +0.0226 | -0.0197 | +0.0145 | +0.0058 | -0.0197–+0.0226 |

These means/ranges are arithmetic summaries of **rounded saved scores**, not metrics of an ensemble. Seed21 improves over baseline but does not remove the earlier sensitivity: the observed sample-wise span is **0.0423** and macro span **0.0603**. The three-seed mean exceeds baseline by only approximately **0.0058**, with one of three runs below baseline. Safety (span **0.1298**) and Other (**0.1325**) vary most; Efficacy varies less, while Funding and Recruitment have unchanged displayed F1. Equal F1 values do not establish identical predictions.

The unresolved share spans **15.0299 percentage points**. Seed0 recovers the most labels but has the weakest validation scores, again showing that recovery coverage is not a measure of label correctness. Three seeds on the same small, repeatedly used validation set are insufficient to claim robust generalization or select a lucky seed.

Suggested next stability experiment, **not implemented**: increase K-means `n_init` from20 to100 at the same fixed model configuration, then repeat seeds0/21/42. K-means selects the lowest training inertia among its initializations, not the highest validation F1. More restarts may reduce optimization variation, but do not guarantee more meaningful clusters or improved reason predictions. Track inertia alongside all current metrics if that test is approved; do not autochange any parameters in this analysis.

### 2026-11-06 — Pending n_init100 experiment at seed21

At the user's request, increased K-means restarts to **100**. Added `N_INIT=100` to the pipeline configuration, passed it explicitly to `train_kmeans`, and changed the helper's default from20 to100 for consistency. Each stored run now records `n_init` for provenance. K-means seed21, fixedK21, kernel PCA/logistic seeds42, gamma1/75, 50components, threshold50%, and equal weights remain unchanged.

This run is pending user execution. Compare against n_init20/seed21: sample F1 0.8799, macro0.7186, Safety0.8000, Funding0.3333, Efficacy0.8966, Recruitment0.9800, Other0.5833, unresolved49.1033%; baseline sample0.8654. Repeating other clustering seeds is a future step, not implemented here. Cleared outputs/counts in helper cell4 and pipeline/comparison cells12–13; JSON, syntax, and active-call checks passed. No model was executed. Rerun cells4,12,13.

### 2026-11-06 — Observed n_init100 result at seed21

Verified fresh saved helper execution **352** and pipeline/comparison executions **359/360**. The helper default is100 and the pipeline explicitly sets `N_INIT=100` and passes it to `train_kmeans`. K-means seed **21**, fixed **K21**, kernel PCA/logistic seeds42, gamma1/75, 50components, threshold50%, and equal weights remain unchanged. The n_init100/seed21 test is now **observed**. No models were executed or notebook code changed during this analysis.

| Metric at K21/seed21 | n_init20 | n_init100 |
|---|---:|---:|
| Sample-wise F1 | 0.8799 | 0.8799 |
| Macro F1 | 0.7186 | 0.7186 |
| Safety F1 | 0.8000 | 0.8000 |
| Funding F1 | 0.3333 | 0.3333 |
| Efficacy F1 | 0.8966 | 0.8966 |
| Recruitment F1 | 0.9800 | 0.9800 |
| Other F1 | 0.5833 | 0.5833 |
| Percent unresolved | 49.1033% | 49.1033% |
| Delta versus baseline sample-wise F1 | +0.0145 | +0.0145 |

Every displayed metric and recovery-coverage value is unchanged. The baseline remains0.8654. More restarts have not improved this particular seed's observable validation result; this does not establish identical cluster assignments, recovered labels, or inertia. Inertia is not included in the saved output, so no inertia comparison is claimed.

For context only, the n_init20/seed42 reference was0.8880 sample-wise and0.7351 macro; that comparison changes seed and cannot isolate the restart-count effect. A single n_init100 seed does not demonstrate reduced cross-seed variability. The next informative stability comparison is **seed0 with n_init100**, since seed0/init20 underperformed baseline; hold every other setting fixed and compare all categories and recovery coverage. This next test is proposed only, not configured or run. Do not choose a seed by the highest validation score.

### 2026-11-06 — Pending seed0/n_init100 stability test

At the user's request, changed only `KMEANS_RANDOM_STATE` from21 to **0** in cell12. Kept `N_INIT=100`, fixedK21, kernel PCA/logistic seeds42, gamma1/75, 50components, threshold50%, and equal weights. Cleared pipeline/comparison outputs and execution counts in cells12–13. JSON and syntax checks passed; no models were run.

The test is pending user execution. Compare directly against seed0/n_init20: sample F1 **0.8457**, macro **0.6748**, Safety **0.7273**, Funding **0.3333**, Efficacy **0.8571**, Recruitment **0.9800**, Other **0.4762**, and unresolved **34.0734%**; baseline0.8654. The seed21/n_init100 result above remains historical and matched its n_init20 scores. Analyze all reason metrics and coverage rather than selecting a lucky seed. Rerun cells12 and13.

### 2026-11-06 — Observed seed0/n_init100 result

Verified fresh saved executions **371/372**: K-means seed **0**, **n_init100**, fixed **K21**, kernel PCA/logistic seeds42, RBF gamma1/75, 50components, and threshold50%. The baseline remains **0.8654**. This experiment is now **observed**. The logging agent only inspected saved output and modified this log; no notebook changes or model execution occurred.

| Metric | Seed0/init20 | Seed0/init100 | Seed21/init100 | Reference seed42/init20 |
|---|---:|---:|---:|---:|
| Sample-wise F1 | 0.8457 | 0.8816 | 0.8799 | 0.8880 |
| Macro F1 | 0.6748 | 0.7392 | 0.7186 | 0.7351 |
| Safety F1 | 0.7273 | 0.9000 | 0.8000 | 0.8571 |
| Funding F1 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| Efficacy F1 | 0.8571 | 0.8571 | 0.8966 | 0.8966 |
| Recruitment F1 | 0.9800 | 0.9899 | 0.9800 | 0.9800 |
| Other F1 | 0.4762 | 0.6154 | 0.5833 | 0.6087 |
| Percent unresolved | 34.0734% | 45.2605% | 49.1033% | 46.7122% |
| Sample-wise delta versus baseline | -0.0197 | +0.0162 | +0.0145 | +0.0226 |

At the same seed0, increasing restarts improves displayed sample-wise F1 by **0.0359**, macro by **0.0644**, Safety by **0.1727**, Other by **0.1392**, and Recruitment by **0.0099**. Funding and Efficacy are unchanged. The unresolved share increases by **11.1871 percentage points**; using fewer recovered rows accompanies better predictions here, but the exact recovered-label accuracy was not measured. This is evidence that100 restarts helped the formerly weak seed0 run, not a claim that every seed or category must improve.

For the matched pair of seeds0/21, sample-wise range shrinks from **0.8457–0.8799** at init20 (span0.0342) to **0.8799–0.8816** at init100 (span0.0017); macro span shrinks from0.0438 to0.0206. Both observed init100 runs exceed baseline, but two seeds remain insufficient to establish broad stability. Safety still differs by0.1000 between these two init100 runs, while Efficacy and Other also show category tradeoffs.

The new run's macro F1 (0.7392) exceeds the historical chosen seed42/init20 reference (0.7351), but sample-wise F1 is lower (0.8816 versus0.8880), with Efficacy lower and Safety/Other higher. That cross-seed comparison is contextual, not an isolated restart-count comparison or a reason to select a lucky seed. No saved inertia values are available to compare optimizer objectives.

Proposed next stability check: **seed42 with init100**, keeping every other setting fixed, to complete the same three-seed comparison before judging stability. It remains unconfigured and unrun; no next parameters were changed. All numerical differences use rounded saved metrics, and validation-selection uncertainty remains.

### 2026-11-06 — Pending seed42/n_init100 stability check

At the user's request, changed only `KMEANS_RANDOM_STATE` from0 to **42** in cell12, retaining `N_INIT=100`, fixedK21, kernel PCA/logistic seeds42, gamma1/75, 50components, threshold50%, and equal weights. Cleared pipeline/comparison outputs and execution counts in cells12–13. JSON and syntax checks passed; no models were run.

This final seed test is pending user execution. Preserve the observed init100 outcomes for seed0 (sample0.8816, macro0.7392, unresolved45.2605%) and seed21 (sample0.8799, macro0.7186, unresolved49.1033%). Once seed42 is saved, compare every reason F1, sample/macro F1, and unresolved coverage and summarize the matched three-seed mean/ranges; do not select a lucky seed. The historical seed42/init20 result remains the direct comparison reference (sample0.8880, macro0.7351, unresolved46.7122%). Rerun cells12 and13.

### 2026-11-06 — Observed seed42/n_init100 and completed stability trio

Verified fresh saved executions **383/384**: K-means seed **42**, `N_INIT=100` explicitly passed, fixed **K21**, kernel PCA/logistic seeds42, gamma1/75, 50components, and threshold50%. Baseline sample-wise F1 remains **0.8654**. The final init100 seed test is now **observed**. This inspection changed only the log; no notebook code or models were changed or run.

| Metric at n_init100 | Seed0 | Seed21 | Seed42 | Approximate mean |
|---|---:|---:|---:|---:|
| Sample-wise F1 | 0.8816 | 0.8799 | 0.8880 | 0.8832 |
| Macro F1 | 0.7392 | 0.7186 | 0.7351 | 0.7310 |
| Safety F1 | 0.9000 | 0.8000 | 0.8571 | 0.8524 |
| Funding F1 | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| Efficacy F1 | 0.8571 | 0.8966 | 0.8966 | 0.8834 |
| Recruitment F1 | 0.9899 | 0.9800 | 0.9800 | 0.9833 |
| Other F1 | 0.6154 | 0.5833 | 0.6087 | 0.6025 |
| Percent unresolved | 45.2605% | 49.1033% | 46.7122% | 47.0253% |
| Sample-wise delta versus baseline | +0.0162 | +0.0145 | +0.0226 | +0.0178 |

Seed42/init100 matches every displayed seed42/init20 metric and coverage value; this does not establish identical clusters or labels. The meaningful result is the matched three-seed comparison:

| Summary across seeds0/21/42 | n_init20 | n_init100 |
|---|---:|---:|
| Mean sample-wise F1 | 0.8712 | 0.8832 |
| Sample-wise range | 0.8457–0.8880 | 0.8799–0.8880 |
| Sample-wise span | 0.0423 | 0.0081 |
| Mean macro F1 | 0.7095 | 0.7310 |
| Macro range | 0.6748–0.7351 | 0.7186–0.7392 |
| Macro span | 0.0603 | 0.0206 |
| Unresolved percentage range | 34.0734%–49.1033% | 45.2605%–49.1033% |
| Unresolved percentage span | 15.0299 points | 3.8428 points |

All three init100 runs now exceed the observed-only sample-wise baseline; the mean gain is approximately **0.0178**, versus0.0058 for init20. The sample/macro means increase by approximately0.0120/0.0215. These are summaries of rounded saved scores, not performance of an ensemble or confidence estimates. The improvement mainly comes from rescuing seed0; seed21 and seed42 retain their displayed results.

Category variability remains: Safety spans **0.8000–0.9000**, Efficacy **0.8571–0.8966**, Other **0.5833–0.6154**, and Recruitment **0.9800–0.9899**. Funding remains a stable weakness at0.3333. Other's span improves from0.1325 to0.0321; Safety's span improves from0.1298 to0.1000 but is still substantial. Coverage alone does not establish pseudo-label accuracy.

Conclusion: **100 restarts improves observed stability across the three preselected seeds**, without proving generalization to future trials or all initializations. Stop searching random seeds for validation gains. Retaining the originally documented seed42 with init100 is a reproducible working choice; it is not a new seed selected by a larger validation search. Any future consensus model should be a separately defined method rather than an average of these reported metrics. No new settings or tests were configured.

## 2026-11-06 — Part 1 Kaggle test export

Executed the selected, previously validated pipeline to generate `part1_kmeans_kernel_pca_submission.csv`. Configuration: training-fitted StandardScaler on the 150 reason embeddings; RBF kernel PCA with 50 components, gamma 1/75 and seed42 for label recovery only; K-means K21, seed42, n_init100; reason threshold50%; equal-weight logistic regression (C1, liblinear, max_iter500, seed42) on original scaled embeddings. No minimum-label-support rule or recovered-label weights. Validation labels were used only for evaluation; no training on validation or test labels.

Actual recomputed validation F1: Sample-wise F1=0.8880, Macro F1=0.7351, Safety=0.8571, Funding=0.3333, Efficacy=0.8966, Recruitment=0.9800, Other=0.6087. Observed/recovered/unresolved training counts: {'unresolved': 547, 'cluster_prediction': 467, 'observed': 157}.

Exported 312 test rows with exactly `ID,Predicted`, original test order and unique IDs matching the provided sample submission. Labels are space-separated names from the five allowed reasons. 9 rows have no predicted reasons; these remain empty strings, with no forced label. Checked binary predictions shape, CSV readback, ID set/order, allowed names and the final classifier’s 150-dimensional input (clustering has 50-dimensional input).

Added modular `export_part1_predictions()` and a selected-model export call in notebook cells14–15. All nonempty notebook cells were executed successfully in the existing Cornell environment and real outputs/counts saved. Execution and CSV checks took 17.65 seconds. No Kaggle upload was performed; the user will submit the CSV.

## 2026-11-06 — Nonempty Kaggle prediction correction

Kaggle rejected the initial CSV because nine all-zero model predictions serialized as empty reason fields. Added shared `predict_reason_labels(model, X)`: preserve all existing positive predictions; for all-zero rows only, select the single reason with highest predicted probability. Both validation evaluation and export now use this same postprocessing. Training, cluster mappings, and unresolved training rows remain unchanged.

Regenerated `part1_kmeans_kernel_pca_submission.csv` with 312 rows and zero null/blank fields. Checked exact columns `ID,Predicted`, unique IDs in original test order matching sample submission, allowed reason names, and per-row argmax fallbacks. Exactly nine formerly empty predictions changed; all 303 originally nonempty predictions are preserved. Assignments: [{'ID': 'TR09690', 'Predicted': 'Other'}, {'ID': 'TR06209', 'Predicted': 'Other'}, {'ID': 'TR03889', 'Predicted': 'Recruitment'}, {'ID': 'TR02120', 'Predicted': 'Recruitment'}, {'ID': 'TR11120', 'Predicted': 'Recruitment'}, {'ID': 'TR08185', 'Predicted': 'Recruitment'}, {'ID': 'TR00180', 'Predicted': 'Funding'}, {'ID': 'TR00275', 'Predicted': 'Safety'}, {'ID': 'TR07421', 'Predicted': 'Other'}].

Actual selected-pipeline validation fallbacks: 2; baseline validation fallbacks: 4. Consistent postprocessed selected F1: {'Sample-wise F1': 0.888034188034188, 'Macro F1': 0.7272271078039481, 'Safety': 0.8571428571428571, 'Funding': 0.3333333333333333, 'Efficacy': 0.8666666666666667, 'Recruitment': 0.9702970297029703, 'Other': 0.6086956521739131}; baseline F1: {'Sample-wise F1': 0.8782051282051284, 'Macro F1': 0.6989897959183673, 'Safety': 0.8571428571428571, 'Funding': 0.2857142857142857, 'Efficacy': 0.8125, 'Recruitment': 0.9795918367346939, 'Other': 0.56}. Previous raw selected sample/macro F1 were 0.8880341880/0.7351447134 and raw baseline sample F1 was 0.8653846154; the postprocessing changes some class/macro and baseline metrics, so the old comparison must not be reused.

The selected kernel PCA50/gamma1/75 + K-means21/seed42/n_init100 + threshold50% pipeline was executed successfully in the existing Cornell environment. The final classifier continues to use the original 150 standardized embedding features. During initial output capture, the user concurrently ran/saved the same updated notebook (executions400–414); newer source and actual outputs were preserved rather than overwritten. Notebook cells8–9 contain the helper/evaluator, cell15 the exporter, and cell16 exports the CSV. An independent rerun verified fallback counts and artifact consistency. No Kaggle upload was performed.

## 2026-11-06 — User-reported Kaggle score approximately 0.82

The user reports a Kaggle sample-wise F1 of **approximately 0.82** for the selected exported model. This is a **user-reported approximate score**, not independently verified through Kaggle; no test per-reason metrics are available or inferred.

Checked the current saved notebook, including pipeline/comparison executions **441/442** and export execution444. The configuration remains RBF kernel PCA50/gamma1/75, K-meansK21/seed42/n_init100, reason threshold50%, equal-weight logistic regression on original scaled embeddings, and the shared mandatory-nonempty prediction fallback. Saved validation sample-wise F1 is **0.8880**, macro F1 **0.7272**, Safety **0.8571**, Funding **0.3333**, Efficacy **0.8667**, Recruitment **0.9703**, Other **0.6087**, and unresolved training fraction **46.7122%**. The consistently postprocessed supervised-only baseline sample-wise F1 is **0.8782**, making the current validation improvement **+0.0098**. Historical raw-output macro0.7351 and baseline0.8654 are not the current like-for-like comparison. The saved export has312 rows, no empty predictions, and9 argmax fallback cases.

The approximate validation-to-reported-Kaggle gap is **0.8880−0.82≈0.068**, or about6.8 percentage points. Plausible explanations include the assignment's chronological shift (Part1 test contains later2010–2016 trials; validation has only78 earlier trials), extensive selection on the same small validation set, and cluster-label estimates inherited from157 visible training labels biased toward earlier trials and selected rare reasons. These are hypotheses, not measured attributions of the score drop.

A single user-reported Kaggle score does not establish whether augmentation improves or harms hidden-label performance. If a controlled comparator is desired, an observed-label-only submission using the **same nonempty postprocessing** would isolate that question more fairly; no comparator export/upload or code change was made here. Further hyperparameter selection should not be driven by the hidden-test leaderboard score. Only this log was modified; no models were run and no Kaggle access occurred.

## 2026-11-06 — Planned modular GMM clustering comparison

The user requested a GMM comparison. Added separate `train_gmm(X_train, n_components, covariance_type="diag", n_init=5, max_iter=300, reg_covar=1e-4, random_state=42)` to fit and return a Gaussian mixture using EM. The pipeline now defaults to `CLUSTERING_METHOD="gmm"`, retaining a readable `"kmeans"` branch and the original K-means helper. Generalized reason inspection to obtain group counts from either `n_clusters` or `n_components`.

First GMM configuration: **21 components, diagonal covariance, 5 initializations, max_iter300, reg_covar0.0001, seed42**, on the existing **50-component RBF kernel PCA representation (gamma1/75, seed42)**. Keep the **50%** reason threshold, hard `.predict()` cluster assignments, equal-weight logistic regression on original standardized150-dimensional embeddings, and the same nonempty prediction fallback for evaluation/export. Soft membership propagation is a later possible experiment, not implemented here. Kernel PCA already makes the current clustering nonlinear in the original embedding space; GMM tests a different cluster-shape/density assumption, not an automatic fix or guaranteed improvement.

Each run records a generic clustering model, method-specific model key (`gmm_model` or `kmeans_model`), and actual method-specific settings. GMM convergence and iteration count are printed when the real pipeline runs. The current fixed group count is21, without a new component/covariance search. The like-for-like saved K-means reference remains sample-wise F1 **0.8880**, macro **0.7272**, Safety **0.8571**, Funding **0.3333**, Efficacy **0.8667**, Recruitment **0.9703**, Other **0.6087**, unresolved **46.7122%**, and observed-only baseline sample-wise **0.8782**, all with current nonempty postprocessing.

The selected export now derives its filename from the stored clustering method: `part1_gmm_kernel_pca_submission.csv` for GMM versus the original K-means filename. The existing K-means CSV was not changed, and no GMM CSV was generated. Cleared changed/dependent outputs and counts, including the selected export call, to avoid showing K-means results as GMM results.

Verification: notebook JSON, unique cell IDs, Python syntax, and representation routing passed. A small synthetic-only GMM fit verified convergence, hard assignments, multi-label cluster mapping/recovery, preservation of original labels/input, and K-means-compatible inspection. **The real GMM trial-data experiment has not run; no GMM validation or Kaggle score is claimed.** Run updated inspector cell5, new GMM helper cell13, pipeline cell14, and comparison cell15 after the existing imports/data/functions are available. No actual model export or Kaggle upload occurred.

The user also reports an approximate **0.62 supervised-only Kaggle baseline** versus approximately **0.82** for the selected augmented model. Both are user-reported and unverified through Kaggle; identical submission/postprocessing conditions have not been independently established here. These reports are historical context, not the basis for tuning new GMM parameters on hidden test scores.

## 2026-11-06 — Separate K-means and GMM notebooks

At the user's request, restored `Part1_Clustering_Walkthrough.ipynb` to the selected K-means-only workflow and created independent `Part1_GMM_Walkthrough.ipynb`. Each notebook contains its own imports, CSV loading, scaling, PCA/kernel-PCA helpers, clustering helper, cluster interpretation, label recovery, logistic regression, nonempty prediction helper, evaluation, and exporter; neither depends on the other's kernel variables.

The K-means notebook retains fixed K21, seed42/n_init100, RBF kernel PCA50/gamma1/75, threshold50%, equal-weight logistic regression on original standardized150-dimensional embeddings, unresolved-label exclusion, and the shared nonempty prediction fallback. Removed the GMM helper, method switch, GMM configuration, and generic inspection branches from this notebook. The GMM notebook uses 21 diagonal-covariance components, n_init5, max_iter300, reg_covar0.0001, seed42, and the same preprocessing/label-recovery/classifier settings. It uses hard membership assignments for this first comparison. All GMM notebook cells are unexecuted to avoid carrying results from the mixed notebook into the new file.

Before splitting, inspected real saved user-run outputs in the mixed notebook (pipeline457, metrics458, export460): GMM converged in37 iterations; sample-wise F1=0.8547, macro F1=0.6763, Safety=0.8571, Funding=0.2857, Efficacy=0.7500, Recruitment=0.9703, Other=0.5185, unresolved=44.7481%, observed-only baseline sample=0.8782, and displayed baseline delta=-0.0235. The saved export reported312 rows, zero empty reason strings, and11 argmax fallback cases. These are previously saved user outputs read during the split, not a new run by this agent. Their associated outputs were cleared from the restored K-means pipeline so they cannot be mistaken for K-means results.

Exports now have explicit distinct filenames: `part1_kmeans_kernel_pca_submission.csv` and `part1_gmm_kernel_pca_submission.csv`. Both existing CSV artifacts were preserved byte-for-byte; no CSV was generated or uploaded. JSON, Python syntax, unique cell IDs, independent imports/model routing, original150-dimensional classifier input, prediction fallback routing, and separate export paths were checked statically. No model fitting, notebook execution, dependency installation, or Kaggle access occurred. Run either notebook from the top in its own kernel to reproduce its method.

### 2026-11-06 — GMM component-count sweep configured

Updated only `Part1_GMM_Walkthrough.ipynb` to try 2–30 mixture components, inclusive. Kernel PCA is fitted once outside the loop; each run stores its GMM, hard-assignment reason mapping, recovered training DataFrame, classifier, all seven validation F1 metrics, label-source counts, and GMM configuration. GMM convergence and iteration count are stored and shown alongside F1, unresolved percentage, and the baseline difference.

Settings remain RBF kernel PCA with 50 components and gamma 1/75, reason threshold 50%, diagonal covariance, five GMM initializations, maximum 300 EM iterations, covariance regularization 1e-4, and seed 42. Logistic regression still uses the original scaled embeddings, observed and recovered labels equally, and the shared nonempty-prediction fallback. This sweep is configured but unexecuted: no new metrics or exports are claimed. Review all per-reason scores, macro/sample F1, unresolved coverage, and convergence; compare the 21-component reference as well as sweep winners.

The export cell explicitly selects `GMM_EXPORT_COMPONENTS = 21` from `pipeline_results`. Change this after reviewing the table to export another candidate. It cannot silently export the loop's final 30-component classifier. The export filename remains `part1_gmm_kernel_pca_submission.csv`. Existing CSVs and the separate K-means notebook were preserved. Notebook JSON, source syntax, and sweep/export routing checks passed; no training or export was run.

## 2026-11-06 — Observed diagonal-GMM component sweep (2–30)

Read actual saved user-run outputs in `Part1_GMM_Walkthrough.ipynb`, pipeline execution12 and comparison execution13. This is inspection of completed user execution, not a new training run by the logging agent. Confirmed RBF kernel PCA50/gamma1/75/seed42, reason threshold50%, diagonal GMM covariance, n_init5, max_iter300, reg_covar1e-4, GMM seed42, hard cluster assignments, equal-weight logistic regression on original scaled150-dimensional embeddings, and shared mandatory-nonempty prediction fallback.

All **29 mixtures converged**, taking **9–62 EM iterations**. The selected 21-component fit took37 iterations. Convergence establishes optimization termination, not good reason-label recovery. The consistently postprocessed observed-only baseline sample-wise F1 is **0.8782**. **Every GMM configuration scores below that baseline.**

Best sample-wise F1 and best macro F1 occur at the same count, **21 components**, with no reported best-sample tie. Like-for-like comparison uses the historical postfallback K-means reference verified in this log; the restored K-means notebook currently has cleared dependent outputs and was not rerun here. Do not substitute its historical pre-fallback macro0.7351.

| Metric | K-means21 / n_init100 / seed42 | Best GMM21 |

|---|---:|---:|

| Sample-wise F1 | **0.8880** | 0.8547 |

| Macro F1 | **0.7272** | 0.6763 |

| Safety | 0.8571 | 0.8571 |

| Funding | **0.3333** | 0.2857 |

| Efficacy | **0.8667** | 0.7500 |

| Recruitment | 0.9703 | 0.9703 |

| Other | **0.6087** | 0.5185 |

| Unresolved training fraction | 46.7122% | 44.7481% |


GMM21 loses approximately0.0333 sample-wise F1 and0.0509 macro F1 versus K-means using rounded outputs; its baseline delta is−0.0235. Its roughly1.96-percentage-point increase in recovery coverage does not compensate for worse Funding, Efficacy, and Other classification. Safety and Recruitment scores tie.

No distinct GMM configuration has higher macro F1 than the sample winner. GMM19 improves Other to0.6667 but has sample0.8162, macro0.6007, Safety0.5333, Funding0.2857, Efficacy0.5833, Recruitment0.9346, and38.1725% unresolved; this is a category tradeoff, not a stronger balanced model. GMM3/4 Funding0.4000 likewise accompanies much weaker overall scores.

These findings apply to **diagonal-covariance GMM plus hard membership label propagation on this representation**; they do not establish that every GMM configuration or soft-propagation method is worse. More flexible density modeling does not guarantee reason-aligned clusters. Retain K-means as the stronger current validation reference. User-reported approximate Kaggle0.82 for K-means and0.62 for the supervised baseline remain unverified contextual reports; no GMM Kaggle score was measured here. No notebook, model, CSV, or upload was changed by this analysis.

### Full saved sweep table

Values below are copied from the notebook table rounded to four decimals; selection in the notebook used original unrounded scores.

| Components | Sample F1 | Macro F1 | Safety | Funding | Efficacy | Recruitment | Other | Unresolved % | Delta vs baseline | Converged | EM iterations |

| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| 2 | 0.6496 | 0.2351 | 0.2000 | 0.0000 | 0.0000 | 0.7937 | 0.1818 | 0.0000 | -0.2286 | True | 13 |

| 3 | 0.7308 | 0.5003 | 0.6154 | 0.4000 | 0.3529 | 0.8475 | 0.2857 | 32.2801 | -0.1474 | True | 9 |

| 4 | 0.7350 | 0.4853 | 0.5333 | 0.4000 | 0.3529 | 0.8547 | 0.2857 | 26.7293 | -0.1432 | True | 25 |

| 5 | 0.7714 | 0.5696 | 0.6667 | 0.3333 | 0.4444 | 0.8772 | 0.5263 | 36.0376 | -0.1068 | True | 18 |

| 6 | 0.7799 | 0.5821 | 0.6250 | 0.3333 | 0.5000 | 0.8807 | 0.5714 | 34.2442 | -0.0983 | True | 16 |

| 7 | 0.8521 | 0.6642 | 0.8000 | 0.2857 | 0.7143 | 0.9608 | 0.5600 | 52.5192 | -0.0261 | True | 28 |

| 8 | 0.7774 | 0.5786 | 0.6667 | 0.3333 | 0.4444 | 0.8772 | 0.5714 | 34.7566 | -0.1009 | True | 19 |

| 9 | 0.7628 | 0.5502 | 0.5714 | 0.3333 | 0.4211 | 0.8696 | 0.5556 | 31.5969 | -0.1154 | True | 23 |

| 10 | 0.7607 | 0.5536 | 0.7143 | 0.3333 | 0.5000 | 0.8673 | 0.3529 | 29.3766 | -0.1175 | True | 22 |

| 11 | 0.7308 | 0.4559 | 0.6154 | 0.0000 | 0.2353 | 0.8403 | 0.5882 | 24.5944 | -0.1474 | True | 62 |

| 12 | 0.7479 | 0.4850 | 0.6154 | 0.0000 | 0.4211 | 0.8621 | 0.5263 | 25.5337 | -0.1303 | True | 25 |

| 13 | 0.8205 | 0.6145 | 0.6667 | 0.2857 | 0.6400 | 0.9615 | 0.5185 | 40.2220 | -0.0577 | True | 19 |

| 14 | 0.8120 | 0.5921 | 0.5000 | 0.2857 | 0.7407 | 0.9515 | 0.4828 | 42.5278 | -0.0662 | True | 25 |

| 15 | 0.8226 | 0.6229 | 0.7273 | 0.2857 | 0.6400 | 0.9615 | 0.5000 | 43.7233 | -0.0556 | True | 22 |

| 16 | 0.8291 | 0.6226 | 0.6250 | 0.2857 | 0.7407 | 0.9615 | 0.5000 | 44.0649 | -0.0491 | True | 20 |

| 17 | 0.7863 | 0.5682 | 0.5000 | 0.3333 | 0.6154 | 0.9159 | 0.4762 | 37.4893 | -0.0919 | True | 21 |

| 18 | 0.6962 | 0.5627 | 0.4706 | 0.3333 | 0.7407 | 0.8478 | 0.4211 | 38.2579 | -0.1821 | True | 15 |

| 19 | 0.8162 | 0.6007 | 0.5333 | 0.2857 | 0.5833 | 0.9346 | 0.6667 | 38.1725 | -0.0620 | True | 22 |

| 20 | 0.7286 | 0.5931 | 0.5333 | 0.3333 | 0.7200 | 0.8333 | 0.5455 | 34.2442 | -0.1496 | True | 22 |

| 21 | 0.8547 | 0.6763 | 0.8571 | 0.2857 | 0.7500 | 0.9703 | 0.5185 | 44.7481 | -0.0235 | True | 37 |

| 22 | 0.7094 | 0.5796 | 0.4571 | 0.3333 | 0.6923 | 0.8696 | 0.5455 | 43.8941 | -0.1688 | True | 16 |

| 23 | 0.8235 | 0.6265 | 0.7059 | 0.2857 | 0.7143 | 0.9600 | 0.4667 | 45.3459 | -0.0547 | True | 25 |

| 24 | 0.8120 | 0.6148 | 0.5000 | 0.3333 | 0.7500 | 0.9524 | 0.5385 | 38.5995 | -0.0662 | True | 30 |

| 25 | 0.8355 | 0.6420 | 0.6667 | 0.3333 | 0.6667 | 0.9346 | 0.6087 | 39.4535 | -0.0427 | True | 32 |

| 26 | 0.8184 | 0.6193 | 0.6154 | 0.3333 | 0.7200 | 0.9515 | 0.4762 | 36.0376 | -0.0598 | True | 16 |

| 27 | 0.7115 | 0.5855 | 0.5161 | 0.3333 | 0.7500 | 0.8478 | 0.4800 | 39.7950 | -0.1667 | True | 31 |

| 28 | 0.7218 | 0.5221 | 0.5143 | 0.3333 | 0.4444 | 0.9184 | 0.4000 | 23.9112 | -0.1564 | True | 17 |

| 29 | 0.7842 | 0.5437 | 0.4286 | 0.3333 | 0.4762 | 0.9091 | 0.5714 | 35.9522 | -0.0940 | True | 18 |

| 30 | 0.7030 | 0.4482 | 0.4571 | 0.0000 | 0.3333 | 0.9053 | 0.5455 | 24.5944 | -0.1752 | True | 12 |

## 2026-11-06 — Pending soft-membership GMM label recovery

Added `label_unlabeled_points_soft(model, X_train, train_df, percentage_threshold=50)` in its own cell in **Part1_GMM_Walkthrough.ipynb**. It returns `(augmented_df, soft_details)`, where details expose per-trial reason probabilities, component reason rates, and effective labeled support. Only originally visible labels estimate component rates: `rates = (R_known.T @ Y_known) / R_known.sum(axis=0)[:, None]`. Each trial then receives its own mixture `R @ rates`. These five reason estimates are independent and need not sum to one; they are not demonstrated calibrated probabilities.

Components with negligible known-label mass fall back to overall visible-training reason rates. Unknown trials are assigned binary labels independently at50%; all-zero estimates remain masked/unresolved. Original known labels and the input DataFrame are preserved. Recovered rows use `label_source="probability_prediction"`.

The GMM sweep now calls this soft helper rather than hard cluster mapping. It retains2–30 components, diagonal covariance/n_init5/max_iter300/reg_covar1e-4/seed42, kernel PCA50/gamma1/75, equal-weight classifiers on original scaled embeddings, and consistent nonempty evaluation/export. Runs retain soft probabilities/rates/support, and the metrics table adds minimum effective labeled support. Export still explicitly selects21 components and the same GMM filename, but rejects stale results not generated by the soft pipeline. Existing CSVs were not regenerated.

The real soft sweep is **pending and unexecuted**; prior hard-GMM metrics are historical, not soft results. Cleared pipeline/comparison/export outputs. JSON/syntax/routing checks and a hand-computable responsibility-matrix test passed, including weighted rates, per-trial mixing, zero-support fallback, multi-label independence, threshold/unresolved behavior, nonstandard indices, and preservation of known labels/input. No real-data fits or exports were run. The separate K-means notebook remains **byte-identical**. Run new helper cell13, pipeline14, comparison15; export17 only after reviewing/selecting the soft run.

## 2026-11-06 — Observed soft-membership GMM sweep

Verified saved user-run pipeline/comparison executions **28/29** in `Part1_GMM_Walkthrough.ipynb`. The sweep calls `label_unlabeled_points_soft`, estimates rates from original known-label responsibilities, and mixes them per trial; it does not call hard assignment helpers. Settings remain2–30 diagonal-GMM components, n_init5/max_iter300/reg_covar1e-4/GMM seed42, kernel PCA50/gamma1/75/seed42, reason threshold50%, equal-weight classifiers on original scaled embeddings, and shared nonempty prediction fallback. All29 mixtures converged in9–62 iterations. The soft sweep is now **observed**; this inspection did not edit notebooks, execute models, export predictions, or access Kaggle.

The soft sweep's **sample-wise and macro winner is7 components**, with no reported sample tie. Every soft-GMM setting is below the consistent supervised baseline (**sample0.8782, macro0.6990**).

| Metric | K-means21 reference | Hard GMM21 | Soft GMM21 | Soft GMM7 winner |
|---|---:|---:|---:|---:|
| Sample-wise F1 | **0.8880** | 0.8547 | 0.8205 | 0.8581 |
| Macro F1 | **0.7272** | 0.6763 | 0.6526 | 0.6727 |
| Safety F1 | **0.8571** | **0.8571** | **0.8571** | 0.7619 |
| Funding F1 | **0.3333** | 0.2857 | 0.2857 | 0.2857 |
| Efficacy F1 | **0.8667** | 0.7500 | 0.6429 | 0.7857 |
| Recruitment F1 | **0.9703** | **0.9703** | 0.9388 | **0.9703** |
| Other F1 | **0.6087** | 0.5185 | 0.5385 | 0.5600 |
| Percent unresolved | 46.7122% | 44.7481% | 40.0512% | 51.0675% |
| Delta versus baseline sample F1 | +0.0098 | -0.0235 | -0.0577 | -0.0201 |

Soft GMM7 converged in28 iterations; soft GMM21 in37. At fixed21 components, soft propagation lowers sample F1 by0.0342 and macro by0.0237 versus hard propagation: Safety/Funding are unchanged, Efficacy/Recruitment worsen, and Other improves0.0200. Selecting7 components raises soft sample F1 above the hard sweep's best by only0.0034 while macro is0.0036 lower, and that comparison changes component count. At fixed7, hard-to-soft sample/macro improve0.8521/0.6642 to0.8581/0.6727, with lower Safety but better Efficacy/Recruitment. Softness therefore has mixed effects rather than consistently repairing label recovery.

No alternative has better macro than the soft sample winner. Category-specialist alternatives show substantial tradeoffs:

| Components | Sample F1 | Macro F1 | Safety | Funding | Efficacy | Recruitment | Other | Unresolved% | Min effective support | EM iterations |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.7308 | 0.5033 | 0.6154 | **0.4000** | 0.3529 | 0.8403 | 0.3077 | 31.5969 | 23.3339 | 9 |
| 4 | 0.7350 | 0.4853 | 0.5333 | **0.4000** | 0.3529 | 0.8547 | 0.2857 | 26.7293 | 24.0000 | 25 |
| 19 | 0.8248 | 0.6121 | 0.5333 | 0.2857 | 0.6400 | 0.9346 | **0.6667** | 35.7814 | 0.0000 | 22 |

Minimum effective known-label support prints **0.0000 for every count from7 upward**, including the winner and21-component reference. This is a fractional sum of known-trial responsibilities, not a hard count. Rounded zero may mean exact zero or tiny positive mass; the saved output cannot distinguish these or establish how often the ≤1e-12 global-rate fallback was used. Component reason estimates can therefore be weakly supported even when EM converges. Conversely,3/4 components have much stronger minimum support but poor overall scores, so sparse support alone is not a complete explanation.

Conclusion: keep the K-means pipeline as the stronger current reference. Diagonal GMM plus responsibility-weighted label recovery does not beat baseline or improve overall category balance in this sweep. Changing density geometry and using soft memberships do not guarantee reason-aligned components; no cause or superior test performance is proved. All differences use rounded saved values, coverage is not label accuracy, and repeated selection on78 validation trials remains a limitation. No new configuration was proposed or applied here.


### Direct GMM predictor notebook

Added `Part1_GMM_Direct_Predictor.ipynb` as a separate method. It fits one GMM on all training embeddings, estimates five component reason rates from originally visible training labels using soft responsibilities, and predicts validation/test reason probabilities directly as responsibilities times component rates. It does not create pseudo-labels and refit a second classifier. The notebook sweeps 2–30 components and reports all F1 metrics, fallback rows, effective labeled support, and EM convergence. GMM submission output has its own filename. Notebook code is syntax checked but has not been run on the real data; no Kaggle upload was performed.


## Class-conditional generative GMM — pending run

Created `Part1_ClassConditional_GMM.ipynb` as a separate experiment. It fits a positive-density and negative-density Gaussian mixture for each reason using only originally labeled training examples, then predicts each reason directly with class-conditional likelihoods and empirical priors. It uses 50-component RBF Kernel PCA at gamma 1/75 for the embedding representation; Kernel PCA is fit on training embeddings only. Each class mixture uses diagonal covariance, at most 2 components, 5 starts, up to 300 EM iterations, covariance regularization 1e-4, and seed 42. Validation and test prediction use the same nonempty fallback. This experiment does not pseudo-label the unlabeled training rows and does not train logistic regression. The real-data run and export are pending; no new performance result is claimed. Notebook JSON and cell syntax were checked, with no real model fit.

### Class-conditional notebook completion and checks

Reviewed the existing unexecuted notebook from the preceding handoff and completed its safeguards without changing other experiments. The fit helper now exposes its GMM settings explicitly, caps components by class support, and rejects classes with fewer than two examples (the GMM library's minimum). Training uses only originally observed binary labels; actual positive supports are Safety16, Funding8, Efficacy17, Recruitment101, Other25. Kernel PCA/scaling may use all training embeddings, but labels for the1,014 masked rows do not fit either class density.

Posterior prediction uses stable `expit(log p(x|positive)+log prior_positive−log p(x|negative)−log prior_negative)` independently for each reason. Threshold0.5 and argmax fallback are shared between evaluation and the distinct optional export helper. The final export example is commented out, so running the notebook does not automatically write a submission. Support/component-count/convergence/iteration diagnostics and all assignment F1 metrics are included.

This is an **additional labeled-only generative-classifier comparison**, not a replacement for the assignment's required unsupervised label-recovery work in the existing K-means notebook. Historical validation references0.8782 (observed-only logistic baseline) and0.8880 (selected K-means augmentation) are context only; there are no class-conditional validation results yet.

Notebook JSON, unique IDs, and source syntax passed. A formula-only synthetic check verified Bayes posterior values, stable extreme log-joints, independent thresholding, and nonempty fallback using fake density scores—no GMM fitting. All cells remain unexecuted. Hash checks confirmed17 other existing notebook/CSV artifacts unchanged. No real-data model fit, export, upload, or new metrics occurred.

### Class-conditional runtime correction and actual validation run

The user encountered `NameError: expit is not defined` in validation cell16. The scoring helper referenced `expit`, but the import-cell edit had silently failed because its string replacement assumed a trailing newline. The earlier formula-only check supplied `expit` directly to its test namespace and therefore did not verify the notebook import path. Added the explicit `from scipy.special import expit` import in **cell2**, preserving the user's configuration and all other source.

Executed the actual notebook from imports through validation and diagnostics (cells2,4,5,7,9,11,13,14,16,18) in a fresh IPython context using the existing Cornell environment. All completed successfully in **1.67seconds** and real outputs/execution counts were saved. Config remains kernel PCA50/RBF/gamma1/75; positive/negative class densities use up to2 diagonal components,5 initializations,max300,reg1e-4,seed42, trained only on originally labeled rows. All **10 GMMs converged**, taking **2–11 EM iterations**, without convergence warnings. Four validation trials used the nonempty argmax fallback.

| Actual validation metric | Result |
|---|---:|
| Sample-wise F1 | 0.5982905983 |
| Macro F1 | 0.3435006836 |
| Safety F1 | 0.4615384615 |
| Funding F1 | 0.0000000000 |
| Efficacy F1 | 0.3703703704 |
| Recruitment F1 | 0.7476635514 |
| Other F1 | 0.1379310345 |

This verifies successful execution, not good predictive performance; the direct class-conditional model does not improve on the historical validation references. No hyperparameter tuning occurred. The optional export remains commented and was not executed; no CSV or upload was generated, and other notebooks/existing root CSVs were hash-checked unchanged. For the user's existing kernel, rerun importcell2, validationcell16, and diagnosticscell18.

## 2026-11-06 — Integrated teammates’ main branch

Fast-forwarded `prith_trial` from `c8db1cd` to main `0233983c491ae36f96516085e82d15b8a47c7eaf` after backing up all12 local modified/untracked files to `/private/tmp/applied-ml-before-main-zmi12vhm` and stashing them. Reapplied local work; retained stash `243c6e0c8fb66d9af5b999c5553dfef084b93968` for recovery. All personal experiment notebooks, submissions, and figures matched pre-pull hashes before appending this entry.

The starter now includes teammates’ full Part1 and Part2 implementations, with the local Cornell kernel metadata preserved. The old local path-error output was omitted from the integrated starter; its original raw content remains in the backup and stash. Resolved the requirements overlap to `matplotlib>=3.10,<4`, which also meets the previous local requirement. No experiments were rerun.

See `TEAM_COMPARISON.md` for method/result comparison and recommended follow-up work.

### K-means clone — full cluster-count sweep restored

`Part1_KMeans_Experiments.ipynb` now sweeps 2–30 clusters, with kernel PCA50/RBF gamma1/75, recovery threshold50%, K-means seed42/n_init100, and equal-weight logistic regression C1 unchanged. All validation F1 columns, unresolved percentage, baseline difference, and tied-winner reporting remain. The clone exports an explicitly selected run (`KMEANS_EXPORT_CLUSTERS = 21`) to `part1_kmeans_experiments_submission.csv`, preserving the reference CSV. Modified sweep/display/export outputs were cleared. JSON and Python syntax checks passed; no sweep or CSV generation was executed. The original `Part1_Clustering_Walkthrough.ipynb` is byte-unchanged.

### K-means clone — classifier regularization experiment

Added separate unexecuted cells in `Part1_KMeans_Experiments.ipynb` to reuse the existing recovered labels from `pipeline_results[14]` and compare logistic regression C values 0.01, 0.1, 1.0, and 10.0. Smaller C means stronger L2 regularization; C1 is the control. Clustering and label recovery are not repeated or changed. Each augmented model is compared with an observed-label-only baseline at the same C. Results retain all five per-reason F1 scores, sample/macro F1, the fixed unresolved percentage, and baseline delta. Sample and macro winners are reported with ties; no model is automatically chosen for export. The original cluster sweep and explicit cluster-sweep export behavior remain. JSON and syntax checks passed; no classifiers were fitted, no metrics were newly observed, and no CSV was generated. The original K-means notebook is byte-unchanged.

### Observed clone experiment — fixed K14 labels, logistic regularization

Read saved `Part1_KMeans_Experiments.ipynb` outputs (executions44/45); no new models were run for this analysis. The experiment reuses the same recovered labels from K14. KernelPCA50/RBF gamma1/75, recovery threshold50%, K-means seed42/n_init100, equal label weighting, and classifier seed42 remain unchanged. The original K-means notebook is byte-unchanged.

| C | Sample F1 | Macro F1 | Safety | Funding | Efficacy | Recruitment | Other | Unresolved % | Same-C baseline sample F1 | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | .8175 | .5887 | .5143 | .2308 | .7000 | .9691 | .5294 | 44.4065 | .7201 | +.0974 |
| 0.1 | .8739 | .6477 | .6667 | .1818 | .8485 | .9899 | .5517 | 44.4065 | .8534 | +.0205 |
| 1 | **.8949** | **.7221** | .7826 | .3333 | .9333 | .9899 | .5714 | 44.4065 | .8782 | +.0167 |
| 10 | .8782 | .6867 | .8000 | .2857 | .8966 | .9899 | .4615 | 44.4065 | .8782 | .0000 |

C1 uniquely maximizes both sample-wise and macro F1 among these four values, and has the best Funding, Efficacy, and Other scores. C10 raises Safety slightly (.7826→.8000) but weakens those three reasons and both aggregate metrics. The larger augmentation delta at C0.01 reflects its much weaker same-C supervised baseline, not a better final classifier. Unresolved coverage is identical because labels were held fixed. Retain C1 for this tested configuration; the conclusion is limited to these settings and the78-trial validation set. No notebook parameters, sources, exports, or CSV files were changed.

### Observed clone experiment — nearest-neighbor recovery in kernel PCA space

Added `recover_labels_with_neighbors()` and separate experiment/comparison cells17–19 in the clone, with a NearestNeighbors import in cell1. The helper searches only originally known-label trials, using the existing 50-dimensional kernel PCA embeddings for distances. It combines raw observed cluster rates and uniform3-neighbor vote rates equally, then applies the unchanged50% threshold. A cluster with no originally labeled members uses neighbor votes alone. Observed labels are preserved; empty recovered sets remain -100/unresolved. An audit DataFrame stores neighbor IDs/distances and component, neighbor, and blended reason scores. The final classifier remains equal-weight logistic regression C1 on the original scaled150-dimensional embeddings. Prior cluster-count and regularization experiments and the explicit cluster-only export remain intact. JSON/AST and a small synthetic example passed (different labels within one cluster, unknown-cluster fallback, positional alignment, empty sets unresolved); no project models were fitted during implementation.

The user then executed the new cells. Read fresh saved outputs (executions64/65), verifying fixedK14,3uniform neighbors,50/50 blend,50% threshold,C1, and kernelPCA50/gamma1/75. Results:

| Metric | Cluster only | Cluster + neighbors |
|---|---:|---:|
| Sample-wise F1 | **.8949** | .8013 |
| Macro F1 | **.7221** | .6470 |
| Safety | .7826 | **.8235** |
| Funding | .3333 | **.4000** |
| Efficacy | **.9333** | .5714 |
| Recruitment | **.9899** | .9400 |
| Other | **.5714** | .5000 |
| Unresolved training trials | 44.4065% | 22.6302% |
| Recovered Funding positives | 0 | 9 |
| Observed-only C1 baseline sample F1 | .8782 | .8782 |
| Delta vs baseline | +.0167 | -.0769 |

Neighbor votes improve Safety and Funding and recover more rows, but Efficacy drops sharply and both overall F1 measures deteriorate. Greater coverage is not evidence of more accurate recovered labels. The score changes reflect this specific neighbor space/blending/threshold configuration; they do not isolate a general effect of KNN. Teammates' hybrid uses different neighbor space, thresholds, smoothing, regularization, and pseudo-label weights, so it is not a controlled comparator for this single addition. No parameter changes, project model fits, or exports were made during analysis. The original K-means notebook remains byte-unchanged.

### Planned clone experiment — neighbor search in original scaled150 dimensions

Updated only clone cells17–19 to separate clustering and neighbor representations. `recover_labels_with_neighbors(..., X_neighbors=None)` keeps cluster assignments in `X_cluster`; neighbor fit/query uses the supplied `X_neighbors` and only originally known labels. Omitting the new argument retains the previous shared-space behavior. Both arrays must have the same row order and number of trials.

The experiment now passes `X_neighbors=X_train` (original150 standardized embeddings), while K-means still uses the existing kernelPCA50 embedding. It prints both dimensions to verify routing. K14, three uniform neighbors, equal cluster/neighbor score blend,50% threshold, raw cluster rates, LR C1/equal label weights, seed42/n_init100, and kernelPCA gamma1/75 remain fixed. No-known-label clusters use neighbor votes alone; empty predicted sets remain unresolved. Original labels are preserved. The explicit reference export is unchanged.

The preceding kernelPCA-neighbor result remains recorded above: sampleF1 .8013, macroF1 .6470, Safety .8235, Funding .4000, Efficacy .5714, Recruitment .9400, Other .5000, unresolved22.6302%, and nine recovered Funding positives. Cleared those saved outputs in the changed cells so they are not attributed to the new raw-neighbor configuration. The raw-neighbor experiment is pending; no new project models, validation metrics, or CSVs were generated.

Notebook JSON/AST checks and a focused synthetic test with separate50- and150-dimensional representations passed: the clustering model received only50 features, neighbor geometry changed the recovered labels, default behavior stayed compatible, and observed labels/row alignment were preserved. `Part1_Clustering_Walkthrough.ipynb` remains byte-unchanged.

### Observed clone experiment — raw150-dimensional neighbor distances

Read fresh saved clone outputs (executions48/49). The pipeline prints clustering dimensions50 and neighbor dimensions150, and passes `X_neighbors=X_train` while K-means assignment stays on `X_train_cluster`. Fixed K14,3uniform originally labeled neighbors, equal cluster/neighbor score blending,50% threshold, LR C1/equal label weights, seed42/n_init100, kernelPCA50/RBF gamma1/75 remain unchanged. Zero-known-label clusters still use neighbor votes alone; no new smoothing or label-weighting rule was introduced.

| Metric | Cluster only | Prior kernelPCA-neighbor hybrid | Raw150-neighbor hybrid |
|---|---:|---:|---:|
| Sample-wise F1 | **.8949** | .8013 | .8162 |
| Macro F1 | **.7221** | .6470 | .6395 |
| Safety | .7826 | **.8235** | **.8235** |
| Funding | .3333 | **.4000** | **.4000** |
| Efficacy | **.9333** | .5714 | .6364 |
| Recruitment | **.9899** | .9400 | .9091 |
| Other | **.5714** | .5000 | .4286 |
| Unresolved % | 44.4065 | 22.6302 | 12.1264 |
| Recovered Funding positives | 0 | 9 | 5 |
| Baseline sample-wise F1 | .8782 | .8782 | .8782 |
| Delta vs baseline | +.0167 | -.0769 | -.0620 |

Raw-space neighbors improve sample F1 and Efficacy relative to the previous kernelPCA-neighbor experiment, but macro F1, Recruitment, and Other worsen; Safety and Funding stay unchanged. Both neighbor versions remain substantially below the cluster-only model and observed-label-only baseline overall. The raw-neighbor version recovers more labels but this does not establish greater label accuracy. Five Funding pseudo-positives replace the prior nine while validation Funding F1 stays .4. The result rejects this particular50/50 blending configuration as a clear improvement; it does not establish that neighbor methods in general cannot help. No parameters were selected or changed during this analysis. No notebook/source edits, project model fits, CSV writes, or exports were made. The original K-means notebook remains byte-unchanged.

### 2026-11-06 — Decision: selective reason-specific recovery experiment

The user approved isolating KNN label recovery to Safety and Funding after the global cluster/KNN blend improved these two reasons while harming Efficacy, Recruitment, and Other. This decision is based on the observed behavior, not rarity alone: originally labeled Safety and Efficacy have similarly small positive counts (16 and17), yet their observed responses to blending differed. This is a reversible experiment, not an established improvement.

Added unexecuted clone cells20–22: a small five-estimator container, `fit_selective_reason_model()`, one configured fit/evaluation cell, and comparison/coverage/lineage output. The configuration reuses K14's fitted cluster-only model; three uniformly weighted neighbors are drawn only from the157 originally labeled trials in the original150-dimensional standardized embedding space. For Safety and Funding, a50% independent vote threshold assigns both binary positives and negatives to all1014 originally masked trials. Those two logistic regressions are refitted on all original observations plus these recovered labels with C1, liblinear,max_iter500,seed42, and equal row weights. Observed labels stay unchanged.

Efficacy, Recruitment, and Other reuse the exact fitted binary estimators from the K14 cluster-only reference. Their classifiers are not refitted on newly recovered Safety/Funding rows. A per-reason coverage table records observed/recovered/training/unresolved counts, recovered positives/negatives, and unresolved percentages; a per-trial lineage DataFrame records each reason's actual training labels and label source. There is no misleading single fully-labeled-dataframe coverage value because the five training masks now differ.

The existing final prediction policy still adds the most probable reason when all five raw predictions are zero. Although the three protected estimators and their raw probabilities/predictions are identical, a new Safety/Funding positive can prevent this row-level fallback from adding a protected reason. Their final postprocessed class scores can therefore change. The new comparison explicitly audits protected raw and postprocessed prediction changes; it does not promise unchanged final metrics.

The original kernelPCA50/gamma1/75 clustering, K14 labels, seed42/n_init100, global-blend experiments and their saved outputs, regularization results, and reference export remain intact. This selective model is not automatically chosen for export. New experiment results are pending: no project models were fitted, no new validation metrics are claimed, and no CSV was generated by this implementation.

For future revisiting, alternatives include selectively blending cluster and KNN scores for only Safety/Funding, separate reason thresholds, or whole-row augmentation after assessing confidence. None is implemented here. The current experiment deliberately uses KNN-only scores for its two selected reasons.

JSON/AST checks and a focused synthetic test passed: exactly two classifiers were fitted; original true labels and both input dataframes were preserved; only selected reasons received all recovered binary labels; protected estimator identities, coefficients, raw probabilities, and raw predictions were unchanged; and per-reason coverage respected the original reference masks. `Part1_Clustering_Walkthrough.ipynb` remains byte-identical (SHA25630d347b0c04b42f686d951201dde12025da3d7d4d4fcba54df430c9c74902ee6).

### Observed selective recovery — KNN-only Safety/Funding

Read saved clone outputs (executions71/72), confirming K14, three uniform originally observed neighbors in the original scaled150-dimensional embedding space,50% independent threshold,C1/equal weights, and unchanged kernelPCA50/gamma1/75 clustering. Safety and Funding receive KNN-only recovered positives and negatives; the other three estimators are the exact fitted cluster-only reference objects.

| Metric | Cluster-only K14 | Selective KNN recovery |
|---|---:|---:|
| Sample-wise F1 | **.8949** | .8910 |
| Macro F1 | **.7221** | .7141 |
| Safety | **.7826** | .7059 |
| Funding | .3333 | **.4000** |
| Efficacy (final postprocessing) | **.9333** | .9032 |
| Recruitment | .9899 | .9899 |
| Other | .5714 | .5714 |
| Observed-only C1 baseline sample F1 | .8782 | .8782 |
| Delta vs baseline | +.0167 | +.0128 |

Per-reason coverage differs as intended. Safety/Funding each use1171 training rows:157 observed plus1014 recovered, zero unresolved. Safety recovers75 positives and939 negatives; Funding recovers16 positives and998 negatives. Efficacy/Recruitment/Other each retain651 training rows:157 observed plus494 recovered, with520 unresolved (44.4065%). Their recovered positive/negative counts remain16/478,458/36, and20/474, respectively. Originally observed labels remain intact by the helper's input preservation checks and masked-only assignment.

The saved preservation audit confirms all three protected estimators are the same objects and have zero raw prediction changes. Final nonempty fallback changes one Efficacy prediction, causing its final F1 to change despite an identical fitted classifier. Recruitment and Other have no final prediction changes. Coverage is therefore reported per reason, not as one global fully-labeled-row percentage.

The selective model stays above the observed-only baseline and avoids most of the global blend's damage, but it does not improve overall sample/macro F1 over the cluster-only reference. Funding improves, while Safety worsens; the fallback interaction modestly worsens Efficacy. This does not support the initial assumption that KNN-only recovery helps both selected reasons. Their original positive supports are only16 Safety and8 Funding observations; the78-trial validation comparison remains limited. The prior global raw-neighbor blend scored.8162/.6395 overall, so isolating the affected estimators prevented much of its damage, without establishing a new winner. The decision is tested but remains reversible; no model was automatically selected for export.

No notebook sources, parameters, project fits, or CSVs were changed during this analysis. Only this log was updated; the original K-means notebook remains byte-unchanged.

### 2026-11-06 — Decision revision: Funding-only KNN recovery

The user approved restoring cluster-only Safety and testing KNN recovery for Funding alone. The preceding two-reason run improved Funding F1 from.3333 to.4000 but reduced Safety from.7826 to.7059; overall sample/macro F1 were.8910/.7141 versus the cluster reference.8949/.7221. That tested alternative remains recorded above.

Clone cells20–22 now set `KNN_RECOVERY_REASONS=["Funding"]` and protect Safety, Efficacy, Recruitment, and Other. The helper is generic and its default now also selects Funding only. The four protected classifiers reuse exact fitted K14 reference estimator objects, with no new training rows or refits. Funding alone uses the157 original labels plus1014 binary0/1 labels recovered from three uniform originally labeled neighbors in the original150-dimensional standardized space, at50% threshold,C1,equal weights,seed42. Its prior unchanged vote rule produced16 recovered Funding positives, but the new run is pending and no newly observed results are claimed.

Per-reason coverage and the raw-vs-final fallback audit remain. Funding should have1171 training rows; the four protected reference models retain651 rows and520 unresolved under the current K14 reference. The nonempty row fallback can still change some final protected labels despite identical raw estimators. Prior global-neighbor experiments, regularization sweep, full cluster sweep, and reference export remain unchanged. Cleared only the selective helper/run/comparison outputs to prevent old Safety/Funding results appearing under Funding-only configuration.

JSON/AST checks and a focused synthetic test passed: exactly one new classifier fitted, original Funding observations preserved, four protected estimator identities/raw probabilities/raw predictions preserved, and per-reason training masks retained. No actual project model fits, CSV writes, or exports were executed. The original K-means notebook remains byte-unchanged. Funding-only is a pending reversible decision.

### Observed clone experiment — Funding-only KNN recovery

Read saved selective outputs (executions94/95). Verified Funding alone uses three uniform original labeled neighbors in scaled150 space,50% independent vote threshold,C1/equal row weights, with all1014 originally masked Funding values recovered. Safety, Efficacy, Recruitment, and Other reuse the exact K14 cluster-only classifiers. KernelPCA50/gamma1/75, K-means seed42/n_init100, and all other settings remain fixed.

| Metric | Cluster-only K14 | Prior Safety/Funding KNN | Funding-only KNN |
|---|---:|---:|---:|
| Sample-wise F1 | .8949 | .8910 | **.8970** |
| Macro F1 | .7221 | .7141 | **.7355** |
| Safety | .7826 | .7059 | .7826 |
| Funding | .3333 | .4000 | **.4000** |
| Efficacy | .9333 | .9032 | .9333 |
| Recruitment | .9899 | .9899 | .9899 |
| Other | .5714 | .5714 | .5714 |
| Observed-only C1 baseline sample F1 | .8782 | .8782 | .8782 |
| Delta vs baseline | +.0167 | +.0128 | **+.0188** |

Funding-only improves reported sample F1 by approximately.0021 and macro by.0134 over the pure-cluster reference while all other class F1 values remain unchanged. The saved preservation audit confirms all four protected estimators are identical, with zero raw prediction changes and zero final fallback changes, including Safety. Unlike the preceding two-reason run, this run has no protected Efficacy fallback change. This is a measured validation improvement, not proof of wider generalization; the validation set is small and Funding contains few positives.

Per-reason coverage: Funding trains on1171 rows (157 observed plus1014 recovered), including16 recovered positives and998 recovered negatives, with zero unresolved Funding values. Each protected reason retains651 training rows (157 observed plus494 recovered) and520 unresolved values (44.4065%). Protected recovered positive/negative counts remain Safety0/494, Efficacy16/478, Recruitment458/36, and Other20/474. Originally observed labels are preserved by masked-only recovery and the helper's preservation check. One global unresolved percentage would obscure these different masks.

The Funding-only decision is now tested and remains reversible. The model was not automatically chosen for export. During this analysis only this log was updated: no notebook sources, settings, fits, CSVs, or exports changed. The original K-means notebook remains byte-unchanged.

### Funding-only hybrid Kaggle export

At the user's request, appended a separate export cell to `Part1_KMeans_Experiments.ipynb` and generated `part1_kmeans_funding_knn_submission.csv`. A fresh execution fitted the exact saved Funding-only configuration with K14, kernel PCA50/gamma1/75 for clustering, and three original labeled neighbors in scaled150 space for Funding. Its validation metrics reproduced sample-wise F1 .8970 and macro F1 .7355. The existing K21 cluster-only submission and original notebook were not changed.

The new CSV has312 rows in test-ID order, exactly `ID,Predicted`, unique IDs, allowed reason names, and no null or empty predictions. Ten all-zero raw predictions received the existing highest-probability reason fallback. It differs from the earlier K21 cluster-only export on43 test rows. Kaggle performance has not yet been observed.

The user subsequently reported a Kaggle public score of **0.82264** for this Funding-only hybrid and judged it worse. The earlier personal K-means public score is recorded only as approximately0.82, so its exact ordering against0.82264 is not established here. The validation improvement (.8949 to.8970 sample F1 versus the K14 reference) did not provide a clear public-score gain.

Read-only comparison of the exported hybrid against the earlier **K21** cluster-only CSV shows43 changed test rows. Funding positives fall from23 to7 (one added,17 removed). Other test prediction counts also differ: Safety76 to76, Efficacy58 to57, Recruitment171 to169, and Other84 to91. Because the files use different cluster counts as well as different Funding recovery, these changes cannot isolate the effect of Funding-only KNN. No models or CSVs were modified for this analysis.

### Funding-only hybrid at K=21

After the user reported public Kaggle score **0.82264** for the K14 Funding-only hybrid, they requested the same hybrid with K21. An isolated cell in `Part1_KMeans_Experiments.ipynb` reuses the K21 cluster-only run, retrains Funding alone with the same three originally labeled neighbors in scaled150 space,50% vote threshold,C1/equal weights,seed42, and exports `part1_kmeans_funding_knn_k21_submission.csv`. Kernel PCA50/gamma1/75, K-means n_init100/seed42, and the nonempty prediction fallback are unchanged. The K14 hybrid cell and prior CSVs remain intact.

Fresh execution yielded validation sample-wise F1 **.890171**, macro F1 **.740560**; Safety **.857143**, Funding **.400000**, Efficacy **.866667**, Recruitment **.970297**, Other **.608696**. Against K21 cluster-only (.8880 sample/.7272 macro), this is a small validation improvement. Against K14 Funding-only (.8970/.7355), sample F1 decreases while macro F1 increases. Funding trains on1171 rows (157 observed,1014 recovered;16 recovered positives,998 negatives) with zero unresolved. Each protected reason retains624 training rows (157 observed,467 cluster recovered) and547 unresolved (46.712%).

The new CSV has312 unique test IDs in test order, exactly `ID,Predicted`, allowed reason names, and no null/empty predictions. Nine all-zero raw rows received the existing fallback. It changes18 rows versus the K21 cluster-only export and27 versus the K14 Funding-only export. Predicted reason counts for K21 cluster-only → K21 hybrid are Safety76→76, Funding23→7, Efficacy58→59, Recruitment171→171, Other84→84. Kaggle score for this new K21 file has not yet been observed.

The user subsequently reported a public Kaggle score of **approximately .83** for the K21 Funding-only file, calling it their highest personal score so far. This is a rounded user report, not an independently queried exact score. It is above the exact user-reported **.82264** for K14 Funding-only, while the earlier personal K-means score remains only approximately .82. The teammate hybrid's **.87927** is a separate score recorded in project notes, so the K21 hybrid has not surpassed it on the available public results. No model or export was changed after this score report.

### K21 selective smoothed cluster and KNN blend

The user requested a controlled test blending teammate-style smoothed cluster rates and three original-label neighbor votes equally for Safety, Funding, and Other. Clone cells26–27 implement it with smoothing strength2 toward the observed-label prior and a50% recovery threshold. Efficacy and Recruitment reuse the exact K21 cluster-only estimators. Kernel PCA50/gamma1/75, KMeans21/n_init100/seed42, scaled150 neighbor space, C1/equal-weight logistic regression, and nonempty fallback remain fixed. Only the157 originally labeled rows estimate cluster rates and neighbor votes; their true labels are preserved.

A fresh **single K21** run gave validation sample-wise F1 **.850427**, macro F1 **.616744**; Safety **.615385**, Funding **.400000**, Efficacy **.764706**, Recruitment **.970297**, Other **.333333**. This is below K21 Funding-only (.890171/.740560) and the teammate selected hybrid (.886752/.779234) on both aggregate metrics. Recovered positives among1014 originally masked rows were Safety44, Funding5, Other32; each selected reason has zero unresolved rows. Efficacy and Recruitment each retain547 unresolved rows (46.712%). Their estimator objects and raw validation predictions are identical to K21 cluster-only; final nonempty fallback changes four Efficacy predictions and zero Recruitment predictions.

Exported `part1_kmeans_blend_safety_funding_other_k21_submission.csv` with312 unique test IDs in order, exact `ID,Predicted` columns, allowed reason names, and no null/empty predictions;41 raw all-zero rows used the fallback. It differs on91 rows from K21 cluster-only and88 from K21 Funding-only. Test predicted positive counts are Safety42, Funding5, Efficacy75, Recruitment177, Other25. This is a new experiment only; old notebooks and submission CSVs were not overwritten. Kaggle score has not been observed.

**Correction to selective blend prediction rule.** The user clarified that blending must affect only Safety, Funding, and Other, including final postprocessing. The first exported version above used the shared global nonempty fallback, which changed four final Efficacy validation predictions and some protected test predictions even though their fitted estimators were reused. Its metrics and CSV counts above describe that superseded version.

The corrected blend-only adapter now copies the K21 cluster-only model's **final** Efficacy and Recruitment predictions on every row, combines them with the blend model's raw Safety/Funding/Other predictions, and, only if the row is still empty, chooses the highest-probability reason among Safety/Funding/Other. The generic fallback for every other experiment is untouched. Assertions verified the protected fitted estimator identities and **zero Efficacy/Recruitment final changes on both validation and test**. All final rows remain nonempty.

Fresh corrected validation sample-wise F1 is **.850427**, macro F1 **.600726**; Safety **.500000**, Funding **.333333**, Efficacy **.866667**, Recruitment **.970297**, Other **.333333**. The corrected CSV replaced only this experiment's distinct `part1_kmeans_blend_safety_funding_other_k21_submission.csv`:312 test IDs, exact two columns, no null/empty predictions,37 selected-reason fallbacks. It differs on77 test rows from K21 cluster-only and76 from K21 Funding-only. Corrected predicted positive counts are Safety50, Funding12, Efficacy58, Recruitment171, Other33. No Kaggle result is claimed for this corrected file.

### Quarter-weight pseudo labels in the corrected K21 selective blend

The user explicitly requested quarter weight for pseudo labels in this experiment, superseding their earlier preference for equal weights **for this variant only**. The helper now accepts `pseudo_weight` (default1, preserving the prior unweighted run). A separate notebook cell trains Safety/Funding/Other binary logistic regressions with weight1 on157 observed rows and weight0.25 on1014 recovered rows. All recovery labels, cluster/neighbor settings, C1, and Efficacy/Recruitment protected estimators and final predictions remain unchanged.

Fresh validation sample-wise F1 was **.880342** and macro F1 **.704059**, compared with the corrected equal-weight blend's .850427/.600726. Per-reason F1: Safety **.750000**, Funding **.400000**, Efficacy **.866667**, Recruitment **.970297**, Other **.533333**. This improves over the equal-weight blend but remains below K21 Funding-only (.890171/.740560) on both aggregate metrics. Recovered positives remain Safety44, Funding5, Other32, with zero unresolved for those reasons. Efficacy/Recruitment each retain547 unresolved rows (46.712%); assertions confirm their estimator identities and zero raw/final validation and final test changes.

Exported separate `part1_kmeans_blend_sfo_k21_weight025_submission.csv`, verified312 exact test IDs, `ID,Predicted`, allowed labels, and no null/empty values. There were33 selected-reason fallback trials. Twelve test rows differ from the unweighted corrected blend. Test positive counts are Safety50, Funding11, Efficacy58, Recruitment171, Other39. Earlier notebooks/CSVs remain untouched; no Kaggle score for this variant has been observed.

### Quarter-weight K21 selective blend with 30% recovery threshold

The user requested lowering the selected-reason recovery threshold. A separate clone cell changes only Safety/Funding/Other's smoothed-cluster and three-neighbor blended-score threshold from50% to30%, retaining pseudo-label weight0.25, smoothing strength2, 50/50 blend, C1, K21 on kernel PCA50/gamma1/75, and scaled150 neighbors. Efficacy/Recruitment keep exact K21 cluster-only estimators and final predictions. Originally observed labels are unchanged.

Fresh validation sample-wise F1 was **.893162**, macro F1 **.775236**, versus threshold50 weighted .880342/.704059 and K21 Funding-only .890171/.740560. Per-reason F1: Safety **.705882**, Funding **.666667**, Efficacy **.866667**, Recruitment **.970297**, Other **.666667**. Safety fell from .750000 at threshold50 while Funding and Other improved. Of1014 originally masked rows, recovered positives are Safety75, Funding16, Other214; their remaining rows are recovered negatives, leaving zero unresolved for those reasons. Efficacy and Recruitment retain547 unresolved each (46.712%). Assertions confirmed zero protected final prediction changes on validation and test and preserved original labels.

Distinct CSV `part1_kmeans_blend_sfo_k21_weight025_threshold30_submission.csv` has312 exact test IDs, `ID,Predicted`, allowed reason names, and no null/empty predictions. Fifteen rows used the selected-reason fallback. It changes54 test rows versus the threshold50 weighted CSV. Predicted test positive counts at threshold50→30 are Safety50→49, Funding11→6, Efficacy58→58, Recruitment171→171, Other39→85. No Kaggle score has yet been observed; earlier cells and CSVs remain intact.

### C=0.1 regularization on selected reasons

At the user's request, a separate K21 threshold30/quarter-weight blend cell changes only logistic regression C from1 to0.1 for Safety, Funding, and Other. The recovered binary labels, weights, smoothed cluster/neighbor scores, K21/kernel PCA setup, and protected Efficacy/Recruitment models and final decisions stay fixed. Assertions confirm the recovered label matrix exactly matches the C1 experiment and the157 original labels are unchanged.

Fresh validation sample-wise F1 increased from **.893162 to .914530**, and macro F1 from **.775236 to .822504**. Per-reason F1 for C0.1: Safety **.888889**, Funding **.666667**, Efficacy **.866667**, Recruitment **.970297**, Other **.720000**. Safety and Other improved; Funding and protected classes were unchanged. Recovered positives remain Safety75, Funding16, Other214 among1014 masked rows; each selected reason has no unresolved labels. Efficacy/Recruitment each retain547 unresolved (46.712%); their estimator identities and final validation/test predictions are unchanged.

Separate `part1_kmeans_blend_sfo_k21_weight025_threshold30_c01_submission.csv` was exported and verified to contain312 exact test IDs, `ID,Predicted`, allowed reason names, and no null/empty values. Thirteen test rows used selected-reason fallback. It differs on25 rows from the C1 threshold30 file; predicted test positive counts change Safety49→47, Funding6→6, Other85→91, with Efficacy58 and Recruitment171 unchanged. Kaggle score for the C0.1 variant is not yet observed. Earlier cells and CSVs remain intact.

The user later reported exact public Kaggle scores for these two submissions: **0.84508** for C1 and **0.83974** for C0.1. Thus stronger regularization improved validation F1 (.893162→.914530 sample-wise) but reduced this reported public score by **0.00534**. The C1 threshold30 quarter-weight blend is the highest **personal** public score reported so far; the teammate hybrid's separately recorded score is **0.87927**. These are public-subset results and do not isolate why validation and public ordering differ.

Read-only comparison of the two saved CSVs confirms25 changed test rows. C1→C0.1 positive counts and changes: Safety49→47 (one added, three removed); Funding6→6 (no changes); Other85→91 (14 added, eight removed); Efficacy58→58 and Recruitment171→171 (no changes). No model or CSV was changed for this score analysis.

### K14 versus K21 for the public-best C1 blend

The user requested the same C1/quarter-weight/threshold30 selective Safety/Funding/Other blend at K14. A separate clone cell uses the K14 reference from the shared sweep; kernel PCA50/gamma1/75, smoothing strength2, equal cluster/three-neighbor blend, raw scaled150 neighbor space, n_init100/seed42, and selected-only fallback remain fixed. Efficacy/Recruitment reuse exact K14 cluster-only estimators and final decisions. Original observed labels are preserved.

Fresh K14 validation sample-wise F1 was **.910256**, macro F1 **.792490**, versus K21 C1 .893162/.775236. Per-reason K14 F1: Safety **.705882**, Funding **.666667**, Efficacy **.933333**, Recruitment **.989899**, Other **.666667**. Selected recovered positives among1014 masked rows: Safety75, Funding16, Other216; all selected reasons have zero unresolved. Efficacy/Recruitment each retain520 unresolved (44.40649%) and their protected final validation/test predictions exactly match the K14 reference.

Exported separate `part1_kmeans_blend_sfo_k14_weight025_threshold30_c1_submission.csv`, verified312 exact test IDs, `ID,Predicted`, allowed reason names, and no null/empty predictions. Thirteen rows used selected-only fallback. It differs on15 test rows from the K21 C1 threshold30 export. Test positive counts K21→K14: Safety49→49, Funding6→6, Efficacy58→57, Recruitment171→169, Other85→83. The existing K21 C1 Kaggle score is user-reported **.84508**; no K14 public score has yet been observed. Previous notebooks and CSVs remain intact.

### K14 Safety-only C=0.1 regularization

The user observed that Safety trails the teammate model, so a new isolated notebook cell changes **only Safety's logistic regression C from1 to0.1** in the K14 quarter-weight/threshold30 selective blend. It reuses the same157 observed and1014 recovered Safety labels (75 recovered positives,939 negatives) with weights1 and0.25. Funding, Efficacy, Recruitment, and Other reuse their exact fitted K14 C1 estimator objects. A Safety-only postprocessing adapter preserves those four **final** reference predictions; an otherwise empty row receives Safety, keeping predictions nonempty. No submission CSV was requested or generated.

Fresh validation sample-wise F1 increased **.910256→.923077** and macro F1 **.792490→.829091**. Safety F1 increased **.705882→.888889**; Funding **.666667**, Efficacy **.933333**, Recruitment **.989899**, and Other **.666667** are unchanged. The audit confirms zero raw validation and final validation/test changes for all four fixed reasons. Safety final predictions differ on3 validation and3 test trials. The Safety-only fallback applied to0 validation and8 test trials. This is a validation result; public Kaggle performance has not been measured.

### Separate Efficacy/Recruitment cluster argmax fallback

Starting from the Safety-only C0.1 K14 model above, a separate notebook cell fills **only the520 Efficacy/Recruitment rows** left unresolved by the K14 cluster mapping. It computes unsmoothed five-reason rates from originally observed members in each K14 cluster, using the global observed-label prior for a cluster with no observed members, and assigns one argmax fallback reason in fixed Safety/Funding/Efficacy/Recruitment/Other order. Efficacy and Recruitment each become binary1 exactly when selected as that reason. The157 original and494 previously cluster-recovered E/R labels remain unchanged. Two C1/equal-weight E/R logistic regressions are refit on all1171 rows. Safety, Funding, and Other reuse their exact fitted models and final predictions through an experiment-specific nonempty fallback restricted to E/R.

Fresh validation sample-wise F1 **fell .923077→.790598** and macro F1 **.829091→.636400**. Efficacy F1 fell **.933333→.133333**; Recruitment fell **.989899→.826446**. Safety .888889, Funding .666667, and Other .666667 remained unchanged, with zero raw/final validation and final test changes for all three. The Efficacy validation confusion is TP1/FP0/FN13; Recruitment TP50/FP21/FN0.

All520 fallback rows chose **Recruitment** (there is one cluster with no originally observed member). The resulting full E/R training labels contain Efficacy33 positives/1138 negatives and Recruitment1079 positives/92 negatives; neither has unresolved labels. This strong Recruitment skew explains the poor fitted balance as a plausible mechanism, but the validation result alone does not prove a general test effect. The E/R-only nonempty fallback fired on0 validation and2 test rows. No Kaggle CSV was generated or submission made; prior notebooks, cells, and CSVs remain intact.

### Separate Efficacy/Recruitment 3NN fallback

The user requested nearest neighbors for the same E/R unresolved rows. A new notebook cell starts from the K14 Safety-only C0.1 reference and changes only E/R recovery on the520 rows where the original K14 cluster mapping was unresolved. Three uniform nearest neighbors are drawn from the157 originally labeled trials in the original scaled150 embedding space. Efficacy and Recruitment are assigned independently at vote rate≥0.30; the157 observed and494 previously cluster-recovered E/R labels remain unchanged. Only the two E/R C1/equal-weight logistic regressions are refit on all1171 trials. Safety, Funding, and Other keep their exact fitted objects and final decisions via the E/R-only nonempty fallback adapter.

Fresh validation sample-wise F1 was **.873504** and macro F1 **.781101**. This exceeds the cluster-argmax E/R fallback (.790598/.636400) but remains below the Safety-only reference (.923077/.829091). Per-reason F1: Safety **.888889**, Funding **.666667**, Efficacy **.774194** (reference .933333), Recruitment **.909091** (reference .989899), Other **.666667**. Efficacy validation confusion was TP12/FP5/FN2; Recruitment TP50/FP10/FN0. All S/F/O estimator identities and raw/final validation and final test predictions were unchanged.

Among the520 KNN fallback rows, Efficacy receives142 positives/378 negatives and Recruitment364 positives/156 negatives. Final fully labeled E/R training totals are Efficacy175 positives/996 negatives and Recruitment923 positives/248 negatives. Coverage rises to1171/1171 for both E/R, but the additional labels reduce their validation F1. E/R-only nonempty fallback applied to0 validation and5 test rows. No Kaggle CSV was generated, and prior experiment cells and files remain intact.

### Explicit export of K14 Safety-only C=0.1 model

At the user's request, an explicit notebook export cell selects the K14 selective S/F/O blend with threshold30, pseudo-label weight0.25, and **Safety C0.1 only**. Funding/Other retain C1; Efficacy/Recruitment retain their original K14 cluster-only fitted models and unresolved training rows. Neither E/R fallback experiment above is used. Fresh execution reproduced validation sample-wise F1 **.923077**, macro F1 **.829091**, and per-reason F1 Safety **.888889**, Funding **.666667**, Efficacy **.933333**, Recruitment **.989899**, Other **.666667**.

Generated separate `part1_kmeans_blend_sfo_k14_safety_c01_submission.csv` in `prith_attempts/part_1/`. It has312 exact test IDs in order, `ID,Predicted`, allowed reason names, and no null/empty predictions. Eight test rows needed the Safety-only nonempty fallback. Against the previous K14 all-selected-C1 export, only three test rows changed, all in Safety; the other four reasons' test decisions are identical. No prior CSV was overwritten and no Kaggle submission was made. Public score for this exported model is not yet observed.

The user subsequently reported this K14 Safety-only C0.1 model's public Kaggle score as **approximately .84**, a rounded value. The K21 C1 blend's score is the exact user-reported **.84508**; the rounded K14 report does not establish their exact ordering. The teammate hybrid's separately recorded public score is **.87927**. The higher K14 Safety-only validation F1 (.923077 versus K21 C1 .893162) did not translate into a clearly higher reported public score, consistent with the user's concern about validation selection on78 trials; this observation alone does not establish the cause.

Read-only CSV comparison: K14 Safety-only differs on3 rows from K14 all-selected-C1, solely in Safety; predicted Safety positives are48 versus49, while Funding6, Efficacy57, Recruitment169, Other83 match. Against K21 C1, it differs on18 test rows, with per-reason prediction changes Safety3, Funding0, Efficacy7, Recruitment6, Other10. Predicted positive counts K14 Safety-only versus K21 C1 are Safety48/49, Funding6/6, Efficacy57/58, Recruitment169/171, Other83/85. No model or CSV changed for this analysis.

### K21 uniform C=0.1 regularization across all five classifiers

The user requested stronger regularization on every reason classifier. A separate K21 cell refits five independent binary logistic regressions at **C=0.1** on the *same* recovered labels and weights as the K21 C1 threshold30/quarter-weight blend. Safety/Funding/Other each use all1171 rows (157 observed weight1,1014 recovered weight0.25). Efficacy/Recruitment each use624 rows (157 observed plus467 K21 cluster-recovered, equal weight1), leaving547 unresolved per reason; neither receives an E/R fallback. Original observed labels are checked unchanged. KPCA50/gamma1/75, K21 n_init100, smoothed 50/50 cluster/3NN recovery, threshold30, and scaled150 classifier inputs are unchanged.

The earlier K21 C1 export used an experiment-specific fallback that preserved the K21 cluster-only final E/R decisions. Because both E/R classifiers change here, this experiment evaluates **both C1 and C0.1 using the same global five-reason argmax nonempty fallback**. Under that common rule, C1 validation sample-wise/macro F1 is **.897436/.783153**, versus the earlier protected-fallback C1 **.893162/.775236**. C0.1 improves the common-rule score to **.923077/.828091**. C0.1 per-reason F1: Safety **.888889**, Funding **.666667**, Efficacy **.875000**, Recruitment **.989899**, Other **.720000**. The K14 Safety-only reference is .923077/.829091 on validation under its own postprocessing, so these aggregate validation scores are close, though the two models and fallback policies differ. Raw all-zero rows before the common fallback: C1 1 validation/18 test; C0.1 0 validation/18 test. No public score has been observed for C0.1 all-five.

Because the C0.1 variant is validation-competitive, a distinct `part1_kmeans_blend_sfo_k21_all_five_c01_submission.csv` was generated in `prith_attempts/part_1/`. It has312 exact test IDs, `ID,Predicted`, allowed reason names, and no null/empty predictions. It differs on47 test rows from the original K21 C1 protected-fallback CSV (which also uses a different fallback policy). Test positive counts are Safety43, Funding6, Efficacy76, Recruitment168, Other89. Prior notebooks and CSVs were preserved; no Kaggle upload occurred.

### Modular rebuild of teammate selected Part 1 hybrid

At the user's request, `Part1_Teammate_Hybrid_Rebuilt.ipynb` is a separate five-code-cell notebook with small functions for train-only scaling, KMeans10 clustering on the raw standardized150 embeddings, cluster-rate smoothing toward observed-label priors with strength2, three uniform originally labeled neighbors, equal score blending, 30% pseudo-label threshold plus argmax for empty pseudo rows, and five independent C0.1/liblinear logistic regressions with observed weight1 and recovered weight0.25. Final predictions use the teammate's 50% probability threshold plus highest-probability nonempty fallback. It uses only Part 1 data and fixed selected settings, without rerunning the teammate search grid.

Fresh execution exactly reproduced the saved teammate artifacts: all1171 recovered training label vectors, all78 validation prediction vectors, and all312 test reason sets matched. Validation sample-wise F1 was **.886752**, macro F1 **.779234**; per-reason Safety **.888889**, Funding **.666667**, Efficacy **.800000**, Recruitment **.952381**, Other **.588235**. Pseudo-positive counts among1014 masked rows were Safety95, Funding16, Efficacy147, Recruitment766, Other112; zero unresolved. Separate `part1_teammate_hybrid_rebuilt_submission.csv` in `prith_attempts/part_1/` was an exact dataframe match to `part1-results/knn-experiment/best_augmented_candidate_submission.csv` and has312 valid nonempty predictions. The original starter and all prior personal notebooks and CSVs were preserved. No Kaggle upload occurred.

### Teammate hybrid with KernelPCA for clustering only

The user requested one controlled representation change to the rebuilt teammate method. New isolated notebook cells fit RBF KernelPCA50 with gamma1/75 and seed42 on **training** scaled embeddings, then fit the same KMeans10/n_init10/seed42 on those transformed training points. Original-label 3NN votes still use raw scaled150; rate smoothing strength2, 50/50 score blend, threshold30 with pseudo-row argmax, all-five label recovery, C0.1/quarter-weight final logistic regressions on raw scaled150, and final nonempty fallback remain fixed. The baseline cells and CSV are unchanged.

Fresh validation sample-wise F1 changed **.886752→.873932**, macro F1 **.779234→.773888**. Per-reason baseline→KernelPCA F1: Safety .888889→.888889, Funding .666667→.666667, Efficacy .800000→.695652, Recruitment .952381→.925926, Other .588235→.692308. Recovered positives among1014 masked rows changed Safety95→75, Funding16→16, Efficacy147→89, Recruitment766→842, Other112→219; zero unresolved under both. In total222 training rows received a different recovered reason set. Thus this isolated KernelPCA clustering change did not improve validation sample or macro F1, although Other improved.

Separate `part1_teammate_hybrid_kernelpca_clusters_submission.csv` was generated in `prith_attempts/part_1/`, verified312 exact test IDs, `ID,Predicted`, allowed nonempty labels and no null values. It differs on84 test rows from the exact teammate baseline. Test positive counts baseline→KernelPCA are Safety49→44, Funding6→6, Efficacy47→34, Recruitment191→212, Other44→91. No Kaggle upload occurred.

### Prespecified KernelPCA dimension sweep in rebuilt teammate hybrid

The user requested a clustering-representation dimension sweep while keeping every other teammate setting fixed. Separate helper/run cells in `Part1_Teammate_Hybrid_Rebuilt.ipynb` independently fit train-only RBF KernelPCA at20,30,50,75,100,150 components (gamma1/75,seed42), followed by KMeans10/n_init10 on the transformed **training** embeddings. The raw scaled150 teammate model is a separate reference row. The original-label NN3 votes, smoothing2, 50/50 blending, 30% recovery threshold, quarter-weight C0.1 logistic regression on raw scaled150, and final fallback are unchanged. The 50-component run exactly reproduced the earlier independently fitted50 result in recovered labels and validation predictions.

| Clustering input | Sample F1 | Macro F1 | Safety | Funding | Efficacy | Recruitment | Other | Recovered positives S/F/E/R/O | Changed recovered rows vs raw |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Raw scaled150 | **.8868** | **.7792** | .8889 | .6667 | .8000 | .9524 | .5882 | 95/16/147/766/112 | 0 |
| KernelPCA20 | .8697 | .7688 | .8889 | .6667 | .6957 | .9259 | .6667 | 75/16/89/842/220 | 223 |
| KernelPCA30 | .8547 | .7514 | .8889 | .6667 | .6957 | .9174 | .5882 | 75/16/84/838/89 | 141 |
| KernelPCA50 | .8739 | .7739 | .8889 | .6667 | .6957 | .9259 | **.6923** | 75/16/89/842/219 | 222 |
| KernelPCA75 | .8547 | .7514 | .8889 | .6667 | .6957 | .9174 | .5882 | 75/16/84/838/89 | 141 |
| KernelPCA100 | .8547 | .7514 | .8889 | .6667 | .6957 | .9174 | .5882 | 75/16/84/838/89 | 141 |
| KernelPCA150 | .8697 | .7671 | .8889 | .6667 | .6957 | .9174 | .6667 | 75/16/94/838/210 | 214 |

All variants recover every masked row, so unresolved coverage is0 for each. Raw scaled150 remains the strongest validation sample/macro setting among this prespecified comparison. No model or CSV was selected or exported from the sweep, and prior baseline/KernelPCA50 CSVs were preserved. No Kaggle upload occurred.

### Actual RBF kernel K-means in teammate hybrid

The user requested kernel K-means. Isolated notebook cells implement the actual implicit-RKHS centroid-distance update from a **training-only** RBF Gram matrix (gamma1/75) on original scaled150 embeddings. Ten seeded kernel K-means++ starts use K10, deterministic seed42, empty-cluster repair, and selection by lowest final within-cluster kernel objective. The best start had objective **897.987279**, converged in15 iterations; all10 starts converged. Best-run cluster sizes were22/22/21/205/51/49/97/106/575/23 (all nonempty), with originally observed member counts2/1/1/33/9/3/12/13/78/5. Since clustering is used only to recover labels on training rows, no out-of-sample cluster assignment is needed.

The resulting train cluster IDs feed the same teammate observed-only smoothed rates (alpha2), three originally labeled raw-scaled150 neighbors, equal blend, 30% threshold plus pseudo-row argmax, all-five recovery, C0.1/quarter-weight logistic regressions on raw scaled150, and final nonempty fallback. Validation sample-wise/macro F1 were **.878205/.779427**, versus raw teammate KMeans **.886752/.779234** and KernelPCA50+ordinary KMeans **.873932/.773888**. Per-reason kernel K-means F1: Safety **.888889**, Funding **.666667**, Efficacy **.695652**, Recruitment **.925926**, Other **.720000**. Recovered positives among1014 masked rows were Safety75, Funding16, Efficacy90, Recruitment842, Other212; zero unresolved. The recovered label set differed on216 training rows from raw teammate KMeans. Thus actual kernel K-means did not improve the primary validation sample F1, though macro F1 was nearly equal and Other improved.

Separate `part1_teammate_hybrid_kernel_kmeans_submission.csv` was generated in `prith_attempts/part_1/` with312 exact test IDs, `ID,Predicted`, allowed nonempty reasons, and no nulls. It changes81 test rows versus raw teammate KMeans and9 versus KernelPCA50+KMeans. Test positive counts are Safety43, Funding6, Efficacy35, Recruitment213, Other87, compared with raw teammate49/6/47/191/44. No prior CSV was overwritten and no Kaggle upload occurred.

### Team Part 1 submission decision

The user states the team will use the shared `CS5785_Midterm_Starter.ipynb` and its **selected K-means/3NN hybrid** as the main Part 1 approach. The corresponding submission file is `part1-results/knn-experiment/best_augmented_candidate_submission.csv`, with a recorded public Kaggle score of **.87927**. The root `part1_submission.csv` is the older K-means output and is **not** that selected hybrid file. Personal notebooks and their submissions remain comparison experiments. This note records the decision only; no model, notebook source, CSV, or Kaggle submission was changed.
