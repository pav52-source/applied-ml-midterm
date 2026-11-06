# Team comparison and next experiments

Reviewed upstream `origin/main` at `0233983` alongside Prith's local notebooks. This document summarizes saved outputs and code; no new experiments were run for this review.

## Part 1: results

| Metric | Prith: kernel PCA + K-means | Teammate: K-means/KNN hybrid |
|---|---:|---:|
| Validation sample-wise F1 | **0.8880** | 0.8868 |
| Validation macro F1 | 0.7272 | **0.7792** |
| Safety F1 | 0.8571 | **0.8889** |
| Funding F1 | 0.3333 | **0.6667** |
| Efficacy F1 | **0.8667** | 0.8000 |
| Recruitment F1 | **0.9703** | 0.9524 |
| Other F1 | **0.6087** | 0.5882 |
| Training trials still unresolved | 46.7% | **0%** |
| Public Kaggle sample-wise F1 | approximately 0.82, user-reported | 0.87927, recorded in notes/report |

Both validation columns use the nonempty-prediction fallback: if no reason reaches the final threshold, select the highest-scoring reason. The matching observed-label-only C=1 baseline is 0.8782 sample-wise F1. Historical scores before that fallback are different and should not be mixed into this comparison.

## What each method does

- **Prith:** standardize all training embeddings; RBF kernel PCA to 50 components with gamma 1/75 for clustering; K-means with 21 clusters, seed 42, and 100 initializations. Use raw cluster reason frequencies and a 50% threshold. Keep empty cluster mappings unresolved. Train logistic regression with C=1 on the original 150 scaled embeddings, giving observed and recovered rows equal weight.
- **Teammate:** K-means on the original 150 scaled embeddings, with 10 clusters, seed 42, and 10 initializations. Smooth cluster rates toward visible-training rates with strength 2. Blend those rates equally with votes from three uniformly weighted labeled neighbors. Threshold recovery scores at 0.30, using an argmax fallback for empty sets. Train logistic regression with C=0.1, observed weight 1, and recovered-label weight 0.25.
- The teammate hybrid fills all 1,014 masked rows. It generates 16 Funding pseudo-positives. Its validation Funding predictions are 2 true positives, 0 false positives, and 2 false negatives, among only 4 true Funding positives.
- KNN-only recovery plus LR has sample F1 0.8803 and macro F1 0.7873. Direct supervised KNN is weaker and finds no Funding-positive validation examples. Local label recovery, rather than merely switching the final classifier to KNN, is the useful finding.

Sources: [upstream starter notebook](../../CS5785_Midterm_Starter.ipynb), especially cells 8–11 and 13–20; [Part 1 notes](../../PART1_NOTES.md); [hybrid settings](../../part1-results/knn-experiment/best_augmented_candidate.json); [all KNN results](../../part1-results/knn-experiment/results.csv); [hybrid validation predictions](../../part1-results/knn-experiment/validation_kmeans_knn_hybrid_k3_uniform.csv); [recovered training labels](../../part1-results/knn-experiment/best_augmented_training_labels.csv).

## Split safety and strength of evidence

- The teammate scaler, cluster model, KNN reference set, text vocabulary, and final classifier are fitted on training data. Neighbors come only from the 157 originally labeled training rows. Validation trials are not neighbors or fitted classifier rows.
- Observed training labels are preserved by assertions and in the saved augmented labels. All saved hybrid rows have at least one label. The validation predictions independently reproduce sample F1 0.886752 and macro F1 0.779234 when joined to truth by ID.
- Validation is used to select hyperparameters, as permitted. Repeated selection across 99 configurations and only 78 validation trials makes tiny gains uncertain. Four Funding positives are too few to establish broad rare-class superiority.
- The teammate's 0.87927 public score is documented in [Part 1 notes](../../PART1_NOTES.md), [report](../../writeup/main.tex), and [writeup README](../../writeup/README.md). No separate Part 1 Kaggle API-status JSON was found, so this is a recorded result, not independently verified against Kaggle during this review. Prith's approximately 0.82 is user-reported.
- Zero unresolved rows means greater recovery coverage, not proof of correct labels. The methods also differ in smoothing, threshold, weighting, and LR regularization, so the score difference cannot be attributed solely to KNN or kernel PCA.

## Safe combined experiments

1. Freeze both existing model outputs and settings as references. Retain their distinct submission filenames.
2. Start from the teammate hybrid and change **only clustering input** from raw scaled embeddings to Prith's 50-component RBF kernel PCA (gamma 1/75). Keep 10 clusters, KNN references in the raw 150-dimensional space, smoothing 2, equal score blending, threshold 0.30, LR C=0.1, and weight 0.25. This isolates the representation change.
3. Separately compare 21 clusters with 10 under that same pipeline. Do not change cluster count and representation in the same comparison.
4. Separately test recovered-label weight 1 versus 0.25. Prith deliberately removed weighting from their notebook; reproducing the teammate method should explicitly disclose it rather than silently restore it.
5. Review sample/macro/per-reason F1, unresolved percentage, class support, and seed sensitivity. Compare validation first; do not tune repeatedly against public Kaggle scores.

No combined experiment has been executed yet.

## Part 2: use the completed teammate work

| Phase | Selected representation/model | Calibration | Baseline F1 → selected | Baseline loss → selected |
|---|---|---|---:|---:|
| I | Eligibility-criteria TF-IDF + LR, C=10 | None | 0.7006 → **0.7821** | 0.8791 → **0.5603** |
| II | Eligibility-criteria TF-IDF + LR, C=1 | Training-only, 3-fold sigmoid | 0.6141 → **0.6288** | 1.0603 → **0.6127** |
| III | Separate criteria/context/ICD TF-IDF blocks + LR, C=1 | Training-only, 3-fold sigmoid | 0.7739 → **0.7915** | 0.8808 → **0.5485** |

Equal-phase validation log loss improves from **0.9401 to 0.5738**. F1, ROC-AUC, and average precision improve in every phase. Selection JSON records no refitting on validation. Calibration preprocessing is refitted within training folds; Part 1 reasons are used only for the error audit, not as outcome-model features.

The important tradeoff is Phase III: overall false-success rate rises from **49.59% to 59.35%** even as aggregate F1 and log loss improve. The report includes category sizes, observed/predicted reason provenance, subgroup regressions, and a measured balanced alternative. Keep that discussion when building on these results.

The machine-readable submission record reports `COMPLETE`, public log loss **0.61046**, and submission reference `58039392`. This review read that record; it did not query Kaggle. Public-subset aggregate performance does not establish full-test or per-phase test results, or prove why performance differs from validation.

Sources: [Part 2 walkthrough](PART2_WALKTHROUGH.md), [selected model specifications](part2-results/selected_models.json), [metric comparison](part2-results/metric_comparison.csv), [reason error comparison](part2-results/reason_error_comparison.csv), and [recorded submission](part2-results/kaggle_submission.json).

The existing Part 2 pipeline, provenance artifacts, and report should be used rather than rebuilt. The report at [writeup/main.tex](../../writeup/main.tex) is described as a four-page draft in its documentation; this review did not rebuild the PDF. Team names and responsibilities still require user-provided facts.
