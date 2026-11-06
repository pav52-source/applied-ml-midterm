# Additional Part 1 attempts in the shared starter notebook

The shared [starter notebook](../../CS5785_Midterm_Starter.ipynb) contains more than the selected K-means/KNN hybrid. Its Part 1 cells 8–20 have saved execution outputs. The most distinct follow-up is a **Funding-only recovery-threshold experiment** in cells 15–17.

## Funding threshold experiment

The team held its initial K-means recovery model fixed: 10 clusters on the original 150 scaled embeddings, smoothed cluster reason rates, all five recovered labels on all 1,014 masked rows, logistic regression `C=0.1`, and recovered-row weight `0.25`. It changed only the **Funding pseudo-label threshold**. This threshold decides training labels; the final validation prediction threshold remained 0.5. Original observed labels and the other four recovered reason columns were unchanged.

| Funding recovery threshold | Recovered Funding positives | Validation Funding TP / FP / FN | Funding precision / recall / F1 | Sample F1 | Macro F1 |
|---:|---:|---:|---:|---:|---:|
| 0.30 (control) | 0 | 1 / 0 / 3 | 1.00 / 0.25 / 0.40 | 0.8547 | 0.6923 |
| 0.20 | 112 | 1 / 3 / 3 | 0.25 / 0.25 / 0.25 | 0.8419 | 0.6623 |
| 0.15 | 112 | 1 / 3 / 3 | 0.25 / 0.25 / 0.25 | 0.8419 | 0.6623 |
| 0.10 | 112 | 1 / 3 / 3 | 0.25 / 0.25 / 0.25 | 0.8419 | 0.6623 |

The same 112 masked trials in cluster 7 cross all three lower thresholds; no other cluster crosses even 0.10. Lowering the threshold gave the final classifier **no additional true positive** on the four Funding-positive validation trials and three extra false positives. The team retained 0.30. This does **not** show that Funding recovery is useless: it shows that promoting a whole cluster with this fixed representation and classifier did not help. Only four validation Funding positives make the comparison fragile. The notebook saves results and validation predictions under [`part1-results/funding-threshold-experiment`](../../part1-results/funding-threshold-experiment/); it does not claim a Kaggle score for this follow-up.

## Other executed Part 1 comparisons

- **Initial 70-configuration grid (cells 8–11):** K-means and diagonal GMM with 5, 10, or 20 components; recovery thresholds 0.30 or 0.50; pseudo-label weights 0.25 or 1; logistic regression `C=0.1` or `1`. It also tested adding start year and text/drug features to the final classifier, while clustering stayed on the 150 embeddings. The best augmented result in this grid was K-means 10 at **sample F1 0.8547, macro F1 0.6923**. Best GMM was **0.8312 / 0.6418**; added year/text did not improve the selected embedding-only K-means result. A supervised embedding-only control scored **0.8782 / 0.6990** with the same nonempty prediction rule.
- **Local-neighbor follow-up (cells 18–20):** 25 saved configurations compare direct KNN, KNN-only recovery then logistic regression, and the K-means/KNN blend. The best **KNN-only recovery** uses three uniform original-labeled neighbors and scores **0.8803 / 0.7873**, with 45 Funding pseudo-positives and Funding F1 0.6667. The best **hybrid** uses the same neighbors plus equal smoothed cluster/neighbor scores and scores **0.8868 / 0.7792**, with 16 Funding pseudo-positives and Funding F1 0.6667. This hybrid is the previously reviewed team selection. Direct KNN's best sample F1 is **0.8632**, and it finds no Funding-positive validation examples. The hybrid's public **0.87927** is recorded in team notes, not in the notebook's execution outputs.

## Comparison with Prith's current approach

Prith's [latest K=14 selective blend](Part1_KMeans_Experiments.ipynb) uses 50-dimensional kernel PCA for clustering, a 30% recovery threshold, quarter weight, and equal cluster/neighbor blending for **Safety, Funding, and Other**. Efficacy and Recruitment keep cluster-only classifiers, with 520 masked training labels unresolved for each. Its saved validation scores are **sample F1 0.9103, macro F1 0.7925**; it has no Kaggle result. Prith's submitted K=21 version has validation **0.8932 / 0.7752** and user-reported public **0.84508**.

The Funding threshold failure is relevant: a lower cluster-only threshold added 112 Funding positives without improving Funding recall. The later **local** recovery identified Funding positives more selectively and improved validation Funding F1. Prith already uses local blending for Funding, so repeating the starter's cluster-only Funding sweep would not test the same method. Scores from different pipelines cannot isolate the cause of the public-score gap, and repeated choices on 78 validation trials are uncertain.

Sources: [starter notebook](../../CS5785_Midterm_Starter.ipynb), [initial grid](../../part1-results/experiments.csv), [Funding threshold results](../../part1-results/funding-threshold-experiment/results.csv), [neighbor follow-up results](../../part1-results/knn-experiment/results.csv), [team notes](../../PART1_NOTES.md), and [Prith's experiment log](EXPERIMENT_LOG.md).
