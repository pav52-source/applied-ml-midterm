# Part 1 — Clinical Trial Stopping-Reason Classification

## Objective and dataset

The task was to predict five potentially overlapping reasons for clinical trial termination: **Safety, Funding, Efficacy, Recruitment, and Other**. There were 1,171 training trials, including **157 labeled** and **1,014 unlabeled** trials; the validation set had **78 labeled** trials, and the test set had **312** trials. We used the 150-dimensional stopping-reason embeddings as features. Our goals were to establish a supervised baseline, use unsupervised learning to exploit the unlabeled data, compare alternative approaches, and select a model based primarily on validation sample-wise F1 while monitoring macro and per-reason F1.

## Results at a glance

| Experiment | Approach | Sample-wise F1 | Macro F1 |
|---|---|---:|---:|
| A | Supervised one-vs-rest Logistic Regression | 0.8483 | 0.7453 |
| B | K-Means pseudo-labels + Logistic Regression | 0.865 | 0.770 |
| C | GMM pseudo-labels + Logistic Regression | 0.827 | 0.683 |
| D | Per-label K-Means pseudo-labeling | 0.795 | 0.635 |
| E | K-Means cluster-distance features + Logistic Regression | 0.848 | 0.745 |
| F | Random pseudo-label selection control | 0.739 | 0.530 |
| G | Experiment B with per-reason thresholds | **0.8825** | **0.7867** |
| H | RBF SVM | 0.803 | 0.674 |

All results were measured on the same 78-trial validation set. Rounded values reflect the precision of the recorded outputs.

## Experiment A — Supervised baseline

**Goal:** Establish performance without using the unlabeled trials.

**Method:** Standardize the 150 embedding features and train five separate, class-balanced Logistic Regression classifiers on the 157 labeled trials. Convert probabilities to labels using 0.5 thresholds.

**Result:** Sample-wise F1 **0.8483**, macro F1 **0.7453**. Per-reason F1: Safety **0.824**, Funding **0.667**, Efficacy **0.720**, Recruitment **0.990**, Other **0.526**.

**Lesson:** The embeddings already provided a strong signal, particularly for Recruitment. Funding and Other were substantially more difficult.

## Experiment B — K-Means pseudo-labeling

**Goal:** Exploit unlabeled embeddings to enlarge the supervised training set.

**Method:** Fit K-Means with **8 clusters** on all 1,171 training embeddings. Estimate five smoothed label probabilities for each cluster using labeled members. Pseudo-label a trial only if **all five** probabilities are either at most **0.2** or at least **0.8**, and the cluster has at least **5 labeled** examples. Retrain Logistic Regression with pseudo-label sample weight **0.3**.

**Result:** **285** trials selected; sample-wise F1 **0.865**, macro F1 **0.770**. Safety F1 rose from 0.824 to **0.889**, Other from 0.526 to **0.636**, while Efficacy declined from 0.720 to **0.667**.

**Lesson:** Selective pseudo-labeling improved validation performance, but the improvement was uneven. Inspection revealed that **every one of the 285 selected trials was labeled `[0, 0, 0, 1, 0]`**, i.e., Recruitment only. The all-label confidence criterion selected a narrow subset rather than diverse reason combinations.

## Experiment C — Gaussian Mixture Model pseudo-labeling

**Goal:** See whether soft cluster memberships would improve pseudo-label selection.

**Method:** Fit an **8-component diagonal-covariance GMM**; estimate reason probabilities using soft membership weights and apply confidence-based pseudo-labeling.

**Result:** **482** pseudo-labeled trials; sample-wise F1 **0.827**, macro F1 **0.683**. Funding F1 fell to **0.400**; Efficacy rose to **0.750**.

**Lesson:** More pseudo-labels did not necessarily help. Possible label noise was one explanation, but pseudo-label accuracy could not be directly checked without ground truth.

## Experiment D — Per-label K-Means pseudo-labeling

**Goal:** Avoid discarding an entire trial because one or two of its five label estimates were uncertain.

**Method:** Apply the 0.2/0.8 confidence rule separately for each reason, giving each binary classifier its own pseudo-labeled training subset.

| Reason | Positive pseudo-labels | Negative pseudo-labels |
|---|---:|---:|
| Safety | 0 | 702 |
| Funding | 0 | 914 |
| Efficacy | 0 | 517 |
| Recruitment | 418 | 212 |
| Other | 0 | 569 |

**Result:** Sample-wise F1 **0.795**, macro F1 **0.635**.

**Lesson:** The cluster probabilities never exceeded 0.8 for Safety, Funding, Efficacy, or Other. Per-label selection therefore generated hundreds of negatives but no positives for these four reasons. The validation model predicted just **1 Funding positive versus 4 actual**, and **3 Other positives versus 10 actual**. Adding more training rows with one-sided labels could degrade recall and overall performance.

## Experiment E — K-Means distances as features

**Goal:** Use unsupervised structure without inventing labels.

**Method:** Append standardized distances to the **8 cluster centers** to the original 150 embeddings, yielding **158 features**; fit Logistic Regression on the 157 labeled trials.

**Result:** Sample-wise F1 **0.848**, macro F1 **0.745**, matching the baseline's reported metrics.

**Lesson:** The added cluster-distance features did not improve validation F1 under this setup. This does not prove that they carried no information.

## Experiment F — Random pseudo-label control

**Goal:** Test whether the improvement in Experiment B came from *which trials* K-Means selected rather than merely adding 285 examples.

**Method:** Randomly choose **285** unlabeled trials, assign all the Recruitment-only vector `[0, 0, 0, 1, 0]`, and use the same pseudo-label weight **0.3**.

**Result:** Sample-wise F1 **0.739**, macro F1 **0.530**.

**Lesson:** Selecting the right trials mattered. The poor result from random assignment was consistent with label noise, while K-Means selection was more useful. This was a **single random draw**, so it does not establish the average behavior of random selection or prove that K-Means pseudo-labels were correct.

## Experiment G — Per-reason prediction thresholds

**Goal:** Improve the best existing model without retraining it.

**Method:** Use Experiment B's predicted probabilities and select a separate decision threshold for each reason by maximizing that reason's validation F1 over a grid from **0.20 to 0.80**, in steps of **0.05**.

| Reason | Selected threshold | F1 after tuning |
|---|---:|---:|
| Safety | 0.20 | 0.900 |
| Funding | 0.20 | 0.667 |
| Efficacy | 0.20 | 0.741 |
| Recruitment | 0.40 | 0.990 |
| Other | 0.40 | 0.636 |

**Result:** Sample-wise F1 **0.8825**, macro F1 **0.7867** — the highest observed validation scores.

**Lesson:** The default 0.5 cutoff was not ideal for every label, especially categories with missed positives. **Caution:** Thresholds were selected and evaluated on the same small validation set, making these scores potentially optimistic; they are not independent estimates of test performance.

## Experiment H — RBF Support Vector Machine

**Goal:** Check whether nonlinear classification could outperform Logistic Regression.

**Method:** Train five class-balanced RBF-kernel SVM classifiers on the 157 labeled trials.

**Result:** Sample-wise F1 **0.803**, macro F1 **0.674**.

**Lesson:** This nonlinear alternative did not improve performance under the tested settings. Logistic Regression remained preferable.

## Overall conclusions and selected approach

The best observed configuration was **8-cluster K-Means pseudo-label selection followed by weighted one-vs-rest Logistic Regression, with per-reason thresholds `[0.20, 0.20, 0.20, 0.40, 0.40]`**. Its observed validation sample-wise F1 was **0.8825**, compared with **0.8483** for the original supervised baseline.

Across the experiments, the main findings were:

1. The pretrained embeddings and a simple linear classifier provided a strong baseline.
2. Selective pseudo-labeling was more helpful than simply adding large numbers of unlabeled examples.
3. The K-Means confidence rule selected exclusively Recruitment-only pseudo-labels, limiting diversity.
4. Per-label pseudo-labeling increased coverage but introduced severe positive/negative imbalance and worsened performance.
5. Cluster-distance features and RBF SVM did not improve on the baseline in the tested configurations.
6. Label-specific prediction thresholds provided the largest additional improvement, though they were tuned on a small validation set.

**Limitations:** Only 157 ground-truth training labels and 78 validation trials were available. Labeled and test trials also differed in their temporal distributions, so validation performance might not generalize fully to later test trials. Unlabeled pseudo-label accuracy could not be directly verified. The reported best score should therefore be understood as the **best observed validation result**, not a guaranteed test score.

**Next step:** Generate the Part 1 test predictions with the selected model and thresholds, verify the Kaggle submission format, and proceed to Part 2.
