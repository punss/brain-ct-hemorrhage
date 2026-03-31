# Test Set Evaluation — Tier 1 + Tier 2

## Setup

**Split:** Held-out test set — 440 cases, never seen during training or hyperparameter search.
**Thresholds:** Val-set calibrated thresholds from `calibration_summary.json`. These were not re-optimised on the test set — doing so would be data leakage.
**Metrics:**
- **Macro AUC-ROC** — threshold-free ranking quality, equal weight per label
- **Macro AUPRC** — area under precision-recall curve; more informative than AUC-ROC for imbalanced labels
- **Macro F1 / Precision / Recall** — hard-prediction quality at val-calibrated thresholds

**Test label positive rates:**

| Label | Positive rate |
|-------|:-------------:|
| epidural | 17.3% |
| intraparenchymal | 44.5% |
| intraventricular | 8.6% |
| subarachnoid | 25.7% |
| subdural | 25.2% |

---

## Summary Results

| Model | AUC | AUPRC | F1 (cal) | Precision | Recall |
|-------|:---:|:-----:|:--------:|:---------:|:------:|
| [T2] random_forest | 0.7439 | 0.4688 | 0.4752 | 0.4420 | 0.5629 |
| [T2] xgboost | 0.7308 | 0.4693 | 0.4617 | 0.4356 | 0.6056 |
| [T2] svm | 0.7214 | 0.4459 | 0.4709 | 0.3826 | 0.6631 |
| [T1] logistic_regression | 0.6965 | 0.4099 | 0.4409 | 0.3526 | 0.6388 |
| [T1] qda | 0.6884 | 0.3806 | 0.4538 | 0.3410 | 0.7051 |
| [T1] knn | 0.6823 | 0.4020 | 0.4161 | 0.2934 | 0.7511 |
| [T1] lda | 0.6757 | 0.3948 | 0.4339 | 0.3541 | 0.6537 |
| [T1] gnb | 0.6590 | 0.3584 | 0.4290 | 0.3174 | 0.6902 |

---

## Per-Label Results

### [T2] random_forest

**Macro:** AUC 0.7439 | AUPRC 0.4688 | F1 0.4752 | Precision 0.4420 | Recall 0.5629

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.8106 | 0.5007 | 0.4720 | 0.4471 | 0.5000 | 0.28 |
| intraparenchymal | 0.7273 | 0.6955 | 0.6589 | 0.5331 | 0.8622 | 0.35 |
| intraventricular | 0.7690 | 0.2838 | 0.2545 | 0.4118 | 0.1842 | 0.32 |
| subarachnoid | 0.7368 | 0.4608 | 0.5243 | 0.4545 | 0.6195 | 0.32 |
| subdural | 0.6757 | 0.4033 | 0.4660 | 0.3636 | 0.6486 | 0.27 |

### [T2] xgboost

**Macro:** AUC 0.7308 | AUPRC 0.4693 | F1 0.4617 | Precision 0.4356 | Recall 0.6056

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.8202 | 0.4897 | 0.5402 | 0.4796 | 0.6184 | 0.12 |
| intraparenchymal | 0.7476 | 0.7238 | 0.6639 | 0.5548 | 0.8265 | 0.21 |
| intraventricular | 0.7600 | 0.2999 | 0.2400 | 0.5000 | 0.1579 | 0.33 |
| subarachnoid | 0.6859 | 0.4445 | 0.4430 | 0.2945 | 0.8938 | 0.05 |
| subdural | 0.6400 | 0.3887 | 0.4214 | 0.3491 | 0.5315 | 0.14 |

### [T2] svm

**Macro:** AUC 0.7214 | AUPRC 0.4459 | F1 0.4709 | Precision 0.3826 | Recall 0.6631

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.8289 | 0.5072 | 0.5169 | 0.4510 | 0.6053 | 0.22 |
| intraparenchymal | 0.7188 | 0.6896 | 0.6406 | 0.5840 | 0.7092 | 0.37 |
| intraventricular | 0.7402 | 0.2673 | 0.3077 | 0.2642 | 0.3684 | 0.17 |
| subarachnoid | 0.6685 | 0.3805 | 0.4513 | 0.3009 | 0.9027 | 0.19 |
| subdural | 0.6509 | 0.3851 | 0.4378 | 0.3127 | 0.7297 | 0.21 |

### [T1] logistic_regression

**Macro:** AUC 0.6965 | AUPRC 0.4099 | F1 0.4409 | Precision 0.3526 | Recall 0.6388

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.7692 | 0.4034 | 0.4159 | 0.3133 | 0.6184 | 0.50 |
| intraparenchymal | 0.6913 | 0.6517 | 0.6174 | 0.5379 | 0.7245 | 0.42 |
| intraventricular | 0.7438 | 0.2494 | 0.2740 | 0.2857 | 0.2632 | 0.76 |
| subarachnoid | 0.6688 | 0.3973 | 0.4652 | 0.3191 | 0.8584 | 0.40 |
| subdural | 0.6093 | 0.3477 | 0.4320 | 0.3068 | 0.7297 | 0.42 |

### [T1] qda

**Macro:** AUC 0.6884 | AUPRC 0.3806 | F1 0.4538 | Precision 0.3410 | Recall 0.7051

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.7383 | 0.3329 | 0.4724 | 0.3821 | 0.6184 | 0.28 |
| intraparenchymal | 0.6888 | 0.6303 | 0.6299 | 0.5128 | 0.8163 | 0.05 |
| intraventricular | 0.7265 | 0.2285 | 0.2698 | 0.1932 | 0.4474 | 0.94 |
| subarachnoid | 0.6479 | 0.3668 | 0.4527 | 0.3149 | 0.8053 | 0.07 |
| subdural | 0.6405 | 0.3446 | 0.4439 | 0.3019 | 0.8378 | 0.08 |

### [T1] knn

**Macro:** AUC 0.6823 | AUPRC 0.4020 | F1 0.4161 | Precision 0.2934 | Recall 0.7511

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.7687 | 0.4313 | 0.4595 | 0.3493 | 0.6711 | 0.20 |
| intraparenchymal | 0.6757 | 0.6177 | 0.6248 | 0.4565 | 0.9898 | 0.12 |
| intraventricular | 0.6821 | 0.1793 | 0.1607 | 0.1216 | 0.2368 | 0.16 |
| subarachnoid | 0.6320 | 0.4030 | 0.4224 | 0.2757 | 0.9027 | 0.13 |
| subdural | 0.6532 | 0.3787 | 0.4133 | 0.2637 | 0.9550 | 0.06 |

### [T1] lda

**Macro:** AUC 0.6757 | AUPRC 0.3948 | F1 0.4339 | Precision 0.3541 | Recall 0.6537

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.7265 | 0.3727 | 0.4135 | 0.2895 | 0.7237 | 0.14 |
| intraparenchymal | 0.6688 | 0.6302 | 0.6194 | 0.4882 | 0.8469 | 0.28 |
| intraventricular | 0.6984 | 0.2491 | 0.2903 | 0.3750 | 0.2368 | 0.30 |
| subarachnoid | 0.6648 | 0.3756 | 0.4324 | 0.2826 | 0.9204 | 0.16 |
| subdural | 0.6202 | 0.3465 | 0.4138 | 0.3352 | 0.5405 | 0.26 |

### [T1] gnb

**Macro:** AUC 0.6590 | AUPRC 0.3584 | F1 0.4290 | Precision 0.3174 | Recall 0.6902

| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |
|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|
| epidural | 0.7093 | 0.2944 | 0.3922 | 0.3125 | 0.5263 | 0.22 |
| intraparenchymal | 0.6414 | 0.5974 | 0.6174 | 0.4841 | 0.8520 | 0.28 |
| intraventricular | 0.6808 | 0.2008 | 0.2778 | 0.2143 | 0.3947 | 0.30 |
| subarachnoid | 0.6501 | 0.3623 | 0.4449 | 0.2982 | 0.8761 | 0.16 |
| subdural | 0.6136 | 0.3373 | 0.4130 | 0.2781 | 0.8018 | 0.16 |

---

## Val vs Test Comparison

| Model | Val AUC | Test AUC | Val F1 (cal) | Test F1 (cal) |
|-------|:-------:|:--------:|:------------:|:-------------:|
| [T2] Random Forest | 0.7211 | **0.7439** | 0.508 | 0.475 |
| [T2] XGBoost | **0.7333** | 0.7308 | **0.512** | 0.462 |
| [T2] SVM | 0.7205 | 0.7214 | 0.498 | 0.471 |
| [T1] Logistic Regression | 0.7023 | 0.6965 | 0.491 | 0.441 |
| [T1] QDA | 0.6708 | 0.6884 | 0.454 | 0.454 |
| [T1] KNN | 0.6604 | 0.6823 | 0.456 | 0.416 |
| [T1] LDA | 0.6845 | 0.6757 | 0.490 | 0.434 |
| [T1] GNB | 0.6406 | 0.6590 | 0.438 | 0.429 |

---

## Observations

### Random Forest overtakes XGBoost on test AUC
RF achieves the highest test AUC (0.744) despite ranking second on val (0.721). XGBoost remains stronger on val (0.733) but its test AUC (0.731) is slightly lower. The gap between them is small (0.013) and likely within sampling variance given the test set size (440 cases). Both AUPRC scores are nearly identical (RF: 0.469, XGBoost: 0.469), suggesting comparable precision-recall performance. For the progress report, both should be cited as the top Tier 2 models rather than declaring a single winner.

### Val-to-test generalisation is strong across all models
AUC differences between val and test are within ±0.023 for all models. No model degrades substantially, indicating that the 70/15/15 stratified split produced representative partitions and that overfitting to the val set did not occur at the model level.

### Calibrated F1 drops modestly on test — expected
Test F1 is 0.03–0.05 lower than val F1 for all models. This is the expected cost of optimising thresholds on val: the thresholds were chosen to maximise F1 on exactly those 440 val cases, so some drop on unseen test cases is inevitable. The drop is small, confirming that threshold calibration generalises reasonably to the test distribution.

### AUPRC reveals the true cost of label imbalance
Macro AUC-ROC (0.66–0.74) is considerably higher than Macro AUPRC (0.36–0.47) for every model. AUC-ROC rewards correctly ranking the many negative cases; AUPRC focuses on how precisely the model identifies the rare positives. The large gap — roughly 0.28–0.30 points — shows that while models rank cases correctly, their precision on positive predictions is low, particularly for intraventricular and epidural. AUPRC is the more honest metric for this problem and should be the primary metric in the final paper.

### Intraventricular is the hardest label at test time despite high AUC
Intraventricular AUC on test is 0.68–0.77 across models — high relative to the label, but F1 at the val-calibrated threshold is very low (0.16–0.31). The val-optimised thresholds for IVH (0.16–0.94 depending on model) are sensitive to the small number of positive cases (38 in val, ~38 in test), making threshold transfer noisy. This is the label where threshold calibration's in-sample optimism is most visible.

### SVM retains the best epidural performance
SVM's epidural AUC (0.829) remains the highest on test, matching its val performance closely (0.830 val → 0.829 test). The RBF kernel with balanced class weights and PCA(120) generalises well for this label.
