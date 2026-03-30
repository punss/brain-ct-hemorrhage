# Tier 2 Model Results + Enhancements

## Setup

**Task:** Multi-label classification — predict presence/absence of 5 hemorrhage types independently.
**Labels:** epidural, intraparenchymal, intraventricular, subarachnoid, subdural
**Approach:** `OneVsRestClassifier` wrapping each base model. After OvR training, two post-hoc enhancements were applied to all 8 models (Tier 1 + Tier 2): per-label threshold calibration and classifier chains.
**Data:** 2,929 cases (train: 2,049 | val: 440 | test: 440), stratified by hemorrhage type.
**Preprocessing:**
- SVM: `StandardScaler` → `PCA` (n_components tuned) → `OvR(SVC, probability=True)`
- Random Forest: `StandardScaler` → `OvR(RandomForestClassifier)` (tree-based; internal feature selection makes PCA unnecessary)
- XGBoost: `OvR(XGBClassifier)` (tree-based, scale-invariant; no scaler or PCA)
**Tuning:** `RandomizedSearchCV`, 20 iterations, 5-fold CV stratified on hemorrhage type. Primary CV metric: macro AUC.
**Evaluation:** Val split only. Test split held out for final cross-tier evaluation.

---

## Overall Performance (OvR, default threshold)

| Model | CV Macro AUC | Val Macro AUC | Val Macro F1 | Train Time |
|-------|:------------:|:-------------:|:------------:|:----------:|
| SVM | 0.7156 | 0.7068 | 0.177 | 34.1s |
| Random Forest | 0.7222 | 0.7010 | 0.195 | 404.6s |
| XGBoost | **0.7336** | **0.7333** | **0.326** | 95.2s |

**Best AUC:** XGBoost (0.733) — also best Tier 1 model (Logistic Regression) at 0.698; Tier 2 improves by +0.036.

---

## Per-Label AUC (OvR)

| Label | SVM | Random Forest | XGBoost |
|-------|:---:|:-------------:|:-------:|
| epidural | **0.800** | 0.760 | 0.797 |
| intraparenchymal | **0.694** | 0.685 | 0.700 |
| intraventricular | 0.816 | 0.805 | **0.855** |
| subarachnoid | 0.629 | 0.636 | **0.661** |
| subdural | 0.594 | 0.618 | **0.654** |

XGBoost improves over Tier 1 on every label, most notably subdural (+0.043 vs Logistic Regression) and subarachnoid (+0.056). SVM achieves the highest epidural AUC (0.800) across all Tier 2 models.

---

## Best Hyperparameters

| Model | Best Parameters |
|-------|----------------|
| SVM | `pca__n_components=120`, `C=1.0`, `gamma=0.01` |
| Random Forest | `n_estimators=300`, `min_samples_leaf=2`, `max_features='sqrt'`, `max_depth=None` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=1` |

---

## Enhancement 1: Per-Label Threshold Calibration

All models were trained with a default 0.5 decision threshold. Because the labels are imbalanced, this threshold is poorly calibrated — the optimal threshold per label ranges from 0.05 to 0.33. Calibration sweeps thresholds 0.05–0.95 in 0.01 steps on the val set, picking the threshold that maximises binary F1 per label (no retraining).

### Calibrated Thresholds

| Model | epidural | intraparenchymal | intraventricular | subarachnoid | subdural |
|-------|:--------:|:----------------:|:----------------:|:------------:|:--------:|
| SVM | 0.32 | 0.27 | 0.26 | 0.17 | 0.16 |
| Random Forest | 0.26 | 0.33 | 0.30 | 0.22 | 0.25 |
| XGBoost | 0.12 | 0.21 | 0.33 | 0.05 | 0.14 |

### F1 Before and After Calibration (all 8 models)

| Model | Default F1 | Calibrated F1 | Gain |
|-------|:----------:|:-------------:|:----:|
| SVM | 0.177 | **0.506** | +0.329 |
| Random Forest | 0.195 | 0.492 | +0.297 |
| XGBoost | 0.326 | **0.512** | +0.185 |
| Logistic Regression | 0.203 | 0.485 | +0.282 |
| LDA | 0.229 | 0.490 | +0.261 |
| KNN | 0.211 | 0.456 | +0.245 |
| GNB | 0.251 | 0.438 | +0.187 |
| QDA | **0.404** | 0.454 | +0.051 |

Calibration delivers the largest gains to SVM (+0.329), Random Forest (+0.297), and Logistic Regression (+0.282) — all models whose raw probabilities are poorly spread around 0.5 on imbalanced data. QDA has the smallest gain because it was already using aggressive thresholds internally. After calibration, XGBoost leads at **0.512 macro F1**, followed by SVM (0.506).

---

## Enhancement 2: Classifier Chains

Classifier chains replace the independent OvR setup by feeding each label's predicted probability as an additional feature to subsequent classifiers. Label order was set by descending OvR AUC: intraventricular → epidural → intraparenchymal → subdural → subarachnoid. A `cv=5` inner fold was used during chain training to prevent label leakage.

### OvR vs Chain Comparison (val macro AUC)

| Model | OvR AUC | Chain AUC | Delta | OvR F1 (cal) | Chain F1 (cal) |
|-------|:-------:|:---------:|:-----:|:------------:|:--------------:|
| XGBoost | **0.7333** | 0.7279 | −0.0054 | 0.512 | **0.513** |
| SVM | 0.7068 | 0.7066 | −0.0002 | 0.506 | 0.505 |
| Random Forest | 0.7010 | 0.7046 | +0.0036 | 0.492 | 0.500 |
| Logistic Regression | 0.6976 | 0.6979 | +0.0003 | 0.485 | 0.485 |
| LDA | 0.6845 | 0.6855 | +0.0010 | 0.490 | 0.489 |
| QDA | 0.6708 | 0.6720 | +0.0012 | 0.454 | 0.454 |
| KNN | 0.6604 | 0.6596 | −0.0008 | 0.456 | 0.456 |
| GNB | 0.6406 | 0.6384 | −0.0022 | 0.438 | 0.438 |

Chains produce no meaningful AUC improvement over OvR for any model — deltas are within ±0.005. The label dependency structure is either too weak to exploit, or the 269 handcrafted features already encode the inter-label correlations implicitly. The chains do not hurt performance either, so there is no reason to prefer OvR in inference.

---

## Tier 1 vs Tier 2 Summary

| Model | Val Macro AUC | Val Macro F1 (cal) |
|-------|:-------------:|:------------------:|
| **XGBoost** | **0.733** | **0.512** |
| SVM | 0.707 | 0.506 |
| Random Forest | 0.701 | 0.492 |
| Logistic Regression | 0.698 | 0.485 |
| LDA | 0.685 | 0.490 |
| KNN | 0.660 | 0.456 |
| QDA | 0.671 | 0.454 |
| GNB | 0.641 | 0.438 |

---

## Observations

### Tier 2 closes the gap on the hard labels
The largest per-label AUC improvements from Tier 1 to Tier 2 (XGBoost) are on subdural (+0.043) and subarachnoid (+0.056) — the two labels that were hardest for linear models. XGBoost's ability to model non-linear interactions between ring zone features, histogram bins, and GLCM statistics provides a genuine advantage over Logistic Regression for diffuse, surface-level hemorrhages.

### XGBoost has the best CV-to-val fidelity
CV AUC (0.7336) and val AUC (0.7333) differ by only 0.0003 — the tightest alignment across all 8 models. This reflects XGBoost's regularization (subsample=0.8, colsample_bytree=0.6, min_child_weight=1) preventing fold-level overfitting. Random Forest shows a larger gap (0.7222 CV vs 0.7010 val), despite longer training time.

### SVM achieves the best epidural AUC (0.800)
The RBF kernel with PCA(120) + C=1.0 + gamma=0.01 provides the sharpest epidural discriminator of any model. This is consistent with the feature importance analysis: epidural blood produces a distinctive multi-component peripheral pattern that an RBF kernel can cleanly separate in PCA space. XGBoost is close at 0.797.

### Default F1 is misleadingly low — calibration is mandatory
Raw F1 scores at 0.5 threshold (0.177–0.326) dramatically understate model quality. After calibration, F1 rises to 0.438–0.512. The gap is caused by label imbalance: a model that predicts 0.25 probability for a label present in 15% of cases will be systematically cut off by a 0.5 threshold. Calibrated thresholds should be treated as part of the model, not as post-processing.

### Classifier chains add no value at this feature tier
The inter-label correlations in this dataset (e.g., intraventricular and intraparenchymal often co-occurring) are already captured by shared features in the 269-dim vector. Providing chain predictions as additional inputs does not give classifiers information they don't already have access to implicitly. This may change at Tier 3 with raw image inputs, where the feature space is not designed to capture cross-label patterns.

### Random Forest is slow with modest returns
RF took 6.7× longer than XGBoost (404s vs 95s) for a val AUC that is 0.032 lower. The unlimited depth (`max_depth=None`) likely explains both the training time and the larger CV-val gap. A constrained depth search would be worth exploring if RF were a candidate for deployment.
