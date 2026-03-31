# Tier 2 Model Results + Enhancements

## Setup

**Task:** Multi-label classification — predict presence/absence of 5 hemorrhage types independently.
**Labels:** epidural, intraparenchymal, intraventricular, subarachnoid, subdural
**Approach:** `OneVsRestClassifier` wrapping each base model. After OvR training, two post-hoc enhancements were applied to all 8 models (Tier 1 + Tier 2): per-label threshold calibration and classifier chains.
**Data:** 2,929 cases (train: 2,049 | val: 440 | test: 440), stratified by hemorrhage type.
**Preprocessing:**
- SVM: `StandardScaler` → `PCA` (n_components tuned) → `OvR(SVC, probability=True, class_weight='balanced')`
- Random Forest: `StandardScaler` → `OvR(RandomForestClassifier, class_weight='balanced')`
- XGBoost: `OvR(XGBClassifier)` — XGBoost's `scale_pos_weight` cannot be set per-label inside OvR; imbalance is addressed via threshold calibration instead
**Tuning:** `RandomizedSearchCV`, 20 iterations, 5-fold CV stratified on hemorrhage type. Primary CV metric: macro AUC-ROC, averaged equally across all 5 labels regardless of positive rate. Macro averaging treats each hemorrhage type as equally important — consistent with the clinical framing where rare types carry the same diagnostic weight as common ones.
**Evaluation:** Val split only. Test split held out for final cross-tier evaluation.

---

## Overall Performance (OvR, default threshold)

| Model | CV Macro AUC | Val Macro AUC | Val Macro F1 | Train Time |
|-------|:------------:|:-------------:|:------------:|:----------:|
| SVM | 0.7233 | 0.7205 | 0.416 | 38.2s |
| Random Forest | **0.7335** | 0.7211 | 0.241 | 322.4s |
| XGBoost | 0.7336 | **0.7333** | **0.326** | 98.6s |

**Best AUC:** XGBoost (0.733) — best model across Tier 1 and Tier 2.
SVM and RF are tied at 0.721.

---

## Per-Label AUC (OvR)

| Label | SVM | Random Forest | XGBoost |
|-------|:---:|:-------------:|:-------:|
| epidural | **0.830** | 0.792 | 0.797 |
| intraparenchymal | 0.692 | 0.691 | **0.700** |
| intraventricular | 0.837 | 0.847 | **0.855** |
| subarachnoid | 0.638 | 0.655 | **0.661** |
| subdural | 0.607 | 0.620 | **0.654** |

XGBoost leads on every label except epidural, where SVM achieves 0.830 — the highest epidural AUC across all Tier 1 and Tier 2 models. XGBoost shows the most improvement on the hard labels (subdural: 0.654, subarachnoid: 0.661) compared to the best Tier 1 model.

---

## Best Hyperparameters

| Model | Best Parameters |
|-------|----------------|
| SVM | `pca__n_components=120`, `C=1.0`, `gamma=0.01` |
| Random Forest | `n_estimators=500`, `min_samples_leaf=2`, `max_features='sqrt'`, `max_depth=30` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=8`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=1` |

---

## Enhancement 1: Per-Label Threshold Calibration

All models were evaluated at their natural output threshold. Because the labels are imbalanced, this threshold is not optimal for F1. Calibration sweeps thresholds 0.05–0.95 in 0.01 steps on the val set, picking the threshold that maximises binary F1 per label (no retraining).

### Calibrated Thresholds

| Model | epidural | intraparenchymal | intraventricular | subarachnoid | subdural |
|-------|:--------:|:----------------:|:----------------:|:------------:|:--------:|
| Logistic Regression | 0.50 | 0.42 | 0.76 | 0.40 | 0.42 |
| LDA | 0.14 | 0.28 | 0.30 | 0.16 | 0.26 |
| QDA | 0.28 | 0.05 | 0.94 | 0.07 | 0.08 |
| GNB | 0.22 | 0.28 | 0.30 | 0.16 | 0.16 |
| KNN | 0.20 | 0.12 | 0.16 | 0.13 | 0.06 |
| SVM | 0.22 | 0.37 | 0.17 | 0.19 | 0.21 |
| Random Forest | 0.28 | 0.35 | 0.32 | 0.32 | 0.27 |
| XGBoost | 0.12 | 0.21 | 0.33 | 0.05 | 0.14 |

Logistic Regression's thresholds are close to 0.4–0.5 across all labels, reflecting well-calibrated probability estimates from class weighting. Models without class weighting (LDA, GNB, KNN) require thresholds well below 0.3, indicating their raw probabilities are biased toward the majority class.

### F1 Before and After Calibration (all 8 models)

| Model | Default F1 | Calibrated F1 | Gain |
|-------|:----------:|:-------------:|:----:|
| XGBoost | 0.326 | **0.512** | +0.185 |
| Random Forest | 0.241 | **0.508** | +0.267 |
| SVM | 0.416 | 0.498 | +0.082 |
| Logistic Regression | 0.447 | 0.491 | +0.044 |
| LDA | 0.229 | 0.490 | +0.261 |
| KNN | 0.211 | 0.456 | +0.245 |
| GNB | 0.251 | 0.438 | +0.187 |
| QDA | 0.404 | 0.454 | +0.051 |

Models with `class_weight='balanced'` (LR, SVM) show the smallest calibration gains — their probabilities are already reasonably calibrated and threshold adjustment provides only marginal improvement. Models without class weighting require large threshold shifts to reach comparable F1.

---

## Enhancement 2: Classifier Chains

Classifier chains replace the independent OvR setup by feeding each label's predicted probability as an additional feature to subsequent classifiers. Label order was set by descending OvR AUC: intraventricular → epidural → intraparenchymal → subdural → subarachnoid. A `cv=5` inner fold was used during chain training to prevent label leakage.

### OvR vs Chain Comparison (val macro AUC)

| Model | OvR AUC | Chain AUC | Delta | OvR F1 (cal) | Chain F1 (cal) |
|-------|:-------:|:---------:|:-----:|:------------:|:--------------:|
| XGBoost | **0.7333** | 0.7279 | −0.0054 | **0.512** | 0.513 |
| Random Forest | 0.7211 | 0.7229 | +0.0018 | 0.508 | 0.507 |
| SVM | 0.7205 | 0.7208 | +0.0003 | 0.498 | 0.499 |
| Logistic Regression | 0.7023 | 0.7019 | −0.0004 | 0.491 | 0.491 |
| LDA | 0.6845 | 0.6855 | +0.0010 | 0.490 | 0.489 |
| QDA | 0.6708 | 0.6720 | +0.0012 | 0.454 | 0.454 |
| KNN | 0.6604 | 0.6596 | −0.0008 | 0.456 | 0.456 |
| GNB | 0.6406 | 0.6384 | −0.0022 | 0.438 | 0.438 |

Chains produce no meaningful AUC improvement over OvR — all deltas are within ±0.006. The label dependency structure is either too weak to exploit, or the 269 handcrafted features already encode inter-label correlations implicitly.

---

## Tier 1 vs Tier 2 Summary (post-calibration)

| Model | Val Macro AUC | Val Macro F1 (cal) |
|-------|:-------------:|:------------------:|
| **XGBoost** | **0.733** | **0.512** |
| Random Forest | 0.721 | 0.508 |
| SVM | 0.721 | 0.498 |
| Logistic Regression | 0.702 | 0.491 |
| LDA | 0.685 | 0.490 |
| KNN | 0.660 | 0.456 |
| QDA | 0.671 | 0.454 |
| GNB | 0.641 | 0.438 |

---

## Observations

### XGBoost is the best overall model
XGBoost leads on both metrics (AUC 0.733, calibrated F1 0.512) and has the strongest generalization — CV AUC (0.734) and val AUC (0.733) differ by only 0.001. Its regularization (subsample=0.8, colsample_bytree=0.6) prevents overfitting without requiring explicit class weighting.

### Tier 2 closes the gap on the hard labels
The largest per-label improvements from Tier 1 to Tier 2 are on subdural (LR: 0.608 → XGB: 0.654, +0.046) and subarachnoid (LR: 0.608 → XGB: 0.661, +0.053). XGBoost's ability to model non-linear interactions between ring zone features, histogram bins, and GLCM statistics provides a genuine advantage over linear models for diffuse, surface-level hemorrhages.

### SVM achieves the best epidural AUC (0.830)
The RBF kernel with PCA(120) + `class_weight='balanced'` provides the sharpest epidural discriminator of any model. Epidural is the most imbalanced label; class weighting has the largest effect on the most imbalanced classes, and the kernel boundary can cleanly separate the multi-component peripheral signal in PCA space.

### RF raw F1 is low despite class weighting
RF's raw F1 (0.241) remains the lowest among Tier 2 models despite `class_weight='balanced'`. RF with `max_depth=30` produces near-binary leaf predictions — probability estimates cluster near 0 and 1 rather than spanning a calibrated range. Calibration corrects this, but the issue is structural to tree ensembles with deep trees rather than a class imbalance problem per se.

### Calibration is mandatory for models without class weighting
LDA, QDA, GNB, and KNN have large calibration gains (0.19–0.26 F1) because their probability outputs are systematically biased by label imbalance. Calibrated thresholds for these models sit at 0.05–0.30, far below 0.5. These calibrated thresholds should be treated as part of the model specification, not as post-processing.

### Classifier chains add no value at this feature tier
Label co-occurrence analysis on the 2,929 cases reveals meaningful inter-label dependencies: 65% of intraventricular cases also have intraparenchymal, 49% also have subarachnoid; subdural and subarachnoid co-occur in 18–19% of their respective positive cases. Epidural is almost entirely independent (1–3% co-occurrence with any other type), consistent with its traumatic peripheral origin.

Despite this real dependency structure, chains provide no AUC improvement. The likely explanation is that the 269-dim handcrafted feature vector already encodes the spatial and textural signals that drive co-occurrence — ring zone features capture center vs. periphery, GLCM captures homogeneity patterns, and region descriptors capture blob count and location. The chain mechanism would only add value if the chain prediction carried information about the lesion that the features did not already encode. At this feature tier, it does not.

This conclusion may not hold at Tier 3, where models operate on raw image arrays without explicit spatial summaries. A CNN processing the same pixel array for intraparenchymal and intraventricular labels may benefit from chain conditioning in a way that a tree ensemble operating on a pre-computed spatial decomposition does not.

### Random Forest is slow with modest returns
RF took 3.3× longer than XGBoost (322s vs 99s) for a val AUC that is 0.012 lower. The selected `max_depth=30` likely explains both the training time and the structural probability miscalibration that requires large threshold corrections.
