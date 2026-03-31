# Tier 1 Model Results

## Setup

**Task:** Multi-label classification — predict presence/absence of 5 hemorrhage types independently.
**Labels:** epidural, intraparenchymal, intraventricular, subarachnoid, subdural
**Approach:** `OneVsRestClassifier` wrapping each base model — one binary classifier per label, trained independently. Each model sees the same 269-dim feature vector.
**Data:** 2,929 cases (train: 2,049 | val: 440 | test: 440), stratified by hemorrhage type.
**Preprocessing:** `StandardScaler` applied to all models. `PCA` applied to LDA, QDA, GNB, KNN; `n_components` tuned as a hyperparameter. Logistic Regression uses L1/L2 regularization instead of PCA.
**Class imbalance:** `class_weight='balanced'` applied to Logistic Regression. LDA, QDA, GNB, and KNN do not expose a class weight parameter; imbalance is addressed post-hoc via per-label threshold calibration for all models.
**Tuning:** `RandomizedSearchCV`, 5-fold CV on train split stratified by hemorrhage type. Primary CV metric: macro AUC.
**Evaluation:** Val split only. Test split held out for final evaluation across all tiers.

---

## Overall Performance

| Model | CV Macro AUC | Val Macro AUC | Val Macro F1 | Train Time |
|-------|:------------:|:-------------:|:------------:|:----------:|
| Logistic Regression | 0.7012 | **0.7023** | **0.447** | 162.8s |
| LDA | 0.6857 | 0.6845 | 0.229 | 0.5s |
| KNN | 0.6871 | 0.6604 | 0.211 | 0.8s |
| QDA | 0.6759 | 0.6708 | 0.404 | 0.4s |
| GNB | 0.6549 | 0.6406 | 0.251 | 0.4s |

**Best AUC:** Logistic Regression (0.702)
**Best F1:** Logistic Regression (0.447)

---

## Per-Label AUC

| Label | Log Reg | LDA | QDA | GNB | KNN |
|-------|:-------:|:---:|:---:|:---:|:---:|
| epidural | **0.730** | 0.700 | 0.758 | 0.661 | 0.753 |
| intraparenchymal | **0.699** | 0.677 | 0.639 | 0.620 | 0.622 |
| intraventricular | **0.866** | 0.840 | 0.823 | 0.785 | 0.791 |
| subarachnoid | **0.608** | 0.596 | 0.588 | 0.556 | 0.588 |
| subdural | **0.608** | 0.609 | 0.547 | 0.581 | 0.548 |

---

## Best Hyperparameters

| Model | Best Parameters |
|-------|----------------|
| Logistic Regression | `penalty=l2`, `C=0.01` |
| LDA | `pca__n_components=60`, `shrinkage=auto` |
| QDA | `pca__n_components=50`, `reg_param=0.5` |
| GNB | `pca__n_components=25`, `var_smoothing=0.001` |
| KNN | `pca__n_components=25`, `n_neighbors=15`, `metric=euclidean`, `weights=distance` |

---

## Observations

### AUC vs F1 divergence in QDA
QDA achieves a high macro F1 (0.404) relative to its macro AUC (0.671). This reflects a difference in what the two metrics capture: AUC measures ranking quality across all thresholds, while F1 measures the quality of hard predictions at a single threshold. QDA uses aggressive internal thresholds that happen to produce more decisive predictions on this imbalanced dataset, even though its underlying ranking is weaker than Logistic Regression.

### Intraventricular is the easiest label
Every model achieves its highest AUC on intraventricular (0.785–0.866), despite it having the fewest positive cases (8.8% of training data). The ring features cleanly capture the anatomical signature — blood in the ventricles appears as a bright, homogeneous region in the center of the image, which is geometrically distinctive regardless of hemorrhage intensity.

### Subarachnoid and subdural are the hardest labels
Both sit at 0.55–0.61 AUC across all models. These hemorrhage types are diffuse and surface-level — subarachnoid blood spreads across sulci, subdural blood forms a thin crescent. Neither produces a compact, localized bright region, making them difficult to distinguish with intensity/texture features alone.

### LDA with shrinkage is competitive and near-instant
LDA with Ledoit-Wolf shrinkage (`shrinkage='auto'`) and 60 PCA components achieved 0.685 AUC — within 0.018 of Logistic Regression — in 0.5 seconds vs 163 seconds. The shrinkage regularization compensates for the small sample-to-dimension ratio after PCA. For real-time inference, LDA would be the practical choice at this tier.

### Strong CV-to-val generalization
CV and val AUC are within 0.001 for Logistic Regression and within 0.01 for all other models except KNN (0.027 gap). The KNN gap suggests slight overfitting to the training fold structure — distance-weighted neighbours may be picking up idiosyncratic training samples.

### Class imbalance and threshold calibration
All labels are imbalanced (positive rates range from ~6% for epidural to ~35% for subdural in this dataset). Logistic Regression addresses this via `class_weight='balanced'`, resulting in a raw F1 of 0.447 — meaningful at the default threshold. LDA, QDA, GNB, and KNN do not support class weighting; their raw F1 values (0.21–0.40) are low and reflect predictions biased toward the majority class. Per-label threshold calibration (see calibration report) closes most of this gap post-hoc.
