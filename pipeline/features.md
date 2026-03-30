# Feature Engineering & Dimensionality Reduction

## Input

Each case provides a `(512, 512, 4)` float32 image tensor with channels:

| Index | Window | What it emphasizes |
|-------|--------|--------------------|
| 0 | `brain_bone_window` | Bone and soft tissue contrast |
| 1 | `brain_window` | Standard parenchyma |
| 2 | `max_contrast_window` | Lesions vs. normal tissue |
| 3 | `subdural_window` | Subdural space and surface |

All pixel values are in `[0, 1]`. Features are extracted from this tensor and assembled into a single flat vector per case.

---

## Feature Groups

### 1. Per-Channel Intensity Histograms

**What:** Bin each channel into 32 equal-width buckets across `[0, 1]` and record the normalized bin counts.

**Why:** Hemorrhage is hyperdense on CT — it creates a characteristic tail at high intensities. The histogram directly encodes the distribution of bright pixels without requiring spatial reasoning.

**Dims:** 4 channels × 32 bins = **128**

---

### 2. Per-Channel Summary Statistics

**What:** For each channel compute: mean, std, skewness, kurtosis, 25th, 50th, 75th, 95th, 99th percentiles, and the fraction of pixels above a high threshold (0.75).

**Why:** Skewness and the high-percentile values are sensitive to the presence of a bright hemorrhage blob even when it occupies a small fraction of the image. The threshold fraction is a direct "how much bright stuff is here" signal.

**Dims:** 4 channels × 11 statistics = **44**

---

### 3. Spatial Zone Statistics (Annular Rings)

**What:** Divide the image into 4 concentric rings centered on the image midpoint. For each (channel, ring) pair compute mean, max, and std intensity.

```
Ring 0 — center    (r < 0.15 × half-width)   ventricles
Ring 1 — mid-inner (0.15 – 0.40)             deep parenchyma
Ring 2 — mid-outer (0.40 – 0.70)             cortex / sulci
Ring 3 — periphery (0.70 – 1.00)             surface / subdural / epidural space
```

**Why:** Hemorrhage type is largely determined by anatomical location. This decomposition gives Tier 1/2 models a spatial proxy without requiring them to process pixel grids.

| Type | Expected bright zone |
|------|----------------------|
| Intraventricular | Ring 0 |
| Intraparenchymal | Rings 1–2 |
| Subarachnoid | Ring 2 |
| Subdural / Epidural | Ring 3 |

**Dims:** 4 channels × 4 rings × 3 stats = **48**

---

### 4. Thresholded Region Descriptors

**What:** Threshold each channel at 0.75 to produce a binary mask of candidate hemorrhage pixels. Describe the resulting blobs using connected-component analysis:
- Total bright pixel area (fraction of image)
- Number of connected components
- Area of the largest component
- Centroid (x, y) of the largest component
- Distance of centroid from image center (normalized)
- Compactness of the largest component: `area / convex_hull_area`

**Why:** These descriptors answer "is there a blob, how big is it, how many pieces, and where is it" — a compact spatial summary that does not require the model to reason over pixels.

**Dims:** 4 channels × 6 descriptors = **24**

---

### 5. Cross-Channel Difference Statistics

**What:** For each of the 3 non-redundant channel pairs, compute the pixel-wise difference image and record its mean, std, and 95th percentile.

Pairs: `(0,2)`, `(0,3)`, `(2,3)` — i.e. brain_bone vs max_contrast, brain_bone vs subdural, max_contrast vs subdural.

**Why:** The windows were radiologically designed to complement each other. Differences isolate what one window sees that another does not — particularly useful for distinguishing subdural from epidural, where the subdural window provides extra signal.

**Dims:** 3 pairs × 3 stats = **9**

---

### 6. GLCM Texture Features

**What:** Compute the Gray Level Co-occurrence Matrix for each channel at 4 angles (0°, 45°, 90°, 135°) and extract: contrast, energy, homogeneity, correlation (averaged across angles).

**Why:** Hemorrhage tends to be smooth and homogeneous; normal brain parenchyma has more complex texture. GLCM features capture this local structure without spatial reasoning.

**Dims:** 4 channels × 4 properties = **16**

---

### Total Feature Vector

| Group | Dims |
|-------|------|
| Intensity histograms | 128 |
| Summary statistics | 44 |
| Spatial zone statistics | 48 |
| Thresholded region descriptors | 24 |
| Cross-channel differences | 9 |
| GLCM texture | 16 |
| **Total** | **269** |

All features are computed on the train split only for fitting any scalers. A `StandardScaler` is applied before passing to any model (PCA and most classifiers assume zero-mean, unit-variance input).

---

## Dimensionality Reduction

The full 269-dim vector is suitable for Tier 2 models directly. Tier 1 models require reduction first because they make stronger assumptions about the feature space.

### PCA

Fits on the training set only. Components are ordered by explained variance. Applied before KNN, GNB, LDA, QDA, and SVM.

**Target dimensionalities:**

| Model | Target dims | Reason |
|-------|-------------|--------|
| GNB | 30 | Decorrelates features; GNB assumes independence |
| LDA | `n_classes − 1` = 5 | LDA projects to at most `n_classes − 1` discriminant axes regardless; PCA beforehand makes the covariance matrix invertible |
| QDA | 50 | Fits one covariance matrix per class; needs `dims ≪ min(class_size)` to stay non-singular |
| KNN | 30 | Euclidean distance degrades past ~50 dims; fewer is better |
| Logistic Regression | — | L1 penalty used instead; PCA not applied |
| SVM | 80 | Benefits from reduction but can handle more than Tier 1 models |
| Random Forest | — | Internal feature selection; raw vector used |
| XGBoost | — | Internal feature selection; raw vector used |

### L1 Regularization (Logistic Regression only)

Logistic Regression receives the full scaled vector with an L1 penalty (`solver='saga'`). L1 drives irrelevant feature weights to exactly zero, acting as built-in feature selection. The regularization strength `C` is tuned via cross-validation on the train split.

---

## Pipeline Per Model

```
Raw image tensor (512, 512, 4)
        │
        ▼
  Feature extraction  →  269-dim vector
        │
        ▼
  StandardScaler (fit on train only)
        │
   ┌────┴────────────────────────────────────┐
   │                                         │
   ▼                                         ▼
PCA → reduced vector              Raw scaled vector
(GNB, LDA, QDA, KNN, SVM)       (LogReg L1, RF, XGBoost)
   │
   ▼
 Model
```

All steps are wrapped in a `sklearn.pipeline.Pipeline` so that no fitting step sees val or test data.
