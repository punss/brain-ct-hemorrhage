# EDA Results & Modeling Implications

## Raw Findings

### Label Distribution (752K scans)
- Extreme class imbalance across all types:
  - `epidural`: **238:1** negative-to-positive (~3,100 positive cases)
  - `subdural`: **15:1** (~47K positive) — most common
  - `intraparenchymal`: 19.8:1
  - `intraventricular`: 27.7:1
  - `subarachnoid`: 20.1:1

### Segmentation Records
- 4,736 total annotated images across 5 types (no intraventricular records)
- Average **2.2 ROIs per image**, ranging up to 12
- Difficulty scores mostly near 0 but with high-difficulty outliers
- Agreement varies widely — some annotations below 0.5

### Channel Intensity (epidural, 1,000 images — all positive)
- `Brain Bone`: mean **35.6**
- `Subdural`: mean **33.3**
- `Max Contrast`: mean **95.6** — very different dynamic range from the other two

### ROI Mask Areas (% of 512×512 image)
| Type | Mean | Median | Max |
|---|---|---|---|
| epidural | 0.79% | 0.45% | 8.6% |
| intraparenchymal | 0.77% | 0.38% | 31.2% |
| subarachnoid | 0.39% | 0.23% | 6.3% |
| subdural | 0.67% | 0.37% | 6.5% |
| multi | 0.69% | 0.33% | 6.8% |

### Data Availability
- Only `epidural` images are currently extracted. Other types need to be unpacked.
- The `renders/` type folders contain **only positive cases**. Negatives come from `renders/normal/`.

---

## Modeling Implications

### 1. Data Construction

**Renders folders only contain positives.** Each type folder holds hemorrhage-positive images only. Negatives come from `renders/normal/` once extracted. The dataset must combine each type folder with normals — use `hemorrhage-labels.csv` as ground truth, not folder name alone.

**Per-channel normalization is required before anything else.** Max Contrast has mean ~95.6 vs ~33–35 for the others. Any distance-based model (KNN, SVM with RBF) or anything treating channels as comparable magnitudes will be dominated by Max Contrast. Normalize each channel independently — z-score per channel per image is standard for medical imaging.

---

### 2. Class Imbalance

Ratios are severe: **238:1 for epidural, 15–28:1 for the others**. A model predicting always-negative gets 99.6% accuracy on epidural. Accuracy is a useless metric across all tiers.

| Tier | Recommended approach |
|---|---|
| Tier 1 (LR, LDA, GNB, KNN) | `class_weight='balanced'` for LR; adjust priors for LDA/QDA/GNB; SMOTE oversampling on feature vectors for KNN |
| Tier 2 (SVM, RF, XGBoost) | `class_weight='balanced'` universally; `scale_pos_weight` for XGBoost |
| Tier 3 classification (MLP, LSTM, CNN) | Weighted cross-entropy loss, inversely proportional to class frequency |
| U-Net segmentation | **Dice loss** (or Dice + BCE combo) — BCE converges to all-background since <0.5% of pixels are hemorrhage |

**Evaluation metrics for all tiers:** AUC-ROC, F1 at optimal threshold, precision-recall curve. Never report accuracy alone.

---

### 3. Hemorrhage Size — Most Critical Finding

Median ROI is **~0.45% of the image** (~1,200 pixels out of 262,144). This is the finding with the most downstream impact.

**For Tier 1 & 2 (feature extraction):**
Global statistics like overall mean intensity will be nearly useless — the hemorrhage signal is diluted across 99.5% non-hemorrhage pixels. Features need to be sensitive to local intensity anomalies:

- **High percentiles** (95th, 99th, max) per channel — fresh blood appears as bright spots; high percentiles capture this even when the mean is unaffected
- **Histogram tails** — right tail of intensity distribution, especially in Max Contrast
- **Spatial grid features** — divide the brain into a 4×4 or 8×8 grid, compute statistics per cell; hemorrhages aren't uniformly distributed (confirmed by spatial heatmap)
- **Texture features (GLCM)** — contrast, dissimilarity, energy; hemorrhage has distinctive texture vs. brain tissue
- **After skull stripping** — features computed only inside the brain mask are far more informative than whole-image statistics; the skull ring inflates high-percentile features and adds noise

The quality ceiling of all Tier 1 and Tier 2 models is entirely bounded by feature quality. Good feature engineering matters more here than which classifier is chosen.

**For Tier 3:**

- **MLP (Dense NN):** Flattens to 786K inputs; hemorrhage is ~1,200 of those. No way to localize spatially — must learn which pixel positions are relevant from scratch. Expect poor performance. This is informative for the controlled comparison: it shows why spatial structure matters.
- **LSTM:** Treats image rows (or patches) as a sequence. No spatial invariance — if hemorrhage appears in row 200 in training but row 210 in test, it may miss it. Expected to be the weakest Tier 3 model, which is useful to demonstrate.
- **CNN:** Right inductive bias — spatial invariance means it can detect hemorrhage wherever it appears. However, with tiny ROIs, global average pooling at the final layer risks averaging away the hemorrhage signal. Keep spatial resolution higher for longer; use max pooling sparingly.
- **U-Net:** Best suited. Skip connections preserve fine spatial detail at the pixel level, which is exactly what sub-1% ROI segmentation requires. Dice loss handles pixel-level imbalance. Most architecture investment should go here.

---

### 4. Multi-label Nature

Images can have multiple hemorrhage types simultaneously. The folder structure gives a primary type, but the labels CSV has the full picture.

- **Tier 1 & 2:** Train one binary classifier per hemorrhage type (one-vs-rest). Simplest, most interpretable, directly answers whether each type is present independently.
- **Tier 3 classification:** Use **sigmoid output per class** (not softmax). Softmax enforces mutual exclusion; sigmoid does not — which is correct for multi-label.
- **U-Net:** Each type needs its own output channel, or train separate U-Nets per type. Given the small annotated dataset per type, separate U-Nets are safer.

---

### 5. Segmentation Annotation Quality

4,676 valid masks across 5 types — small for a U-Net. Key points:

- **Subarachnoid hemorrhages are smallest** (median 0.23%) and likely hardest to segment. **Intraparenchymal can be very large** (up to 31%) — the model must handle a 100× range in hemorrhage size.
- **Average 2.2 ROIs per image** — the mask-building pipeline must composite all ROIs for the same image into one mask, not just use the first row.
- **Filter by agreement** before training U-Net — consider using only annotations with agreement > 0.6 and treating low-agreement cases as unlabeled rather than as noisy labels.

---

### 6. Recommended Transformation Stack

| Transform | Apply? | Reason |
|---|---|---|
| Per-channel z-score normalization | **Yes, always** | Max Contrast dynamic range mismatch; required for KNN/SVM correctness |
| CLAHE | **Yes, for Tier 3** | Enhances local contrast; may help CNN/U-Net detect faint bleeds; less useful for hand-crafted features |
| Skull stripping | **Yes, for Tier 1/2 features** | Removes bone signal that inflates high-percentile features; keeps focus on brain tissue |
| Min-max normalization | **Only if z-score not used** | Redundant with z-score; sensitive to outlier pixels |
| ROI crop + resize | **Only for U-Net augmentation** | Helpful during training; loses spatial context at inference |
| Horizontal flip augmentation | **Yes, for Tier 3** | Doubles training data; brain is nearly bilaterally symmetric |

---

## Summary

Three findings shape everything else:

1. **Hemorrhages occupy <0.5% of pixels** — global features will fail; distance-based models need skull-stripped local features focused on intensity tails and spatial grids.
2. **Max Contrast is on a different scale** — per-channel z-score normalization is required before any model.
3. **Extreme class imbalance** — Dice loss for U-Net, balanced class weights everywhere else, evaluate only on AUC/F1.

The tiered comparison will naturally show Tier 1/2 performance bounded by feature engineering quality, MLP/LSTM struggling with spatial locality, and CNN/U-Net benefiting from their structural inductive biases — which is the core narrative the project is designed to demonstrate.
