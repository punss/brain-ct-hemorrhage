# Brain CT Hemorrhage Segmentation — Data Definition and Project Guidelines

## 1. Data Structure Overview

Your dataset has **three distinct data layers** that serve different purposes:

### Layer 1: Rendered Images (Classification Input)
```
renders/
├── epidural/
├── intraparenchymal/
├── intraventricular/
├── multi/
├── normal/
├── subarachnoid/
└── subdural/
    ├── brain_bone_window/
    ├── bone_window/
    ├── max_contrast_window/
    └── subdural_window/
```

Each hemorrhage folder contains 4 CT window renderings of the same slices. The folder structure itself provides **classification labels** (the folder name = the hemorrhage type). The `normal/` folder has no hemorrhaging, and `multi/` has multiple hemorrhage types present.

### Layer 2: Classification Labels (`hemorrhage-labels.csv`)
| Column | Description |
|--------|-------------|
| `Image` | Unique image ID (e.g., `ID_00027c277`) |
| `any` | Binary — 1 if any hemorrhage is present |
| `epidural` | Binary — 1 if epidural hemorrhage |
| `intraparenchymal` | Binary — 1 if intraparenchymal hemorrhage |
| `intraventricular` | Binary — 1 if intraventricular hemorrhage |
| `subarachnoid` | Binary — 1 if subarachnoid hemorrhage |
| `subdural` | Binary — 1 if subdural hemorrhage |

**Key observations from the sample:**
- This is a **multi-label** classification problem — a single image can have multiple hemorrhage types (e.g., `subarachnoid=1` AND `subdural=1` simultaneously)
- The `any` column is a convenience flag: 1 if any of the 5 types are present
- Images with `any=0` and all types=0 are **normal** (no hemorrhaging)
- Images with multiple types=1 correspond to the `multi/` folder

### Layer 3: Segmentation Annotations (`Results_*.csv` files)
These are the most complex and most important for the segmentation task.

| Column | Description | Usage Priority |
|--------|-------------|----------------|
| `Case ID` | Unique internal identifier | Join key |
| `Origin` | Original filename (e.g., `ID_00042829c.jpg`) | **Links to image files** |
| `Labeling State` | Annotation quality tier (e.g., "Gold Standard") | Filter for quality |
| `ROI` | **Region of Interest — polygon coordinates for segmentation** | **Primary target for segmentation** |
| `All Annotations` | All labeler annotations (multiple polygons from multiple labelers) | Fallback / ensemble |
| `Number of Annotations` | Count of labeler annotations | Data quality indicator |
| `Number of ROIs` | Count of distinct hemorrhage regions | Multiple regions per image possible |
| `Difficulty` | Area outside Correct Label / area of union of All Labels | Difficulty metric |

**Critical findings about the segmentation data:**

1. **Polygon format:** Annotations are stored as **normalized polygon coordinates** (x, y values between 0 and 1). Example:
   ```json
   [{"x": 0.3125, "y": 0.457}, {"x": 0.3125, "y": 0.455}, ...]
   ```
   These need to be scaled to actual pixel dimensions to create binary masks.

2. **Multiple ROIs per image:** Some images have 2+ hemorrhage regions. These appear as **separate rows** in the CSV with the same `Origin` and `Case ID`. When building masks, you need to combine all ROIs for the same image.

3. **Multiple annotators:** The `All Annotations` column contains polygon lists from multiple qualified labelers. `ROI` appears to be the consensus/gold-standard polygon.

4. **Label priority** (from project description):
   - Use **Correct Label / ROI** if available (Gold Standard)
   - Fall back to **Majority Label** if no Correct Label
   - **Skip images** with neither (no reliable annotation)

5. **One Results CSV per hemorrhage type** — there are 5–6 separate files, one for each subtype.

---

## 2. Data Pipeline: From Raw Files to Model Input

### Step 1: Build a Master Index
```
For each hemorrhage type:
    Parse Results_{type}.csv
    For each row:
        Extract Origin (image ID)
        Extract ROI polygons
        Record hemorrhage type
        Record number of ROIs
        Record labeling quality (Gold Standard, etc.)
```

### Step 2: Convert Polygon Annotations to Binary Masks
```python
# Pseudocode
from PIL import Image, ImageDraw
import json

def polygons_to_mask(polygons_json, image_width, image_height):
    mask = Image.new('L', (image_width, image_height), 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons_json:
        # Scale normalized coords to pixel coords
        points = [(p['x'] * image_width, p['y'] * image_height) for p in polygon]
        draw.polygon(points, fill=255)
    return mask
```

### Step 3: Match Images to Masks
```
For each image in renders/:
    Look up image ID in master index
    If segmentation exists → pair (image, mask) for segmentation training
    If only classification label exists → use for classification training
    If no annotation → skip (or use for normal class if in normal/ folder)
```

### Step 4: Choose CT Window Strategy
**Option B (Recommended):** Stack 3 window types as RGB-like channels for richer input:
```python
# Example: stack 3 windows as 3-channel input
import numpy as np
ch1 = load_grayscale("brain_bone_window/img.png")
ch2 = load_grayscale("subdural_window/img.png")
ch3 = load_grayscale("max_contrast_window/img.png")
input_image = np.stack([ch1, ch2, ch3], axis=-1)  # Shape: (H, W, 3)
```

---

# Data Consistency Requirements

All processed data must conform to the following specifications before any model-specific transformations are applied.

**Image Arrays:** Every image is stored as a numpy array of shape `(512, 512, 3)` with dtype `uint8` and pixel values in `[0, 255]`. The three channels correspond to a fixed window ordering — channel 0 is `brain_bone_window`, channel 1 is `subdural_window`, channel 2 is `max_contrast_window` — and this ordering must never change between samples. If a window rendering is missing for a given image, that image is excluded from the dataset entirely rather than being zero-filled or duplicated from another channel.

**Segmentation Masks:** Masks are stored as numpy arrays of shape `(512, 512)` with dtype `uint8` and exactly two values: `0` (background) and `255` (hemorrhage). Polygon annotations are rasterized at the same `(512, 512)` resolution as the images so that pixel indices correspond directly. Images with multiple ROIs have all regions composited into a single mask. Only images with a `Gold Standard` or `Majority Label` annotation are assigned masks; images lacking both are given no mask file and are flagged `has_segmentation=False` in the metadata.

**Metadata CSV:** One row per image, serving as the single join key between images, masks, and labels. Required columns: `image_id` (unique string), `hemorrhage_type` (categorical string from the folder structure), binary indicator columns `any`, `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural` (int 0/1), `has_segmentation` (bool), `num_rois` (int), `labeling_quality` (string), `difficulty` (float, NaN if no segmentation), and `split` (one of `train`, `val`, `test` — assigned via stratified random split with a fixed seed). No image may appear in more than one split. The `multi` category is used exclusively for images confirmed to have more than one hemorrhage type; the `normal` category is used exclusively for images confirmed to have no hemorrhage.

---

## 3. Model Framework

### Goal

The project employs a deliberately tiered model framework to serve two purposes. First, by progressing from simple statistical methods through classical machine learning to deep neural networks, we create a controlled comparison that isolates the contribution of model complexity — every tier shares the same underlying data, so performance differences are attributable to the model rather than the input. Second, the diversity of approaches within each tier highlights the strengths, weaknesses, and appropriate use cases of individual model families, particularly with respect to the unique challenges of medical image classification and segmentation (high dimensionality, spatial structure, class imbalance, and the need for interpretability in clinical contexts).

The three tiers also trace a natural boundary between **human-engineered features** (Tiers 1–2, which operate on handcrafted descriptors extracted from the images) and **learned features** (Tier 3, which operates on raw pixel arrays). This boundary is itself a central finding of the project: how far can expert feature engineering take us, and at what point does end-to-end representation learning become necessary?

### Tier 1 — Statistical / Probabilistic Models

All Tier 1 models require a preceding feature extraction step. They do not receive raw images; instead, they receive fixed-length feature vectors derived from the images (e.g., intensity histograms, texture descriptors, shape statistics). The quality ceiling of this tier is therefore bounded by the quality of the features chosen.

- Logistic Regression
- Linear and Quadratic Discriminant Analysis
- Gaussian Naive Bayes
- K-Nearest Neighbors (KNN)

### Tier 2 — Traditional Machine Learning

Like Tier 1, these models operate on extracted feature vectors rather than raw pixels. However, they bring substantially greater modeling power — non-linear decision boundaries, ensemble methods, regularization, and kernel-based transformations — allowing them to capture more complex relationships within the same feature space.

- Support Vector Machine (SVM)
- Random Forest
- XGBoost (Gradient Boosted Trees)

### Tier 3 — Neural Networks

Tier 3 models receive the raw `(512, 512, 3)` image arrays directly and learn their own internal representations. Each architecture introduces a different **inductive bias** — a structural assumption about the data — and the comparison between them reveals how architectural choices determine what a network can and cannot learn from spatial image data.

- Dense Neural Network (MLP) — classification only
- Recurrent Neural Network (LSTM) — classification only
- Convolutional Neural Network (CNN) — classification only
- U-Net — segmentation