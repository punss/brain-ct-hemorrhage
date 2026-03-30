"""
pipeline/single.py

Builds pipeline/metadata_epidural.csv — one row per available rendered image,
serving as the master join key between images, masks, and labels.

Scoped to epidural renders only (all 1,694 images, including non-segmented).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths (all relative to Project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RENDERS_DIR = os.path.join(PROJECT_ROOT, "renders", "epidural", "brain_bone_window")
LABELS_CSV = os.path.join(PROJECT_ROOT, "HemorrhageLabels", "hemorrhage-labels.csv")
BHT_CSV = os.path.join(
    PROJECT_ROOT,
    "HemorrhageLabels",
    "Results_Brain Hemorrhage Tracing_2020-09-28_15.21.52.597.csv",
)
EPI_DET_CSV = os.path.join(
    PROJECT_ROOT,
    "HemorrhageLabels",
    "Results_Epidural Hemorrhage Detection_2020-11-16_21.31.26.148.csv",
)
FLAGGED_TXT = os.path.join(PROJECT_ROOT, "HemorrhageLabels", "flagged.txt")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "pipeline", "metadata_epidural.csv")

LABEL_COLS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
QUALITY_STATES = {"Gold Standard", "Labeled"}
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Step 1: Image IDs from renders
# ---------------------------------------------------------------------------
image_ids = sorted(
    f.replace(".jpg", "")
    for f in os.listdir(RENDERS_DIR)
    if f.endswith(".jpg")
)
print(f"Found {len(image_ids)} images in renders/epidural/brain_bone_window/")

# ---------------------------------------------------------------------------
# Step 2: Classification labels
# ---------------------------------------------------------------------------
labels_raw = pd.read_csv(LABELS_CSV, index_col="Image")
labels = labels_raw.loc[labels_raw.index.isin(image_ids), LABEL_COLS]
print(f"Matched {len(labels)} image IDs in hemorrhage-labels.csv")

# ---------------------------------------------------------------------------
# Step 3: Segmentation CSVs — load, filter to quality rows, aggregate by image
# ---------------------------------------------------------------------------

def load_seg_csv(path, polygon_col, source_tag):
    """
    Load a Results CSV, filter to annotatable quality states, and return a
    dict mapping image_id -> {num_rois, labeling_quality, difficulty, polygon_source}.
    Multiple rows per image (multiple ROIs) are collapsed into one record.
    """
    df = pd.read_csv(path)
    df["image_id"] = df["Origin"].str.replace(".jpg", "", regex=False)
    df = df[df["Labeling State"].isin(QUALITY_STATES)].copy()

    agg = (
        df.groupby("image_id")
        .agg(
            num_rois=("Number of ROIs", "max"),
            labeling_quality=("Labeling State", "first"),
            difficulty=("Difficulty", "first"),
        )
        .to_dict("index")
    )
    for v in agg.values():
        v["polygon_source"] = source_tag
    return agg


bht_seg = load_seg_csv(BHT_CSV, polygon_col="ROI", source_tag="BHT")
epi_seg = load_seg_csv(EPI_DET_CSV, polygon_col="Correct Label", source_tag="Detection")
print(f"BHT quality entries: {len(bht_seg)}  |  Epidural Detection quality entries: {len(epi_seg)}")

# ---------------------------------------------------------------------------
# Step 4: Flagged image IDs (treat as no reliable annotation)
# ---------------------------------------------------------------------------
with open(FLAGGED_TXT) as fh:
    flagged = {line.strip().replace(".jpg", "") for line in fh if line.strip()}
print(f"Flagged images: {len(flagged)}")

# ---------------------------------------------------------------------------
# Step 5: Build one row per image
# ---------------------------------------------------------------------------
rows = []
for img_id in image_ids:
    row = {"image_id": img_id, "hemorrhage_type": "epidural"}

    # Classification labels
    if img_id in labels.index:
        for col in LABEL_COLS:
            row[col] = int(labels.at[img_id, col])
    else:
        for col in LABEL_COLS:
            row[col] = np.nan

    # Segmentation — BHT takes priority; fall back to Detection CSV
    seg = None
    if img_id not in flagged:
        seg = bht_seg.get(img_id) or epi_seg.get(img_id)

    if seg:
        row["has_segmentation"] = True
        row["num_rois"] = int(seg["num_rois"])
        row["labeling_quality"] = seg["labeling_quality"]
        row["difficulty"] = float(seg["difficulty"]) if pd.notna(seg["difficulty"]) else np.nan
        row["polygon_source"] = seg["polygon_source"]
    else:
        row["has_segmentation"] = False
        row["num_rois"] = 0
        row["labeling_quality"] = None
        row["difficulty"] = np.nan
        row["polygon_source"] = None

    rows.append(row)

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Step 6: Stratified 70 / 15 / 15 split on has_segmentation
# ---------------------------------------------------------------------------
train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["has_segmentation"], random_state=SPLIT_SEED
)
train, val = train_test_split(
    train_val,
    test_size=0.15 / 0.85,
    stratify=train_val["has_segmentation"],
    random_state=SPLIT_SEED,
)
df.loc[train.index, "split"] = "train"
df.loc[val.index, "split"] = "val"
df.loc[test.index, "split"] = "test"

# ---------------------------------------------------------------------------
# Step 7: Write output
# ---------------------------------------------------------------------------
col_order = (
    ["image_id", "hemorrhage_type"]
    + LABEL_COLS
    + ["has_segmentation", "num_rois", "labeling_quality", "difficulty", "polygon_source", "split"]
)
df = df[col_order]
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Verification summary
# ---------------------------------------------------------------------------
print("\n--- Verification ---")
print(f"Total rows: {len(df)}  (expected 1694)")
print(f"\nSplit counts:\n{df['split'].value_counts().sort_index()}")
print(f"\nhas_segmentation:\n{df['has_segmentation'].value_counts()}")
print(f"\nhas_segmentation by split:")
print(df.groupby("split")["has_segmentation"].value_counts().unstack(fill_value=0))
print(f"\nNull counts in required columns:")
req_cols = ["image_id", "hemorrhage_type"] + LABEL_COLS + ["has_segmentation", "split"]
print(df[req_cols].isnull().sum())
seg_df = df[df["has_segmentation"]]
print(f"\nSegmented images with null difficulty: {seg_df['difficulty'].isna().sum()}")
print(f"polygon_source breakdown:\n{df['polygon_source'].value_counts(dropna=False)}")
