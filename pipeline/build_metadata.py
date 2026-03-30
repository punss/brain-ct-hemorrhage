"""
pipeline/build_metadata.py

Builds pipeline/metadata.csv for all 2,929 cases in the TFRecord, covering
five hemorrhage types: epidural, intraparenchymal, multi, subarachnoid, subdural.

Sources:
  - tf/cases_manifest.csv       : master list of TFRecord cases (id, render_directory, seg_label_source)
  - HemorrhageLabels/hemorrhage-labels.csv : classification labels
  - HemorrhageLabels/Results_*_Detection_*.csv : num_rois, labeling_quality, difficulty
  - HemorrhageLabels/flagged.txt

All 2,929 cases have has_segmentation=True (they are in the TFRecord by construction).
Split is stratified on render_directory (hemorrhage type), 70/15/15, seed=42.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MANIFEST_CSV  = os.path.join(PROJECT_ROOT, "tf", "cases_manifest.csv")
LABELS_CSV    = os.path.join(PROJECT_ROOT, "HemorrhageLabels", "hemorrhage-labels.csv")
SEG_CSV_GLOB  = os.path.join(PROJECT_ROOT, "HemorrhageLabels", "Results_*Detection*.csv")
FLAGGED_TXT   = os.path.join(PROJECT_ROOT, "HemorrhageLabels", "flagged.txt")
OUTPUT_CSV    = os.path.join(PROJECT_ROOT, "pipeline", "metadata_all.csv")

LABEL_COLS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Step 1: Load manifest
# ---------------------------------------------------------------------------
manifest = pd.read_csv(MANIFEST_CSV)
print(f"Manifest: {len(manifest)} cases")
print(f"  Types: {manifest['render_directory'].value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# Step 2: Classification labels (filter to manifest IDs only)
# ---------------------------------------------------------------------------
labels_raw = pd.read_csv(LABELS_CSV, index_col="Image")
manifest_ids = set(manifest["id"])
labels = labels_raw.loc[labels_raw.index.isin(manifest_ids), LABEL_COLS]
print(f"Matched {len(labels)} manifest IDs in hemorrhage-labels.csv")

# ---------------------------------------------------------------------------
# Step 3: Aggregate segmentation metadata from Detection CSVs
#
# For each image, pick the "best" row using the same priority as the repo:
#   1. correct > majority  (source_rank)
#   2. higher Agreement
#   3. higher Total Qualified Reads
#   4. higher Total Reads
#   5. earlier CSV name (alphabetical)
#   6. earlier Case ID
# ---------------------------------------------------------------------------

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def is_empty(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.isna() | s.isin(["", "[]", "nan", "None", "null"])

candidate_frames = []
for csv_path in sorted(glob.glob(SEG_CSV_GLOB)):
    df = pd.read_csv(csv_path)
    df["id"] = df["Origin"].astype(str).str.strip().str.replace(r"\.jpg$", "", regex=True)

    # only rows whose image is in the manifest
    df = df[df["id"].isin(manifest_ids)].copy()
    if df.empty:
        continue

    for col in ["Agreement", "Total Qualified Reads", "Total Reads", "Case ID"]:
        if col not in df.columns:
            df[col] = pd.NA

    correct_present = ~is_empty(df["Correct Label"])
    majority_present = ~is_empty(df["Majority Label"])
    usable = df[correct_present | majority_present].copy()
    if usable.empty:
        continue

    usable["source_rank"] = 1  # majority default
    usable.loc[correct_present.loc[usable.index], "source_rank"] = 0  # correct preferred

    usable["_agreement"]   = safe_numeric(usable["Agreement"]).fillna(-1)
    usable["_tqr"]         = safe_numeric(usable["Total Qualified Reads"]).fillna(-1)
    usable["_tr"]          = safe_numeric(usable["Total Reads"]).fillna(-1)
    usable["_csv"]         = os.path.basename(csv_path)
    usable["_case_id_str"] = usable["Case ID"].astype("string").fillna("")

    candidate_frames.append(
        usable[["id", "Labeling State", "Number of ROIs", "Difficulty",
                "source_rank", "_agreement", "_tqr", "_tr", "_csv", "_case_id_str"]]
    )

candidates = pd.concat(candidate_frames, ignore_index=True)

# sort by priority, keep the best row per image
candidates = candidates.sort_values(
    by=["id", "source_rank", "_agreement", "_tqr", "_tr", "_csv", "_case_id_str"],
    ascending=[True, True, False, False, False, True, True],
)
best_seg = (
    candidates.drop_duplicates(subset=["id"], keep="first")
    .set_index("id")
)
print(f"Segmentation metadata resolved for {len(best_seg)} images")

# ---------------------------------------------------------------------------
# Step 4: Flagged IDs
# ---------------------------------------------------------------------------
with open(FLAGGED_TXT) as fh:
    flagged = {line.strip().replace(".jpg", "") for line in fh if line.strip()}
print(f"Flagged image IDs: {len(flagged)}")

# ---------------------------------------------------------------------------
# Step 5: Build one row per manifest case
# ---------------------------------------------------------------------------
rows = []
for _, m_row in manifest.iterrows():
    img_id = m_row["id"]
    row = {
        "image_id":        img_id,
        "hemorrhage_type": m_row["render_directory"],
        "has_segmentation": True,
        "seg_label_source": m_row["seg_label_source"],   # 'correct' or 'majority'
    }

    # classification labels
    if img_id in labels.index:
        for col in LABEL_COLS:
            row[col] = int(labels.at[img_id, col])
    else:
        for col in LABEL_COLS:
            row[col] = np.nan

    # segmentation metadata
    if img_id in best_seg.index:
        seg = best_seg.loc[img_id]
        row["num_rois"]        = int(seg["Number of ROIs"]) if pd.notna(seg["Number of ROIs"]) else np.nan
        row["labeling_quality"] = str(seg["Labeling State"]) if pd.notna(seg["Labeling State"]) else None
        row["difficulty"]      = float(seg["Difficulty"]) if pd.notna(seg["Difficulty"]) else np.nan
    else:
        row["num_rois"]        = np.nan
        row["labeling_quality"] = None
        row["difficulty"]      = np.nan

    rows.append(row)

df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Step 6: Stratified 70/15/15 split on render_directory
# ---------------------------------------------------------------------------
train_val, test = train_test_split(
    df, test_size=0.15, stratify=df["hemorrhage_type"], random_state=SPLIT_SEED
)
train, val = train_test_split(
    train_val, test_size=0.15 / 0.85,
    stratify=train_val["hemorrhage_type"], random_state=SPLIT_SEED
)
df.loc[train.index, "split"] = "train"
df.loc[val.index,  "split"] = "val"
df.loc[test.index, "split"] = "test"

# ---------------------------------------------------------------------------
# Step 7: Write output
# ---------------------------------------------------------------------------
col_order = (
    ["image_id", "hemorrhage_type"]
    + LABEL_COLS
    + ["has_segmentation", "seg_label_source", "num_rois",
       "labeling_quality", "difficulty", "split"]
)
df = df[col_order]
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nWrote {len(df)} rows to {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
print("\n--- Verification ---")
print(f"Total rows: {len(df)}  (expected 2929)")
print(f"\nSplit counts:\n{df['split'].value_counts().sort_index()}")
print(f"\nRows by type and split:")
print(df.groupby(["hemorrhage_type", "split"]).size().unstack(fill_value=0))
print(f"\nNull counts in required columns:")
req_cols = ["image_id", "hemorrhage_type"] + LABEL_COLS + ["has_segmentation", "split"]
print(df[req_cols].isnull().sum())
print(f"\ndifficulty null: {df['difficulty'].isna().sum()} / {len(df)}")
print(f"\nseg_label_source:\n{df['seg_label_source'].value_counts()}")
print(f"\nlabeling_quality:\n{df['labeling_quality'].value_counts(dropna=False)}")
