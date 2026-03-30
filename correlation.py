"""
Channel Correlation Analysis for Brain CT Hemorrhage Segmentation
=================================================================
Loads the 4 CT window renderings separately, computes pairwise Pearson
correlation across a sample of images, and recommends which channel to
drop (the one most redundant with the others).

The 4 windows per the data spec:
  - brain_bone_window
  - bone_window
  - max_contrast_window
  - subdural_window

We keep 3 to form an RGB-like input tensor of shape (512, 512, 3).
"""

import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from itertools import combinations

# ──────────────────────────────────────────────────────────────────────
# CONFIG — adjust these paths to match your directory layout
# ──────────────────────────────────────────────────────────────────────
RENDERS_ROOT = "/Users/hridikpunukollu/Documents/Acads/MATH 7243/Project/renders"  # top-level renders directory
WINDOW_NAMES = [
    "brain_bone_window",
    "brain_window",
    "max_contrast_window",
    "subdural_window",
]
MAX_SAMPLES = 500        # max images to sample (set None for all)
RANDOM_SEED = 42
IMG_SIZE = (512, 512)

# ──────────────────────────────────────────────────────────────────────
# STEP 1: Discover images that have all 4 window renderings
# ──────────────────────────────────────────────────────────────────────
def discover_images(renders_root: str, window_names: list[str]) -> list[dict]:
    """
    Walk the hemorrhage-type folders and collect image paths grouped by
    image ID, keeping only images where all 4 windows exist.
    
    Returns a list of dicts: {image_id, hemorrhage_type, windows: {name: path}}
    """
    hemorrhage_types = [
        d for d in os.listdir(renders_root)
        if os.path.isdir(os.path.join(renders_root, d))
    ]

    records = []
    for htype in hemorrhage_types:
        htype_dir = os.path.join(renders_root, htype)

        # Collect filenames from the first available window subfolder
        first_window_dir = os.path.join(htype_dir, window_names[0])
        if not os.path.isdir(first_window_dir):
            continue

        # Get all image filenames from the first window
        image_files = [
            f for f in os.listdir(first_window_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        for fname in image_files:
            # Check all 4 windows exist for this image
            window_paths = {}
            all_exist = True
            for wname in window_names:
                wpath = os.path.join(htype_dir, wname, fname)
                if os.path.isfile(wpath):
                    window_paths[wname] = wpath
                else:
                    all_exist = False
                    break

            if all_exist:
                image_id = os.path.splitext(fname)[0]
                records.append({
                    "image_id": image_id,
                    "hemorrhage_type": htype,
                    "windows": window_paths,
                })

    return records


# ──────────────────────────────────────────────────────────────────────
# STEP 2: Load a single channel as a flattened grayscale vector
# ──────────────────────────────────────────────────────────────────────
def load_grayscale_flat(path: str, size: tuple = IMG_SIZE) -> np.ndarray:
    """Load image, convert to grayscale, resize, return as flat float32 array."""
    img = Image.open(path).convert("L").resize(size)
    return np.asarray(img, dtype=np.float32).ravel()


# ──────────────────────────────────────────────────────────────────────
# STEP 3: Compute pairwise Pearson correlation across sampled images
# ──────────────────────────────────────────────────────────────────────
def compute_channel_correlations(
    records: list[dict],
    window_names: list[str],
    max_samples,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each sampled image, compute the per-image Pearson r between every
    pair of windows.  Return a DataFrame of mean correlations.
    """
    rng = np.random.default_rng(seed)
    if max_samples and len(records) > max_samples:
        indices = rng.choice(len(records), size=max_samples, replace=False)
        records = [records[i] for i in indices]

    n = len(records)
    pairs = list(combinations(range(len(window_names)), 2))
    pair_corrs = {pair: [] for pair in pairs}

    print(f"Computing correlations over {n} images …")
    for idx, rec in enumerate(records):
        if (idx + 1) % 100 == 0:
            print(f"  processed {idx + 1}/{n}")

        # Load all 4 channels for this image
        channels = []
        for wname in window_names:
            channels.append(load_grayscale_flat(rec["windows"][wname]))

        # Pairwise Pearson r
        for (i, j) in pairs:
            r = np.corrcoef(channels[i], channels[j])[0, 1]
            pair_corrs[(i, j)].append(r)

    # Build a symmetric correlation matrix
    corr_matrix = np.eye(len(window_names))
    for (i, j), values in pair_corrs.items():
        mean_r = np.nanmean(values)
        corr_matrix[i, j] = mean_r
        corr_matrix[j, i] = mean_r

    corr_df = pd.DataFrame(corr_matrix, index=window_names, columns=window_names)
    return corr_df


# ──────────────────────────────────────────────────────────────────────
# STEP 4: Recommend which channel to drop
# ──────────────────────────────────────────────────────────────────────
def recommend_drop(corr_df: pd.DataFrame) -> str:
    """
    The channel to drop is the one that is *most redundant* — i.e., has
    the highest average correlation with the other three channels.
    Dropping it loses the least unique information.
    """
    window_names = corr_df.columns.tolist()
    avg_corr = {}
    for w in window_names:
        # Mean correlation with the OTHER three channels
        others = [c for c in window_names if c != w]
        avg_corr[w] = corr_df.loc[w, others].mean()

    avg_series = pd.Series(avg_corr).sort_values(ascending=False)
    return avg_series


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Discovering images with all 4 windows …")
    records = discover_images(RENDERS_ROOT, WINDOW_NAMES)
    print(f"Found {len(records)} images with all 4 windows.\n")

    if len(records) == 0:
        print("ERROR: No images found. Check RENDERS_ROOT path and folder structure.")
        raise SystemExit(1)

    corr_df = compute_channel_correlations(
        records, WINDOW_NAMES, max_samples=MAX_SAMPLES, seed=RANDOM_SEED
    )

    print("\n" + "=" * 60)
    print("PAIRWISE CORRELATION MATRIX (mean Pearson r across images)")
    print("=" * 60)
    print(corr_df.round(4).to_string())

    avg_corr = recommend_drop(corr_df)
    print("\n" + "=" * 60)
    print("MEAN CORRELATION WITH OTHER 3 CHANNELS (higher = more redundant)")
    print("=" * 60)
    for name, val in avg_corr.items():
        print(f"  {name:25s}  {val:.4f}")

    drop = avg_corr.idxmax()
    print(f"\n>>> RECOMMENDATION: Drop '{drop}' — it is most correlated with")
    print(f"    the remaining channels and contributes the least unique info.")
    print(f"\n>>> KEEP: {[w for w in WINDOW_NAMES if w != drop]}")
