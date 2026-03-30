"""
pipeline/test_features.py

Smoke-tests extract_features on a small subset of images from the TFRecord.
Run from the project root:
    python pipeline/test_features.py
"""

import time
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.load_tfrecord import load_tfrecord
from pipeline.extract_features import (
    extract_features,
    extract_features_batch,
    DIM_TOTAL,
    DIM_HIST, DIM_STATS, DIM_RINGS, DIM_REGION, DIM_CROSS, DIM_GLCM,
)

TFRECORD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "tf", "cases.tfrecord.gz")
N_BATCHES  = 3   # 3 batches × 4 images = 12 images
BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Load a small subset
# ---------------------------------------------------------------------------
print(f"Loading {N_BATCHES} batches (batch_size={BATCH_SIZE}) from TFRecord...")
ds = load_tfrecord(TFRECORD_PATH, batch_size=BATCH_SIZE, include_metadata=True)

images_list, ids_list, types_list = [], [], []
for i, (x, y, meta) in enumerate(ds):
    if i >= N_BATCHES:
        break
    images_list.append(x.numpy())
    ids_list.extend([s.decode() for s in meta["id"].numpy()])
    types_list.extend([s.decode() for s in meta["render_directory"].numpy()])

images = np.concatenate(images_list, axis=0)   # (12, 512, 512, 4)
print(f"Loaded {len(images)} images")
print(f"  hemorrhage types: {types_list}")
print(f"  image tensor  — shape: {images.shape}, dtype: {images.dtype}, "
      f"range: [{images.min():.3f}, {images.max():.3f}]")

# ---------------------------------------------------------------------------
# Single-image extraction + timing
# ---------------------------------------------------------------------------
print(f"\n--- Single image extraction ---")
t0 = time.perf_counter()
vec = extract_features(images[0])
t1 = time.perf_counter()

print(f"Output shape : {vec.shape}  (expected ({DIM_TOTAL},))")
print(f"dtype        : {vec.dtype}")
print(f"Time         : {(t1 - t0) * 1000:.1f} ms")
print(f"NaN          : {np.isnan(vec).sum()}")
print(f"Inf          : {np.isinf(vec).sum()}")

# ---------------------------------------------------------------------------
# Batch extraction + timing
# ---------------------------------------------------------------------------
print(f"\n--- Batch extraction ({len(images)} images) ---")
t0 = time.perf_counter()
feat_matrix = extract_features_batch(images)
t1 = time.perf_counter()

print(f"Output shape : {feat_matrix.shape}  (expected ({len(images)}, {DIM_TOTAL}))")
print(f"Time         : {(t1 - t0) * 1000:.1f} ms  "
      f"({(t1 - t0) / len(images) * 1000:.1f} ms/image)")
print(f"NaN          : {np.isnan(feat_matrix).sum()}")
print(f"Inf          : {np.isinf(feat_matrix).sum()}")

# ---------------------------------------------------------------------------
# Per-group breakdown
# ---------------------------------------------------------------------------
print(f"\n--- Feature group breakdown ---")
groups = [
    ("Histograms",   DIM_HIST,   0),
    ("Stats",        DIM_STATS,  DIM_HIST),
    ("Rings",        DIM_RINGS,  DIM_HIST + DIM_STATS),
    ("Regions",      DIM_REGION, DIM_HIST + DIM_STATS + DIM_RINGS),
    ("Cross-ch",     DIM_CROSS,  DIM_HIST + DIM_STATS + DIM_RINGS + DIM_REGION),
    ("GLCM",         DIM_GLCM,   DIM_HIST + DIM_STATS + DIM_RINGS + DIM_REGION + DIM_CROSS),
]
print(f"{'Group':<14} {'Dims':>5}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
print("-" * 58)
for name, dim, start in groups:
    seg = feat_matrix[:, start:start + dim]
    print(f"{name:<14} {dim:>5}  {seg.mean():>8.4f}  {seg.std():>8.4f}  "
          f"{seg.min():>8.4f}  {seg.max():>8.4f}")
print("-" * 58)
print(f"{'Total':<14} {DIM_TOTAL:>5}")

# ---------------------------------------------------------------------------
# Sanity: features differ between images
# ---------------------------------------------------------------------------
print(f"\n--- Inter-image variation (std across images per feature) ---")
feat_std = feat_matrix.std(axis=0)
print(f"Features with std = 0 : {(feat_std == 0).sum()} / {DIM_TOTAL}  (should be low)")
print(f"Mean feature std      : {feat_std.mean():.4f}")
print(f"Min feature std       : {feat_std.min():.6f}")
print(f"Max feature std       : {feat_std.max():.4f}")

print("\nAll checks passed." if np.isnan(feat_matrix).sum() == 0 and
      np.isinf(feat_matrix).sum() == 0 else "\nWARNING: NaN or Inf detected.")
