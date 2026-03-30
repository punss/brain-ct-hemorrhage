"""
pipeline/extract_features.py

Extracts a 269-dimensional feature vector from a (512, 512, 4) CT image tensor.
Feature groups and rationale are documented in pipeline/features.md.
"""

import numpy as np
from scipy import stats as sp_stats
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_SIZE  = 512
HALF        = IMAGE_SIZE // 2
THRESHOLD   = 0.75
GLCM_LEVELS = 64
HIST_BINS   = 32
N_CHANNELS  = 4
N_RINGS     = 4
CROSS_PAIRS = [(0, 2), (0, 3), (2, 3)]

# Feature dimensionalities
DIM_HIST   = N_CHANNELS * HIST_BINS       # 128
DIM_STATS  = N_CHANNELS * 11              #  44
DIM_RINGS  = N_CHANNELS * N_RINGS * 3    #  48
DIM_REGION = N_CHANNELS * 6              #  24
DIM_CROSS  = len(CROSS_PAIRS) * 3        #   9
DIM_GLCM   = N_CHANNELS * 4             #  16
DIM_TOTAL  = DIM_HIST + DIM_STATS + DIM_RINGS + DIM_REGION + DIM_CROSS + DIM_GLCM  # 269

# Precompute ring masks once at import time
_y, _x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
_dist = np.sqrt((_x - HALF) ** 2 + (_y - HALF) ** 2)
_RING_MASKS = [
    _dist < 0.15 * HALF,
    (_dist >= 0.15 * HALF) & (_dist < 0.40 * HALF),
    (_dist >= 0.40 * HALF) & (_dist < 0.70 * HALF),
    _dist >= 0.70 * HALF,
]

# ---------------------------------------------------------------------------
# Per-channel extractors
# ---------------------------------------------------------------------------

def _histograms(ch: np.ndarray) -> np.ndarray:
    """32-bin normalized histogram over [0, 1]. Shape: (32,)"""
    counts, _ = np.histogram(ch, bins=HIST_BINS, range=(0.0, 1.0))
    return (counts / ch.size).astype(np.float32)


def _summary_stats(ch: np.ndarray) -> np.ndarray:
    """mean, std, skew, kurt, p5/25/50/75/95/99, fraction above threshold. Shape: (11,)"""
    flat = ch.ravel()
    return np.array([
        flat.mean(),
        flat.std(),
        sp_stats.skew(flat),
        sp_stats.kurtosis(flat),
        np.percentile(flat,  5),
        np.percentile(flat, 25),
        np.percentile(flat, 50),
        np.percentile(flat, 75),
        np.percentile(flat, 95),
        np.percentile(flat, 99),
        np.mean(flat > THRESHOLD),
    ], dtype=np.float32)


def _ring_stats(ch: np.ndarray) -> np.ndarray:
    """Mean, max, std intensity per concentric ring. Shape: (12,)"""
    feats = []
    for mask in _RING_MASKS:
        vals = ch[mask]
        if vals.size == 0:
            feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([vals.mean(), vals.max(), vals.std()])
    return np.array(feats, dtype=np.float32)


def _region_descriptors(ch: np.ndarray) -> np.ndarray:
    """
    Thresholded bright-region shape/location descriptors. Shape: (6,)
      [total_bright_area, num_components, largest_area,
       centroid_x_norm, centroid_y_norm, compactness]
    """
    bright = (ch > THRESHOLD).astype(np.uint8)
    total_area = float(bright.sum()) / (IMAGE_SIZE * IMAGE_SIZE)

    labeled  = label(bright)
    regions  = regionprops(labeled)
    n_comps  = float(len(regions))

    if not regions:
        return np.zeros(6, dtype=np.float32)

    largest      = max(regions, key=lambda r: r.area)
    largest_area = float(largest.area) / (IMAGE_SIZE * IMAGE_SIZE)
    cy, cx       = largest.centroid
    cx_norm      = (cx - HALF) / HALF
    cy_norm      = (cy - HALF) / HALF
    convex       = float(largest.area_convex)
    compactness  = float(largest.area) / convex if convex > 0 else 0.0

    return np.array(
        [total_area, n_comps, largest_area, cx_norm, cy_norm, compactness],
        dtype=np.float32,
    )


def _glcm_features(ch: np.ndarray) -> np.ndarray:
    """GLCM contrast, energy, homogeneity, correlation averaged over 4 angles. Shape: (4,)"""
    quantized = (ch * (GLCM_LEVELS - 1)).clip(0, GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    return np.array(
        [float(graycoprops(glcm, p).mean())
         for p in ("contrast", "energy", "homogeneity", "correlation")],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Cross-channel extractor
# ---------------------------------------------------------------------------

def _cross_channel_diffs(image: np.ndarray) -> np.ndarray:
    """Mean, std, p95 of pixel-wise difference for 3 channel pairs. Shape: (9,)"""
    feats = []
    for i, j in CROSS_PAIRS:
        diff = image[:, :, i] - image[:, :, j]
        feats.extend([diff.mean(), diff.std(), float(np.percentile(diff, 95))])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Extract a 269-dimensional feature vector from one CT image.

    Parameters
    ----------
    image : np.ndarray  shape (512, 512, 4), float32, values in [0, 1]

    Returns
    -------
    np.ndarray  shape (269,), float32
    """
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, N_CHANNELS):
        raise ValueError(f"Expected (512, 512, 4), got {image.shape}")

    hist_parts   = []
    stats_parts  = []
    ring_parts   = []
    region_parts = []
    glcm_parts   = []

    for c in range(N_CHANNELS):
        ch = image[:, :, c]
        hist_parts.append(_histograms(ch))
        stats_parts.append(_summary_stats(ch))
        ring_parts.append(_ring_stats(ch))
        region_parts.append(_region_descriptors(ch))
        glcm_parts.append(_glcm_features(ch))

    vec = np.concatenate(
        hist_parts + stats_parts + ring_parts + region_parts
        + [_cross_channel_diffs(image)]
        + glcm_parts
    )

    assert vec.shape == (DIM_TOTAL,), f"Bug: expected ({DIM_TOTAL},), got {vec.shape}"
    return vec


def extract_features_batch(images: np.ndarray) -> np.ndarray:
    """
    Extract features for a batch of images.

    Parameters
    ----------
    images : np.ndarray  shape (N, 512, 512, 4)

    Returns
    -------
    np.ndarray  shape (N, 269)
    """
    return np.stack([extract_features(images[i]) for i in range(len(images))])
