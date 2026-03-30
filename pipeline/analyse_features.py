"""
pipeline/analyse_features.py

Feature importance analysis for Tier 1 models.

Primary analysis: Logistic Regression coefficients (highest AUC model,
most interpretable via signed weights).
Secondary analysis: LDA coefficients.

Outputs:
  pipeline/results/feature_importance_lr.json  — top features per label
  pipeline/results/group_importance.png        — feature group heatmap
  pipeline/results/top_features_{label}.png    — per-label bar charts
  pipeline/results/lr_coef_heatmap.png         — full coefficient heatmap
"""

import os, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
RESULTS_DIR  = os.path.join(PIPELINE_DIR, "results")

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
CHANNELS   = ["brain_bone", "brain_win", "max_contrast", "subdural_win"]
CH         = ["bb", "bw", "mc", "sd"]   # short names

# ---------------------------------------------------------------------------
# Build 269 feature names (must match concatenation order in extract_features.py)
# ---------------------------------------------------------------------------
feature_names = []
feature_groups = []   # group label per feature

# Group 1: Histograms — 4 ch × 32 bins = 128
for c in CH:
    for b in range(32):
        feature_names.append(f"{c}_hist_b{b:02d}")
        feature_groups.append("Histogram")

# Group 2: Summary stats — 4 ch × 11 = 44
STAT_NAMES = ["mean", "std", "skew", "kurt", "p5", "p25", "p50", "p75", "p95", "p99", "frac_hi"]
for c in CH:
    for s in STAT_NAMES:
        feature_names.append(f"{c}_{s}")
        feature_groups.append("Stats")

# Group 3: Ring stats — 4 ch × 4 rings × 3 = 48
for c in CH:
    for r in range(4):
        for s in ["mean", "max", "std"]:
            feature_names.append(f"{c}_ring{r}_{s}")
            feature_groups.append("Rings")

# Group 4: Region descriptors — 4 ch × 6 = 24
REGION_NAMES = ["total_area", "n_comps", "largest_area", "cx_norm", "cy_norm", "compactness"]
for c in CH:
    for rd in REGION_NAMES:
        feature_names.append(f"{c}_{rd}")
        feature_groups.append("Regions")

# Group 5: Cross-channel diffs — 3 pairs × 3 = 9
for pair in [("bb","mc"), ("bb","sd"), ("mc","sd")]:
    for s in ["mean", "std", "p95"]:
        feature_names.append(f"diff_{pair[0]}_{pair[1]}_{s}")
        feature_groups.append("Cross-ch")

# Group 6: GLCM — 4 ch × 4 = 16
GLCM_PROPS = ["contrast", "energy", "homogeneity", "correlation"]
for c in CH:
    for p in GLCM_PROPS:
        feature_names.append(f"{c}_glcm_{p}")
        feature_groups.append("GLCM")

assert len(feature_names) == 269, f"Expected 269, got {len(feature_names)}"

feature_names  = np.array(feature_names)
feature_groups = np.array(feature_groups)
GROUP_ORDER    = ["Histogram", "Stats", "Rings", "Regions", "Cross-ch", "GLCM"]

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
lr_model  = joblib.load(os.path.join(PIPELINE_DIR, "models", "tier1", "logistic_regression.pkl"))
lda_model = joblib.load(os.path.join(PIPELINE_DIR, "models", "tier1", "lda.pkl"))

# ---------------------------------------------------------------------------
# Extract LR coefficients
# LR pipeline: scaler → OvR(LR)
# After scaler, features are standardised — coefficients are comparable
# OvR has one estimator per label; each has coef_ shape (1, n_features_after_scaler)
# No PCA in LR pipeline so coef_ aligns directly with feature_names
# ---------------------------------------------------------------------------
lr_ovr = lr_model.named_steps["clf"]
coef_matrix = np.vstack([est.coef_[0] for est in lr_ovr.estimators_])  # (5, 269)

print("LR coefficient matrix shape:", coef_matrix.shape)

# ---------------------------------------------------------------------------
# Extract LDA coefficients
# LDA pipeline: scaler → PCA → OvR(LDA)
# coef_ is in PCA space — map back to original feature space via PCA components
# ---------------------------------------------------------------------------
pca_lda    = lda_model.named_steps["pca"]
lda_ovr    = lda_model.named_steps["clf"]
lda_coefs_pca = np.vstack([est.coef_[0] for est in lda_ovr.estimators_])  # (5, n_components)
lda_coef_matrix = lda_coefs_pca @ pca_lda.components_   # (5, 269) — back in feature space

# ---------------------------------------------------------------------------
# 1. Top features per label (LR)
# ---------------------------------------------------------------------------
TOP_N = 15
importance_data = {}

for i, label in enumerate(LABEL_COLS):
    coef = coef_matrix[i]
    top_pos_idx = np.argsort(coef)[-TOP_N:][::-1]
    top_neg_idx = np.argsort(coef)[:TOP_N]

    importance_data[label] = {
        "top_positive": [
            {"feature": feature_names[j], "group": feature_groups[j], "coef": round(float(coef[j]), 4)}
            for j in top_pos_idx
        ],
        "top_negative": [
            {"feature": feature_names[j], "group": feature_groups[j], "coef": round(float(coef[j]), 4)}
            for j in top_neg_idx
        ],
    }

with open(os.path.join(RESULTS_DIR, "feature_importance_lr.json"), "w") as f:
    json.dump(importance_data, f, indent=2)
print("Saved feature_importance_lr.json")

# ---------------------------------------------------------------------------
# 2. Feature group importance heatmap (LR)
# Mean |coef| per (label, group) — shows which groups drive each label
# ---------------------------------------------------------------------------
group_imp = pd.DataFrame(index=LABEL_COLS, columns=GROUP_ORDER, dtype=float)
for i, label in enumerate(LABEL_COLS):
    for grp in GROUP_ORDER:
        mask = feature_groups == grp
        group_imp.loc[label, grp] = np.abs(coef_matrix[i, mask]).mean()

# Normalise per label so colours show relative group importance
group_imp_norm = group_imp.div(group_imp.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.heatmap(
    group_imp.astype(float), ax=axes[0],
    annot=True, fmt=".4f", cmap="YlOrRd",
    linewidths=0.5, cbar_kws={"label": "Mean |coef|"},
)
axes[0].set_title("Mean |LR coefficient| by feature group and label")
axes[0].set_xlabel("Feature group")
axes[0].set_ylabel("Label")

sns.heatmap(
    group_imp_norm.astype(float), ax=axes[1],
    annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, cbar_kws={"label": "Relative importance"},
)
axes[1].set_title("Relative group importance (row-normalised)")
axes[1].set_xlabel("Feature group")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "group_importance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved group_importance.png")

# ---------------------------------------------------------------------------
# 3. Per-label top-feature bar charts (LR, top 12 positive + negative)
# ---------------------------------------------------------------------------
TOP_BAR = 12
fig, axes = plt.subplots(1, len(LABEL_COLS), figsize=(22, 6), sharey=False)

GROUP_COLORS = {
    "Histogram": "#4C72B0", "Stats": "#DD8452", "Rings": "#55A868",
    "Regions": "#C44E52", "Cross-ch": "#8172B3", "GLCM": "#937860",
}

for ax, label in zip(axes, LABEL_COLS):
    coef = coef_matrix[LABEL_COLS.index(label)]
    # top positives and negatives by abs magnitude
    top_idx = np.argsort(np.abs(coef))[-TOP_BAR:][::-1]
    vals    = coef[top_idx]
    names   = [n.replace("_", "\n") for n in feature_names[top_idx]]
    colors  = [GROUP_COLORS[feature_groups[j]] for j in top_idx]

    bars = ax.barh(range(TOP_BAR), vals[::-1], color=colors[::-1])
    ax.set_yticks(range(TOP_BAR))
    ax.set_yticklabels([n for n in names[::-1]], fontsize=6.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("LR coefficient")

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=g) for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_elements, loc="lower center", ncol=6,
           bbox_to_anchor=(0.5, -0.02), fontsize=8, title="Feature group")

plt.suptitle("Top LR features per label (by |coefficient|)", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "top_features_per_label.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved top_features_per_label.png")

# ---------------------------------------------------------------------------
# 4. Full coefficient heatmap (LR vs LDA side by side, grouped features)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(20, 5))

for ax, mat, title in [
    (axes[0], coef_matrix,     "Logistic Regression"),
    (axes[1], lda_coef_matrix, "LDA (mapped to feature space)"),
]:
    # Normalise each row to [-1, 1] for visual clarity
    row_max = np.abs(mat).max(axis=1, keepdims=True)
    mat_norm = mat / np.where(row_max == 0, 1, row_max)

    im = ax.imshow(mat_norm, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_yticks(range(len(LABEL_COLS)))
    ax.set_yticklabels(LABEL_COLS)
    ax.set_xlabel("Feature index (0–268)")
    ax.set_title(title)

    # Group boundary lines
    boundaries = [0, 128, 172, 220, 244, 253, 269]
    labels_grp = ["Hist", "Stats", "Rings", "Regions", "X-ch", "GLCM"]
    for b in boundaries[1:-1]:
        ax.axvline(b - 0.5, color="white", linewidth=1.2, alpha=0.8)
    for k in range(len(labels_grp)):
        mid = (boundaries[k] + boundaries[k+1]) / 2
        ax.text(mid, len(LABEL_COLS) - 0.2, labels_grp[k],
                ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Normalised coefficient")

plt.suptitle("Coefficient patterns across all features (row-normalised per label)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "lr_lda_coef_heatmap.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved lr_lda_coef_heatmap.png")

# ---------------------------------------------------------------------------
# 5. Print narrative summary
# ---------------------------------------------------------------------------
print("\n" + "="*65)
print("FEATURE IMPORTANCE SUMMARY (Logistic Regression)")
print("="*65)
for label in LABEL_COLS:
    d = importance_data[label]
    print(f"\n{label.upper()}")
    print(f"  Top positive (predicts presence):")
    for f in d["top_positive"][:5]:
        print(f"    {f['feature']:<35} coef={f['coef']:+.4f}  [{f['group']}]")
    print(f"  Top negative (predicts absence):")
    for f in d["top_negative"][:5]:
        print(f"    {f['feature']:<35} coef={f['coef']:+.4f}  [{f['group']}]")

print("\n" + "="*65)
print("GROUP-LEVEL RELATIVE IMPORTANCE (LR, row-normalised)")
print("="*65)
print(group_imp_norm.astype(float).round(3).to_string())
