"""
pipeline/evaluate_test.py

Final evaluation of all Tier 1 + Tier 2 OvR models on the held-out test split.

Important: decision thresholds are taken from calibration_summary.json, which
was derived on the val split. They are NOT re-optimised on the test set.

Metrics reported per model:
  - Macro AUC-ROC        (threshold-free ranking quality)
  - Macro AUPRC          (avg. precision; better than AUC-ROC for imbalanced labels)
  - Per-label AUC-ROC
  - Per-label AUPRC
  - Macro F1             (at val-calibrated thresholds)
  - Per-label F1, Precision, Recall

Outputs:
  pipeline/results/test_summary.json
  pipeline/results/test_results.md
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
RESULTS_DIR  = os.path.join(PIPELINE_DIR, "results")
MODELS_DIR   = os.path.join(PIPELINE_DIR, "models")

# ---------------------------------------------------------------------------
# Data — test split only
# ---------------------------------------------------------------------------
X     = np.load(os.path.join(PIPELINE_DIR, "X.npy"))
y_all = np.load(os.path.join(PIPELINE_DIR, "y_cls.npy"))
meta  = pd.read_csv(os.path.join(PIPELINE_DIR, "metadata_all.csv"))

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
y = y_all[:, 1:].astype(int)

test_mask = (meta["split"] == "test").values
X_test, y_test = X[test_mask], y[test_mask]

print(f"Test set: {X_test.shape[0]} cases, {X_test.shape[1]} features")
print(f"Label positive rates on test set:")
for i, label in enumerate(LABEL_COLS):
    rate = y_test[:, i].mean() * 100
    print(f"  {label:<22}: {rate:.1f}%")
print()

# ---------------------------------------------------------------------------
# Load calibrated thresholds (from val set — not re-optimised on test)
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "calibration_summary.json")) as f:
    cal = json.load(f)

# ---------------------------------------------------------------------------
# Load models
# ---------------------------------------------------------------------------
model_paths = {}
for subdir in ["tier1", "tier2"]:
    d = os.path.join(MODELS_DIR, subdir)
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "")
            model_paths[name] = os.path.join(d, fname)

# Tier ordering for display
TIER = {
    "logistic_regression": "T1", "lda": "T1", "qda": "T1", "gnb": "T1", "knn": "T1",
    "svm": "T2", "random_forest": "T2", "xgboost": "T2",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def apply_thresholds(y_prob, thresholds):
    y_pred = np.zeros(y_prob.shape, dtype=int)
    for i, label in enumerate(LABEL_COLS):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[label]).astype(int)
    return y_pred

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
print("=" * 70)
print("TEST SET EVALUATION — ALL TIER 1 + TIER 2 OvR MODELS")
print("Thresholds: val-set calibrated (not re-optimised on test)")
print("=" * 70)

results = {}

for name, path in model_paths.items():
    model = joblib.load(path)
    y_prob = model.predict_proba(X_test)

    thresholds = cal[name]["thresholds"]
    y_pred_cal = apply_thresholds(y_prob, thresholds)

    # --- Macro metrics ---
    macro_auc  = round(float(roc_auc_score(y_test, y_prob, average="macro")), 4)
    macro_auprc = round(float(average_precision_score(y_test, y_prob, average="macro")), 4)
    macro_f1   = round(float(f1_score(y_test, y_pred_cal, average="macro", zero_division=0)), 4)
    macro_prec = round(float(precision_score(y_test, y_pred_cal, average="macro", zero_division=0)), 4)
    macro_rec  = round(float(recall_score(y_test, y_pred_cal, average="macro", zero_division=0)), 4)

    # --- Per-label metrics ---
    per_label = {}
    for i, label in enumerate(LABEL_COLS):
        try:
            auc  = round(float(roc_auc_score(y_test[:, i], y_prob[:, i])), 4)
        except ValueError:
            auc  = None
        auprc = round(float(average_precision_score(y_test[:, i], y_prob[:, i])), 4)
        f1    = round(float(f1_score(y_test[:, i], y_pred_cal[:, i], zero_division=0)), 4)
        prec  = round(float(precision_score(y_test[:, i], y_pred_cal[:, i], zero_division=0)), 4)
        rec   = round(float(recall_score(y_test[:, i], y_pred_cal[:, i], zero_division=0)), 4)
        per_label[label] = {
            "auc": auc, "auprc": auprc,
            "f1": f1, "precision": prec, "recall": rec,
            "threshold": thresholds[label],
        }

    tier = TIER.get(name, "?")
    print(f"\n[{tier}] {name}")
    print(f"  Macro AUC: {macro_auc:.4f}  |  Macro AUPRC: {macro_auprc:.4f}  |  "
          f"Macro F1: {macro_f1:.4f}  |  Prec: {macro_prec:.4f}  |  Rec: {macro_rec:.4f}")
    print(f"  {'Label':<22} {'AUC':>6} {'AUPRC':>7} {'F1':>6} {'Prec':>6} {'Rec':>6} {'Thresh':>7}")
    print(f"  {'-'*62}")
    for label, m in per_label.items():
        print(f"  {label:<22} {m['auc']:>6.4f} {m['auprc']:>7.4f} "
              f"{m['f1']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} {m['threshold']:>7.2f}")

    results[name] = {
        "tier": tier,
        "macro_auc":   macro_auc,
        "macro_auprc": macro_auprc,
        "macro_f1":    macro_f1,
        "macro_precision": macro_prec,
        "macro_recall":    macro_rec,
        "per_label":   per_label,
    }

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'MODEL':<24} {'AUC':>7} {'AUPRC':>7} {'F1(cal)':>8} {'Prec':>6} {'Rec':>6}")
print("-" * 70)
for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_auc"]):
    tier = r["tier"]
    print(f"[{tier}] {name:<22} {r['macro_auc']:>7.4f} {r['macro_auprc']:>7.4f} "
          f"{r['macro_f1']:>8.4f} {r['macro_precision']:>6.4f} {r['macro_recall']:>6.4f}")
print("=" * 70)

# ---------------------------------------------------------------------------
# Save JSON
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "test_summary.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved test_summary.json")

# ---------------------------------------------------------------------------
# Write markdown report
# ---------------------------------------------------------------------------
lines = []
lines.append("# Test Set Evaluation — Tier 1 + Tier 2\n")
lines.append("## Setup\n")
lines.append(f"**Split:** Held-out test set — {X_test.shape[0]} cases, never seen during training or hyperparameter search.")
lines.append("**Thresholds:** Val-set calibrated thresholds from `calibration_summary.json`. "
             "These were not re-optimised on the test set — doing so would be data leakage.")
lines.append("**Metrics:**")
lines.append("- **Macro AUC-ROC** — threshold-free ranking quality, equal weight per label")
lines.append("- **Macro AUPRC** — area under precision-recall curve; more informative than AUC-ROC for imbalanced labels")
lines.append("- **Macro F1 / Precision / Recall** — hard-prediction quality at val-calibrated thresholds")
lines.append("")

lines.append("**Test label positive rates:**\n")
lines.append("| Label | Positive rate |")
lines.append("|-------|:-------------:|")
for i, label in enumerate(LABEL_COLS):
    lines.append(f"| {label} | {y_test[:, i].mean()*100:.1f}% |")
lines.append("")

lines.append("---\n")
lines.append("## Summary Results\n")
lines.append(f"| Model | AUC | AUPRC | F1 (cal) | Precision | Recall |")
lines.append("|-------|:---:|:-----:|:--------:|:---------:|:------:|")
for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_auc"]):
    tier = r["tier"]
    lines.append(f"| [{tier}] {name} | {r['macro_auc']:.4f} | {r['macro_auprc']:.4f} | "
                 f"{r['macro_f1']:.4f} | {r['macro_precision']:.4f} | {r['macro_recall']:.4f} |")
lines.append("")

lines.append("---\n")
lines.append("## Per-Label Results\n")
for name, r in sorted(results.items(), key=lambda x: -x[1]["macro_auc"]):
    tier = r["tier"]
    lines.append(f"### [{tier}] {name}\n")
    lines.append(f"**Macro:** AUC {r['macro_auc']:.4f} | AUPRC {r['macro_auprc']:.4f} | "
                 f"F1 {r['macro_f1']:.4f} | Precision {r['macro_precision']:.4f} | Recall {r['macro_recall']:.4f}\n")
    lines.append("| Label | AUC | AUPRC | F1 | Precision | Recall | Threshold |")
    lines.append("|-------|:---:|:-----:|:--:|:---------:|:------:|:---------:|")
    for label, m in r["per_label"].items():
        lines.append(f"| {label} | {m['auc']:.4f} | {m['auprc']:.4f} | {m['f1']:.4f} | "
                     f"{m['precision']:.4f} | {m['recall']:.4f} | {m['threshold']:.2f} |")
    lines.append("")

md_path = os.path.join(RESULTS_DIR, "test_results.md")
with open(md_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved test_results.md")
