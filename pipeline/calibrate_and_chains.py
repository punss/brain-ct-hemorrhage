"""
pipeline/calibrate_and_chains.py

Two improvements applied to all Tier 1 + Tier 2 models:

  1. Per-label threshold calibration on the val set
     — finds the threshold per label that maximises F1
     — zero retraining; applied post-hoc to existing OvR models

  2. Classifier chains
     — replaces OneVsRestClassifier with ClassifierChain
     — reuses best hyperparameters from OvR search
     — label order: intraventricular → epidural → intraparenchymal → subdural → subarachnoid
       (highest AUC first so downstream classifiers receive more reliable features)
     — cv=5 inside the chain to prevent label-leakage during training

Outputs:
  pipeline/results/calibration_summary.json
  pipeline/results/chains_summary.json
  pipeline/models/chains/{model_name}.pkl
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
RESULTS_DIR  = os.path.join(PIPELINE_DIR, "results")
CHAINS_DIR   = os.path.join(PIPELINE_DIR, "models", "chains")
os.makedirs(CHAINS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
X     = np.load(os.path.join(PIPELINE_DIR, "X.npy"))
y_all = np.load(os.path.join(PIPELINE_DIR, "y_cls.npy"))
meta  = pd.read_csv(os.path.join(PIPELINE_DIR, "metadata_all.csv"))

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
y = y_all[:, 1:].astype(int)

train_mask = (meta["split"] == "train").values
val_mask   = (meta["split"] == "val").values

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

# Chain label order: highest AUC first (intraventricular=2, epidural=0,
# intraparenchymal=1, subdural=4, subarachnoid=3)
CHAIN_ORDER = [2, 0, 1, 4, 3]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def macro_auc_val(model, X, y):
    y_prob = model.predict_proba(X)
    return round(float(roc_auc_score(y, y_prob, average="macro")), 4)

def per_label_auc_val(model, X, y):
    y_prob = model.predict_proba(X)
    return {
        label: round(float(roc_auc_score(y[:, i], y_prob[:, i])), 4)
        for i, label in enumerate(LABEL_COLS)
    }

def find_thresholds(y_true, y_prob):
    """Per-label threshold that maximises F1 on the provided set."""
    thresholds = {}
    for i, label in enumerate(LABEL_COLS):
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            pred = (y_prob[:, i] >= t).astype(int)
            f1 = f1_score(y_true[:, i], pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[label] = round(float(best_t), 2)
    return thresholds

def apply_thresholds(y_prob, thresholds):
    y_pred = np.zeros(y_prob.shape, dtype=int)
    for i, label in enumerate(LABEL_COLS):
        y_pred[:, i] = (y_prob[:, i] >= thresholds[label]).astype(int)
    return y_pred

def macro_f1(y_true, y_pred):
    return round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)

# ---------------------------------------------------------------------------
# Part 1: Threshold calibration on all existing OvR models
# ---------------------------------------------------------------------------
print("=" * 65)
print("PART 1 — PER-LABEL THRESHOLD CALIBRATION")
print("=" * 65)

model_paths = {}
for tier, subdir in [("tier1", "tier1"), ("tier2", "tier2")]:
    d = os.path.join(PIPELINE_DIR, "models", subdir)
    for fname in sorted(os.listdir(d)):
        if fname.endswith(".pkl"):
            name = fname.replace(".pkl", "")
            model_paths[name] = os.path.join(d, fname)

calibration_results = {}

print(f"\n{'MODEL':<22} {'OvR F1':>8} {'Cal F1':>8} {'Delta':>7}  Thresholds")
print("-" * 75)

for name, path in model_paths.items():
    model = joblib.load(path)
    y_prob = model.predict_proba(X_val)

    # Default threshold (0.5)
    y_pred_default = model.predict(X_val)
    f1_default = macro_f1(y_val, y_pred_default)

    # Calibrated threshold (found on val set)
    thresholds = find_thresholds(y_val, y_prob)
    y_pred_cal = apply_thresholds(y_prob, thresholds)
    f1_cal = macro_f1(y_val, y_pred_cal)

    delta = round(f1_cal - f1_default, 4)
    print(f"{name:<22} {f1_default:>8.4f} {f1_cal:>8.4f} {delta:>+7.4f}  {thresholds}")

    calibration_results[name] = {
        "f1_default":   f1_default,
        "f1_calibrated": f1_cal,
        "delta_f1":     delta,
        "thresholds":   thresholds,
        "val_macro_auc": macro_auc_val(model, X_val, y_val),
    }

with open(os.path.join(RESULTS_DIR, "calibration_summary.json"), "w") as f:
    json.dump(calibration_results, f, indent=2)
print(f"\nSaved calibration_summary.json")

# ---------------------------------------------------------------------------
# Part 2: Classifier chains — best params from OvR search
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
print("PART 2 — CLASSIFIER CHAINS")
print("=" * 65)

# Load best params found during OvR search
with open(os.path.join(RESULTS_DIR, "tier1_summary.json")) as f:
    t1_params = {k: v["best_params"] for k, v in json.load(f).items()}
with open(os.path.join(RESULTS_DIR, "tier2_summary.json")) as f:
    t2_params = {k: v["best_params"] for k, v in json.load(f).items()}

def make_chain(base_estimator):
    return ClassifierChain(base_estimator, order=CHAIN_ORDER, cv=5, random_state=42)

p = t1_params

CHAIN_MODELS = {
    # Tier 1
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", make_chain(LogisticRegression(
            solver="saga", max_iter=3000, random_state=42,
            class_weight="balanced",
            penalty=p["logistic_regression"]["clf__estimator__penalty"],
            C=p["logistic_regression"]["clf__estimator__C"],
        ))),
    ]),
    "lda": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",   PCA(n_components=p["lda"]["pca__n_components"], random_state=42)),
        ("clf",   make_chain(LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=p["lda"]["clf__estimator__shrinkage"],
        ))),
    ]),
    "qda": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",   PCA(n_components=p["qda"]["pca__n_components"], random_state=42)),
        ("clf",   make_chain(QuadraticDiscriminantAnalysis(
            reg_param=p["qda"]["clf__estimator__reg_param"],
        ))),
    ]),
    "gnb": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",   PCA(n_components=p["gnb"]["pca__n_components"], random_state=42)),
        ("clf",   make_chain(GaussianNB(
            var_smoothing=p["gnb"]["clf__estimator__var_smoothing"],
        ))),
    ]),
    "knn": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",   PCA(n_components=p["knn"]["pca__n_components"], random_state=42)),
        ("clf",   make_chain(KNeighborsClassifier(
            n_neighbors=p["knn"]["clf__estimator__n_neighbors"],
            metric=p["knn"]["clf__estimator__metric"],
            weights=p["knn"]["clf__estimator__weights"],
        ))),
    ]),
    # Tier 2
    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("pca",   PCA(n_components=t2_params["svm"]["pca__n_components"], random_state=42)),
        ("clf",   make_chain(SVC(
            kernel="rbf", probability=True, random_state=42,
            class_weight="balanced",
            C=t2_params["svm"]["clf__estimator__C"],
            gamma=t2_params["svm"]["clf__estimator__gamma"],
        ))),
    ]),
    "random_forest": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",   make_chain(RandomForestClassifier(
            random_state=42, n_jobs=-1,
            class_weight="balanced",
            n_estimators=t2_params["random_forest"]["clf__estimator__n_estimators"],
            min_samples_leaf=t2_params["random_forest"]["clf__estimator__min_samples_leaf"],
            max_features=t2_params["random_forest"]["clf__estimator__max_features"],
            max_depth=t2_params["random_forest"]["clf__estimator__max_depth"],
        ))),
    ]),
    "xgboost": Pipeline([
        ("clf", make_chain(XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0,
            n_estimators=t2_params["xgboost"]["clf__estimator__n_estimators"],
            learning_rate=t2_params["xgboost"]["clf__estimator__learning_rate"],
            max_depth=t2_params["xgboost"]["clf__estimator__max_depth"],
            subsample=t2_params["xgboost"]["clf__estimator__subsample"],
            colsample_bytree=t2_params["xgboost"]["clf__estimator__colsample_bytree"],
            min_child_weight=t2_params["xgboost"]["clf__estimator__min_child_weight"],
        ))),
    ]),
}

chains_results = {}

print(f"\n{'MODEL':<22} {'Chain AUC':>10} {'Chain F1':>9} {'Cal F1':>9}  vs OvR AUC")
print("-" * 70)

ovr_aucs = {**{k: v["val_macro_auc"] for k, v in json.load(open(os.path.join(RESULTS_DIR, "tier1_summary.json"))).items()},
            **{k: v["val_macro_auc"] for k, v in json.load(open(os.path.join(RESULTS_DIR, "tier2_summary.json"))).items()}}

for name, pipeline in CHAIN_MODELS.items():
    print(f"  Fitting {name}...", end=" ", flush=True)
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_val)
    y_pred = pipeline.predict(X_val)

    auc   = round(float(roc_auc_score(y_val, y_prob, average="macro")), 4)
    f1_default = macro_f1(y_val, y_pred)

    thresholds = find_thresholds(y_val, y_prob)
    y_pred_cal = apply_thresholds(y_prob, thresholds)
    f1_cal = macro_f1(y_val, y_pred_cal)

    delta_auc = round(auc - ovr_aucs.get(name, 0), 4)

    print(f"done")
    print(f"  {name:<22} {auc:>10.4f} {f1_default:>9.4f} {f1_cal:>9.4f}  {delta_auc:+.4f} vs OvR")

    joblib.dump(pipeline, os.path.join(CHAINS_DIR, f"{name}.pkl"))

    chains_results[name] = {
        "val_macro_auc":    auc,
        "val_macro_f1":     f1_default,
        "val_f1_calibrated": f1_cal,
        "delta_auc_vs_ovr": delta_auc,
        "thresholds":       thresholds,
        "val_per_label_auc": per_label_auc_val(pipeline, X_val, y_val),
    }

with open(os.path.join(RESULTS_DIR, "chains_summary.json"), "w") as f:
    json.dump(chains_results, f, indent=2)
print(f"\nSaved chains_summary.json")

# ---------------------------------------------------------------------------
# Final comparison table
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print(f"{'MODEL':<22} {'OvR AUC':>9} {'Chain AUC':>10} {'OvR F1(cal)':>12} {'Chain F1(cal)':>14}")
print("-" * 75)

tiers = {"logistic_regression": "T1", "lda": "T1", "qda": "T1", "gnb": "T1", "knn": "T1",
         "svm": "T2", "random_forest": "T2", "xgboost": "T2"}

for name in CHAIN_MODELS:
    tier = tiers[name]
    ovr_auc = ovr_aucs.get(name, 0)
    ovr_f1_cal = calibration_results[name]["f1_calibrated"]
    ch_auc = chains_results[name]["val_macro_auc"]
    ch_f1_cal = chains_results[name]["val_f1_calibrated"]
    print(f"[{tier}] {name:<20} {ovr_auc:>9.4f} {ch_auc:>10.4f} {ovr_f1_cal:>12.4f} {ch_f1_cal:>14.4f}")

print("=" * 75)
