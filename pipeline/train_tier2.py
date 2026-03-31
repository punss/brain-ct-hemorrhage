"""
pipeline/train_tier2.py

Trains all Tier 2 models (SVM, Random Forest, XGBoost) on the 269-dim feature
vectors using OneVsRestClassifier for 5-label multi-label classification.

Preprocessing per model (from features.md / scaling memory):
  SVM    : StandardScaler → PCA (n_components tuned) → OvR(SVC, probability=True)
  RF     : StandardScaler → OvR(RandomForestClassifier)   [internal feature selection]
  XGBoost: OvR(XGBClassifier)                             [tree-based, scale-invariant]

Labels: epidural, intraparenchymal, intraventricular, subarachnoid, subdural

Outputs:
  pipeline/results/tier2_summary.json
  pipeline/models/tier2/{model_name}.pkl
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")
MODELS_DIR   = os.path.join(PIPELINE_DIR, "models", "tier2")
RESULTS_DIR  = os.path.join(PIPELINE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
X     = np.load(os.path.join(PIPELINE_DIR, "X.npy"))
y_all = np.load(os.path.join(PIPELINE_DIR, "y_cls.npy"))
meta  = pd.read_csv(os.path.join(PIPELINE_DIR, "metadata_all.csv"))

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
y = y_all[:, 1:].astype(int)   # drop 'any' (col 0)

train_mask = (meta["split"] == "train").values
val_mask   = (meta["split"] == "val").values

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

print(f"Train: {X_train.shape}  Val: {X_val.shape}")
print(f"Labels: {LABEL_COLS}\n")

# ---------------------------------------------------------------------------
# CV — same strategy as Tier 1: stratify folds on hemorrhage_type
# ---------------------------------------------------------------------------
cv_strat = meta.loc[train_mask, "hemorrhage_type"].values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_splits = list(cv.split(X_train, cv_strat))

# ---------------------------------------------------------------------------
# Scorer — macro AUC across all 5 labels
# ---------------------------------------------------------------------------
def macro_auc(estimator, X, y):
    y_prob = estimator.predict_proba(X)
    try:
        return roc_auc_score(y, y_prob, average="macro")
    except ValueError:
        return 0.0

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
MODELS = {
    "svm": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(random_state=42)),
            ("clf",    OneVsRestClassifier(
                SVC(kernel="rbf", probability=True, random_state=42,
                    class_weight="balanced")
            )),
        ]),
        "param_dist": {
            "pca__n_components":      [40, 60, 80, 100, 120],
            "clf__estimator__C":      [0.1, 1.0, 10.0, 100.0],
            "clf__estimator__gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
        },
        "n_iter": 20,
        "n_jobs": -1,
    },
    "random_forest": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    OneVsRestClassifier(
                RandomForestClassifier(random_state=42, n_jobs=1,
                                       class_weight="balanced")
            )),
        ]),
        "param_dist": {
            "clf__estimator__n_estimators":    [100, 200, 300, 500],
            "clf__estimator__max_depth":       [None, 10, 20, 30],
            "clf__estimator__min_samples_leaf": [1, 2, 4, 8],
            "clf__estimator__max_features":    ["sqrt", "log2", 0.3, 0.5],
        },
        "n_iter": 20,
        "n_jobs": -1,
    },
    "xgboost": {
        "pipeline": Pipeline([
            ("clf", OneVsRestClassifier(
                XGBClassifier(
                    eval_metric="logloss",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            )),
        ]),
        "param_dist": {
            "clf__estimator__n_estimators":     [100, 200, 300],
            "clf__estimator__learning_rate":    [0.01, 0.05, 0.1, 0.2],
            "clf__estimator__max_depth":        [3, 4, 5, 6, 8],
            "clf__estimator__subsample":        [0.6, 0.7, 0.8, 1.0],
            "clf__estimator__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "clf__estimator__min_child_weight": [1, 3, 5],
        },
        "n_iter": 20,
        "n_jobs": -1,
    },
}

# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
all_results = {}

for name, config in MODELS.items():
    print(f"{'='*60}")
    print(f"Training: {name.upper()}")
    t0 = time.perf_counter()

    search = RandomizedSearchCV(
        estimator           = config["pipeline"],
        param_distributions = config["param_dist"],
        n_iter              = config["n_iter"],
        scoring             = macro_auc,
        cv                  = cv_splits,
        refit               = True,
        n_jobs              = config["n_jobs"],
        random_state        = 42,
        error_score         = 0.0,
        verbose             = 1,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best   = search.best_estimator_
    y_prob = best.predict_proba(X_val)
    y_pred = best.predict(X_val)

    per_label_auc = {}
    for i, label in enumerate(LABEL_COLS):
        try:
            per_label_auc[label] = round(float(roc_auc_score(y_val[:, i], y_prob[:, i])), 4)
        except ValueError:
            per_label_auc[label] = None

    macro_auc_val = round(float(roc_auc_score(y_val, y_prob, average="macro")), 4)
    macro_f1_val  = round(float(f1_score(y_val, y_pred, average="macro", zero_division=0)), 4)

    result = {
        "best_cv_macro_auc": round(float(search.best_score_), 4),
        "val_macro_auc":     macro_auc_val,
        "val_macro_f1":      macro_f1_val,
        "val_per_label_auc": per_label_auc,
        "best_params":       search.best_params_,
        "train_time_s":      round(elapsed, 1),
    }
    all_results[name] = result

    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(best, model_path)

    print(f"  CV macro AUC : {result['best_cv_macro_auc']}")
    print(f"  Val macro AUC: {result['val_macro_auc']}")
    print(f"  Val macro F1 : {result['val_macro_f1']}")
    print(f"  Per-label AUC: {per_label_auc}")
    print(f"  Best params  : {search.best_params_}")
    print(f"  Time         : {elapsed:.1f}s\n")

# ---------------------------------------------------------------------------
# Save + summary
# ---------------------------------------------------------------------------
results_path = os.path.join(RESULTS_DIR, "tier2_summary.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {results_path}")

# Combined Tier 1 + Tier 2 comparison
tier1_path = os.path.join(RESULTS_DIR, "tier1_summary.json")
if os.path.exists(tier1_path):
    with open(tier1_path) as f:
        tier1 = json.load(f)
    combined = {**tier1, **all_results}
else:
    combined = all_results

print(f"\n{'='*65}")
print(f"{'MODEL':<26} {'CV AUC':>8} {'VAL AUC':>9} {'VAL F1':>8}")
print(f"{'-'*55}")
print("-- Tier 1 --")
if os.path.exists(tier1_path):
    for name, r in tier1.items():
        print(f"  {name:<24} {r['best_cv_macro_auc']:>8.4f} {r['val_macro_auc']:>9.4f} {r['val_macro_f1']:>8.4f}")
print("-- Tier 2 --")
for name, r in all_results.items():
    print(f"  {name:<24} {r['best_cv_macro_auc']:>8.4f} {r['val_macro_auc']:>9.4f} {r['val_macro_f1']:>8.4f}")
print(f"{'='*65}")
