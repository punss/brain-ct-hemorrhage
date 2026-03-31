"""
pipeline/train_tier1.py

Trains all Tier 1 models on the 269-dim feature vectors using
OneVsRestClassifier for 5-label multi-label classification.

Labels: epidural, intraparenchymal, intraventricular, subarachnoid, subdural
        (dropping 'any' — always 1 in this dataset)

Outputs:
  pipeline/results/tier1_summary.json
  pipeline/models/tier1/{model_name}.pkl
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "pipeline")

MODELS_DIR  = os.path.join(PIPELINE_DIR, "models", "tier1")
RESULTS_DIR = os.path.join(PIPELINE_DIR, "results")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
X    = np.load(os.path.join(PIPELINE_DIR, "X.npy"))
y_all = np.load(os.path.join(PIPELINE_DIR, "y_cls.npy"))
meta = pd.read_csv(os.path.join(PIPELINE_DIR, "metadata_all.csv"))

# Drop 'any' (col 0) — always 1, zero variance
LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
y = y_all[:, 1:].astype(int)   # (2929, 5)

train_mask = (meta["split"] == "train").values
val_mask   = (meta["split"] == "val").values

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]

print(f"Train: {X_train.shape}  Val: {X_val.shape}")
print(f"Labels: {LABEL_COLS}\n")

# ---------------------------------------------------------------------------
# Cross-validation — stratify folds on hemorrhage_type
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
    "logistic_regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", OneVsRestClassifier(
                LogisticRegression(solver="saga", max_iter=3000, random_state=42,
                                   class_weight="balanced")
            )),
        ]),
        "param_dist": {
            "clf__estimator__penalty": ["l1", "l2"],
            "clf__estimator__C":       [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "n_iter": 12,
    },
    "lda": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",   PCA(random_state=42)),
            ("clf",   OneVsRestClassifier(
                LinearDiscriminantAnalysis(solver="lsqr")
            )),
        ]),
        "param_dist": {
            "pca__n_components":          [20, 30, 40, 50, 60],
            "clf__estimator__shrinkage":  [None, "auto", 0.1, 0.3, 0.5, 0.7],
        },
        "n_iter": 15,
    },
    "qda": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",   PCA(random_state=42)),
            ("clf",   OneVsRestClassifier(QuadraticDiscriminantAnalysis())),
        ]),
        "param_dist": {
            "pca__n_components":             [20, 30, 40, 50],
            "clf__estimator__reg_param":     [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "n_iter": 15,
    },
    "gnb": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",   PCA(random_state=42)),
            ("clf",   OneVsRestClassifier(GaussianNB())),
        ]),
        "param_dist": {
            "pca__n_components":              [15, 20, 25, 30, 40],
            "clf__estimator__var_smoothing":  [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        },
        "n_iter": 15,
    },
    "knn": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("pca",   PCA(random_state=42)),
            ("clf",   OneVsRestClassifier(KNeighborsClassifier())),
        ]),
        "param_dist": {
            "pca__n_components":          [15, 20, 25, 30, 35, 40],
            "clf__estimator__n_neighbors": [3, 5, 7, 11, 15, 21],
            "clf__estimator__metric":     ["euclidean", "manhattan"],
            "clf__estimator__weights":    ["uniform", "distance"],
        },
        "n_iter": 20,
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
        estimator  = config["pipeline"],
        param_distributions = config["param_dist"],
        n_iter     = config["n_iter"],
        scoring    = macro_auc,
        cv         = cv_splits,
        refit      = True,
        n_jobs     = -1,
        random_state = 42,
        error_score  = 0.0,
    )
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    best   = search.best_estimator_
    y_prob = best.predict_proba(X_val)
    y_pred = best.predict(X_val)

    # Per-label AUC
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

    # Save fitted pipeline
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(best, model_path)

    print(f"  CV macro AUC : {result['best_cv_macro_auc']}")
    print(f"  Val macro AUC: {result['val_macro_auc']}")
    print(f"  Val macro F1 : {result['val_macro_f1']}")
    print(f"  Per-label AUC: {per_label_auc}")
    print(f"  Best params  : {search.best_params_}")
    print(f"  Time         : {elapsed:.1f}s\n")

# ---------------------------------------------------------------------------
# Save results + summary table
# ---------------------------------------------------------------------------
results_path = os.path.join(RESULTS_DIR, "tier1_summary.json")
with open(results_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Results saved to {results_path}")

print(f"\n{'='*60}")
print(f"{'MODEL':<22} {'CV AUC':>8} {'VAL AUC':>8} {'VAL F1':>8}")
print(f"{'-'*50}")
for name, r in all_results.items():
    print(f"{name:<22} {r['best_cv_macro_auc']:>8.4f} {r['val_macro_auc']:>8.4f} {r['val_macro_f1']:>8.4f}")
print(f"{'='*60}")
