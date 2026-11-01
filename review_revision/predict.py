from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from .common import binary_entropy


def _ece(y_true, p, bins=10):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if mask.any():
            out += mask.mean() * abs(y_true[mask].mean() - p[mask].mean())
    return float(out)


class PlattScaler:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y):
        self.model.fit(self._logit(p), np.asarray(y, dtype=int))
        return self

    def predict(self, p):
        return self.model.predict_proba(self._logit(p))[:, 1]


def infer_types(X: pd.DataFrame):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def make_preprocessor(numeric, categorical):
    num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("num", num, numeric),
        ("cat", cat, categorical),
    ])


def make_models(seed: int, threads: int):
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2500, class_weight="balanced", random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=260, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=threads,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=260, max_depth=4, learning_rate=0.04,
            subsample=0.90, colsample_bytree=0.90,
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            random_state=seed, n_jobs=threads,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=260, learning_rate=0.04, num_leaves=31,
            subsample=0.90, subsample_freq=1, colsample_bytree=0.90,
            class_weight="balanced", deterministic=True, force_col_wise=True,
            random_state=seed, verbosity=-1, n_jobs=threads,
        ),
    }


def fit_calibrated(model, preprocessor, X_train, y_train, X_test, seed):
    fit_idx, cal_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.20, random_state=seed, stratify=y_train
    )
    pipe = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(model)),
    ])
    pipe.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
    raw_cal = pipe.predict_proba(X_train.iloc[cal_idx])[:, 1]
    raw_test = pipe.predict_proba(X_test)[:, 1]
    scaler = PlattScaler().fit(raw_cal, y_train.iloc[cal_idx])
    return np.clip(scaler.predict(raw_test), 1e-6, 1 - 1e-6)


def run_oof(X: pd.DataFrame, y: pd.Series, folds=5, repeats=3, seed=20261031, threads=8):
    X = X.reset_index(drop=True)
    y = y.astype(int).reset_index(drop=True)
    numeric, categorical = infer_types(X)
    pre = make_preprocessor(numeric, categorical)
    split = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    names = list(make_models(seed, threads))
    preds = {r: {m: np.full(len(X), np.nan) for m in names} for r in range(repeats)}

    for split_no, (tr, te) in enumerate(split.split(X, y)):
        r = split_no // folds
        f = split_no % folds
        Xtr = X.iloc[tr].reset_index(drop=True)
        ytr = y.iloc[tr].reset_index(drop=True)
        Xte = X.iloc[te]
        models = make_models(seed + 100 * r + f, threads)
        for name, model in models.items():
            preds[r][name][te] = fit_calibrated(
                model, pre, Xtr, ytr, Xte, seed + 1000 * r + 10 * f
            )

    metric_rows = []
    ens_repeats = []
    for r in range(repeats):
        matrix = []
        for name in names:
            p = preds[r][name]
            if np.isnan(p).any():
                raise RuntimeError(f"Incomplete OOF predictions for repeat={r}, model={name}")
            metric_rows.append({
                "repeat": r + 1,
                "model": name,
                "auroc": roc_auc_score(y, p),
                "pr_auc": average_precision_score(y, p),
                "brier": brier_score_loss(y, p),
                "ece": _ece(y, p),
            })
            matrix.append(p)
        ens = np.vstack(matrix).mean(axis=0)
        ens_repeats.append(ens)
        metric_rows.append({
            "repeat": r + 1,
            "model": "Ensemble",
            "auroc": roc_auc_score(y, ens),
            "pr_auc": average_precision_score(y, ens),
            "brier": brier_score_loss(y, ens),
            "ece": _ece(y, ens),
        })

    p_final = np.vstack(ens_repeats).mean(axis=0)
    entropy = binary_entropy(p_final)
    metrics = pd.DataFrame(metric_rows)
    return p_final, entropy, metrics
