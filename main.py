#!/usr/bin/env python3
"""
Knowledge-Guided Decision Intelligence for Cost-Aware Customer Churn
Intervention Under Predictive Uncertainty

Single-file experimental pipeline for a Knowledge-Based Systems short communication.

Design principles
-----------------
1) leakage-safe repeated out-of-fold prediction;
2) separate calibration partition inside each outer training fold;
3) predictive entropy as the primary uncertainty measure;
4) model disagreement retained as a separate diagnostic only;
5) unit-consistent simulated business utility;
6) IBM CLTV used only as a dimensionless strategic-priority index;
7) fair, budget-matched policy comparisons;
8) paired policy-rerun bootstrap for the main KGDI effect estimate;
9) ablation, subgroup audit, and sensitivity analyses as supplementary evidence;
10) exactly two primary tables and two primary figures.

Observed fields come from the IBM Telco Customer Churn workbook.
Business costs, intervention success, margin, review accuracy, and budget are
explicit simulation assumptions and must not be described as observed outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import warnings
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
import lightgbm
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


TARGET = "Churn Value"

LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason"]

IDENTIFIER_GEO_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
]

# IBM CLTV is treated as a customer-value index, not currency.
BUSINESS_ONLY_COLUMNS = ["CLTV"]

# Excluded from the primary model; retained only for post-hoc auditing.
AUDIT_ONLY_COLUMNS = ["Gender", "Senior Citizen"]

# Explicit publication-grade feature allowlist. Unknown workbook columns are never
# admitted to the predictive model implicitly.
MODEL_FEATURES = [
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
]


@dataclass
class BusinessConfig:
    offer_cost: float = 100.0
    retention_success: float = 0.35
    human_review_cost: float = 25.0
    human_accuracy: float = 0.85
    budget_fraction: float = 0.20

    gross_margin_rate: float = 0.60
    value_horizon_months: int = 12

    minimum_churn_risk: float = 0.20
    uncertainty_threshold: float = 0.65
    moderate_uncertainty_threshold: float = 0.45
    high_value_quantile: float = 0.75
    cltv_priority_weight: float = 0.03
    review_utility_tolerance: float = 0.90


# -----------------------------------------------------------------------------
# CLI / validation
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Leakage-safe churn prediction + knowledge-guided decision intelligence"
    )
    p.add_argument("--data", required=True, help="IBM .xlsx or ZIP containing the workbook")
    p.add_argument("--output", default="results/paper")
    p.add_argument("--mode", choices=["quick", "paper"], default="paper")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap", type=int, default=None)

    p.add_argument("--offer-cost", type=float, default=100.0)
    p.add_argument("--retention-success", type=float, default=0.35)
    p.add_argument("--human-review-cost", type=float, default=25.0)
    p.add_argument("--human-accuracy", type=float, default=0.85)
    p.add_argument("--budget-fraction", type=float, default=0.20)
    p.add_argument("--gross-margin-rate", type=float, default=0.60)
    p.add_argument("--value-horizon-months", type=int, default=12)

    # KGDI policy parameters are exposed explicitly for reproducibility.
    p.add_argument("--minimum-churn-risk", type=float, default=0.20)
    p.add_argument("--uncertainty-threshold", type=float, default=0.65)
    p.add_argument("--moderate-uncertainty-threshold", type=float, default=0.45)
    p.add_argument("--high-value-quantile", type=float, default=0.75)
    p.add_argument("--cltv-priority-weight", type=float, default=0.03)
    p.add_argument("--review-utility-tolerance", type=float, default=0.90)
    return p.parse_args()


def validate_args(args):
    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.offer_cost <= 0:
        raise ValueError("--offer-cost must be > 0")
    if args.human_review_cost < 0:
        raise ValueError("--human-review-cost must be >= 0")
    if args.value_horizon_months < 1:
        raise ValueError("--value-horizon-months must be >= 1")

    for name in [
        "retention_success", "human_accuracy", "budget_fraction", "gross_margin_rate"
    ]:
        value = getattr(args, name)
        if not (0 < value <= 1):
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")

    if args.repeats is not None and args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if args.bootstrap is not None and args.bootstrap < 100:
        raise ValueError("--bootstrap must be >= 100")

    for name in [
        "minimum_churn_risk",
        "uncertainty_threshold",
        "moderate_uncertainty_threshold",
        "high_value_quantile",
    ]:
        value = getattr(args, name)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.moderate_uncertainty_threshold > args.uncertainty_threshold:
        raise ValueError(
            "--moderate-uncertainty-threshold must be <= --uncertainty-threshold"
        )
    if args.cltv_priority_weight < 0:
        raise ValueError("--cltv-priority-weight must be >= 0")
    if args.review_utility_tolerance < 0:
        raise ValueError("--review-utility-tolerance must be >= 0")


# -----------------------------------------------------------------------------
# Data loading / audit
# -----------------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as z:
            candidates = [
                f for f in z.namelist()
                if f.lower().endswith(".xlsx")
                and not f.startswith("__MACOSX")
                and not Path(f).name.startswith("~$")
            ]
            if not candidates:
                raise ValueError("No .xlsx workbook found inside ZIP")

            exact = [
                f for f in candidates
                if Path(f).name.lower() == "telco_customer_churn.xlsx"
            ]
            selected = exact[0] if exact else sorted(candidates)[0]
            print(f"Workbook selected: {selected}")
            with z.open(selected) as f:
                return pd.read_excel(io.BytesIO(f.read()))

    raise ValueError("Dataset must be .xlsx or .zip containing an .xlsx workbook")


def clean_data(df: pd.DataFrame):
    required = [
        "CustomerID", "Tenure Months", "Monthly Charges",
        "Total Charges", "CLTV", TARGET,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df = df.copy()

    for c in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[TARGET].isna().any():
        raise ValueError("Churn Value contains missing/non-numeric values")

    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn Value must be binary 0/1")

    if df["CustomerID"].isna().any() or df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be complete and unique")

    zero_tenure_blank = df["Total Charges"].isna() & (df["Tenure Months"] == 0)
    zero_tenure_fixed = int(zero_tenure_blank.sum())
    df.loc[zero_tenure_blank, "Total Charges"] = 0.0

    # Any remaining Total Charges missingness is intentionally preserved here.
    # It is imputed inside each training fold by the preprocessing pipeline,
    # preventing full-dataset distribution information from entering CV folds.
    remaining_total_missing = int(df["Total Charges"].isna().sum())

    if df["Monthly Charges"].isna().any() or df["CLTV"].isna().any():
        raise ValueError("Monthly Charges and CLTV must be present for the decision layer")
    if (df["Monthly Charges"] < 0).any() or (df["CLTV"] < 0).any():
        raise ValueError("Negative Monthly Charges or CLTV values detected")

    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "churners": int(df[TARGET].sum()),
        "non_churners": int((df[TARGET] == 0).sum()),
        "churn_rate": float(df[TARGET].mean()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "zero_tenure_total_charges_set_to_zero": zero_tenure_fixed,
        "remaining_total_charges_left_for_fold_local_imputation": remaining_total_missing,
    }
    return df, audit


def select_features(df):
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Required predictive features missing: {missing}")

    # Fixed allowlist prevents schema drift or newly added outcome-derived fields
    # from silently entering the predictive model.
    features = list(MODEL_FEATURES)
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return features, numeric, categorical


def save_column_roles(df, output_dir):
    rows = []
    for c in df.columns:
        if c == TARGET:
            role = "target"
        elif c in LEAKAGE_COLUMNS:
            role = "exclude_direct_leakage"
        elif c in BUSINESS_ONLY_COLUMNS:
            role = "decision_layer_only"
        elif c in AUDIT_ONLY_COLUMNS:
            role = "audit_only"
        elif c in IDENTIFIER_GEO_COLUMNS:
            role = "exclude_identifier_or_geography"
        elif c in MODEL_FEATURES:
            role = "ai_feature"
        else:
            role = "excluded_unrecognized_column"

        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique(dropna=True)),
            "role": role,
        })

    pd.DataFrame(rows).to_csv(
        output_dir / "SUPPLEMENT_column_roles.csv", index=False
    )


# -----------------------------------------------------------------------------
# Predictive modeling / calibration
# -----------------------------------------------------------------------------

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


def make_models(seed: int, threads: int, mode: str):
    trees = 160 if mode == "quick" else 260
    lr = 0.055 if mode == "quick" else 0.04

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2500,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=trees,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=threads,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=trees,
            max_depth=4,
            learning_rate=lr,
            subsample=0.90,
            colsample_bytree=0.90,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=threads,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=trees,
            learning_rate=lr,
            num_leaves=31,
            subsample=0.90,
            subsample_freq=1,
            colsample_bytree=0.90,
            class_weight="balanced",
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
            verbosity=-1,
            n_jobs=threads,
        ),
    }


class PlattScaler:
    """Sigmoid calibrator fitted only on a dedicated calibration partition."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, solver="lbfgs")

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_probability, y):
        self.model.fit(self._logit(raw_probability), np.asarray(y))
        return self

    def predict(self, raw_probability):
        return self.model.predict_proba(self._logit(raw_probability))[:, 1]


def fit_calibrated_model(model, preprocessor, X_train, y_train, X_test, seed):
    fit_idx, cal_idx = train_test_split(
        np.arange(len(X_train)),
        test_size=0.20,
        random_state=seed,
        stratify=y_train,
    )

    pipeline = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", clone(model)),
    ])
    pipeline.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])

    raw_cal = pipeline.predict_proba(X_train.iloc[cal_idx])[:, 1]
    raw_test = pipeline.predict_proba(X_test)[:, 1]

    scaler = PlattScaler().fit(raw_cal, y_train.iloc[cal_idx])
    calibrated = scaler.predict(raw_test)
    return np.clip(calibrated, 1e-6, 1 - 1e-6)


def expected_calibration_error(y_true, p, bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi if i == bins - 1 else p < hi)
        if not np.any(mask):
            continue
        ece += np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(p[mask]))
    return float(ece)


def predictive_metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
    }


def binary_entropy(p):
    """Normalized binary predictive entropy in [0, 1]."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def ensemble_probability_and_diagnostics(model_probabilities):
    """
    Equal-weight calibrated ensemble.

    Primary uncertainty = predictive entropy of the ensemble probability.
    Model disagreement is retained as a separate diagnostic and is NOT mixed
    with entropy using arbitrary weights.
    """
    probs = np.asarray(model_probabilities, dtype=float)
    mean_p = probs.mean(axis=0)
    entropy = binary_entropy(mean_p)
    disagreement = probs.std(axis=0, ddof=0)
    return mean_p, entropy, disagreement


def run_repeated_oof(df, features, numeric, categorical, args):
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "quick" else 5)
    splitter = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=repeats,
        random_state=args.seed,
    )

    X = df[features].reset_index(drop=True)
    y = df[TARGET].astype(int).reset_index(drop=True)
    n = len(df)
    preprocessor = make_preprocessor(numeric, categorical)

    model_names = list(make_models(args.seed, args.threads, args.mode).keys())
    repeat_predictions = {
        r: {name: np.full(n, np.nan) for name in model_names}
        for r in range(repeats)
    }

    for split_number, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        repeat_id = split_number // args.folds
        fold_id = split_number % args.folds

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx]

        models = make_models(
            seed=args.seed + repeat_id * 100 + fold_id,
            threads=args.threads,
            mode=args.mode,
        )

        for model_name, model in models.items():
            p_test = fit_calibrated_model(
                model=model,
                preprocessor=preprocessor,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                seed=args.seed + repeat_id * 1000 + fold_id * 10,
            )
            repeat_predictions[repeat_id][model_name][test_idx] = p_test

        print(
            f"Completed repeat {repeat_id + 1}/{repeats}, "
            f"fold {fold_id + 1}/{args.folds}"
        )

    metric_rows = []
    ensemble_by_repeat = {}

    for repeat_id in range(repeats):
        matrix = []
        for model_name, probs in repeat_predictions[repeat_id].items():
            if np.isnan(probs).any():
                raise RuntimeError(
                    f"Incomplete OOF predictions: repeat={repeat_id}, model={model_name}"
                )
            row = {"repeat": repeat_id + 1, "model": model_name}
            row.update(predictive_metrics(y.to_numpy(), probs))
            metric_rows.append(row)
            matrix.append(probs)

        matrix = np.vstack(matrix)
        ensemble_p, entropy, disagreement = ensemble_probability_and_diagnostics(matrix)
        ensemble_by_repeat[repeat_id] = {
            "probability": ensemble_p,
            "uncertainty": entropy,
            "predictive_entropy": entropy,
            "model_disagreement": disagreement,
            "model_matrix": matrix,
        }

        row = {"repeat": repeat_id + 1, "model": "Ensemble"}
        row.update(predictive_metrics(y.to_numpy(), ensemble_p))
        metric_rows.append(row)

    return pd.DataFrame(metric_rows), ensemble_by_repeat


# -----------------------------------------------------------------------------
# Decision intelligence
# -----------------------------------------------------------------------------

def percentile_rank(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def build_decision_table(df, ensemble_info, cfg):
    d = pd.DataFrame({
        "CustomerID": df["CustomerID"].astype(str).values,
        TARGET: df[TARGET].astype(int).values,
        "churn_probability": np.asarray(ensemble_info["probability"], dtype=float),
        "uncertainty": np.asarray(ensemble_info["uncertainty"], dtype=float),
        "predictive_entropy": np.asarray(
            ensemble_info.get("predictive_entropy", ensemble_info["uncertainty"]),
            dtype=float,
        ),
        "model_disagreement": np.asarray(
            ensemble_info.get("model_disagreement", np.zeros(len(df))), dtype=float
        ),
        "CLTV": df["CLTV"].astype(float).values,
        "Monthly Charges": df["Monthly Charges"].astype(float).values,
        "Tenure Months": df["Tenure Months"].astype(float).values,
    })

    d["cltv_percentile"] = percentile_rank(d["CLTV"].values)
    d["annual_margin_value"] = (
        d["Monthly Charges"]
        * cfg.value_horizon_months
        * cfg.gross_margin_rate
    )
    return d


def business_vectors(d, cfg: BusinessConfig):
    p = d["churn_probability"].to_numpy(float)
    uncertainty = d["uncertainty"].to_numpy(float)
    cltv_pct = d["cltv_percentile"].to_numpy(float)
    annual_value = d["annual_margin_value"].to_numpy(float)

    offer_utility = p * cfg.retention_success * annual_value - cfg.offer_cost

    # A reviewer can either recommend an offer correctly or create a false positive.
    expected_review_offer_probability = (
        p * cfg.human_accuracy
        + (1 - p) * (1 - cfg.human_accuracy)
    )
    review_expected_cost = (
        cfg.human_review_cost
        + cfg.offer_cost * expected_review_offer_probability
    )
    review_utility = (
        p * cfg.human_accuracy * cfg.retention_success * annual_value
        - review_expected_cost
    )

    return {
        "p": p,
        "uncertainty": uncertainty,
        "cltv_pct": cltv_pct,
        "annual_value": annual_value,
        "offer_utility": offer_utility,
        "review_utility": review_utility,
        "review_expected_cost": review_expected_cost,
    }


def allocate_mixed(candidates, budget):
    """
    Greedy utility-per-expected-cost allocation under a common budget.
    Candidate tuple = (row_index, action, priority, expected_cost).
    """
    ranked = sorted(
        candidates,
        key=lambda x: (x[2] / max(x[3], 1e-12), x[2]),
        reverse=True,
    )
    selected = {}
    spent = 0.0

    for idx, action, priority, expected_cost in ranked:
        if priority <= 0:
            continue
        if spent + expected_cost <= budget:
            selected[idx] = action
            spent += expected_cost
    return selected, spent


def finalize_actions(n, mapping):
    result = np.array(["No Action"] * n, dtype=object)
    for idx, action in mapping.items():
        result[idx] = action
    return result


def policy_probability_threshold(d, cfg, budget):
    v = business_vectors(d, cfg)
    candidates = [
        (i, "Retention Offer", float(v["p"][i]), float(cfg.offer_cost))
        for i in range(len(d))
        if v["p"][i] >= 0.50
    ]
    selected, _ = allocate_mixed(candidates, budget)
    return finalize_actions(len(d), selected)


def policy_cost_aware(d, cfg, budget):
    v = business_vectors(d, cfg)
    candidates = [
        (i, "Retention Offer", float(v["offer_utility"][i]), float(cfg.offer_cost))
        for i in range(len(d))
        if v["p"][i] >= cfg.minimum_churn_risk and v["offer_utility"][i] > 0
    ]
    selected, _ = allocate_mixed(candidates, budget)
    return finalize_actions(len(d), selected)


def policy_uncertainty_review(d, cfg, budget):
    v = business_vectors(d, cfg)
    candidates = []

    for i in range(len(d)):
        if v["p"][i] < cfg.minimum_churn_risk:
            continue

        if (
            v["uncertainty"][i] >= cfg.uncertainty_threshold
            and v["review_utility"][i] > 0
        ):
            candidates.append((
                i,
                "Human Review",
                float(v["review_utility"][i]),
                float(v["review_expected_cost"][i]),
            ))
        elif v["offer_utility"][i] > 0:
            candidates.append((
                i,
                "Retention Offer",
                float(v["offer_utility"][i]),
                float(cfg.offer_cost),
            ))

    selected, _ = allocate_mixed(candidates, budget)
    return finalize_actions(len(d), selected)


def policy_kgdi(
    d,
    cfg,
    budget,
    use_uncertainty=True,
    use_cltv=True,
    use_human_review=True,
):
    """
    Proposed KGDI policy.

    Economic utility determines eligibility.
    Predictive entropy determines ambiguous cases for human review.
    CLTV modifies strategic ranking only and is never interpreted as currency.
    All baselines receive the same budget.
    """
    v = business_vectors(d, cfg)
    candidates = []

    for i in range(len(d)):
        if v["p"][i] < cfg.minimum_churn_risk:
            continue

        high_value = v["cltv_pct"][i] >= cfg.high_value_quantile
        standard_uncertain = v["uncertainty"][i] >= cfg.uncertainty_threshold
        value_sensitive_review = (
            high_value
            and v["uncertainty"][i] >= cfg.moderate_uncertainty_threshold
            and v["review_utility"][i] > 0
            and (
                v["offer_utility"][i] <= 0
                or v["review_utility"][i]
                >= cfg.review_utility_tolerance * v["offer_utility"][i]
            )
        )

        route_review = (
            use_human_review
            and use_uncertainty
            and (standard_uncertain or value_sensitive_review)
        )

        if route_review and v["review_utility"][i] > 0:
            base_utility = float(v["review_utility"][i])
            expected_cost = float(v["review_expected_cost"][i])
            action = "Human Review"
        elif v["offer_utility"][i] > 0:
            base_utility = float(v["offer_utility"][i])
            expected_cost = float(cfg.offer_cost)
            action = "Retention Offer"
        else:
            continue

        priority = base_utility
        if use_cltv:
            priority *= 1.0 + cfg.cltv_priority_weight * float(v["cltv_pct"][i])

        candidates.append((i, action, priority, expected_cost))

    selected, _ = allocate_mixed(candidates, budget)
    return finalize_actions(len(d), selected)


def customer_contributions(d, actions, cfg):
    y = d[TARGET].to_numpy(int)
    cltv_pct = d["cltv_percentile"].to_numpy(float)
    annual_value = d["annual_margin_value"].to_numpy(float)

    offer = actions == "Retention Offer"
    review = actions == "Human Review"
    intervention = offer | review

    cost = np.zeros(len(d), dtype=float)
    saved = np.zeros(len(d), dtype=float)
    unnecessary_cost = np.zeros(len(d), dtype=float)

    # Direct offer path.
    cost[offer] = cfg.offer_cost
    saved[offer] = y[offer] * cfg.retention_success * annual_value[offer]
    unnecessary_cost[offer & (y == 0)] = cfg.offer_cost

    # Human-review path.
    # True churner -> correct-positive follow-up offer with probability accuracy.
    # Non-churner -> false-positive follow-up offer with probability 1-accuracy.
    review_offer_prob = (
        y[review] * cfg.human_accuracy
        + (1 - y[review]) * (1 - cfg.human_accuracy)
    )
    cost[review] = cfg.human_review_cost + cfg.offer_cost * review_offer_prob
    saved[review] = (
        y[review]
        * cfg.human_accuracy
        * cfg.retention_success
        * annual_value[review]
    )

    nonchurn_review = review & (y == 0)
    unnecessary_cost[nonchurn_review] = (
        cfg.human_review_cost
        + cfg.offer_cost * (1 - cfg.human_accuracy)
    )

    return pd.DataFrame({
        "cost": cost,
        "saved_value": saved,
        "net_benefit": saved - cost,
        "unnecessary_intervention_cost": unnecessary_cost,
        "intervened": intervention.astype(int),
        "true_churn": y,
        "high_value_churn": (
            (y == 1) & (cltv_pct >= cfg.high_value_quantile)
        ).astype(int),
    })


def expected_selected_cost(d, actions, cfg):
    """Ex-ante expected cost used by the budget-constrained policy allocator."""
    v = business_vectors(d, cfg)
    offer = actions == "Retention Offer"
    review = actions == "Human Review"
    return float(
        np.sum(offer) * cfg.offer_cost
        + np.sum(v["review_expected_cost"][review])
    )


def evaluate_policy(d, actions, cfg, planning_budget=None):
    c = customer_contributions(d, actions, cfg)
    total_churn = int(c["true_churn"].sum())
    total_high_value_churn = int(c["high_value_churn"].sum())

    reached_churn = int(
        ((c["true_churn"] == 1) & (c["intervened"] == 1)).sum()
    )
    reached_high = int(
        ((c["high_value_churn"] == 1) & (c["intervened"] == 1)).sum()
    )

    expected_cost = expected_selected_cost(d, actions, cfg)
    outcome_cost = float(c["cost"].sum())

    result = {
        "offers": int(np.sum(actions == "Retention Offer")),
        "human_reviews": int(np.sum(actions == "Human Review")),
        "interventions": int(c["intervened"].sum()),
        "intervention_rate": float(c["intervened"].mean()),
        "churn_reach_rate": reached_churn / total_churn if total_churn else np.nan,
        "high_value_churn_reach_rate": (
            reached_high / total_high_value_churn if total_high_value_churn else np.nan
        ),
        "total_cost": outcome_cost,
        "outcome_anchored_cost": outcome_cost,
        "expected_selected_cost": expected_cost,
        "expected_vs_outcome_cost_delta": outcome_cost - expected_cost,
        "saved_value_proxy": float(c["saved_value"].sum()),
        "net_benefit_proxy": float(c["net_benefit"].sum()),
        "unnecessary_intervention_cost": float(
            c["unnecessary_intervention_cost"].sum()
        ),
    }
    if planning_budget is not None:
        result["planning_budget"] = float(planning_budget)
        result["budget_utilization"] = (
            expected_cost / planning_budget if planning_budget > 0 else np.nan
        )
    return result


def run_decision_experiment(df, ensemble_by_repeat, cfg):
    rows = []
    decision_tables = {}

    for repeat_id, info in ensemble_by_repeat.items():
        d = build_decision_table(df, info, cfg)
        budget = len(d) * cfg.offer_cost * cfg.budget_fraction

        policies = {
            "Probability_Threshold": policy_probability_threshold(d, cfg, budget),
            "Cost_Aware": policy_cost_aware(d, cfg, budget),
            "Uncertainty_Review": policy_uncertainty_review(d, cfg, budget),
            "Proposed_KGDI": policy_kgdi(d, cfg, budget),
        }

        for name, actions in policies.items():
            row = {"repeat": repeat_id + 1, "strategy": name, "budget": float(budget)}
            row.update(evaluate_policy(d, actions, cfg, planning_budget=budget))
            rows.append(row)
            d[f"action_{name}"] = actions

        decision_tables[repeat_id] = d

    return pd.DataFrame(rows), decision_tables


# -----------------------------------------------------------------------------
# Robustness / inference
# -----------------------------------------------------------------------------

def repeated_stability_summary(decision_metrics):
    """
    Descriptive repeated-CV stability only. No independence claim is made across
    repeated CV runs on the same dataset.
    """
    rows = []
    pivot = decision_metrics.pivot(
        index="repeat", columns="strategy", values="net_benefit_proxy"
    )
    if "Proposed_KGDI" not in pivot.columns:
        return pd.DataFrame()

    for baseline in ["Probability_Threshold", "Cost_Aware", "Uncertainty_Review"]:
        if baseline not in pivot.columns:
            continue
        diff = (pivot["Proposed_KGDI"] - pivot[baseline]).dropna()
        rows.append({
            "comparison": f"Proposed_KGDI - {baseline}",
            "repeats": int(len(diff)),
            "mean_delta": float(diff.mean()),
            "sd_delta": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
            "median_delta": float(diff.median()),
            "fraction_repeats_positive": float(np.mean(diff > 0)),
        })
    return pd.DataFrame(rows)


def bootstrap_policy_delta(d, cfg, n_boot, seed):
    """
    Paired customer bootstrap with policy re-optimization inside each resample.

    For every bootstrap sample:
    1) resample customers with replacement;
    2) recompute CLTV percentile ranks;
    3) recompute the available budget;
    4) rerun Uncertainty Review and KGDI policies;
    5) evaluate their net-benefit difference.

    Predictive probabilities are treated as fixed OOF estimates. Model instability
    is evaluated separately by repeated OOF cross-validation.
    """
    rng = np.random.default_rng(seed)
    n = len(d)
    deltas = np.empty(n_boot, dtype=float)

    base_budget = len(d) * cfg.offer_cost * cfg.budget_fraction
    base_unc = policy_uncertainty_review(d, cfg, base_budget)
    base_kgdi = policy_kgdi(d, cfg, base_budget)
    observed_delta = (
        evaluate_policy(d, base_kgdi, cfg, planning_budget=base_budget)["net_benefit_proxy"]
        - evaluate_policy(d, base_unc, cfg, planning_budget=base_budget)["net_benefit_proxy"]
    )

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = d.iloc[idx].reset_index(drop=True).copy()
        sample["cltv_percentile"] = percentile_rank(sample["CLTV"].to_numpy(float))

        budget = len(sample) * cfg.offer_cost * cfg.budget_fraction
        unc = policy_uncertainty_review(sample, cfg, budget)
        kgdi = policy_kgdi(sample, cfg, budget)

        unc_nb = evaluate_policy(sample, unc, cfg, planning_budget=budget)["net_benefit_proxy"]
        kgdi_nb = evaluate_policy(sample, kgdi, cfg, planning_budget=budget)["net_benefit_proxy"]
        deltas[b] = kgdi_nb - unc_nb

    return {
        "observed_delta": float(observed_delta),
        "bootstrap_mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(deltas, 2.5)),
        "ci_high": float(np.percentile(deltas, 97.5)),
        "bootstrap_probability_gt_0": float(np.mean(deltas > 0)),
        "bootstrap_samples": int(n_boot),
    }


def run_ablations(d, cfg):
    budget = len(d) * cfg.offer_cost * cfg.budget_fraction
    variants = {
        "Full_KGDI": policy_kgdi(d, cfg, budget, True, True, True),
        "No_Uncertainty": policy_kgdi(d, cfg, budget, False, True, True),
        "No_CLTV_Priority": policy_kgdi(d, cfg, budget, True, False, True),
        "No_Human_Review": policy_kgdi(d, cfg, budget, True, True, False),
    }

    rows = []
    for name, actions in variants.items():
        row = {"variant": name}
        row.update(evaluate_policy(d, actions, cfg, planning_budget=budget))
        rows.append(row)
    return pd.DataFrame(rows)



def policy_distinctness_report(d, cfg, kgdi, uncertainty, costaware):
    """Quantify whether KGDI materially changes actions and which components do so."""
    budget = len(d) * cfg.offer_cost * cfg.budget_fraction
    variants = {
        "KGDI_vs_Uncertainty_Review": (kgdi, uncertainty),
        "KGDI_vs_Cost_Aware": (kgdi, costaware),
        "CLTV_component_Full_vs_No_CLTV": (
            kgdi, policy_kgdi(d, cfg, budget, True, False, True)
        ),
        "Uncertainty_component_Full_vs_No_Uncertainty": (
            kgdi, policy_kgdi(d, cfg, budget, False, True, True)
        ),
        "HumanReview_component_Full_vs_No_Human_Review": (
            kgdi, policy_kgdi(d, cfg, budget, True, True, False)
        ),
    }
    rows = []
    for comparison, (a, b) in variants.items():
        changed = a != b
        rows.append({
            "comparison": comparison,
            "different_actions": int(np.sum(changed)),
            "disagreement_rate": float(np.mean(changed)),
            "kgdi_no_action": int(np.sum(a == "No Action")),
            "kgdi_retention_offer": int(np.sum(a == "Retention Offer")),
            "kgdi_human_review": int(np.sum(a == "Human Review")),
            "comparator_no_action": int(np.sum(b == "No Action")),
            "comparator_retention_offer": int(np.sum(b == "Retention Offer")),
            "comparator_human_review": int(np.sum(b == "Human Review")),
        })
    return pd.DataFrame(rows)


def subgroup_audit(df, d, actions):
    audit = pd.DataFrame({
        "Gender": df["Gender"].astype(str).values if "Gender" in df else "Unknown",
        "Senior Citizen": (
            df["Senior Citizen"].astype(str).values
            if "Senior Citizen" in df else "Unknown"
        ),
        TARGET: d[TARGET].values,
        "intervened": (actions != "No Action").astype(int),
    })

    rows = []
    for attribute in ["Gender", "Senior Citizen"]:
        for group, g in audit.groupby(attribute):
            churners = g[g[TARGET] == 1]
            rows.append({
                "attribute": attribute,
                "group": group,
                "n": int(len(g)),
                "churn_rate": float(g[TARGET].mean()),
                "intervention_rate": float(g["intervened"].mean()),
                "churn_reach_rate": (
                    float(churners["intervened"].mean()) if len(churners) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def evaluate_sensitivity_scenario(d, base_cfg, scenario_name, **changes):
    cfg = BusinessConfig(**asdict(base_cfg))
    for key, value in changes.items():
        setattr(cfg, key, value)

    # annual_margin_value depends on margin rate and horizon, so recompute it.
    s = d.copy()
    s["annual_margin_value"] = (
        s["Monthly Charges"] * cfg.value_horizon_months * cfg.gross_margin_rate
    )

    budget = len(s) * cfg.offer_cost * cfg.budget_fraction
    uncertainty = policy_uncertainty_review(s, cfg, budget)
    kgdi = policy_kgdi(s, cfg, budget)
    costaware = policy_cost_aware(s, cfg, budget)

    m_u = evaluate_policy(s, uncertainty, cfg, planning_budget=budget)
    m_k = evaluate_policy(s, kgdi, cfg, planning_budget=budget)
    m_c = evaluate_policy(s, costaware, cfg, planning_budget=budget)

    return {
        "scenario": scenario_name,
        **changes,
        "kgdi_net_benefit": m_k["net_benefit_proxy"],
        "uncertainty_net_benefit": m_u["net_benefit_proxy"],
        "costaware_net_benefit": m_c["net_benefit_proxy"],
        "kgdi_delta_vs_uncertainty": (
            m_k["net_benefit_proxy"] - m_u["net_benefit_proxy"]
        ),
        "kgdi_high_value_churn_reach": m_k["high_value_churn_reach_rate"],
    }


def sensitivity_analysis(d, base_cfg):
    """
    Compact but broad robustness design:
    A) 3x3x3 operational grid: offer cost x success x budget;
    B) one-factor-at-a-time reviewer-proofing for margin, reviewer accuracy,
       horizon, uncertainty threshold, and CLTV priority weight.
    """
    rows = []

    for cost in [50.0, 100.0, 150.0]:
        for success in [0.20, 0.35, 0.50]:
            for budget_fraction in [0.10, 0.20, 0.30]:
                row = evaluate_sensitivity_scenario(
                    d,
                    base_cfg,
                    "operational_grid",
                    offer_cost=cost,
                    retention_success=success,
                    budget_fraction=budget_fraction,
                )
                rows.append(row)

    ofat = {
        "gross_margin_rate": [0.40, 0.60, 0.80],
        "human_accuracy": [0.70, 0.85, 0.95],
        "value_horizon_months": [6, 12, 18],
        "uncertainty_threshold": [0.55, 0.65, 0.75],
        "cltv_priority_weight": [0.00, 0.03, 0.06],
    }

    for parameter, values in ofat.items():
        for value in values:
            rows.append(evaluate_sensitivity_scenario(
                d,
                base_cfg,
                f"OFAT_{parameter}",
                **{parameter: value},
            ))

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Tables / figures
# -----------------------------------------------------------------------------

def mean_sd_string(values, digits=3):
    v = np.asarray(values, dtype=float)
    if len(v) <= 1:
        return f"{v.mean():.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def make_table_1(model_metrics):
    rows = []
    order = [
        "LogisticRegression", "RandomForest", "XGBoost", "LightGBM", "Ensemble"
    ]
    for model in order:
        g = model_metrics[model_metrics["model"] == model]
        if g.empty:
            continue
        rows.append({
            "Model": model,
            "AUROC": mean_sd_string(g["roc_auc"], 3),
            "PR-AUC": mean_sd_string(g["pr_auc"], 3),
            "F1": mean_sd_string(g["f1"], 3),
            "Brier ↓": mean_sd_string(g["brier"], 3),
            "ECE ↓": mean_sd_string(g["ece"], 3),
        })
    return pd.DataFrame(rows)


def make_table_2(decision_metrics, bootstrap_result):
    rows = []
    order = [
        "Probability_Threshold", "Cost_Aware", "Uncertainty_Review", "Proposed_KGDI"
    ]

    for strategy in order:
        g = decision_metrics[decision_metrics["strategy"] == strategy]
        if g.empty:
            continue

        delta = ""
        ci = ""
        p_positive = ""
        if strategy == "Proposed_KGDI":
            delta = f"{bootstrap_result['observed_delta']:.2f}"
            ci = (
                f"[{bootstrap_result['ci_low']:.2f}, "
                f"{bootstrap_result['ci_high']:.2f}]"
            )
            p_positive = f"{bootstrap_result['bootstrap_probability_gt_0']:.3f}"

        rows.append({
            "Strategy": strategy,
            "Intervention %": mean_sd_string(100 * g["intervention_rate"], 1),
            "Churn reach %": mean_sd_string(100 * g["churn_reach_rate"], 1),
            "High-value churn reach %": mean_sd_string(
                100 * g["high_value_churn_reach_rate"], 1
            ),
            "Net benefit proxy ($)": mean_sd_string(g["net_benefit_proxy"], 2),
            "KGDI Δ vs uncertainty ($)": delta,
            "95% policy-rerun bootstrap CI": ci,
            "Bootstrap P(Δ>0)": p_positive,
        })
    return pd.DataFrame(rows)


def figure_decision_map(d, actions, output_path):
    action_order = ["No Action", "Retention Offer", "Human Review"]
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}

    plt.figure(figsize=(8.6, 6.2))
    sizes = 12 + 45 * (
        d["annual_margin_value"].to_numpy()
        / max(d["annual_margin_value"].max(), 1e-12)
    )

    for action in action_order:
        mask = actions == action
        if not np.any(mask):
            continue
        plt.scatter(
            d.loc[mask, "churn_probability"],
            d.loc[mask, "predictive_entropy"],
            s=sizes[mask],
            alpha=0.45,
            marker=markers[action],
            label=action,
        )

    plt.xlabel("Calibrated ensemble churn probability")
    plt.ylabel("Predictive entropy")
    plt.title("KGDI Decision Map")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def figure_sensitivity(sensitivity, base_budget_fraction, output_path):
    s = sensitivity[
        (sensitivity["scenario"] == "operational_grid")
        & np.isclose(sensitivity["budget_fraction"], base_budget_fraction)
    ].copy()

    costs = sorted(s["offer_cost"].dropna().unique())
    success = sorted(s["retention_success"].dropna().unique())
    matrix = np.full((len(success), len(costs)), np.nan)

    for i, sr in enumerate(success):
        for j, cost in enumerate(costs):
            value = s[
                np.isclose(s["retention_success"], sr)
                & np.isclose(s["offer_cost"], cost)
            ]["kgdi_delta_vs_uncertainty"]
            if len(value):
                matrix[i, j] = float(value.iloc[0])

    plt.figure(figsize=(7.5, 5.5))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="KGDI net-benefit advantage ($)")
    plt.xticks(range(len(costs)), [f"${int(x)}" for x in costs])
    plt.yticks(range(len(success)), [f"{int(100*x)}%" for x in success])
    plt.xlabel("Retention-offer cost")
    plt.ylabel("Retention-success assumption")
    plt.title(
        f"KGDI Robustness at {int(100 * base_budget_fraction)}% Budget Fraction"
    )

    for i in range(len(success)):
        for j in range(len(costs)):
            if not np.isnan(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def current_script_sha256():
    try:
        return sha256_file(Path(__file__).resolve())
    except Exception:
        return None


def reproducibility_metadata(args, cfg, repeats, n_boot):
    return {
        "study_title": (
            "Knowledge-Guided Decision Intelligence for Cost-Aware Customer "
            "Churn Intervention Under Predictive Uncertainty"
        ),
        "data_path": str(args.data),
        "dataset_sha256": sha256_file(args.data),
        "main_py_sha256": current_script_sha256(),
        "git_commit_sha": git_commit_sha(),
        "mode": args.mode,
        "folds": args.folds,
        "repeats": repeats,
        "seed": args.seed,
        "bootstrap_samples": n_boot,
        "threads_per_model": args.threads,
        "cpu_count_visible": os.cpu_count(),
        "business_parameters": asdict(cfg),
        "model_feature_allowlist": MODEL_FEATURES,
        "budget_definition": (
            "Planning constraint is enforced using ex-ante expected selected cost. "
            "Outcome-anchored realized/simulated cost is reported separately."
        ),
        "business_value_definition": (
            "Simulated revenue-at-risk proxy = Monthly Charges × value horizon months "
            "× gross margin rate. IBM CLTV is used only as a dimensionless strategic "
            "priority index and is not interpreted as currency."
        ),
        "uncertainty_definition": (
            "Primary uncertainty = normalized binary predictive entropy of the "
            "calibrated ensemble probability. Cross-model standard deviation is "
            "reported separately as a model-disagreement diagnostic."
        ),
        "inference_definition": (
            "Main KGDI effect uses paired customer bootstrap with policy rerouting and "
            "budget reallocation inside every bootstrap resample. Repeated OOF CV is "
            "used for stability summaries, not as independent-sample inference."
        ),
        "methodological_disclosure": (
            "IBM customer variables and churn outcomes are observed dataset fields. "
            "Retention cost, intervention success, human-review cost/accuracy, gross "
            "margin, value horizon, and budget are simulated assumptions."
        ),
        "software_versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "platform": platform.platform(),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Do not globally suppress warnings in research runs. Filter only this known
    # benign sklearn/LightGBM feature-name interoperability warning.
    warnings.filterwarnings(
        "ignore",
        message=(
            "X does not have valid feature names, but LGBMClassifier was fitted "
            "with feature names"
        ),
        category=UserWarning,
    )
    args = parse_args()
    validate_args(args)

    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "quick" else 5)
    n_boot = args.bootstrap if args.bootstrap is not None else (200 if args.mode == "quick" else 1000)

    cfg = BusinessConfig(
        offer_cost=args.offer_cost,
        retention_success=args.retention_success,
        human_review_cost=args.human_review_cost,
        human_accuracy=args.human_accuracy,
        budget_fraction=args.budget_fraction,
        gross_margin_rate=args.gross_margin_rate,
        value_horizon_months=args.value_horizon_months,
        minimum_churn_risk=args.minimum_churn_risk,
        uncertainty_threshold=args.uncertainty_threshold,
        moderate_uncertainty_threshold=args.moderate_uncertainty_threshold,
        high_value_quantile=args.high_value_quantile,
        cltv_priority_weight=args.cltv_priority_weight,
        review_utility_tolerance=args.review_utility_tolerance,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("KBS CHURN DECISION INTELLIGENCE")
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Outer CV: {args.folds} folds × {repeats} repeats")
    print(f"Threads/model: {args.threads}")
    print(f"Policy-rerun bootstrap samples: {n_boot}")

    df, audit = clean_data(load_data(args.data))
    features, numeric, categorical = select_features(df)

    audit.update({
        "ai_feature_count": len(features),
        "ai_features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "leakage_excluded": LEAKAGE_COLUMNS,
        "business_only": BUSINESS_ONLY_COLUMNS,
        "audit_only": AUDIT_ONLY_COLUMNS,
        "model_feature_allowlist": MODEL_FEATURES,
    })

    with open(output_dir / "data_audit.json", "w") as f:
        json.dump(audit, f, indent=2)
    save_column_roles(df, output_dir)

    print(f"Rows: {len(df)} | Churn rate: {df[TARGET].mean():.3f}")
    print(f"AI predictors: {len(features)}")

    model_metrics, ensemble_by_repeat = run_repeated_oof(
        df, features, numeric, categorical, args
    )
    model_metrics.to_csv(
        output_dir / "SUPPLEMENT_predictive_metrics_by_repeat.csv", index=False
    )

    decision_metrics, _ = run_decision_experiment(df, ensemble_by_repeat, cfg)
    decision_metrics.to_csv(
        output_dir / "SUPPLEMENT_decision_metrics_by_repeat.csv", index=False
    )

    stability = repeated_stability_summary(decision_metrics)
    stability.to_csv(
        output_dir / "SUPPLEMENT_repeated_cv_stability.csv", index=False
    )

    # Aggregate repeated OOF predictions for final descriptive/inferential policy.
    stacked_p = np.vstack([
        ensemble_by_repeat[r]["probability"] for r in sorted(ensemble_by_repeat)
    ])
    stacked_dis = np.vstack([
        ensemble_by_repeat[r]["model_disagreement"] for r in sorted(ensemble_by_repeat)
    ])

    final_probability = stacked_p.mean(axis=0)
    final_entropy = binary_entropy(final_probability)
    final_disagreement = stacked_dis.mean(axis=0)

    final_info = {
        "probability": final_probability,
        "uncertainty": final_entropy,
        "predictive_entropy": final_entropy,
        "model_disagreement": final_disagreement,
    }
    final_d = build_decision_table(df, final_info, cfg)
    budget = len(final_d) * cfg.offer_cost * cfg.budget_fraction

    final_probability_threshold = policy_probability_threshold(final_d, cfg, budget)
    final_costaware = policy_cost_aware(final_d, cfg, budget)
    final_uncertainty = policy_uncertainty_review(final_d, cfg, budget)
    final_kgdi = policy_kgdi(final_d, cfg, budget)

    distinctness = policy_distinctness_report(
        final_d, cfg, final_kgdi, final_uncertainty, final_costaware
    )
    distinctness.to_csv(
        output_dir / "SUPPLEMENT_policy_distinctness.csv", index=False
    )
    kgdi_vs_unc = distinctness.loc[
        distinctness["comparison"] == "KGDI_vs_Uncertainty_Review",
        "different_actions",
    ]
    if len(kgdi_vs_unc) and int(kgdi_vs_unc.iloc[0]) == 0:
        warnings.warn(
            "KGDI and Uncertainty Review generated identical actions under the "
            "current configuration. Do not claim incremental KGDI benefit for "
            "this configuration.",
            RuntimeWarning,
        )

    bootstrap_result = bootstrap_policy_delta(
        final_d, cfg, n_boot, args.seed + 909
    )
    with open(
        output_dir / "SUPPLEMENT_bootstrap_KGDI_vs_uncertainty.json", "w"
    ) as f:
        json.dump(bootstrap_result, f, indent=2)

    ablations = run_ablations(final_d, cfg)
    ablations.to_csv(output_dir / "SUPPLEMENT_ablation_study.csv", index=False)

    groups = subgroup_audit(df, final_d, final_kgdi)
    groups.to_csv(output_dir / "SUPPLEMENT_subgroup_audit.csv", index=False)

    sensitivity = sensitivity_analysis(final_d, cfg)
    sensitivity.to_csv(output_dir / "SUPPLEMENT_sensitivity_full.csv", index=False)

    table1 = make_table_1(model_metrics)
    table2 = make_table_2(decision_metrics, bootstrap_result)
    table1.to_csv(output_dir / "TABLE_1_predictive_performance.csv", index=False)
    table2.to_csv(output_dir / "TABLE_2_decision_performance.csv", index=False)

    figure_decision_map(
        final_d,
        final_kgdi,
        output_dir / "FIGURE_1_KGDI_decision_map.png",
    )
    figure_sensitivity(
        sensitivity,
        cfg.budget_fraction,
        output_dir / "FIGURE_2_KGDI_sensitivity.png",
    )

    final_predictions = final_d.copy()
    final_predictions["action_Probability_Threshold"] = final_probability_threshold
    final_predictions["action_Cost_Aware"] = final_costaware
    final_predictions["action_Uncertainty_Review"] = final_uncertainty
    final_predictions["action_Proposed_KGDI"] = final_kgdi
    final_predictions.to_csv(
        output_dir / "SUPPLEMENT_customer_level_oof_decisions.csv", index=False
    )

    run_config = reproducibility_metadata(args, cfg, repeats, n_boot)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print("\nPRIMARY TABLE 1")
    print(table1.to_string(index=False))
    print("\nPRIMARY TABLE 2")
    print(table2.to_string(index=False))
    print("\nPOLICY-RERUN BOOTSTRAP: KGDI vs UNCERTAINTY REVIEW")
    print(json.dumps(bootstrap_result, indent=2))
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
