#!/usr/bin/env python3
"""
Knowledge-Guided Decision Intelligence for Cost-Aware Customer Churn
Intervention Under Predictive Uncertainty

Single-file experimental pipeline for a Knowledge-Based Systems short communication.

Core design:
1) leakage-safe repeated out-of-fold churn prediction;
2) probability calibration using a held-out calibration partition;
3) composite predictive uncertainty from entropy + model disagreement;
4) unit-consistent business utility using a 12-month gross-margin revenue-at-risk proxy;
5) CLTV used only as a dimensionless strategic-priority index, not as dollars;
6) fair budget-matched policy comparisons;
7) repeated-CV robustness, customer-level paired bootstrap, ablations, subgroup audit;
8) only two primary tables and two manuscript-ready figures.

IBM variables are observed dataset fields.
Business costs/effectiveness/margin assumptions are explicit simulation parameters.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import platform
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
from scipy.stats import wilcoxon

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
from lightgbm import LGBMClassifier


TARGET = "Churn Value"

LEAKAGE_COLUMNS = ["Churn Label", "Churn Score", "Churn Reason"]

IDENTIFIER_GEO_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City",
    "Zip Code", "Lat Long", "Latitude", "Longitude",
]

# CLTV is an IBM customer-value index. It is not treated as currency.
BUSINESS_ONLY_COLUMNS = ["CLTV"]

# Excluded from the primary model; retained only for post-hoc audit.
AUDIT_ONLY_COLUMNS = ["Gender", "Senior Citizen"]


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Q1-style AI + Decision Intelligence churn experiment"
    )
    p.add_argument("--data", required=True, help=".xlsx or ZIP containing the IBM workbook")
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
    return p.parse_args()


def validate_args(args):
    if args.folds < 3:
        raise ValueError("--folds must be >= 3")
    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    for name in [
        "retention_success", "human_accuracy", "budget_fraction", "gross_margin_rate"
    ]:
        value = getattr(args, name.replace("-", "_"), None)
        if value is not None and not (0 < value <= 1):
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1]")


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
                raise ValueError("No .xlsx workbook found inside ZIP.")
            exact = [
                f for f in candidates
                if Path(f).name.lower() == "telco_customer_churn.xlsx"
            ]
            selected = exact[0] if exact else sorted(candidates)[0]
            print(f"Workbook selected: {selected}")
            with z.open(selected) as f:
                return pd.read_excel(io.BytesIO(f.read()))

    raise ValueError("Dataset must be .xlsx or .zip containing an .xlsx workbook.")


def clean_data(df: pd.DataFrame):
    required = [
        "CustomerID", "Tenure Months", "Monthly Charges",
        "Total Charges", "CLTV", TARGET
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df = df.copy()

    # Numeric normalization.
    for c in ["Tenure Months", "Monthly Charges", "Total Charges", "CLTV", TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if df[TARGET].isna().any():
        raise ValueError("Churn Value contains non-numeric/missing values.")

    df[TARGET] = df[TARGET].astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn Value must be binary 0/1.")

    if df["CustomerID"].isna().any() or df["CustomerID"].duplicated().any():
        raise ValueError("CustomerID must be complete and unique.")

    zero_tenure_blank = df["Total Charges"].isna() & (df["Tenure Months"] == 0)
    fixed_zero = int(zero_tenure_blank.sum())
    df.loc[zero_tenure_blank, "Total Charges"] = 0.0

    remaining_total_missing = int(df["Total Charges"].isna().sum())
    if remaining_total_missing:
        # Conservative fallback; model pipeline also imputes remaining numeric values.
        df["Total Charges"] = df["Total Charges"].fillna(df["Total Charges"].median())

    if (df["Monthly Charges"] < 0).any() or (df["CLTV"] < 0).any():
        raise ValueError("Negative Monthly Charges or CLTV values detected.")

    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "churners": int(df[TARGET].sum()),
        "non_churners": int((df[TARGET] == 0).sum()),
        "churn_rate": float(df[TARGET].mean()),
        "duplicate_customer_ids": int(df["CustomerID"].duplicated().sum()),
        "zero_tenure_total_charges_set_to_zero": fixed_zero,
        "remaining_total_charges_imputed": remaining_total_missing,
    }
    return df, audit


def select_features(df):
    excluded = set(
        LEAKAGE_COLUMNS
        + IDENTIFIER_GEO_COLUMNS
        + BUSINESS_ONLY_COLUMNS
        + AUDIT_ONLY_COLUMNS
        + [TARGET]
    )
    features = [c for c in df.columns if c not in excluded]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return features, numeric, categorical


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
            colsample_bytree=0.90,
            class_weight="balanced",
            random_state=seed,
            verbosity=-1,
            n_jobs=threads,
        ),
    }


class PlattScaler:
    """Leakage-safe sigmoid calibrator fitted only on a calibration partition."""
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    @staticmethod
    def _logit(p):
        p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, raw_probability, y):
        self.model.fit(self._logit(raw_probability), np.asarray(y))
        return self

    def predict(self, raw_probability):
        return self.model.predict_proba(self._logit(raw_probability))[:, 1]


def fit_calibrated_model(
    model,
    preprocessor,
    X_train,
    y_train,
    X_test,
    seed,
):
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
    calibrated_test = scaler.predict(raw_test)
    return np.clip(calibrated_test, 1e-6, 1 - 1e-6)


def expected_calibration_error(y_true, p, bins=10):
    y_true = np.asarray(y_true)
    p = np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
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
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def composite_uncertainty(model_probabilities):
    """
    model_probabilities: shape [n_models, n_customers]
    Entropy captures closeness to 0.5.
    Disagreement captures model-to-model dispersion.
    Both are normalized to approximately [0,1].
    """
    probs = np.asarray(model_probabilities, dtype=float)
    mean_p = probs.mean(axis=0)
    entropy = binary_entropy(mean_p)
    disagreement = np.clip(2.0 * probs.std(axis=0), 0.0, 1.0)
    uncertainty = np.clip(0.70 * entropy + 0.30 * disagreement, 0.0, 1.0)
    return mean_p, uncertainty, entropy, disagreement


def percentile_rank(values):
    s = pd.Series(values)
    return s.rank(method="average", pct=True).to_numpy(dtype=float)


def business_vectors(d, cfg: BusinessConfig):
    p = d["churn_probability"].to_numpy(float)
    uncertainty = d["uncertainty"].to_numpy(float)
    cltv_pct = d["cltv_percentile"].to_numpy(float)
    annual_value = d["annual_margin_value"].to_numpy(float)

    offer_utility = p * cfg.retention_success * annual_value - cfg.offer_cost

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
    candidates: list of (index, action, priority, expected_cost)
    Greedy utility-per-cost allocation under a common budget.
    """
    actions = {}
    ranked = sorted(
        candidates,
        key=lambda x: (x[2] / max(x[3], 1e-9), x[2]),
        reverse=True,
    )
    spent = 0.0
    for idx, action, priority, expected_cost in ranked:
        if priority <= 0:
            continue
        if spent + expected_cost <= budget:
            actions[idx] = action
            spent += expected_cost
    return actions, spent


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
                i, "Human Review",
                float(v["review_utility"][i]),
                float(v["review_expected_cost"][i]),
            ))
        elif v["offer_utility"][i] > 0:
            candidates.append((
                i, "Retention Offer",
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
    Proposed KGDI:
    - economic eligibility from expected utility;
    - uncertainty routes ambiguous cases to human review;
    - high-value CLTV index adds strategic priority but never becomes currency;
    - same intervention budget as all baselines.
    """
    v = business_vectors(d, cfg)
    candidates = []

    for i in range(len(d)):
        if v["p"][i] < cfg.minimum_churn_risk:
            continue

        high_value = v["cltv_pct"][i] >= cfg.high_value_quantile

        # Standard uncertainty routing is retained. For strategically high-value
        # customers, a moderate-uncertainty case may also be reviewed, but only
        # when its predicted review utility is close to the direct-offer utility.
        # This prevents CLTV knowledge from overriding the economic objective.
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

    # Direct retention offer.
    cost[offer] = cfg.offer_cost
    saved[offer] = (
        y[offer] * cfg.retention_success * annual_value[offer]
    )

    # Human-review path.
    # If true churner: correct positive with probability accuracy.
    # If non-churner: false positive with probability (1-accuracy).
    review_offer_prob = (
        y[review] * cfg.human_accuracy
        + (1 - y[review]) * (1 - cfg.human_accuracy)
    )
    cost[review] = (
        cfg.human_review_cost
        + cfg.offer_cost * review_offer_prob
    )
    saved[review] = (
        y[review]
        * cfg.human_accuracy
        * cfg.retention_success
        * annual_value[review]
    )

    net = saved - cost

    return pd.DataFrame({
        "cost": cost,
        "saved_value": saved,
        "net_benefit": net,
        "intervened": intervention.astype(int),
        "true_churn": y,
        "high_value_churn": (
            (y == 1) & (cltv_pct >= cfg.high_value_quantile)
        ).astype(int),
        "false_offer": ((y == 0) & offer).astype(int),
    })


def evaluate_policy(d, actions, cfg):
    c = customer_contributions(d, actions, cfg)
    total_churn = int(c["true_churn"].sum())
    high_value_churn = int(c["high_value_churn"].sum())

    reached_churn = int(
        ((c["true_churn"] == 1) & (c["intervened"] == 1)).sum()
    )
    reached_high = int(
        ((c["high_value_churn"] == 1) & (c["intervened"] == 1)).sum()
    )

    return {
        "offers": int(np.sum(actions == "Retention Offer")),
        "human_reviews": int(np.sum(actions == "Human Review")),
        "interventions": int(c["intervened"].sum()),
        "intervention_rate": float(c["intervened"].mean()),
        "churn_reach_rate": reached_churn / total_churn if total_churn else np.nan,
        "high_value_churn_reach_rate": (
            reached_high / high_value_churn if high_value_churn else np.nan
        ),
        "total_cost": float(c["cost"].sum()),
        "saved_value_proxy": float(c["saved_value"].sum()),
        "net_benefit_proxy": float(c["net_benefit"].sum()),
        "unnecessary_offer_cost": float(c["false_offer"].sum() * cfg.offer_cost),
    }


def bootstrap_delta(d, actions_a, actions_b, cfg, n_boot, seed):
    """
    Paired customer bootstrap conditional on the already-selected policies.
    Repeated CV separately captures model/policy instability.
    """
    ca = customer_contributions(d, actions_a, cfg)["net_benefit"].to_numpy()
    cb = customer_contributions(d, actions_b, cfg)["net_benefit"].to_numpy()
    delta = ca - cb

    rng = np.random.default_rng(seed)
    n = len(delta)
    stats = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = delta[idx].sum()

    return {
        "delta_mean": float(delta.sum()),
        "ci_low": float(np.percentile(stats, 2.5)),
        "ci_high": float(np.percentile(stats, 97.5)),
        "bootstrap_probability_gt_0": float(np.mean(stats > 0)),
    }


def mean_sd_string(values, digits=3):
    v = np.asarray(values, dtype=float)
    if len(v) <= 1:
        return f"{v.mean():.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


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

    # repeat -> model -> probability array
    repeat_predictions = {
        r: {
            name: np.full(n, np.nan)
            for name in make_models(args.seed, args.threads, args.mode).keys()
        }
        for r in range(repeats)
    }

    fold_rows = []

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

    # Validate complete OOF coverage and build per-repeat metrics.
    per_repeat_model_rows = []
    ensemble_by_repeat = {}

    for repeat_id in range(repeats):
        model_matrix = []
        for model_name, probs in repeat_predictions[repeat_id].items():
            if np.isnan(probs).any():
                raise RuntimeError(
                    f"Incomplete OOF predictions: repeat={repeat_id}, model={model_name}"
                )
            row = {"repeat": repeat_id + 1, "model": model_name}
            row.update(predictive_metrics(y.to_numpy(), probs))
            per_repeat_model_rows.append(row)
            model_matrix.append(probs)

        model_matrix = np.vstack(model_matrix)
        ensemble_p, uncertainty, entropy, disagreement = composite_uncertainty(model_matrix)
        ensemble_by_repeat[repeat_id] = {
            "probability": ensemble_p,
            "uncertainty": uncertainty,
            "entropy": entropy,
            "disagreement": disagreement,
            "model_matrix": model_matrix,
        }

        row = {"repeat": repeat_id + 1, "model": "Ensemble"}
        row.update(predictive_metrics(y.to_numpy(), ensemble_p))
        per_repeat_model_rows.append(row)

    return (
        pd.DataFrame(per_repeat_model_rows),
        ensemble_by_repeat,
        repeat_predictions,
    )


def build_decision_table(df, ensemble_info, cfg):
    d = pd.DataFrame({
        "CustomerID": df["CustomerID"].astype(str).values,
        TARGET: df[TARGET].astype(int).values,
        "churn_probability": ensemble_info["probability"],
        "uncertainty": ensemble_info["uncertainty"],
        "entropy": ensemble_info["entropy"],
        "model_disagreement": ensemble_info["disagreement"],
        "CLTV": df["CLTV"].astype(float).values,
        "cltv_percentile": percentile_rank(df["CLTV"].astype(float).values),
        "Monthly Charges": df["Monthly Charges"].astype(float).values,
        "Tenure Months": df["Tenure Months"].astype(float).values,
    })

    d["annual_margin_value"] = (
        d["Monthly Charges"]
        * cfg.value_horizon_months
        * cfg.gross_margin_rate
    )
    return d


def run_decision_experiment(df, ensemble_by_repeat, cfg):
    all_rows = []
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
            row = {
                "repeat": repeat_id + 1,
                "strategy": name,
                "budget": float(budget),
            }
            row.update(evaluate_policy(d, actions, cfg))
            all_rows.append(row)
            d[f"action_{name}"] = actions

        decision_tables[repeat_id] = d

    return pd.DataFrame(all_rows), decision_tables


def repeated_significance_tests(decision_metrics):
    out = []
    pivot = decision_metrics.pivot(
        index="repeat",
        columns="strategy",
        values="net_benefit_proxy",
    )
    if "Proposed_KGDI" not in pivot.columns:
        return pd.DataFrame()

    for baseline in [
        "Probability_Threshold",
        "Cost_Aware",
        "Uncertainty_Review",
    ]:
        if baseline not in pivot.columns:
            continue
        diff = (pivot["Proposed_KGDI"] - pivot[baseline]).dropna()

        if len(diff) < 3 or np.allclose(diff, 0):
            stat, pvalue = np.nan, np.nan
        else:
            try:
                stat, pvalue = wilcoxon(diff, alternative="greater")
            except ValueError:
                stat, pvalue = np.nan, np.nan

        out.append({
            "comparison": f"Proposed_KGDI > {baseline}",
            "repeats": int(len(diff)),
            "mean_delta_net_benefit": float(diff.mean()),
            "median_delta_net_benefit": float(diff.median()),
            "wilcoxon_statistic": stat,
            "one_sided_p_value": pvalue,
        })
    return pd.DataFrame(out)


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
        row.update(evaluate_policy(d, actions, cfg))
        rows.append(row)
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
                    float(churners["intervened"].mean())
                    if len(churners) else np.nan
                ),
            })
    return pd.DataFrame(rows)


def sensitivity_analysis(d, base_cfg):
    offer_costs = [50.0, 100.0, 150.0]
    success_rates = [0.20, 0.35, 0.50]
    budget_fractions = [0.10, 0.20, 0.30]

    rows = []
    for cost in offer_costs:
        for success in success_rates:
            for budget_fraction in budget_fractions:
                cfg = BusinessConfig(**asdict(base_cfg))
                cfg.offer_cost = cost
                cfg.retention_success = success
                cfg.budget_fraction = budget_fraction

                budget = len(d) * cfg.offer_cost * cfg.budget_fraction

                policies = {
                    "Cost_Aware": policy_cost_aware(d, cfg, budget),
                    "Uncertainty_Review": policy_uncertainty_review(d, cfg, budget),
                    "Proposed_KGDI": policy_kgdi(d, cfg, budget),
                }

                metrics = {
                    k: evaluate_policy(d, a, cfg)
                    for k, a in policies.items()
                }

                rows.append({
                    "offer_cost": cost,
                    "retention_success": success,
                    "budget_fraction": budget_fraction,
                    "kgdi_net_benefit": metrics["Proposed_KGDI"]["net_benefit_proxy"],
                    "uncertainty_net_benefit": metrics["Uncertainty_Review"]["net_benefit_proxy"],
                    "costaware_net_benefit": metrics["Cost_Aware"]["net_benefit_proxy"],
                    "kgdi_delta_vs_uncertainty": (
                        metrics["Proposed_KGDI"]["net_benefit_proxy"]
                        - metrics["Uncertainty_Review"]["net_benefit_proxy"]
                    ),
                    "kgdi_high_value_churn_reach": (
                        metrics["Proposed_KGDI"]["high_value_churn_reach_rate"]
                    ),
                })
    return pd.DataFrame(rows)


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
        "Probability_Threshold", "Cost_Aware",
        "Uncertainty_Review", "Proposed_KGDI"
    ]
    base = decision_metrics[
        decision_metrics["strategy"] == "Uncertainty_Review"
    ]["net_benefit_proxy"].to_numpy()

    for strategy in order:
        g = decision_metrics[decision_metrics["strategy"] == strategy]
        if g.empty:
            continue

        delta = ""
        ci = ""
        if strategy == "Proposed_KGDI":
            kg = g["net_benefit_proxy"].to_numpy()
            if len(kg) == len(base):
                delta = f"{np.mean(kg - base):.2f}"
            ci = (
                f"[{bootstrap_result['ci_low']:.2f}, "
                f"{bootstrap_result['ci_high']:.2f}]"
            )

        rows.append({
            "Strategy": strategy,
            "Intervention %": mean_sd_string(100 * g["intervention_rate"], 1),
            "Churn reach %": mean_sd_string(100 * g["churn_reach_rate"], 1),
            "High-value churn reach %": mean_sd_string(
                100 * g["high_value_churn_reach_rate"], 1
            ),
            "Net benefit proxy ($)": mean_sd_string(g["net_benefit_proxy"], 2),
            "Δ vs uncertainty": delta,
            "95% paired bootstrap CI": ci,
        })
    return pd.DataFrame(rows)


def figure_decision_map(d, actions, output_path):
    action_order = ["No Action", "Retention Offer", "Human Review"]
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}

    plt.figure(figsize=(8.6, 6.2))

    # Point size reflects annual margin-value proxy only; no colors are forced.
    sizes = 12 + 45 * (
        d["annual_margin_value"].to_numpy()
        / max(d["annual_margin_value"].max(), 1e-9)
    )

    for action in action_order:
        mask = actions == action
        if not np.any(mask):
            continue
        plt.scatter(
            d.loc[mask, "churn_probability"],
            d.loc[mask, "uncertainty"],
            s=sizes[mask],
            alpha=0.45,
            marker=markers[action],
            label=action,
        )

    plt.xlabel("Calibrated ensemble churn probability")
    plt.ylabel("Composite predictive uncertainty")
    plt.title("KGDI Decision Map")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def figure_sensitivity(sensitivity, base_budget_fraction, output_path):
    s = sensitivity[
        np.isclose(sensitivity["budget_fraction"], base_budget_fraction)
    ].copy()

    costs = sorted(s["offer_cost"].unique())
    success = sorted(s["retention_success"].unique())
    matrix = np.zeros((len(success), len(costs)))

    for i, sr in enumerate(success):
        for j, cost in enumerate(costs):
            value = s[
                np.isclose(s["retention_success"], sr)
                & np.isclose(s["offer_cost"], cost)
            ]["kgdi_delta_vs_uncertainty"]
            matrix[i, j] = float(value.iloc[0]) if len(value) else np.nan

    plt.figure(figsize=(7.5, 5.5))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="KGDI net-benefit advantage ($)")
    plt.xticks(range(len(costs)), [f"${int(x)}" for x in costs])
    plt.yticks(range(len(success)), [f"{int(100*x)}%" for x in success])
    plt.xlabel("Retention-offer cost")
    plt.ylabel("Retention-success assumption")
    plt.title(
        f"KGDI Robustness at {int(100*base_budget_fraction)}% Budget Fraction"
    )

    for i in range(len(success)):
        for j in range(len(costs)):
            plt.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


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
        else:
            role = "ai_feature"

        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "missing": int(df[c].isna().sum()),
            "unique": int(df[c].nunique(dropna=True)),
            "role": role,
        })

    pd.DataFrame(rows).to_csv(output_dir / "SUPPLEMENT_column_roles.csv", index=False)


def main():
    warnings.filterwarnings("ignore")
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
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("KBS CHURN DECISION INTELLIGENCE")
    print("=" * 72)
    print(f"Mode: {args.mode}")
    print(f"Outer CV: {args.folds} folds × {repeats} repeats")
    print(f"Threads/model: {args.threads}")
    print(f"Paired bootstrap samples: {n_boot}")

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
    })

    with open(output_dir / "data_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    save_column_roles(df, output_dir)

    print(f"Rows: {len(df)} | Churn rate: {df[TARGET].mean():.3f}")
    print(f"AI predictors: {len(features)}")

    model_metrics, ensemble_by_repeat, _ = run_repeated_oof(
        df, features, numeric, categorical, args
    )
    model_metrics.to_csv(
        output_dir / "SUPPLEMENT_predictive_metrics_by_repeat.csv", index=False
    )

    decision_metrics, decision_tables = run_decision_experiment(
        df, ensemble_by_repeat, cfg
    )
    decision_metrics.to_csv(
        output_dir / "SUPPLEMENT_decision_metrics_by_repeat.csv", index=False
    )

    significance = repeated_significance_tests(decision_metrics)
    significance.to_csv(
        output_dir / "SUPPLEMENT_repeated_cv_significance.csv", index=False
    )

    # Aggregate repeated OOF ensemble predictions for the final descriptive policy.
    stacked_p = np.vstack([
        ensemble_by_repeat[r]["probability"] for r in sorted(ensemble_by_repeat)
    ])
    stacked_u = np.vstack([
        ensemble_by_repeat[r]["uncertainty"] for r in sorted(ensemble_by_repeat)
    ])
    stacked_entropy = np.vstack([
        ensemble_by_repeat[r]["entropy"] for r in sorted(ensemble_by_repeat)
    ])
    stacked_dis = np.vstack([
        ensemble_by_repeat[r]["disagreement"] for r in sorted(ensemble_by_repeat)
    ])

    final_info = {
        "probability": stacked_p.mean(axis=0),
        "uncertainty": stacked_u.mean(axis=0),
        "entropy": stacked_entropy.mean(axis=0),
        "disagreement": stacked_dis.mean(axis=0),
    }
    final_d = build_decision_table(df, final_info, cfg)
    budget = len(final_d) * cfg.offer_cost * cfg.budget_fraction

    final_uncertainty = policy_uncertainty_review(final_d, cfg, budget)
    final_kgdi = policy_kgdi(final_d, cfg, budget)

    bootstrap_result = bootstrap_delta(
        final_d, final_kgdi, final_uncertainty, cfg, n_boot, args.seed + 909
    )
    with open(output_dir / "SUPPLEMENT_bootstrap_KGDI_vs_uncertainty.json", "w") as f:
        json.dump(bootstrap_result, f, indent=2)

    ablations = run_ablations(final_d, cfg)
    ablations.to_csv(output_dir / "SUPPLEMENT_ablation_study.csv", index=False)

    audit_groups = subgroup_audit(df, final_d, final_kgdi)
    audit_groups.to_csv(output_dir / "SUPPLEMENT_subgroup_audit.csv", index=False)

    sensitivity = sensitivity_analysis(final_d, cfg)
    sensitivity.to_csv(output_dir / "SUPPLEMENT_sensitivity.csv", index=False)

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
    final_predictions["action_Uncertainty_Review"] = final_uncertainty
    final_predictions["action_Proposed_KGDI"] = final_kgdi
    final_predictions.to_csv(
        output_dir / "SUPPLEMENT_customer_level_oof_decisions.csv", index=False
    )

    run_config = {
        "study_title": (
            "Knowledge-Guided Decision Intelligence for Cost-Aware Customer "
            "Churn Intervention Under Predictive Uncertainty"
        ),
        "data_path": str(args.data),
        "mode": args.mode,
        "folds": args.folds,
        "repeats": repeats,
        "seed": args.seed,
        "bootstrap_samples": n_boot,
        "threads": args.threads,
        "business_parameters": asdict(cfg),
        "business_value_definition": (
            "12-month revenue-at-risk proxy = Monthly Charges × horizon months × "
            "gross-margin rate. CLTV is used only as a dimensionless priority index."
        ),
        "uncertainty_definition": (
            "0.70 × binary predictive entropy + 0.30 × normalized disagreement "
            "across calibrated AI models."
        ),
        "methodological_disclosure": (
            "IBM customer variables and churn outcomes are observed dataset fields. "
            "Retention cost, intervention success, human-review cost/accuracy, "
            "gross-margin rate, horizon, and budget are simulated assumptions."
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print("\nPRIMARY TABLE 1")
    print(table1.to_string(index=False))
    print("\nPRIMARY TABLE 2")
    print(table2.to_string(index=False))
    print("\nBOOTSTRAP KGDI vs UNCERTAINTY REVIEW")
    print(json.dumps(bootstrap_result, indent=2))
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
