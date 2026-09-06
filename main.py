#!/usr/bin/env python3

"""
Knowledge-Guided Decision Intelligence for Cost-Aware
Customer Churn Intervention Under Predictive Uncertainty

Single-file reproducible experimental pipeline.

Real IBM data:
- customer characteristics
- services
- charges
- tenure
- churn outcome
- CLTV

Simulated business parameters:
- retention offer cost
- retention success probability
- management budget
- human review cost
- human review accuracy
- uncertainty penalty
"""

import argparse
import io
import json
import math
import platform
import sys
import warnings
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "Churn Value"

LEAKAGE_COLUMNS = [
    "Churn Label",
    "Churn Score",
    "Churn Reason",
]

ID_GEO_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
]

# CLTV is intentionally NOT given to the AI churn predictor.
# It is reserved for the downstream business decision layer.
BUSINESS_ONLY_COLUMNS = [
    "CLTV",
]

# Excluded from primary prediction model but retained in source data.
AUDIT_ONLY_COLUMNS = [
    "Gender",
    "Senior Citizen",
]


@dataclass
class BusinessConfig:
    offer_cost: float = 100.0
    retention_success: float = 0.35

    human_review_cost: float = 25.0
    human_accuracy: float = 0.85

    budget_fraction: float = 0.20

    minimum_churn_risk: float = 0.20
    uncertainty_threshold: float = 0.65

    high_value_quantile: float = 0.75

    uncertainty_penalty: float = 0.20

    top_risk_fraction: float = 0.20


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="KBS AI + Decision Intelligence churn experiment"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="IBM Telco .xlsx file or ZIP containing it"
    )

    parser.add_argument(
        "--output",
        default="results/paper",
        help="Output directory"
    )

    parser.add_argument(
        "--mode",
        choices=["quick", "paper"],
        default="paper"
    )

    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional seeds such as 42,52,62,72,82"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20
    )

    parser.add_argument(
        "--offer-cost",
        type=float,
        default=100.0
    )

    parser.add_argument(
        "--retention-success",
        type=float,
        default=0.35
    )

    parser.add_argument(
        "--human-review-cost",
        type=float,
        default=25.0
    )

    parser.add_argument(
        "--human-accuracy",
        type=float,
        default=0.85
    )

    parser.add_argument(
        "--budget-fraction",
        type=float,
        default=0.20
    )

    return parser.parse_args()


# ============================================================
# DATA LOADING
# ============================================================

def load_data(path):

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)

    if path.suffix.lower() == ".zip":

        with zipfile.ZipFile(path, "r") as z:

            files = [
                f for f in z.namelist()
                if f.lower().endswith(".xlsx")
                and not f.startswith("__MACOSX")
            ]

            if not files:
                raise ValueError("No .xlsx file found inside ZIP")

            preferred = [
                f for f in files
                if Path(f).name.lower()
                == "telco_customer_churn.xlsx"
            ]

            selected = preferred[0] if preferred else files[0]

            print(f"Workbook selected: {selected}")

            with z.open(selected) as f:
                return pd.read_excel(io.BytesIO(f.read()))

    raise ValueError(
        "Dataset must be .xlsx or .zip containing .xlsx"
    )


# ============================================================
# DATA VALIDATION AND CLEANING
# ============================================================

def clean_data(df):

    required = [
        "CustomerID",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "CLTV",
        TARGET,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    df = df.copy()

    # IBM sample contains blank Total Charges values for
    # customers with zero tenure.
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    new_customer_mask = (
        df["Total Charges"].isna()
        & (df["Tenure Months"] == 0)
    )

    new_customer_count = int(
        new_customer_mask.sum()
    )

    df.loc[
        new_customer_mask,
        "Total Charges"
    ] = 0.0

    # Defensive handling for any remaining missing values.
    if df["Total Charges"].isna().any():

        median_value = df[
            "Total Charges"
        ].median()

        df["Total Charges"] = (
            df["Total Charges"]
            .fillna(median_value)
        )

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="raise"
    ).astype(int)

    if not set(
        df[TARGET].unique()
    ).issubset({0, 1}):

        raise ValueError(
            "Churn Value must contain only 0 and 1"
        )

    if df["CustomerID"].duplicated().any():
        raise ValueError(
            "Duplicate CustomerID values detected"
        )

    audit = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "churners": int(df[TARGET].sum()),
        "non_churners": int(
            (df[TARGET] == 0).sum()
        ),
        "churn_rate": float(
            df[TARGET].mean()
        ),
        "zero_tenure_total_charges_fixed":
            new_customer_count,
    }

    return df, audit


# ============================================================
# LEAKAGE-SAFE FEATURE SELECTION
# ============================================================

def select_features(df):

    excluded = set(
        LEAKAGE_COLUMNS
        + ID_GEO_COLUMNS
        + BUSINESS_ONLY_COLUMNS
        + AUDIT_ONLY_COLUMNS
        + [TARGET]
    )

    features = [
        c for c in df.columns
        if c not in excluded
    ]

    numeric = [
        c for c in features
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical = [
        c for c in features
        if c not in numeric
    ]

    return features, numeric, categorical


# ============================================================
# PREPROCESSING
# ============================================================

def create_preprocessor(
    numeric,
    categorical
):

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical
            ),
        ]
    )


# ============================================================
# AI MODELS
# ============================================================

def create_models(seed, mode):

    n_estimators = (
        200
        if mode == "quick"
        else 350
    )

    learning_rate = (
        0.06
        if mode == "quick"
        else 0.04
    )

    return {

        "LogisticRegression":
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=seed
            ),

        "RandomForest":
            RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=n_estimators,
                max_depth=4,
                learning_rate=learning_rate,
                subsample=0.90,
                colsample_bytree=0.90,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1
            ),

        "LightGBM":
            LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                num_leaves=31,
                subsample=0.90,
                colsample_bytree=0.90,
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
                n_jobs=-1
            ),
    }


# ============================================================
# AI METRICS
# ============================================================

def expected_calibration_error(
    y_true,
    probabilities,
    bins=10
):

    boundaries = np.linspace(
        0,
        1,
        bins + 1
    )

    result = 0.0

    for low, high in zip(
        boundaries[:-1],
        boundaries[1:]
    ):

        if high == 1:
            mask = (
                (probabilities >= low)
                & (probabilities <= high)
            )
        else:
            mask = (
                (probabilities >= low)
                & (probabilities < high)
            )

        if not np.any(mask):
            continue

        observed = np.mean(
            y_true[mask]
        )

        confidence = np.mean(
            probabilities[mask]
        )

        result += (
            np.mean(mask)
            * abs(
                observed
                - confidence
            )
        )

    return float(result)


def predictive_uncertainty(probabilities):

    probabilities = np.clip(
        probabilities,
        1e-12,
        1 - 1e-12
    )

    # Binary entropy.
    # Maximum value = 1 at p = 0.5.
    uncertainty = -(
        probabilities
        * np.log2(probabilities)
        +
        (1 - probabilities)
        * np.log2(
            1 - probabilities
        )
    )

    return uncertainty


def model_metrics(
    y_true,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "roc_auc":
            roc_auc_score(
                y_true,
                probabilities
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                probabilities
            ),

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "precision":
            precision_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                predictions,
                zero_division=0
            ),

        "brier":
            brier_score_loss(
                y_true,
                probabilities
            ),

        "log_loss":
            log_loss(
                y_true,
                probabilities
            ),

        "ece":
            expected_calibration_error(
                y_true,
                probabilities
            ),
    }


# ============================================================
# BUDGET ALLOCATION
# ============================================================

def allocate_budget(
    candidates,
    priority,
    costs,
    budget,
    action_name
):

    actions = np.array(
        ["No Action"] * len(priority),
        dtype=object
    )

    indices = np.where(
        candidates
    )[0]

    ordered = indices[
        np.argsort(
            -priority[indices]
        )
    ]

    spent = 0.0

    for i in ordered:

        cost = float(
            costs[i]
        )

        if (
            spent + cost
            <= budget
        ):

            actions[i] = (
                action_name
            )

            spent += cost

    return actions


# ============================================================
# BASELINE DECISION STRATEGIES
# ============================================================

def threshold_strategy(
    d,
    cfg,
    budget
):

    p = d[
        "churn_probability"
    ].to_numpy()

    costs = np.full(
        len(d),
        cfg.offer_cost
    )

    return allocate_budget(
        p >= 0.50,
        p,
        costs,
        budget,
        "Retention Offer"
    )


def top_risk_strategy(
    d,
    cfg,
    budget
):

    p = d[
        "churn_probability"
    ].to_numpy()

    number = max(
        1,
        int(
            math.ceil(
                len(d)
                * cfg.top_risk_fraction
            )
        )
    )

    selected = np.argsort(
        -p
    )[:number]

    mask = np.zeros(
        len(d),
        dtype=bool
    )

    mask[selected] = True

    costs = np.full(
        len(d),
        cfg.offer_cost
    )

    return allocate_budget(
        mask,
        p,
        costs,
        budget,
        "Retention Offer"
    )


def cost_aware_strategy(
    d,
    cfg,
    budget
):

    p = d[
        "churn_probability"
    ].to_numpy()

    cltv = d[
        "CLTV"
    ].to_numpy()

    utility = (
        p
        * cfg.retention_success
        * cltv
        - cfg.offer_cost
    )

    candidates = (
        utility > 0
    )

    costs = np.full(
        len(d),
        cfg.offer_cost
    )

    return allocate_budget(
        candidates,
        utility,
        costs,
        budget,
        "Retention Offer"
    )


# ============================================================
# PROPOSED KNOWLEDGE-GUIDED DECISION INTELLIGENCE
# ============================================================

def kgdi_strategy(
    d,
    cfg,
    budget,
    high_value_threshold
):

    p = d[
        "churn_probability"
    ].to_numpy(
        dtype=float
    )

    uncertainty = d[
        "uncertainty"
    ].to_numpy(
        dtype=float
    )

    cltv = d[
        "CLTV"
    ].to_numpy(
        dtype=float
    )

    # Expected utility from sending
    # a retention offer.
    offer_utility = (
        p
        * cfg.retention_success
        * cltv
        - cfg.offer_cost
        -
        cfg.uncertainty_penalty
        * uncertainty
        * cfg.retention_success
        * cltv
    )

    # Expected review cost assumes that
    # some reviewed cases will subsequently
    # receive a retention offer.
    review_cost = (
        cfg.human_review_cost
        +
        p
        * cfg.human_accuracy
        * cfg.offer_cost
    )

    review_utility = (
        p
        * cfg.human_accuracy
        * cfg.retention_success
        * cltv
        - review_cost
        -
        0.5
        * cfg.uncertainty_penalty
        * uncertainty
        * cfg.retention_success
        * cltv
    )

    # KNOWLEDGE RULE:
    # High-value + uncertain + meaningful
    # churn risk -> Human Review.
    review_mask = (
        (p >= cfg.minimum_churn_risk)
        &
        (
            uncertainty
            >= cfg.uncertainty_threshold
        )
        &
        (
            cltv
            >= high_value_threshold
        )
        &
        (
            review_utility > 0
        )
    )

    # Lower uncertainty or lower-value
    # profitable cases receive direct offers.
    offer_mask = (
        (p >= cfg.minimum_churn_risk)
        &
        (~review_mask)
        &
        (
            offer_utility > 0
        )
    )

    actions = np.array(
        ["No Action"] * len(d),
        dtype=object
    )

    candidates = []

    for i in np.where(
        review_mask
    )[0]:

        candidates.append(
            (
                i,
                "Human Review",
                float(
                    review_utility[i]
                ),
                float(
                    review_cost[i]
                ),
            )
        )

    for i in np.where(
        offer_mask
    )[0]:

        candidates.append(
            (
                i,
                "Retention Offer",
                float(
                    offer_utility[i]
                ),
                float(
                    cfg.offer_cost
                ),
            )
        )

    candidates.sort(
        key=lambda x: x[2],
        reverse=True
    )

    spent = 0.0

    for (
        index,
        action,
        utility,
        cost
    ) in candidates:

        if (
            spent + cost
            <= budget
        ):

            actions[index] = (
                action
            )

            spent += cost

    return actions


# ============================================================
# BUSINESS EVALUATION
# ============================================================

def evaluate_decisions(
    d,
    actions,
    cfg
):

    y = d[TARGET].to_numpy(
        dtype=int
    )

    p = d[
        "churn_probability"
    ].to_numpy(
        dtype=float
    )

    cltv = d[
        "CLTV"
    ].to_numpy(
        dtype=float
    )

    offer = (
        actions
        == "Retention Offer"
    )

    review = (
        actions
        == "Human Review"
    )

    intervened = (
        offer | review
    )

    expected_review_followups = (
        p[review]
        * cfg.human_accuracy
    ).sum()

    total_cost = (
        offer.sum()
        * cfg.offer_cost

        +

        review.sum()
        * cfg.human_review_cost

        +

        expected_review_followups
        * cfg.offer_cost
    )

    expected_saved_value = (

        np.sum(
            p[offer]
            * cfg.retention_success
            * cltv[offer]
        )

        +

        np.sum(
            p[review]
            * cfg.human_accuracy
            * cfg.retention_success
            * cltv[review]
        )
    )

    expected_utility = (
        expected_saved_value
        - total_cost
    )

    # This uses observed churn outcomes
    # only as an evaluation anchor.
    # It remains a simulated utility proxy.
    outcome_saved_value = (

        np.sum(
            y[offer]
            * cfg.retention_success
            * cltv[offer]
        )

        +

        np.sum(
            y[review]
            * cfg.human_accuracy
            * cfg.retention_success
            * cltv[review]
        )
    )

    outcome_utility = (
        outcome_saved_value
        - total_cost
    )

    total_churners = (
        y == 1
    ).sum()

    churners_reached = (
        (y == 1)
        & intervened
    ).sum()

    unnecessary_offers = (
        (y == 0)
        & offer
    ).sum()

    high_value_threshold = (
        np.quantile(
            cltv,
            0.75
        )
    )

    high_value_churners = (
        (y == 1)
        &
        (
            cltv
            >= high_value_threshold
        )
    )

    hv_total = (
        high_value_churners.sum()
    )

    hv_reached = (
        high_value_churners
        & intervened
    ).sum()

    return {

        "offers":
            int(offer.sum()),

        "human_reviews":
            int(review.sum()),

        "interventions":
            int(intervened.sum()),

        "intervention_rate":
            float(
                intervened.mean()
            ),

        "churn_reach_rate":
            float(
                churners_reached
                / total_churners
            )
            if total_churners
            else np.nan,

        "high_value_churn_reach_rate":
            float(
                hv_reached
                / hv_total
            )
            if hv_total
            else np.nan,

        "total_expected_cost":
            float(total_cost),

        "expected_utility":
            float(expected_utility),

        "outcome_utility_proxy":
            float(outcome_utility),

        "unnecessary_offer_cost":
            float(
                unnecessary_offers
                * cfg.offer_cost
            ),

        "mean_cltv_intervened":
            float(
                cltv[
                    intervened
                ].mean()
            )
            if intervened.any()
            else 0.0,
    }


# ============================================================
# RESULT SUMMARY
# ============================================================

def summarize(
    df,
    groups,
    metrics
):

    output = []

    grouped = df.groupby(
        groups,
        dropna=False
    )

    for keys, group in grouped:

        if not isinstance(
            keys,
            tuple
        ):
            keys = (keys,)

        row = dict(
            zip(
                groups,
                keys
            )
        )

        for metric in metrics:

            values = pd.to_numeric(
                group[metric],
                errors="coerce"
            )

            row[
                f"{metric}_mean"
            ] = float(
                values.mean()
            )

            row[
                f"{metric}_std"
            ] = (
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0
            )

        output.append(row)

    return pd.DataFrame(
        output
    )


# ============================================================
# STATISTICAL TESTS
# ============================================================

def statistical_tests(
    results
):

    pivot = (
        results
        .pivot_table(
            index=[
                "seed",
                "model"
            ],
            columns="strategy",
            values="outcome_utility_proxy",
            aggfunc="mean"
        )
    )

    if (
        "Proposed_KGDI"
        not in pivot.columns
    ):
        return pd.DataFrame()

    output = []

    proposed = pivot[
        "Proposed_KGDI"
    ]

    for baseline in pivot.columns:

        if (
            baseline
            == "Proposed_KGDI"
        ):
            continue

        paired = pd.concat(
            [
                proposed,
                pivot[baseline]
            ],
            axis=1
        ).dropna()

        if len(paired) < 2:
            continue

        difference = (
            paired.iloc[:, 0]
            - paired.iloc[:, 1]
        )

        try:

            statistic, pvalue = (
                wilcoxon(
                    difference,
                    alternative="greater"
                )
            )

        except ValueError:

            statistic = np.nan
            pvalue = np.nan

        output.append(
            {
                "comparison":
                    f"Proposed_KGDI > {baseline}",

                "pairs":
                    len(paired),

                "mean_difference":
                    difference.mean(),

                "median_difference":
                    difference.median(),

                "wilcoxon_statistic":
                    statistic,

                "one_sided_p_value":
                    pvalue,
            }
        )

    return pd.DataFrame(
        output
    )


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def sensitivity_analysis(
    decision_tables,
    base_config
):

    offer_costs = [
        50.0,
        100.0,
        150.0
    ]

    success_rates = [
        0.20,
        0.35,
        0.50
    ]

    budget_fractions = [
        0.10,
        0.20,
        0.30
    ]

    rows = []

    for (
        seed,
        model_name,
        d,
        high_value_threshold
    ) in decision_tables:

        for offer_cost in offer_costs:

            for success in success_rates:

                for budget_fraction in budget_fractions:

                    cfg = BusinessConfig(
                        **asdict(
                            base_config
                        )
                    )

                    cfg.offer_cost = (
                        offer_cost
                    )

                    cfg.retention_success = (
                        success
                    )

                    cfg.budget_fraction = (
                        budget_fraction
                    )

                    budget = (
                        len(d)
                        * cfg.offer_cost
                        * cfg.budget_fraction
                    )

                    strategies = {

                        "Probability_Threshold":
                            threshold_strategy(
                                d,
                                cfg,
                                budget
                            ),

                        "Top_Risk":
                            top_risk_strategy(
                                d,
                                cfg,
                                budget
                            ),

                        "Cost_Aware":
                            cost_aware_strategy(
                                d,
                                cfg,
                                budget
                            ),

                        "Proposed_KGDI":
                            kgdi_strategy(
                                d,
                                cfg,
                                budget,
                                high_value_threshold
                            ),
                    }

                    for (
                        strategy_name,
                        actions
                    ) in strategies.items():

                        metrics = (
                            evaluate_decisions(
                                d,
                                actions,
                                cfg
                            )
                        )

                        rows.append(
                            {
                                "seed":
                                    seed,

                                "model":
                                    model_name,

                                "strategy":
                                    strategy_name,

                                "offer_cost":
                                    offer_cost,

                                "retention_success":
                                    success,

                                "budget_fraction":
                                    budget_fraction,

                                **metrics,
                            }
                        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FIGURES
# ============================================================

def create_figures(
    model_summary,
    decision_summary,
    prediction_file,
    output_dir
):

    # MODEL ROC-AUC
    if not model_summary.empty:

        chart = model_summary.sort_values(
            "roc_auc_mean",
            ascending=False
        )

        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            chart["model"],
            chart["roc_auc_mean"]
        )

        plt.ylabel(
            "Mean ROC-AUC"
        )

        plt.title(
            "Calibrated Churn Prediction Performance"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / "figure_model_roc_auc.png",
            dpi=250
        )

        plt.close()

    # DECISION UTILITY
    if not decision_summary.empty:

        chart = (
            decision_summary
            .groupby(
                "strategy",
                as_index=False
            )[
                "outcome_utility_proxy_mean"
            ]
            .mean()
            .sort_values(
                "outcome_utility_proxy_mean",
                ascending=False
            )
        )

        plt.figure(
            figsize=(10, 5)
        )

        plt.bar(
            chart["strategy"],
            chart[
                "outcome_utility_proxy_mean"
            ]
        )

        plt.ylabel(
            "Outcome-Anchored Utility Proxy"
        )

        plt.title(
            "Business Decision Strategy Comparison"
        )

        plt.xticks(
            rotation=20
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / "figure_decision_utility.png",
            dpi=250
        )

        plt.close()

    # RISK-UNCERTAINTY SPACE
    if (
        prediction_file
        and prediction_file.exists()
    ):

        d = pd.read_csv(
            prediction_file
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.scatter(
            d["churn_probability"],
            d["uncertainty"],
            s=12,
            alpha=0.35
        )

        plt.xlabel(
            "Calibrated Churn Probability"
        )

        plt.ylabel(
            "Predictive Uncertainty"
        )

        plt.title(
            "Risk-Uncertainty Decision Space"
        )

        plt.tight_layout()

        plt.savefig(
            output_dir
            / "figure_risk_uncertainty.png",
            dpi=250
        )

        plt.close()


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore"
    )

    args = parse_args()

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_dir = (
        output_dir
        / "predictions"
    )

    prediction_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if args.seeds:

        seeds = [
            int(x.strip())
            for x
            in args.seeds.split(",")
            if x.strip()
        ]

    elif args.mode == "quick":

        seeds = [42]

    else:

        seeds = [
            42,
            52,
            62,
            72,
            82
        ]

    cfg = BusinessConfig(
        offer_cost=
            args.offer_cost,

        retention_success=
            args.retention_success,

        human_review_cost=
            args.human_review_cost,

        human_accuracy=
            args.human_accuracy,

        budget_fraction=
            args.budget_fraction,
    )

    print()
    print(
        "=" * 70
    )

    print(
        "KBS CHURN DECISION INTELLIGENCE"
    )

    print(
        "=" * 70
    )

    print(
        f"Dataset: {args.data}"
    )

    print(
        f"Mode: {args.mode}"
    )

    print(
        f"Seeds: {seeds}"
    )

    # --------------------------------------------------------
    # LOAD AND CLEAN
    # --------------------------------------------------------

    raw_data = load_data(
        args.data
    )

    df, audit = clean_data(
        raw_data
    )

    features, numeric, categorical = (
        select_features(
            df
        )
    )

    audit.update(
        {
            "ai_feature_count":
                len(features),

            "numeric_features":
                numeric,

            "categorical_features":
                categorical,

            "ai_features":
                features,

            "leakage_excluded":
                LEAKAGE_COLUMNS,

            "business_only":
                BUSINESS_ONLY_COLUMNS,

            "audit_only":
                AUDIT_ONLY_COLUMNS,
        }
    )

    with open(
        output_dir
        / "data_audit.json",
        "w"
    ) as file:

        json.dump(
            audit,
            file,
            indent=2
        )

    print()
    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Churn rate: "
        f"{df[TARGET].mean():.4f}"
    )

    print(
        f"AI features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # COLUMN ROLE TABLE
    # --------------------------------------------------------

    role_rows = []

    for column in df.columns:

        if column == TARGET:
            role = "target"

        elif column in LEAKAGE_COLUMNS:
            role = "exclude_leakage"

        elif column in BUSINESS_ONLY_COLUMNS:
            role = "business_decision_layer"

        elif column in AUDIT_ONLY_COLUMNS:
            role = "audit_only"

        elif column in ID_GEO_COLUMNS:
            role = "exclude_identifier_geography"

        else:
            role = "ai_feature"

        role_rows.append(
            {
                "column":
                    column,

                "dtype":
                    str(
                        df[column].dtype
                    ),

                "missing":
                    int(
                        df[column]
                        .isna()
                        .sum()
                    ),

                "unique":
                    int(
                        df[column]
                        .nunique(
                            dropna=True
                        )
                    ),

                "role":
                    role,
            }
        )

    pd.DataFrame(
        role_rows
    ).to_csv(
        output_dir
        / "column_roles.csv",
        index=False
    )

    # --------------------------------------------------------
    # EXPERIMENT VARIABLES
    # --------------------------------------------------------

    X = df[
        features
    ].copy()

    y = df[
        TARGET
    ].astype(int)

    customer_ids = df[
        "CustomerID"
    ].astype(str)

    business = df[
        [
            "CLTV",
            "Monthly Charges",
            "Total Charges",
            "Tenure Months",
        ]
    ].copy()

    model_rows = []
    decision_rows = []

    tables_for_sensitivity = []

    representative_prediction = None

    # --------------------------------------------------------
    # MULTI-SEED EXPERIMENT
    # --------------------------------------------------------

    for seed in seeds:

        print()
        print(
            f"SEED {seed}"
        )

        indices = np.arange(
            len(df)
        )

        (
            train_indices,
            test_indices
        ) = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=seed,
            stratify=y
        )

        X_train = X.iloc[
            train_indices
        ]

        X_test = X.iloc[
            test_indices
        ]

        y_train = y.iloc[
            train_indices
        ]

        y_test = y.iloc[
            test_indices
        ]

        ids_test = customer_ids.iloc[
            test_indices
        ]

        business_train = business.iloc[
            train_indices
        ]

        business_test = business.iloc[
            test_indices
        ]

        high_value_threshold = float(
            business_train[
                "CLTV"
            ].quantile(
                cfg.high_value_quantile
            )
        )

        preprocessor = (
            create_preprocessor(
                numeric,
                categorical
            )
        )

        models = create_models(
            seed,
            args.mode
        )

        for (
            model_name,
            estimator
        ) in models.items():

            print(
                f"Training {model_name}"
            )

            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        clone(
                            preprocessor
                        )
                    ),
                    (
                        "model",
                        estimator
                    ),
                ]
            )

            # Calibrated probabilities are important
            # because Decision Intelligence uses
            # probabilities as risk estimates.
            calibrated_model = (
                CalibratedClassifierCV(
                    estimator=pipeline,
                    method="sigmoid",
                    cv=3,
                    n_jobs=-1
                )
            )

            calibrated_model.fit(
                X_train,
                y_train
            )

            probabilities = (
                calibrated_model
                .predict_proba(
                    X_test
                )[:, 1]
            )

            uncertainty = (
                predictive_uncertainty(
                    probabilities
                )
            )

            metrics = model_metrics(
                y_test.to_numpy(),
                probabilities
            )

            model_rows.append(
                {
                    "seed":
                        seed,

                    "model":
                        model_name,

                    "train_rows":
                        len(train_indices),

                    "test_rows":
                        len(test_indices),

                    **metrics,
                }
            )

            # ------------------------------------------------
            # BUILD BUSINESS DECISION TABLE
            # ------------------------------------------------

            d = pd.DataFrame(
                {
                    "CustomerID":
                        ids_test.values,

                    TARGET:
                        y_test.values,

                    "churn_probability":
                        probabilities,

                    "uncertainty":
                        uncertainty,

                    "CLTV":
                        business_test[
                            "CLTV"
                        ].astype(
                            float
                        ).values,

                    "Monthly Charges":
                        business_test[
                            "Monthly Charges"
                        ].values,

                    "Total Charges":
                        business_test[
                            "Total Charges"
                        ].values,

                    "Tenure Months":
                        business_test[
                            "Tenure Months"
                        ].values,
                }
            )

            # Same cost-based budget is used
            # for all strategies.
            budget = (
                len(d)
                * cfg.offer_cost
                * cfg.budget_fraction
            )

            strategies = {

                "Probability_Threshold":
                    threshold_strategy(
                        d,
                        cfg,
                        budget
                    ),

                "Top_Risk":
                    top_risk_strategy(
                        d,
                        cfg,
                        budget
                    ),

                "Cost_Aware":
                    cost_aware_strategy(
                        d,
                        cfg,
                        budget
                    ),

                "Proposed_KGDI":
                    kgdi_strategy(
                        d,
                        cfg,
                        budget,
                        high_value_threshold
                    ),
            }

            for (
                strategy_name,
                actions
            ) in strategies.items():

                evaluation = (
                    evaluate_decisions(
                        d,
                        actions,
                        cfg
                    )
                )

                decision_rows.append(
                    {
                        "seed":
                            seed,

                        "model":
                            model_name,

                        "strategy":
                            strategy_name,

                        "budget":
                            budget,

                        "high_value_threshold":
                            high_value_threshold,

                        **evaluation,
                    }
                )

                d[
                    f"action_{strategy_name}"
                ] = actions

            prediction_file = (
                prediction_dir
                / (
                    f"predictions_"
                    f"seed{seed}_"
                    f"{model_name}.csv"
                )
            )

            d.to_csv(
                prediction_file,
                index=False
            )

            if (
                representative_prediction
                is None
                and model_name
                == "XGBoost"
            ):

                representative_prediction = (
                    prediction_file
                )

            tables_for_sensitivity.append(
                (
                    seed,
                    model_name,
                    d[
                        [
                            "CustomerID",
                            TARGET,
                            "churn_probability",
                            "uncertainty",
                            "CLTV",
                            "Monthly Charges",
                            "Total Charges",
                            "Tenure Months",
                        ]
                    ].copy(),
                    high_value_threshold,
                )
            )

    # ========================================================
    # SAVE AI RESULTS
    # ========================================================

    model_results = pd.DataFrame(
        model_rows
    )

    model_results.to_csv(
        output_dir
        / "model_metrics_all_runs.csv",
        index=False
    )

    model_summary = summarize(
        model_results,
        ["model"],
        [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "brier",
            "log_loss",
            "ece",
        ]
    )

    model_summary.to_csv(
        output_dir
        / "model_metrics_summary.csv",
        index=False
    )

    # ========================================================
    # SAVE DECISION INTELLIGENCE RESULTS
    # ========================================================

    decision_results = pd.DataFrame(
        decision_rows
    )

    decision_results.to_csv(
        output_dir
        / "decision_metrics_all_runs.csv",
        index=False
    )

    decision_summary = summarize(
        decision_results,
        [
            "model",
            "strategy"
        ],
        [
            "intervention_rate",
            "churn_reach_rate",
            "high_value_churn_reach_rate",
            "total_expected_cost",
            "expected_utility",
            "outcome_utility_proxy",
            "unnecessary_offer_cost",
            "mean_cltv_intervened",
        ]
    )

    decision_summary.to_csv(
        output_dir
        / "decision_metrics_summary.csv",
        index=False
    )

    # ========================================================
    # STATISTICAL TEST
    # ========================================================

    significance = statistical_tests(
        decision_results
    )

    significance.to_csv(
        output_dir
        / "decision_significance_tests.csv",
        index=False
    )

    # ========================================================
    # PAPER-MODE SENSITIVITY ANALYSIS
    # ========================================================

    if args.mode == "paper":

        print()
        print(
            "Running sensitivity analysis"
        )

        sensitivity = (
            sensitivity_analysis(
                tables_for_sensitivity,
                cfg
            )
        )

        sensitivity.to_csv(
            output_dir
            / "sensitivity_analysis.csv",
            index=False
        )

        sensitivity_summary = summarize(
            sensitivity,
            [
                "strategy",
                "offer_cost",
                "retention_success",
                "budget_fraction",
            ],
            [
                "outcome_utility_proxy",
                "churn_reach_rate",
                "unnecessary_offer_cost",
            ]
        )

        sensitivity_summary.to_csv(
            output_dir
            / "sensitivity_summary.csv",
            index=False
        )

    # ========================================================
    # CREATE FIGURES
    # ========================================================

    create_figures(
        model_summary,
        decision_summary,
        representative_prediction,
        output_dir
    )

    # ========================================================
    # SAVE REPRODUCIBILITY CONFIGURATION
    # ========================================================

    run_config = {

        "title":
            (
                "Knowledge-Guided Decision "
                "Intelligence for Cost-Aware "
                "Customer Churn Intervention "
                "Under Predictive Uncertainty"
            ),

        "dataset":
            args.data,

        "mode":
            args.mode,

        "seeds":
            seeds,

        "test_size":
            args.test_size,

        "business_parameters":
            asdict(cfg),

        "methodological_disclosure":
            (
                "IBM customer characteristics, "
                "churn outcomes, service variables "
                "and CLTV are dataset variables. "
                "Retention cost, retention success, "
                "human review cost/accuracy, budget "
                "and uncertainty penalty are "
                "simulation assumptions."
            ),

        "python":
            sys.version,

        "platform":
            platform.platform(),

        "pandas":
            pd.__version__,

        "numpy":
            np.__version__,
    }

    with open(
        output_dir
        / "run_config.json",
        "w"
    ) as file:

        json.dump(
            run_config,
            file,
            indent=2
        )

    # ========================================================
    # TERMINAL SUMMARY
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXPERIMENT COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Results saved to: "
        f"{output_dir.resolve()}"
    )

    print()
    print(
        "AI MODEL SUMMARY"
    )

    print(
        model_summary.to_string(
            index=False
        )
    )

    print()
    print(
        "DECISION INTELLIGENCE SUMMARY"
    )

    print(
        decision_summary.to_string(
            index=False
        )
    )

    if not significance.empty:

        print()
        print(
            "STATISTICAL COMPARISONS"
        )

        print(
            significance.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
