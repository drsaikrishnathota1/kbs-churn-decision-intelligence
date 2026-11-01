from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .common import percentile_rank


@dataclass
class DatasetBundle:
    name: str
    frame: pd.DataFrame
    target: pd.Series
    features: pd.DataFrame
    strategic_value_raw: pd.Series
    economic_value_raw: pd.Series
    strategic_value: np.ndarray
    economic_value: np.ndarray
    metadata: dict


def _norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _lookup(df: pd.DataFrame, *candidates: str) -> str:
    mapping = {_norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = _norm_col(cand)
        if key in mapping:
            return mapping[key]
    raise KeyError(f"Could not find any of columns {candidates}; available={list(df.columns)}")


def _coerce_binary(s: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        x = pd.to_numeric(s, errors="coerce")
        vals = sorted(pd.Series(x.dropna().unique()).tolist())
        if set(vals).issubset({0, 1}):
            return x.astype("Int64")
    raw = s.astype(str).str.strip().str.lower()
    positive = {"1", "yes", "true", "churn", "churned"}
    negative = {"0", "no", "false", "stay", "stayed", "non-churn", "nonchurn"}
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[raw.isin(positive)] = 1
    out[raw.isin(negative)] = 0
    if out.isna().any():
        bad = sorted(raw[out.isna()].unique())[:10]
        raise ValueError(f"Cannot coerce {name} to binary; unresolved values={bad}")
    return out.astype(int)


def load_ibm_oof(path: str | Path) -> DatasetBundle:
    p = Path(path)
    df = pd.read_csv(p)
    required = [
        "CustomerID", "Churn Value", "churn_probability", "predictive_entropy",
        "CLTV", "Monthly Charges", "cltv_percentile", "annual_margin_value",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"IBM OOF file missing columns: {missing}")
    keep = df.dropna(subset=["Churn Value", "churn_probability", "CLTV", "Monthly Charges"]).copy()
    y = _coerce_binary(keep["Churn Value"], "Churn Value")
    q = percentile_rank(pd.to_numeric(keep["CLTV"], errors="raise"))
    e = percentile_rank(pd.to_numeric(keep["Monthly Charges"], errors="raise"))
    score_frame = pd.DataFrame({
        "customer_id": keep["CustomerID"].astype(str),
        "target": y.astype(int),
        "churn_probability": pd.to_numeric(keep["churn_probability"], errors="raise"),
        "predictive_entropy": pd.to_numeric(keep["predictive_entropy"], errors="raise"),
        "strategic_value": q,
        "economic_value": e,
        "strategic_value_raw": pd.to_numeric(keep["CLTV"], errors="raise"),
        "economic_value_raw": pd.to_numeric(keep["Monthly Charges"], errors="raise"),
        "annual_margin_value": pd.to_numeric(keep["annual_margin_value"], errors="raise"),
        "cltv_percentile": pd.to_numeric(keep["cltv_percentile"], errors="raise"),
    }).reset_index(drop=True)
    return DatasetBundle(
        name="ibm_telco",
        frame=score_frame,
        target=score_frame["target"],
        features=pd.DataFrame(index=score_frame.index),
        strategic_value_raw=score_frame["strategic_value_raw"],
        economic_value_raw=score_frame["economic_value_raw"],
        strategic_value=score_frame["strategic_value"].to_numpy(float),
        economic_value=score_frame["economic_value"].to_numpy(float),
        metadata={
            "source_type": "frozen_experiment_a_oof",
            "rows": int(len(score_frame)),
            "strategic_field": "CLTV",
            "economic_field": "Monthly Charges",
            "prediction_source": "frozen Experiment A calibrated OOF ensemble",
        },
    )


def load_iranian(path: str | Path) -> DatasetBundle:
    p = Path(path)
    df = pd.read_csv(p)
    target_col = _lookup(df, "Churn")
    strategic_col = _lookup(df, "Customer Value")
    economic_col = _lookup(df, "Charge Amount")
    y = _coerce_binary(df[target_col], target_col)
    strategic = pd.to_numeric(df[strategic_col], errors="coerce")
    economic = pd.to_numeric(df[economic_col], errors="coerce")
    valid = y.notna() & strategic.notna() & economic.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    strategic = strategic.loc[valid].reset_index(drop=True)
    economic = economic.loc[valid].reset_index(drop=True)
    # Customer Value is decision-layer-only. Charge Amount remains a legitimate observed predictor.
    exclude = {target_col, strategic_col}
    features = df[[c for c in df.columns if c not in exclude]].copy()
    return DatasetBundle(
        name="iranian_churn",
        frame=df,
        target=y,
        features=features,
        strategic_value_raw=strategic,
        economic_value_raw=economic,
        strategic_value=percentile_rank(strategic),
        economic_value=percentile_rank(economic),
        metadata={
            "source": "UCI Machine Learning Repository dataset 563",
            "doi": "10.24432/C5JW3Z",
            "license": "CC BY 4.0",
            "rows": int(len(df)),
            "strategic_field": strategic_col,
            "economic_field": economic_col,
            "target_field": target_col,
        },
    )


def load_cell2cell(path: str | Path) -> DatasetBundle:
    p = Path(path)
    df = pd.read_csv(p)
    target_col = _lookup(df, "Churn")
    strategic_col = _lookup(df, "TotalRecurringCharge", "Total Recurring Charge")
    economic_col = _lookup(df, "MonthlyRevenue", "Monthly Revenue")
    y = _coerce_binary(df[target_col], target_col)
    strategic = pd.to_numeric(df[strategic_col], errors="coerce")
    economic = pd.to_numeric(df[economic_col], errors="coerce")
    valid = y.notna() & strategic.notna() & economic.notna()
    df = df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].astype(int).reset_index(drop=True)
    strategic = strategic.loc[valid].reset_index(drop=True)
    economic = economic.loc[valid].reset_index(drop=True)

    # Exclude target, strategic/economic decision fields, obvious identifiers, and prior-retention process fields.
    exclude_norm = {
        _norm_col(target_col), _norm_col(strategic_col), _norm_col(economic_col),
        "customerid", "customer", "id",
    }
    retention_patterns = (
        "retentioncall", "retentioncalls", "retentionoffer", "retentionoffers",
        "retentionofferaccepted", "retentionoffersaccepted",
    )
    selected = []
    excluded_intervention = []
    for c in df.columns:
        n = _norm_col(c)
        if n in exclude_norm:
            continue
        if any(token in n for token in retention_patterns):
            excluded_intervention.append(c)
            continue
        selected.append(c)
    features = df[selected].copy()
    return DatasetBundle(
        name="cell2cell",
        frame=df,
        target=y,
        features=features,
        strategic_value_raw=strategic,
        economic_value_raw=economic,
        strategic_value=percentile_rank(strategic),
        economic_value=percentile_rank(economic),
        metadata={
            "source": "Cell2Cell labeled training partition (Duke/Teradata origin)",
            "rows": int(len(df)),
            "strategic_field": strategic_col,
            "economic_field": economic_col,
            "target_field": target_col,
            "excluded_prior_retention_process_fields": excluded_intervention,
            "strategic_value_note": "revenue/service-value proxy, not CLTV",
        },
    )
