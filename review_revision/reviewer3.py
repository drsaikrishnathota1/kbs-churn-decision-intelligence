from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

import main as kgdi


def load_cfg(run_config_path: str | Path):
    with open(run_config_path, encoding="utf-8") as f:
        rc = json.load(f)
    params = rc["business_parameters"]
    fields = asdict(kgdi.BusinessConfig()).keys()
    return kgdi.BusinessConfig(**{k: params[k] for k in fields})


def planning_budget(d, cfg):
    return float(len(d) * cfg.offer_cost * cfg.budget_fraction)


def fixed_cost_benefit_policy(d: pd.DataFrame, cfg, budget: float):
    p = d["churn_probability"].to_numpy(float)
    vref = float(np.median(d["annual_margin_value"].to_numpy(float)))
    expected = p * cfg.retention_success * vref - cfg.offer_cost
    eligible = (p >= cfg.minimum_churn_risk) & (expected > 0)
    order = np.argsort(-expected)
    actions = np.array(["No Action"] * len(d), dtype=object)
    spent = 0.0
    for i in order:
        if not eligible[i]:
            continue
        if spent + cfg.offer_cost <= budget + 1e-9:
            actions[i] = "Retention Offer"
            spent += cfg.offer_cost
    return actions, vref


def p_times_value_policy(d: pd.DataFrame, cfg, budget: float):
    p = d["churn_probability"].to_numpy(float)
    q = d["cltv_percentile"].to_numpy(float)
    v = kgdi.business_vectors(d, cfg)
    score = p * q
    eligible = (p >= cfg.minimum_churn_risk) & (v["offer_utility"] > 0)
    order = np.argsort(-score)
    actions = np.array(["No Action"] * len(d), dtype=object)
    spent = 0.0
    for i in order:
        if not eligible[i]:
            continue
        if spent + cfg.offer_cost <= budget + 1e-9:
            actions[i] = "Retention Offer"
            spent += cfg.offer_cost
    return actions


def metric_row(d, cfg, budget, label, actions):
    m = kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget)
    return {"strategy": label, **m}


def run_comparators(d: pd.DataFrame, cfg):
    budget = planning_budget(d, cfg)
    cost = kgdi.policy_cost_aware(d, cfg, budget)
    matrix, vref = fixed_cost_benefit_policy(d, cfg, budget)
    pxq = p_times_value_policy(d, cfg, budget)
    rows = [
        metric_row(d, cfg, budget, "Cost_Aware", cost),
        metric_row(d, cfg, budget, "Fixed_Cost_Benefit_Matrix", matrix),
        metric_row(d, cfg, budget, "p_x_CLTV_Percentile", pxq),
    ]
    return pd.DataFrame(rows), {"V_ref": vref, "budget": budget}, {
        "Cost_Aware": cost,
        "Fixed_Cost_Benefit_Matrix": matrix,
        "p_x_CLTV_Percentile": pxq,
    }


def paired_bootstrap(d: pd.DataFrame, cfg, n_boot=1000, seed=20261031):
    rng = np.random.default_rng(seed)
    labels = ["Fixed_Cost_Benefit_Matrix", "p_x_CLTV_Percentile"]
    store = {k: {"hv": [], "churn": [], "nb": []} for k in labels}

    base_metrics, meta, _ = run_comparators(d, cfg)
    base_cost = base_metrics[base_metrics.strategy == "Cost_Aware"].iloc[0]
    observed = {}
    for label in labels:
        r = base_metrics[base_metrics.strategy == label].iloc[0]
        observed[label] = {
            "delta_hv": float(r.high_value_churn_reach_rate - base_cost.high_value_churn_reach_rate),
            "delta_churn": float(r.churn_reach_rate - base_cost.churn_reach_rate),
            "delta_nb": float(r.net_benefit_proxy - base_cost.net_benefit_proxy),
        }

    for _ in range(n_boot):
        sample = d.iloc[rng.integers(0, len(d), size=len(d))].reset_index(drop=True).copy()
        # Recompute cohort-relative strategic rank in every bootstrap resample.
        sample["cltv_percentile"] = kgdi.percentile_rank(sample["CLTV"].to_numpy(float))
        sample["annual_margin_value"] = (
            sample["Monthly Charges"].to_numpy(float)
            * cfg.value_horizon_months
            * cfg.gross_margin_rate
        )
        metrics, _, _ = run_comparators(sample, cfg)
        c = metrics[metrics.strategy == "Cost_Aware"].iloc[0]
        for label in labels:
            r = metrics[metrics.strategy == label].iloc[0]
            store[label]["hv"].append(float(r.high_value_churn_reach_rate - c.high_value_churn_reach_rate))
            store[label]["churn"].append(float(r.churn_reach_rate - c.churn_reach_rate))
            store[label]["nb"].append(float(r.net_benefit_proxy - c.net_benefit_proxy))

    out = {"seed": seed, "bootstrap_samples": n_boot, "reference": "Cost_Aware", "comparisons": {}}
    for label in labels:
        out["comparisons"][label] = {"observed": observed[label]}
        for key in ["hv", "churn", "nb"]:
            arr = np.asarray(store[label][key], dtype=float)
            out["comparisons"][label][key] = {
                "ci_2_5": float(np.nanpercentile(arr, 2.5)),
                "ci_97_5": float(np.nanpercentile(arr, 97.5)),
                "fraction_positive": float(np.nanmean(arr > 0)),
            }
    return out
