#!/usr/bin/env python3
"""Reviewer-requested analyses on frozen OOF (no model refit)."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import experiment_c as xc
import main as kgdi

EXP_A = Path("results/experiment_a_frozen")
DATA = Path("data/Telco_customer_churn.xlsx")
OUT = Path("results/paper_letter")
SHARE = 0.85


def summarize(d, actions, cfg, budget, name: str) -> dict:
    row = xc.metrics_row(d, actions, cfg, budget, name)
    row["offers"] = int((actions == "Retention Offer").sum())
    row["human_reviews"] = int((actions == "Human Review").sum())
    return row


def residual_table(d) -> pd.DataFrame:
    y = d[kgdi.TARGET].to_numpy(int)
    p = d["churn_probability"].to_numpy(float)
    r = d["knowledge_residual"].to_numpy(float)
    pos = r[r > 0]
    cuts = np.quantile(pos, [1 / 3, 2 / 3]) if np.any(r > 0) else [0, 0]
    bins = np.full(len(d), "k-p ≤ 0", dtype=object)
    bins[r > 0] = "small +"
    bins[r > cuts[0]] = "medium +"
    bins[r > cuts[1]] = "large +"
    order = ["k-p ≤ 0", "small +", "medium +", "large +"]
    rows = []
    for lab in order:
        m = bins == lab
        rows.append(
            {
                "bin": lab,
                "n": int(m.sum()),
                "mean_p": float(p[m].mean()),
                "observed_churn": float(y[m].mean()),
                "observed_minus_p_pp": float(100 * (y[m].mean() - p[m].mean())),
            }
        )
    return pd.DataFrame(rows)


def disagreement_table(d) -> dict:
    y = d[kgdi.TARGET].to_numpy(int)
    r = d["knowledge_residual"].to_numpy(float)
    disc = d["model_disagreement"].to_numpy(float)
    graph = r > 0
    top_g = r >= np.quantile(r, 0.90)
    top_d = disc >= np.quantile(disc, 0.90)
    groups = {
        "graph_only": graph & (disc < np.median(disc)),
        "disagreement_only": (~graph) & (disc >= np.quantile(disc, 0.75)),
        "both_high": graph & (disc >= np.quantile(disc, 0.75)),
        "neither": (~graph) & (disc < np.median(disc)),
    }
    # reviewer four-way on top 10%
    four = {
        "graph_conflict_only": top_g & ~top_d,
        "model_disagreement_only": ~top_g & top_d,
        "both": top_g & top_d,
        "neither_top10": ~top_g & ~top_d,
    }
    out = {
        "corr_residual_disagreement": float(np.corrcoef(r, disc)[0, 1]),
        "top10_overlap": float(np.mean(top_g & top_d)),
        "top10_jaccard": float(np.sum(top_g & top_d) / np.maximum(np.sum(top_g | top_d), 1)),
    }
    for name, m in four.items():
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_churn"] = float(y[m].mean()) if m.any() else float("nan")
    return out, groups


def sensitivity(d, cfg) -> pd.DataFrame:
    scenarios = [
        ("base", cfg),
        ("offer_80", replace(cfg, offer_cost=80.0)),
        ("offer_120", replace(cfg, offer_cost=120.0)),
        ("success_0.25", replace(cfg, retention_success=0.25)),
        ("success_0.45", replace(cfg, retention_success=0.45)),
        ("review_15", replace(cfg, human_review_cost=15.0)),
        ("review_40", replace(cfg, human_review_cost=40.0)),
        ("reviewer_0.75", replace(cfg, human_sensitivity=0.75, human_specificity=0.75)),
        ("reviewer_0.95", replace(cfg, human_sensitivity=0.95, human_specificity=0.95)),
        ("budget_0.15", replace(cfg, budget_fraction=0.15)),
        ("budget_0.25", replace(cfg, budget_fraction=0.25)),
    ]
    rows = []
    for name, c in scenarios:
        budget = xc.planning_budget(d, c)
        kg = xc.policy_kgdi_c(d, c, budget, SHARE)
        ca = kgdi.policy_cost_aware(d, c, budget)
        mk = kgdi.evaluate_policy(d, kg, c, planning_budget=budget)
        mc = kgdi.evaluate_policy(d, ca, c, planning_budget=budget)
        rows.append(
            {
                "scenario": name,
                "hv_kgdi": mk["high_value_churn_reach_rate"],
                "hv_cost_aware": mc["high_value_churn_reach_rate"],
                "delta_hv_pp": 100
                * (
                    mk["high_value_churn_reach_rate"]
                    - mc["high_value_churn_reach_rate"]
                ),
                "nb_kgdi": mk["net_benefit_proxy"],
                "nb_cost_aware": mc["net_benefit_proxy"],
                "delta_nb": mk["net_benefit_proxy"] - mc["net_benefit_proxy"],
                "churn_reach_kgdi": mk["churn_reach_rate"],
                "churn_reach_cost_aware": mc["churn_reach_rate"],
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, cfg, raw, _ = xc.load_inputs(EXP_A, DATA)
    d, inventory = xc.attach_knowledge(d, raw)
    budget = xc.planning_budget(d, cfg)

    ab = xc.ablations(d, cfg, budget, SHARE)
    ab.to_csv(OUT / "SUPPLEMENT_component_ablation.csv", index=False)

    actions = {
        r["variant"] if "variant" in r else r.get("strategy"): None for _, r in ab.iterrows()
    }
    # distinctness vs full
    full = xc.policy_kgdi_c(d, cfg, budget, SHARE)
    variants_act = {
        "Cost_aware": kgdi.policy_cost_aware(d, cfg, budget),
        "CLTV_only": xc.policy_kgdi_c(
            d, cfg, budget, SHARE, use_cltv=True, use_entropy=False, use_graph=False
        ),
        "CLTV_entropy": xc.policy_kgdi_c(
            d, cfg, budget, SHARE, use_cltv=True, use_entropy=True, use_graph=False
        ),
        "CLTV_graph": xc.policy_kgdi_c(
            d, cfg, budget, SHARE, use_cltv=True, use_entropy=False, use_graph=True
        ),
        "Full_KGDI": full,
    }
    dist = []
    for name, act in variants_act.items():
        dist.append(xc.distinctness_row(full, act, f"Full vs {name}"))
    dist_df = pd.DataFrame(dist)

    res = residual_table(d)
    disc_summary, _ = disagreement_table(d)
    sens = sensitivity(d, cfg)

    ca = kgdi.evaluate_policy(
        d, kgdi.policy_cost_aware(d, cfg, budget), cfg, planning_budget=budget
    )
    full_m = kgdi.evaluate_policy(d, full, cfg, planning_budget=budget)
    tradeoff_pp = 100 * (
        full_m["high_value_churn_reach_rate"] - ca["high_value_churn_reach_rate"]
    )
    tradeoff_nb = full_m["net_benefit_proxy"] - ca["net_benefit_proxy"]

    payload = {
        "inventory": inventory,
        "disagreement": disc_summary,
        "hv_delta_pp": tradeoff_pp,
        "nb_delta_point": tradeoff_nb,
        "nb_per_hv_pp": float(tradeoff_nb / tradeoff_pp) if tradeoff_pp else None,
    }
    res.to_csv(OUT / "SUPPLEMENT_graph_residual_bins.csv", index=False)
    dist_df.to_csv(OUT / "SUPPLEMENT_ablation_distinctness.csv", index=False)
    sens.to_csv(OUT / "SUPPLEMENT_scenario_sensitivity.csv", index=False)
    pd.Series(disc_summary).to_json(OUT / "SUPPLEMENT_graph_vs_disagreement.json", indent=2)

    print("ABLATION")
    cols = [
        "variant",
        "offers",
        "human_reviews",
        "churn_reach_rate",
        "high_value_churn_reach_rate",
        "expected_selected_cost",
        "net_benefit_proxy",
    ]
    print(ab[cols].to_string(index=False))
    print("\nDISTINCTNESS vs FULL")
    print(dist_df.to_string(index=False))
    print("\nRESIDUAL BINS")
    print(res.to_string(index=False))
    print("\nGRAPH vs DISAGREEMENT")
    print(pd.Series(disc_summary).to_string())
    print("\nSENSITIVITY HV Δ pp")
    print(sens[["scenario", "delta_hv_pp", "delta_nb"]].to_string(index=False))
    print("\nTRADEOFF $ per HV pp (point):", payload["nb_per_hv_pp"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
