#!/usr/bin/env python3
"""Camera-ready letter outputs: TABLE_1, TABLE_2, FIGURE_1, FIGURE_2."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiment_c as xc
import main as kgdi

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "No Action": "#9AA0A6",
    "Retention Offer": "#E45756",
    "Human Review": "#4C78A8",
    "Probability threshold": "#9AA0A6",
    "Cost-aware": "#F58518",
    "Uncertainty review": "#54A24B",
    "Proposed KGDI": "#4C78A8",
    "Allocation path": "#8B8C89",
}


def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--oof", required=True, help="Frozen customer-level OOF CSV")
    p.add_argument("--run-config", required=True, help="Frozen run_config.json")
    p.add_argument("--data", required=True, help="IBM Telco workbook")
    p.add_argument("--table1", required=True, help="Frozen predictive TABLE_1 CSV")
    p.add_argument("--output", default="results/paper_letter")
    p.add_argument("--economic-share", type=float, default=0.85)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def bootstrap_vs_cost_aware(d, cfg, n_boot, seed, share):
    rng = np.random.default_rng(seed)
    n = len(d)
    nb = np.empty(n_boot)
    hv = np.empty(n_boot)
    budget0 = xc.planning_budget(d, cfg)
    proposed = xc.policy_kgdi_c(d, cfg, budget0, share)
    cost_aware = kgdi.policy_cost_aware(d, cfg, budget0)
    m_p = kgdi.evaluate_policy(d, proposed, cfg, planning_budget=budget0)
    m_c = kgdi.evaluate_policy(d, cost_aware, cfg, planning_budget=budget0)
    obs = {
        "nb": m_p["net_benefit_proxy"] - m_c["net_benefit_proxy"],
        "hv": m_p["high_value_churn_reach_rate"] - m_c["high_value_churn_reach_rate"],
    }
    for b in range(n_boot):
        sample = d.iloc[rng.integers(0, n, size=n)].reset_index(drop=True).copy()
        sample["cltv_percentile"] = kgdi.percentile_rank(sample["CLTV"].to_numpy(float))
        sample["annual_margin_value"] = (
            sample["Monthly Charges"].to_numpy(float)
            * cfg.value_horizon_months
            * cfg.gross_margin_rate
        )
        budget = xc.planning_budget(sample, cfg)
        p_act = xc.policy_kgdi_c(sample, cfg, budget, share)
        c_act = kgdi.policy_cost_aware(sample, cfg, budget)
        mp = kgdi.evaluate_policy(sample, p_act, cfg, planning_budget=budget)
        mc = kgdi.evaluate_policy(sample, c_act, cfg, planning_budget=budget)
        nb[b] = mp["net_benefit_proxy"] - mc["net_benefit_proxy"]
        hv[b] = mp["high_value_churn_reach_rate"] - mc["high_value_churn_reach_rate"]
    return {
        "nb_observed": float(obs["nb"]),
        "nb_ci": (float(np.percentile(nb, 2.5)), float(np.percentile(nb, 97.5))),
        "nb_p": float(np.mean(nb > 0)),
        "hv_observed": float(obs["hv"]),
        "hv_ci": (float(np.percentile(hv, 2.5)), float(np.percentile(hv, 97.5))),
        "hv_p": float(np.mean(hv > 0)),
        "bootstrap_samples": n_boot,
    }


def table_2(policy_rows, boot) -> pd.DataFrame:
    labels = {
        "Probability_Threshold": "Probability threshold",
        "Cost_Aware": "Cost-aware",
        "Uncertainty_Review": "Uncertainty review",
        "Proposed_KGDI": "Proposed KGDI",
    }
    keep = list(labels)
    rows = []
    for _, r in policy_rows.iterrows():
        if r["strategy"] not in keep:
            continue
        is_prop = r["strategy"] == "Proposed_KGDI"
        rows.append({
            "Strategy": labels[r["strategy"]],
            "Intervention %": f"{100 * r['intervention_rate']:.1f}",
            "Churn reach %": f"{100 * r['churn_reach_rate']:.1f}",
            "High-value churn reach %": f"{100 * r['high_value_churn_reach_rate']:.1f}",
            "Offers / reviews": f"{int(r['offers'])} / {int(r['human_reviews'])}",
            "Net-benefit proxy ($)": f"{r['net_benefit_proxy']:.0f}",
            "Δ vs cost-aware ($)": (
                f"{boot['nb_observed']:.0f}" if is_prop else ""
            ),
            "95% CI (net benefit)": (
                f"[{boot['nb_ci'][0]:.0f}, {boot['nb_ci'][1]:.0f}]" if is_prop else ""
            ),
            "P(Δ>0) net benefit": f"{boot['nb_p']:.3f}" if is_prop else "",
            "Δ high-value reach (pp)": (
                f"{100 * boot['hv_observed']:.1f}" if is_prop else ""
            ),
            "P(Δ>0) high-value reach": f"{boot['hv_p']:.3f}" if is_prop else "",
        })
    return pd.DataFrame(rows)


def figure_1(d, actions, path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    order = ["No Action", "Retention Offer", "Human Review"]
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}
    z = {"No Action": 1, "Retention Offer": 3, "Human Review": 4}
    risk = np.clip(d["knowledge_risk"].to_numpy(float), 0.0, 1.0)
    for action in order:
        mask = actions == action
        if not np.any(mask):
            continue
        ax.scatter(
            d.loc[mask, "churn_probability"],
            d.loc[mask, "cltv_percentile"],
            s=8 + 42 * risk[mask],
            c=COLORS[action],
            marker=markers[action],
            alpha=0.42,
            linewidths=0,
            zorder=z[action],
            label=action,
        )
    ax.axhline(0.75, color="#444444", ls=":", lw=0.9, zorder=0)
    ax.set_xlabel("Calibrated ensemble churn probability")
    ax.set_ylabel("Strategic value (CLTV percentile)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def figure_2(share_grid, letter_rows, path: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.plot(
        100 * share_grid["high_value_churn_reach_rate"],
        share_grid["net_benefit_proxy"] / 1000.0,
        color=COLORS["Allocation path"],
        marker="o",
        ms=4.5,
        lw=1.2,
        label="Knowledge-guided allocation path",
    )
    display = {
        "Probability_Threshold": ("Probability threshold", "D"),
        "Cost_Aware": ("Cost-aware", "X"),
        "Uncertainty_Review": ("Uncertainty review", "P"),
        "Proposed_KGDI": ("Proposed KGDI", "*"),
    }
    for key, (label, marker) in display.items():
        r = letter_rows[letter_rows["strategy"] == key].iloc[0]
        color = {
            "Probability threshold": COLORS["Probability threshold"],
            "Cost-aware": COLORS["Cost-aware"],
            "Uncertainty review": COLORS["Uncertainty review"],
            "Proposed KGDI": COLORS["Proposed KGDI"],
        }[label]
        ax.scatter(
            [100 * r["high_value_churn_reach_rate"]],
            [r["net_benefit_proxy"] / 1000.0],
            s=90 if key == "Proposed_KGDI" else 70,
            marker=marker,
            color=color,
            zorder=5,
            label=label,
        )
    ax.set_xlabel("High-value churn reach (%)")
    ax.set_ylabel("Net-benefit proxy ($ thousands)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    args = parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.oof)
    with open(args.run_config, encoding="utf-8") as f:
        run_config = json.load(f)
    cfg = kgdi.BusinessConfig(
        **{k: run_config["business_parameters"][k] for k in asdict(kgdi.BusinessConfig())}
    )
    raw = kgdi.load_data(args.data)
    d, inventory = xc.attach_knowledge(d, raw)
    budget = xc.planning_budget(d, cfg)

    frozen, frozen_actions = xc.reconstruct_experiment_a(d, cfg, budget)
    proposed = xc.policy_kgdi_c(d, cfg, budget, args.economic_share)
    proposed_row = xc.metrics_row(d, proposed, cfg, budget, "Proposed_KGDI")
    letter_rows = pd.concat(
        [
            frozen[frozen["strategy"].isin(
                ["Probability_Threshold", "Cost_Aware", "Uncertainty_Review"]
            )],
            pd.DataFrame([proposed_row]),
        ],
        ignore_index=True,
    )
    share_grid = xc.run_share_grid(d, cfg, budget, xc.DEFAULT_SHARE_GRID)
    boot = bootstrap_vs_cost_aware(d, cfg, args.bootstrap, args.seed, args.economic_share)

    table1 = pd.read_csv(args.table1)
    table1.to_csv(out / "TABLE_1_predictive_performance.csv", index=False)
    t2 = table_2(letter_rows, boot)
    t2.to_csv(out / "TABLE_2_decision_performance.csv", index=False)

    figure_1(d, proposed, out / "FIGURE_1_decision_map.png")
    figure_2(share_grid, letter_rows, out / "FIGURE_2_tradeoff.png")

    letter_rows.to_csv(out / "SUPPLEMENT_letter_policy_metrics.csv", index=False)
    with (out / "SUPPLEMENT_letter_bootstrap.json").open("w", encoding="utf-8") as f:
        json.dump({"bootstrap": boot, "knowledge_graph": inventory}, f, indent=2)

    print("TABLE 1")
    print(table1.to_string(index=False))
    print("\nTABLE 2")
    print(t2.to_string(index=False))
    print("\nBOOTSTRAP vs COST-AWARE")
    print(
        f"Net benefit Δ=${boot['nb_observed']:.0f} "
        f"CI=[{boot['nb_ci'][0]:.0f}, {boot['nb_ci'][1]:.0f}] P(Δ>0)={boot['nb_p']:.3f}"
    )
    print(
        f"High-value reach Δ={100 * boot['hv_observed']:.1f} pp "
        f"CI=[{100 * boot['hv_ci'][0]:.1f}, {100 * boot['hv_ci'][1]:.1f}] "
        f"P(Δ>0)={boot['hv_p']:.3f}"
    )
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
