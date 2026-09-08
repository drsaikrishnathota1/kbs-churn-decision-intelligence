#!/usr/bin/env python3
"""Experiment C: knowledge-graph decision intelligence on frozen Experiment A OOF."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def _import_frozen_main():
    """Import frozen Experiment A helpers without requiring native boosting libs."""
    if "xgboost" not in sys.modules:
        xg = types.ModuleType("xgboost")
        xg.XGBClassifier = object
        sys.modules["xgboost"] = xg
    if "lightgbm" not in sys.modules:
        lg = types.ModuleType("lightgbm")
        lg.LGBMClassifier = object
        sys.modules["lightgbm"] = lg
    import main as kgdi  # noqa: WPS433

    return kgdi


kgdi = _import_frozen_main()

DEFAULT_SHARE_GRID = [0.70, 0.75, 0.80, 0.85, 0.90, 1.00]
PRIMARY_ECONOMIC_SHARE = 0.85
KNOWLEDGE_FOLDS = 5
KNOWLEDGE_SEED = 42

SERVICE_YES_FIELDS = [
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Experiment C knowledge-graph decision layer on frozen OOF",
        allow_abbrev=False,
    )
    p.add_argument("--experiment-a", required=True, help="Experiment A results directory")
    p.add_argument("--data", required=True, help="IBM Telco workbook (.xlsx)")
    p.add_argument("--output", default="results/experiment_c")
    p.add_argument("--economic-share", type=float, default=PRIMARY_ECONOMIC_SHARE)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20260908)
    return p.parse_args()


def is_yes(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.strip().str.lower().isin(["yes", "true", "1"]).to_numpy()


def motif_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    contract = df["Contract"].astype(str)
    internet = df["Internet Service"].astype(str)
    pay = df["Payment Method"].astype(str)
    tenure = df["Tenure Months"].to_numpy(float)
    has_internet = internet.ne("No").to_numpy()
    return {
        "contract:Month-to-month": (contract == "Month-to-month").to_numpy(float),
        "contract:One year": (contract == "One year").to_numpy(float),
        "contract:Two year": (contract == "Two year").to_numpy(float),
        "internet:Fiber optic": (internet == "Fiber optic").to_numpy(float),
        "internet:DSL": (internet == "DSL").to_numpy(float),
        "pay:Electronic check": (pay == "Electronic check").to_numpy(float),
        "motif:fiber_month_to_month": (
            (internet == "Fiber optic") & (contract == "Month-to-month")
        ).to_numpy(float),
        "motif:fiber_m2m_electronic_check": (
            (internet == "Fiber optic")
            & (contract == "Month-to-month")
            & (pay == "Electronic check")
        ).to_numpy(float),
        "motif:unprotected_internet": (
            has_internet & ~is_yes(df["Online Security"]) & ~is_yes(df["Tech Support"])
        ).astype(float),
        "motif:tenure_le_6": (tenure <= 6).astype(float),
        "motif:tenure_7_12": ((tenure > 6) & (tenure <= 12)).astype(float),
    }


def graph_inventory(df: pd.DataFrame, motifs: dict[str, np.ndarray]) -> dict:
    n_customers = int(len(df))
    type_nodes = (
        int(df["Contract"].nunique())
        + int(df["Internet Service"].nunique())
        + int(df["Payment Method"].nunique())
        + len(SERVICE_YES_FIELDS)
        + int(len(motifs))
    )
    has_contract = n_customers
    uses_internet = int((df["Internet Service"].astype(str) != "No").sum())
    pays = n_customers
    service_edges = 0
    for col in SERVICE_YES_FIELDS:
        service_edges += int(is_yes(df[col]).sum())
    motif_edges = int(sum(v.sum() for v in motifs.values()))
    return {
        "customer_nodes": n_customers,
        "knowledge_type_nodes": type_nodes,
        "has_contract_edges": has_contract,
        "uses_internet_edges": uses_internet,
        "pays_via_edges": pays,
        "uses_protection_or_addon_edges": service_edges,
        "motif_membership_edges": motif_edges,
        "total_edges": has_contract + uses_internet + pays + service_edges + motif_edges,
    }


def oof_knowledge_risk(y: np.ndarray, motifs: dict[str, np.ndarray], seed: int) -> np.ndarray:
    """Training-fold motif churn rates aggregated onto held-out customers."""
    n = len(y)
    risk = np.full(n, np.nan)
    names = list(motifs)
    matrix = np.column_stack([motifs[name] for name in names])
    splitter = StratifiedKFold(n_splits=KNOWLEDGE_FOLDS, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(np.zeros(n), y):
        base = float(y[train_idx].mean())
        rates = np.array(
            [
                float(y[train_idx][matrix[train_idx, j] > 0].mean())
                if np.any(matrix[train_idx, j] > 0)
                else base
                for j in range(matrix.shape[1])
            ]
        )
        active = matrix[test_idx]
        weight = active.sum(axis=1)
        agg = (active * rates).sum(axis=1)
        risk[test_idx] = np.where(weight > 0, agg / np.maximum(weight, 1e-12), base)
    if np.isnan(risk).any():
        raise RuntimeError("Incomplete OOF knowledge-risk scores")
    return np.clip(risk, 1e-6, 1 - 1e-6)


def attach_knowledge(d: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    raw = raw.copy()
    raw["CustomerID"] = raw["CustomerID"].astype(str)
    d = d.copy()
    d["CustomerID"] = d["CustomerID"].astype(str)
    keep = ["CustomerID", "Contract", "Internet Service", "Payment Method", "Tenure Months"]
    keep += [c for c in SERVICE_YES_FIELDS if c in raw.columns]
    merged = d.merge(raw[keep], on="CustomerID", how="left", suffixes=("", "_raw"))
    missing = [c for c in keep if c != "CustomerID" and merged[c].isna().any()]
    if missing:
        raise RuntimeError(f"Knowledge-graph fields missing after join: {missing}")
    motifs = motif_matrix(merged)
    y = merged[kgdi.TARGET].to_numpy(int)
    knowledge = oof_knowledge_risk(y, motifs, KNOWLEDGE_SEED)
    merged["knowledge_risk"] = knowledge
    merged["knowledge_residual"] = knowledge - merged["churn_probability"].to_numpy(float)
    inventory = graph_inventory(merged, motifs)
    inventory.update(
        {
            "knowledge_risk_auroc": float(roc_auc_score(y, knowledge)),
            "ml_probability_auroc": float(roc_auc_score(y, merged["churn_probability"])),
            "knowledge_pr_auc": float(average_precision_score(y, knowledge)),
            "ml_pr_auc": float(average_precision_score(y, merged["churn_probability"])),
            "corr_knowledge_vs_ml": float(
                np.corrcoef(knowledge, merged["churn_probability"].to_numpy(float))[0, 1]
            ),
        }
    )
    return merged, inventory


def planning_budget(d, cfg) -> float:
    return float(len(d) * cfg.offer_cost * cfg.budget_fraction)


def policy_kgdi_c(d, cfg, budget, economic_share: float):
    """Majority cost-aware offers, minority knowledge-guided high-value review."""
    if not (0.0 <= economic_share <= 1.0):
        raise ValueError("economic_share must be in [0, 1]")
    v = kgdi.business_vectors(d, cfg)
    n = len(d)
    actions = np.array(["No Action"] * n, dtype=object)
    spent = 0.0
    offer_cap = economic_share * budget
    knowledge_residual = (
        d["knowledge_residual"].to_numpy(float)
        if "knowledge_residual" in d.columns
        else np.zeros(n)
    )

    offer_order = np.argsort(-v["offer_utility"])
    for i in offer_order:
        if v["p"][i] < cfg.minimum_churn_risk or v["offer_utility"][i] <= 0:
            continue
        if spent + cfg.offer_cost <= offer_cap + 1e-9:
            actions[i] = "Retention Offer"
            spent += cfg.offer_cost

    review_ok = (
        (actions == "No Action")
        & (v["p"] >= cfg.minimum_churn_risk)
        & (v["cltv_pct"] >= cfg.high_value_quantile)
        & (v["review_utility"] > 0)
        & (
            (v["uncertainty"] >= cfg.moderate_uncertainty_threshold)
            | (knowledge_residual > 0)
        )
    )
    review_priority = v["review_utility"] / np.maximum(v["review_expected_cost"], 1e-12)
    for i in np.argsort(-review_priority):
        if not review_ok[i]:
            continue
        cost_i = float(v["review_expected_cost"][i])
        if spent + cost_i <= budget + 1e-9:
            actions[i] = "Human Review"
            spent += cost_i

    for i in offer_order:
        if actions[i] != "No Action":
            continue
        if v["p"][i] < cfg.minimum_churn_risk or v["offer_utility"][i] <= 0:
            continue
        if spent + cfg.offer_cost <= budget + 1e-9:
            actions[i] = "Retention Offer"
            spent += cfg.offer_cost
    return actions


def distinctness_row(a, b, comparison: str) -> dict:
    changed = a != b
    return {
        "comparison": comparison,
        "different_actions": int(np.sum(changed)),
        "disagreement_rate": float(np.mean(changed)),
        "left_no_action": int(np.sum(a == "No Action")),
        "left_retention_offer": int(np.sum(a == "Retention Offer")),
        "left_human_review": int(np.sum(a == "Human Review")),
        "right_no_action": int(np.sum(b == "No Action")),
        "right_retention_offer": int(np.sum(b == "Retention Offer")),
        "right_human_review": int(np.sum(b == "Human Review")),
    }


def metrics_row(d, actions, cfg, budget, strategy: str) -> dict:
    row = {"strategy": strategy}
    row.update(kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget))
    v = kgdi.business_vectors(d, cfg)
    offer = actions == "Retention Offer"
    review = actions == "Human Review"
    row["expected_utility"] = float(
        np.sum(v["offer_utility"][offer]) + np.sum(v["review_utility"][review])
    )
    return row


def reconstruct_experiment_a(d, cfg, budget) -> tuple[pd.DataFrame, dict]:
    mapping = {
        "Probability_Threshold": "action_Probability_Threshold",
        "Cost_Aware": "action_Cost_Aware",
        "Uncertainty_Review": "action_Uncertainty_Review",
        "Proposed_KGDI_A": "action_Proposed_KGDI",
    }
    rows = []
    actions = {}
    for name, col in mapping.items():
        if col not in d.columns:
            raise RuntimeError(f"Frozen Experiment A missing {col}")
        a = d[col].to_numpy(dtype=object)
        actions[name] = a
        rows.append(metrics_row(d, a, cfg, budget, name))
    return pd.DataFrame(rows), actions


def run_share_grid(d, cfg, budget, shares) -> pd.DataFrame:
    rows = []
    for share in shares:
        actions = policy_kgdi_c(d, cfg, budget, share)
        row = metrics_row(d, actions, cfg, budget, f"KGDI_C_share_{share:.2f}")
        row["economic_allocation_fraction"] = share
        rows.append(row)
    return pd.DataFrame(rows)


def ablations(d, cfg, budget, economic_share: float) -> pd.DataFrame:
    variants = {
        "KGDI_C_full": policy_kgdi_c(d, cfg, budget, economic_share),
        "No_review_reserve": policy_kgdi_c(d, cfg, budget, 1.0),
        "No_knowledge_residual": policy_kgdi_c(
            d.assign(knowledge_residual=0.0), cfg, budget, economic_share
        ),
        "No_CLTV_gate": policy_kgdi_c(
            d.assign(cltv_percentile=1.0), cfg, budget, economic_share
        ),
    }
    rows = []
    for name, actions in variants.items():
        row = {"variant": name}
        row.update(kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_deltas(d, cfg, n_boot: int, seed: int, economic_share: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(d)
    budget0 = planning_budget(d, cfg)
    observed = {
        "Cost_Aware": kgdi.policy_cost_aware(d, cfg, budget0),
        "Uncertainty_Review": kgdi.policy_uncertainty_review(d, cfg, budget0),
        "Probability_Threshold": kgdi.policy_probability_threshold(d, cfg, budget0),
        "KGDI_C": policy_kgdi_c(d, cfg, budget0, economic_share),
    }
    obs_nb = {
        k: kgdi.evaluate_policy(d, a, cfg, planning_budget=budget0)["net_benefit_proxy"]
        for k, a in observed.items()
    }
    baselines = ["Cost_Aware", "Uncertainty_Review", "Probability_Threshold"]
    store = {b: np.empty(n_boot) for b in baselines}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = d.iloc[idx].reset_index(drop=True).copy()
        sample["cltv_percentile"] = kgdi.percentile_rank(sample["CLTV"].to_numpy(float))
        sample["annual_margin_value"] = (
            sample["Monthly Charges"].to_numpy(float)
            * cfg.value_horizon_months
            * cfg.gross_margin_rate
        )
        budget = planning_budget(sample, cfg)
        kgc = policy_kgdi_c(sample, cfg, budget, economic_share)
        kgc_nb = kgdi.evaluate_policy(sample, kgc, cfg, planning_budget=budget)[
            "net_benefit_proxy"
        ]
        comps = {
            "Cost_Aware": kgdi.policy_cost_aware(sample, cfg, budget),
            "Uncertainty_Review": kgdi.policy_uncertainty_review(sample, cfg, budget),
            "Probability_Threshold": kgdi.policy_probability_threshold(sample, cfg, budget),
        }
        for name, act in comps.items():
            nb = kgdi.evaluate_policy(sample, act, cfg, planning_budget=budget)[
                "net_benefit_proxy"
            ]
            store[name][b] = kgc_nb - nb

    rows = []
    for name in baselines:
        deltas = store[name]
        rows.append(
            {
                "comparison": f"KGDI_C - {name}",
                "observed_delta": float(obs_nb["KGDI_C"] - obs_nb[name]),
                "bootstrap_mean_delta": float(deltas.mean()),
                "ci_low": float(np.percentile(deltas, 2.5)),
                "ci_high": float(np.percentile(deltas, 97.5)),
                "bootstrap_probability_gt_0": float(np.mean(deltas > 0)),
                "bootstrap_samples": int(n_boot),
            }
        )
    return pd.DataFrame(rows)


def make_figures(d, policy_table, share_grid, kgc_actions, output_dir: Path, cfg):
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}
    plt.figure(figsize=(8.6, 6.2))
    for action, marker in markers.items():
        mask = kgc_actions == action
        if not np.any(mask):
            continue
        plt.scatter(
            d.loc[mask, "churn_probability"],
            d.loc[mask, "cltv_percentile"],
            s=14 + 40 * np.clip(d.loc[mask, "knowledge_risk"].to_numpy(float), 0, 1),
            alpha=0.45,
            marker=marker,
            label=action,
        )
    plt.xlabel("Calibrated ensemble churn probability")
    plt.ylabel("CLTV percentile")
    plt.title("KGDI-C: risk × strategic value (marker size = knowledge risk)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_C1_knowledge_decision_map.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.4, 6.0))
    plt.plot(
        100 * share_grid["high_value_churn_reach_rate"],
        share_grid["net_benefit_proxy"],
        marker="o",
        label="KGDI-C economic-share grid",
    )
    for _, r in policy_table.iterrows():
        plt.scatter(
            [100 * r["high_value_churn_reach_rate"]],
            [r["net_benefit_proxy"]],
            marker="x",
            s=70,
            label=r["strategy"],
        )
    plt.xlabel("High-value churn reach (%)")
    plt.ylabel("Outcome-anchored net benefit proxy ($)")
    plt.title("Budget-matched economic–strategic tradeoff")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_C2_economic_strategic_tradeoff.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.4, 5.8))
    plt.scatter(
        d["churn_probability"],
        d["knowledge_risk"],
        c=d["knowledge_residual"],
        s=8,
        alpha=0.35,
        cmap="coolwarm",
    )
    plt.plot([0, 1], [0, 1], ls="--", lw=1, color="black")
    plt.xlabel("Calibrated ensemble probability")
    plt.ylabel("OOF knowledge-graph risk")
    plt.title("Knowledge–ML residual (points above the diagonal: graph > ML)")
    plt.colorbar(label="knowledge residual k − p")
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_C3_knowledge_ml_residual.png", dpi=300)
    plt.close()


def primary_table(policy_table: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    boot_map = {r["comparison"]: r for _, r in boot.iterrows()}
    rows = []
    for _, r in policy_table.iterrows():
        delta = ci = ppos = ""
        if r["strategy"] == "Proposed_KGDI_C":
            vs_ca = boot_map["KGDI_C - Cost_Aware"]
            delta = f"{vs_ca['observed_delta']:.2f}"
            ci = f"[{vs_ca['ci_low']:.2f}, {vs_ca['ci_high']:.2f}]"
            ppos = f"{vs_ca['bootstrap_probability_gt_0']:.3f}"
        rows.append(
            {
                "Strategy": r["strategy"],
                "Intervention %": f"{100 * r['intervention_rate']:.1f}",
                "Churn reach %": f"{100 * r['churn_reach_rate']:.1f}",
                "High-value churn reach %": f"{100 * r['high_value_churn_reach_rate']:.1f}",
                "Offers": int(r["offers"]),
                "Human reviews": int(r["human_reviews"]),
                "Expected selected cost ($)": f"{r['expected_selected_cost']:.2f}",
                "Net benefit proxy ($)": f"{r['net_benefit_proxy']:.2f}",
                "KGDI-C Δ vs Cost-Aware ($)": delta,
                "95% bootstrap CI vs Cost-Aware": ci,
                "Bootstrap P(Δ>0) vs Cost-Aware": ppos,
            }
        )
    return pd.DataFrame(rows)


def load_inputs(exp_a: Path, data_path: Path):
    d = pd.read_csv(exp_a / "SUPPLEMENT_customer_level_oof_decisions.csv")
    with (exp_a / "run_config.json").open(encoding="utf-8") as f:
        run_config = json.load(f)
    params = run_config["business_parameters"]
    cfg = kgdi.BusinessConfig(**{k: params[k] for k in asdict(kgdi.BusinessConfig())})
    raw = kgdi.load_data(str(data_path))
    return d, cfg, raw, run_config


def main():
    args = parse_args()
    if not (0.0 <= args.economic_share <= 1.0):
        raise ValueError("--economic-share must be in [0, 1]")
    if args.bootstrap < 100:
        raise ValueError("--bootstrap must be >= 100")

    exp_a = Path(args.experiment_a).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    d, cfg, raw, run_config = load_inputs(exp_a, Path(args.data))
    d, inventory = attach_knowledge(d, raw)
    budget = planning_budget(d, cfg)

    frozen, frozen_actions = reconstruct_experiment_a(d, cfg, budget)
    kgc = policy_kgdi_c(d, cfg, budget, args.economic_share)
    kgc_metrics = metrics_row(d, kgc, cfg, budget, "Proposed_KGDI_C")
    policy_table = pd.concat([frozen, pd.DataFrame([kgc_metrics])], ignore_index=True)

    distinct_rows = [
        distinctness_row(kgc, frozen_actions["Cost_Aware"], "KGDI_C_vs_Cost_Aware"),
        distinctness_row(kgc, frozen_actions["Uncertainty_Review"], "KGDI_C_vs_Uncertainty_Review"),
        distinctness_row(kgc, frozen_actions["Proposed_KGDI_A"], "KGDI_C_vs_KGDI_A"),
        distinctness_row(
            frozen_actions["Proposed_KGDI_A"],
            frozen_actions["Uncertainty_Review"],
            "KGDI_A_vs_Uncertainty_Review",
        ),
    ]

    share_grid = run_share_grid(d, cfg, budget, DEFAULT_SHARE_GRID)
    abl = ablations(d, cfg, budget, args.economic_share)
    boot = bootstrap_deltas(d, cfg, args.bootstrap, args.seed, args.economic_share)
    table = primary_table(policy_table, boot)

    d_out = d.copy()
    d_out["action_Proposed_KGDI_C"] = kgc
    d_out.to_csv(output_dir / "EXPERIMENT_C_customer_decisions.csv", index=False)
    policy_table.to_csv(output_dir / "EXPERIMENT_C_policy_metrics.csv", index=False)
    share_grid.to_csv(output_dir / "EXPERIMENT_C_economic_share_grid.csv", index=False)
    abl.to_csv(output_dir / "EXPERIMENT_C_ablation.csv", index=False)
    boot.to_csv(output_dir / "EXPERIMENT_C_bootstrap.csv", index=False)
    pd.DataFrame(distinct_rows).to_csv(
        output_dir / "EXPERIMENT_C_policy_distinctness.csv", index=False
    )
    table.to_csv(output_dir / "TABLE_C_decision_performance.csv", index=False)
    with (output_dir / "EXPERIMENT_C_knowledge_graph.json").open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2)

    make_figures(d, policy_table, share_grid, kgc, output_dir, cfg)

    manifest = {
        "study": "Experiment C - knowledge-graph decision intelligence on frozen Experiment A OOF",
        "experiment_a_directory": str(exp_a),
        "experiment_a_git_commit": run_config.get("git_commit_sha"),
        "experiment_c_script_sha256": sha256_file(Path(__file__).resolve()),
        "primary_economic_allocation_fraction": args.economic_share,
        "economic_share_grid": DEFAULT_SHARE_GRID,
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "knowledge_graph": inventory,
        "business_parameters_frozen_from_experiment_a": asdict(cfg),
        "disclosure": (
            "Predictive models are frozen from Experiment A. Experiment C changes only the "
            "decision layer: a heterogeneous service/contract knowledge graph, OOF motif risk, "
            "and a prespecified 85/15 economic-offer vs knowledge-review budget split."
        ),
    }
    with (output_dir / "EXPERIMENT_C_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 84)
    print("EXPERIMENT C COMPLETED")
    print("=" * 84)
    print(table.to_string(index=False))
    print("\nBOOTSTRAP")
    print(boot.to_string(index=False))
    print("\nDISTINCTNESS")
    print(pd.DataFrame(distinct_rows).to_string(index=False))
    print(f"\nResults saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
