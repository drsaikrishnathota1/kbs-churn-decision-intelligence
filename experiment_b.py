#!/usr/bin/env python3
"""Prespecified Experiment B: multi-objective KGDI epsilon-constraint analysis."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, subprocess, sys, tarfile, tempfile
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import multiprocessing as mp
from scipy.optimize import Bounds, LinearConstraint, milp

import main as kgdi

EXPECTED_MAIN_SHA256 = "5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d"
EXPECTED_EXPERIMENT_A_COMMIT = "157fe440d97ad41178b2dc1c7183e1c8195ec6ba"
DEFAULT_EPSILON_GRID = [round(x / 100, 2) for x in range(101)]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def parse_grid(text: str | None) -> list[float]:
    if not text:
        return DEFAULT_EPSILON_GRID.copy()
    values = sorted({round(float(x.strip()), 6) for x in text.split(",") if x.strip()})
    if not values or any(x < 0 or x > 1 for x in values):
        raise ValueError("--epsilon-grid values must be in [0,1]")
    return values


def safe_extract_tar(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:*") as tf:
        root = destination.resolve()
        for member in tf.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        tf.extractall(destination)


def resolve_experiment_a(source: Path):
    source = source.resolve(); temp = None; root = source
    if source.is_file():
        temp = tempfile.TemporaryDirectory(prefix="kgdi_experiment_a_")
        root = Path(temp.name); safe_extract_tar(source, root)
    elif not source.is_dir():
        raise FileNotFoundError(source)
    matches = list(root.rglob("SUPPLEMENT_customer_level_oof_decisions.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Experiment A customer table; found {len(matches)}")
    exp_dir = matches[0].parent
    required = ["run_config.json", "results_manifest.json", "SUPPLEMENT_final_policy_metrics.csv"]
    missing = [x for x in required if not (exp_dir / x).is_file()]
    if missing: raise RuntimeError(f"Experiment A missing: {missing}")
    return exp_dir, temp


def load_source(source: Path):
    exp_dir, temp = resolve_experiment_a(source)
    with (exp_dir / "run_config.json").open(encoding="utf-8") as f:
        run_config = json.load(f)
    local_main = Path(kgdi.__file__).resolve()
    local_sha = sha256_file(local_main)
    if local_sha != EXPECTED_MAIN_SHA256:
        raise RuntimeError(f"Local main.py SHA mismatch: {local_sha}")
    if run_config.get("main_py_sha256") != EXPECTED_MAIN_SHA256:
        raise RuntimeError("Experiment A main.py SHA mismatch")
    if run_config.get("git_commit_sha") != EXPECTED_EXPERIMENT_A_COMMIT:
        raise RuntimeError(f"Experiment A commit mismatch: {run_config.get('git_commit_sha')}")

    with (exp_dir / "results_manifest.json").open(encoding="utf-8") as f:
        manifest = json.load(f)
    for name in ["SUPPLEMENT_customer_level_oof_decisions.csv", "SUPPLEMENT_final_policy_metrics.csv", "run_config.json"]:
        expected = manifest.get("files", {}).get(name, {}).get("sha256")
        if not expected or sha256_file(exp_dir / name) != expected:
            raise RuntimeError(f"Experiment A manifest hash mismatch: {name}")

    params = run_config.get("business_parameters", {})
    fields = asdict(kgdi.BusinessConfig()).keys()
    missing = [k for k in fields if k not in params]
    if missing: raise RuntimeError(f"Missing business parameters: {missing}")
    cfg = kgdi.BusinessConfig(**{k: params[k] for k in fields})

    d = pd.read_csv(exp_dir / "SUPPLEMENT_customer_level_oof_decisions.csv")
    required = [
        "CustomerID", kgdi.TARGET, "churn_probability", "uncertainty", "predictive_entropy",
        "model_disagreement", "CLTV", "Monthly Charges", "Tenure Months", "cltv_percentile",
        "annual_margin_value", "action_Probability_Threshold", "action_Cost_Aware",
        "action_Uncertainty_Review", "action_Proposed_KGDI",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing: raise RuntimeError(f"Customer table missing: {missing}")
    return exp_dir, temp, run_config, cfg, d, {
        "local_main_py_sha256": local_sha,
        "source_archive_sha256": sha256_file(source.resolve()) if source.is_file() else None,
        "source_customer_table_sha256": sha256_file(exp_dir / "SUPPLEMENT_customer_level_oof_decisions.csv"),
    }


def expected_utility_for_actions(d, actions, cfg) -> float:
    v = kgdi.business_vectors(d, cfg)
    return float(np.sum(v["offer_utility"][actions == "Retention Offer"]) + np.sum(v["review_utility"][actions == "Human Review"]))


def reconstruct_baselines(d, exp_dir, cfg):
    source = pd.read_csv(exp_dir / "SUPPLEMENT_final_policy_metrics.csv")
    budget = len(d) * cfg.offer_cost * cfg.budget_fraction
    mapping = {
        "Probability_Threshold": "action_Probability_Threshold",
        "Cost_Aware": "action_Cost_Aware",
        "Uncertainty_Review": "action_Uncertainty_Review",
        "Proposed_KGDI": "action_Proposed_KGDI",
    }
    rows = []
    for strategy, col in mapping.items():
        actions = d[col].to_numpy(dtype=object)
        metrics = kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget)
        src = source[source["strategy"] == strategy]
        if len(src) != 1: raise RuntimeError(f"Missing primary strategy: {strategy}")
        for key in ["expected_selected_cost", "net_benefit_proxy", "churn_reach_rate", "high_value_churn_reach_rate"]:
            if not np.isclose(float(metrics[key]), float(src.iloc[0][key]), rtol=1e-10, atol=1e-7):
                raise RuntimeError(f"Experiment A reconstruction failed: {strategy}/{key}")
        rows.append({"strategy": strategy, "expected_utility": expected_utility_for_actions(d, actions, cfg), **metrics})
    return pd.DataFrame(rows)


def build_candidates(d, cfg):
    v = kgdi.business_vectors(d, cfg); rows = []
    for i in range(len(d)):
        if v["p"][i] < cfg.minimum_churn_risk: continue
        high = v["cltv_pct"][i] >= cfg.high_value_quantile
        standard_uncertain = v["uncertainty"][i] >= cfg.uncertainty_threshold
        value_sensitive_review = (
            high and v["uncertainty"][i] >= cfg.moderate_uncertainty_threshold
            and v["review_utility"][i] > 0
            and (v["offer_utility"][i] <= 0 or v["review_utility"][i] >= cfg.review_utility_tolerance * v["offer_utility"][i])
        )
        if (standard_uncertain or value_sensitive_review) and v["review_utility"][i] > 0:
            action, utility, cost = "Human Review", float(v["review_utility"][i]), float(v["review_expected_cost"][i])
        elif v["offer_utility"][i] > 0:
            action, utility, cost = "Retention Offer", float(v["offer_utility"][i]), float(cfg.offer_cost)
        else: continue
        if utility <= 0 or cost <= 0: continue
        rows.append({
            "row_index": i, "CustomerID": str(d.iloc[i]["CustomerID"]), "action": action,
            "expected_utility": utility, "expected_cost": cost,
            "strategic_mass": float(v["p"][i]) if high else 0.0,
            "high_value": int(high), "churn_probability": float(v["p"][i]),
            "predictive_entropy": float(v["uncertainty"][i]), "cltv_percentile": float(v["cltv_pct"][i]),
        })
    c = pd.DataFrame(rows)
    if c.empty: raise RuntimeError("No Experiment B candidates")
    return c


def solve(candidates, budget, objective, min_strategic, time_limit, mip_rel_gap):
    utility = candidates["expected_utility"].to_numpy(float)
    cost = candidates["expected_cost"].to_numpy(float)
    strategic = candidates["strategic_mass"].to_numpy(float)
    c = -utility if objective == "utility" else -strategic
    constraints = [LinearConstraint(cost, -np.inf, budget)]
    if min_strategic > 0: constraints.append(LinearConstraint(strategic, min_strategic, np.inf))
    result = milp(
        c=c, integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))), constraints=constraints,
        options={"presolve": True, "time_limit": time_limit, "mip_rel_gap": mip_rel_gap, "disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {objective}; status={result.status}; {result.message}")
    return result.x > 0.5, result


def selected_actions(n, candidates, mask):
    actions = np.array(["No Action"] * n, dtype=object)
    for row in candidates.loc[mask].itertuples(index=False): actions[int(row.row_index)] = row.action
    return actions


def run_frontier(d, candidates, cfg, eps_grid, time_limit, mip_rel_gap):
    budget = len(d) * cfg.offer_cost * cfg.budget_fraction
    hv = d["cltv_percentile"].to_numpy(float) >= cfg.high_value_quantile
    total_hv_mass = float(np.sum(d["churn_probability"].to_numpy(float) * hv))
    max_mask, max_result = solve(candidates, budget, "strategic", 0.0, time_limit, mip_rel_gap)
    max_strategic = float(candidates.loc[max_mask, "strategic_mass"].sum())
    rows, action_map = [], {}
    for eps in eps_grid:
        target = eps * max_strategic
        if np.isclose(eps, 1.0): target = max_strategic - max(1e-9, abs(max_strategic) * 1e-9)
        mask, result = solve(candidates, budget, "utility", target, time_limit, mip_rel_gap)
        actions = selected_actions(len(d), candidates, mask); action_map[eps] = actions
        chosen = candidates.loc[mask]; strategic = float(chosen["strategic_mass"].sum())
        rows.append({
            "epsilon_fraction_of_max_strategic": eps, "strategic_constraint_target": target,
            "solver_status": int(result.status), "solver_message": str(result.message),
            "mip_gap_reported": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None else np.nan,
            "expected_utility": float(chosen["expected_utility"].sum()),
            "expected_high_value_mass_reached": strategic,
            "expected_high_value_reach_fraction": strategic / total_hv_mass,
            "fraction_of_budget_constrained_max_strategic": strategic / max_strategic,
            **kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget),
        })
    return pd.DataFrame(rows), action_map, {
        "budget": budget, "total_expected_high_value_mass": total_hv_mass,
        "max_budget_constrained_strategic_mass": max_strategic,
        "max_strategic_solver_status": int(max_result.status),
        "max_strategic_solver_message": str(max_result.message),
    }


def nondominated(frontier):
    f = frontier.copy().sort_values(["expected_high_value_reach_fraction", "expected_utility"])
    keep = []
    vals = f[["expected_high_value_reach_fraction", "expected_utility"]].to_numpy(float)
    for i, (s, u) in enumerate(vals):
        dominated = np.any((vals[:, 0] >= s - 1e-12) & (vals[:, 1] >= u - 1e-9) & ((vals[:, 0] > s + 1e-12) | (vals[:, 1] > u + 1e-9)))
        keep.append(not dominated)
    return f.loc[keep].drop_duplicates(subset=["expected_high_value_reach_fraction", "expected_utility"]).reset_index(drop=True)


def rerun_primary(sample, cfg):
    s = sample.copy().reset_index(drop=True)
    s["cltv_percentile"] = kgdi.percentile_rank(s["CLTV"].to_numpy(float))
    s["annual_margin_value"] = s["Monthly Charges"].to_numpy(float) * cfg.value_horizon_months * cfg.gross_margin_rate
    budget = len(s) * cfg.offer_cost * cfg.budget_fraction
    actions = {
        "Probability_Threshold": kgdi.policy_probability_threshold(s, cfg, budget),
        "Cost_Aware": kgdi.policy_cost_aware(s, cfg, budget),
        "Uncertainty_Review": kgdi.policy_uncertainty_review(s, cfg, budget),
        "Proposed_KGDI": kgdi.policy_kgdi(s, cfg, budget),
    }
    return {k: kgdi.evaluate_policy(s, a, cfg, planning_budget=budget) for k, a in actions.items()}


_BOOT_D = None
_BOOT_CFG = None


def _bootstrap_init(d, cfg):
    global _BOOT_D, _BOOT_CFG
    _BOOT_D, _BOOT_CFG = d, cfg


def _bootstrap_once(sample_seed):
    rng = np.random.default_rng(int(sample_seed)); n = len(_BOOT_D)
    m = rerun_primary(_BOOT_D.iloc[rng.integers(0, n, size=n)].reset_index(drop=True), _BOOT_CFG)
    return (
        m["Proposed_KGDI"]["net_benefit_proxy"] - m["Probability_Threshold"]["net_benefit_proxy"],
        m["Proposed_KGDI"]["net_benefit_proxy"] - m["Cost_Aware"]["net_benefit_proxy"],
        m["Proposed_KGDI"]["net_benefit_proxy"] - m["Uncertainty_Review"]["net_benefit_proxy"],
    )


def bootstrap_primary(d, cfg, n_boot, seed, workers):
    observed = rerun_primary(d, cfg); baselines = ["Probability_Threshold", "Cost_Aware", "Uncertainty_Review"]
    seed_rng = np.random.default_rng(seed); sample_seeds = seed_rng.integers(0, np.iinfo(np.uint32).max, size=n_boot, dtype=np.uint32).tolist()
    workers = min(int(workers), int(n_boot))
    if workers == 1:
        _bootstrap_init(d, cfg); results = [_bootstrap_once(s) for s in sample_seeds]
    else:
        methods = mp.get_all_start_methods(); method = "fork" if "fork" in methods else methods[0]
        ctx = mp.get_context(method); chunksize = max(1, n_boot // (workers * 8))
        with ctx.Pool(processes=workers, initializer=_bootstrap_init, initargs=(d, cfg)) as pool:
            results = pool.map(_bootstrap_once, sample_seeds, chunksize=chunksize)
    matrix = np.asarray(results, dtype=float)
    return pd.DataFrame([{
        "comparison": f"Proposed_KGDI - {b}",
        "observed_delta": observed["Proposed_KGDI"]["net_benefit_proxy"] - observed[b]["net_benefit_proxy"],
        "bootstrap_mean_delta": float(matrix[:, j].mean()), "ci_low": float(np.percentile(matrix[:, j], 2.5)),
        "ci_high": float(np.percentile(matrix[:, j], 97.5)),
        "bootstrap_proportion_delta_gt_0": float(np.mean(matrix[:, j] > 0)), "bootstrap_samples": n_boot,
    } for j, b in enumerate(baselines)])


def save_customer_actions(d, action_map, path):
    base = d[["CustomerID", kgdi.TARGET, "churn_probability", "predictive_entropy", "model_disagreement", "CLTV", "cltv_percentile", "annual_margin_value"]].copy()
    action_cols = {f"action_epsilon_{eps:.2f}".replace(".", "p"): actions for eps, actions in action_map.items()}
    out = pd.concat([base, pd.DataFrame(action_cols, index=base.index)], axis=1)
    out.to_csv(path, index=False)


def expected_strategic_fraction(d, actions, cfg):
    hv = d["cltv_percentile"].to_numpy(float) >= cfg.high_value_quantile
    mass = d["churn_probability"].to_numpy(float) * hv; denom = float(mass.sum())
    return float(mass[actions != "No Action"].sum() / denom)


def make_figures(d, baselines, frontier, output_dir, cfg):
    actions = d["action_Proposed_KGDI"].to_numpy(dtype=object)
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}
    plt.figure(figsize=(8.6, 6.2)); sizes = 10 + 45 * d["predictive_entropy"].to_numpy(float)
    for a in markers:
        mask = actions == a
        if np.any(mask): plt.scatter(d.loc[mask, "churn_probability"], d.loc[mask, "cltv_percentile"], s=sizes[mask], alpha=.42, marker=markers[a], label=a)
    plt.xlabel("Calibrated ensemble churn probability"); plt.ylabel("CLTV percentile"); plt.title("Knowledge-Guided Decision Map: Risk x Strategic Value"); plt.legend(frameon=False); plt.tight_layout(); plt.savefig(output_dir / "FIGURE_B0_knowledge_decision_map.png", dpi=300); plt.close()

    plt.figure(figsize=(8.2, 5.9)); plt.plot(100 * frontier["expected_high_value_reach_fraction"], frontier["expected_utility"], marker="o", label="Experiment B frontier")
    mapping = {"Probability_Threshold":"action_Probability_Threshold","Cost_Aware":"action_Cost_Aware","Uncertainty_Review":"action_Uncertainty_Review","Proposed_KGDI":"action_Proposed_KGDI"}
    for strategy, col in mapping.items():
        a = d[col].to_numpy(dtype=object); x = 100 * expected_strategic_fraction(d, a, cfg); y = float(baselines.loc[baselines["strategy"] == strategy, "expected_utility"].iloc[0]); plt.scatter([x],[y],marker="x",s=65,label=strategy)
    plt.xlabel("Expected high-value churn mass reached (%)"); plt.ylabel("Expected economic utility ($)"); plt.title("Experiment B: Epsilon-Constraint Pareto Frontier"); plt.legend(frameon=False,fontsize=8); plt.tight_layout(); plt.savefig(output_dir / "FIGURE_B1_pareto_frontier.png", dpi=300); plt.close()

    plt.figure(figsize=(8.2, 5.9)); plt.plot(100 * frontier["high_value_churn_reach_rate"], frontier["net_benefit_proxy"], marker="o", label="Experiment B frontier")
    for _, r in baselines.iterrows(): plt.scatter([100*r["high_value_churn_reach_rate"]],[r["net_benefit_proxy"]],marker="x",s=65,label=r["strategy"])
    plt.xlabel("Realized high-value churn reach (%)"); plt.ylabel("Outcome-anchored net benefit proxy ($)"); plt.title("Experiment B: Realized Economic-Strategic Tradeoff"); plt.legend(frameon=False,fontsize=8); plt.tight_layout(); plt.savefig(output_dir / "FIGURE_B2_realized_tradeoff.png", dpi=300); plt.close()


def build_manifest(output_dir, source, exp_dir, run_config, verification, eps_grid, cfg, args, solver_info):
    files = {p.name:{"sha256":sha256_file(p),"bytes":p.stat().st_size} for p in sorted(output_dir.iterdir()) if p.is_file() and p.name != "EXPERIMENT_B_manifest.json"}
    return {
        "study":"Experiment B - prespecified multi-objective KGDI epsilon-constraint analysis",
        "source_experiment_a":str(source.resolve()), "source_experiment_a_directory":str(exp_dir.resolve()),
        "source_experiment_a_git_commit":run_config.get("git_commit_sha"), "source_main_py_sha256":run_config.get("main_py_sha256"),
        "experiment_b_git_commit":git_value("rev-parse","HEAD"), "experiment_b_script_sha256":sha256_file(Path(__file__).resolve()),
        "verification":verification, "python_version":sys.version, "platform":platform.platform(), "scipy_version":scipy.__version__,
        "epsilon_grid_fraction_of_budget_constrained_max_strategic":eps_grid,
        "business_parameters_frozen_from_experiment_a":asdict(cfg), "bootstrap_samples_primary_comparisons":args.bootstrap, "bootstrap_seed":args.seed, "bootstrap_workers":args.bootstrap_workers,
        "optimization_disclosure":"MILP uses only frozen OOF probability, predictive entropy, CLTV percentile, and frozen business assumptions. Churn Value is excluded from optimization and used only for ex-post evaluation. All prespecified epsilon points are reported; no post-hoc winning point is selected.",
        "strategic_objective_definition":"Expected high-value churn mass = sum(p_i) over selected customers at or above the frozen CLTV high-value quantile; epsilon is a fraction of its budget-constrained maximum.",
        "solver":solver_info, "files":files,
    }


def main_cli():
    p = argparse.ArgumentParser(description="Experiment B multi-objective KGDI epsilon-constraint analysis", allow_abbrev=False)
    p.add_argument("--experiment-a", required=True); p.add_argument("--output", default="results/experiment_b")
    p.add_argument("--epsilon-grid", default=None, help="Optional comma-separated override; default is 0.00..1.00 by 0.01")
    p.add_argument("--bootstrap", type=int, default=1000); p.add_argument("--bootstrap-workers", type=int, default=8); p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--solver-time-limit", type=float, default=120.0); p.add_argument("--mip-rel-gap", type=float, default=1e-9)
    args = p.parse_args()
    if args.bootstrap < 100 or args.bootstrap_workers < 1 or args.solver_time_limit <= 0 or args.mip_rel_gap < 0: raise ValueError("Invalid bootstrap/solver arguments")
    eps_grid = parse_grid(args.epsilon_grid); source = Path(args.experiment_a); output_dir = Path(args.output).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    exp_dir, temp, run_config, cfg, d, verification = load_source(source)
    try:
        baselines = reconstruct_baselines(d, exp_dir, cfg); candidates = build_candidates(d, cfg)
        frontier, action_map, solver_info = run_frontier(d, candidates, cfg, eps_grid, args.solver_time_limit, args.mip_rel_gap)
        nd = nondominated(frontier); boot = bootstrap_primary(d, cfg, args.bootstrap, args.seed, args.bootstrap_workers)
        baselines.to_csv(output_dir / "EXPERIMENT_A_reconstructed_baselines.csv", index=False); boot.to_csv(output_dir / "EXPERIMENT_A_bootstrap_all_comparisons.csv", index=False)
        candidates.to_csv(output_dir / "EXPERIMENT_B_candidate_table.csv", index=False); frontier.to_csv(output_dir / "EXPERIMENT_B_frontier_all_epsilon.csv", index=False); nd.to_csv(output_dir / "EXPERIMENT_B_pareto_nondominated.csv", index=False)
        save_customer_actions(d, action_map, output_dir / "EXPERIMENT_B_customer_actions.csv"); make_figures(d, baselines, nd, output_dir, cfg)
        atomic_json(output_dir / "EXPERIMENT_B_manifest.json", build_manifest(output_dir, source, exp_dir, run_config, verification, eps_grid, cfg, args, solver_info))
        print("="*84); print("EXPERIMENT B COMPLETED"); print("="*84); print("Frozen Experiment A reconstruction: PASS"); print(f"Candidates: {len(candidates)} | Budget: ${solver_info['budget']:.2f} | Epsilon points: {len(eps_grid)} | Non-dominated points: {len(nd)}")
        print("\nNON-DOMINATED EXPERIMENT B FRONTIER"); print(nd[["epsilon_fraction_of_max_strategic","expected_utility","expected_high_value_reach_fraction","net_benefit_proxy","high_value_churn_reach_rate","churn_reach_rate","interventions"]].to_string(index=False))
        print("\nPRIMARY POLICY-RERUN BOOTSTRAP COMPARISONS"); print(boot.to_string(index=False)); print(f"\nResults saved to: {output_dir}"); return 0
    finally:
        if temp is not None: temp.cleanup()


if __name__ == "__main__": raise SystemExit(main_cli())
