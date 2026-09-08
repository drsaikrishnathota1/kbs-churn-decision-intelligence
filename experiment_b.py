#!/usr/bin/env python3
"""
Experiment B: locked post-primary multi-objective KGDI secondary analysis.

This script consumes the immutable Experiment A archive and does NOT refit any
predictive model. It solves a knowledge-gated binary MILP over feasible direct-
offer and human-review actions to quantify the tradeoff between expected
economic utility and expected engagement of high-value churn risk.

Observed Churn Value is excluded from candidate construction, objectives,
constraints, and operating-point selection. It is used only for ex-post,
outcome-anchored descriptive evaluation after an allocation has been selected.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import main as kgdi

EXPECTED_MAIN_SHA256 = "5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d"
EXPECTED_EXPERIMENT_A_COMMIT = "157fe440d97ad41178b2dc1c7183e1c8195ec6ba"
EXPECTED_EXPERIMENT_A_ARCHIVE_SHA256 = "362b250a5fa5d4ba8e9ac8177d9aa6253e26933321fcd5b8a26ff474c39b3c2a"
EXPECTED_DATASET_SHA256 = "9b7d90f9e4c0f9b607126640e4279f0afd4120e7b3485614770753c9f393aaae"
EXPECTED_ROWS = 7043
DEFAULT_EPSILON_GRID = [round(i / 100.0, 2) for i in range(101)]
UTILITY_LOSS_LEVELS = [0.00, 0.01, 0.02, 0.05]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260907
MIP_REL_GAP = 1e-6
ALLOWED_ACTIONS = {"No Action", "Retention Offer", "Human Review"}

PACKAGE_MAP = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit_learn": "scikit-learn",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}


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
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def git_is_clean() -> bool:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        ).strip() == ""
    except Exception:
        return False


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for run_key, package_name in PACKAGE_MAP.items():
        try:
            out[run_key] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            out[run_key] = None
    return out


def safe_extract_tar(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as tf:
        for member in tf.getmembers():
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise RuntimeError(f"Unsafe archive member type: {member.name}")
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
        tf.extractall(destination)


def ensure_fresh_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            f"Output directory is not empty: {output_dir}. "
            "Use a new directory for the definitive Experiment B run."
        )


def resolve_experiment_a(source: Path):
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            "Experiment B requires the immutable Experiment A .tar.gz archive: "
            f"{source}"
        )
    archive_sha = sha256_file(source)
    if archive_sha != EXPECTED_EXPERIMENT_A_ARCHIVE_SHA256:
        raise RuntimeError(
            "Experiment A archive SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_EXPERIMENT_A_ARCHIVE_SHA256}\n"
            f"Observed: {archive_sha}"
        )
    temp = tempfile.TemporaryDirectory(prefix="kgdi_experiment_a_")
    root = Path(temp.name)
    safe_extract_tar(source, root)
    matches = list(root.rglob("SUPPLEMENT_customer_level_oof_decisions.csv"))
    if len(matches) != 1:
        temp.cleanup()
        raise RuntimeError(
            f"Expected exactly one Experiment A customer table; found {len(matches)}"
        )
    return source, matches[0].parent, temp, archive_sha


def verify_manifest(exp_dir: Path) -> dict:
    manifest_path = exp_dir / "results_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Experiment A results_manifest.json is missing")
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("git_commit_sha") != EXPECTED_EXPERIMENT_A_COMMIT:
        raise RuntimeError("Experiment A manifest Git commit mismatch")
    if manifest.get("main_py_sha256") != EXPECTED_MAIN_SHA256:
        raise RuntimeError("Experiment A manifest main.py SHA mismatch")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Experiment A manifest has no file inventory")

    verified = 0
    for name, meta in files.items():
        p = exp_dir / name
        if not p.is_file():
            raise RuntimeError(f"Experiment A manifest file missing: {name}")
        expected_sha = meta.get("sha256")
        expected_bytes = meta.get("bytes")
        observed_sha = sha256_file(p)
        observed_bytes = p.stat().st_size
        if observed_sha != expected_sha:
            raise RuntimeError(f"Experiment A manifest SHA mismatch: {name}")
        if expected_bytes is not None and int(expected_bytes) != int(observed_bytes):
            raise RuntimeError(f"Experiment A manifest byte-size mismatch: {name}")
        verified += 1
    return {"manifest_files_verified": verified, "manifest": manifest}


def verify_environment(run_config: dict, allow_version_drift: bool) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Definitive Experiment B requires Python 3.11; observed {sys.version.split()[0]}"
        )
    observed = package_versions()
    expected_all = run_config.get("software_versions", {})
    mismatches = {}
    for run_key in PACKAGE_MAP:
        expected = expected_all.get(run_key)
        current = observed.get(run_key)
        if expected is not None and current != expected:
            mismatches[run_key] = {"expected": expected, "observed": current}
    if mismatches and not allow_version_drift:
        raise RuntimeError(
            "Package-version drift from definitive Experiment A environment: "
            + json.dumps(mismatches, indent=2)
            + "\nInstall requirements-runpod.txt or use --allow-version-drift only for development."
        )
    return {
        "python": sys.version,
        "packages": observed,
        "version_mismatches": mismatches,
        "version_drift_allowed": bool(allow_version_drift),
    }


def validate_customer_table(d: pd.DataFrame) -> dict:
    required = [
        "CustomerID", kgdi.TARGET, "churn_probability", "uncertainty",
        "predictive_entropy", "model_disagreement", "CLTV", "Monthly Charges",
        "Tenure Months", "cltv_percentile", "annual_margin_value",
        "action_Probability_Threshold", "action_Cost_Aware",
        "action_Uncertainty_Review", "action_Proposed_KGDI",
    ]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise RuntimeError(f"Experiment A customer table missing columns: {missing}")
    if len(d) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} customer rows; observed {len(d)}")
    if d["CustomerID"].isna().any() or d["CustomerID"].duplicated().any():
        raise RuntimeError("CustomerID must be complete and unique")

    target = pd.to_numeric(d[kgdi.TARGET], errors="coerce")
    if target.isna().any() or not set(target.astype(int).unique()).issubset({0, 1}):
        raise RuntimeError("Churn Value must be complete binary 0/1")

    bounded = ["churn_probability", "uncertainty", "predictive_entropy", "cltv_percentile"]
    for col in bounded:
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all() or np.any(x < -1e-12) or np.any(x > 1.0 + 1e-12):
            raise RuntimeError(f"Invalid finite/range values in {col}")
    disagreement = pd.to_numeric(d["model_disagreement"], errors="coerce").to_numpy(float)
    if not np.isfinite(disagreement).all() or np.any(disagreement < -1e-12):
        raise RuntimeError("Invalid model_disagreement values")
    for col in ["CLTV", "Monthly Charges", "Tenure Months", "annual_margin_value"]:
        x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
        if not np.isfinite(x).all() or np.any(x < -1e-12):
            raise RuntimeError(f"Invalid nonnegative values in {col}")
    if not np.allclose(
        d["uncertainty"].to_numpy(float),
        d["predictive_entropy"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Experiment A uncertainty and predictive_entropy are inconsistent")
    for col in [
        "action_Probability_Threshold", "action_Cost_Aware",
        "action_Uncertainty_Review", "action_Proposed_KGDI",
    ]:
        values = set(d[col].astype(str).unique())
        if not values.issubset(ALLOWED_ACTIONS):
            raise RuntimeError(f"Unexpected action labels in {col}: {sorted(values)}")
    return {
        "customer_rows_validated": int(len(d)),
        "customer_ids_unique": True,
        "numeric_ranges_validated": True,
        "action_labels_validated": True,
    }


def load_source(source: Path, allow_version_drift: bool):
    source, exp_dir, temp, archive_sha = resolve_experiment_a(source)
    manifest_info = verify_manifest(exp_dir)

    run_config_path = exp_dir / "run_config.json"
    run_status_path = exp_dir / "run_status.json"
    if not run_config_path.is_file() or not run_status_path.is_file():
        raise RuntimeError("Experiment A run_config.json or run_status.json is missing")
    with run_config_path.open(encoding="utf-8") as f:
        run_config = json.load(f)
    with run_status_path.open(encoding="utf-8") as f:
        run_status = json.load(f)

    if run_status.get("status") != "COMPLETED":
        raise RuntimeError(f"Experiment A status is not COMPLETED: {run_status.get('status')}")
    if run_config.get("git_commit_sha") != EXPECTED_EXPERIMENT_A_COMMIT:
        raise RuntimeError("Experiment A run_config Git commit mismatch")
    if run_config.get("main_py_sha256") != EXPECTED_MAIN_SHA256:
        raise RuntimeError("Experiment A run_config main.py SHA mismatch")
    if run_config.get("dataset_sha256") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Experiment A dataset SHA mismatch")

    local_main = Path(kgdi.__file__).resolve()
    local_main_sha = sha256_file(local_main)
    if local_main_sha != EXPECTED_MAIN_SHA256:
        raise RuntimeError(
            f"Local frozen main.py SHA mismatch: {local_main_sha}. "
            "Do not modify Experiment A main.py."
        )

    environment = verify_environment(run_config, allow_version_drift)

    params = run_config.get("business_parameters", {})
    fields = list(asdict(kgdi.BusinessConfig()).keys())
    missing_params = [k for k in fields if k not in params]
    if missing_params:
        raise RuntimeError(f"Experiment A business parameters missing: {missing_params}")
    cfg = kgdi.BusinessConfig(**{k: params[k] for k in fields})

    d = pd.read_csv(exp_dir / "SUPPLEMENT_customer_level_oof_decisions.csv")
    customer_validation = validate_customer_table(d)

    verification = {
        "source_archive_sha256": archive_sha,
        "local_main_py_sha256": local_main_sha,
        "source_customer_table_sha256": sha256_file(
            exp_dir / "SUPPLEMENT_customer_level_oof_decisions.csv"
        ),
        "source_git_commit": run_config.get("git_commit_sha"),
        "source_dataset_sha256": run_config.get("dataset_sha256"),
        "manifest_files_verified": manifest_info["manifest_files_verified"],
        **customer_validation,
        "environment": environment,
    }
    return source, exp_dir, temp, run_config, cfg, d, verification


def expected_utility_for_actions(d, actions, cfg) -> float:
    v = kgdi.business_vectors(d, cfg)
    offer = actions == "Retention Offer"
    review = actions == "Human Review"
    return float(np.sum(v["offer_utility"][offer]) + np.sum(v["review_utility"][review]))


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
        if len(src) != 1:
            raise RuntimeError(f"Missing Experiment A strategy: {strategy}")
        src_row = src.iloc[0]
        for key, observed in metrics.items():
            if key in src.columns and pd.notna(src_row[key]):
                expected = float(src_row[key])
                if not np.isclose(float(observed), expected, rtol=1e-10, atol=1e-7):
                    raise RuntimeError(
                        f"Experiment A reconstruction failed: {strategy}/{key}: "
                        f"expected={expected}, reconstructed={observed}"
                    )
        rows.append({
            "strategy": strategy,
            "expected_utility": expected_utility_for_actions(d, actions, cfg),
            **metrics,
        })
    return pd.DataFrame(rows)


def build_candidates(d, cfg):
    """Build the knowledge-gated action set without using observed Churn Value."""
    v = kgdi.business_vectors(d, cfg)
    rows: list[dict] = []
    for i in range(len(d)):
        p = float(v["p"][i])
        if p < cfg.minimum_churn_risk:
            continue

        high_value = bool(v["cltv_pct"][i] >= cfg.high_value_quantile)
        standard_uncertain = bool(v["uncertainty"][i] >= cfg.uncertainty_threshold)
        value_sensitive_review = bool(
            high_value
            and v["uncertainty"][i] >= cfg.moderate_uncertainty_threshold
            and v["review_utility"][i] > 0
            and (
                v["offer_utility"][i] <= 0
                or v["review_utility"][i]
                >= cfg.review_utility_tolerance * v["offer_utility"][i]
            )
        )

        common = {
            "row_index": int(i),
            "CustomerID": str(d.iloc[i]["CustomerID"]),
            "high_value": int(high_value),
            "churn_probability": p,
            "predictive_entropy": float(v["uncertainty"][i]),
            "cltv_percentile": float(v["cltv_pct"][i]),
            "strategic_mass": p if high_value else 0.0,
        }

        if v["offer_utility"][i] > 0:
            rows.append({
                **common,
                "action": "Retention Offer",
                "expected_utility": float(v["offer_utility"][i]),
                "expected_cost": float(cfg.offer_cost),
            })

        review_eligible = standard_uncertain or value_sensitive_review
        if review_eligible and v["review_utility"][i] > 0:
            rows.append({
                **common,
                "action": "Human Review",
                "expected_utility": float(v["review_utility"][i]),
                "expected_cost": float(v["review_expected_cost"][i]),
            })

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise RuntimeError("No Experiment B candidates were generated")
    if not np.isfinite(candidates[["expected_utility", "expected_cost", "strategic_mass"]].to_numpy(float)).all():
        raise RuntimeError("Non-finite Experiment B candidate values")
    if (candidates["expected_utility"] <= 0).any() or (candidates["expected_cost"] <= 0).any():
        raise RuntimeError("Experiment B candidates must have positive utility and cost")
    pair_dupes = candidates.duplicated(subset=["row_index", "action"]).sum()
    if pair_dupes:
        raise RuntimeError(f"Duplicate customer-action candidates detected: {pair_dupes}")
    return candidates.reset_index(drop=True)


def customer_exclusivity_constraint(candidates: pd.DataFrame, n_customers: int):
    cols = np.arange(len(candidates), dtype=int)
    rows = candidates["row_index"].to_numpy(int)
    data = np.ones(len(candidates), dtype=float)
    matrix = coo_matrix((data, (rows, cols)), shape=(n_customers, len(candidates))).tocsr()
    return LinearConstraint(matrix, -np.inf, np.ones(n_customers, dtype=float))


def solve(
    candidates: pd.DataFrame,
    n_customers: int,
    budget: float,
    objective: str,
    min_strategic: float,
    time_limit: float,
    mip_rel_gap: float,
):
    utility = candidates["expected_utility"].to_numpy(float)
    cost = candidates["expected_cost"].to_numpy(float)
    strategic = candidates["strategic_mass"].to_numpy(float)
    if objective == "utility":
        objective_vector = -utility
    elif objective == "strategic":
        objective_vector = -strategic
    else:
        raise ValueError(f"Unknown objective: {objective}")

    constraints = [
        LinearConstraint(cost, -np.inf, float(budget)),
        customer_exclusivity_constraint(candidates, n_customers),
    ]
    if min_strategic > 0:
        constraints.append(LinearConstraint(strategic, float(min_strategic), np.inf))

    result = milp(
        c=objective_vector,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraints,
        options={
            "presolve": True,
            "time_limit": float(time_limit),
            "mip_rel_gap": float(mip_rel_gap),
            "disp": False,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"MILP failed: objective={objective}; status={result.status}; {result.message}"
        )
    mask = np.asarray(result.x) > 0.5
    return mask, result


def validate_selection(
    candidates: pd.DataFrame,
    mask: np.ndarray,
    budget: float,
    min_strategic: float,
    tolerance: float = 1e-6,
) -> dict:
    chosen = candidates.loc[mask]
    selected_cost = float(chosen["expected_cost"].sum())
    selected_strategic = float(chosen["strategic_mass"].sum())
    if selected_cost > budget + max(tolerance, abs(budget) * 1e-8):
        raise RuntimeError(
            f"Post-solution budget violation: cost={selected_cost}, budget={budget}"
        )
    if selected_strategic + max(tolerance, abs(min_strategic) * 1e-8) < min_strategic:
        raise RuntimeError(
            f"Post-solution epsilon violation: strategic={selected_strategic}, target={min_strategic}"
        )
    if chosen["row_index"].duplicated().any():
        raise RuntimeError("Post-solution customer exclusivity violation")
    return {
        "expected_cost_from_candidates": selected_cost,
        "strategic_mass_from_candidates": selected_strategic,
        "selected_candidates": int(mask.sum()),
        "selected_customers": int(chosen["row_index"].nunique()),
    }


def selected_actions(n_customers: int, candidates: pd.DataFrame, mask: np.ndarray):
    actions = np.array(["No Action"] * n_customers, dtype=object)
    chosen = candidates.loc[mask]
    if chosen["row_index"].duplicated().any():
        raise RuntimeError("Cannot construct actions: more than one action selected for a customer")
    for row in chosen.itertuples(index=False):
        actions[int(row.row_index)] = row.action
    return actions


def solver_metadata(result) -> dict:
    return {
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": (
            float(result.mip_gap)
            if getattr(result, "mip_gap", None) is not None
            else None
        ),
        "mip_node_count": (
            int(result.mip_node_count)
            if getattr(result, "mip_node_count", None) is not None
            else None
        ),
    }


def run_frontier(d, candidates, cfg, eps_grid, time_limit, mip_rel_gap):
    budget = float(len(d) * cfg.offer_cost * cfg.budget_fraction)
    high_value = d["cltv_percentile"].to_numpy(float) >= cfg.high_value_quantile
    expected_hv_mass_all = d["churn_probability"].to_numpy(float) * high_value
    total_hv_mass = float(expected_hv_mass_all.sum())
    if total_hv_mass <= 0:
        raise RuntimeError("Total expected high-value churn-risk mass is non-positive")

    max_mask, max_result = solve(
        candidates, len(d), budget, "strategic", 0.0, time_limit, mip_rel_gap
    )
    max_check = validate_selection(candidates, max_mask, budget, 0.0)
    max_strategic = max_check["strategic_mass_from_candidates"]
    if max_strategic <= 0:
        raise RuntimeError("Budget-constrained maximum strategic mass is non-positive")

    utility_mask, utility_result = solve(
        candidates, len(d), budget, "utility", 0.0, time_limit, mip_rel_gap
    )
    utility_check = validate_selection(candidates, utility_mask, budget, 0.0)
    max_utility = float(candidates.loc[utility_mask, "expected_utility"].sum())
    if max_utility <= 0:
        raise RuntimeError("Maximum expected utility is non-positive")

    rows = []
    action_map: dict[float, np.ndarray] = {}
    endpoint_tolerance = max(1e-8, abs(max_strategic) * 1e-8)

    for j, eps in enumerate(eps_grid, start=1):
        target = float(eps * max_strategic)
        if np.isclose(eps, 1.0):
            target = max(0.0, max_strategic - endpoint_tolerance)
        mask, result = solve(
            candidates, len(d), budget, "utility", target, time_limit, mip_rel_gap
        )
        checks = validate_selection(candidates, mask, budget, target)
        chosen = candidates.loc[mask]
        actions = selected_actions(len(d), candidates, mask)
        action_map[eps] = actions
        expected_utility = float(chosen["expected_utility"].sum())
        strategic = checks["strategic_mass_from_candidates"]
        metrics = kgdi.evaluate_policy(d, actions, cfg, planning_budget=budget)

        if not np.isclose(
            float(metrics["expected_selected_cost"]),
            checks["expected_cost_from_candidates"],
            rtol=1e-10,
            atol=1e-7,
        ):
            raise RuntimeError(f"Expected-cost mismatch at epsilon={eps:.2f}")

        meta = solver_metadata(result)
        rows.append({
            "epsilon_fraction_of_max_strategic": float(eps),
            "strategic_constraint_target": target,
            "solver_status": meta["status"],
            "solver_message": meta["message"],
            "mip_gap_reported": meta["mip_gap"],
            "mip_node_count": meta["mip_node_count"],
            "expected_utility": expected_utility,
            "expected_utility_retained_fraction": expected_utility / max_utility,
            "expected_high_value_mass_reached": strategic,
            "expected_high_value_reach_fraction": strategic / total_hv_mass,
            "fraction_of_budget_constrained_max_strategic": strategic / max_strategic,
            **metrics,
        })
        if j == 1 or j == len(eps_grid) or j % 10 == 0:
            print(
                f"Solved epsilon {eps:.2f} ({j}/{len(eps_grid)}): "
                f"utility=${expected_utility:,.2f}, strategic={strategic:.6f}"
            )

    frontier = pd.DataFrame(rows)

    # Independent endpoint checks against separately solved utility/strategic programs.
    if 0.0 in action_map:
        eps0 = frontier.loc[
            np.isclose(frontier["epsilon_fraction_of_max_strategic"], 0.0)
        ].iloc[0]
        if not np.isclose(float(eps0["expected_utility"]), max_utility, rtol=1e-7, atol=1e-5):
            raise RuntimeError("epsilon=0 does not reproduce the economic-utility optimum")
    if 1.0 in action_map:
        eps1 = frontier.loc[
            np.isclose(frontier["epsilon_fraction_of_max_strategic"], 1.0)
        ].iloc[0]
        if float(eps1["expected_high_value_mass_reached"]) < max_strategic - 2 * endpoint_tolerance:
            raise RuntimeError("epsilon=1 does not reach the strategic endpoint")

    gap_values = pd.to_numeric(frontier["mip_gap_reported"], errors="coerce").dropna()
    if len(gap_values) and float(gap_values.max()) > max(5e-6, mip_rel_gap * 5.0):
        raise RuntimeError(
            f"Reported MIP gap exceeds validation tolerance: {gap_values.max()}"
        )

    solver_info = {
        "budget": budget,
        "total_expected_high_value_mass": total_hv_mass,
        "max_budget_constrained_strategic_mass": max_strategic,
        "max_expected_utility": max_utility,
        "strategic_endpoint_solver": solver_metadata(max_result),
        "economic_endpoint_solver": solver_metadata(utility_result),
        "mip_rel_gap_requested": float(mip_rel_gap),
        "solver_time_limit_seconds": float(time_limit),
        "epsilon_points": int(len(eps_grid)),
        "endpoint_tolerance": endpoint_tolerance,
    }
    return frontier, action_map, solver_info


def nondominated(frontier: pd.DataFrame) -> pd.DataFrame:
    f = frontier.copy().sort_values(
        ["expected_high_value_mass_reached", "expected_utility"],
        ascending=[True, False],
    )
    values = f[["expected_high_value_mass_reached", "expected_utility"]].to_numpy(float)
    keep = []
    for i, (strategic, utility) in enumerate(values):
        dominated = np.any(
            (values[:, 0] >= strategic - 1e-10)
            & (values[:, 1] >= utility - 1e-7)
            & (
                (values[:, 0] > strategic + 1e-10)
                | (values[:, 1] > utility + 1e-7)
            )
        )
        keep.append(not dominated)
    out = f.loc[keep].copy()
    out = out.drop_duplicates(
        subset=["expected_high_value_mass_reached", "expected_utility"]
    )
    return out.sort_values("expected_high_value_mass_reached").reset_index(drop=True)


def utility_retention_summary(frontier: pd.DataFrame) -> pd.DataFrame:
    best_utility = float(frontier["expected_utility"].max())
    if best_utility <= 0:
        raise RuntimeError("Utility-retention operating points require positive economic utility")
    rows = []
    for loss in UTILITY_LOSS_LEVELS:
        floor = best_utility * (1.0 - loss)
        feasible = frontier[frontier["expected_utility"] >= floor - 1e-7].copy()
        if feasible.empty:
            raise RuntimeError(f"No epsilon solution satisfies utility-loss level {loss}")
        chosen = feasible.sort_values(
            ["expected_high_value_mass_reached", "expected_utility", "epsilon_fraction_of_max_strategic"],
            ascending=[False, False, True],
        ).iloc[0]
        row = chosen.to_dict()
        row.update({
            "allowed_expected_utility_loss_fraction": float(loss),
            "expected_utility_floor": float(floor),
            "actual_expected_utility_loss_fraction": float(
                1.0 - float(chosen["expected_utility"]) / best_utility
            ),
        })
        rows.append(row)
    cols_first = [
        "allowed_expected_utility_loss_fraction",
        "expected_utility_floor",
        "actual_expected_utility_loss_fraction",
        "epsilon_fraction_of_max_strategic",
        "expected_utility",
        "expected_utility_retained_fraction",
        "expected_high_value_mass_reached",
        "expected_high_value_reach_fraction",
        "fraction_of_budget_constrained_max_strategic",
    ]
    df = pd.DataFrame(rows)
    return df[cols_first + [c for c in df.columns if c not in cols_first]]


def rerun_primary(sample, cfg):
    """Re-run frozen Experiment A policies inside a bootstrap resample."""
    s = sample.copy().reset_index(drop=True)
    s["cltv_percentile"] = kgdi.percentile_rank(s["CLTV"].to_numpy(float))
    s["annual_margin_value"] = (
        s["Monthly Charges"].to_numpy(float)
        * cfg.value_horizon_months
        * cfg.gross_margin_rate
    )
    budget = len(s) * cfg.offer_cost * cfg.budget_fraction
    actions = {
        "Probability_Threshold": kgdi.policy_probability_threshold(s, cfg, budget),
        "Cost_Aware": kgdi.policy_cost_aware(s, cfg, budget),
        "Uncertainty_Review": kgdi.policy_uncertainty_review(s, cfg, budget),
        "Proposed_KGDI": kgdi.policy_kgdi(s, cfg, budget),
    }
    return {
        name: kgdi.evaluate_policy(s, a, cfg, planning_budget=budget)
        for name, a in actions.items()
    }


_BOOT_D = None
_BOOT_CFG = None


def _bootstrap_init(d, cfg):
    global _BOOT_D, _BOOT_CFG
    _BOOT_D, _BOOT_CFG = d, cfg


def _bootstrap_once(sample_seed):
    rng = np.random.default_rng(int(sample_seed))
    n = len(_BOOT_D)
    sample = _BOOT_D.iloc[rng.integers(0, n, size=n)].reset_index(drop=True)
    metrics = rerun_primary(sample, _BOOT_CFG)
    kgdi_net = metrics["Proposed_KGDI"]["net_benefit_proxy"]
    return (
        kgdi_net - metrics["Probability_Threshold"]["net_benefit_proxy"],
        kgdi_net - metrics["Cost_Aware"]["net_benefit_proxy"],
        kgdi_net - metrics["Uncertainty_Review"]["net_benefit_proxy"],
    )


def bootstrap_primary(d, cfg, n_boot, seed, workers):
    observed = rerun_primary(d, cfg)
    baselines = ["Probability_Threshold", "Cost_Aware", "Uncertainty_Review"]
    rng = np.random.default_rng(seed)
    sample_seeds = rng.integers(
        0, np.iinfo(np.uint32).max, size=n_boot, dtype=np.uint32
    ).tolist()
    workers = min(int(workers), int(n_boot))
    print(f"Running {n_boot} paired Experiment A policy-rerun bootstrap resamples...")
    if workers == 1:
        _bootstrap_init(d, cfg)
        results = [_bootstrap_once(s) for s in sample_seeds]
    else:
        methods = mp.get_all_start_methods()
        method = "fork" if "fork" in methods else methods[0]
        ctx = mp.get_context(method)
        chunksize = max(1, n_boot // (workers * 8))
        with ctx.Pool(
            processes=workers,
            initializer=_bootstrap_init,
            initargs=(d, cfg),
        ) as pool:
            results = pool.map(_bootstrap_once, sample_seeds, chunksize=chunksize)
    matrix = np.asarray(results, dtype=float)
    rows = []
    for j, baseline in enumerate(baselines):
        delta = matrix[:, j]
        rows.append({
            "comparison": f"Proposed_KGDI - {baseline}",
            "observed_delta": float(
                observed["Proposed_KGDI"]["net_benefit_proxy"]
                - observed[baseline]["net_benefit_proxy"]
            ),
            "bootstrap_mean_delta": float(delta.mean()),
            "ci_low": float(np.percentile(delta, 2.5)),
            "ci_high": float(np.percentile(delta, 97.5)),
            "bootstrap_proportion_delta_gt_0": float(np.mean(delta > 0)),
            "bootstrap_samples": int(n_boot),
        })
    return pd.DataFrame(rows)


def save_customer_actions(d, action_map, path: Path):
    base = d[[
        "CustomerID", kgdi.TARGET, "churn_probability", "predictive_entropy",
        "model_disagreement", "CLTV", "cltv_percentile", "annual_margin_value",
    ]].copy()
    action_cols = {
        f"action_epsilon_{eps:.2f}".replace(".", "p"): actions
        for eps, actions in sorted(action_map.items())
    }
    pd.concat([base, pd.DataFrame(action_cols, index=base.index)], axis=1).to_csv(
        path, index=False
    )


def expected_strategic_fraction(d, actions, cfg) -> float:
    high_value = d["cltv_percentile"].to_numpy(float) >= cfg.high_value_quantile
    mass = d["churn_probability"].to_numpy(float) * high_value
    denom = float(mass.sum())
    return float(mass[actions != "No Action"].sum() / denom) if denom > 0 else np.nan


def make_figures(d, baselines, nondom, utility_summary, action_map, output_dir, cfg, max_utility):
    # Locked 2% utility-loss operating point for the decision map.
    op2 = utility_summary.loc[
        np.isclose(utility_summary["allowed_expected_utility_loss_fraction"], 0.02)
    ].iloc[0]
    eps2 = float(op2["epsilon_fraction_of_max_strategic"])
    actions = action_map[eps2]
    markers = {"No Action": "o", "Retention Offer": "^", "Human Review": "s"}
    sizes = 10.0 + 45.0 * d["predictive_entropy"].to_numpy(float)
    plt.figure(figsize=(8.6, 6.2))
    for action, marker in markers.items():
        mask = actions == action
        if np.any(mask):
            plt.scatter(
                d.loc[mask, "churn_probability"],
                d.loc[mask, "cltv_percentile"],
                s=sizes[mask], alpha=0.42, marker=marker, label=action,
            )
    plt.xlabel("Calibrated ensemble churn probability")
    plt.ylabel("CLTV percentile")
    plt.title(f"Experiment B Decision Map: Locked 2% Utility-Loss Rule (ε={eps2:.2f})")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_B0_knowledge_decision_map.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.2, 5.9))
    plt.plot(
        100.0 * nondom["expected_high_value_reach_fraction"],
        100.0 * nondom["expected_utility"] / max_utility,
        marker="o", label="Experiment B non-dominated frontier",
    )
    mapping = {
        "Probability_Threshold": "action_Probability_Threshold",
        "Cost_Aware": "action_Cost_Aware",
        "Uncertainty_Review": "action_Uncertainty_Review",
        "Proposed_KGDI": "action_Proposed_KGDI",
    }
    for strategy, col in mapping.items():
        a = d[col].to_numpy(dtype=object)
        x = 100.0 * expected_strategic_fraction(d, a, cfg)
        utility = float(
            baselines.loc[baselines["strategy"] == strategy, "expected_utility"].iloc[0]
        )
        y = 100.0 * utility / max_utility
        plt.scatter([x], [y], marker="x", s=65, label=strategy)
    plt.xlabel("Expected high-value churn-risk mass engaged (%)")
    plt.ylabel("Expected economic utility retained (%)")
    plt.title("Experiment B: Expected Economic–Strategic Pareto Frontier")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_B1_pareto_frontier.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8.2, 5.9))
    plt.plot(
        100.0 * nondom["high_value_churn_reach_rate"],
        nondom["net_benefit_proxy"], marker="o",
        label="Experiment B non-dominated frontier",
    )
    for _, row in baselines.iterrows():
        plt.scatter(
            [100.0 * row["high_value_churn_reach_rate"]],
            [row["net_benefit_proxy"]], marker="x", s=65, label=row["strategy"],
        )
    plt.xlabel("Ex-post high-value churn reach (%)")
    plt.ylabel("Outcome-anchored simulated net-benefit proxy ($)")
    plt.title("Experiment B Ex-Post Tradeoff (Outcomes Not Used in Optimization)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "FIGURE_B2_realized_tradeoff.png", dpi=300)
    plt.close()


def build_manifest(
    output_dir, source, exp_dir, run_config, verification, eps_grid, cfg, args,
    solver_info, candidates, nondom, utility_summary,
):
    files = {
        p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(output_dir.iterdir())
        if p.is_file() and p.name not in {"EXPERIMENT_B_manifest.json", "EXPERIMENT_B_status.json"}
    }
    counts = candidates.groupby("row_index")["action"].nunique()
    return {
        "study": "Experiment B - locked post-primary multi-objective KGDI secondary analysis",
        "protocol_version": "v2",
        "source_experiment_a_archive": str(source),
        "source_experiment_a_directory": str(exp_dir),
        "source_experiment_a_git_commit": run_config.get("git_commit_sha"),
        "source_main_py_sha256": run_config.get("main_py_sha256"),
        "experiment_b_git_commit": git_value("rev-parse", "HEAD"),
        "experiment_b_script_sha256": sha256_file(Path(__file__).resolve()),
        "verification": verification,
        "python_version": sys.version,
        "platform": platform.platform(),
        "scipy_version": scipy.__version__,
        "epsilon_grid_fraction_of_budget_constrained_max_strategic": eps_grid,
        "utility_loss_levels": UTILITY_LOSS_LEVELS,
        "business_parameters_frozen_from_experiment_a": asdict(cfg),
        "bootstrap_samples_primary_comparisons": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_workers": int(args.bootstrap_workers),
        "candidate_actions": {
            "customer_action_candidates": int(len(candidates)),
            "customers_with_any_candidate": int(candidates["row_index"].nunique()),
            "customers_with_multiple_feasible_actions": int((counts > 1).sum()),
        },
        "non_dominated_points": int(len(nondom)),
        "utility_retention_operating_points": utility_summary[
            [
                "allowed_expected_utility_loss_fraction",
                "epsilon_fraction_of_max_strategic",
                "expected_utility",
                "expected_high_value_reach_fraction",
            ]
        ].to_dict(orient="records"),
        "optimization_disclosure": (
            "MILP uses only frozen OOF probability, predictive entropy, CLTV percentile, "
            "and frozen business assumptions. Observed Churn Value is excluded from "
            "candidate construction, objectives, constraints, and operating-point selection; "
            "it is used only for ex-post descriptive evaluation."
        ),
        "strategic_objective_definition": (
            "Expected high-value churn-risk mass = sum(p_i) over selected customers at or "
            "above the frozen CLTV high-value quantile. Epsilon is a fraction of its "
            "budget-constrained maximum."
        ),
        "solver": solver_info,
        "files": files,
    }


def write_status(output_dir: Path, status: str, **extra) -> None:
    payload = {
        "status": status,
        "git_commit_sha": git_value("rev-parse", "HEAD"),
        "experiment_b_script_sha256": sha256_file(Path(__file__).resolve()),
        **extra,
    }
    atomic_json(output_dir / "EXPERIMENT_B_status.json", payload)


def parse_args():
    p = argparse.ArgumentParser(
        description="Locked Experiment B multi-objective KGDI epsilon-constraint analysis",
        allow_abbrev=False,
    )
    p.add_argument("--experiment-a", required=True, help="Immutable Experiment A .tar.gz archive")
    p.add_argument("--output", default="results/experiment_b_final")
    p.add_argument("--bootstrap-workers", type=int, default=8)
    p.add_argument("--solver-time-limit", type=float, default=120.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--allow-dirty", action="store_true",
        help="Development only. Definitive run should use a clean Git working tree.",
    )
    p.add_argument(
        "--allow-version-drift", action="store_true",
        help="Development only. Definitive run should reproduce Experiment A package versions.",
    )
    return p.parse_args()


def validate_args(args) -> None:
    if args.bootstrap_workers < 1:
        raise ValueError("--bootstrap-workers must be >= 1")
    if args.solver_time_limit <= 0:
        raise ValueError("--solver-time-limit must be > 0")
    if not args.dry_run and (args.allow_dirty or args.allow_version_drift):
        raise RuntimeError(
            "Definitive Experiment B forbids --allow-dirty and --allow-version-drift."
        )
    if not args.allow_dirty and not git_is_clean():
        raise RuntimeError(
            "Git working tree is not clean. Commit/push intended source files before the definitive run."
        )


def main_cli() -> int:
    args = parse_args()
    validate_args(args)
    eps_grid = DEFAULT_EPSILON_GRID.copy()
    source = Path(args.experiment_a)
    output_dir = Path(args.output).expanduser().resolve()

    source, exp_dir, temp, run_config, cfg, d, verification = load_source(
        source, args.allow_version_drift
    )
    try:
        baselines = reconstruct_baselines(d, exp_dir, cfg)
        candidates = build_candidates(d, cfg)
        counts = candidates.groupby("row_index")["action"].nunique()
        print("Frozen Experiment A integrity/reconstruction: PASS")
        print(
            f"Candidate actions: {len(candidates)} | "
            f"customers: {candidates['row_index'].nunique()} | "
            f"customers with both feasible actions: {(counts > 1).sum()}"
        )

        # A dry run validates the source, environment, candidate set, and MILP endpoints.
        if args.dry_run:
            budget = float(len(d) * cfg.offer_cost * cfg.budget_fraction)
            utility_mask, utility_result = solve(
                candidates, len(d), budget, "utility", 0.0,
                args.solver_time_limit, MIP_REL_GAP,
            )
            validate_selection(candidates, utility_mask, budget, 0.0)
            strategic_mask, strategic_result = solve(
                candidates, len(d), budget, "strategic", 0.0,
                args.solver_time_limit, MIP_REL_GAP,
            )
            validate_selection(candidates, strategic_mask, budget, 0.0)
            print(
                "DRY RUN OK: archive, manifest, frozen main.py, environment, baseline "
                "reconstruction, candidate set, exclusivity, and MILP endpoints verified."
            )
            print("Economic endpoint:", solver_metadata(utility_result))
            print("Strategic endpoint:", solver_metadata(strategic_result))
            return 0

        ensure_fresh_output(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_status(output_dir, "RUNNING")
        try:
            frontier, action_map, solver_info = run_frontier(
                d, candidates, cfg, eps_grid,
                args.solver_time_limit, MIP_REL_GAP,
            )
            nondom = nondominated(frontier)
            utility_summary = utility_retention_summary(frontier)
            bootstrap = bootstrap_primary(
                d, cfg, BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED, args.bootstrap_workers
            )

            baselines.to_csv(
                output_dir / "EXPERIMENT_A_reconstructed_baselines.csv", index=False
            )
            bootstrap.to_csv(
                output_dir / "EXPERIMENT_A_bootstrap_all_comparisons.csv", index=False
            )
            candidates.to_csv(
                output_dir / "EXPERIMENT_B_candidate_table.csv", index=False
            )
            frontier.to_csv(
                output_dir / "EXPERIMENT_B_frontier_all_epsilon.csv", index=False
            )
            nondom.to_csv(
                output_dir / "EXPERIMENT_B_pareto_nondominated.csv", index=False
            )
            utility_summary.to_csv(
                output_dir / "EXPERIMENT_B_utility_retention_summary.csv", index=False
            )
            save_customer_actions(
                d, action_map, output_dir / "EXPERIMENT_B_customer_actions.csv"
            )
            make_figures(
                d, baselines, nondom, utility_summary, action_map,
                output_dir, cfg, solver_info["max_expected_utility"],
            )

            manifest = build_manifest(
                output_dir, source, exp_dir, run_config, verification,
                eps_grid, cfg, args, solver_info, candidates, nondom, utility_summary,
            )
            atomic_json(output_dir / "EXPERIMENT_B_manifest.json", manifest)
            write_status(
                output_dir, "COMPLETED",
                epsilon_points=len(frontier),
                non_dominated_points=len(nondom),
                bootstrap_samples=BOOTSTRAP_SAMPLES,
            )

            print("=" * 88)
            print("EXPERIMENT B COMPLETED SUCCESSFULLY")
            print("=" * 88)
            print(
                f"Candidate actions: {len(candidates)} | "
                f"customers: {candidates['row_index'].nunique()} | "
                f"Budget: ${solver_info['budget']:,.2f}"
            )
            print(
                f"Epsilon points: {len(frontier)} | "
                f"Non-dominated points: {len(nondom)}"
            )
            print("\nLOCKED UTILITY-RETENTION OPERATING POINTS")
            print(
                utility_summary[[
                    "allowed_expected_utility_loss_fraction",
                    "epsilon_fraction_of_max_strategic",
                    "expected_utility_retained_fraction",
                    "expected_high_value_reach_fraction",
                    "net_benefit_proxy",
                    "high_value_churn_reach_rate",
                ]].to_string(index=False)
            )
            print("\nEXPERIMENT A PAIRED POLICY-RERUN BOOTSTRAP COMPARISONS")
            print(bootstrap.to_string(index=False))
            print(f"\nResults saved to: {output_dir}")
            return 0
        except Exception as exc:
            write_status(output_dir, "FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
    finally:
        temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main_cli())
