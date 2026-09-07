#!/usr/bin/env python3
"""
RunPod-safe execution wrapper for the frozen KGDI research pipeline.

This wrapper does NOT modify the scientific methodology in main.py. It adds:
- frozen main.py SHA-256 verification;
- clean Git working-tree preflight;
- durable run.log capture;
- RUNNING / COMPLETED / FAILED run_status.json;
- pre-run reproducibility metadata;
- post-run output integrity validation;
- SHA-256 results_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_MAIN_SHA256 = (
    "5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d"
)

REQUIRED_OUTPUTS = [
    "data_audit.json",
    "SUPPLEMENT_column_roles.csv",
    "SUPPLEMENT_predictive_metrics_by_repeat.csv",
    "SUPPLEMENT_decision_metrics_by_repeat.csv",
    "SUPPLEMENT_repeated_cv_stability.csv",
    "SUPPLEMENT_final_policy_metrics.csv",
    "SUPPLEMENT_policy_distinctness.csv",
    "SUPPLEMENT_bootstrap_KGDI_vs_uncertainty.json",
    "SUPPLEMENT_ablation_study.csv",
    "SUPPLEMENT_subgroup_audit.csv",
    "SUPPLEMENT_sensitivity_full.csv",
    "SUPPLEMENT_customer_level_oof_decisions.csv",
    "TABLE_1_predictive_performance.csv",
    "TABLE_2_decision_performance.csv",
    "FIGURE_1_KGDI_decision_map.png",
    "FIGURE_2_KGDI_sensitivity.png",
    "run_config.json",
]

PACKAGE_NAMES = [
    "pandas",
    "numpy",
    "openpyxl",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "matplotlib",
    "scipy",
]

ALLOWED_ACTIONS = {"No Action", "Retention Offer", "Human Review"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            ["git", *args],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def write_status(output_dir: Path, status: str, **extra) -> None:
    payload = {
        "status": status,
        "updated_at_utc": utc_now(),
        "git_commit_sha": git_value("rev-parse", "HEAD"),
        "main_py_sha256": (
            sha256_file(Path("main.py").resolve())
            if Path("main.py").is_file()
            else None
        ),
    }
    payload.update(extra)
    atomic_json(output_dir / "run_status.json", payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RunPod-safe wrapper around frozen KGDI main.py",
        allow_abbrev=False,
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/paper")
    parser.add_argument("--main", default="main.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty Git tree. Not recommended for the definitive paper run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform preflight checks only; do not execute main.py.",
    )
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def preflight(args, passthrough):
    main_path = Path(args.main).resolve()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not main_path.is_file():
        raise FileNotFoundError(f"main.py not found: {main_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    main_sha = sha256_file(main_path)
    if main_sha != EXPECTED_MAIN_SHA256:
        raise RuntimeError(
            "Frozen scientific code verification failed. "
            f"main.py SHA256={main_sha}, expected={EXPECTED_MAIN_SHA256}"
        )

    dirty = git_value("status", "--porcelain") or ""
    if dirty and not args.allow_dirty:
        raise RuntimeError(
            "Git working tree is not clean. Commit/stash changes before the definitive "
            "paper run. Use --allow-dirty only for a non-definitive test."
        )

    subprocess.run(
        [args.python, "-m", "py_compile", str(main_path)],
        check=True,
    )

    command = [
        args.python,
        str(main_path),
        "--data",
        str(data_path),
        "--output",
        str(output_dir),
        *passthrough,
    ]

    metadata = {
        "wrapper_status": "PRE_RUN",
        "created_at_utc": utc_now(),
        "command": command,
        "main_path": str(main_path),
        "main_py_sha256": main_sha,
        "expected_main_py_sha256": EXPECTED_MAIN_SHA256,
        "dataset_path": str(data_path),
        "dataset_sha256": sha256_file(data_path),
        "git_commit_sha": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "git_working_tree_clean": not bool(dirty),
        "python_executable": args.python,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count_visible": os.cpu_count(),
        "package_versions": package_versions(),
    }
    atomic_json(output_dir / "run_config.json", metadata)
    return output_dir, command, metadata


def run_with_tee(command: list[str], log_path: Path) -> int:
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        header = f"\n===== RUN START {utc_now()} =====\n"
        sys.stdout.write(header)
        log.write(header)
        log.write("COMMAND: " + " ".join(command) + "\n")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None

        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)

        code = process.wait()
        footer = f"===== RUN END {utc_now()} exit={code} =====\n"
        sys.stdout.write(footer)
        log.write(footer)
        return code


def finite_float(value: str, label: str) -> float:
    x = float(value)
    if not math.isfinite(x):
        raise RuntimeError(f"{label} is not finite: {value}")
    return x


def validate_customer_outputs(output_dir: Path) -> dict:
    path = output_dir / "SUPPLEMENT_customer_level_oof_decisions.csv"
    rows = 0

    required = {
        "churn_probability",
        "predictive_entropy",
        "model_disagreement",
        "action_Probability_Threshold",
        "action_Cost_Aware",
        "action_Uncertainty_Review",
        "action_Proposed_KGDI",
    }

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"Customer-level output missing columns: {sorted(missing)}"
            )

        for row in reader:
            rows += 1

            p = finite_float(row["churn_probability"], "churn_probability")
            entropy = finite_float(
                row["predictive_entropy"],
                "predictive_entropy",
            )
            disagreement = finite_float(
                row["model_disagreement"],
                "model_disagreement",
            )

            if not 0.0 <= p <= 1.0:
                raise RuntimeError(
                    f"churn_probability outside [0,1]: {p}"
                )

            if not 0.0 <= entropy <= 1.0:
                raise RuntimeError(
                    f"predictive_entropy outside [0,1]: {entropy}"
                )

            if disagreement < 0.0:
                raise RuntimeError(
                    f"model_disagreement cannot be negative: {disagreement}"
                )

            for column in (
                "action_Probability_Threshold",
                "action_Cost_Aware",
                "action_Uncertainty_Review",
                "action_Proposed_KGDI",
            ):
                if row[column] not in ALLOWED_ACTIONS:
                    raise RuntimeError(
                        f"Unexpected action in {column}: {row[column]!r}"
                    )

    if rows == 0:
        raise RuntimeError("Customer-level output is empty")

    return {
        "customer_rows_validated": rows,
    }


def validate_budget_outputs(output_dir: Path) -> dict:
    path = output_dir / "SUPPLEMENT_final_policy_metrics.csv"
    checked = 0

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "strategy",
            "expected_selected_cost",
            "planning_budget",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                f"Final policy metrics missing columns: {sorted(missing)}"
            )

        for row in reader:
            checked += 1

            cost = finite_float(
                row["expected_selected_cost"],
                "expected_selected_cost",
            )

            budget = finite_float(
                row["planning_budget"],
                "planning_budget",
            )

            tolerance = max(
                1e-6,
                abs(budget) * 1e-9,
            )

            if cost > budget + tolerance:
                raise RuntimeError(
                    f"Budget violation for {row['strategy']}: "
                    f"cost={cost}, budget={budget}"
                )

    if checked == 0:
        raise RuntimeError(
            "Final policy metrics are empty"
        )

    return {
        "policies_budget_validated": checked,
    }


def build_results_manifest(
    output_dir: Path,
    validations: dict,
) -> dict:
    missing = [
        name
        for name in REQUIRED_OUTPUTS
        if not (output_dir / name).is_file()
    ]

    if missing:
        raise RuntimeError(
            f"Missing required result files: {missing}"
        )

    files = {}

    for name in REQUIRED_OUTPUTS:
        path = output_dir / name

        files[name] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }

    return {
        "created_at_utc": utc_now(),
        "git_commit_sha": git_value(
            "rev-parse",
            "HEAD",
        ),
        "main_py_sha256": EXPECTED_MAIN_SHA256,
        "validations": validations,
        "files": files,
    }


def main() -> int:
    args, passthrough = parse_args()

    output_dir = Path(
        args.output
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        output_dir, command, pre_metadata = preflight(
            args,
            passthrough,
        )

        write_status(
            output_dir,
            "DRY_RUN"
            if args.dry_run
            else "RUNNING",
            main_py_sha256=EXPECTED_MAIN_SHA256,
            dataset_sha256=pre_metadata[
                "dataset_sha256"
            ],
        )

        if args.dry_run:
            write_status(
                output_dir,
                "DRY_RUN_OK",
                main_py_sha256=EXPECTED_MAIN_SHA256,
                dataset_sha256=pre_metadata[
                    "dataset_sha256"
                ],
            )

            print(
                "DRY RUN OK: frozen main.py, dataset, "
                "Git state, and Python syntax verified."
            )

            return 0

        code = run_with_tee(
            command,
            output_dir / "run.log",
        )

        if code != 0:
            raise RuntimeError(
                f"main.py exited with code {code}"
            )

        validations = {}

        validations.update(
            validate_customer_outputs(
                output_dir
            )
        )

        validations.update(
            validate_budget_outputs(
                output_dir
            )
        )

        run_config_path = (
            output_dir
            / "run_config.json"
        )

        with run_config_path.open(
            encoding="utf-8"
        ) as f:
            final_config = json.load(f)

        final_config[
            "runpod_wrapper_preflight"
        ] = pre_metadata

        final_config[
            "runpod_wrapper_validations"
        ] = validations

        final_config[
            "runpod_wrapper_completed_at_utc"
        ] = utc_now()

        atomic_json(
            run_config_path,
            final_config,
        )

        manifest = build_results_manifest(
            output_dir,
            validations,
        )

        atomic_json(
            output_dir
            / "results_manifest.json",
            manifest,
        )

        write_status(
            output_dir,
            "COMPLETED",
            main_py_sha256=EXPECTED_MAIN_SHA256,
            results_manifest_sha256=sha256_file(
                output_dir
                / "results_manifest.json"
            ),
        )

        print(
            "RUNPOD WRAPPER COMPLETED SUCCESSFULLY"
        )

        print(
            f"Results: {output_dir}"
        )

        return 0

    except Exception as exc:
        try:
            write_status(
                output_dir,
                "FAILED",
                error_type=type(exc).__name__,
                error=str(exc),
            )

        except Exception:
            pass

        with (
            output_dir / "run.log"
        ).open(
            "a",
            encoding="utf-8",
        ) as log:
            log.write(
                f"\nWRAPPER FAILURE {utc_now()}: "
                f"{type(exc).__name__}: {exc}\n"
            )

        raise


if __name__ == "__main__":
    raise SystemExit(main())
