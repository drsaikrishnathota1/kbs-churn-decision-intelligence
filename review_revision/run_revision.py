from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RevisionScenario, scenario_to_dict, sha256_file, write_json
from .datasets import load_ibm_oof, load_iranian, load_cell2cell
from .predict import run_oof
from .optimizer import frontier, operating_points, knowledge_ablation
from .reviewer3 import load_cfg, run_comparators, paired_bootstrap


def parse_args():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--ibm-oof", required=True, help="Frozen Experiment A SUPPLEMENT_customer_level_oof_decisions.csv")
    p.add_argument("--run-config", required=True, help="Frozen Experiment A run_config.json")
    p.add_argument("--iranian", required=True, help="UCI Iranian Churn CSV")
    p.add_argument("--cell2cell", required=True, help="Cell2Cell labeled training CSV")
    p.add_argument("--output", default="results/reviewer_revision_final")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="Development only: 2 folds x 1 repeat, 5 bootstrap, epsilon step .5")
    return p.parse_args()


def scores_from_bundle(bundle, p, entropy):
    return pd.DataFrame({
        "customer_id": np.arange(len(bundle.target)).astype(str),
        "target": bundle.target.to_numpy(int),
        "churn_probability": np.asarray(p, dtype=float),
        "predictive_entropy": np.asarray(entropy, dtype=float),
        "strategic_value": np.asarray(bundle.strategic_value, dtype=float),
        "economic_value": np.asarray(bundle.economic_value, dtype=float),
        "strategic_value_raw": bundle.strategic_value_raw.to_numpy(),
        "economic_value_raw": bundle.economic_value_raw.to_numpy(),
    })


def run_dataset(bundle, out, cfg, threads, predictive=True):
    out.mkdir(parents=True, exist_ok=True)
    if predictive:
        p, entropy, metrics = run_oof(
            bundle.features, bundle.target,
            folds=cfg.folds, repeats=cfg.repeats, seed=cfg.seed, threads=threads
        )
        scores = scores_from_bundle(bundle, p, entropy)
    else:
        scores = bundle.frame[[
            "customer_id", "target", "churn_probability", "predictive_entropy",
            "strategic_value", "economic_value", "strategic_value_raw", "economic_value_raw"
        ]].copy()
        metrics = pd.DataFrame([{
            "repeat": 0, "model": "Frozen Experiment A ensemble",
            "auroc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece": np.nan,
        }])

    metrics.to_csv(out / "predictive_metrics.csv", index=False)
    scores.to_csv(out / "customer_scores.csv", index=False)
    f, meta = frontier(scores, cfg)
    f.to_csv(out / "epsilon_frontier.csv", index=False)
    op = operating_points(f)
    op.to_csv(out / "operating_points.csv", index=False)
    ab = knowledge_ablation(scores, cfg)
    ab.to_csv(out / "knowledge_ablation.csv", index=False)
    write_json(out / "dataset_manifest.json", bundle.metadata)
    write_json(out / "run_manifest.json", {
        "dataset": bundle.name,
        "scenario": scenario_to_dict(cfg),
        "frontier_meta": meta,
        "python": sys.version,
        "platform": platform.platform(),
        "predictive_refit": bool(predictive),
    })
    return metrics, f, op, ab


def main():
    args = parse_args()
    cfg = RevisionScenario()
    if args.quick:
        cfg = RevisionScenario(folds=2, repeats=1, bootstrap=5, epsilon_step=0.50)
    else:
        cfg = RevisionScenario(bootstrap=args.bootstrap)

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ibm = load_ibm_oof(args.ibm_oof)
    iranian = load_iranian(args.iranian)
    cell = load_cell2cell(args.cell2cell)

    summaries_pred = []
    summaries_decision = []
    for bundle, predictive in [(ibm, False), (iranian, True), (cell, True)]:
        m, f, op, ab = run_dataset(bundle, out / bundle.name, cfg, args.threads, predictive=predictive)
        if predictive:
            ens = m[m.model == "Ensemble"].groupby("model").agg(
                auroc=("auroc", "mean"), pr_auc=("pr_auc", "mean"),
                brier=("brier", "mean"), ece=("ece", "mean")
            ).reset_index().iloc[0].to_dict()
        else:
            # Predictive performance is already reported by frozen Experiment A; avoid silently recomputing it here.
            ens = {"auroc": np.nan, "pr_auc": np.nan, "brier": np.nan, "ece": np.nan}
        summaries_pred.append({"dataset": bundle.name, "n": len(bundle.target), **ens})
        for _, r in op.iterrows():
            summaries_decision.append({"dataset": bundle.name, **r.to_dict()})

    # Reviewer 3 direct IBM comparators under the original Experiment A economic scenario.
    ibm_raw = pd.read_csv(args.ibm_oof)
    ibm_cfg = load_cfg(args.run_config)
    r3, r3meta, _ = run_comparators(ibm_raw, ibm_cfg)
    r3.to_csv(out / "reviewer3_comparators.csv", index=False)
    write_json(out / "reviewer3_comparator_definition.json", r3meta)
    boot_n = cfg.bootstrap
    r3boot = paired_bootstrap(ibm_raw, ibm_cfg, n_boot=boot_n, seed=cfg.seed)
    write_json(out / "reviewer3_bootstrap.json", r3boot)

    pd.DataFrame(summaries_pred).to_csv(out / "cross_dataset_predictive_summary.csv", index=False)
    pd.DataFrame(summaries_decision).to_csv(out / "cross_dataset_decision_summary.csv", index=False)
    write_json(out / "revision_manifest.json", {
        "protocol": "REVISION_PROTOCOL.md",
        "scenario": scenario_to_dict(cfg),
        "inputs": {
            "ibm_oof": {"path": str(Path(args.ibm_oof).name), "sha256": sha256_file(args.ibm_oof)},
            "run_config": {"path": str(Path(args.run_config).name), "sha256": sha256_file(args.run_config)},
            "iranian": {"path": str(Path(args.iranian).name), "sha256": sha256_file(args.iranian)},
            "cell2cell": {"path": str(Path(args.cell2cell).name), "sha256": sha256_file(args.cell2cell)},
        },
        "quick_development_run": bool(args.quick),
    })
    print(f"Revision outputs written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
