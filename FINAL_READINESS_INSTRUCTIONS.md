# Final RunPod scientific safeguards

This package adds **read-only** input and result validation and tests. It does not alter the optimizer, model fits, or research assumptions.

## Install in your existing repository

Unzip into repository root; then run `python -m pytest -q tests/test_final_readiness.py`.

## Before running RunPod

Make sure the four input files are under `data/` and the tracked checksum manifest is at `results/experiment_a_frozen/experiment_manifest.json`.

```
python final_readiness.py pre \
  --ibm-oof data/SUPPLEMENT_customer_level_oof_decisions.csv \
  --run-config data/run_config.json \
  --iranian 'data/Customer Churn.csv' \
  --cell2cell data/cell2celltrain.csv \
  --manifest results/experiment_a_frozen/experiment_manifest.json
```

The Iranian CSV SHA-256 is recorded as evidence, not equated to the archived ZIP hash in the manifest. Its columns and row count are verified.

## After full run, before using any numbers in manuscript

```
python final_readiness.py post --output results/reviewer_revision_final
```

## Research guardrails

- The IBM baseline was regenerated under a specific macOS environment, so do not claim it is byte-identical to older historical results.
- In that regenerated Experiment A, KGDI and Uncertainty Review chose identical actions and had zero observed incremental economic benefit. Preserve this limitation in paper and response.
- These tests prove structural readiness and basic numeric feasibility, not causal lift or scientific superiority.
- Do not commit raw data, customer predictions, or full experimental outputs to public GitHub.
- Save `pip freeze > requirements-runpod-exact.txt` on the target RunPod **after** the installation, for the actual execution environment. Different platforms can resolve different dependency versions.

## Recheck 2026-10-31: corrected package
- Post-run checks now require the expected dataset score counts (IBM 7,043; Iranian 3,150; Cell2Cell 50,891), unique/non-null customer IDs, a complete epsilon grid, consistent predictive summary counts, decision-summary coverage, and SHA-256 provenance entries in the run manifest.
- This does **not** verify that the recorded run-manifest hashes equal the source files on RunPod. Re-run the `pre` validation against the actual files on the pod before execution.
- The 5 input-validator tests were executed on the supplied data and passed. Optimizer edge-case tests have **not** been executed against the full repository in this environment; run both suites locally before committing.
