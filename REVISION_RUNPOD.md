# RunPod execution guide — Reviewer 2 + Reviewer 3 revision

## 1. Add the patch to the repository
Place these files at the repository root:

- `REVISION_PROTOCOL.md`
- `REVISION_RUNPOD.md`
- `review_revision/`
- `tests/test_revision_invariants.py`

Do not edit the frozen Experiment A `main.py`.

## 2. Data files
Prepare these inputs on RunPod:

1. Frozen Experiment A output:
   - `SUPPLEMENT_customer_level_oof_decisions.csv`
   - `run_config.json`
2. UCI Iranian Churn:
   - `Customer Churn.csv` from UCI dataset 563 (DOI 10.24432/C5JW3Z).
3. Cell2Cell:
   - labeled training file, commonly `cell2celltrain.csv`.

Do not commit raw datasets to GitHub.

## 3. Environment
Use the repository's existing RunPod environment first:

```bash
python -m pip install --upgrade pip
pip install -r requirements-runpod.txt
pip install pytest
```

## 4. Static checks

```bash
python -m py_compile review_revision/*.py
pytest -q tests/test_revision_invariants.py
```

## 5. Development smoke run
This deliberately uses reduced folds/bootstrap and a three-point epsilon grid. Do not report these numbers in the paper.

```bash
python -m review_revision.run_revision \
  --ibm-oof /workspace/data/SUPPLEMENT_customer_level_oof_decisions.csv \
  --run-config /workspace/data/run_config.json \
  --iranian '/workspace/data/Customer Churn.csv' \
  --cell2cell /workspace/data/cell2celltrain.csv \
  --output results/reviewer_revision_smoke \
  --threads 16 \
  --quick
```

Inspect for exceptions and verify all output files exist.

## 6. Definitive run
Only after the protocol and code are committed and the smoke run passes:

```bash
python -m review_revision.run_revision \
  --ibm-oof /workspace/data/SUPPLEMENT_customer_level_oof_decisions.csv \
  --run-config /workspace/data/run_config.json \
  --iranian '/workspace/data/Customer Churn.csv' \
  --cell2cell /workspace/data/cell2celltrain.csv \
  --output results/reviewer_revision_final \
  --threads 16 \
  --bootstrap 1000
```

The definitive run uses:
- 5 folds × 3 repeats for external predictive validation;
- epsilon 0.00–1.00 by 0.01;
- 1,000 paired IBM bootstrap resamples;
- seed 20261031;
- the same normalized decision scenario across IBM/Iranian/Cell2Cell.

## 7. Outputs to return for manuscript revision
Please download or zip:

```bash
cd results
zip -r reviewer_revision_final.zip reviewer_revision_final
```

The most important files are:

- `reviewer3_comparators.csv`
- `reviewer3_bootstrap.json`
- `cross_dataset_predictive_summary.csv`
- `cross_dataset_decision_summary.csv`
- each dataset's `epsilon_frontier.csv`
- each dataset's `operating_points.csv`
- each dataset's `knowledge_ablation.csv`
- `revision_manifest.json`

Do not rewrite the manuscript with numerical claims until the definitive outputs are inspected.
