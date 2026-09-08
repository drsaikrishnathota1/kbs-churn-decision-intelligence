# Knowledge-Guided Decision Intelligence for Customer Churn

Research code for:

**Knowledge-Guided Decision Intelligence for Cost-Aware Customer Churn Intervention Under Predictive Uncertainty**

## Repository structure

- `main.py` — frozen Experiment A predictive + decision-intelligence pipeline.
- `runpod_runner.py` — integrity/reproducibility wrapper for Experiment A.
- `experiment_b.py` — locked post-primary Experiment B multi-objective MILP analysis.
- `experiment_c.py` — knowledge-gated decision layer on frozen Experiment A OOF (no refit).
- `letter_outputs.py` — camera-ready Tables 1–2 and Figs. 1–2 from frozen OOF.
- `review_analyses.py` — component ablation, residual bins, disagreement overlap, scenario sensitivity.
- `EXPERIMENT_B_PROTOCOL.md` — frozen Experiment B protocol and reporting rules.
- `requirements.txt` — portable minimum dependency versions.
- `requirements-runpod.txt` — exact top-level package versions used by the definitive Experiment A RunPod environment and required for the definitive Experiment B run.

## Frozen Experiment A provenance

- Git commit: `157fe440d97ad41178b2dc1c7183e1c8195ec6ba`
- `main.py` SHA-256: `5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d`
- Definitive archive: `KGDI_paper_final_157fe440.tar.gz`
- Archive SHA-256: `362b250a5fa5d4ba8e9ac8177d9aa6253e26933321fcd5b8a26ff474c39b3c2a`
- IBM dataset SHA-256: `9b7d90f9e4c0f9b607126640e4279f0afd4120e7b3485614770753c9f393aaae`

Experiment A remains immutable. Experiment B consumes the frozen Experiment A out-of-fold customer-level outputs and does not refit predictive models.

## Experiment A

The predictive pipeline uses Logistic Regression, Random Forest, XGBoost, LightGBM, and an equal-weight calibrated ensemble with repeated leakage-safe out-of-fold evaluation. Predictive entropy is the primary uncertainty quantity, while model/repeat disagreement is retained separately as a diagnostic.

The decision layer evaluates Probability Threshold, Cost-Aware, Uncertainty Review, and Proposed KGDI policies under a common expected-cost budget. IBM CLTV is treated only as a dimensionless strategic-priority index.

Example definitive Experiment A command:

```bash
python runpod_runner.py \
  --data data/Telco_customer_churn.xlsx.zip \
  --output results/paper_final \
  --mode paper \
  --threads 16
```

## Experiment B

Experiment B is a locked post-primary secondary analysis. It solves a binary mixed-integer linear program over knowledge-eligible direct-offer and human-review actions. The optimizer maximizes expected economic utility while progressively constraining expected high-value churn-risk engagement over a fixed 101-point epsilon grid.

Observed `Churn Value` is excluded from Experiment B candidate construction, feasibility rules, objectives, constraints, and operating-point selection. It is used only for ex-post descriptive evaluation.

### Exact environment

Use Python 3.11 and install:

```bash
python -m pip install --upgrade pip
pip install -r requirements-runpod.txt
```

### Preflight

Upload the immutable Experiment A archive, then run:

```bash
python experiment_b.py \
  --experiment-a KGDI_paper_final_157fe440.tar.gz \
  --dry-run
```

Expected final message:

```text
DRY RUN OK: archive, manifest, frozen main.py, environment, baseline reconstruction, candidate set, exclusivity, and MILP endpoints verified.
```

### Definitive Experiment B run

Use a clean Git working tree and a fresh output directory:

```bash
python experiment_b.py \
  --experiment-a KGDI_paper_final_157fe440.tar.gz \
  --output results/experiment_b_final \
  --bootstrap-workers 8
```

The definitive script locks:

- epsilon grid: `0.00` to `1.00` by `0.01`;
- bootstrap samples: `1000`;
- bootstrap seed: `20260907`;
- relative MIP gap: `1e-6`;
- utility-retention operating rules: 0%, 1%, 2%, and 5% allowed expected-utility loss.

## Methodological disclosure

Customer characteristics, churn outcomes, charges, and CLTV come from the IBM Telco Customer Churn dataset. Retention cost, intervention success probability, human-review cost/accuracy, gross-margin assumption, value horizon, and management budget are simulated scenario assumptions.

Reported outcome-anchored net benefit is a **simulated proxy**, not causal realized revenue or causal retention impact.

## Data and generated outputs

The IBM dataset and generated results are intentionally excluded from Git. Do not commit the raw workbook or definitive result archives to the public repository.
