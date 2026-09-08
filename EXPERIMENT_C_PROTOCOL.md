# Experiment C Protocol

## Status
Prespecified upgrade of the **decision layer only**. Experiment A (`main.py`) remains immutable and is not overwritten.

## Why Experiment C exists
Experiment A showed that the original KGDI rule was not a distinct policy:

- KGDI and Uncertainty Review disagreed on 2 of 7,043 customers.
- Outcome-anchored net benefit of KGDI was **$59,247** versus **$65,209** for Cost-Aware.
- The 95% policy-rerun bootstrap CI for KGDI versus Uncertainty Review included zero.

Those results are retained as Experiment A. They are not retuned.

## Research question
Can a knowledge-graph decision layer improve **budget-matched** intervention by combining:

1. expected economic utility from calibrated OOF probabilities, and
2. IBM CLTV plus structural service-graph risk to reserve a minority review budget for high-value uncertain or knowledge-conflict cases?

## Frozen inputs
- Frozen Experiment A OOF table: `SUPPLEMENT_customer_level_oof_decisions.csv`
- Frozen business parameters from Experiment A `run_config.json`
- Predictive models are **not** refit

## Knowledge graph (leakage-safe)
Heterogeneous graph constructed from observed IBM fields only:

- Customer nodes
- Contract, internet-service, payment-method, and protection-service nodes
- Edges: `HAS_CONTRACT`, `USES_SERVICE`, `PAYS_VIA`, `LACKS_PROTECTION`

One-hop knowledge risk for each customer is the **training-fold** empirical churn rate of incident motifs, averaged out-of-fold with the same 5-fold stratified split seed. Labels from a test fold never enter that fold's motif rates.

IBM CLTV is a node attribute used as a dimensionless strategic-priority index, not as currency.

## Decision policy (prespecified)
Let `economic_allocation_fraction = 0.85`.

1. **Economic stage.** Spend 85% of the planning budget on `Retention Offer` using Experiment A Cost-Aware ranking (positive expected offer utility, `p >= minimum_churn_risk`).
2. **Knowledge-guided review stage.** Spend remaining budget on `Human Review` for customers who are still unselected and satisfy **all** of:
   - `p >= minimum_churn_risk`
   - CLTV percentile ≥ frozen high-value quantile (0.75)
   - expected review utility > 0
   - predictive entropy ≥ frozen moderate-uncertainty threshold **or** OOF knowledge residual `(k - p) > 0`
3. **Fill stage.** Any leftover budget returns to Cost-Aware offers.

The 85/15 split is chosen *a priori* as a majority economic allocation with a minority knowledge-review reserve. It is not selected by peeking at net benefit. The grid `{0.70, 0.75, 0.80, 0.85, 0.90, 1.00}` is reported in full.

## Primary comparators
Budget-matched:

- Probability Threshold (Experiment A)
- Cost-Aware (Experiment A; primary economic baseline)
- Uncertainty Review (Experiment A)
- Proposed KGDI from Experiment A
- Proposed KGDI-C (this experiment)

## Inference
1,000 paired customer bootstraps with policy rerouting and budget reallocation inside every resample. Report delta versus **Cost-Aware** and versus Uncertainty Review, with 95% percentile intervals and bootstrap P(Δ>0).

## Reporting rule
Report favorable and unfavorable metrics. Do not drop Cost-Aware if it wins on a subset of metrics. No post-hoc threshold search to force superiority.
