# Experiment B Protocol

## Status
Locked secondary analysis performed after the primary Experiment A result. The design is intentionally separated from Experiment A: `main.py` and the definitive Experiment A outputs remain immutable.

This protocol was refined during code review and development validation before the definitive Experiment B RunPod execution. It is therefore a post-primary secondary analysis, not a preregistered confirmatory experiment. The epsilon grid, feasible-action rules, optimization objectives, utility-retention operating points, bootstrap seed/sample count, business assumptions, and reporting rules below are fixed before the definitive Experiment B result is reported.

## Frozen Experiment A provenance
- Experiment A Git commit: `157fe440d97ad41178b2dc1c7183e1c8195ec6ba`
- Frozen `main.py` SHA-256: `5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d`
- Definitive Experiment A archive: `KGDI_paper_final_157fe440.tar.gz`
- Archive SHA-256: `362b250a5fa5d4ba8e9ac8177d9aa6253e26933321fcd5b8a26ff474c39b3c2a`
- IBM dataset SHA-256 recorded by Experiment A: `9b7d90f9e4c0f9b607126640e4279f0afd4120e7b3485614770753c9f393aaae`

Experiment B consumes the frozen final out-of-fold customer-level probabilities and decision variables from this archive. Predictive models are not refit.

## Research question
Under the same planning budget and frozen business assumptions, how much expected economic utility must be traded to increase engagement of strategically high-value churn risk?

## Knowledge-gated feasible action set
Experiment B uses the frozen risk threshold, uncertainty thresholds, high-value threshold, human-review assumptions, value horizon, margin assumption, intervention cost, retention success, and budget from Experiment A.

For a customer to enter the feasible action set, frozen out-of-fold churn probability must meet the Experiment A minimum-risk threshold. A direct retention offer is feasible only when its expected utility is positive. Human review is feasible only when the Experiment A uncertainty/high-value routing condition is met and expected review utility is positive.

When both actions are feasible for a customer, the optimizer may choose either action, but never both. This is enforced by a per-customer exclusivity constraint. This differs deliberately from Experiment A's fixed routing: Experiment B evaluates multi-objective allocation over the knowledge-eligible action set rather than reusing the Experiment A greedy ranking rule.

## Optimization design
A binary mixed-integer linear program (MILP) is solved using an epsilon-constraint formulation.

Primary objective:
1. maximize expected economic utility.

Constraints:
1. expected selected cost must not exceed the common planning budget;
2. at most one action may be selected per customer; and
3. expected high-value churn-risk mass engaged must be at least epsilon times its budget-constrained maximum.

Expected high-value churn-risk mass engaged is the sum of frozen OOF churn probabilities for selected customers whose CLTV percentile is at or above the frozen high-value quantile. IBM CLTV remains a dimensionless priority index and is not interpreted as currency.

The solver uses a locked relative MIP-gap tolerance of `1e-6`. Post-solution checks independently verify the budget, customer-action exclusivity, epsilon constraint, epsilon=0 economic endpoint, epsilon=1 strategic endpoint, and reported solver gaps.

## Locked epsilon grid
All 101 epsilon points from `0.00` through `1.00` in increments of `0.01` are evaluated and retained. No epsilon is selected based on observed churn outcomes.

## Locked utility-retention operating points
For decision interpretation, four operating rules are reported from the complete epsilon grid:
- 0% allowed loss of expected economic utility;
- 1% allowed loss;
- 2% allowed loss; and
- 5% allowed loss.

Within each rule, the reported point is the epsilon solution with the greatest expected high-value churn-risk engagement while satisfying the corresponding expected-utility floor. These levels are fixed before the definitive Experiment B report.

## Outcome separation
Observed `Churn Value` is excluded from candidate construction, feasibility rules, objectives, constraints, and operating-point selection. It is used only after allocation for descriptive ex-post evaluation of churn reach, high-value churn reach, outcome-anchored simulated cost, and outcome-anchored net-benefit proxy.

No Experiment B result may be described as causal revenue saved or causal retention impact.

## Statistical extension retained from Experiment A
The primary Experiment A KGDI policy is compared with all three Experiment A baselines using 1,000 paired customer bootstrap resamples with policy rerouting and budget reallocation inside each resample:
- Probability Threshold;
- Cost Aware;
- Uncertainty Review.

The bootstrap sample count remains `1000` and the locked bootstrap seed remains `20260907`, preserving the original Experiment B draft choice. The reported bootstrap quantity is the proportion of resamples with delta greater than zero. It is not labeled as a conventional frequentist p-value.

The Experiment B Pareto frontier itself is reported as a deterministic scenario/optimization analysis conditional on the frozen OOF probabilities and business assumptions; no confirmatory significance claim is attached to individual epsilon points.

## Reproducibility and integrity rules
The definitive Experiment B execution must:
- use Python 3.11;
- reproduce the top-level package versions recorded by Experiment A via `requirements-runpod.txt`;
- verify the exact Experiment A archive SHA-256 before extraction;
- verify every file listed in the Experiment A results manifest;
- verify the frozen local `main.py` SHA-256;
- reconstruct all four final Experiment A policies before solving Experiment B;
- run from a clean Git working tree; and
- write to a fresh, non-overwritten output directory.

Development-only bypass flags may be used only with `--dry-run`; they are rejected for the definitive execution.

## Reporting rules
The manuscript/supplement must report:
- the full 101-point epsilon grid;
- the non-dominated economic-strategic frontier;
- the four locked utility-retention operating points;
- Experiment A baseline locations for context;
- all three Experiment A paired-bootstrap comparisons;
- both favorable and unfavorable tradeoffs;
- the fact that Experiment B is a post-primary secondary analysis;
- the distinction between expected optimization quantities and ex-post outcome-anchored simulated quantities; and
- no post-hoc parameter tuning to force a favorable KGDI result.

## Protocol revision history
- **v1 — 2026-09-07:** initial secondary epsilon-constraint protocol added in commit `79895bad049786f6c9f2798853f27220ced7495d`.
- **v2 — 2026-09-08, before definitive Experiment B RunPod execution:** code-audit refinement. The MILP feasible set was upgraded to permit a choice between knowledge-eligible direct-offer and human-review actions with a one-action-per-customer constraint; full Experiment A manifest verification, the exact Experiment A archive hash, locked `1e-6` MIP-gap tolerance, exact environment checks, clean-tree/fresh-output safeguards, and locked 0/1/2/5% utility-retention reporting rules were added. Experiment A remained unchanged.
