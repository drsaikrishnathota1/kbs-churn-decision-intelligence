# Experiment B Protocol

## Status
Prespecified secondary analysis. Experiment A remains immutable and is not overwritten.

## Frozen Experiment A provenance
- Git commit: `157fe440d97ad41178b2dc1c7183e1c8195ec6ba`
- `main.py` SHA-256: `5e4345496225856b398e8c85b3358dff0f2f680c22ac56f33d6aedfb2f628c2d`
- Experiment A archive reviewed before this protocol: `KGDI_paper_final_157fe440.tar.gz`

## Research question
How does a knowledge-guided churn intervention policy trade expected economic utility against expected coverage of strategically high-value churn risk under the same planning budget?

## Optimization design
Experiment B uses the frozen final OOF probabilities from Experiment A. Predictive models are not refit.

The decision candidate-routing rules, uncertainty thresholds, human-review assumptions, high-value threshold, and business parameters are frozen from Experiment A.

The allocation is solved as a binary mixed-integer linear program using an epsilon-constraint formulation:
1. maximize expected economic utility;
2. subject to the common planning budget; and
3. require expected high-value churn mass reached to be at least epsilon times its budget-constrained maximum.

Expected high-value churn mass is defined as the sum of frozen OOF churn probabilities among selected customers whose CLTV percentile is at or above the frozen high-value quantile.

## Prespecified epsilon grid
All 101 points from 0.00 through 1.00 in increments of 0.01 are evaluated and reported. No point is selected or tuned after observing outcomes.

## Outcome separation
Observed `Churn Value` is excluded from optimization. It is used only after allocation for ex-post evaluation of churn reach, high-value churn reach, cost, and outcome-anchored net-benefit proxy.

## Statistical extension
The primary Experiment A KGDI policy is compared with all three baselines using 1,000 paired customer bootstrap resamples with policy rerouting and budget reallocation inside every resample:
- Probability Threshold;
- Cost Aware;
- Uncertainty Review.

The reported quantity is the bootstrap proportion with delta greater than zero, not a conventional p-value.

## Reporting rule
The manuscript will report:
- the full epsilon grid;
- the non-dominated frontier;
- all baseline comparisons;
- both favorable and unfavorable tradeoffs;
- no post-hoc parameter tuning to force KGDI superiority.
