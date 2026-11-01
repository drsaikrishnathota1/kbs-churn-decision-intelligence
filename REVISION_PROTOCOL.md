# Reviewer 2 + Reviewer 3 Scientific Revision Protocol

## Status
This is a post-peer-review analysis protocol for manuscript KNOSYS-D-26-21292. It is not preregistered and must not be described as confirmatory preregistration. The purpose is to prospectively freeze the reviewer-motivated analyses before the definitive revision results are inspected.

The original Experiment A (`main.py`) and its definitive outputs remain immutable. All new work lives under `review_revision/` and consumes either frozen Experiment A out-of-fold (OOF) outputs or independently processed external datasets.

## Reviewer questions addressed

### Reviewer 2
1. The original service-contract graph is non-binding at the nominal point and must not be presented as the operative source of the allocation gain.
2. The hard 75th-percentile CLTV gate is removed from the revised optimization so the strategic result is not mechanically constructed by the same threshold used for evaluation.
3. Strategic value becomes a continuous decision quantity.
4. The primary revised decision method is an exact epsilon-constraint optimizer.
5. Binding-knowledge ablations quantify how allocations change when strategic, economic, reviewer-quality, review-capacity, or action-availability knowledge is removed.
6. Generalization is tested on external telecom churn datasets using the same decision equations and no dataset-specific threshold tuning.

### Reviewer 3
1. Predictive preprocessing is documented from the actual frozen code: fold-local numeric median imputation, standardization, categorical most-frequent imputation, one-hot encoding, model fitting, and nested calibration; tree learners provide nonlinear interaction capacity while logistic regression supplies a linear additive component.
2. A fixed cost-benefit-matrix comparator is added under the same IBM planning budget.
3. A direct `p × strategic-value` comparator is added under the same IBM planning budget.
4. No uplift/treatment-effect experiment is attempted on IBM because the dataset does not contain intervention assignment plus treatment outcome information sufficient for causal uplift estimation. This is reported as a future-data requirement rather than simulated evidence.

## Core revised formulation
For each customer i:

- p_i: calibrated churn probability.
- q_i in [0,1]: percentile-normalized strategic-value score supplied by the application.
- e_i in [0,1]: percentile-normalized economic-value proxy for cross-dataset validation.
- r_i = p_i q_i: continuous strategic value-at-risk score.

For a selected action x_ia:

S(x) = sum_i,a x_ia p_i q_i

The optimizer maximizes expected economic utility subject to:

1. common planning budget;
2. at most one action per customer;
3. optional human-review capacity;
4. epsilon strategic requirement `S(x) >= epsilon * S_max`.

The epsilon grid is frozen at 0.00 through 1.00 in increments of 0.01.

The 75th-percentile strategic-value subgroup is retained only as an ex-post interpretability metric and never appears in candidate eligibility, the strategic objective, or the epsilon constraint.

## Knowledge layer
The revised knowledge layer is declarative and operational:

| Knowledge item | Representation | Decision role |
|---|---|---|
| Churn risk | calibrated p_i | expected intervention benefit |
| Strategic customer value | q_i percentile | continuous strategic-risk objective |
| Economic value | e_i or IBM scenario value | intervention utility |
| Offer economics | success and offer cost | offer feasibility/utility |
| Reviewer capability | sensitivity, specificity, review cost | review utility |
| Uncertainty | predictive entropy | review eligibility only |
| Resource availability | budget | global constraint |
| Review capacity | max review count | scarce-human-resource constraint |
| Action exclusivity | one action/customer | feasibility constraint |

The original auxiliary service-contract graph may remain in the manuscript as an honest diagnostic but is not part of the revised primary knowledge claim.

## Datasets and frozen mappings

### IBM Telco (primary/frozen OOF)
- Prediction: use frozen Experiment A calibrated OOF ensemble probabilities; do not refit.
- Strategic value: `CLTV` percentile.
- Cross-dataset economic proxy: `Monthly Charges` percentile.
- Reviewer 3 IBM comparators use the original Experiment A economic scenario (`annual_margin_value`, offer cost, success rate, budget).

### Iranian Churn (UCI 563)
- Source: UCI Machine Learning Repository, DOI 10.24432/C5JW3Z, CC BY 4.0.
- Target: `Churn`.
- Strategic value: `Customer Value` percentile.
- Economic proxy: `Charge Amount` percentile.
- Predictive features: all pre-intervention/non-target variables except `Customer Value`; `Charge Amount` may remain a predictive feature because it is an observed customer attribute, analogous to monthly charges in IBM.
- No dataset-specific decision threshold tuning.

### Cell2Cell
- Source: public Cell2Cell training data originating from the Duke/Teradata churn modeling data; use the labeled training partition only.
- Target: `Churn`.
- Strategic-value proxy: `TotalRecurringCharge` percentile (or exact case-insensitive equivalent).
- Economic proxy: `MonthlyRevenue` percentile (or exact case-insensitive equivalent).
- Explicitly label these as value/revenue proxies, not CLTV.
- Exclude obvious prior-retention-intervention variables from predictive features when present (for example retention calls/offers) so prior intervention process variables do not contaminate the predictive layer.
- Drop rows only when the target or required value proxies are missing; all remaining predictor missingness is handled fold-locally.

## External predictive protocol
External datasets use the same learner family as Experiment A:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- equal-weight calibrated ensemble

Preprocessing for all four models:
- numeric: fold-local median imputation + standardization;
- categorical: fold-local most-frequent imputation + one-hot encoding with unknown-category ignore;
- no global winsorization or outlier deletion;
- no synthetic oversampling;
- calibration fitted only on a dedicated 20% partition inside each outer training fold.

External validation uses 5 folds × 3 repeats, seed 20261031. This is fixed for the revision analyses.

## Common cross-dataset decision scenario
To avoid falsely interpreting heterogeneous external value fields as dollars, cross-dataset decision validation uses normalized utility units:

- offer cost c_o = 0.05
- retention success s = 0.35
- human review cost c_h = 0.0125
- reviewer sensitivity Se = 0.85
- reviewer specificity Sp = 0.85
- minimum churn probability = 0.20
- review entropy threshold = 0.45
- planning budget = N * c_o * 0.20
- review capacity = floor(0.05 * N)

Economic value is e_i in [0,1], a percentile-normalized dataset-native economic proxy. These normalized results are scenario-comparative, not causal realized profit.

## Reviewer 3 comparator definitions on IBM

### Fixed cost-benefit matrix
Use a common reference value `V_ref = median(annual_margin_value)` computed from the frozen OOF customer table. For a direct offer:
- would churn: s * V_ref - c_o
- would not churn: -c_o
- no action: 0 in either state

Expected value is `p_i * s * V_ref - c_o`. Eligible positive-value customers are ranked by expected value and offered until the same Experiment A planning budget is exhausted.

### p × strategic-value comparator
Define `q_i = CLTV percentile`, `R_i = p_i q_i`. Require the same minimum churn-risk threshold and positive original offer utility, rank direct-offer candidates by R_i, and allocate until the same planning budget is exhausted.

## Binding-knowledge ablation
Run at the predeclared midpoint diagnostic epsilon = 0.50 in the normalized cross-dataset scenario.

Variants:
1. Full decision knowledge.
2. No strategic-value knowledge: replace q_i by 1 for the strategic objective.
3. No economic heterogeneity: replace e_i by the dataset median e.
4. No reviewer-quality knowledge: replace reviewer sensitivity/specificity with an uninformative 0.50/0.50 reviewer.
5. No review-capacity knowledge: remove the review-count cap while retaining the budget.
6. Offer-only: remove human review from the feasible action set.
7. Risk-only comparator: direct offers ranked only by p_i under the same budget.

For each variant report:
- number and percentage of customer actions different from the full allocation;
- expected utility;
- strategic-risk mass engaged;
- strategic-risk fraction of S_max;
- ex-post 75th-percentile high-value churn reach;
- overall churn reach;
- number of offers/reviews;
- budget utilization.

No claim is made in advance that every ablation must alter decisions; whether a knowledge item is binding is an empirical result.

## Frontier reporting
For every dataset save all 101 epsilon solutions. Also report utility-retention operating points using fixed allowed expected-utility losses of 0%, 1%, 2%, and 5% relative to the epsilon=0 economic optimum. Within each floor, select the solution with the greatest strategic-risk engagement.

## Bootstrap
Reviewer 3 IBM comparator deltas versus cost-aware targeting use 1,000 paired customer bootstrap resamples with reranking/reallocation inside each resample; seed 20261031. These resample-positive fractions are not classical p-values.

External-dataset frontier analyses are deterministic conditional on the OOF probabilities and scenario assumptions. No significance claim is attached to individual epsilon points.

## Invariants
Code must verify:
- observed churn labels never enter candidate utilities, ranking, objectives, constraints, or epsilon selection;
- budget is never exceeded beyond numerical tolerance;
- at most one action per customer;
- all selected actions satisfy feasibility rules;
- q_i and e_i are within [0,1];
- epsilon requirement is met;
- epsilon=0 reproduces the economic optimum of the declared candidate set;
- epsilon=1 reaches the maximum budget-feasible strategic objective within solver tolerance;
- the same optimizer and scenario parameters are used across datasets;
- dataset adapters only map schema and do not tune decision thresholds.

## Required outputs
Per dataset:
- `dataset_manifest.json`
- `predictive_metrics.csv`
- `customer_scores.csv`
- `epsilon_frontier.csv`
- `operating_points.csv`
- `knowledge_ablation.csv`
- `run_manifest.json`

IBM Reviewer 3 outputs:
- `reviewer3_comparators.csv`
- `reviewer3_bootstrap.json`

Combined outputs:
- `cross_dataset_predictive_summary.csv`
- `cross_dataset_decision_summary.csv`
- `revision_manifest.json`

## Interpretation restrictions
- External normalized utility is not dollar profit.
- Outcome-anchored IBM net benefit remains a simulated proxy, not causal realized revenue.
- No uplift/treatment-effect result is reported without intervention assignment and treatment-outcome data.
- No post-hoc threshold tuning is permitted after seeing definitive RunPod results.
