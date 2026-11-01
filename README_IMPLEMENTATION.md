# Pre-RunPod scientific hardening v2

Target branch: `reviewer2-reviewer3-revision`. **Do not** apply patches on `main` or overwrite frozen Experiment A. Apply in a local branch with a clean working tree.

## Apply

1. Extract this package at the repository root, preserving `tests/` and `.github/workflows/` paths. `apply_hardening.py` and `apply_additional.py` belong at the root.
2. Create a new work branch: `git switch -c pre-runpod-scientific-hardening`.
3. Run `python apply_hardening.py` and `python apply_additional.py`.
4. Run `python preflight.py`, `python -m py_compile review_revision/*.py revision_validation.py`, and `python -m pytest -q`.
5. Review `git diff`, including the `.before_hardening` and `.pre_v2` backup files. **Do not commit backup files.** Remove them before commit, after reviewing.
6. Execute the documented **quick smoke run** using frozen IBM OOF and both external datasets. Inspect all result CSVs and invariants. Then execute the definitive RunPod run, never report quick-mode results.

## Implementation scope

- Reviewed/guarded original MILP semantics; effectiveness-adjusted review mass and supplemental coverage are now proposed rather than conflating review assignments with retained customers.
- Common reference economics for knowledge ablations. The altered-assumption objective must not be compared as though economically interchangeable with the baseline objective.
- Stronger MILP status, integrality, budget, exclusivity, review-capacity and epsilon-feasibility assertions.
- Strict prediction input checks and identifier checks on data adapters.
- Independent post-hoc validation helpers for calibration, action economic value, conditional descriptive bootstrap, selection constraints and environment hashes.
- CI workflow and tests; **these are not a substitute for full scientific reproduction**.

## Still mandatory before paper submission

- **Nested CV:** The original predictive hyperparameters are frozen. Do not tune on outer held-out folds. If tuning is added later, put the search wholly within training folds; do not describe fixed-parameter OOF as nested *hyperparameter search*.
- **Temporal leakage:** No automatic regex can prove all external features predate the churn label. Manually review data dictionaries and remove future-looking, derived-outcome, retention/intervention and ID fields. Freeze allowlists before the definitive run.
- **Additional baselines:** Use matched-budget action policies (risk-only, fixed matrix, CLTV, p×CLTV, offer-only, KGDI heuristic, revised exact optimizer); preserve equal eligibility and evaluate under common fixed economics.
- **Uncertainty:** `bootstrap_fixed_actions` is *conditional and descriptive*; it is NOT a replacement for Reviewer 3's rerouted paired bootstrap. Neither captures training-set uncertainty unless the model is refit per bootstrap replicate.
- **Reviewer effects:** Sensitivity-adjusted review engagement is scenario-defined, not demonstrated treatment uplift or causal retention. Offer success also remains assumed.
- **External validation:** Dataset-specific economic percentiles are dimensionless proxies. Do not mix them with IBM dollar benefits or claim a common real-world currency.
- **Sensitivity:** Perform an explicitly prespecified grid varying budget, offer cost, retention success, sensitivity *independently* of specificity, review cost, and alternate strategic rank transforms. Report all, including unfavorable outcomes.
- **Optimizer:** SciPy MILP's gap tolerance is approximate, not proof of exact global optimality. Report solver status and precision; 'exact' denotes an integer formulation unless zero certified gap is shown.
- **Model metrics:** Calibration statistics are estimated using held-out OOF scores but calibration slope/intercept are descriptive and should not be advertised as independent external testing.

## Status

This deliverable is a guarded source patch plus automated helper/test package. No raw IBM/UCI/Cell2Cell datasets, frozen OOF tables, or substantive RunPod outputs are bundled. Definitive statistical claims require those inputs and successful execution. Preserve the original paper's numbers unchanged until validated.
