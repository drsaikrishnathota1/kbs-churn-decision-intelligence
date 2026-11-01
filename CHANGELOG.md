# Changes and verification

## Changes made
1. `apply_hardening.py`: sensitivity-adjusted review strategic mass, assignment vs effectiveness metrics, explicit MILP constraint checks, reference-economics ablation diagnostics, dataset ID checks.
2. `apply_additional.py`: solver integrality checks, prediction data guards, quick-run publication warning, metric provenance label.
3. `revision_validation.py`: calibration intercept/slope and binned reliability; safe external evaluation of selected actions against common assumptions; input and solver result assertions; conditional action-fixed bootstrap; environment fingerprint.
4. `tests/test_scientific_validation.py`: six synthetic tests covering malformed probabilities, scoring, utility identity, MILP feasibility, duplicate actions, reproducible fixed-action bootstrap.
5. `preflight.py`: parser and required-file checks.
6. `.github/workflows/scientific-checks.yml`: run tests automatically on supported GitHub push/PR.

## Verified in this execution environment
- Python syntax compilation of all new scripts/tests: PASS.
- Standalone calibration diagnostics smoke calculation: PASS.
- Standalone fixed-action bootstrap determinism example: PASS.

## Not verified
- Full test suite against actual checkout; patch anchors against live original files; dataset leakage completeness; RunPod environment; experimental numerical claims. GitHub write permission returned HTTP 403, so the repository is unchanged.

## Known cautions
- The original manuscript's reported 'reach' is action assignment, not observed churn reduction.
- The original patch modifies the optimization target. Prior frontier numbers must NOT be reused.
- For true treatment effectiveness, intervention-assignment/outcome data are required.
- Run *all* CI and smoke tests on the actual checkout; adjust guarded anchors if upstream files changed.
