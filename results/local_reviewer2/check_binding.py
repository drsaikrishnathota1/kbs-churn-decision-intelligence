from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import milp as scipy_milp
from review_revision import optimizer as opt
from review_revision.common import RevisionScenario

# Local diagnostic only: do not change repository optimizer code.
def bounded_milp(*args, **kwargs):
    options = dict(kwargs.get("options") or {})
    options.update(time_limit=12.0, mip_rel_gap=0.001, disp=False)
    kwargs["options"] = options
    return scipy_milp(*args, **kwargs)

opt.milp = bounded_milp
cfg = RevisionScenario()
root = Path("results/restored_runpod/kbs-churn-decision-intelligence/results/reviewer_revision_smoke_fast")
out = Path("results/local_reviewer2")
rows = []

for dataset in ["ibm_telco", "iranian_churn"]:
    scores = pd.read_csv(root / dataset / "customer_scores.csv")
    n = len(scores)
    budget = n * cfg.offer_cost * cfg.budget_fraction
    capacity = int(np.floor(n * cfg.review_capacity_fraction))
    print(f"\n=== {dataset} ===", flush=True)

    for variant, override in [
        ("Full knowledge", None),
        ("No strategic knowledge", 1.0),
    ]:
        cand = opt.build_candidates(scores, cfg, strategic_override=override)
        reference = opt.build_candidates(scores, cfg)
        ref_lookup = {
            (int(r.row_index), r.action):
            (float(r.expected_utility), float(r.strategic_mass),
             float(r.expected_cost))
            for r in reference.itertuples()
        }

        def run(objective, threshold=0.0):
            c = (-cand["expected_utility"].to_numpy()
                 if objective == "utility"
                 else -cand["strategic_mass"].to_numpy())
            t = time.monotonic()
            result = scipy_milp(
                c=c,
                integrality=np.ones(len(c), dtype=int),
                bounds=opt.Bounds(np.zeros(len(c)), np.ones(len(c))),
                constraints=opt._constraints(
                    cand, n, budget, threshold, capacity
                ),
                options={
                    "presolve": True,
                    "mip_rel_gap": 0.001,
                    "time_limit": 12.0,
                    "disp": False,
                },
            )
            elapsed = time.monotonic() - t
            if result.x is None:
                print(f"  {variant}: {objective} no solution "
                      f"(status {result.status})", flush=True)
                return None, result, elapsed
            selected = cand.loc[result.x > 0.5].copy()
            return selected, result, elapsed

        strategic, sr, _ = run("strategic")
        if strategic is None:
            continue
        smax = float(strategic.strategic_mass.sum())

        for epsilon in [0.0, 0.5, 0.75, 1.0]:
            selected, result, elapsed = run(
                "utility", float(epsilon * smax)
            )
            if selected is None:
                continue

            keys = [(int(r.row_index), r.action)
                    for r in selected.itertuples()]
            if any(k not in ref_lookup for k in keys):
                raise RuntimeError("Reference candidate mismatch")

            ref_utility = sum(ref_lookup[k][0] for k in keys)
            ref_strategic = sum(ref_lookup[k][1] for k in keys)
            ref_cost = sum(ref_lookup[k][2] for k in keys)
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "epsilon": epsilon,
                "reference_utility": ref_utility,
                "reference_strategic_mass": ref_strategic,
                "reference_cost": ref_cost,
                "budget": budget,
                "offers": sum(k[1] == "Retention Offer" for k in keys),
                "reviews": sum(k[1] == "Human Review" for k in keys),
                "solver_status": result.status,
                "solver_gap": getattr(result, "mip_gap", np.nan),
                "seconds": round(elapsed, 2),
                "selected_rows": tuple(sorted(keys)),
            })
            print(
                f"  {variant}, ε={epsilon}: "
                f"utility={ref_utility:.4f}, "
                f"strategic={ref_strategic:.4f}, "
                f"status={result.status}, "
                f"gap={getattr(result, 'mip_gap', np.nan):.5f}",
                flush=True,
            )

    subset = [r for r in rows if r["dataset"] == dataset]
    for eps in [0.0, 0.5, 0.75, 1.0]:
        matches = [r for r in subset if r["epsilon"] == eps]
        if len(matches) == 2:
            a, b = matches
            diff = len(set(a["selected_rows"]) ^ set(b["selected_rows"]))
            for r in matches:
                r["selection_difference_symmetric"] = diff

    pd.DataFrame([
        {k: v for k, v in r.items() if k != "selected_rows"}
        for r in rows
    ]).to_csv(out / "binding_diagnostic.csv", index=False)

print("\nSaved:", out / "binding_diagnostic.csv")
print("DEVELOPMENT DIAGNOSTIC ONLY — NOT FINAL PUBLICATION RESULTS")
