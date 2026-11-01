#!/usr/bin/env python3
"""Apply conservative review hardening to reviewer2-reviewer3-revision checkout.
Creates a backup of each changed source file. Run from repository root.
"""
from pathlib import Path
import shutil

ROOT=Path.cwd()

def modify(name, transformations):
    path=ROOT/name
    if not path.is_file(): raise FileNotFoundError(f'Missing {name}; run from repository root')
    original=path.read_text(encoding='utf8')
    result=original
    for old,new in transformations:
        n=result.count(old)
        if n != 1: raise RuntimeError(f'{name}: expected exactly one occurrence, found {n}: {old[:90]!r}')
        result=result.replace(old,new,1)
    if result!=original:
        backup=path.with_suffix(path.suffix+'.before_hardening')
        if not backup.exists(): shutil.copy2(path,backup)
        path.write_text(result,encoding='utf8')
        print('Updated',name)

def main():
    modify('review_revision/optimizer.py', [
        ('        if (not offer_only and entropy[i] >= cfg.review_entropy_threshold and review_u[i] > 0):\n            rows.append({\n                **common, "action": "Human Review",',
         '        if (not offer_only and entropy[i] >= cfg.review_entropy_threshold and review_u[i] > 0):\n            # Expected identified-risk engagement differs from mere review assignment.\n            # Sensitivity applies only to the true-churn component of the reviewed population.\n            rows.append({\n                **common, "strategic_mass": float(p[i] * q[i] * (reviewer_override[0] if reviewer_override is not None else cfg.human_sensitivity)), "action": "Human Review",'),
        ('    acted = actions != "No Action"\n    hv_den',
         '    acted = actions != "No Action"\n    # Assignment coverage is operational workload, not equivalent to effective engagement.\n    # Effective figures are assumptions-based, NOT actual retention/causal uplift.\n    effective = np.where(actions == "Retention Offer", 1.0,\n                         np.where(actions == "Human Review", cfg.human_sensitivity, 0.0))\n    hv_den'),
        ('        "churn_reach": float(np.sum(acted & churn) / ch_den) if ch_den else np.nan,',
         '        "churn_reach": float(np.sum(acted & churn) / ch_den) if ch_den else np.nan,\n        "effective_high_value_churn_reach": float(np.sum(effective * (high & churn)) / hv_den) if hv_den else np.nan,\n        "effective_churn_reach": float(np.sum(effective * churn) / ch_den) if ch_den else np.nan,'),
        ('            "churn_reach": m["churn_reach"],',
         '            "churn_reach": m["churn_reach"],\n            "effective_high_value_churn_reach": m["effective_high_value_churn_reach"],\n            "effective_churn_reach": m["effective_churn_reach"],'),
        ('    if not res.success or res.x is None:\n        raise RuntimeError(f"MILP failed: status={res.status}, message={res.message}")\n    chosen = res.x > 0.5',
         '    if not res.success or res.x is None:\n        raise RuntimeError(f"MILP failed: status={res.status}, message={res.message}")\n    chosen = res.x > 0.5\n    selected_tmp = candidates.loc[chosen]\n    if selected_tmp["expected_cost"].sum() > budget + 1e-7 * max(1, budget):\n        raise RuntimeError("MILP returned over-budget solution")\n    if selected_tmp["row_index"].duplicated().any():\n        raise RuntimeError("MILP returned multiple actions per customer")\n    if min_strategic > 0 and selected_tmp["strategic_mass"].sum() + 1e-7 * max(1, min_strategic) < min_strategic:\n        raise RuntimeError("MILP returned strategic-infeasible solution")\n    if max_reviews is not None and (selected_tmp["action"] == "Human Review").sum() > max_reviews:\n        raise RuntimeError("MILP exceeded review capacity")'),
        ('        ev = evaluate(scores, sel, cfg, budget, smax)\n        actions = ev.pop("actions")',
         '        ev = evaluate(scores, sel, cfg, budget, smax)\n        actions = ev.pop("actions")\n        # Common-reference evaluation: do NOT interpret the ablated selection\n        # objective as comparable realized economic utility.\n        u_ref_offer, u_ref_review, c_ref_review = utilities(scores, cfg)\n        ref_utility = float(np.sum(np.where(actions == "Retention Offer", u_ref_offer,\n                              np.where(actions == "Human Review", u_ref_review, 0.0))))\n        ref_cost = float(np.sum(np.where(actions == "Retention Offer", cfg.offer_cost,\n                           np.where(actions == "Human Review", c_ref_review, 0.0))))\n        ev["reference_expected_utility"] = ref_utility\n        ev["reference_expected_cost"] = ref_cost\n        ev["reference_budget_feasible"] = bool(ref_cost <= budget + 1e-8)'),
    ])
    modify('review_revision/datasets.py', [
        ('    y = _coerce_binary(df[target_col], target_col)\n    strategic = pd.to_numeric(df[strategic_col], errors="coerce")',
         '    y = _coerce_binary(df[target_col], target_col)\n    strategic = pd.to_numeric(df[strategic_col], errors="coerce")'),
        ('    return DatasetBundle(\n        name="ibm_telco",',
         '    if score_frame["customer_id"].isna().any() or score_frame["customer_id"].duplicated().any():\n        raise ValueError("IBM frozen OOF has missing or duplicated customer identifiers")\n    if len(score_frame) != len(df):\n        raise ValueError("IBM frozen OOF has missing rows; silent row-dropping is prohibited")\n    return DatasetBundle(\n        name="ibm_telco",'),
        ('    features = df[[c for c in df.columns if c not in exclude]].copy()\n    return DatasetBundle(\n        name="iranian_churn",',
         '    features = df[[c for c in df.columns if c not in exclude]].copy()\n    if any(_norm_col(c) in {"customerid", "id", "customer"} for c in features):\n        raise ValueError("Iranian predictive features include possible identity field; audit and explicitly exclude")\n    return DatasetBundle(\n        name="iranian_churn",'),
    ])
    print('Complete. Run pytest and smoke tests. DO NOT report revised numbers without full rerun.')

if __name__=='__main__':main()
