from __future__ import annotations

from dataclasses import replace
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .common import RevisionScenario, assert_unit_interval


def utilities(scores: pd.DataFrame, cfg: RevisionScenario, *, reviewer_override=None, economic_override=None):
    p = scores["churn_probability"].to_numpy(float)
    e = scores["economic_value"].to_numpy(float)
    if economic_override is not None:
        e = np.full(len(scores), float(economic_override))
    se = cfg.human_sensitivity
    sp = cfg.human_specificity
    if reviewer_override is not None:
        se, sp = reviewer_override
    offer = p * cfg.retention_success * e - cfg.offer_cost
    review_offer_prob = p * se + (1 - p) * (1 - sp)
    review_cost = cfg.human_review_cost + cfg.offer_cost * review_offer_prob
    review = p * se * cfg.retention_success * e - review_cost
    return offer, review, review_cost


def build_candidates(
    scores: pd.DataFrame,
    cfg: RevisionScenario,
    *,
    strategic_override=None,
    economic_override=None,
    reviewer_override=None,
    offer_only=False,
):
    required = ["churn_probability", "predictive_entropy", "strategic_value", "economic_value"]
    missing = [c for c in required if c not in scores.columns]
    if missing:
        raise ValueError(f"scores missing columns: {missing}")
    assert_unit_interval("churn_probability", scores["churn_probability"])
    assert_unit_interval("strategic_value", scores["strategic_value"])
    assert_unit_interval("economic_value", scores["economic_value"])

    p = scores["churn_probability"].to_numpy(float)
    q = scores["strategic_value"].to_numpy(float)
    if strategic_override is not None:
        q = np.full(len(scores), float(strategic_override))
    entropy = scores["predictive_entropy"].to_numpy(float)
    offer_u, review_u, review_cost = utilities(
        scores, cfg, reviewer_override=reviewer_override, economic_override=economic_override
    )
    rows = []
    for i in range(len(scores)):
        if p[i] < cfg.minimum_churn_risk:
            continue
        common = {
            "row_index": i,
            "strategic_mass": float(p[i] * q[i]),
            "churn_probability": float(p[i]),
            "strategic_value": float(q[i]),
        }
        if offer_u[i] > 0:
            rows.append({
                **common, "action": "Retention Offer",
                "expected_utility": float(offer_u[i]),
                "expected_cost": float(cfg.offer_cost),
            })
        if (not offer_only and entropy[i] >= cfg.review_entropy_threshold and review_u[i] > 0):
            rows.append({
                **common, "action": "Human Review",
                "expected_utility": float(review_u[i]),
                "expected_cost": float(review_cost[i]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No feasible candidates")
    return out.reset_index(drop=True)


def _constraints(candidates, n_customers, budget, min_strategic=0.0, max_reviews=None):
    m = len(candidates)
    idx = np.arange(m)
    rows = candidates["row_index"].to_numpy(int)
    exclusivity = coo_matrix(
        (np.ones(m), (rows, idx)), shape=(n_customers, m)
    ).tocsr()
    cons = [
        LinearConstraint(candidates["expected_cost"].to_numpy(float), -np.inf, budget),
        LinearConstraint(exclusivity, -np.inf, np.ones(n_customers)),
    ]
    if min_strategic > 0:
        cons.append(LinearConstraint(
            candidates["strategic_mass"].to_numpy(float), min_strategic, np.inf
        ))
    if max_reviews is not None:
        review_vec = (candidates["action"].to_numpy(object) == "Human Review").astype(float)
        cons.append(LinearConstraint(review_vec, -np.inf, float(max_reviews)))
    return cons


def solve(candidates, n_customers, budget, objective, min_strategic=0.0, max_reviews=None):
    if objective == "utility":
        c = -candidates["expected_utility"].to_numpy(float)
    elif objective == "strategic":
        c = -candidates["strategic_mass"].to_numpy(float)
    else:
        raise ValueError(objective)
    res = milp(
        c=c,
        integrality=np.ones(len(candidates), dtype=int),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=_constraints(candidates, n_customers, budget, min_strategic, max_reviews),
        options={"presolve": True, "mip_rel_gap": 1e-7, "disp": False},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"MILP failed: status={res.status}, message={res.message}")
    chosen = res.x > 0.5
    selected = candidates.loc[chosen].copy()
    return selected


def actions_from_selected(selected, n):
    actions = np.array(["No Action"] * n, dtype=object)
    for r in selected.itertuples(index=False):
        if actions[r.row_index] != "No Action":
            raise RuntimeError("Action exclusivity violated")
        actions[r.row_index] = r.action
    return actions


def evaluate(scores, selected, cfg: RevisionScenario, budget, smax):
    actions = actions_from_selected(selected, len(scores))
    y = scores["target"].to_numpy(int)
    q = scores["strategic_value"].to_numpy(float)
    strategic = float(selected["strategic_mass"].sum())
    utility = float(selected["expected_utility"].sum())
    spent = float(selected["expected_cost"].sum())
    high = q >= 0.75
    churn = y == 1
    acted = actions != "No Action"
    hv_den = int(np.sum(high & churn))
    ch_den = int(np.sum(churn))
    return {
        "expected_utility": utility,
        "strategic_risk_mass": strategic,
        "strategic_fraction_of_smax": strategic / smax if smax > 0 else np.nan,
        "high_value_churn_reach": float(np.sum(acted & high & churn) / hv_den) if hv_den else np.nan,
        "churn_reach": float(np.sum(acted & churn) / ch_den) if ch_den else np.nan,
        "offers": int(np.sum(actions == "Retention Offer")),
        "reviews": int(np.sum(actions == "Human Review")),
        "budget_used": spent,
        "budget_utilization": spent / budget if budget > 0 else np.nan,
        "actions": actions,
    }


def frontier(scores: pd.DataFrame, cfg: RevisionScenario):
    n = len(scores)
    budget = n * cfg.offer_cost * cfg.budget_fraction
    max_reviews = int(np.floor(cfg.review_capacity_fraction * n))
    cand = build_candidates(scores, cfg)
    econ = solve(cand, n, budget, "utility", max_reviews=max_reviews)
    strategic = solve(cand, n, budget, "strategic", max_reviews=max_reviews)
    smax = float(strategic["strategic_mass"].sum())
    u0 = float(econ["expected_utility"].sum())
    rows = []
    eps_values = np.round(np.arange(0, 1.0000001, cfg.epsilon_step), 10)
    for eps in eps_values:
        sel = solve(
            cand, n, budget, "utility", min_strategic=float(eps * smax), max_reviews=max_reviews
        )
        m = evaluate(scores, sel, cfg, budget, smax)
        rows.append({
            "epsilon": float(eps),
            "expected_utility": m["expected_utility"],
            "utility_retained": m["expected_utility"] / u0 if u0 else np.nan,
            "strategic_risk_mass": m["strategic_risk_mass"],
            "strategic_fraction_of_smax": m["strategic_fraction_of_smax"],
            "high_value_churn_reach": m["high_value_churn_reach"],
            "churn_reach": m["churn_reach"],
            "offers": m["offers"],
            "reviews": m["reviews"],
            "budget_used": m["budget_used"],
            "budget_utilization": m["budget_utilization"],
        })
    return pd.DataFrame(rows), {"budget": budget, "max_reviews": max_reviews, "smax": smax, "u0": u0}


def operating_points(frontier_df: pd.DataFrame):
    rows = []
    for loss in [0.00, 0.01, 0.02, 0.05]:
        floor = 1.0 - loss
        eligible = frontier_df[frontier_df["utility_retained"] >= floor - 1e-10]
        if eligible.empty:
            continue
        r = eligible.sort_values(["strategic_risk_mass", "epsilon"], ascending=[False, False]).iloc[0]
        d = r.to_dict()
        d["allowed_utility_loss"] = loss
        rows.append(d)
    return pd.DataFrame(rows)


def _midpoint_selected(scores, cfg, *, strategic_override=None, economic_override=None,
                       reviewer_override=None, offer_only=False, remove_review_capacity=False):
    n = len(scores)
    budget = n * cfg.offer_cost * cfg.budget_fraction
    max_reviews = None if remove_review_capacity else int(np.floor(cfg.review_capacity_fraction * n))
    cand = build_candidates(
        scores, cfg, strategic_override=strategic_override, economic_override=economic_override,
        reviewer_override=reviewer_override, offer_only=offer_only
    )
    s_sel = solve(cand, n, budget, "strategic", max_reviews=max_reviews)
    smax = float(s_sel["strategic_mass"].sum())
    sel = solve(cand, n, budget, "utility", min_strategic=0.50 * smax, max_reviews=max_reviews)
    return sel, budget, smax


def risk_only(scores, cfg):
    n = len(scores)
    budget = n * cfg.offer_cost * cfg.budget_fraction
    p = scores["churn_probability"].to_numpy(float)
    order = np.argsort(-p)
    actions = np.array(["No Action"] * n, dtype=object)
    spent = 0.0
    for i in order:
        if p[i] < cfg.minimum_churn_risk:
            continue
        if spent + cfg.offer_cost <= budget + 1e-12:
            actions[i] = "Retention Offer"
            spent += cfg.offer_cost
    return actions


def knowledge_ablation(scores: pd.DataFrame, cfg: RevisionScenario):
    variants = {
        "Full decision knowledge": {},
        "No strategic-value knowledge": {"strategic_override": 1.0},
        "No economic heterogeneity": {"economic_override": float(scores["economic_value"].median())},
        "No reviewer-quality knowledge": {"reviewer_override": (0.50, 0.50)},
        "No review-capacity knowledge": {"remove_review_capacity": True},
        "Offer-only": {"offer_only": True},
    }
    full_sel, full_budget, full_smax = _midpoint_selected(scores, cfg)
    full_eval = evaluate(scores, full_sel, cfg, full_budget, full_smax)
    full_actions = full_eval.pop("actions")
    rows = [{"variant": "Full decision knowledge", "different_actions_vs_full": 0, **full_eval}]
    for name, kw in variants.items():
        if name == "Full decision knowledge":
            continue
        sel, budget, smax = _midpoint_selected(scores, cfg, **kw)
        ev = evaluate(scores, sel, cfg, budget, smax)
        actions = ev.pop("actions")
        rows.append({
            "variant": name,
            "different_actions_vs_full": int(np.sum(actions != full_actions)),
            **ev,
        })

    r_actions = risk_only(scores, cfg)
    y = scores["target"].to_numpy(int)
    q = scores["strategic_value"].to_numpy(float)
    p = scores["churn_probability"].to_numpy(float)
    acted = r_actions != "No Action"
    strategic = float(np.sum(p[acted] * q[acted]))
    high = q >= 0.75
    churn = y == 1
    rows.append({
        "variant": "Risk-only",
        "different_actions_vs_full": int(np.sum(r_actions != full_actions)),
        "expected_utility": np.nan,
        "strategic_risk_mass": strategic,
        "strategic_fraction_of_smax": strategic / full_smax if full_smax else np.nan,
        "high_value_churn_reach": float(np.sum(acted & high & churn) / np.sum(high & churn)),
        "churn_reach": float(np.sum(acted & churn) / np.sum(churn)),
        "offers": int(np.sum(acted)),
        "reviews": 0,
        "budget_used": float(np.sum(acted) * cfg.offer_cost),
        "budget_utilization": float(np.sum(acted) * cfg.offer_cost / full_budget),
    })
    return pd.DataFrame(rows)
