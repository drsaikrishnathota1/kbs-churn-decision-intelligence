import numpy as np
import pandas as pd

from review_revision.common import RevisionScenario
from review_revision.optimizer import frontier, knowledge_ablation


def synthetic_scores(n=120, seed=7):
    rng = np.random.default_rng(seed)
    p = np.clip(rng.beta(2.2, 3.0, n), 0.02, 0.98)
    q = np.clip(rng.random(n), 0.001, 0.999)
    e = np.clip(0.25 + 0.65 * rng.random(n), 0.001, 0.999)
    entropy = -(p*np.log2(p) + (1-p)*np.log2(1-p))
    y = rng.binomial(1, p)
    return pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(n)],
        "target": y,
        "churn_probability": p,
        "predictive_entropy": entropy,
        "strategic_value": q,
        "economic_value": e,
    })


def test_frontier_invariants():
    scores = synthetic_scores()
    cfg = RevisionScenario(epsilon_step=0.50)
    f, meta = frontier(scores, cfg)
    assert list(f["epsilon"]) == [0.0, 0.5, 1.0]
    assert (f["budget_used"] <= meta["budget"] + 1e-8).all()
    assert (f["strategic_fraction_of_smax"] + 1e-8 >= f["epsilon"]).all()
    assert abs(f.iloc[0]["utility_retained"] - 1.0) < 1e-8
    assert f.iloc[-1]["strategic_fraction_of_smax"] >= 1.0 - 1e-7


def test_ablation_runs_and_has_full_reference():
    scores = synthetic_scores()
    cfg = RevisionScenario(epsilon_step=0.50)
    a = knowledge_ablation(scores, cfg)
    assert "Full decision knowledge" in set(a["variant"])
    full = a[a.variant == "Full decision knowledge"].iloc[0]
    assert full["different_actions_vs_full"] == 0
    assert (a["budget_utilization"] <= 1.0 + 1e-8).all()
