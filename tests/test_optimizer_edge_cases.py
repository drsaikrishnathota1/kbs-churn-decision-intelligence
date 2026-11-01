"""Run against the repository optimizer, not a duplicate implementation."""
import numpy as np
import pandas as pd
import pytest
from review_revision.common import RevisionScenario
from review_revision.optimizer import build_candidates, solve


def toy():
    return pd.DataFrame({
        'churn_probability':[.9,.8,.7,.6],
        'predictive_entropy':[.5,.6,.7,.8],
        'strategic_value':[.9,.8,.7,.6],
        'economic_value':[1.,.9,.8,.7],
        'target':[1,0,1,0],
    })


def test_empty_eligible_candidate_set_is_explicit():
    df=toy();df['churn_probability']=0.
    with pytest.raises(RuntimeError,match='No feasible candidates'):
        build_candidates(df, RevisionScenario())


def test_zero_budget_selects_nothing():
    df=toy();cand=build_candidates(df,RevisionScenario())
    selected=solve(cand,len(df),budget=0.,objective='utility')
    assert selected.empty


def test_infeasible_strategic_target_is_explicit():
    df=toy();cand=build_candidates(df,RevisionScenario())
    with pytest.raises(RuntimeError,match='MILP failed'):
        solve(cand,len(df),budget=0.,objective='utility',min_strategic=100.)


def test_never_assign_two_actions_per_customer():
    df=toy();cand=build_candidates(df,RevisionScenario())
    selected=solve(cand,len(df),budget=2.,objective='utility')
    assert not selected.row_index.duplicated().any()
    assert selected.expected_cost.sum() <= 2 + 1e-7
