import numpy as np
import pandas as pd
import pytest
from revision_validation import check_probabilities,calibration_diagnostics,validate_selection,bootstrap_fixed_actions
from review_revision.common import RevisionScenario
from review_revision.optimizer import build_candidates,solve,utilities,evaluate


def scores(n=32):
    p=np.linspace(.2,.85,n);q=np.linspace(.1,.99,n);e=np.linspace(.3,.95,n)
    return pd.DataFrame(dict(churn_probability=p,predictive_entropy=-(p*np.log2(p)+(1-p)*np.log2(1-p)),strategic_value=q,economic_value=e,target=(p>.55).astype(int)))


def test_probability_rejection():
    with pytest.raises(ValueError):check_probabilities([.2,np.nan],[0,1])
    with pytest.raises(ValueError):check_probabilities([-.1,.7],[0,1])
    with pytest.raises(ValueError):check_probabilities([.1,.9],[0,2])


def test_calibration_summary_valid():
    y=np.array([0,0,0,1,0,1,1,1,1,0]);p=np.array([.1,.2,.3,.8,.4,.6,.9,.7,.8,.2]);d=calibration_diagnostics(y,p)
    assert 0<=d['brier']<=1 and 0<=d['ece']<=1
    assert sum(b['n'] for b in d['reliability'])==len(y)


def test_utility_identity():
    d=scores();cfg=RevisionScenario();u,v,c=utilities(d,cfg)
    p=d.churn_probability.to_numpy();e=d.economic_value.to_numpy()
    assert np.allclose(u,p*cfg.retention_success*e-cfg.offer_cost)
    assert np.all(c>=cfg.human_review_cost)


def test_milp_properties():
    d=scores();cfg=RevisionScenario();cand=build_candidates(d,cfg)
    budget=len(d)*cfg.offer_cost*cfg.budget_fraction
    cap=int(np.floor(cfg.review_capacity_fraction*len(d)))
    sel=solve(cand,len(d),budget,'utility',max_reviews=cap)
    x=validate_selection(d,sel,cfg,budget=budget,max_reviews=cap)
    assert x['within_budget']
    ev=evaluate(d,sel,cfg,budget,max(1.,float(sel.strategic_mass.sum())))
    assert ev['offers']+ev['reviews']==len(sel)


def test_validator_rejects_double_action():
    d=scores();cfg=RevisionScenario();sel=pd.DataFrame({'row_index':[2,2],'action':['Retention Offer','Human Review'],'expected_cost':[.05,.02],'expected_utility':[.1,.1],'strategic_mass':[.1,.1]})
    with pytest.raises(AssertionError):validate_selection(d,sel,cfg)


def test_fixed_actions_bootstrap_is_reproducible():
    y=np.array([0,1]*20);q=np.array([.2,.9]*20);a=np.array(['No Action','Retention Offer']*20);b=np.array(['No Action']*40)
    x=bootstrap_fixed_actions(y,q,a,b,n_boot=25,seed=3)
    assert x==bootstrap_fixed_actions(y,q,a,b,n_boot=25,seed=3)
