"""Side-effect-free external audit helpers; does not replace frozen Experiment A."""
from __future__ import annotations
import json
import hashlib
import platform
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression


def check_probabilities(p, y):
    p=np.asarray(p,dtype=float); y=np.asarray(y)
    if p.ndim!=1 or y.ndim!=1 or len(p)!=len(y) or len(p)==0:
        raise ValueError('Invalid probability and target dimensions')
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Invalid probability values')
    if pd.isna(y).any() or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Target must be binary and nonmissing')
    return p,y.astype(int)


def calibration_diagnostics(y,p,bins=10):
    p,y=check_probabilities(p,y)
    if len(np.unique(y))<2:
        raise ValueError('Requires both outcome classes')
    clipped=np.clip(p,1e-8,1-1e-8)
    logits=np.log(clipped/(1-clipped))
    lr=LogisticRegression(C=1e6,max_iter=3000,solver='lbfgs').fit(logits.reshape(-1,1),y)
    edges=np.linspace(0,1,bins+1)
    idx=np.minimum(np.searchsorted(edges,p,side='right')-1,bins-1)
    reliability=[]
    ece=0.
    for k in range(bins):
        take=idx==k
        if not take.any(): continue
        actual=float(y[take].mean()); mean=float(p[take].mean())
        ece+=float(take.mean())*abs(actual-mean)
        reliability.append({'bin':k,'n':int(take.sum()),'predicted':mean,'observed':actual})
    return {'n':len(y),'auroc':float(roc_auc_score(y,p)),
        'pr_auc':float(average_precision_score(y,p)),
        'brier':float(brier_score_loss(y,p)),
        'log_loss':float(log_loss(y,clipped)),
        'ece':ece,'calibration_intercept':float(lr.intercept_[0]),
        'calibration_slope':float(lr.coef_[0][0]), 'reliability':reliability}


def validate_selection(scores,selected,cfg,*,budget=None,max_reviews=None,minimum_strategic=None):
    n=len(scores)
    if budget is None: budget=n*cfg.offer_cost*cfg.budget_fraction
    required={'row_index','action','expected_cost','expected_utility','strategic_mass'}
    if not required.issubset(selected.columns): raise ValueError(f'Missing selection fields: {required-set(selected.columns)}')
    ids=selected.row_index.to_numpy()
    if len(set(ids))!=len(ids) or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any() or (ids>=n).any():
        raise AssertionError('Invalid or repeated row index')
    if not selected.action.isin(['Retention Offer','Human Review']).all():raise AssertionError('Unexpected action')
    for col in ['expected_cost','expected_utility','strategic_mass']:
        if not np.isfinite(selected[col].astype(float).to_numpy()).all():raise AssertionError(f'Invalid {col}')
    cost=float(selected.expected_cost.sum())
    if (selected.expected_cost<0).any() or cost>budget+1e-7*max(1,budget):raise AssertionError('Invalid budget')
    if max_reviews is not None and (selected.action=='Human Review').sum()>max_reviews:raise AssertionError('Exceeded review capacity')
    if minimum_strategic is not None and float(selected.strategic_mass.sum())+1e-7*max(1,minimum_strategic)<minimum_strategic:raise AssertionError('Missing strategic target')
    return {'n_selected':len(ids),'budget':float(budget),'cost':cost,'within_budget':True}


def expected_action_values(scores,actions,cfg):
    from review_revision.optimizer import utilities
    offer,review,review_cost=utilities(scores,cfg)
    actions=np.asarray(actions)
    if len(actions)!=len(scores):raise ValueError('Action length mismatch')
    if not np.isin(actions,['No Action','Retention Offer','Human Review']).all():raise ValueError('Unknown action')
    util=float(np.where(actions=='Retention Offer',offer,np.where(actions=='Human Review',review,0)).sum())
    cost=float(np.where(actions=='Retention Offer',cfg.offer_cost,np.where(actions=='Human Review',review_cost,0)).sum())
    return {'reference_expected_utility':util,'reference_expected_cost':cost}


def bootstrap_fixed_actions(y,q,actions_a,actions_b,n_boot=1000,seed=20261031):
    """Descriptive conditional CI only; fixed policies; does NOT substitute rerouted bootstrap."""
    y=np.asarray(y,int);q=np.asarray(q,float);a=np.asarray(actions_a);b=np.asarray(actions_b)
    if not(len(y)==len(q)==len(a)==len(b)):raise ValueError('Length mismatch')
    if n_boot<20:raise ValueError('Need >=20 bootstrap replicates')
    rng=np.random.default_rng(seed);differences=[]
    for _ in range(n_boot):
        ix=rng.integers(0,len(y),len(y)); mask=(y[ix]==1)&(q[ix]>=.75)
        if not mask.any():continue
        differences.append(float(np.mean(a[ix][mask]!='No Action')-np.mean(b[ix][mask]!='No Action')))
    if not differences:raise ValueError('No eligible observed churn events')
    return {'delta_reach':float(np.mean(differences)),'ci95':list(map(float,np.quantile(differences,[.025,.975]))), 'n_valid':len(differences), 'interpretation':'Fixed-actions conditional descriptive CI, not refit or policy rerouting'}


def emit_environment_manifest(paths,out):
    files={}
    for path in paths:
        p=Path(path)
        if not p.is_file():raise FileNotFoundError(path)
        files[str(p.name)]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}
    import importlib.metadata as md
    libs=['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','openpyxl']
    versions={x:(md.version(x) if any(d.metadata['Name'].lower()==x for d in md.distributions()) else 'not-installed') for x in libs}
    obj={'python':platform.python_version(),'platform':platform.platform(),'versions':versions,'files':files}
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    Path(out).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return obj
