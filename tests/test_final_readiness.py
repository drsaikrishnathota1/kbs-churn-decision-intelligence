import json
import numpy as np
import pandas as pd
import pytest
from final_readiness import check_input, check_post, require

def test_guard_rejects_bad_condition():
    with pytest.raises(ValueError):require(False,'bad')

def test_ibm_rejects_duplicates(tmp_path):
    df=pd.DataFrame({'CustomerID':['one']*7043,'Churn Value':[0]*7042+[1], 'churn_probability':[.5]*7043,'predictive_entropy':[.5]*7043,'CLTV':[1]*7043,'Monthly Charges':[1]*7043,'cltv_percentile':[.5]*7043,'annual_margin_value':[1]*7043})
    p=tmp_path/'x.csv';df.to_csv(p,index=False)
    with pytest.raises(ValueError,match='duplicate'):check_input('ibm_oof',p)

def test_iranian_rejects_missing_economic(tmp_path):
    p=tmp_path/'x.csv';pd.DataFrame({'Churn':[0,1]*1575,'Customer Value':[3]*3150,'Charge Amount':[None]*3150}).to_csv(p,index=False)
    with pytest.raises(ValueError,match='missing'):check_input('iranian',p)

def test_post_rejects_missing_manifest(tmp_path):
    with pytest.raises(ValueError,match='revision_manifest'):check_post(tmp_path)

def test_post_rejects_quick(tmp_path):
    (tmp_path/'revision_manifest.json').write_text(json.dumps({'quick_development_run':True}))
    with pytest.raises(ValueError,match='QUICK'):check_post(tmp_path)
