#!/usr/bin/env python3
"""Read-only pre/post run checks. Does not fit models, alter data, or modify claims."""
import argparse, hashlib, json, math, re
from pathlib import Path
import pandas as pd
import numpy as np

REQUIRED_IBM = ['CustomerID','Churn Value','churn_probability','predictive_entropy','CLTV','Monthly Charges','cltv_percentile','annual_margin_value']
SHA_FIELDS = {'ibm_oof':'SUPPLEMENT_customer_level_oof_decisions.csv','run_config':'run_config.json','cell2cell':'cell2celltrain.csv'}

def checksum(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def require(c,msg):
    if not c: raise ValueError(msg)

def normalized(df, name):
    matches={re.sub('[^a-z0-9]','',str(c).lower()):c for c in df.columns}
    key=re.sub('[^a-z0-9]','',name.lower())
    require(key in matches, f'Missing {name!r}; available: {list(df.columns)}')
    return matches[key]

def numeric(df,col,nullable=False):
    x=pd.to_numeric(df[col],errors='coerce')
    require(nullable or x.notna().all(),f'{col}: missing/non-numeric values')
    require(np.isfinite(x.dropna().astype(float)).all(),f'{col}: nonfinite values')
    return x

def check_binary(s,name):
    raw=s.dropna().astype(str).str.lower().str.strip()
    require(not s.isna().any(),f'{name}: missing target')
    require(raw.isin(['0','1','0.0','1.0','yes','no','true','false']).all(),f'{name}: non-binary targets')
    require(len(set(raw.map(lambda x:x in ['1','1.0','yes','true'])))==2,f'{name}: requires both classes')

def check_input(label, path):
    df=pd.read_csv(path,low_memory=False)
    require(len(df)>0,f'{label}: empty file')
    if label=='ibm_oof':
        require(not (set(REQUIRED_IBM)-set(df.columns)),f'IBM missing: {set(REQUIRED_IBM)-set(df.columns)}')
        require(len(df)==7043,f'IBM row count changed: {len(df)} != 7043; investigate before using frozen results')
        require(df.CustomerID.notna().all() and df.CustomerID.astype(str).str.strip().ne('').all(),'IBM: missing IDs')
        require(not df.CustomerID.duplicated().any(),'IBM: duplicate CustomerID')
        check_binary(df['Churn Value'],'IBM Churn Value')
        for c in ['churn_probability','predictive_entropy','cltv_percentile']:
            x=numeric(df,c);require(x.between(-1e-12,1+1e-12).all(),f'IBM {c} outside [0,1]')
        for c in ['CLTV','Monthly Charges','annual_margin_value']:numeric(df,c)
        if {'action_Uncertainty_Review','action_Proposed_KGDI'}.issubset(df.columns):
            identical=bool((df.action_Uncertainty_Review==df.action_Proposed_KGDI).all())
        else:identical=None
        return {'rows':len(df),'duplicate_ids':0,'kgdi_equals_uncertainty_actions':identical}
    if label=='iranian':
        ch=normalized(df,'Churn');v=normalized(df,'Customer Value');e=normalized(df,'Charge Amount')
        check_binary(df[ch],ch); numeric(df,v);numeric(df,e)
        require(len(df)==3150,f'Iranian row count changed: {len(df)} != 3150')
        return {'rows':len(df),'value_field':v,'economic_field':e}
    if label=='cell2cell':
        ch=normalized(df,'Churn');v=normalized(df,'TotalRecurringCharge');e=normalized(df,'MonthlyRevenue');idc=normalized(df,'CustomerID')
        require(len(df)==51047,f'Cell2Cell row count changed: {len(df)} != 51047')
        require(not df[idc].isna().any() and not df[idc].duplicated().any(),'Cell2Cell duplicate/missing ID')
        check_binary(df[ch],ch)
        mv=int(pd.to_numeric(df[e],errors='coerce').isna().sum())
        require(mv==156,f'Cell2Cell missing MonthlyRevenue count {mv} != 156')
        eligible=df[df[e].notna() & df[v].notna()]
        require(len(eligible)>0,'Cell2Cell no eligible records')
        return {'rows':len(df),'missing_monthly_revenue':mv,'usable_rows_after_required_value_filter':len(eligible),'strategic_field':v}
    raise ValueError(label)

def check_pre(args):
    manifest=json.loads(Path(args.manifest).read_text())
    expected={item['filename']:item['sha256'] for item in manifest['files']}
    paths={'ibm_oof':args.ibm_oof,'run_config':args.run_config,'iranian':args.iranian,'cell2cell':args.cell2cell}
    report={'checksums':{},'datasets':{}}
    for key,p in paths.items():
        p=Path(p)
        require(p.is_file(),f'Missing {key}: {p}')
        value=checksum(p)
        # Manifest stores the Iranian ZIP hash, not necessarily the extracted CSV hash.
        filename=SHA_FIELDS.get(key, 'iranian+churn+dataset.zip')
        is_extracted_iranian=(key=='iranian' and p.name!='iranian+churn+dataset.zip')
        if filename in expected and not is_extracted_iranian:
            require(value==expected[filename],f'{key}: SHA-256 mismatch: {value}; expected {expected[filename]}')
            report['checksums'][key]='MATCH'
        else:report['checksums'][key]={'sha256':value,'note':'source archive hash is in manifest; extracted CSV hash recorded here'}
        if key!='run_config':report['datasets'][key]=check_input(key,p)
        else:
            cfg=json.loads(p.read_text());require(isinstance(cfg,dict),'run_config must be JSON object')
            report['run_config_keys']=list(cfg)
    print(json.dumps(report,indent=2))
    print('PASS: dataset schemas, counts, target values, and frozen file hashes')

def check_post(output):
    root=Path(output)
    require((root/'revision_manifest.json').is_file(),'revision_manifest.json missing: run incomplete')
    man=json.loads((root/'revision_manifest.json').read_text())
    require(not man.get('quick_development_run',False),'QUICK run: not for publication')
    expected_rows={'ibm_telco':7043,'iranian_churn':3150,'cell2cell':50891}
    for dataset in ['ibm_telco','iranian_churn','cell2cell']:
        folder=root/dataset
        scores=pd.read_csv(folder/'customer_scores.csv')
        require(len(scores)==expected_rows[dataset],f'{dataset}: score count changed: {len(scores)} != {expected_rows[dataset]}')
        require('customer_id' in scores.columns and scores.customer_id.notna().all() and not scores.customer_id.duplicated().any(), f'{dataset}: customer IDs missing or duplicated')
        required=['churn_probability','predictive_entropy','strategic_value','economic_value','target']
        require(not(set(required)-set(scores.columns)),f'{dataset}: missing score columns')
        for c in ['churn_probability','strategic_value','economic_value']:
            a=pd.to_numeric(scores[c],errors='coerce')
            require(a.notna().all() and a.between(-1e-9,1+1e-9).all(),f'{dataset}: invalid {c}')
        f=pd.read_csv(folder/'epsilon_frontier.csv')
        require(len(f)>0,f'{dataset}: no frontier')
        for c in ['epsilon','budget_used','strategic_risk_mass','strategic_fraction_of_smax','expected_utility']:
            require(c in f,f'{dataset}: missing frontier metric {c}')
            require(np.isfinite(pd.to_numeric(f[c],errors='coerce')).all(),f'{dataset}: nonfinite {c}')
        runmeta=json.loads((folder/'run_manifest.json').read_text())
        meta=runmeta['frontier_meta']
        step=float(runmeta['scenario']['epsilon_step'])
        require(step>0 and step<=1, f'{dataset}: invalid epsilon step')
        require(len(f)==len(np.arange(0,1.0000001,step)),f'{dataset}: incomplete epsilon frontier')
        require((f.budget_used<=float(meta['budget'])+1e-6*max(1,float(meta['budget']))).all(),f'{dataset}: budget breached')
        require(f.epsilon.min()>=-1e-10 and f.epsilon.max()<=1+1e-10,f'{dataset}: epsilon invalid')
        require((f.strategic_fraction_of_smax + 1e-5 >= f.epsilon).all(),f'{dataset}: epsilon requirement violated')
        print(f'PASS: {dataset}: {len(scores)} scores, {len(f)} frontier points, finite values, budget and epsilon checks')
    for p in ['reviewer3_comparators.csv','reviewer3_bootstrap.json','cross_dataset_predictive_summary.csv','cross_dataset_decision_summary.csv']:
        require((root/p).exists(),f'Missing reviewer output: {p}')
    pred=pd.read_csv(root/'cross_dataset_predictive_summary.csv')
    require(len(pred)==3 and set(pred.dataset)==set(expected_rows), 'Predictive summary missing or duplicated dataset')
    for r in pred.itertuples(index=False):
        require(int(r.n)==expected_rows[r.dataset],f'{r.dataset}: summary customer count mismatch')
    decisions=pd.read_csv(root/'cross_dataset_decision_summary.csv')
    require(set(decisions.dataset)==set(expected_rows),'Decision summary missing dataset')
    require(all(int((decisions.dataset==d).sum())>0 for d in expected_rows),'Decision summary empty dataset')
    require(all(set(['ibm_oof','run_config','iranian','cell2cell']) <= set(man['inputs'])), 'Run manifest missing input provenance')
    for name, entry in man['inputs'].items():
        if name in ['ibm_oof','run_config','iranian','cell2cell']:
            require(re.fullmatch(r'[0-9a-f]{64}',str(entry.get('sha256',''))) is not None,f'{name}: invalid input checksum in run manifest')
    print('PASS: complete full-run output structure (does not establish causal effect or manuscript superiority)')

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='task',required=True)
    a=sub.add_parser('pre')
    for x in ['ibm-oof','run-config','iranian','cell2cell','manifest']:a.add_argument('--'+x,required=True)
    b=sub.add_parser('post');b.add_argument('--output',required=True)
    args=p.parse_args()
    try:
        if args.task=='pre':check_pre(args)
        else:check_post(args.output)
    except (ValueError,OSError,KeyError,AssertionError) as exc:
        print(f'FAIL: {exc}')
        raise SystemExit(1)
if __name__=='__main__':main()
