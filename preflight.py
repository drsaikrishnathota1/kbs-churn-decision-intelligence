#!/usr/bin/env python3
"""Non-mutating readiness check in the repository root."""
from pathlib import Path
import ast
import sys

REQUIRED=['main.py','review_revision/common.py','review_revision/datasets.py','review_revision/predict.py','review_revision/optimizer.py','review_revision/reviewer3.py','review_revision/run_revision.py','requirements.txt']

def run():
    bad=[]
    for f in REQUIRED:
        p=Path(f)
        if not p.exists():bad.append(f'MISSING: {f}');continue
        try:ast.parse(p.read_text(encoding='utf-8'),filename=f)
        except SyntaxError as e:bad.append(f'SYNTAX: {f}: {e}')
    for f in ['results/experiment_a_frozen/SUPPLEMENT_customer_level_oof_decisions.csv','results/experiment_a_frozen/run_config.json']:
        if not Path(f).exists():print(f'INFO: frozen input not at default path: {f} (may be supplied with CLI flags)')
    if bad:
        for x in bad:print(x,file=sys.stderr)
        return 1
    print('PASS: required sources exist and parse')
    print('NEXT: python -m pytest -q tests/test_revision_invariants.py tests/test_scientific_validation.py')
    print('NEXT: smoke-run using real inputs, inspect output, then full RunPod experiment')
    return 0
if __name__=='__main__':sys.exit(run())
