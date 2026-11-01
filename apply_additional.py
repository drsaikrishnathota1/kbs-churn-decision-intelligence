#!/usr/bin/env python3
"""Additional guarded modifications for reviewer revision checkout; run after apply_hardening.py."""
from pathlib import Path
import shutil


def change(path, old, new):
    f = Path(path)
    if not f.is_file():
        raise FileNotFoundError(f"Run from project root: {path}")
    current = f.read_text(encoding='utf-8')
    if new in current:
        print(f"Already patched: {path}")
        return
    occurrences = current.count(old)
    if occurrences != 1:
        raise RuntimeError(f"Expected unique anchor in {path}, got {occurrences}: {old[:70]}")
    shutil.copy2(f, f.with_suffix(f.suffix + '.pre_v2'))
    f.write_text(current.replace(old, new, 1), encoding='utf-8')
    print(f"Updated: {path}")


def main():
    p='review_revision/optimizer.py'
    # Hardening V1 evaluates reference sensitivity; for reviewer quality ablations pass actual reference scenario.
    change(p,
       '        "expected_utility": utility,\n        "strategic_risk_mass": strategic,',
       '        "expected_utility": utility,\n        "strategic_risk_mass": strategic,\n        "metric_definition": "review assignment is distinct from sensitivity-adjusted identification; not causal retained customers",')
    # Add optional solver diagnostics without altering the frozen optimization configuration.
    change(p,
       '    chosen = res.x > 0.5\n    selected_tmp = candidates.loc[chosen]',
       '    chosen = res.x > 0.5\n    if not np.all(np.isfinite(res.x)) or np.max(np.abs(res.x - chosen.astype(float))) > 1e-5:\n        raise RuntimeError("MILP returned non-integral or non-finite selection")\n    selected_tmp = candidates.loc[chosen]')
    # Required safeguards for revised prediction data integrity.
    p='review_revision/predict.py'
    change(p,
       '    X = X.reset_index(drop=True)\n    y = y.astype(int).reset_index(drop=True)',
       '    if folds < 2 or repeats < 1:\n        raise ValueError("folds must be >=2 and repeats >=1")\n    if len(X) != len(y) or len(X) == 0:\n        raise ValueError("X/y empty or inconsistent")\n    if X.columns.duplicated().any():\n        raise ValueError("Duplicate predictive feature columns")\n    if pd.Series(y).isna().any() or not set(pd.Series(y).unique()).issubset({0, 1}):\n        raise ValueError("Target must be nonmissing binary")\n    if int(pd.Series(y).value_counts().min()) < folds:\n        raise ValueError("Each class must contain at least folds examples")\n    X = X.reset_index(drop=True)\n    y = y.astype(int).reset_index(drop=True)')
    # Guard accidental quick-run publication by marking outputs unmistakably.
    p='review_revision/run_revision.py'
    change(p,
       '    print(f"Revision outputs written to {out}")',
       '    if args.quick:\n        (out / "NOT_FOR_PUBLICATION_QUICK_RUN.txt").write_text(\n            "DEVELOPMENT-ONLY: reduced folds, repeats, epsilon grid and bootstrap. Do not cite these results.\\n",\n            encoding="utf-8",\n        )\n    print(f"Revision outputs written to {out}")')
    print('V2 guarded additions complete. Execute pytest and preflight before RunPod.')

if __name__ == '__main__':
    main()
