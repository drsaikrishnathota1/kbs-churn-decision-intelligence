from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RevisionScenario:
    offer_cost: float = 0.05
    retention_success: float = 0.35
    human_review_cost: float = 0.0125
    human_sensitivity: float = 0.85
    human_specificity: float = 0.85
    budget_fraction: float = 0.20
    minimum_churn_risk: float = 0.20
    review_entropy_threshold: float = 0.45
    review_capacity_fraction: float = 0.05
    seed: int = 20261031
    folds: int = 5
    repeats: int = 3
    bootstrap: int = 1000
    epsilon_step: float = 0.01


def percentile_rank(values) -> np.ndarray:
    s = pd.Series(values, dtype="float64")
    if s.isna().any():
        raise ValueError("percentile_rank received missing values")
    return s.rank(method="average", pct=True).to_numpy(float)


def binary_entropy(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def scenario_to_dict(cfg: RevisionScenario) -> dict:
    return asdict(cfg)


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def assert_unit_interval(name: str, x) -> None:
    arr = np.asarray(x, dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains non-finite values")
    if (arr < -1e-12).any() or (arr > 1 + 1e-12).any():
        raise RuntimeError(f"{name} must be in [0,1]")
