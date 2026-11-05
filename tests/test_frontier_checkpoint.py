"""Small, fast tests for optimization checkpoint equivalence and invalidation."""
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal
from review_revision.common import RevisionScenario
from review_revision.optimizer import frontier


def toy_scores():
    n = 16
    return pd.DataFrame({
        "target": np.array([0, 1] * (n // 2)),
        "churn_probability": np.linspace(.55, .95, n),
        "predictive_entropy": np.full(n, .60),
        "strategic_value": np.linspace(.20, 1.00, n),
        "economic_value": np.linspace(.60, 1.00, n),
    })


def test_checkpoint_same_result_on_restart(tmp_path):
    scores = toy_scores()
    cfg = RevisionScenario(epsilon_step=0.50)
    first, meta1 = frontier(scores, cfg, checkpoint_dir=tmp_path)
    assert len(first) == 3
    assert (tmp_path / "frontier_checkpoint.json").exists()
    second, meta2 = frontier(scores, cfg, checkpoint_dir=tmp_path)
    assert_frame_equal(first, second, check_exact=True)
    assert meta1 == meta2


def test_changed_scores_reject_old_checkpoint(tmp_path):
    scores = toy_scores()
    cfg = RevisionScenario(epsilon_step=0.50)
    frontier(scores, cfg, checkpoint_dir=tmp_path)
    changed = scores.copy()
    changed.loc[0, "strategic_value"] = .30
    with pytest.raises(RuntimeError, match="Checkpoint identity mismatch"):
        frontier(changed, cfg, checkpoint_dir=tmp_path)
