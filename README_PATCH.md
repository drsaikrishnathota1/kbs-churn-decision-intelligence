## Peer-review revision experiments

The `review_revision/` package contains reviewer-motivated post-review analyses for manuscript KNOSYS-D-26-21292. The original Experiment A remains immutable.

The revision package adds:

- Reviewer 3 fixed cost-benefit-matrix and `p × strategic-value` comparators on frozen IBM OOF predictions;
- a continuous strategic-risk objective `p_i q_i` that removes the hard 75th-percentile CLTV gate from revised optimization;
- an exact epsilon-constraint decision allocator;
- binding-knowledge ablations;
- external validation adapters for UCI Iranian Churn and Cell2Cell;
- invariant tests and reproducibility manifests.

See `REVISION_PROTOCOL.md` before running any definitive analysis and `REVISION_RUNPOD.md` for execution commands.
