# Regenerated IBM Experiment A — provenance

This directory tracks a regenerated IBM Experiment A run. The raw files `SUPPLEMENT_customer_level_oof_decisions.csv` and `run_config.json` are **not committed**; obtain them from the researcher's locally archived `experiment_a_inputs.zip`. Verify their SHA-256 hashes against `experiment_manifest.json` before executing the revision.

**Important:** This is a regenerated run under the environment recorded in the manifest; it is not assumed bit-identical to an earlier publication run. The observed KGDI and Uncertainty Review policies selected identical actions under this configuration, with observed incremental benefit zero. Avoid claims of positive incremental benefit.

The other two datasets are external data: UCI Iranian Churn (`Customer Churn.csv`) and Cell2Cell (`cell2celltrain.csv`). Check exact input file hashes against the manifest. The Cell2Cell source has 51,047 rows, including 156 missing MonthlyRevenue values; filtering may remove additional rows depending on pipeline rules.

Input check (from repo root):

```bash
python -m pytest -q
python preflight.py
```

**Do not mistake test success for completed dataset-level verification or end-to-end reproducibility.**
