# Experiment A reproducibility and decision-claim register

## Regenerated baseline
- Code commit: `ce95115e2045bc42da7c405ff2e9ba2312f4589f`
- Source workbook: `Telco_customer_churn.xlsx` (SHA-256 recorded in manifest)
- Command used: `python main.py --data ~/Downloads/Telco_customer_churn.xlsx --output results/experiment_a_frozen --mode paper --folds 5 --repeats 5 --seed 20261031 --bootstrap 1000 --threads 4`
- Original baseline rerun may not be bit-identical to earlier historical files. Software versions and exact scenario assumptions are recorded in `results/experiment_a_frozen/experiment_manifest.json`.

## Observed run results (user terminal log)
- Ensemble AUROC 0.860 ± 0.001; PR-AUC 0.681 ± 0.002; Brier 0.130 ± 0.000; ECE 0.012 ± 0.002.
- Under the regenerated baseline, KGDI and Uncertainty Review generated **identical assignments**. Both reported intervention 24.5%, churn reach 56.5%, high-value churn reach 51.8%, expected selected cost $140,842.37, and outcome-anchored net-benefit proxy $59,217.17.
- Observed KGDI minus Uncertainty Review net-benefit proxy $0; paired policy-rerun bootstrap CI [-$161.61, $152.39] using 1,000 resamples. No observed incremental superiority in this scenario.
- Note: monetary benefit is an **assumption-based proxy**, not observed causal retention impact.

## External validity and labeling
- Iranian Churn: UCI external validation dataset, `Customer Churn.csv`; customer value and charge amount are normalized per-cohort decision inputs.
- Cell2Cell: original labeled training CSV from an Empulse-described public GitHub mirror; `TotalRecurringCharge` is a revenue proxy rather than CLTV.
- Do not compare normalized utility across datasets as if they are observed currencies.
- Review assignment reach and sensitivity-adjusted expected identification are different concepts. Neither equals causal retention uplift.
- Reviewer 2/3 validation results are **not yet completed**. The quick-run flag is not for publication.

## Git hygiene
Commit only this report, the manifest, and the directory README. Store the datasets and generated customer predictions privately outside Git.
