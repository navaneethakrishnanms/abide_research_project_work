# Integrity checks -- representation audit

| Check | Result | Evidence |
|---|---|---|
| Development subjects only (600) | PASS (with disclosure) | Manifest-defined development IDs; time-series files opened only for those IDs (600/600 found). The cohort CSV and `Phenotypic_V1_0b.csv` were read into memory in full and filtered to development IDs immediately, before any analysis; no non-development row was inspected, printed or used. |
| No access to holdout list, labels, features, predictions or the restricted ID file | PASS | Script contains no reference to them except the docstring. |
| CC200 not used | PASS | Only AAL-116 files and the cached AAL-116 matrix. (The CC200 mentions in the inventory are historical results from earlier reports.) |
| No model trained | PASS | Only descriptive statistics (FFT power, autocorrelation, split-half reliability, per-edge variance shares). Diagnosis labels of development subjects entered only the per-edge association magnitudes, reported descriptively and not used to select anything. |
| No outer-fold scores used to choose a representation | PASS | No predictions were read in this audit; the inventory quotes previously reported results. |
| Preprocessing claims | PASS | Each claim is marked VERIFIED, CONSISTENT (data only) or NOT VERIFIABLE LOCALLY in `preprocessing_inventory.csv`; nothing is asserted without a script, argument or measurement. |
| Baseline reported accurately | PASS | 64.2% +/- 4.3 accuracy, AUC 0.694 +/- 0.037 (five-fold development CV). |
| Previously hashed files unchanged | PASS | SHA-256 of the 311 baseline files: 0 changed; newest timestamps of earlier result folders predate this run. |
| Background processes | PASS | None left running. |

Limitations: the TR and the exact PCP nuisance regressors are unavailable locally; reliability and variance shares are descriptive statistics on the development cohort and ignore model training.
