# ABIDE-I ASD vs typical control: source code and data

Numerical resting-state fMRI connectivity (CC200 atlas, ABIDE-I, CPAC pipeline, band-pass 0.01-0.1 Hz, global signal regression).
Source of the raw time series: public ABIDE Preprocessed Connectomes Project on Amazon S3 (`fcp-indi`), downloaded with nilearn (`code/download_cc200_from_s3.py`).

## data/
| File | Content |
|---|---|
| fc_raw_CC200.npy | 600 subjects x 19,900 Pearson correlations (upper triangle of the 200x200 CC200 matrix), raw (not Fisher-Z) |
| dev_split_manifest.csv | the 600 development subjects: sid, diagnosis, site, y (1 = ASD), outer_fold (0-4) |
| Phenotypic_V1_0b.csv | ABIDE phenotype file (age, sex, site, DX_GROUP, ...) |
| phase2_train_subjects.txt | subject ids of the development/train split |
| fd_filtered_subjects.csv | subjects remaining after the motion (framewise displacement) filter |

## code/
| File | Purpose |
|---|---|
| run_cc200_torch_baseline.py | PyTorch (GPU) logistic regression / MLP on CC200 FC + age + sex, nested hyper-parameter selection |
| run_cc200_combat_ensemble.py | same + in-fold ComBat harmonization + ensemble of the top-3 configurations |
| combat_harmonizer.py | label-free ComBat-style harmonizer used by the ensemble script |
| download_cc200_from_s3.py | downloads the CC200 ROI time series from S3 (nilearn) |

## Run (conda env torch_gpu: torch, scikit-learn, pandas, numpy)
```
python code/run_cc200_torch_baseline.py
python code/run_cc200_combat_ensemble.py
```
Outputs are written to a new `outputs/` folder at run time. All fitting (imputation, scaling, ComBat, feature selection, hyper-parameters)
uses training rows of each fold only; the same 5 outer folds (`outer_fold` column) are used throughout.
