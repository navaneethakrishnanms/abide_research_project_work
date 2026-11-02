# Dynamic-data inventory -- 600 development subjects only

Read-only. Only the development subjects' time-series files were opened; holdout subjects' files and the pooled `ts_cc200.npy` were not. Per-subject details: `dynamic_data_inventory.csv`; sliding-window cost table: `dynamic_cost_estimates.csv`; machine-readable summary: `dynamic_inventory_summary.json`.

## Availability and alignment
- **Files:** `*_rois_aal.1D` (116 ROIs) and `*_rois_cc200.1D` (200 ROIs) exist for **all 600** development subjects (600/600 each). They are the only derivatives present; there are no per-frame framewise-displacement or confound files anywhere under `abide_pcp`.
- **Alignment:** 600 unique IDs match the manifest; subject-to-fold assignment (120 per fold, 19 sites) is stored in the same manifest; time-series length is identical across the two atlases for every subject.
- **Length (TRs):** min 116, median 176, mean 196.4, max 316. None < 100; 82 subjects < 120. Length varies by site, and the repetition time (seconds per TR) is not recorded in these files, so a window defined in TRs covers different durations across sites.
- **Missing values:** none (0 NaN) in either atlas.
- **Constant (zero) ROIs:** AAL-116: 40 subjects, 113 ROI instances; CC200: 12 subjects, 42 ROI instances. These make some correlations undefined in every window that contains them.
- **Motion metadata:** only **subject-level mean FD** (`func_mean_fd`, finite for all 600, range 0.016-0.200 after the FD <= 0.2 filter). No per-frame FD, so volume scrubbing/censoring and window-level motion regressors are impossible with current files.

## Storage and compute for a small dynamic-connectivity experiment
| Window / step (TRs) | Atlas | Windows per subject (min / mean / max) | Total windows | All windows stored as float32 | Per-subject summary features |
|---|---|---|---|---|---|
| 50 / 10 | AAL-116 | 7 / 15.1 / 27 | 9,033 | 0.24 GB | 13,340 (variance + mean absolute change per edge) |
| 50 / 10 | CC200 | 7 / 15.1 / 27 | 9,033 | 0.72 GB | 39,800 |
| 60 / 20 | AAL-116 | 3 / 7.1 / 13 | 4,270 | 0.11 GB | 13,340 |
| 40 / 10 | AAL-116 | 8 / 16.1 / 28 | 9,633 | 0.26 GB | 13,340 |

- Raw time series on disk: 135 MB (AAL-116) and 236 MB (CC200).
- Computing the windowed correlations for AAL-116 is a few minutes at most on this machine (about 9,000 small correlation matrices); CC200 is about 3x larger. Summaries can be computed per subject with no cross-subject statistics, so they leak nothing across folds. Any variance-type summary should be treated as a nuisance-sensitive feature (short windows are noisy, and motion inflates variability).
- Limitations: heterogeneous scan length and unknown TR; constant ROIs; mean FD only; window parameters would be fixed in advance.
