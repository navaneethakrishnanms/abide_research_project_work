"""Input validation and feature construction for the demo model. Pure functions, no Streamlit.

An upload is accepted only if it is an AAL-116 ROI time-series text file in the ABIDE PCP "rois_aal.1D" layout:
a '#'-prefixed tab/space-separated header with the exact 116 ROI codes in the training order, then numeric rows.
What can be verified is the FORMAT (header, shape, finiteness, variance). The upstream preprocessing
(CPAC, band-pass 0.01-0.1 Hz, global signal regression) cannot be verified from numbers alone and is
stated by the user.
"""
from __future__ import annotations

import io

import numpy as np

MIN_TIMEPOINTS = 100
MAX_TIMEPOINTS = 2000
MAX_CONSTANT_ROIS = 20          # the training data itself contained constant ROIs (undefined correlations are imputed)
N_ROI = 116


class InputError(ValueError):
    pass


def parse_timeseries(raw: bytes, expected_header: list[str]) -> np.ndarray:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise InputError("File is not UTF-8 text.") from e
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise InputError("File has no data rows.")
    if not lines[0].lstrip().startswith("#"):
        raise InputError("Missing '#'-prefixed ROI header line. The ROI order cannot be verified, so the file is rejected.")
    header = lines[0].replace("#", " ").split()
    if len(header) != N_ROI:
        raise InputError(f"Header lists {len(header)} ROI codes; the AAL-116 model needs exactly {N_ROI}.")
    if header != expected_header:
        raise InputError("ROI codes or their order differ from the training data (ABIDE PCP rois_aal.1D order).")
    if len(lines) - 1 > MAX_TIMEPOINTS:
        raise InputError(f"More than {MAX_TIMEPOINTS} rows; this is not a single-subject ROI time series.")
    try:
        rows = [[float(v) for v in ln.split()] for ln in lines[1:]]
    except ValueError as e:
        raise InputError("Non-numeric value in the data rows.") from e
    if any(len(r) != N_ROI for r in rows):
        raise InputError(f"Every data row must have exactly {N_ROI} columns.")
    return check_array(np.asarray(rows, dtype=float))


def check_array(ts: np.ndarray) -> np.ndarray:
    if ts.ndim != 2 or ts.shape[1] != N_ROI:
        raise InputError(f"Expected a (timepoints x {N_ROI}) array, got {ts.shape}.")
    if ts.shape[0] < MIN_TIMEPOINTS:
        raise InputError(f"Only {ts.shape[0]} timepoints; at least {MIN_TIMEPOINTS} are required (as in the training cohort).")
    if not np.isfinite(ts).all():
        raise InputError("Time series contains NaN or infinite values.")
    n_const = int((np.ptp(ts, axis=0) == 0).sum())
    if n_const > MAX_CONSTANT_ROIS:
        raise InputError(f"{n_const} ROIs are constant (limit {MAX_CONSTANT_ROIS}); the data look corrupted or not ROI-averaged.")
    return ts


def build_features(ts: np.ndarray) -> np.ndarray:
    """Upper-triangle Pearson correlations, same construction and order as training: np.triu_indices(116, k=1)."""
    iu = np.triu_indices(N_ROI, k=1)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fc = np.corrcoef(ts, rowvar=False)[iu]
    return fc[None, :]
