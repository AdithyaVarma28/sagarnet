"""Astronomical tide utilities.

Two separate things live here:

1. ``tidal_phase_features`` - the cyclical sin/cos inputs from spec 3.1
   (station-local astronomical time tau = t + lon/15).
2. ``fit_tide`` / ``predict_tide`` - a per-station least-squares harmonic
   analysis. It is fitted on the *training period only* and used to split the
   observed water level into  total = mean + trend + tide + residual.  The
   network forecasts the zero-centred residual (meteorological surge), which
   removes the ~2.2 m chart-datum bias seen in the v1 LOSO results.
"""
from __future__ import annotations

import numpy as np

_YEAR_H = 8766.0


def hours_since_epoch(times) -> np.ndarray:
    """datetime64 array/Index -> float64 hours since 1970-01-01 UTC."""
    t = np.asarray(times, dtype="datetime64[s]").astype(np.int64)
    return t.astype(np.float64) / 3600.0


def tidal_phase_features(t_hours: np.ndarray, lon_deg: np.ndarray, periods: dict) -> np.ndarray:
    """Return [T, N, 2*P] float32 sin/cos features using tau = t + lon/15."""
    tau = t_hours[:, None] + np.asarray(lon_deg, dtype=np.float64)[None, :] / 15.0
    feats = []
    for period in periods.values():
        ang = 2.0 * np.pi * tau / period
        feats.append(np.sin(ang))
        feats.append(np.cos(ang))
    return np.stack(feats, axis=-1).astype(np.float32)


def _design(t_hours: np.ndarray, periods: dict, t0: float) -> np.ndarray:
    tt = t_hours - t0
    cols = [np.ones_like(tt), tt / _YEAR_H]          # mean + linear trend (sea-level rise / drift)
    for period in periods.values():
        ang = 2.0 * np.pi * tt / period
        cols.append(np.cos(ang))
        cols.append(np.sin(ang))
    return np.stack(cols, axis=1)


def fit_tide(t_hours: np.ndarray, y: np.ndarray, valid: np.ndarray, periods: dict,
             t0: float, robust_iters: int = 2, min_points: int = 24 * 60) -> np.ndarray | None:
    """Robust least-squares harmonic fit. Returns coefficient vector or None."""
    idx = np.flatnonzero(valid & np.isfinite(y))
    if idx.size < min_points:
        return None
    A =_design(t_hours[idx], periods, t0)
    yy = y[idx].astype(np.float64)
    keep = np.ones(idx.size, dtype=bool)
    coef = None
    for _ in range(robust_iters + 1):
        coef, *_ = np.linalg.lstsq(A[keep], yy[keep], rcond=None)
        r = yy - A @ coef
        mad = np.median(np.abs(r[keep] - np.median(r[keep]))) * 1.4826 + 1e-6
        # only drop gross outliers (sensor spikes); real surges stay in.
        new_keep = np.abs(r) < max(8.0 * mad, 1.0)
        if new_keep.sum() == keep.sum():
            break
        keep = new_keep
    return coef


def predict_tide(t_hours: np.ndarray, coef: np.ndarray, periods: dict, t0: float) -> np.ndarray:
    return (_design(t_hours, periods, t0) @ coef).astype(np.float32)
