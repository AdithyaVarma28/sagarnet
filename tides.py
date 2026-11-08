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


# Hours from 1970-01-01 00:00 UTC to the J2000 epoch (2000-01-01 12:00).
_J2000_H = 262980.0


def node_longitude_deg(t_hours: np.ndarray) -> np.ndarray:
    """Mean longitude of the Moon's ascending node N(t) [deg] (18.61-year cycle)."""
    return 125.0445479 - 0.0529537648 * (t_hours - _J2000_H) / 24.0


def _base_nodal(N_deg: np.ndarray) -> dict:
    """Nodal amplitude factor f and phase u [rad] of the base constituents (Schureman / Pugh)."""
    N = np.radians(N_deg)
    c1, c2, c3 = np.cos(N), np.cos(2 * N), np.cos(3 * N)
    s1, s2, s3 = np.sin(N), np.sin(2 * N), np.sin(3 * N)
    d = np.radians
    return {
        "M2": (1.0004 - 0.0373 * c1 + 0.0002 * c2, d(-2.14 * s1)),
        "K1": (1.0060 + 0.1150 * c1 - 0.0088 * c2 + 0.0006 * c3, d(-8.86 * s1 + 0.68 * s2 - 0.07 * s3)),
        "O1": (1.0089 + 0.1871 * c1 - 0.0147 * c2 + 0.0014 * c3, d(10.80 * s1 - 1.34 * s2 + 0.19 * s3)),
        "K2": (1.0241 + 0.2863 * c1 + 0.0083 * c2 - 0.0015 * c3, d(-17.74 * s1 + 0.68 * s2 - 0.04 * s3)),
        "Mf": (1.0429 + 0.4135 * c1 - 0.0040 * c2, d(-23.74 * s1 + 2.68 * s2 - 0.38 * s3)),
        "Mm": (1.0000 - 0.1300 * c1 + 0.0013 * c2, np.zeros_like(N)),
        "J1": (1.0129 + 0.1676 * c1 - 0.0170 * c2 + 0.0016 * c3, d(-12.94 * s1 + 1.34 * s2 - 0.19 * s3)),
        "OO1": (1.1027 + 0.6504 * c1 + 0.0317 * c2 - 0.0014 * c3, d(-36.68 * s1 + 4.02 * s2 - 0.57 * s3)),
    }


def _norm(constituents: dict) -> dict:
    """Accept {name: period} (old style) or {name: (period, nodal_recipe)}."""
    return {k: (v, {}) if np.isscalar(v) else (float(v[0]), dict(v[1])) for k, v in constituents.items()}


def _design(t_hours: np.ndarray, constituents: dict, t0: float, names=None, nodal: bool = True) -> np.ndarray:
    cons = _norm(constituents)
    names = list(cons) if names is None else names
    tt = t_hours - t0
    cols = [np.ones_like(tt), tt / _YEAR_H]          # mean + linear trend (sea-level rise / drift)
    base = _base_nodal(node_longitude_deg(t_hours)) if nodal else {}
    for k in names:
        period, recipe = cons[k]
        ang = 2.0 * np.pi * tt / period
        f = np.ones_like(tt)
        if nodal:
            for b, p in recipe.items():
                fb, ub = base[b]
                f = f * fb ** abs(p)
                ang = ang + p * ub
        cols.append(f * np.cos(ang))
        cols.append(f * np.sin(ang))
    return np.stack(cols, axis=1)


def resolvable(constituents: dict, span_hours: float, rayleigh: float = 1.0) -> list:
    """Constituents (in priority order) separable from each other and from the mean over the record."""
    kept, freqs = [], []
    for k, (period, _) in _norm(constituents).items():
        fr = 1.0 / period
        if fr * span_hours < rayleigh:
            continue
        if any(abs(fr - g) * span_hours < rayleigh for g in freqs):
            continue
        kept.append(k)
        freqs.append(fr)
    return kept


def fit_tide(t_hours: np.ndarray, y: np.ndarray, valid: np.ndarray, constituents: dict,
             t0: float, robust_iters: int = 2, min_points: int = 24 * 60, nodal: bool = True):
    """Robust least-squares harmonic fit with nodal corrections.

    Returns a fit dict {"coef", "names", "nodal"} or None if there are too few points.
    Constituents the record is too short to resolve are left out (Rayleigh criterion).
    """
    idx = np.flatnonzero(valid & np.isfinite(y))
    if idx.size < min_points:
        return None
    names = resolvable(constituents, float(t_hours[idx[-1]] - t_hours[idx[0]]))
    A = _design(t_hours[idx], constituents, t0, names, nodal)
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
    return {"coef": coef, "names": names, "nodal": nodal}


def predict_tide(t_hours: np.ndarray, fit, constituents: dict, t0: float) -> np.ndarray:
    if not isinstance(fit, dict):                     # bare coefficient vector, all constituents, no nodal
        fit = {"coef": fit, "names": None, "nodal": False}
    return (_design(t_hours, constituents, t0, fit["names"], fit["nodal"]) @ fit["coef"]).astype(np.float32)
