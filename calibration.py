"""Conformalized Quantile Regression (Romano et al., 2019), per horizon.

Why v1 only reached 74 % instead of 80 %:
  CQR's guarantee assumes calibration and test scores are exchangeable. v1
  calibrated on gauged behaviour but tested an ungauged station with a 2.2 m
  bias, so the score distributions differed. v2 fixes this by calibrating on
  *pseudo-ungauged* scores (training stations with their history masked, see
  evaluate.py) and offers:
    * mode="scaled": normalised scores s / (q90 - q10) -> adaptive widths
    * mondrian=True: separate q_hat for cyclone vs. calm conditions
  Coverage remains approximate under temporal dependence / spatial shift; it is
  an empirical, not a strict, guarantee for time series.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        return 0.0
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


class CQR:
    def __init__(self, alpha: float = 0.2, mode: str = "additive", mondrian: bool = False, min_group: int = 200):
        assert mode in ("additive", "scaled")
        self.alpha, self.mode, self.mondrian, self.min_group = alpha, mode, mondrian, min_group
        self.qhat: dict = {}

    def _score(self, lo, hi, y):
        s = np.maximum(lo - y, y - hi)
        if self.mode == "scaled":
            s = s / np.maximum(hi - lo, 1e-6)
        return s

    def fit(self, lo, hi, y, group=None):
        """lo/hi/y: [n, H] arrays (NaN y ignored). group: [n, H] ints (0 calm, 1 cyclone)."""
        H = lo.shape[1]
        self.qhat = {}
        for h in range(H):
            ok = np.isfinite(y[:, h])
            s = self._score(lo[ok, h], hi[ok, h], y[ok, h])
            self.qhat[(h, -1)] = conformal_quantile(s, self.alpha)
            if self.mondrian and group is not None:
                g = group[ok, h]
                for k in (0, 1):
                    if (g == k).sum() >= self.min_group:
                        self.qhat[(h, k)] = conformal_quantile(s[g == k], self.alpha)
        return self

    def apply(self, lo, hi, group=None):
        lo_c, hi_c = lo.copy(), hi.copy()
        for h in range(lo.shape[1]):
            q = np.full(lo.shape[0], self.qhat[(h, -1)])
            if self.mondrian and group is not None:
                for k in (0, 1):
                    if (h, k) in self.qhat:
                        q[group[:, h] == k] = self.qhat[(h, k)]
            pad = q * (hi[:, h] - lo[:, h]) if self.mode == "scaled" else q
            lo_c[:, h] = lo[:, h] - pad
            hi_c[:, h] = hi[:, h] + pad
        return lo_c, hi_c

    def global_qhat(self) -> list[float]:
        return [self.qhat[(h, -1)] for h in sorted({k[0] for k in self.qhat})]
