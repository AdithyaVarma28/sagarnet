"""Pinball + physics-guided regularisers (spec 3.3), all in residual space.

Everything is in normalised residual units (metres / resid_scale) so the terms
are comparable. The inverted-barometer quantity comes from ERA5 msl in Pa:
    eta_IB = -(msl - msl_ref) / (rho_w * g)

Changes vs v1 (motivated by the Stage-4 ablation where a fixed IB weight helped
+6 h but hurt +12..+72 h):
* Physics weights decay with lead time: w_h = exp(-h / tau_h).
* IB is applied mostly where there is NO label (held-out / masked / missing
  nodes) - it acts as a semi-supervised prior for ungauged coasts instead of
  fighting the data at gauged nodes.
* Losses use Huber, not MSE, so cyclone extremes do not dominate.
* Physics weights are ramped in after a warm-up (set by train.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pinball_loss(q, y, w, quantiles):
    """q [B,N,H,Q], y [B,N,H], w [B,N,H] (0/1 or weights)."""
    tau = torch.as_tensor(quantiles, device=q.device, dtype=q.dtype)
    diff = y.unsqueeze(-1) - q
    l = torch.maximum(tau * diff, (tau - 1) * diff).mean(-1)
    return (l * w).sum() / w.sum().clamp_min(1.0)


class SagarLoss(nn.Module):
    def __init__(self, quantiles, horizons, lam_ib=0.02, lam_grad=0.01, lam_mass=0.002,
                 decay_h=18.0, labelled_ib_weight=0.1, huber_delta=1.0):
        super().__init__()
        self.quantiles = tuple(quantiles)
        self.lam_ib, self.lam_grad, self.lam_mass = lam_ib, lam_grad, lam_mass
        self.labelled_ib_weight = labelled_ib_weight
        self.delta = huber_delta
        h = torch.as_tensor(horizons, dtype=torch.float32)
        self.register_buffer("hw", torch.exp(-h / decay_h))
        self.register_buffer("dh", (h[1:] - h[:-1]))

    def _huber(self, x):
        return F.huber_loss(x, torch.zeros_like(x), delta=self.delta, reduction="none")

    def forward(self, q, y, w, ib, edge_index, edge_weight, phys_scale: float = 1.0):
        """
        q  [B,N,H,3] quantiles; y [B,N,H] target; w [B,N,H] label weight (0 where unlabelled)
        ib [B,N,H] inverted-barometer residual at target time (normalised)
        """
        parts = {"pinball": pinball_loss(q, y, w, self.quantiles)}
        total = parts["pinball"]
        if phys_scale <= 0:
            return total, {k: float(v) for k, v in parts.items()}

        q50 = q[..., 1]
        hw = self.hw.view(1, 1, -1)

        if self.lam_ib > 0:
            node_w = torch.where(w > 0, self.labelled_ib_weight, 1.0)
            l_ib = (self._huber(q50 - ib) * node_w * hw).mean()
            parts["ib"] = l_ib
            total = total + phys_scale * self.lam_ib * l_ib

        src, dst = edge_index
        ew = edge_weight.view(1, -1, 1)
        if self.lam_grad > 0:
            # spatial sea-surface slope along marine edges should follow the pressure slope
            d_eta = q50[:, src] - q50[:, dst]
            d_ib = ib[:, src] - ib[:, dst]
            l_grad = (self._huber(d_eta - d_ib) * ew * hw).sum() / (ew.sum() * q50.size(0) * q50.size(2))
            parts["grad"] = l_grad
            total = total + phys_scale * self.lam_grad * l_grad

        if self.lam_mass > 0 and q50.size(2) > 1:
            # continuity proxy: rate of change of surface height should not diverge sharply
            # between connected coastal nodes (graph-Laplacian smoothness of d(eta)/dt)
            rate = (q50[..., 1:] - q50[..., :-1]) / self.dh.view(1, 1, -1)
            div = rate[:, src] - rate[:, dst]
            l_mass = (div.pow(2) * ew).sum() / (ew.sum() * rate.size(0) * rate.size(2))
            parts["mass"] = l_mass
            total = total + phys_scale * self.lam_mass * l_mass

        return total, {k: float(v) for k, v in parts.items()}
