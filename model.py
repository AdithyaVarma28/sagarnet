"""SAGAR-Net v2 model.

Same five-stage layout as v1 (spec section 3):
    TCN (5 causal dilated blocks, RF = 63 h) -> GRU residual -> static context
    -> sparse coastal GATv2 -> multi-horizon non-crossing quantile heads

What changed and why
--------------------
* Target is the de-tided, de-meaned residual (see sagar_dataset.py), so the
  graph never mixes chart datums -> removes the ~2.2 m LOSO bias.
* Inputs include the observed residual history + availability mask; a learned,
  horizon-dependent persistence skip (alpha_h * last_obs) gives gauged nodes a
  persistence floor the network only has to improve on.
* BatchNorm1d -> GroupNorm(1, C): batch statistics were dominated by gauged
  nodes and shifted when an ungauged node was evaluated alone.
* Stackable GATv2 blocks (default 2 hops) with edge features
  [sim_geo, sim_bath, sin/cos bearing, is_self] and a feed-forward sublayer.
  Implemented in plain PyTorch (same maths as PyG GATv2Conv) so no
  torch-geometric install is needed on Windows.
* One shared horizon head with a learned horizon embedding (more data per
  parameter than five separate heads), fed with stacked future forcing at
  lags FUTURE_LAGS before each target time.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvBlock(nn.Module):
    """Same layout as v1 CausalConv1dBlock (dilated conv -> norm -> ReLU -> dropout -> 1x1 conv -> norm -> ReLU,
    + residual), with GroupNorm instead of BatchNorm and no ReLU on the residual branch output."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm1 = nn.GroupNorm(1, ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.norm2 = nn.GroupNorm(1, ch)

    def forward(self, x):                          # [B, C, L]
        y = self.conv1(F.pad(x, (self.pad, 0)))    # left pad only -> causal
        y = self.drop(F.relu(self.norm1(y)))
        y = self.norm2(self.conv2(y))              # v1 applied ReLU here -> residual could only grow
        return F.relu(x + y)


class EdgeGATv2(nn.Module):
    """GATv2 attention (Brody et al. 2022) with edge features, batched over B graphs that share topology."""

    def __init__(self, in_dim: int, out_dim: int, heads: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.h, self.c = heads, out_dim
        self.lin_l = nn.Linear(in_dim, heads * out_dim)
        self.lin_r = nn.Linear(in_dim, heads * out_dim)
        self.lin_e = nn.Linear(edge_dim, heads * out_dim, bias=False)
        self.att = nn.Parameter(torch.empty(1, 1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.drop = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index, edge_attr):   # x [B,N,D]
        B, N, _ = x.shape
        src, dst = edge_index
        E = src.numel()
        xl = self.lin_l(x).view(B, N, self.h, self.c)
        xr = self.lin_r(x).view(B, N, self.h, self.c)
        e = self.lin_e(edge_attr).view(1, E, self.h, self.c)
        z = F.leaky_relu(xl[:, src] + xr[:, dst] + e, 0.2)
        score = (z * self.att).sum(-1)                                    # [B,E,H]
        idx = dst.view(1, E, 1).expand(B, E, self.h)
        smax = torch.full((B, N, self.h), float("-inf"), device=x.device, dtype=score.dtype)
        smax = smax.scatter_reduce(1, idx, score.detach(), reduce="amax", include_self=True)
        ex = torch.exp(score - smax.gather(1, idx))
        denom = torch.zeros(B, N, self.h, device=x.device, dtype=ex.dtype).index_add_(1, dst, ex)
        alpha = self.drop(ex / (denom.gather(1, idx) + 1e-16))
        msg = xl[:, src] * alpha.unsqueeze(-1)
        out = torch.zeros(B, N, self.h, self.c, device=x.device, dtype=msg.dtype).index_add_(1, dst, msg)
        return out.reshape(B, N, self.h * self.c) + self.bias


class GraphBlock(nn.Module):
    def __init__(self, ch: int, heads: int, edge_dim: int, dropout: float):
        super().__init__()
        assert ch % heads == 0
        self.gat = EdgeGATv2(ch, ch // heads, heads, edge_dim, dropout)
        self.fuse = nn.Linear(2 * ch, ch)
        self.norm1 = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, 2 * ch), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * ch, ch))
        self.norm2 = nn.LayerNorm(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, ei, ea):
        s = F.elu(self.gat(h, ei, ea))
        h = self.norm1(h + self.drop(self.fuse(torch.cat([h, s], -1))))   # LN(Linear([h_temp,h_spat]) + h)
        return self.norm2(h + self.drop(self.ffn(h)))


class SagarNet(nn.Module):
    def __init__(self, in_dim: int, fut_dim: int, static_dim: int, edge_index: torch.Tensor,
                 edge_attr: torch.Tensor, horizons=(6, 12, 24, 48, 72), hidden: int = 64, heads: int = 4,
                 gnn_layers: int = 2, dropout: float = 0.1, dilations=(1, 2, 4, 8, 16), kernel: int = 3):
        super().__init__()
        C = hidden
        self.horizons = list(horizons)
        Hn = len(self.horizons)

        # graph with self-loops; extra is_self edge feature
        n_nodes = int(edge_index.max()) + 1
        loops = torch.arange(n_nodes)
        ea = torch.cat([edge_attr, torch.zeros(edge_attr.size(0), 1)], 1)
        self_attr = torch.zeros(n_nodes, ea.size(1))
        self_attr[:, :2] = 1.0     # sim_geo = sim_bath = 1 to itself
        self_attr[:, -1] = 1.0
        self.register_buffer("edge_index", torch.cat([edge_index, torch.stack([loops, loops])], 1))
        self.register_buffer("edge_attr", torch.cat([ea, self_attr], 0))
        sim = self.edge_attr[:, 0].clone()
        sim[self.edge_attr[:, -1] > 0] = 0.0
        self.register_buffer("edge_weight", sim)     # used by physics losses (no self loops)

        # 1. TCN
        self.inp = nn.Linear(in_dim, C)
        self.tcn = nn.ModuleList([CausalConvBlock(C, d, kernel, dropout) for d in dilations])
        # 2. GRU residual
        self.gru = nn.GRU(C, C, num_layers=1, batch_first=True)
        self.temp_norm = nn.LayerNorm(C)
        # 3. static context
        self.static_mlp = nn.Sequential(nn.LayerNorm(static_dim), nn.Linear(static_dim, C), nn.ReLU(), nn.Linear(C, C))
        # 4. graph
        self.gnn = nn.ModuleList([GraphBlock(C, heads, self.edge_attr.size(1), dropout) for _ in range(gnn_layers)])
        # 5. heads
        self.fut_proj = nn.Sequential(nn.Linear(fut_dim, C), nn.GELU(), nn.Linear(C, C))
        self.h_emb = nn.Embedding(Hn, C // 2)
        self.head = nn.Sequential(nn.Linear(2 * C + C // 2, 2 * C), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(2 * C, C), nn.GELU(), nn.Linear(C, 3))
        init = torch.tensor([math.exp(-h / 24.0) for h in self.horizons]).clamp(1e-3, 1 - 1e-3)
        self.persist_logit = nn.Parameter(torch.log(init / (1 - init)))

    def forward(self, x, fut, static, last=None, last_mask=None):
        """x [B,N,L,Fin], fut [B,N,Hn,Ff], static [N,S] -> quantiles [B,N,Hn,3] (q10,q50,q90)."""
        B, N, L, _ = x.shape
        h = self.inp(x).reshape(B * N, L, -1).transpose(1, 2)
        for blk in self.tcn:
            h = blk(h)
        h_tcn = h.transpose(1, 2)
        g, _ = self.gru(h_tcn)
        h = self.temp_norm(g[:, -1] + h_tcn[:, -1]).view(B, N, -1)
        h = h + self.static_mlp(static).unsqueeze(0)
        for blk in self.gnn:
            h = blk(h, self.edge_index, self.edge_attr)

        Hn = fut.size(2)
        z = torch.cat([h.unsqueeze(2).expand(-1, -1, Hn, -1), self.fut_proj(fut),
                       self.h_emb.weight.view(1, 1, Hn, -1).expand(B, N, -1, -1)], -1)
        raw = self.head(z)
        q50 = raw[..., 0]
        if last is not None:
            skip = (last * last_mask).unsqueeze(-1) * torch.sigmoid(self.persist_logit)
            q50 = q50 + skip
        q10 = q50 - F.softplus(raw[..., 1])
        q90 = q50 + F.softplus(raw[..., 2])
        return torch.stack([q10, q50, q90], -1)
