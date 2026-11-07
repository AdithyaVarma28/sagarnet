"""Small helpers shared by train.py and evaluate.py."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_graph(path: str, station_names: list[str]) -> dict:
    g = torch.load(path, map_location="cpu", weights_only=False)
    if "station_names" not in g:          # v1 graph file (node_names / 2 edge features)
        raise ValueError(f"{path} is a v1 graph (no bearing features / land audit). "
                         "Rebuild it with the v2 construct_graph.py.")
    gn = list(g["station_names"])
    if gn != list(station_names):
        missing = sorted(set(station_names) - set(gn))
        extra = sorted(set(gn) - set(station_names))
        raise ValueError("Graph/dataset station order mismatch. Rebuild the graph with construct_graph.py "
                         f"on the same CSV. missing in graph={missing} extra in graph={extra}")
    return g


def to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def mask_history(batch: dict, drop: torch.Tensor):
    """Hide the observed-residual channels (last 2 input features) for nodes where drop[B,N] is True."""
    keep = (~drop).to(batch["x"].dtype)
    batch["x"][..., -2:] = batch["x"][..., -2:] * keep[..., None, None]
    batch["last"] = batch["last"] * keep
    batch["last_mask"] = batch["last_mask"] * keep
    return batch


def random_node_mask(B: int, N: int, p: float, protect: torch.Tensor, device, generator=None) -> torch.Tensor:
    """Bernoulli(p) node mask over nodes where protect is False; guarantees >=1 masked node per sample."""
    r = torch.rand(B, N, generator=generator, device="cpu").to(device)
    cand = ~protect.view(1, N).expand(B, N)
    drop = (r < p) & cand
    if p > 0:
        none = drop.sum(1) == 0
        if none.any():
            rr = r.masked_fill(~cand, 2.0)
            drop[none, rr[none].argmin(1)] = True
    return drop
