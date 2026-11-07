"""Build the audited sparse marine graph (spec 3.2) for the 34-station dataset.

Changes vs v1
-------------
* Reads ``gebco_mean_5km_m`` (fallback gebco_nearest_m -> gebco_min_5km_m ->
  elevation_m) instead of the removed ``elevation_m`` column.
* Reads the 3M-row CSV in chunks, only the station columns.
* Over-land pruning is explicit: each candidate edge is sampled along its great
  circle and rejected if it crosses a contiguous land stretch longer than
  ``--max-land-run-km`` (small islands are tolerated, peninsulas are not).
* Edge attributes: [sim_geo, sim_bath, sin(bearing), cos(bearing)]. Bearing lets
  attention learn direction-dependent surge propagation.
* Writes an audit CSV of every edge so you can verify no cross-land links.

Usage
-----
    .\\.venv\\Scripts\\python.exe construct_graph.py \
        --dataset combined34/34_stations_combined_with_IBTrACs.csv \
        --output static_graph.pt --k-neighbors 4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_DATASET, DEFAULT_GRAPH, ELEVATION_ALIASES, GEBCO_COLS

EARTH_R_KM = 6371.0088


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat), np.radians(lon)
    dla = la[:, None] - la[None, :]
    dlo = lo[:, None] - lo[None, :]
    a = np.sin(dla / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None, :]) * np.sin(dlo / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def initial_bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)  # radians, 0 = north, clockwise positive


def great_circle_points(lat1, lon1, lat2, lon2, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n points along the great circle (slerp on unit vectors)."""
    def to_vec(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    a, b = to_vec(lat1, lon1), to_vec(lat2, lon2)
    omega = np.arccos(np.clip(a @ b, -1, 1))
    f = np.linspace(0, 1, n)
    if omega < 1e-9:
        pts = np.repeat(a[None], n, 0)
    else:
        pts = (np.sin((1 - f) * omega)[:, None] * a + np.sin(f * omega)[:, None] * b) / np.sin(omega)
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lat, lon


class LandChecker:
    def __init__(self, mode: str, max_land_run_km: float, coast_buffer_km: float, step_km: float):
        self.max_run = max_land_run_km
        self.buffer = coast_buffer_km
        self.step = step_km
        self.globe = None
        if mode != "none":
            try:
                from global_land_mask import globe  # pip install global-land-mask
                self.globe = globe
            except ImportError as e:
                if mode == "require":
                    raise SystemExit(
                        "global-land-mask is required for over-land pruning: "
                        "pip install global-land-mask  (or pass --land-check none)") from e
                print("[warn] global-land-mask not installed -> over-land pruning DISABLED. "
                      "pip install global-land-mask")

    @property
    def enabled(self) -> bool:
        return self.globe is not None

    def longest_land_run_km(self, lat1, lon1, lat2, lon2, dist_km) -> float:
        if not self.enabled:
            return 0.0
        n = max(int(dist_km / self.step) + 1, 8)
        la, lo = great_circle_points(lat1, lon1, lat2, lon2, n)
        along = np.linspace(0, dist_km, n)
        inner = (along > self.buffer) & (along < dist_km - self.buffer)   # stations sit on the coast
        land = self.globe.is_land(la, lo) & inner
        run = best = 0
        for v in land:
            run = run + 1 if v else 0
            best = max(best, run)
        return best * dist_km / max(n - 1, 1)


# --------------------------------------------------------------------------- #
# Station table
# --------------------------------------------------------------------------- #
def load_stations(csv_path: str, chunksize: int = 500_000) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0).columns
    elev_col = next((c for c in ELEVATION_ALIASES if c in header), None)
    if elev_col is None:
        raise KeyError(f"No elevation column found; expected one of {ELEVATION_ALIASES}")
    extra = [c for c in GEBCO_COLS if c in header and c != elev_col]
    usecols = ["station_name", "station_latitude", "station_longitude", elev_col] + extra
    parts = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        parts.append(chunk.groupby("station_name", sort=False).median(numeric_only=True))
    st = pd.concat(parts).groupby(level=0).median()
    st = st.rename(columns={"station_latitude": "lat", "station_longitude": "lon", elev_col: "elevation"})
    st.attrs["elevation_source"] = elev_col
    return st.sort_index()  # alphabetical -> invariant 1:1 with dataset pivots


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #
def build_graph(st: pd.DataFrame, k: int, max_dist_km: float, sigma_km: float, gamma_bath: float,
                checker: LandChecker, fallback_dist_km: float):
    names = list(st.index)
    lat, lon, z = st["lat"].to_numpy(), st["lon"].to_numpy(), st["elevation"].to_numpy()
    n = len(names)
    D = haversine_matrix(lat, lon)

    marine_cache: dict[tuple[int, int], float] = {}

    def land_run(i, j):
        key = (min(i, j), max(i, j))
        if key not in marine_cache:
            marine_cache[key] = checker.longest_land_run_km(lat[i], lon[i], lat[j], lon[j], D[i, j])
        return marine_cache[key]

    chosen: set[tuple[int, int]] = set()
    rejected = []
    for i in range(n):
        order = [j for j in np.argsort(D[i]) if j != i]
        picked = 0
        for j in order:
            if D[i, j] > max_dist_km or picked >= k:
                break
            lr = land_run(i, j)
            if lr > checker.max_run:
                rejected.append((names[i], names[j], D[i, j], lr))
                continue
            chosen.add((i, j)); chosen.add((j, i))   # reciprocal
            picked += 1
        if picked == 0:   # isolated: allow one longer marine link
            for j in order:
                if D[i, j] > fallback_dist_km:
                    break
                if land_run(i, j) <= checker.max_run:
                    print(f"[warn] {names[i]} isolated within {max_dist_km:.0f} km; "
                          f"linked to {names[j]} at {D[i, j]:.0f} km")
                    chosen.add((i, j)); chosen.add((j, i))
                    break

    edges = sorted(chosen)
    src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    d = D[src, dst]
    sim_geo = np.exp(-(d ** 2) / (2 * sigma_km ** 2))
    sim_bath = np.exp(-np.abs(z[src] - z[dst]) / gamma_bath)
    bearing = np.array([initial_bearing(lat[a], lon[a], lat[b], lon[b]) for a, b in edges])
    edge_attr = np.stack([sim_geo, sim_bath, np.sin(bearing), np.cos(bearing)], 1)
    return names, src, dst, d, edge_attr, rejected


def connected_components(n: int, src, dst) -> list[list[int]]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in zip(src, dst):
        parent[find(a)] = find(b)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output", default=DEFAULT_GRAPH)
    ap.add_argument("--k-neighbors", type=int, default=4)
    ap.add_argument("--max-dist-km", type=float, default=2200.0)
    ap.add_argument("--fallback-dist-km", type=float, default=3500.0)
    ap.add_argument("--sigma-km", type=float, default=750.0)
    ap.add_argument("--gamma-bath", type=float, default=25.0)
    ap.add_argument("--land-check", choices=["auto", "require", "none"], default="require")
    ap.add_argument("--max-land-run-km", type=float, default=40.0,
                    help="reject an edge whose path crosses a contiguous land stretch longer than this")
    ap.add_argument("--coast-buffer-km", type=float, default=15.0)
    ap.add_argument("--step-km", type=float, default=2.0)
    args = ap.parse_args()

    st = load_stations(args.dataset)
    print(f"Loaded {len(st)} stations (elevation from '{st.attrs['elevation_source']}')")
    checker = LandChecker(args.land_check, args.max_land_run_km, args.coast_buffer_km, args.step_km)
    names, src, dst, dist, edge_attr, rejected = build_graph(
        st, args.k_neighbors, args.max_dist_km, args.sigma_km, args.gamma_bath, checker, args.fallback_dist_km)

    comps = connected_components(len(names), src, dst)
    deg = np.bincount(dst, minlength=len(names))
    print(f"Edges (directed): {len(src)} | mean in-degree {deg.mean():.2f} | "
          f"min {deg.min()} | components: {len(comps)}")
    if len(comps) > 1:
        for c in comps:
            print("   component:", [names[i] for i in c])
    print(f"Rejected over-land candidates: {len(rejected)}")
    for a, b, d, lr in rejected[:30]:
        print(f"   x {a:>15s} -> {b:<15s} {d:7.0f} km (land run {lr:5.0f} km)")
    print(f"sim_geo  range [{edge_attr[:, 0].min():.3f}, {edge_attr[:, 0].max():.3f}]  "
          f"sim_bath range [{edge_attr[:, 1].min():.3f}, {edge_attr[:, 1].max():.3f}]")
    if np.median(edge_attr[:, 1]) < 0.05:
        print("[hint] sim_bath is ~0 for most edges; consider a larger --gamma-bath (e.g. 200)")

    out = Path(args.output)
    torch.save({
        "station_names": names,
        "edge_index": torch.tensor(np.stack([src, dst]), dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_attr_names": ["sim_geo", "sim_bath", "sin_bearing", "cos_bearing"],
        "distance_km": torch.tensor(dist, dtype=torch.float32),
        "coords": torch.tensor(st[["lat", "lon"]].to_numpy(), dtype=torch.float32),
        "elevation": torch.tensor(st["elevation"].to_numpy(), dtype=torch.float32),
        "land_check": checker.enabled,
        "config": vars(args),
    }, out)
    audit = pd.DataFrame({"src": [names[i] for i in src], "dst": [names[j] for j in dst],
                          "distance_km": dist.round(1), "sim_geo": edge_attr[:, 0].round(4),
                          "sim_bath": edge_attr[:, 1].round(4)})
    audit.to_csv(out.with_name(out.stem + "_edges.csv"), index=False)
    print(f"Saved {out} and {out.with_name(out.stem + '_edges.csv')}")


if __name__ == "__main__":
    main()
