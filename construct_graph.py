"""Build the audited sparse marine graph (spec 3.2) for the 34-station dataset.

Connectivity is based on SEA-ROUTE distance, not straight lines
----------------------------------------------------------------
v2.0 rejected an edge if its straight great-circle line crossed > 40 km of land.
Along curved coasts (Malacca Strait, Indonesian arcs, Indian peninsula) that
rejected almost every edge: the real 34-station run produced 7 disconnected
components and left Cochin with no neighbours at all.

Now an ocean grid is built from global-land-mask (default 0.1 deg ~ 11 km),
each station is snapped to its nearest water cell, and shortest water-only
paths are computed with Dijkstra. An edge i->j is allowed when
    sea_route_km <= --max-dist-km   and   sea_route_km / great_circle_km <= --max-detour
so neighbours along the same coast connect, while stations on opposite sides of
a peninsula (e.g. Penang vs Gulf of Thailand) do not. sim_geo uses the sea-route
distance. Every node is guaranteed at least --min-degree marine neighbours
(searching up to --fallback-dist-km), and the result must be one connected graph.

Edge attributes: [sim_geo, sim_bath, sin(bearing), cos(bearing)].

Usage
-----
    .\\.venv\\Scripts\\python.exe construct_graph.py \
        --dataset combined34/34_stations_combined_with_IBTrACs.csv \
        --output static_graph.pt --k-neighbors 4
"""
from __future__ import annotations

import argparse
import heapq
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
def haversine(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    return haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :])


def initial_bearing(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return math.atan2(x, y)


# --------------------------------------------------------------------------- #
# Sea-route distances on an ocean grid
# --------------------------------------------------------------------------- #
def sea_route_matrix(lat: np.ndarray, lon: np.ndarray, res_deg: float = 0.1, pad_deg: float = 6.0,
                     snap_km: float = 60.0, limit_km: float = 6000.0) -> np.ndarray:
    """Shortest water-only path length [km] between every station pair (inf if unreachable)."""
    try:
        from global_land_mask import globe
    except ImportError as e:
        raise SystemExit("global-land-mask is required: pip install global-land-mask "
                         "(or pass --marine-check none)") from e

    lats = np.arange(lat.min() - pad_deg, lat.max() + pad_deg + 1e-9, res_deg)
    lons = np.arange(lon.min() - pad_deg, lon.max() + pad_deg + 1e-9, res_deg)
    lats = lats[(lats > -89.9) & (lats < 89.9)]
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    water = ~globe.is_land(LA, ((LO + 180) % 360) - 180)
    ny, nx = water.shape
    idx = -np.ones(water.shape, dtype=np.int64)
    idx[water] = np.arange(water.sum())
    print(f"Ocean grid {ny}x{nx} at {res_deg} deg, {water.sum():,} water cells")

    # 8-connected water graph
    rows, cols, wts = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        y0, y1 = max(0, -dy), ny - max(0, dy)
        x0, x1 = max(0, -dx), nx - max(0, dx)
        a = water[y0:y1, x0:x1] & water[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        ia = idx[y0:y1, x0:x1][a]
        ib = idx[y0 + dy:y1 + dy, x0 + dx:x1 + dx][a]
        w = haversine(LA[y0:y1, x0:x1][a], LO[y0:y1, x0:x1][a],
                      LA[y0 + dy:y1 + dy, x0 + dx:x1 + dx][a], LO[y0 + dy:y1 + dy, x0 + dx:x1 + dx][a])
        rows += [ia, ib]; cols += [ib, ia]; wts += [w, w]
    rows, cols, wts = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)

    # snap stations (they sit on the coast) to the nearest OPEN-SEA water cell: ignore lakes, lagoons
    # and enclosed bays that the land mask disconnects from the ocean at this resolution
    snap_ok = np.ones(int(water.sum()), dtype=bool)
    try:
        from scipy.ndimage import label
        lab, nlab = label(water, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        big = sizes >= max(2000, 0.01 * water.sum())
        snap_ok = big[lab[water]]
    except ImportError:
        pass
    wl, wo = LA[water], LO[water]
    wl = np.where(snap_ok, wl, np.nan)
    src_cells, snap_d = [], []
    for la, lo in zip(lat, lon):
        d = haversine(la, lo, wl, wo)
        k = int(np.nanargmin(d))
        src_cells.append(k); snap_d.append(float(d[k]))
    snap_d = np.array(snap_d)
    if (snap_d > snap_km).any():
        print(f"[warn] {int((snap_d > snap_km).sum())} station(s) are > {snap_km} km from open water on the grid")

    n_cells = int(water.sum())
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        G = coo_matrix((wts, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
        D = dijkstra(G, directed=False, indices=src_cells, limit=limit_km)
    except ImportError:
        print("[info] scipy not found - using pure-Python Dijkstra (slower)")
        order = np.argsort(rows, kind="stable")
        rows, cols, wts = rows[order], cols[order], wts[order]
        ptr = np.searchsorted(rows, np.arange(n_cells + 1))
        D = np.full((len(src_cells), n_cells), np.inf)
        for s, c0 in enumerate(src_cells):
            dist = D[s]; dist[c0] = 0.0
            pq = [(0.0, c0)]
            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u] or d > limit_km:
                    continue
                for k in range(ptr[u], ptr[u + 1]):
                    v, nd = cols[k], d + wts[k]
                    if nd < dist[v]:
                        dist[v] = nd
                        heapq.heappush(pq, (nd, v))
    S = D[:, src_cells] + snap_d[:, None] + snap_d[None, :]
    np.fill_diagonal(S, 0.0)
    return np.minimum(S, S.T)


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
def select_edges(names, gc: np.ndarray, route: np.ndarray, k: int, max_dist: float, max_detour: float,
                 min_degree: int, fallback_dist: float):
    n = len(names)
    ok = np.isfinite(route) & (route <= fallback_dist) & (route <= max_detour * np.maximum(gc, 1.0))
    np.fill_diagonal(ok, False)
    chosen: set[tuple[int, int]] = set()
    for i in range(n):
        cand = [j for j in np.argsort(route[i]) if ok[i, j]]
        near = [j for j in cand if route[i, j] <= max_dist][:k]
        if len(near) < min_degree:   # guarantee a minimum marine degree with longer links
            extra = [j for j in cand if j not in near][: min_degree - len(near)]
            for j in extra:
                print(f"[info] {names[i]}: extra link to {names[j]} (sea route {route[i, j]:.0f} km)")
            near += extra
        for j in near:
            chosen.add((i, j)); chosen.add((j, i))     # reciprocal
    return sorted(chosen)


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
    return sorted(comps.values(), key=len, reverse=True)


def bridge_components(edges: list, route: np.ndarray, names) -> list:
    """Join disconnected components with their shortest finite sea-route link."""
    edges = list(edges)
    n = len(names)
    while True:
        comps = connected_components(n, [e[0] for e in edges], [e[1] for e in edges])
        if len(comps) == 1:
            return edges
        main = set(comps[0])
        best = None
        for comp in comps[1:]:
            for i in comp:
                for j in main:
                    if np.isfinite(route[i, j]) and (best is None or route[i, j] < best[2]):
                        best = (i, j, route[i, j])
        if best is None:
            print("[warn] remaining components are not reachable by sea; leaving them disconnected")
            return edges
        i, j, d = best
        print(f"[info] bridging component via {names[i]} <-> {names[j]} (sea route {d:.0f} km)")
        edges += [(i, j), (j, i)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "--csv", dest="dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output", default=DEFAULT_GRAPH)
    ap.add_argument("--k-neighbors", type=int, default=4)
    ap.add_argument("--max-dist-km", type=float, default=2200.0)
    ap.add_argument("--max-detour", type=float, default=2.5,
                    help="max ratio sea-route / straight-line distance (blocks links around peninsulas)")
    ap.add_argument("--min-degree", type=int, default=2)
    ap.add_argument("--fallback-dist-km", type=float, default=3500.0)
    ap.add_argument("--sigma-km", type=float, default=750.0)
    ap.add_argument("--gamma-bath", type=float, default=25.0)
    ap.add_argument("--marine-check", choices=["sea-route", "none"], default="sea-route")
    ap.add_argument("--grid-res-deg", type=float, default=0.1)
    args = ap.parse_args()

    st = load_stations(args.dataset)
    names = list(st.index)
    lat, lon, z = st["lat"].to_numpy(), st["lon"].to_numpy(), st["elevation"].to_numpy()
    print(f"Loaded {len(st)} stations (elevation from '{st.attrs['elevation_source']}')")
    gc = haversine_matrix(lat, lon)
    if args.marine_check == "sea-route":
        route = sea_route_matrix(lat, lon, args.grid_res_deg, limit_km=args.fallback_dist_km * 1.5)
    else:
        route = gc.copy()

    edges = select_edges(names, gc, route, args.k_neighbors, args.max_dist_km, args.max_detour,
                         args.min_degree, args.fallback_dist_km)
    edges = sorted(set(bridge_components(edges, route, names)))
    src = np.array([e[0] for e in edges]); dst = np.array([e[1] for e in edges])
    d = route[src, dst]
    sim_geo = np.exp(-(d ** 2) / (2 * args.sigma_km ** 2))
    sim_bath = np.exp(-np.abs(z[src] - z[dst]) / args.gamma_bath)
    bearing = np.array([initial_bearing(lat[a], lon[a], lat[b], lon[b]) for a, b in edges])
    edge_attr = np.stack([sim_geo, sim_bath, np.sin(bearing), np.cos(bearing)], 1)

    comps = connected_components(len(names), src, dst)
    deg = np.bincount(dst, minlength=len(names))
    print(f"Edges (directed): {len(src)} | mean in-degree {deg.mean():.2f} | min {deg.min()} | "
          f"components: {len(comps)}")
    for c in comps[1:]:
        print("   separate component:", [names[i] for i in c])
    print(f"sim_geo  range [{edge_attr[:, 0].min():.3f}, {edge_attr[:, 0].max():.3f}]  "
          f"sim_bath range [{edge_attr[:, 1].min():.3f}, {edge_attr[:, 1].max():.3f}]")
    if np.median(edge_attr[:, 1]) < 0.05:
        print("[hint] sim_bath is ~0 for most edges; consider a larger --gamma-bath (e.g. 200)")
    print("\nNeighbours (sea-route km):")
    for i, n in enumerate(names):
        nb = [f"{names[j]}({route[i, j]:.0f})" for s, j in zip(src, dst) if s == i]
        print(f"  {n:>15s}: {', '.join(nb)}")

    out = Path(args.output)
    coords = torch.tensor(st[["lat", "lon"]].to_numpy(), dtype=torch.float32)
    torch.save({
        "station_names": names,
        "edge_index": torch.tensor(np.stack([src, dst]), dtype=torch.long),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
        "edge_attr_names": ["sim_geo", "sim_bath", "sin_bearing", "cos_bearing"],
        "distance_km": torch.tensor(d, dtype=torch.float32),
        "coords": coords,
        "elevation": torch.tensor(z, dtype=torch.float32),
        "land_check": args.marine_check != "none",
        "config": vars(args),
        # v1-compatible keys
        "node_names": names, "node_coords": coords,
        "node_elevations": torch.tensor(z, dtype=torch.float32).unsqueeze(-1),
        "is_virtual": torch.zeros(len(names), dtype=torch.bool),
        "k_neighbors": args.k_neighbors, "max_distance_km": args.max_dist_km,
        "sigma_dist": args.sigma_km, "gamma_bath": args.gamma_bath,
    }, out)
    audit = pd.DataFrame({"src": [names[i] for i in src], "dst": [names[j] for j in dst],
                          "sea_route_km": d.round(1), "straight_km": gc[src, dst].round(1),
                          "sim_geo": edge_attr[:, 0].round(4), "sim_bath": edge_attr[:, 1].round(4)})
    audit.to_csv(out.with_name(out.stem + "_edges.csv"), index=False)
    print(f"\nSaved {out} and {out.with_name(out.stem + '_edges.csv')}")


if __name__ == "__main__":
    main()
