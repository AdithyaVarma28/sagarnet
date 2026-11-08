"""Data pipeline for SAGAR-Net v2.

Key changes vs v1
-----------------
1. Schema: reads the 34-station IBTrACS file (``cyclone_*`` columns, with the
   legacy ``storm_*`` names accepted as aliases) and fills physical defaults
   (P = 1013.25 hPa, d = 10000 km, active = 0, cyclone lat/lon = station).
2. Chart-datum fix: each gauge is decomposed on the TRAINING period only into
       sea_level = mean + trend + astronomical tide + residual
   and the network predicts the zero-centred *residual* (surge). Datum offsets
   (0.75 m at Cochin vs 3-4 m elsewhere) therefore never enter message passing.
   Total water level is reconstructed as tide + residual for reporting.
3. Observed residual history is an input channel with an availability mask.
   Held-out / ungauged stations always have mask = 0; during training a random
   subset of gauged stations is masked too (see train.py), so the same model
   does gauged forecasting AND ungauged transfer.
4. Physics-derived inputs: wind stress, inverted-barometer height, 3-h pressure
   tendency, cyclone proximity / relative position.
5. The 3M-row CSV is parsed once and cached as .npz (seconds to reload).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import config as C
from tides import fit_tide, hours_since_epoch, predict_tide, tidal_phase_features

CACHE_VERSION = 5

# Columns that should not be z-scored (binary / already bounded).
_NO_NORM_PREFIX = ("tide_", "doy_")
_NO_NORM = {"cyc_active"}


# --------------------------------------------------------------------------- #
# Schema resolution
# --------------------------------------------------------------------------- #
def resolve_schema(header) -> dict:
    header = list(header)
    missing = [c for c in ["time", "station_name", "station_latitude", "station_longitude", "sea_level"]
               if c not in header]
    if missing:
        raise KeyError(f"Dataset is missing required columns: {missing}")
    cyclone = {}
    for canon, aliases in C.CYCLONE_ALIASES.items():
        hit = next((a for a in aliases if a in header), None)
        if hit is not None:
            cyclone[canon] = hit
    if not cyclone:
        print("[warn] No cyclone columns found - cyclone features will be constant defaults. "
              "Use combined34/34_stations_combined_with_IBTrACs.csv")
    era5 = [c for c in C.ERA5_COLS if c in header]
    gebco = [c for c in C.GEBCO_COLS if c in header] or [c for c in ["elevation_m"] if c in header]
    return {"cyclone": cyclone, "era5": era5, "gebco": gebco,
            "target_valid": "target_valid" in header}


def _robust_fill(a: np.ndarray, limit: int = 6) -> np.ndarray:
    """[T, N] forward/back fill short gaps, then station mean, then global mean."""
    df = pd.DataFrame(a).ffill(limit=limit).bfill(limit=limit)
    out = df.to_numpy(dtype=np.float32)
    col_mean = np.nanmean(np.where(np.isfinite(out), out, np.nan), axis=0)
    glob = np.nanmean(col_mean) if np.isfinite(col_mean).any() else 0.0
    col_mean = np.where(np.isfinite(col_mean), col_mean, glob)
    bad = ~np.isfinite(out)
    out[bad] = np.take(col_mean, np.nonzero(bad)[1])
    return out


def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


# --------------------------------------------------------------------------- #
# Cache builder
# --------------------------------------------------------------------------- #
def _cache_path(csv_path: str, cache_dir: str) -> Path:
    st = os.stat(csv_path)
    key = f"{os.path.abspath(csv_path)}|{st.st_size}|{int(st.st_mtime)}|{CACHE_VERSION}|{C.TRAIN_END}"
    h = hashlib.md5(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"{Path(csv_path).stem}_{h}.npz"


def build_cache(csv_path: str, cache_dir: str = C.DEFAULT_CACHE_DIR, force: bool = False) -> dict:
    path = _cache_path(csv_path, cache_dir)
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    header = pd.read_csv(csv_path, nrows=0).columns
    sch = resolve_schema(header)
    usecols = (["time", "station_name", "station_latitude", "station_longitude", "sea_level"]
               + sch["era5"] + sch["gebco"] + list(sch["cyclone"].values())
               + (["target_valid"] if sch["target_valid"] else []))
    print(f"Reading {csv_path} ({len(usecols)} columns) ...")
    df = pd.read_csv(csv_path, usecols=usecols, parse_dates=["time"], low_memory=False)
    df = df.rename(columns={v: k for k, v in sch["cyclone"].items()})
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)
    df["time"] = df["time"].dt.floor("h")

    names = sorted(df["station_name"].unique())           # alphabetical == graph order
    sidx = {n: i for i, n in enumerate(names)}
    times = pd.date_range(df["time"].min(), df["time"].max(), freq="h")
    T, N = len(times), len(names)
    ti = ((df["time"] - times[0]) // pd.Timedelta(hours=1)).to_numpy()
    si = df["station_name"].map(sidx).to_numpy()
    print(f"  {N} stations x {T} hourly steps ({times[0]} -> {times[-1]})")

    def pivot(col, fill=np.nan):
        a = np.full((T, N), fill, dtype=np.float32)
        if col in df:
            a[ti, si] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
        return a

    st = df.groupby("station_name")[["station_latitude", "station_longitude"] + sch["gebco"]].median()
    st = st.loc[names]
    lat = st["station_latitude"].to_numpy(np.float64)
    lon = st["station_longitude"].to_numpy(np.float64)

    # ---------------- target ----------------
    sea = pivot("sea_level")
    valid = np.isfinite(sea) & (sea > C.SEA_LEVEL_MISSING_BELOW)
    if sch["target_valid"]:
        tv = pivot("target_valid", 0.0)
        valid &= tv > 0.5
    sea = np.where(valid, sea, np.nan).astype(np.float32)

    # ---------------- ERA5 ----------------
    u10 = _robust_fill(pivot("u10"))
    v10 = _robust_fill(pivot("v10"))
    msl = _robust_fill(pivot("msl"))
    if np.nanmedian(msl) < 2000:            # given in hPa -> Pa
        msl = msl * 100.0
    tp = _robust_fill(pivot("tp"))
    if np.nanmax(tp) < 1.0:                 # metres -> mm
        tp = tp * 1000.0
    tp = np.clip(tp, 0, None)
    ws = np.sqrt(u10 ** 2 + v10 ** 2) if "wind_speed" not in df else _robust_fill(pivot("wind_speed"))

    t_hours = hours_since_epoch(times.values)
    train_end = int(np.searchsorted(times.values, np.datetime64(C.TRAIN_END), side="right"))
    val_end = int(np.searchsorted(times.values, np.datetime64(C.VAL_END), side="right"))

    msl_ref = msl[:train_end].mean(0, keepdims=True)                  # per-station climatology
    ib = (-(msl - msl_ref) / C.RHO_G).astype(np.float32)               # inverted barometer [m]
    msl_anom = (msl - msl_ref) / 100.0
    dmsl = np.zeros_like(msl)
    dmsl[3:] = (msl[3:] - msl[:-3]) / 100.0
    tau_x = C.RHO_AIR * C.CD_WIND * ws * u10
    tau_y = C.RHO_AIR * C.CD_WIND * ws * v10

    # ---------------- IBTrACS ----------------
    act = np.nan_to_num(pivot("cyclone_active"), nan=0.0) > 0.5
    dist = pivot("cyclone_distance_km")
    dist = np.where(act & np.isfinite(dist), dist, C.CYCLONE_DEFAULTS["cyclone_distance_km"])
    wind = np.where(act, np.nan_to_num(pivot("cyclone_wind_kts"), nan=0.0), 0.0)
    pres = np.where(act, pivot("cyclone_pressure_hpa"), np.nan)
    pres = np.where(np.isfinite(pres), pres, C.CYCLONE_DEFAULTS["cyclone_pressure_hpa"])
    clat = np.where(act, pivot("cyclone_latitude"), np.nan)
    clon = np.where(act, pivot("cyclone_longitude"), np.nan)
    dlat = np.nan_to_num(clat - lat[None, :], nan=0.0)
    dlon = np.nan_to_num(((clon - lon[None, :] + 180.0) % 360.0) - 180.0, nan=0.0)
    actf = act.astype(np.float32)
    prox = actf * np.exp(-dist / 500.0)
    pdef = np.clip(C.CYCLONE_DEFAULTS["cyclone_pressure_hpa"] - pres, 0, None) * actf
    del df

    # ---------------- calendar / tide phase ----------------
    tide_feats = tidal_phase_features(t_hours, lon, C.TIDE_FEATURE_PERIODS_H)   # [T,N,2P]
    doy = times.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    doy_s = np.repeat(np.sin(doy)[:, None], N, 1)
    doy_c = np.repeat(np.cos(doy)[:, None], N, 1)

    dyn_list = [
        ("u10", u10), ("v10", v10), ("msl_anom_hpa", msl_anom), ("tp_mm", tp), ("wind_speed", ws),
        ("tau_x", tau_x), ("tau_y", tau_y), ("ib_m", ib), ("dmsl_3h_hpa", dmsl),
        ("cyc_active", actf), ("cyc_proximity", prox), ("cyc_logdist", np.log1p(dist)),
        ("cyc_wind_kts", wind), ("cyc_pdef_hpa", pdef), ("cyc_dlat", dlat), ("cyc_dlon", dlon),
        ("cyc_wind_proxy", wind * prox),
    ]
    dyn_names = [n for n, _ in dyn_list]
    dyn = np.stack([a.astype(np.float32) for _, a in dyn_list], -1)
    tide_names = [f"tide_{k}_{f}" for k in C.TIDE_FEATURE_PERIODS_H for f in ("sin", "cos")]
    dyn = np.concatenate([dyn, tide_feats, np.stack([doy_s, doy_c], -1).astype(np.float32)], -1)
    dyn_names += tide_names + ["doy_sin", "doy_cos"]

    # ---------------- harmonic analysis (train period only) ----------------
    tide = np.full((T, N), np.nan, dtype=np.float32)
    tide_src = np.zeros(N, dtype=np.int8)       # 0 none, 1 train, 2 train+val (fallback)
    t0 = float(t_hours[0])
    for j in range(N):
        coef = fit_tide(t_hours[:train_end], sea[:train_end, j], valid[:train_end, j],
                        C.HARMONIC_CONSTITUENTS, t0, nodal=C.TIDE_NODAL)
        src = 1
        if coef is None:   # station starts late: fall back to pre-test data, never test data
            coef = fit_tide(t_hours[:val_end], sea[:val_end, j], valid[:val_end, j],
                            C.HARMONIC_CONSTITUENTS, t0, nodal=C.TIDE_NODAL)
            src = 2
        if coef is None:
            print(f"  [warn] {names[j]}: too little data for harmonic analysis -> no targets")
            valid[:, j] = False
            continue
        tide[:, j] = predict_tide(t_hours, coef, C.HARMONIC_CONSTITUENTS, t0)
        tide_src[j] = src

    # ---------------- despike residual (label QC only) ----------------
    resid = sea - tide
    med = pd.DataFrame(resid).rolling(7, center=True, min_periods=3).median().to_numpy()
    dev = np.abs(resid - med)
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid[:train_end], 0)), 0) * 1.4826
    spike = dev > np.maximum(8.0 * mad, 0.75)[None, :]
    print(f"  despiked {int(np.nansum(spike))} label points")
    valid &= ~spike & np.isfinite(resid)

    gebco = st[sch["gebco"]].to_numpy(np.float64) if sch["gebco"] else np.zeros((N, 1))
    static = np.concatenate([
        lat[:, None], lon[:, None], np.sin(np.radians(lon))[:, None], np.cos(np.radians(lon))[:, None],
        _slog(gebco)], 1).astype(np.float32)
    static_names = ["lat", "lon", "sin_lon", "cos_lon"] + [f"slog_{c}" for c in (sch["gebco"] or ["gebco"])]

    for j, n in enumerate(names):
        if tide_src[j]:
            r = resid[:train_end, j][valid[:train_end, j]]
            print(f"  {n:>15s}: mean {np.nanmean(sea[:train_end, j]):6.3f} m | "
                  f"tide fit {'train' if tide_src[j] == 1 else 'train+val'} | "
                  f"residual std {np.std(r) if r.size else float('nan'):.3f} m")

    out = dict(
        times=times.values.astype("datetime64[s]").astype(np.int64), station_names=np.array(names),
        lat=lat.astype(np.float32), lon=lon.astype(np.float32),
        dyn=dyn, dyn_names=np.array(dyn_names), static=static, static_names=np.array(static_names),
        sea_level=sea, tide=tide, valid=valid, ib=ib, cyc_active=actf, tide_src=tide_src,
        train_end=np.int64(train_end), val_end=np.int64(val_end),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"  cached -> {path}")
    return out


# --------------------------------------------------------------------------- #
# Normalised container shared by all splits
# --------------------------------------------------------------------------- #
class SagarData:
    def __init__(self, cache: dict, holdout: list[str] | None = None, history: int = C.HISTORY_HOURS,
                 horizons=C.HORIZONS, future_lags=C.FUTURE_LAGS, use_future_forcing: bool = True,
                 stats: dict | None = None, target: str = "surge", lf_window: int = C.LF_WINDOW_HOURS,
                 static_set: str = "all"):
        """
        target="surge" (default): residual = slow mean-sea-level anomaly + fast surge.
            label  = residual(t+h) - centred 30-day mean     (fast surge; robust to datum jumps)
            inputs = residual history - trailing 30-day mean at the anchor (causal)
            total  = tide + trailing 30-day mean (gauged) + predicted surge
        target="full": label = full residual (first v2 run).
        """
        self.names = [str(s) for s in cache["station_names"]]
        self.N = len(self.names)
        holdout = holdout or []
        unknown = [h for h in holdout if h not in self.names]
        if unknown:
            raise KeyError(f"Unknown holdout station(s) {unknown}. Available: {self.names}")
        self.holdout = np.array([n in holdout for n in self.names])
        self.history, self.horizons = history, np.asarray(horizons)
        self.future_lags = np.asarray(future_lags)
        self.times = cache["times"].astype("datetime64[s]")
        self.train_end, self.val_end = int(cache["train_end"]), int(cache["val_end"])
        self.T = len(self.times)

        self.sea_level, self.tide = cache["sea_level"], cache["tide"]
        self.target_mode = target
        valid_raw = cache["valid"].astype(bool) & np.isfinite(self.tide)
        resid = np.where(valid_raw, self.sea_level - self.tide, np.nan).astype(np.float32)
        ib = cache["ib"]
        if target == "surge":
            mp = lf_window // 3
            rdf = pd.DataFrame(resid)
            lf_c = rdf.rolling(lf_window, center=True, min_periods=mp).mean().to_numpy(np.float32)
            lf_t = rdf.rolling(lf_window, min_periods=mp).mean().to_numpy(np.float32)
            ib_lf = pd.DataFrame(ib).rolling(lf_window, center=True, min_periods=1).mean().to_numpy(np.float32)
            self.valid = valid_raw & np.isfinite(lf_c)
            tgt = resid - lf_c
            self.off = np.nan_to_num(lf_t)                 # causal slow level known at gauged anchors
            self.off_ok = np.isfinite(lf_t)
            ib_t = ib - ib_lf
        elif target == "full":
            self.valid = valid_raw
            tgt = resid
            self.off = np.zeros_like(resid)
            self.off_ok = np.ones(resid.shape, dtype=bool)
            ib_t = ib
        else:
            raise ValueError(target)
        self.valid_obs = valid_raw
        self.resid_full = np.nan_to_num(resid)              # metres, 0 where invalid
        self.target = np.where(self.valid, tgt, 0.0).astype(np.float32)
        self.ib_t = ib_t.astype(np.float32)
        # slow level usable for total-water-level reconstruction: unknown (0) at ungauged stations
        self.lf_known = np.where(self.holdout[None, :], 0.0, self.off).astype(np.float32)
        self.lf_source = {}
        self.cyc_active = cache["cyc_active"]
        self.dyn_names = [str(s) for s in cache["dyn_names"]]

        if stats is None:   # fit on training period. Forcing exists at ungauged sites, labels do not.
            dyn_tr = cache["dyn"][: self.train_end]
            mean = dyn_tr.reshape(-1, dyn_tr.shape[-1]).mean(0)
            std = dyn_tr.reshape(-1, dyn_tr.shape[-1]).std(0) + 1e-6
            for k, n in enumerate(self.dyn_names):
                if n.startswith(_NO_NORM_PREFIX) or n in _NO_NORM:
                    mean[k], std[k] = 0.0, 1.0
            lab = self.valid[: self.train_end] & ~self.holdout[None, :]
            sigma = float(self.target[: self.train_end][lab].std())
            s = cache["static"]
            stats = dict(dyn_mean=mean, dyn_std=std, resid_scale=sigma,
                         static_mean=s.mean(0), static_std=s.std(0) + 1e-6)
        self.stats = stats
        self.scale = float(stats["resid_scale"])

        self.dyn = ((cache["dyn"] - stats["dyn_mean"]) / stats["dyn_std"]).astype(np.float32)
        if not use_future_forcing:
            keep = np.array([n.startswith(_NO_NORM_PREFIX) for n in self.dyn_names])
            self.fut_dyn = self.dyn * keep[None, None, :]
        else:
            self.fut_dyn = self.dyn
        self.target_n = (self.target / self.scale).astype(np.float32)
        self.resid_full_n = (self.resid_full / self.scale).astype(np.float32)
        self.off_n = (self.off / self.scale).astype(np.float32)
        self.ib_n = (self.ib_t / self.scale).astype(np.float32)
        self.static = ((cache["static"] - stats["static_mean"]) / stats["static_std"]).astype(np.float32)
        if static_set == "physical":
            # drop station-identifying coordinates, keep bathymetry. With 33 training stations, lat/lon
            # let the network memorise per-station behaviour that cannot transfer to an unseen coast.
            names = [str(n) for n in cache["static_names"]]
            geo = [i for i, n in enumerate(names) if n in ("lat", "lon", "sin_lon", "cos_lon")]
            self.static[:, geo] = 0.0
        elif static_set != "all":
            raise ValueError(static_set)

        self.F = self.dyn.shape[-1]
        self.in_dim = self.F + 2
        self.fut_dim = self.F * len(self.future_lags)
        self.static_dim = self.static.shape[-1]

    def split_anchors(self, split: str, stride: int = 1) -> np.ndarray:
        L, H = self.history, int(self.horizons.max())
        if split == "train":
            lo, hi = L - 1, self.train_end - 1 - H
        elif split == "val":
            lo, hi = self.train_end, self.val_end - 1 - H
        elif split == "test":
            lo, hi = self.val_end, self.T - 1 - H
        else:
            raise ValueError(split)
        lo = max(lo, L - 1)
        if hi < lo:
            raise ValueError(f"Split '{split}' is empty with history={L}, max horizon={H}")
        return np.arange(lo, hi + 1, stride)

    def set_neighbour_level(self, edge_index, distance_km, max_km: float = 800.0) -> dict:
        """Slow level at ungauged stations from gauged graph neighbours (causal).

        The 30-day sea-level anomaly is mostly regional, so at a held-out station
        it is estimated as the inverse-distance mean of the trailing 30-day level of
        gauged neighbours within ``max_km`` (sea route). Uses only their past data,
        so it is available at forecast time. No neighbour in range -> stays 0.
        """
        if self.target_mode != "surge":
            return {}
        ei = np.asarray(edge_index)
        dk = np.asarray(distance_km, dtype=np.float64).reshape(-1)
        for j in np.flatnonzero(self.holdout):
            nb = {}
            for (a, b), d in zip(ei.T, dk):
                k = a if b == j else b if a == j else -1
                if k >= 0 and k != j and not self.holdout[k] and d <= max_km:
                    nb[int(k)] = min(d, nb.get(int(k), np.inf))
            if not nb:
                self.lf_source[self.names[j]] = "none in range -> 0"
                continue
            ks = np.array(list(nb)); w = 1.0 / np.maximum(np.array([nb[k] for k in ks]), 50.0)
            ok = self.off_ok[:, ks].astype(np.float64) * w[None, :]
            num = (self.off[:, ks] * ok).sum(1)
            den = ok.sum(1)
            self.lf_known[:, j] = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0).astype(np.float32)
            self.lf_source[self.names[j]] = ", ".join(f"{self.names[k]}({nb[k]:.0f} km)" for k in ks)
        return self.lf_source


class SagarWindows(Dataset):
    """One sample = the whole graph at anchor time t."""

    def __init__(self, data: SagarData, split: str, stride: int = 1):
        self.d, self.split = data, split
        self.anchors = data.split_anchors(split, stride)
        hz, lags = data.horizons, data.future_lags
        self._fut_off = (hz[:, None] - lags[None, :])          # [Hn, nl]
        self._obs_ok = ~data.holdout                           # ungauged nodes never expose labels

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, i):
        d, t = self.d, int(self.anchors[i])
        L = d.history
        sl = slice(t - L + 1, t + 1)
        x = d.dyn[sl].transpose(1, 0, 2)                                       # [N,L,F]
        ok = self._obs_ok & d.off_ok[t]                                        # [N]
        om = (d.valid_obs[sl] & ok[None, :]).T.astype(np.float32)              # [N,L]
        obs = (d.resid_full_n[sl] - d.off_n[t][None, :]).T * om
        x = np.concatenate([x, obs[..., None], om[..., None]], -1)

        tgt = t + d.horizons
        fut = d.fut_dyn[t + self._fut_off]                                     # [Hn,nl,N,F]
        fut = fut.transpose(2, 0, 1, 3).reshape(d.N, len(d.horizons), -1)
        y = d.target_n[tgt].T                                                  # [N,Hn]
        ym = d.valid[tgt].T
        return {
            "x": torch.from_numpy(np.ascontiguousarray(x)),
            "fut": torch.from_numpy(np.ascontiguousarray(fut)),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "ymask": torch.from_numpy(np.ascontiguousarray(ym)),
            "ib": torch.from_numpy(np.ascontiguousarray(d.ib_n[tgt].T)),
            "last": torch.from_numpy(obs[:, -1].copy()),
            "last_mask": torch.from_numpy(om[:, -1].copy()),
            "cyc": torch.from_numpy(np.ascontiguousarray(d.cyc_active[tgt].T)),
            "t": torch.tensor(t, dtype=torch.long),
        }
