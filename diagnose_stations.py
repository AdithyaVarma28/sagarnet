"""Per-station data-quality check for the tide / residual decomposition.

Why: in the first real run, Cochin's harmonic tide (fitted on 2011-2019) missed
the 2022 observations by 0.22 m MAE, while persistence at +24 h was only
0.08 m. That points to a data problem (datum shift, timestamp shift, or a gap
that breaks the tide fit), not a model problem, and it sets a floor that no
model can beat. This script finds which stations are affected and how.

    .\\.venv\\Scripts\\python.exe diagnose_stations.py                  # table for all stations
    .\\.venv\\Scripts\\python.exe diagnose_stations.py --station Cochin # + plot

Output: station_diagnostics.csv, and diag_<Station>.png with --station.

Columns
-------
n_train/n_val/n_test      valid hourly points per split
mean_*, std_*             residual (observed - fitted tide) mean / std per split [m]
tide_mae_*                MAE of the harmonic tide alone per split [m]
lag_*                     time shift (hours) of the tide that best matches the
                          observations in that split; train should be 0. A
                          non-zero test lag (e.g. +5 or +6) means the timestamps
                          changed (e.g. local time vs UTC)
flags                     DATUM_SHIFT, TIME_SHIFT, NOISY, SPARSE_TRAIN, NO_FIT
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import config as C
from sagar_dataset import build_cache


def best_lag(obs, tide, valid, max_lag=12):
    """Lag L (hours) minimising mean |obs(t) - tide(t+L)|, after removing the mean offset."""
    best = (0, np.inf)
    T = len(obs)
    for L in range(-max_lag, max_lag + 1):
        a = slice(max(0, -L), T - max(0, L))
        b = slice(max(0, L), T - max(0, -L))
        m = valid[a] & np.isfinite(tide[b])
        if m.sum() < 500:
            continue
        r = obs[a][m] - tide[b][m]
        e = float(np.mean(np.abs(r - np.median(r))))
        if e < best[1]:
            best = (L, e)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", "--csv", dest="dataset", default=C.DEFAULT_DATASET)
    ap.add_argument("--cache-dir", default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--station", default=None, help="also plot this station")
    ap.add_argument("--out", default="station_diagnostics.csv")
    a = ap.parse_args()

    z = build_cache(a.dataset, a.cache_dir)
    names = [str(s) for s in z["station_names"]]
    times = pd.to_datetime(z["times"].astype("datetime64[s]"))
    sea, tide, valid = z["sea_level"], z["tide"], z["valid"].astype(bool)
    tr, va = int(z["train_end"]), int(z["val_end"])
    splits = {"train": slice(0, tr), "val": slice(tr, va), "test": slice(va, None)}

    rows = []
    for j, n in enumerate(names):
        r = {"station": n, "tide_fit": {0: "none", 1: "train", 2: "train+val"}[int(z["tide_src"][j])]}
        res = np.where(valid[:, j], sea[:, j] - tide[:, j], np.nan)
        for k, s in splits.items():
            v = valid[s, j] & np.isfinite(tide[s, j])
            r[f"n_{k}"] = int(v.sum())
            rr = res[s][v]
            r[f"mean_{k}"] = float(np.mean(rr)) if rr.size else np.nan
            r[f"std_{k}"] = float(np.std(rr)) if rr.size else np.nan
            r[f"tide_mae_{k}"] = float(np.mean(np.abs(rr))) if rr.size else np.nan
            if rr.size > 500:
                L, _ = best_lag(sea[s, j], tide[s, j], v)
                r[f"lag_{k}"] = L
            else:
                r[f"lag_{k}"] = np.nan
        flags = []
        if r["tide_fit"] == "none":
            flags.append("NO_FIT")
        if r["n_train"] < 24 * 365:
            flags.append("SPARSE_TRAIN")
        for k in ("val", "test"):
            if np.isfinite(r[f"mean_{k}"]) and abs(r[f"mean_{k}"]) > 0.05:
                flags.append(f"DATUM_SHIFT_{k}")
            if np.isfinite(r[f"lag_{k}"]) and r[f"lag_{k}"] != r.get("lag_train", 0):
                flags.append(f"TIME_SHIFT_{k}({int(r[f'lag_{k}']):+d}h)")
            if np.isfinite(r[f"std_{k}"]) and r[f"std_{k}"] > 1.5 * max(r["std_train"], 1e-3):
                flags.append(f"NOISY_{k}")
        r["flags"] = " ".join(flags)
        rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    cols = ["station", "tide_fit", "n_train", "n_test", "mean_test", "std_train", "std_test",
            "tide_mae_test", "lag_train", "lag_test", "flags"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(df[cols].round(3).to_string(index=False))
    bad = df[df["flags"] != ""]
    print(f"\n{len(bad)} station(s) flagged. Saved {a.out}")

    if a.station:
        j = names.index(a.station)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        res = pd.Series(np.where(valid[:, j], sea[:, j] - tide[:, j], np.nan), index=times)
        fig, ax = plt.subplots(3, 1, figsize=(13, 10))
        res.resample("MS").mean().plot(ax=ax[0], marker=".")
        ax[0].axvline(times[tr], color="k", ls="--"); ax[0].axvline(times[va], color="k", ls="--")
        ax[0].axhline(0, color="grey", lw=0.8)
        ax[0].set_title(f"{a.station}: monthly mean residual (observed - harmonic tide). Dashed = val / test start")
        for axi, s0, lab in [(ax[1], max(tr - 24 * 10, 0), "last training days"),
                             (ax[2], min(va + 24 * 60, len(times) - 24 * 7), "test period")]:
            sl = slice(s0, s0 + 24 * 7)
            axi.plot(times[sl], sea[sl, j], "k", label="observed")
            axi.plot(times[sl], tide[sl, j], "tab:orange", label="harmonic tide (fit on train)")
            axi.set_title(f"{a.station}: one week, {lab}")
            axi.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(f"diag_{a.station}.png", dpi=130)
        print(f"Saved diag_{a.station}.png")


if __name__ == "__main__":
    main()
