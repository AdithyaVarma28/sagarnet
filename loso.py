"""Run the full Leave-One-Station-Out sweep and aggregate test metrics.

    .\\.venv\\Scripts\\python.exe loso.py                      # all 34 stations
    .\\.venv\\Scripts\\python.exe loso.py --stations Cochin Vishakhapatnam -- --epochs 30

Anything after ``--`` is forwarded to train.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import train
from sagar_dataset import build_cache


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.DEFAULT_DATASET)
    ap.add_argument("--cache-dir", default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--stations", nargs="*", default=None)
    ap.add_argument("--out-dir", default="runs")
    a = ap.parse_args(argv)

    names = [str(s) for s in build_cache(a.dataset, a.cache_dir)["station_names"]]
    stations = a.stations or names
    rows = []
    for st in stations:
        run = f"loso_{st}"
        mfile = Path(a.out_dir) / run / "metrics.json"
        if not mfile.exists():
            print(f"\n######## LOSO fold: {st} ########")
            train.main(["--dataset", a.dataset, "--cache-dir", a.cache_dir, "--out-dir", a.out_dir,
                        "--holdout", st, "--run-name", run] + extra)
        m = json.loads(mfile.read_text())
        rep = m.get("ungauged", {}).get(st)
        if rep and rep["average"]:
            rows.append({"station": st, **rep["average"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out_dir) / "loso_summary.csv", index=False)
    cols = [c for c in ["mae_total", "mae_persistence", "mae_tide_only", "mae_tide_ib", "mae_res",
                        "raw_cov", "cal_cov", "cal_width"] if c in df]
    print(df[["station"] + cols].round(4).to_string(index=False))
    print("\nMean over folds:\n", df[cols].mean().round(4).to_string())


if __name__ == "__main__":
    main()
