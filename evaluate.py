"""Calibrate (CQR on validation) and evaluate (test) a trained SAGAR-Net v2 checkpoint.

    .\\.venv\\Scripts\\python.exe evaluate.py --ckpt runs/loso_Cochin/best.pt

Reports, per horizon, for
  * UNGAUGED  = held-out station(s); history never visible (true LOSO transfer)
  * GAUGED    = training stations in the test year (temporal generalisation)
metrics on residual (surge) and on total water level (= harmonic tide + residual):
  MAE / RMSE, raw & CQR-calibrated 80 % coverage and width, q_hat,
and baselines on the same points:
  persistence      y(t+h) = y(t)                 (reference only for ungauged: needs a gauge)
  tide-only        harmonic tide, zero surge
  tide + IB        tide + inverted barometer
  tide + persist   tide(t+h) + residual(t)       (gauged only)

Note on total level at ungauged sites: the harmonic tide used here is fitted on
the held-out gauge's TRAINING years and stands in for a global tide model
(FES2014/TPXO) that would be used operationally. The network never sees it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import config as C
from calibration import CQR
from model import SagarNet
from sagar_dataset import SagarData, SagarWindows, build_cache
from utils import load_graph, mask_history, pick_device, random_node_mask, to_device


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = SagarNet(edge_index=ck["graph"]["edge_index"], edge_attr=ck["graph"]["edge_attr"], **ck["model_kwargs"])
    state = ck["model_state"]
    if any(k.startswith("head.") for k in state):     # pre-v2.3 checkpoint: head.{0,3,5} -> trunk.{0,3}, out_q
        ren = {"head.0.": "trunk.0.", "head.3.": "trunk.3.", "head.5.": "out_q."}
        state = {next((k.replace(a, b) for a, b in ren.items() if k.startswith(a)), k): v for k, v in state.items()}
    m.load_state_dict(state)
    return m.to(device).eval(), ck


@torch.no_grad()
def predict(model, data: SagarData, split: str, stride: int, device, batch_size=64, mask_prob=0.0, seed=0,
            static_override=None):
    ds = SagarWindows(data, split, stride)
    static = torch.from_numpy(data.static if static_override is None else static_override).to(device)
    dl = DataLoader(ds, batch_size, shuffle=False)
    protect = torch.as_tensor(data.holdout, device=device)
    g = torch.Generator().manual_seed(seed)
    qs, drops, ts = [], [], []
    for b in dl:
        b = to_device(b, device)
        B = b["x"].size(0)
        if mask_prob > 0:
            drop = random_node_mask(B, data.N, mask_prob, protect, device, g)
            b = mask_history(b, drop)
        else:
            drop = torch.zeros(B, data.N, dtype=torch.bool, device=device)
        q = model(b["x"], b["fut"], static, b["last"], b["last_mask"])
        qs.append((q * data.scale).cpu().numpy())
        drops.append(drop.cpu().numpy())
        ts.append(b["t"].cpu().numpy())
    return {"q": np.concatenate(qs), "drop": np.concatenate(drops), "t": np.concatenate(ts)}


_predict_one = predict


def gather_targets(data: SagarData, t: np.ndarray) -> dict:
    """All in metres. y_res = what the model predicts (surge); base = known part of total level:
    harmonic tide + slow 30-day level from the gauge's own past (0 at ungauged stations)."""
    tgt = t[:, None] + data.horizons[None, :]                          # [S,H]
    nan_if = lambda a, m: np.where(m, a, np.nan)
    sea = nan_if(data.sea_level, data.valid_obs)
    y = nan_if(data.target, data.valid)
    sw = lambda a: a[tgt].transpose(0, 2, 1)                            # [S,H,N] -> [S,N,H]
    base = sw(data.tide) + data.lf_known[t][:, :, None]
    obs_now = nan_if(data.resid_full - data.off, data.valid_obs & data.off_ok)
    return {
        "y_res": sw(y), "y_tot": sw(sea), "base": base, "tide": sw(data.tide),
        "ib": sw(data.ib_t), "cyc": sw(data.cyc_active),
        "sea_now": sea[t], "res_now": obs_now[t],                      # [S,N]
    }


def flatten(pred: dict, tg: dict, node_sel: np.ndarray, require_drop: bool = False) -> dict:
    sel = np.broadcast_to(node_sel[None, :], pred["drop"].shape).copy()
    if require_drop:
        sel &= pred["drop"]
    s, n = np.nonzero(sel)
    out = {"q": pred["q"][s, n], "t": pred["t"][s], "node": n}
    for k, v in tg.items():
        out[k] = v[s, n]
    return out


def horizon_report(e: dict, cal: CQR | None, horizons, gauged: bool) -> dict:
    q10, q50, q90 = e["q"][..., 0], e["q"][..., 1], e["q"][..., 2]
    grp = (e["cyc"] > 0.5).astype(int)
    if cal is not None:
        lo_c, hi_c = cal.apply(q10, q90, grp)
    rows = []
    for h, hz in enumerate(horizons):
        ok = np.isfinite(e["y_res"][:, h]) & np.isfinite(e["sea_now"])
        if ok.sum() == 0:
            continue
        ok &= np.isfinite(e["y_tot"][:, h])
        if ok.sum() == 0:
            continue
        y_r, y_t = e["y_res"][ok, h], e["y_tot"][ok, h]
        tide = e["base"][ok, h]          # tide + known slow level
        pred_t = tide + q50[ok, h]
        mae = lambda p, y: float(np.mean(np.abs(p - y)))
        rmse = lambda p, y: float(np.sqrt(np.mean((p - y) ** 2)))
        r = {
            "horizon_h": int(hz), "n": int(ok.sum()),
            "mae_res": mae(q50[ok, h], y_r), "rmse_res": rmse(q50[ok, h], y_r),
            "mae_total": mae(pred_t, y_t), "rmse_total": rmse(pred_t, y_t),
            "bias_total": float(np.mean(pred_t - y_t)),
            "mae_persistence": mae(e["sea_now"][ok], y_t), "rmse_persistence": rmse(e["sea_now"][ok], y_t),
            "mae_tide_only": mae(tide, y_t),
            "mae_tide_ib": mae(tide + e["ib"][ok, h], y_t),
            "mae_surge_zero": float(np.mean(np.abs(y_r))),
            "mae_surge_ib": mae(e["ib"][ok, h], y_r),
            "raw_cov": float(np.mean((y_r >= q10[ok, h]) & (y_r <= q90[ok, h]))),
            "raw_width": float(np.mean(q90[ok, h] - q10[ok, h])),
        }
        if gauged:
            pr = tide + np.nan_to_num(e["res_now"][ok])
            r["mae_tide_persist"] = mae(pr, y_t)
        if cal is not None:
            r["cal_cov"] = float(np.mean((y_r >= lo_c[ok, h]) & (y_r <= hi_c[ok, h])))
            r["cal_width"] = float(np.mean(hi_c[ok, h] - lo_c[ok, h]))
            r["q_hat"] = cal.qhat[(h, -1)]
            cy = grp[ok, h] == 1
            if cy.sum() > 0:
                r["n_cyclone"] = int(cy.sum())
                r["mae_total_cyclone"] = mae(pred_t[cy], y_t[cy])
                r["cal_cov_cyclone"] = float(np.mean((y_r[cy] >= lo_c[ok, h][cy]) & (y_r[cy] <= hi_c[ok, h][cy])))
        r["skill_vs_persistence"] = 1 - r["mae_total"] / max(r["mae_persistence"], 1e-9)
        rows.append(r)
    avg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("horizon_h",) and
           all(k in r for r in rows)} if rows else {}
    return {"per_horizon": rows, "average": avg}


def print_report(title: str, rep: dict, gauged: bool):
    if not rep["per_horizon"]:
        print(f"\n{title}: no valid test points")
        return
    a = rep["average"]
    print(f"\n=== {title} ===")
    base_lbl = "tide+30d-level" if gauged else "tide+nbr-level"
    print(f"Average MAE total {a['mae_total']:.4f} m (persistence {a['mae_persistence']:.4f}, "
          f"{base_lbl} {a['mae_tide_only']:.4f}, +IB {a['mae_tide_ib']:.4f}"
          + (f", tide+persist {a['mae_tide_persist']:.4f}" if gauged else "") + ")")
    print(f"Average RMSE total {a['rmse_total']:.4f} m (persistence {a['rmse_persistence']:.4f}) | "
          f"bias {a['bias_total']:+.4f} m")
    print(f"Surge MAE {a['mae_res']:.4f} m (zero-surge {a['mae_surge_zero']:.4f}, IB-only {a['mae_surge_ib']:.4f}) "
          f"| surge skill vs zero {1 - a['mae_res'] / max(a['mae_surge_zero'], 1e-9):+.1%}")
    if "cal_cov" in a:
        print(f"Raw cov {a['raw_cov']:.1%} (width {a['raw_width']:.3f} m) | "
              f"CQR cov {a['cal_cov']:.1%} (width {a['cal_width']:.3f} m) nominal 80%")
    print("Horizon |  MAE   |  RMSE  | Persist | SurgeMAE | RawCov | CalCov | RawW  | CalW  | q_hat")
    for r in rep["per_horizon"]:
        print(f" +{r['horizon_h']:2d} h  | {r['mae_total']:.4f} | {r['rmse_total']:.4f} | "
              f"{r['mae_persistence']:.4f}  |  {r['mae_res']:.4f}  | {r['raw_cov']:5.1%} | "
              f"{r.get('cal_cov', float('nan')):5.1%} | {r['raw_width']:.3f} | "
              f"{r.get('cal_width', float('nan')):.3f} | {r.get('q_hat', float('nan')):+.4f}")


def save_prediction_plot(e: dict, cal: CQR, horizons, station: str, path: Path, hours: int = 168):
    """v1-style plot: observed vs predicted TOTAL water level with raw and CQR 80 % bands, one panel per horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[info] matplotlib not installed - skipping prediction plot")
        return
    n = len(e["t"])
    plot_len = min(hours, n)
    start = 0
    for i in range(0, max(n - plot_len, 1), 24):       # first fully observed window
        if np.isfinite(e["y_tot"][i:i + plot_len, 0]).all():
            start = i
            break
    sl = slice(start, start + plot_len)
    grp = (e["cyc"] > 0.5).astype(int)
    lo_c, hi_c = cal.apply(e["q"][..., 0], e["q"][..., 2], grp)
    fig, axes = plt.subplots(len(horizons), 1, figsize=(12, 2.2 * len(horizons)), sharex=True)
    x = np.arange(plot_len)
    for h, (ax, hz) in enumerate(zip(np.atleast_1d(axes), horizons)):
        tide = e["base"][sl, h]
        ax.plot(x, e["y_tot"][sl, h], color="black", lw=1.4, label="Observed")
        ax.plot(x, tide + e["q"][sl, h, 1], color="tab:blue", ls="--", label="Predicted q50 (tide + surge)")
        ax.fill_between(x, tide + lo_c[sl, h], tide + hi_c[sl, h], color="tab:blue", alpha=0.12, label="CQR 80 %")
        ax.fill_between(x, tide + e["q"][sl, h, 0], tide + e["q"][sl, h, 2], color="tab:blue", alpha=0.25,
                        label="Raw q10-q90")
        ax.set_ylabel("Level (m)")
        ax.set_title(f"{station}: +{hz} h", fontsize=10)
        if h == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=2)
    np.atleast_1d(axes)[-1].set_xlabel("Hours from window start (test period)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot {path}")


def run_evaluation(ckpt: str, dataset: str = C.DEFAULT_DATASET, cache_dir: str = C.DEFAULT_CACHE_DIR,
                   out_dir: str | None = None, device: str = "auto", alpha: float = 0.2,
                   cqr_mode: str = "additive", mondrian: bool = True, cal_mask_prob: float = 0.3,
                   cal_passes: int = 3, test_stride: int = 1, val_stride: int = 3,
                   nbr_level_km: float = 800.0):
    dev = pick_device(device)
    ckpts = [ckpt] if isinstance(ckpt, (str, Path)) else list(ckpt)
    model, ck = load_model(ckpts[0], dev)
    a = ck["args"]
    cache = build_cache(dataset, cache_dir)
    data = SagarData(cache, ck["holdout"], history=a["history"], stats=ck["stats"],
                     use_future_forcing=not a["no_future_forcing"], target=a.get("target", "full"),
                     static_set=a.get("static_set", "all"))
    if nbr_level_km > 0 and data.holdout.any():
        g = ck.get("graph", {})
        if g.get("distance_km") is None and Path(a.get("graph", C.DEFAULT_GRAPH)).exists():
            g = load_graph(a.get("graph", C.DEFAULT_GRAPH), data.names)
        if g.get("distance_km") is not None and len(np.asarray(g["distance_km"]).reshape(-1)) == np.asarray(g["edge_index"]).shape[1]:
            for n, src in data.set_neighbour_level(g["edge_index"], g["distance_km"], nbr_level_km).items():
                print(f"slow level for ungauged {n}: {src}")
        else:
            print("[warn] graph has no per-edge distance_km; ungauged slow level stays 0")
    # ---- ensemble members: same holdout/target/data, each with its own static feature set ----
    members = [(model, None)]
    for c in ckpts[1:]:
        m2, ck2 = load_model(c, dev)
        a2 = ck2["args"]
        if sorted(ck2["holdout"]) != sorted(ck["holdout"]) or a2.get("target", "full") != a.get("target", "full")                 or a2["history"] != a["history"] or a2["no_future_forcing"] != a["no_future_forcing"]:
            raise ValueError(f"{c}: ensemble members must share holdout, target, history and forcing mode")
        st = None
        if a2.get("static_set", "all") != a.get("static_set", "all"):
            st = SagarData.static_for(cache, ck2["stats"], a2.get("static_set", "all"))
        members.append((m2, st))
    if len(members) > 1:
        print(f"Ensemble of {len(members)} checkpoints: " + ", ".join(str(c) for c in ckpts))

    def predict(_model, d, split, stride, device, mask_prob=0.0, seed=0):   # noqa: shadows module-level predict
        outs = [_predict_one(m, d, split, stride, device, mask_prob=mask_prob, seed=seed, static_override=st)
                for m, st in members]
        q = np.mean([o["q"] for o in outs], 0)          # mean of sorted quantile vectors stays non-crossing
        return {"q": q, "drop": outs[0]["drop"], "t": outs[0]["t"]}

    out = Path(out_dir or Path(ckpts[0]).parent)
    hz = list(C.HORIZONS)
    gauged_nodes = ~data.holdout
    has_tide = np.isfinite(data.tide).any(0)

    # ---- calibration on validation period ----
    pv = predict(model, data, "val", val_stride, dev)
    tv = gather_targets(data, pv["t"])
    eg = flatten(pv, tv, gauged_nodes)
    cal_g = CQR(alpha, cqr_mode, mondrian).fit(eg["q"][..., 0], eg["q"][..., 2], eg["y_res"], (eg["cyc"] > .5).astype(int))
    parts = []
    for k in range(cal_passes):   # pseudo-ungauged scores: mask gauged stations' history
        pm = predict(model, data, "val", val_stride, dev, mask_prob=cal_mask_prob, seed=1000 + k)
        parts.append(flatten(pm, gather_targets(data, pm["t"]), gauged_nodes, require_drop=True))
    eu = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    cal_u = CQR(alpha, cqr_mode, mondrian).fit(eu["q"][..., 0], eu["q"][..., 2], eu["y_res"], (eu["cyc"] > .5).astype(int))
    print(f"q_hat gauged   {np.round(cal_g.global_qhat(), 4).tolist()}")
    print(f"q_hat ungauged {np.round(cal_u.global_qhat(), 4).tolist()}")

    # ---- test ----
    pt = predict(model, data, "test", test_stride, dev)
    tt = gather_targets(data, pt["t"])
    results = {"checkpoint": [str(c) for c in ckpts], "holdout": ck["holdout"], "alpha": alpha, "cqr_mode": cqr_mode,
               "mondrian": mondrian, "q_hat_gauged": cal_g.global_qhat(), "q_hat_ungauged": cal_u.global_qhat()}

    rep_g = horizon_report(flatten(pt, tt, gauged_nodes & has_tide), cal_g, hz, gauged=True)
    print_report("GAUGED stations, test period", rep_g, gauged=True)
    results["gauged"] = rep_g

    if data.holdout.any():
        for j in np.flatnonzero(data.holdout):
            sel = np.zeros(data.N, bool); sel[j] = True
            e = flatten(pt, tt, sel)
            rep = horizon_report(e, cal_u, hz, gauged=False)
            print_report(f"UNGAUGED (LOSO) {data.names[j]}, test period", rep, gauged=False)
            results.setdefault("ungauged", {})[data.names[j]] = rep
            np.savez_compressed(out / f"pred_test_{data.names[j]}.npz",
                                time=data.times[e["t"]].astype("datetime64[s]").astype(str),
                                horizons=np.array(hz), q_res=e["q"], y_res=e["y_res"],
                                y_total=e["y_tot"], tide=e["tide"], base=e["base"])
            save_prediction_plot(e, cal_u, hz, data.names[j], out / f"prediction_plot_{data.names[j]}.png")
    else:   # no held-out station: report pseudo-ungauged test skill
        pm = predict(model, data, "test", test_stride, dev, mask_prob=cal_mask_prob, seed=7)
        rep = horizon_report(flatten(pm, gather_targets(data, pm["t"]), gauged_nodes & has_tide, True),
                             cal_u, hz, gauged=False)
        print_report("PSEUDO-UNGAUGED (masked history), test period", rep, gauged=False)
        results["pseudo_ungauged"] = rep

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out / 'metrics.json'}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, nargs="+",
                    help="one checkpoint, or several (same holdout) to evaluate their averaged ensemble")
    ap.add_argument("--dataset", default=C.DEFAULT_DATASET)
    ap.add_argument("--cache-dir", default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=None, help="default: folder of the first checkpoint (use a new one for ensembles)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--cqr-mode", choices=["additive", "scaled"], default="additive")
    ap.add_argument("--no-mondrian", action="store_true")
    ap.add_argument("--cal-mask-prob", type=float, default=0.3)
    ap.add_argument("--cal-passes", type=int, default=3)
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=3)
    ap.add_argument("--nbr-level-km", type=float, default=800.0,
                    help="ungauged slow level from gauged neighbours within this sea-route distance; 0 = off")
    a = ap.parse_args()
    run_evaluation(a.ckpt if len(a.ckpt) > 1 else a.ckpt[0], a.dataset, a.cache_dir, a.out_dir, a.device, a.alpha, a.cqr_mode,
                   not a.no_mondrian, a.cal_mask_prob, a.cal_passes, a.test_stride, a.val_stride,
                   a.nbr_level_km)


if __name__ == "__main__":
    main()
