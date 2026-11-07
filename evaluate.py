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
from utils import mask_history, pick_device, random_node_mask, to_device


def load_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = SagarNet(edge_index=ck["graph"]["edge_index"], edge_attr=ck["graph"]["edge_attr"], **ck["model_kwargs"])
    m.load_state_dict(ck["model_state"])
    return m.to(device).eval(), ck


@torch.no_grad()
def predict(model, data: SagarData, split: str, stride: int, device, batch_size=64, mask_prob=0.0, seed=0):
    ds = SagarWindows(data, split, stride)
    static = torch.from_numpy(data.static).to(device)
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


def gather_targets(data: SagarData, t: np.ndarray) -> dict:
    tgt = t[:, None] + data.horizons[None, :]                          # [S,H]
    nan_if = lambda a, m: np.where(m, a, np.nan)
    sea = nan_if(data.sea_level, data.valid)
    res = nan_if(data.resid, data.valid)
    sw = lambda a: a[tgt].transpose(0, 2, 1)                            # [S,H,N] -> [S,N,H]
    return {
        "y_res": sw(res), "y_tot": sw(sea), "tide": sw(data.tide),
        "ib": sw(data.ib_n * data.scale), "cyc": sw(data.cyc_active),
        "sea_now": sea[t], "res_now": res[t],                          # [S,N]
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
        y_r, y_t = e["y_res"][ok, h], e["y_tot"][ok, h]
        tide = e["tide"][ok, h]
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
    print(f"Average MAE total {a['mae_total']:.4f} m (persistence {a['mae_persistence']:.4f}, "
          f"tide-only {a['mae_tide_only']:.4f}, tide+IB {a['mae_tide_ib']:.4f}"
          + (f", tide+persist {a['mae_tide_persist']:.4f}" if gauged else "") + ")")
    print(f"Average RMSE total {a['rmse_total']:.4f} m (persistence {a['rmse_persistence']:.4f}) | "
          f"surge MAE {a['mae_res']:.4f} m | bias {a['bias_total']:+.4f} m")
    if "cal_cov" in a:
        print(f"Raw cov {a['raw_cov']:.1%} (width {a['raw_width']:.3f} m) | "
              f"CQR cov {a['cal_cov']:.1%} (width {a['cal_width']:.3f} m) nominal 80%")
    print("Horizon |  MAE   |  RMSE  | Persist | SurgeMAE | RawCov | CalCov | RawW  | CalW  | q_hat")
    for r in rep["per_horizon"]:
        print(f" +{r['horizon_h']:2d} h  | {r['mae_total']:.4f} | {r['rmse_total']:.4f} | "
              f"{r['mae_persistence']:.4f}  |  {r['mae_res']:.4f}  | {r['raw_cov']:5.1%} | "
              f"{r.get('cal_cov', float('nan')):5.1%} | {r['raw_width']:.3f} | "
              f"{r.get('cal_width', float('nan')):.3f} | {r.get('q_hat', float('nan')):+.4f}")


def run_evaluation(ckpt: str, dataset: str = C.DEFAULT_DATASET, cache_dir: str = C.DEFAULT_CACHE_DIR,
                   out_dir: str | None = None, device: str = "auto", alpha: float = 0.2,
                   cqr_mode: str = "additive", mondrian: bool = True, cal_mask_prob: float = 0.3,
                   cal_passes: int = 3, test_stride: int = 1, val_stride: int = 3):
    dev = pick_device(device)
    model, ck = load_model(ckpt, dev)
    a = ck["args"]
    cache = build_cache(dataset, cache_dir)
    data = SagarData(cache, ck["holdout"], history=a["history"], stats=ck["stats"],
                     use_future_forcing=not a["no_future_forcing"])
    out = Path(out_dir or Path(ckpt).parent)
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
    results = {"checkpoint": ckpt, "holdout": ck["holdout"], "alpha": alpha, "cqr_mode": cqr_mode,
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
                                y_total=e["y_tot"], tide=e["tide"])
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default=C.DEFAULT_DATASET)
    ap.add_argument("--cache-dir", default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--cqr-mode", choices=["additive", "scaled"], default="additive")
    ap.add_argument("--no-mondrian", action="store_true")
    ap.add_argument("--cal-mask-prob", type=float, default=0.3)
    ap.add_argument("--cal-passes", type=int, default=3)
    ap.add_argument("--test-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=3)
    a = ap.parse_args()
    run_evaluation(a.ckpt, a.dataset, a.cache_dir, a.out_dir, a.device, a.alpha, a.cqr_mode,
                   not a.no_mondrian, a.cal_mask_prob, a.cal_passes, a.test_stride, a.val_stride)


if __name__ == "__main__":
    main()
