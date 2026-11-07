"""Train SAGAR-Net v2 (one LOSO fold), then calibrate + evaluate.

    .\\.venv\\Scripts\\python.exe train.py --holdout Cochin --run-name loso_cochin

Training recipe
---------------
* AdamW, grad clip 1.0, ReduceLROnPlateau, early stopping (patience 3 by default).
* EMA of weights (decay 0.999) is what gets validated and saved.
* Masked-node training: each step hides the observed history of a random
  ``--mask-prob`` fraction of gauged stations, and the loss is still computed on
  them -> the model learns ungauged transfer from the graph + forcing.
* The held-out station is never in the loss, never exposes history, and is NOT
  used for model selection (v1 selected on it -> optimistic LOSO).
  Model selection uses the mean of gauged MAE and pseudo-ungauged MAE on the
  training stations over the validation period.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, RandomSampler

import config as C
from loss import SagarLoss
from model import SagarNet
from sagar_dataset import SagarData, SagarWindows, build_cache
from utils import load_graph, mask_history, pick_device, random_node_mask, set_seed, to_device


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # v1 flag names are accepted as aliases: --csv, --station, --lambda_phys, --lambda_grad, --lambda_mass,
    # --no_gat, --quick
    ap.add_argument("--dataset", "--csv", dest="dataset", default=C.DEFAULT_DATASET)
    ap.add_argument("--graph", default=C.DEFAULT_GRAPH)
    ap.add_argument("--cache-dir", default=C.DEFAULT_CACHE_DIR)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--holdout", "--station", dest="holdout", nargs="*", default=["Cochin"],
                    help="station(s) treated as ungauged; '' for none")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-dir", default="runs")
    # model
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--gnn-layers", type=int, default=2, help="0 = no graph (v1 --no_gat GRU baseline)")
    ap.add_argument("--no_gat", "--no-gat", action="store_true", help="alias for --gnn-layers 0")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--history", type=int, default=C.HISTORY_HOURS)
    ap.add_argument("--target", choices=["surge", "full"], default="surge",
                    help="surge: forecast residual minus 30-day mean (default); full: whole residual")
    ap.add_argument("--no-future-forcing", action="store_true",
                    help="do not feed ERA5/IBTrACS at future times (only tide/calendar); realistic operational mode")
    # optimisation
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--train-stride", type=int, default=1)
    ap.add_argument("--val-stride", type=int, default=6)
    ap.add_argument("--mask-prob", type=float, default=0.3)
    ap.add_argument("--masked-weight", type=float, default=1.0, help="loss weight multiplier for masked gauged nodes")
    # physics
    ap.add_argument("--lam-ib", "--lambda_phys", dest="lam_ib", type=float, default=0.02)
    ap.add_argument("--lam-grad", "--lambda_grad", dest="lam_grad", type=float, default=0.01)
    ap.add_argument("--lam-mass", "--lambda_mass", dest="lam_mass", type=float, default=0.002)
    ap.add_argument("--phys-decay-h", type=float, default=18.0)
    ap.add_argument("--phys-warmup", type=int, default=2, help="epochs of pure pinball before physics ramps in")
    # misc
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--quick", action="store_true", help="smoke run: 3 epochs x 100 steps (like v1 --quick)")
    a = ap.parse_args(argv)
    a.holdout = [h for h in a.holdout if h]
    if a.no_gat:
        a.gnn_layers = 0
    if a.quick:
        a.epochs, a.steps_per_epoch, a.val_stride = 3, 100, 24
    return a


@torch.no_grad()
def validate(model, loader, data, device, mask_prob, seed=0):
    """Returns (gauged MAE, pseudo-ungauged MAE) in metres over training stations."""
    model.eval()
    protect = torch.as_tensor(data.holdout, device=device)
    g = torch.Generator().manual_seed(seed)
    err_g = n_g = err_u = n_u = 0.0
    for b in loader:
        b = to_device(b, device)
        lab = b["ymask"] & ~protect.view(1, -1, 1)
        q = model(b["x"], b["fut"], b["static"], b["last"], b["last_mask"])
        e = (q[..., 1] - b["y"]).abs() * data.scale
        err_g += float((e * lab).sum()); n_g += float(lab.sum())
        drop = random_node_mask(b["x"].size(0), data.N, mask_prob, protect, device, g)
        bm = mask_history(dict(b), drop)
        q = model(bm["x"], bm["fut"], bm["static"], bm["last"], bm["last_mask"])
        e = (q[..., 1] - b["y"]).abs() * data.scale
        lab_u = lab & drop.unsqueeze(-1)
        err_u += float((e * lab_u).sum()); n_u += float(lab_u.sum())
    return err_g / max(n_g, 1), err_u / max(n_u, 1)


def collate_with_static(static: torch.Tensor):
    def fn(items):
        out = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        out["static"] = static
        return out
    return fn


def main(argv=None):
    args = get_args(argv)
    set_seed(args.seed)
    device = pick_device(args.device)
    run = args.run_name or ("loso_" + "_".join(args.holdout) if args.holdout else "all_gauged")
    out = Path(args.out_dir) / run
    out.mkdir(parents=True, exist_ok=True)

    cache = build_cache(args.dataset, args.cache_dir, force=args.rebuild_cache)
    data = SagarData(cache, args.holdout, history=args.history, use_future_forcing=not args.no_future_forcing,
                     target=args.target)
    graph = load_graph(args.graph, data.names)
    print(f"Device {device} | stations {data.N} | holdout {args.holdout} | residual scale {data.scale:.4f} m")
    print(f"Input dim {data.in_dim} | future dim {data.fut_dim} | static dim {data.static_dim}")

    static = torch.from_numpy(data.static)
    collate = collate_with_static(static)
    ds_tr = SagarWindows(data, "train", args.train_stride)
    ds_va = SagarWindows(data, "val", args.val_stride)
    sampler = RandomSampler(ds_tr, replacement=True, num_samples=args.steps_per_epoch * args.batch_size)
    dl_tr = DataLoader(ds_tr, args.batch_size, sampler=sampler, collate_fn=collate,
                       num_workers=args.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    dl_va = DataLoader(ds_va, args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)
    print(f"train anchors {len(ds_tr)} | val anchors {len(ds_va)}")

    model_kwargs = dict(in_dim=data.in_dim, fut_dim=data.fut_dim, static_dim=data.static_dim,
                        horizons=list(C.HORIZONS), hidden=args.hidden, heads=args.heads,
                        gnn_layers=args.gnn_layers, dropout=args.dropout)
    model = SagarNet(edge_index=graph["edge_index"], edge_attr=graph["edge_attr"], **model_kwargs).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=False)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    crit = SagarLoss(C.QUANTILES, C.HORIZONS, args.lam_ib, args.lam_grad, args.lam_mass, args.phys_decay_h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    protect = torch.as_tensor(data.holdout, device=device)

    best, bad, history = float("inf"), 0, []
    ckpt_path = out / "best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        phys = 0.0 if epoch <= args.phys_warmup else min(1.0, (epoch - args.phys_warmup) / 2.0)
        t0, run_loss, parts_acc = time.time(), 0.0, {}
        for b in dl_tr:
            b = to_device(b, device)
            drop = random_node_mask(b["x"].size(0), data.N, args.mask_prob, protect, device)
            b = mask_history(b, drop)
            w = (b["ymask"] & ~protect.view(1, -1, 1)).float()
            if args.masked_weight != 1.0:
                w = w * (1 + (args.masked_weight - 1) * drop.float().unsqueeze(-1))
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                q = model(b["x"], b["fut"], b["static"], b["last"], b["last_mask"])
            loss, parts = crit(q.float(), b["y"], w, b["ib"], model.edge_index, model.edge_weight, phys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            run_loss += float(loss)
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0.0) + v
        n = len(dl_tr)
        mae_g, mae_u = validate(ema.module, dl_va, data, device, args.mask_prob, seed=args.seed)
        score = 0.5 * (mae_g + mae_u)
        sched.step(score)
        rec = dict(epoch=epoch, loss=run_loss / n, **{k: v / n for k, v in parts_acc.items()},
                   val_mae_gauged=mae_g, val_mae_ungauged=mae_u, val_score=score,
                   lr=opt.param_groups[0]["lr"], secs=time.time() - t0)
        history.append(rec)
        flag = ""
        if score < best - 1e-5:
            best, bad, flag = score, 0, " *"
            torch.save({"model_state": ema.module.state_dict(), "model_kwargs": model_kwargs,
                        "stats": data.stats, "args": vars(args), "station_names": data.names,
                        "dyn_names": data.dyn_names, "holdout": args.holdout,
                        "graph": {"edge_index": graph["edge_index"], "edge_attr": graph["edge_attr"]},
                        "epoch": epoch, "val_score": score}, ckpt_path)
        else:
            bad += 1
        print(f"ep {epoch:3d} | loss {rec['loss']:.4f} | val MAE gauged {mae_g:.4f} m "
              f"ungauged {mae_u:.4f} m | lr {rec['lr']:.1e} | {rec['secs']:.0f}s{flag}")
        if bad >= args.patience:
            print(f"Early stopping (no improvement for {args.patience} epochs)")
            break

    (out / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Best validation score {best:.4f} m -> {ckpt_path}")
    if not args.skip_eval:
        from evaluate import run_evaluation
        run_evaluation(str(ckpt_path), args.dataset, args.cache_dir, str(out), device=str(device))


if __name__ == "__main__":
    main()
