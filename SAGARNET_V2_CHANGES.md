# SAGAR-Net v2: Changes, New Architecture and How to Run

This document covers the move from the v1 codebase described in `CLAUDE_PROMPT.md` to the v2 files in this folder. It explains **what changed**, **why**, **what the new architecture looks like**, and **how to run it**.

---

## 1. Summary

v1 reached a test MAE of **2.20 m** at Cochin, while plain persistence scored **0.21 m**. The calibrated intervals covered only **74%** of observations against an 80% target. The root causes were:

| # | Root cause in v1 | Effect | v2 fix |
|---|---|---|---|
| 1 | Target was raw total water level relative to each gauge's own **chart datum** | GATv2 pulled Cochin (mean 0.75 m) toward its neighbours (3–4 m), giving a ~2.2 m constant bias | Predict the **non-tidal residual (surge)**: total minus mean, trend and harmonic tide, fitted per station on training years only |
| 2 | The model never saw **past sea level** (inputs were only ERA5, IBTrACS and tide phase) | It could not even match persistence at gauged stations | Add the **observed residual history** as an input, with an availability mask |
| 3 | No mechanism to learn the ungauged case explicitly | Transfer to an ungauged node was untrained behaviour | **Masked-node training**: 30% of gauged stations have their history hidden at each step |
| 4 | Model selection and early stopping used the held-out station's validation data | LOSO results were optimistic (leakage) | Select only on **training stations** (gauged + pseudo-ungauged MAE) |
| 5 | Inverted-barometer loss had a fixed weight for every horizon | Helped +6 h, hurt +12 to +72 h (Stage-4 ablation) | Physics weights **decay with lead time**; IB acts mainly as a prior on unlabelled nodes |
| 6 | CQR was calibrated on a different score distribution from the ungauged test | 74% instead of 80% coverage | Calibrate on **pseudo-ungauged** validation scores; per-horizon and optional cyclone/calm (Mondrian) q̂ |
| 7 | Schema mismatch with the 34-station file (`elevation_m`, `storm_*`) | KeyError, or cyclone features silently dropped | Alias-based schema resolution for `gebco_*` and `cyclone_*` columns, with physical defaults |

### 1.1 Review of the actual v1 code (`sagar-main/`)

These findings come from reading the v1 source itself, not only `CLAUDE_PROMPT.md`.

| # | Location (v1) | Finding | Severity | v2 status |
|---|---|---|---|---|
| A | `sagar_dataset.py:266` | `__getitem__` returns `y_current`, which is never defined → **NameError**. The repo snapshot cannot run; the reported results came from an earlier version. | Crash | Fixed: `last` / `last_mask` are computed explicitly |
| B | `train.py:464` | A stray line containing only `x` → **NameError** after evaluation. | Crash | Fixed |
| C | `sagar_dataset.py:189-191` | Inputs are only `dynamic_cols + 6 harmonics`. **Sea level is never an input.** | Accuracy (major) | Residual history + mask channel |
| D | `sagar_dataset.py:130` | Target = raw `sea_level` (chart datum). | Accuracy (major, the 2.2 m bias) | Surge residual target |
| E | `train.py:395-418` | Early stopping, LR scheduler and checkpointing all use **Cochin's** validation MAE. | Leakage | Selection on training stations only |
| F | `train.py:207-237` | CQR is calibrated on Cochin's validation period. In a true ungauged setting those labels do not exist. | Leakage | Pseudo-ungauged calibration on training stations |
| G | `construct_graph.py:30-144` | `max_distance_km=1500`, plain k-NN, **no land check** (the corridor whitelist is commented out). Over-land edges are possible. Uses `elevation_m` → KeyError on the 34-station CSV. | Correctness / schema | Great-circle land check, `gebco_*` fallback |
| H | `loss.py:59` | Wind term divides by `(H_nodes + 10)`. GEBCO depths are negative, so the denominator can be ≈0 or negative → exploding or sign-flipped loss. | Bug | Removed; wind enters as stress features instead |
| I | `loss.py:103-113` | Gradient loss compares the *change* in level difference with the *absolute* pressure difference at t+h (mismatched quantities). | Bug | Compares residual difference with IB difference, the same quantity |
| J | `loss.py:161` | Mass loss penalises `delta - net_inflow`, which pulls every predicted change toward 0 (toward persistence) instead of enforcing continuity. | Design | Laplacian smoothness of the rate of change, decayed by horizon |
| K | `train.py:331` + `model.py:96-97` | Static features `[lat, lon, elev]` are raw, and `LayerNorm(3)` normalises *across* lat/lon/elev of one node (mixed units), which wipes out absolute position. | Accuracy | Standardised per feature across nodes; sin/cos lon; 3 GEBCO depths |
| L | `sagar_dataset.py:51-56` | Looks for `storm_*` names → all cyclone features silently dropped on the 34-station file. | Schema | `cyclone_*` with `storm_*` aliases |
| M | `sagar_dataset.py:98` | `ffill()` with no limit → a station gap of months is filled with a stale value. | Data quality | ffill/bfill limited to 6 h, then station climatology |
| N | `sagar_dataset.py:73` | `df.pivot` raises on duplicate (time, station) rows. | Robustness | Index-based pivot; last value wins |
| O | `sagar_dataset.py:154-155` | 70/15/15 split by row count, so period boundaries move with the data. | Reproducibility | Fixed calendar dates in `config.py` |
| P | `sagar_dataset.py:255-257` | Future head gets only scaled `msl` and `wind_speed` at t+h (8-dim). No cyclone, no wind direction, no preceding hours. | Accuracy | All 29 features at t+h−{0,3,6,12} |
| Q | `model.py:23` | TCN block applies ReLU before the residual add, so the residual branch can only increase activations. | Minor | ReLU moved after the add |
| R | `model.py:11` | BatchNorm1d statistics are mixed over all nodes in a batch. | Minor | GroupNorm |
| S | `train.py:236` | `np.quantile` default (linear interpolation) rather than the conservative "higher" conformal quantile. | Minor | `method="higher"` |

### 1.2 What was kept from v1
- The five-stage TCN → GRU → static → GATv2 → non-crossing quantile pipeline, hidden size 64, 4 heads, dilations 1–16, kernel 3.
- The TCN block layout: dilated conv → norm → ReLU → dropout → **1×1 conv** → norm, plus residual.
- The GRU residual fusion `LayerNorm(gru[-1] + tcn[-1])` and the GAT fusion `LN(Linear([h, h_spat]) + h)`.
- The softplus non-crossing quantile parameterisation.
- The tidal phase features `τ = t + lon/15` (S1, M2, S2), now with K1 and O1 added.
- Graph edge weights `sim_geo` (σ = 750 km) and `sim_bath` (γ = 25 m).
- AdamW (wd 1e-4), grad clip 1.0, ReduceLROnPlateau(×0.5), LOSO by station name.
- The no-graph GRU baseline (`--no_gat`), the quick mode (`--quick`), and the 7-day prediction plot per horizon.
- **All v1 command-line flags work as aliases** (see §9).

---

## 2. File-by-file changes

| File | Status | What it does / what changed |
|---|---|---|
| `config.py` | **new** | One place for constants: paths, SI physics constants (ρw = 1025, g = 9.80665, ρw·g ≈ 10051.8 Pa/m), horizons, quantiles, temporal splits, tide constituents, column aliases, cyclone defaults. |
| `tides.py` | **new** | (a) Cyclical tidal phase features from spec 3.1 with τ = t + lon/15 (S1, M2, S2, plus K1 and O1). (b) Robust least-squares **harmonic analysis** with 17 constituents (M2, S2, N2, K2, K1, O1, P1, Q1, S1, M4, MS4, MN4, M6, Mf, Mm, Ssa, Sa) plus mean and linear trend. |
| `construct_graph.py` | **rewritten** | Elevation column fallback chain `gebco_mean_5km_m → gebco_nearest_m → gebco_min_5km_m → elevation_m`. Reads the 3M-row CSV in chunks. Samples each candidate edge along its great circle and rejects it if it crosses more than 40 km of continuous land (small islands are allowed, peninsulas are not). Edges are reciprocal. Isolated nodes get one longer marine link. Edge attributes are now `[sim_geo, sim_bath, sin(bearing), cos(bearing)]`. Writes `static_graph_edges.csv` for auditing. |
| `sagar_dataset.py` | **rewritten** | `cyclone_*` schema with legacy `storm_*` aliases. Cleans the GESLA missing-value sentinel (−99.9999). Gap-fills forcing. Computes physics-derived features. Runs a train-period tide fit per station and builds the surge residual target. Despikes labels. Caches everything to `cache/*.npz`. Window dataset with history and mask channels. |
| `model.py` | **rewritten** | Same five stages. GroupNorm replaces BatchNorm. Graph attention is now two stacked GATv2 blocks in pure PyTorch (no torch-geometric). One shared horizon head with a horizon embedding. Learned persistence skip. |
| `loss.py` | **rewritten** | Pinball loss, plus horizon-decayed Huber IB, spatial-gradient and mass-continuity terms, with a warm-up ramp. |
| `calibration.py` | **new** | Per-horizon CQR with an additive or scaled score, optional Mondrian groups (cyclone vs calm), and a finite-sample conformal quantile. |
| `train.py` | **rewritten** | AdamW, gradient clipping at 1.0, ReduceLROnPlateau, early stopping, EMA weights, AMP, masked-node training, leak-free model selection. Runs evaluation automatically at the end. |
| `evaluate.py` | **new** | Fits CQR on the validation period (gauged and pseudo-ungauged), evaluates on the test period, prints per-horizon tables and baselines, and writes `metrics.json` and prediction files. |
| `loso.py` | **new** | Runs the full Leave-One-Station-Out sweep and writes `runs/loso_summary.csv`. |
| `utils.py` | **new** | Seeding, device selection, graph/dataset order check, node-masking helpers. |
| `requirements.txt` | **new** | numpy, pandas, torch, global-land-mask, matplotlib. torch-geometric is **no longer needed**. |

---

## 3. New data pipeline

### 3.1 Target decomposition (the chart-datum fix)

For every station *j*, using **training-period data only**:

```
sea_level_j(t) = mean_j + trend_j·t + Σ_k [a_jk cos(ω_k t) + b_jk sin(ω_k t)] + residual_j(t)
                 └──────────────────── tide_j(t) ─────────────────────────┘
```

- The network predicts `residual_j(t+h)`, which is zero-centred, has no datum and transfers between stations.
- Reported total water level is `tide_j(t+h) + predicted residual`.
- The fit uses iterative outlier rejection: points above max(8·MAD, 1 m) are dropped from the fit only. Real surges stay in the data.
- A station with too little training data falls back to train+val data, never test data.
- Residual targets go through a light despike: deviations from a 7-hour rolling median above max(8·MAD, 0.75 m) are masked.
- At an **ungauged** site, the operational tide would come from a global tide model (FES2014/TPXO). For evaluation, the held-out gauge's training-years harmonic fit stands in for it. The network never sees this fit.

### 3.2 Input features (per node, per hour), 29 dynamic + 2 observation channels

| Group | Features | Reason |
|---|---|---|
| ERA5 (5) | u10, v10, msl anomaly (hPa vs station train mean), tp (mm), wind_speed | Base meteorological forcing |
| Physics-derived (4) | wind stress τx, τy = ρa·Cd·\|U\|·U; **IB height** = −(msl − msl_ref)/(ρw·g); 3-hour pressure tendency | Surge responds to stress (∝U²) and pressure, not raw wind |
| IBTrACS (8) | active, proximity = active·exp(−d/500 km), log(1+d), wind (kt), pressure deficit (1013.25 − p), Δlat, Δlon of the eye relative to the station, wind·proximity | Encodes cyclone intensity **and** position relative to each station |
| Tide phase (10) | sin/cos for S1, M2, S2, K1, O1 with τ = t + lon/15 | From spec 3.1; K1 and O1 added because diurnal tides are strong in SE Asia |
| Calendar (2) | day-of-year sin/cos | Monsoon seasonality |
| **Observation (2)** | observed residual × mask, mask | Gives the model its own history; the mask tells it whether the node is gauged |

Cyclone defaults when no cyclone is active: P = 1013.25 hPa, d = 10000 km, active = 0, cyclone lat/lon = station lat/lon (offset 0).

Static node features (7): lat, lon, sin(lon), cos(lon), signed-log of `gebco_mean_5km_m`, `gebco_nearest_m` and `gebco_min_5km_m`. The station datum is **deliberately excluded** because it is unknown at ungauged sites.

### 3.3 Normalisation (leak-free)
- Dynamic features are z-scored with training-period statistics over all nodes. ERA5 exists at ungauged sites, so this is legitimate. Binary, tide and calendar features are left as they are.
- The residual scale σ comes from **training stations and training period only**, excluding the holdout.

### 3.4 Splits
| Split | Period |
|---|---|
| Train | start → 2019-12-31 23:00 |
| Validation | 2020-01-01 → 2021-12-31 |
| Test | 2022-01-01 → 2022-12-31 (the data ends here) |

The window history may reach back into the previous split (inputs only). All targets of a sample lie inside its own split.

---

## 4. New architecture

```
Inputs per sample (whole graph at anchor time t):
  x_hist  [B, N=34, L=72, 31]   29 dynamic features + observed residual + mask
  fut     [B, N, 5 horizons, 29×4]  features at t+h−{0,3,6,12} h for each horizon h
  static  [N, 7]
  last    [B, N]  residual(t) × mask

 ┌──────────────────────────────────────────────────────────────────┐
 │ 1. Multiscale dilated causal TCN encoder                         │
 │    Linear(31→64) → 5 × CausalConv1d(k=3, d=1,2,4,8,16)           │
 │    each: pad → conv → GN → ReLU → Drop → 1×1 conv → GN → +x → ReLU│
 │    receptive field 63 h, no future leakage                        │
 └───────────────────────────────┬──────────────────────────────────┘
 ┌───────────────────────────────▼──────────────────────────────────┐
 │ 2. Residual GRU                                                  │
 │    h_temp = LayerNorm(GRU(h_tcn)[-1] + h_tcn[-1])                 │
 └───────────────────────────────┬──────────────────────────────────┘
 ┌───────────────────────────────▼──────────────────────────────────┐
 │ 3. Static geographic context                                     │
 │    h = h_temp + MLP(LayerNorm(static))                            │
 └───────────────────────────────┬──────────────────────────────────┘
 ┌───────────────────────────────▼──────────────────────────────────┐
 │ 4. Audited coastal graph: 2 × GraphBlock                         │
 │    EdgeGATv2(64 → 16×4 heads, edge_dim=5 incl. self-loop flag)   │
 │    h = LN(h + Linear([h, ELU(gat(h))]))      (spec fusion rule)   │
 │    h = LN(h + FFN(h))                                             │
 │    edges: k=4 marine neighbours ≤ 2200 km, reciprocal, + self     │
 └───────────────────────────────┬──────────────────────────────────┘
 ┌───────────────────────────────▼──────────────────────────────────┐
 │ 5. Shared multi-horizon non-crossing quantile head               │
 │    z_h = [h, MLP(fut_h), horizon_embedding_h] → MLP → raw(3)      │
 │    q50 = raw0 + sigmoid(α_h)·last·mask      (persistence skip)    │
 │    q10 = q50 − softplus(raw1);  q90 = q50 + softplus(raw2)        │
 │    ⇒ q10 ≤ q50 ≤ q90 guaranteed                                   │
 └──────────────────────────────────────────────────────────────────┘
Output: residual quantiles [B, N, 5, 3]   (×σ → metres; + tide → total level)
```

### 4.1 Architecture decisions and reasons

| Decision | Reason |
|---|---|
| Kept TCN → GRU → static → GATv2 → quantile heads | Same research identity as v1; each stage still does the job it did there |
| **GroupNorm** instead of BatchNorm1d | Batch statistics mixed gauged and masked nodes and shifted between training and single-station evaluation. GroupNorm normalises each sequence independently. |
| **2 graph layers** instead of 1 | Two hops let an ungauged node draw on neighbours of its neighbours, which helps in a sparse k=4 graph |
| **Pure-PyTorch GATv2** with self-loops and an `is_self` edge flag | Same maths as PyG `GATv2Conv` (Brody et al. 2022) with no torch-geometric install on Windows. The self-loop lets a node keep its own state. |
| **Bearing** edge features | Surge propagates along the coast in a direction; attention can now tell upstream neighbours from downstream ones |
| **Shared head + horizon embedding** instead of 5 separate heads | 5× more training signal per parameter; smooth behaviour across horizons |
| **Stacked future forcing** at lags 0/3/6/12 h before each target | Surge at t+h depends on wind and pressure over the preceding hours, not only at t+h |
| **Persistence skip** α_h, initialised to exp(−h/24) | At gauged nodes the model starts from persistence and learns the correction. At ungauged nodes mask = 0, so the skip has no effect. |
| Softplus non-crossing quantiles | Kept from v1; guarantees q10 ≤ q50 ≤ q90 |

---

## 5. Loss function

```
L_total = L_pinball + s(epoch)·[ λ_IB·L_IB + λ_grad·L_grad + λ_mass·L_mass ]
```

| Term | Formula (normalised residual units) | Default λ | Reason for the change |
|---|---|---|---|
| Pinball | τ ∈ {0.1, 0.5, 0.9}, masked to labelled **training** stations | 1 | Holdout is never in the loss |
| IB | Huber(q50 − η_IB) · w_h · node_w, with η_IB = −(msl − msl_ref)/(ρw·g) | 0.02 | node_w = 1 on unlabelled nodes and 0.1 on labelled ones, so it works as a **semi-supervised prior for ungauged coasts** |
| Spatial gradient | Huber((q50_i − q50_j) − (η_IB,i − η_IB,j)) · sim_geo · w_h | 0.01 | Sea-surface slope along marine edges should follow the pressure slope |
| Mass continuity | Σ_edges sim_geo · (∂q50_i/∂h − ∂q50_j/∂h)² | 0.002 | Graph-smoothness proxy for flux divergence |

- **Horizon decay** w_h = exp(−h/18): physics is trusted at short lead times and relaxed at long ones, which is the direct fix for the Stage-4 ablation result.
- **Warm-up** s(epoch): 0 for the first 2 epochs, then ramps linearly to 1 over 2 epochs.
- **Huber** instead of MSE, so cyclone extremes do not dominate the gradients.

---

## 6. Training procedure

1. Sample random anchor times from the training period (`--steps-per-epoch × --batch-size` per epoch).
2. Hide the history of the holdout station, always. Also hide the history of a random 30% of gauged stations (`--mask-prob`). At least one station per sample is masked.
3. Compute the loss on every labelled training station, including the masked ones. That is how the model learns transfer.
4. AdamW (lr 1e-3, weight decay 1e-4), gradient clipping max_norm = 1.0, optional AMP.
5. Keep an EMA of the weights (decay 0.999). The EMA model is what gets validated and saved.
6. **Validation score** = ½·(gauged MAE + pseudo-ungauged MAE) on training stations over 2020–21. The holdout is never used.
7. ReduceLROnPlateau (×0.5, patience 1); early stopping with patience 3 (from the spec).
8. Save `runs/<run>/best.pt` (weights, config, normalisation stats, graph, station order). Evaluation then runs automatically.

---

## 7. Uncertainty calibration (CQR)

1. Run the model on the validation period.
2. **Gauged calibrator:** nonconformity scores s = max(q10 − y, y − q90) on training stations with their history visible.
3. **Ungauged calibrator:** run 3 passes, each hiding the history of a random 30% of training stations, and collect scores **only on the hidden stations**. These scores look like the LOSO test case.
4. Per horizon: q̂ = Quantile(s, ⌈(n+1)(1−α)⌉/n), α = 0.2.
5. **Mondrian** (on by default): a separate q̂ for cyclone-active vs calm targets, when a group has at least 200 points. Otherwise the global q̂ is used.
6. Calibrated interval: [q10 − q̂, q90 + q̂] (additive). With `--cqr-mode scaled` it is [q10 − q̂·w, q90 + q̂·w], where w = q90 − q10.
7. The ungauged calibrator is applied to the holdout station; the gauged calibrator to training stations.

> Coverage is **empirical, not a strict guarantee**. Time-series dependence and spatial shift break exchangeability, so report it as observed coverage.

---

## 8. Evaluation outputs

For each horizon (+6, +12, +24, +48, +72 h), two groups are reported:
- **UNGAUGED (LOSO)**: each held-out station. Its history was never visible.
- **GAUGED**: training stations in the 2022 test year.

| Metric | Meaning |
|---|---|
| `mae_total`, `rmse_total`, `bias_total` | Total water level error = (tide + predicted surge) − observation |
| `mae_res`, `rmse_res` | Surge (residual) error. This is the main ungauged-transfer metric. |
| `mae_persistence` | y(t+h) = y(t). Shown for ungauged stations only as a reference to compare with v1; it needs a gauge. |
| `mae_tide_only` | Harmonic tide with zero surge |
| `mae_tide_ib` | Tide + inverted barometer (physics-only baseline) |
| `mae_tide_persist` | Tide(t+h) + residual(t). The strongest simple gauged baseline. |
| `raw_cov`, `raw_width` | Coverage and width of the uncalibrated q10–q90 interval |
| `cal_cov`, `cal_width`, `q_hat` | After CQR (nominal 80%) |
| `mae_total_cyclone`, `cal_cov_cyclone` | Same metrics restricted to cyclone-active targets |
| `skill_vs_persistence` | 1 − MAE_model / MAE_persistence |

Files written to `runs/<run>/`:
- `best.pt`: checkpoint
- `history.json`: loss components and validation MAE per epoch
- `metrics.json`: all metrics above
- `pred_test_<Station>.npz`: time, horizons, residual quantiles, observed residual, observed total, tide (for plotting)
- `prediction_plot_<Station>.png`: v1-style 7-day plot per horizon showing observed vs predicted total level, with raw and CQR bands

`runs/loso_summary.csv` holds the per-station LOSO averages (from `loso.py`).

**Synthetic smoke test** (6 stations, held-out station with a 0.75 m datum among 3–4 m neighbours, 12 short epochs): ungauged MAE 0.026 m vs tide-only 0.050 m and tide+IB 0.043 m; bias −0.01 m (the datum bias is gone); CQR coverage 82.8%. This shows the pipeline works. It does not predict results on the real data.

---

## 9. How to run

All commands run from the project root (the folder containing `combined34/`) on the project laptop.

### Step 0: Copy files and install
Copy every `.py` file plus `requirements.txt` from this folder into the project root, replacing the v1 `construct_graph.py`, `sagar_dataset.py`, `model.py`, `loss.py` and `train.py`. Keep a backup of v1 if you want to compare.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected layout:
```
project/
├─ combined34/34_stations_combined_with_IBTrACs.csv
├─ config.py  tides.py  utils.py
├─ construct_graph.py  sagar_dataset.py  model.py  loss.py  calibration.py
├─ train.py  evaluate.py  loso.py
└─ requirements.txt
```

### Step 1: Build the audited marine graph
```powershell
.\.venv\Scripts\python.exe construct_graph.py --dataset combined34/34_stations_combined_with_IBTrACs.csv --output static_graph.pt --k-neighbors 4
```
Check the console output:
- `components: 1` means the graph is connected. If it is not, the components are listed.
- `Rejected over-land candidates` lists the land-crossing edges that were dropped.
- `sim_bath range`: if it is about 0 for most edges, rebuild with `--gamma-bath 200`.
- Open `static_graph_edges.csv` and confirm there are no links across India or the Malay Peninsula.

### Step 2: Train one LOSO fold (Cochin)
```powershell
.\.venv\Scripts\python.exe train.py --holdout Cochin --amp
```
- The first run parses the CSV once and prints, per station, the mean level, tide-fit source and residual std. It then caches to `cache/`.
- Each epoch prints the validation MAE for gauged and ungauged nodes. `*` marks a new best.
- Calibration and the test report run automatically at the end.

### Step 3: Re-evaluate or recalibrate without retraining
```powershell
.\.venv\Scripts\python.exe evaluate.py --ckpt runs/loso_Cochin/best.pt
.\.venv\Scripts\python.exe evaluate.py --ckpt runs/loso_Cochin/best.pt --cqr-mode scaled
```

### Step 4: Full LOSO sweep (all 34 stations)
```powershell
.\.venv\Scripts\python.exe loso.py -- --amp
.\.venv\Scripts\python.exe loso.py --stations Cochin Vishakhapatnam Penang -- --epochs 30 --amp
```
Folds that already have `metrics.json` are skipped, so the sweep can be resumed.

### Step 5: Optional experiments
| Purpose | Command |
|---|---|
| Operational (no oracle future weather) | `train.py --holdout Cochin --no-future-forcing --run-name cochin_nofuture` |
| Ablation: no physics | `train.py --holdout Cochin --lam-ib 0 --lam-grad 0 --lam-mass 0 --run-name cochin_nophys` |
| Ablation: v1-like single graph layer | `train.py --holdout Cochin --gnn-layers 1 --run-name cochin_1gnn` |
| Ablation: no masked-node training | `train.py --holdout Cochin --mask-prob 0 --run-name cochin_nomask` |
| Emphasise ungauged transfer | `train.py --holdout Cochin --mask-prob 0.5 --masked-weight 1.5` |
| Longer, more patient training | `train.py --holdout Cochin --patience 6 --epochs 100` |
| All stations gauged (no holdout) | `train.py --holdout "" --run-name all_gauged` |
| Rebuild cache after editing CSV/splits | add `--rebuild-cache` |

### v1 commands still work
| v1 command / flag | v2 equivalent |
|---|---|
| `train.py --station Cochin` | same (alias of `--holdout`) |
| `--csv <file>` | same (alias of `--dataset`) |
| `--lambda_phys / --lambda_grad / --lambda_mass` | same (aliases of `--lam-ib / --lam-grad / --lam-mass`) |
| `--no_gat` | same (= `--gnn-layers 0`, GRU-only baseline) |
| `--quick` | same (3 epochs × 100 steps) |
| `--epochs --patience --lr --graph` | same |
| output `best_model_Cochin.pt` | `runs/loso_Cochin/best.pt` |
| output `prediction_plot_Cochin.png` | `runs/loso_Cochin/prediction_plot_Cochin.png` (now total level = tide + surge, with raw and CQR bands) |
| graph keys `node_names`, `node_coords`, `node_elevations` | still saved in `static_graph.pt`, alongside the v2 keys |

A v1 `static_graph.pt` is rejected with a clear message: rebuild it, because v2 needs the bearing edge features.

### Main `train.py` arguments
| Argument | Default | Notes |
|---|---|---|
| `--dataset` | `combined34/34_stations_combined_with_IBTrACs.csv` | |
| `--graph` | `static_graph.pt` | Must be built from the same CSV (station order is checked) |
| `--holdout` | `Cochin` | One or more station names |
| `--hidden / --heads / --gnn-layers` | 64 / 4 / 2 | |
| `--epochs / --steps-per-epoch / --batch-size` | 60 / 800 / 32 | Lower batch size if GPU memory is short |
| `--lr / --weight-decay / --clip` | 1e-3 / 1e-4 / 1.0 | |
| `--patience` | 3 | |
| `--mask-prob` | 0.3 | Fraction of gauged nodes hidden per step |
| `--lam-ib / --lam-grad / --lam-mass` | 0.02 / 0.01 / 0.002 | |
| `--phys-decay-h / --phys-warmup` | 18 / 2 | |
| `--amp` | off | Mixed precision on CUDA |
| `--num-workers` | 0 | Keep 0 on Windows unless loading is slow |

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `global-land-mask is required` | `pip install global-land-mask`, or `--land-check none` (not recommended) |
| `Graph/dataset station order mismatch` | Rebuild `static_graph.pt` from the same CSV |
| `Unknown holdout station(s)` | Use the exact name from the CSV, e.g. `Vishakhapatnam`, `Tanjong_Pagar` |
| `too little data for harmonic analysis` | That station has under 60 days of valid training data. It stays in the graph as forcing-only, with no targets. |
| CUDA out of memory | `--batch-size 16` and/or `--amp` |
| Slow first run | Normal: the CSV is parsed once, and later runs load `cache/*.npz` in seconds |
| Old v1 checkpoint fails to load | Expected: inputs and GAT implementation changed, so retrain |

---

## 11. Known limitations and next steps

- **Nodal corrections:** the harmonic fit ignores the 18.6-year nodal modulation, which leaves a few cm of slow drift. Swap in `utide` inside `tides.py` for a full analysis.
- **Ungauged tide:** total water level at a truly ungauged site needs a global tide model (FES2014/TPXO). Ungauged surge MAE is the clean metric.
- **Oracle forcing:** the default feeds reanalysis (ERA5/IBTrACS) at future times, which is a hindcast setting. Report `--no-future-forcing` results too for an operational comparison.
- **GESLA datum jumps:** sudden instrument datum shifts are not detected automatically. Check the per-station residual std printed at cache build.
- **Coverage:** CQR coverage is empirical under time dependence and spatial shift.
