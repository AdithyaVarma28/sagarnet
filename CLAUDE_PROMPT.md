# SAGAR-Net: Physics-Guided Spatio-Temporal Graph Learning for Coastal Flood & Storm-Surge Forecasting

---

## 1. PROJECT OVERVIEW & RESEARCH IDENTITY

- **Project Title**: SAGAR-Net: Physics-Guided Spatio-Temporal Graph Learning for Coastal Flood Forecasting at Ungauged Coasts
- **Course**: 22AIE498 – Project Phase 1 (Mid Review) | **Team ID**: 41
- **Institution**: Amrita Vishwa Vidyapeetham, Amritapuri Campus
- **Guide**: Dr. Simi Surendran
- **Team Members**:
  - K. Madhuri (AM.SC.U4AIE23117)
  - M. Sri Lakshmi Snigdha (AM.SC.U4AIE23127)
  - W. Venkat Ashrith (AM.SC.U4AIE23162)

### 1.1 Motivation & Problem Statement
India and the Northern Indian Ocean basin feature over 7,500 km of coastline with ~5,700 km exposed to severe cyclonic storms and surge-induced coastal inundation. Over 40% of the regional coastal population lives within 100 km of the shoreline. Despite high vulnerability, physical tide gauges are sparsely distributed, leaving vast stretches of coastline completely **ungauged**.

Existing forecasting methodologies face critical bottlenecks:
1. **Numerical Hydrodynamic Models (ADCIRC, SLOSH, Delft3D)**: Accurate but computationally prohibitive for rapid real-time operational ensemble early warning.
2. **Pure Data-Driven Models (CNN, LSTM)**: Strong temporal sequence modeling, but tied to gauged stations with zero spatial transferability to ungauged coasts.
3. **Existing GNN Approaches**: Often trained on dense, unphysical graphs or synthetic datasets without physical consistency or calibrated uncertainty bounds.

### 1.2 Core Research Question
*How can coastal total water levels and storm surges be accurately forecast at coastal locations without tide-gauge observations (ungauged coasts) across multi-hour horizons (6h to 72h) by transferring spatio-temporal dynamics over a physically constrained marine graph with distribution-free uncertainty calibration?*

### 1.3 Key Research Contributions & Novelty
- **Gauged-to-Ungauged Spatial Transfer**: Formulated via Leave-One-Station-Out (LOSO) cross-validation over an audited marine graph topology.
- **Compound Forcing Integration**: Fuses sea-level history (GESLA-3), atmospheric synoptic forcing (ERA5), coastal bathymetry/elevation (GEBCO), tropical cyclone track dynamics (IBTrACS), and astronomical tidal harmonics.
- **Physics-Guided Loss Regularization**: Incorporates hydrostatic inverted barometer, spatial sea-surface gradients, and mass continuity directly into backpropagation in strict SI units.
- **Calibrated Non-Crossing Uncertainty**: Enforces strict monotonic quantiles ($q_{10} \le q_{50} \le q_{90}$) and applies Conformalized Quantile Regression (CQR) to guarantee empirical coverage on held-out test periods.

---

## 2. DATASET SPECIFICATIONS & SELECTION STRATEGY (`combined34/`)

The directory `combined34/` contains the expanded multi-decadal 34-station dataset spanning **2011-10-01 01:00:00 to 2022-12-31 23:00:00** (~98,640 consecutive hourly timesteps). The 34 stations cover the Bay of Bengal, Arabian Sea, Andaman Sea, Strait of Malacca, South China Sea, and Indonesian archipelago:
- *Stations*: Akyab_Sittwe, Ambon, Benoa, Bitung, Cendering, Cochin, Currimao, Davao, Hong_Kong, Kelang, Keling, Ko_Lak, Ko_Taphao_Noi, Kota_Kinabalu, Kukup, Langkawi, Legaspi, Lembar, Lubang, Lumut, Moulmein, Padang, Penang, Prigi, Qui_Nhon, Sabang, Sandakan, Sedili, Sibolga, Subic_Bay, Tanjong_Pagar, Tawau, Vishakhapatnam, Vung_Tau.

There are **two distinct datasets** inside `combined34/`:

### Dataset 1: `combined34/34_stations_combined_with_IBTrACs.csv` (24 Columns)
- **Base Columns (17)**:
  - `time`, `station_name`, `station_code`, `station_latitude`, `station_longitude`
  - `sea_level` (Target water level in meters), `target_valid` (Binary quality flag)
  - `u10`, `v10` (ERA5 10m eastward/northward wind components in m/s)
  - `msl` (ERA5 Mean sea level atmospheric pressure in Pascals)
  - `tp` (ERA5 Total precipitation in meters)
  - `wind_speed` (Derived magnitude $\sqrt{u10^2 + v10^2}$ in m/s)
  - `era5_latitude`, `era5_longitude`
  - `gebco_nearest_m`, `gebco_min_5km_m`, `gebco_mean_5km_m` (GEBCO bathymetry / seabed depth in meters)
- **EXTRA FEATURES — Tropical Cyclone Forcing from IBTrACS (7 Columns)**:
  1. `cyclone_distance_km`: Haversine distance from the station to the active cyclone eye in kilometers.
  2. `cyclone_wind_kts`: Maximum sustained cyclone wind speed in knots.
  3. `cyclone_pressure_hpa`: Central cyclone atmospheric pressure in hectopascals.
  4. `cyclone_active`: Binary indicator (1 if a cyclone is active within regional basin, 0 otherwise).
  5. `cyclone_latitude`: Instantaneous latitude of the cyclone center.
  6. `cyclone_longitude`: Instantaneous longitude of the cyclone center.
  7. `cyclone_sid`: Storm identification code from IBTrACS.

### Dataset 2: `combined34/34_stations_GESLA_ERA__GEBCO.csv` (17 Columns)
- Contains only the 17 base columns listed above, **omitting the 7 IBTrACS cyclone tracking features**.

### 🎯 Mandatory Dataset Selection for Current Task
**Select `combined34/34_stations_combined_with_IBTrACs.csv` (Dataset 1)**.
*Reasoning*: SAGAR-Net’s primary objective is compound coastal flood and storm-surge prediction under tropical cyclone forcing. Discarding cyclone distance and intensity reduces the model to a background atmospheric tide model, violating the research mandate.

### ⚠️ Critical Schema Incompatibilities in Existing Scripts (MUST BE FIXED)
If you attempt to run `construct_graph.py` or `sagar_dataset.py` with the new dataset without updating column references, the code will either crash with a `KeyError` or silently drop all cyclone features:
1. **Bathymetric Elevation Reference**:
   - `construct_graph.py` currently hardcodes: `usecols=['station_name', 'station_latitude', 'station_longitude', 'elevation_m']`.
   - The new CSV does not have `elevation_m`; it has `gebco_mean_5km_m`, `gebco_nearest_m`, and `gebco_min_5km_m`.
   - **Required Fix**: Update `construct_graph.py` to use `gebco_mean_5km_m` (or fallback to `gebco_nearest_m` or `elevation_m`).
2. **Cyclone Column Names**:
   - `sagar_dataset.py` currently looks for: `storm_wind`, `storm_pressure`, `STORM_SPEED`, `STORM_DIR`, `storm_distance_km`, `storm_present`.
   - The new CSV names them: `cyclone_wind_kts`, `cyclone_pressure_hpa`, `cyclone_distance_km`, `cyclone_active`, `cyclone_latitude`, `cyclone_longitude`.
   - **Required Fix**: Update `all_candidate_cols` and `physical_defaults` in `sagar_dataset.py` to map the `cyclone_*` column names and their physical default fallback values ($P_{\text{cyclone}} = 1013.25\text{ hPa}$, $d_{\text{cyclone}} = 10000.0\text{ km}$, $\text{active} = 0.0$).

---

## 3. SAGAR-NET RECENT SYSTEM ARCHITECTURE

```
Input Tensor: [Batch, Nodes=34, History=72h, Features]
  Features = [ERA5 (5) + IBTrACS (6) + Astronomical Tidal Harmonics (6)]
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 1. Multiscale Dilated Causal TCN Encoder                     │
  │    - 5 CausalConv1d Blocks (kernel=3; dilations d=1,2,4,8,16) │
  │    - 63-hour causal receptive field (no future leakage)       │
  │    - BatchNorm1d + ReLU + Dropout(0.1) + Residual Add         │
  └────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 2. Residual Recurrent Layer (GRU)                             │
  │    - nn.GRU(hidden_dim=64, num_layers=1, batch_first=True)   │
  │    - Recurrent temporal embedding combined via residual sum:  │
  │      h_temp = LayerNorm(h_gru[:, -1, :] + h_tcn[:, -1, :])   │
  └────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 3. Static Geographic Context Projection                       │
  │    - Inputs: [latitude, longitude, gebco_mean_elevation]      │
  │    - LayerNorm -> Linear -> ReLU -> Linear -> hidden_dim      │
  │    - Fused prior to graph propagation: h_temp = h_temp + h_st │
  └────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 4. Audited Sparse Coastal Graph Attention (GATv2Conv)         │
  │    - Sparse marine topology: k=4 marine neighbors, d <= 2200km│
  │    - Edge attributes: [Geographic Sim, Bathymetric Sim]       │
  │    - GATv2Conv(in=64, out=16, heads=4, edge_dim=2)            │
  │    - Residual Fusion: LayerNorm(Linear([h_temp, h_spat]) + h) │
  └────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ 5. Multi-Horizon Non-Crossing Quantile Forecast Heads         │
  │    - Horizons: +6h, +12h, +24h, +48h, +72h                    │
  │    - Concatenates: [h_fused, future_weather_at_horizon, tide] │
  │    - Non-crossing formulation via Softplus offsets:           │
  │        q50 = raw[0]                                           │
  │        q10 = q50 - Softplus(raw[1])                           │
  │        q90 = q50 + Softplus(raw[2])                           │
  │    - Mathematical Guarantee: q10 <= q50 <= q90 everywhere     │
  └───────────────────────────────────────────────────────────────┘
```

### 3.1 Cyclical Astronomical Tidal Harmonics
Total coastal water level is dominated by deterministic gravitational tides. SAGAR-Net explicitly computes station-specific local astronomical solar time:
$$\tau = t_{\text{hours}} + \frac{\text{longitude}}{15.0}$$
And embeds 6 continuous cyclical harmonic phase features:
1. Diurnal solar cycle ($S_1 / K_1$, period = $24.0\text{ h}$): $\sin(2\pi\tau / 24.0)$, $\cos(2\pi\tau / 24.0)$
2. Principal lunar semidiurnal cycle ($M_2$, period = $12.4206012\text{ h}$): $\sin(2\pi\tau / 12.4206)$, $\cos(2\pi\tau / 12.4206)$
3. Principal solar semidiurnal cycle ($S_2$, period = $12.0\text{ h}$): $\sin(2\pi\tau / 12.0)$, $\cos(2\pi\tau / 12.0)$

These features are injected both into the historical input sequence and directly into each future forecast horizon head.

### 3.2 Audited Coastal Topology (`construct_graph.py`)
- Dense, overland connectivity (which previously routed graph edges across continental India or the Malay Peninsula) is completely pruned.
- Nodes are strictly ordered alphabetically to preserve invariant 1:1 mapping with pivot tables.
- Each node connects only to its $k=4$ nearest marine neighbors within $d \le 2200.0\text{ km}$.
- Reciprocal directed edge weights:
  $$\text{sim\_geo} = \exp\left(-\frac{d^2}{2 \sigma_{\text{dist}}^2}\right) \; (\sigma = 750\text{ km}), \qquad \text{sim\_bath} = \exp\left(-\frac{|\Delta z|}{\gamma_{\text{bath}}}\right) \; (\gamma = 25\text{ m})$$

### 3.3 Physics-Guided Loss Formulations (`loss.py`)
All physical quantities are computed in strict **SI units**:
- Seawater density $\rho_w = 1025.0\text{ kg/m}^3$, gravitational acceleration $g = 9.80665\text{ m/s}^2 \implies \rho_w g \approx 10,051.8\text{ Pa/m}$.
- Atmospheric pressure is processed in Pascals ($\text{msl} \approx 101,325\text{ Pa}$).
- **Total Loss Objective**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pinball}} + \lambda_{\text{phys}}\mathcal{L}_{\text{IB}} + \lambda_{\text{grad}}\mathcal{L}_{\text{grad}} + \lambda_{\text{mass}}\mathcal{L}_{\text{mass}}$$
  - **Pinball Loss**: Evaluated at quantiles $\tau \in \{0.1, 0.5, 0.9\}$ masked strictly to training stations.
  - **Hydrostatic Inverted Barometer**: $\Delta \eta_{\text{IB}} = \frac{msl_t - msl_h}{\rho_w g}$.
  - **Spatial Gradient Loss**: Penalizes discrepancies between spatial water level differences and pressure/wind stress gradients along marine edges.
  - **Mass Continuity Loss**: Penalizes divergence of flux across neighboring coastal nodes.
- *Empirical Note from Stage 4 Ablation*: Inverted Barometer loss alone with fixed $\lambda_{\text{phys}} = 0.05$ improved $+6\text{h}$ lead time but degraded long horizons (+12h to +72h) because instantaneous hydrostatic equilibrium ignores progressive wave travel times.

### 3.4 Conformalized Quantile Regression (CQR) Calibration
Raw neural quantiles underestimate real-world epistemic uncertainty under spatial distribution shift. SAGAR-Net applies CQR (Romano et al., 2019):
1. On the validation partition, compute nonconformity scores for each horizon $h$:
   $$s_i = \max\left(q_{10}(x_i) - y_i, \; y_i - q_{90}(x_i)\right)$$
2. Compute the $(1 - \alpha)$ empirical conformal quantile:
   $$\hat{q} = \text{Quantile}\left(\{s_i\}, \; \min\left(1.0, \; (1 - \alpha)\frac{n + 1}{n}\right)\right) \quad (\alpha = 0.20 \text{ for } 80\% \text{ nominal})$$
3. Calibrate the held-out test intervals:
   $$I_{\text{cal}}(x) = [q_{10}(x) - \hat{q}, \;\; q_{90}(x) + \hat{q}]$$
   Mathematically guaranteeing $\ge 80\%$ marginal test coverage.

---

## 4. CURRENT BENCHMARK RESULTS & CRITICAL ROOT-CAUSE ANALYSIS

### 4.1 Latest Held-Out Evaluation (Test Split 2022–2023, Station: Cochin)
```
Best Validation MAE achieved: 2.1064m
Final Test Average MAE:       2.2019m (vs Persistence: 0.2115m)
Final Test Average RMSE:      2.3073m (vs Persistence: 0.2570m)
Raw Test Interval Coverage:   0.2%   (Raw Interval Width: 0.8231m)
Calibrated (CQR) Coverage:    74.2%  (Nominal: 80.0%, Calibrated Width: 5.3227m)

Horizon | MAE (m) | RMSE (m) | Raw Cov | Cal Cov | Raw Width | Cal Width | q_hat
  +6 h  | 2.2004  |  2.3005  |  0.0 %  | 74.0 %  |  0.8581m  |  5.2303m  | +2.1861m
 +12 h  | 2.0770  |  2.2034  |  0.1 %  | 73.6 %  |  0.8162m  |  5.1104m  | +2.1471m
 +24 h  | 2.2240  |  2.3368  |  0.2 %  | 74.6 %  |  0.7885m  |  5.4580m  | +2.3348m
 +48 h  | 2.2477  |  2.3494  |  0.2 %  | 74.6 %  |  0.7944m  |  5.4332m  | +2.3194m
 +72 h  | 2.2602  |  2.3465  |  0.2 %  | 74.0 %  |  0.8581m  |  5.3815m  | +2.2617m
```

### 4.2 The Critical Bottleneck: Local Chart Datum Offset in LOSO
- **The Symptom**: The model achieves MAE $\approx 2.20\text{ m}$, whereas persistence achieves $0.21\text{ m}$. The conformal offset $\hat{q} \approx +2.2\text{ m}$ indicates an almost constant systematic bias.
- **The Physical Cause**: Every tide gauge measures water level relative to its local **Chart Datum** (e.g., Lowest Astronomical Tide or mean sea level of that port).
  - Cochin’s historical mean sea level is **$0.751\text{ m}$**.
  - Neighboring stations in the graph (Chittagong, Kukup, Lumut) have historical mean sea levels of **$3.4\text{ m}$ to $4.2\text{ m}$**.
  - During Leave-One-Station-Out validation on Cochin, the model has never seen Cochin's sea level target. When GATv2 aggregates spatial embeddings from neighbors, it pulls Cochin's prediction up toward the regional network mean ($\approx 3.0\text{ m}$).
  - Prediction error = $3.0\text{ m} - 0.75\text{ m} \approx 2.25\text{ m}$.
- **The Solution Path**:
  1. **Surge Anomaly Decomposition**: Instead of predicting raw total water level, predict the meteorological surge anomaly:
     $$\eta_{\text{surge}}(t) = \eta_{\text{total}}(t) - \eta_{\text{astronomical\_tide}}(t)$$
     Surge anomaly is zero-centered and dynamically transferable across ungauged coasts without datum bias.
  2. **Datum Grounding via Static Context**: If predicting total sea level, provide explicit station datum/mean water level offsets in static node embeddings so message passing does not confound dynamic surge with local vertical datums.

---

## 5. ACTIONABLE DIRECTIVES FOR CLAUDE CODE

When working in this codebase:

1. **Environment**:
   Always execute Python commands using the project's virtual environment:
   ```powershell
   .\.venv\Scripts\python.exe <script.py>
   ```

2. **Step 1: Fix Schema Ingestion for 34 Stations**:
   - Update `construct_graph.py` to ingest `combined34/34_stations_combined_with_IBTrACs.csv` using `gebco_mean_5km_m` for elevation.
   - Update `sagar_dataset.py` to recognize `cyclone_wind_kts`, `cyclone_pressure_hpa`, `cyclone_distance_km`, `cyclone_active`, `cyclone_latitude`, `cyclone_longitude` and set clean physical defaults.

3. **Step 2: Reconstruct the Coastal Graph**:
   - Run `construct_graph.py --dataset combined34/34_stations_combined_with_IBTrACs.csv --output static_graph.pt --k-neighbors 4`.
   - Verify that all 34 stations are connected into sparse marine corridors without cross-land links.

4. **Step 3: Resolve the Chart Datum Offset**:
   - Refactor the target or spatial conditioning in `sagar_dataset.py` and `model.py` to eliminate the constant ~2.2m bias.

5. **Step 4: Train, Validate, and Calibrate**:
   - Train with `train.py` using early stopping (`patience=3`), gradient clipping (`max_norm=1.0`), and AdamW.
   - Run CQR uncertainty calibration on the validation split and evaluate on the test split.
   - Log and verify all metrics against the persistence baseline.

