"""Shared constants for SAGAR-Net (v2).

Everything physical is in SI units unless the name says otherwise.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
DEFAULT_DATASET = "combined34/34_stations_combined_with_IBTrACs.csv"
DEFAULT_GRAPH = "static_graph.pt"
DEFAULT_CACHE_DIR = "cache"

# --------------------------------------------------------------------------- #
# Physics
# --------------------------------------------------------------------------- #
RHO_W = 1025.0            # seawater density [kg m^-3]
G = 9.80665               # gravity [m s^-2]
RHO_G = RHO_W * G         # ~10051.8 Pa per metre of water
P_REF = 101325.0          # reference atmospheric pressure [Pa]
RHO_AIR = 1.225           # air density [kg m^-3]
CD_WIND = 1.3e-3          # bulk drag coefficient for wind stress (dimensionless)

# --------------------------------------------------------------------------- #
# Forecast setup
# --------------------------------------------------------------------------- #
HISTORY_HOURS = 72
HORIZONS = (6, 12, 24, 48, 72)
QUANTILES = (0.1, 0.5, 0.9)
# Hours *before* each target time at which future forcing is sampled and
# stacked for the horizon head (0 = forcing at the target hour itself).
FUTURE_LAGS = (0, 3, 6, 12)

# Window for the slow mean-sea-level anomaly removed from the surge target (30 days).
LF_WINDOW_HOURS = 720

# Temporal split (inclusive end timestamps). Data ends 2022-12-31 23:00.
TRAIN_END = "2019-12-31 23:00:00"
VAL_END = "2021-12-31 23:00:00"
TEST_END = "2022-12-31 23:00:00"

# --------------------------------------------------------------------------- #
# Tides
# --------------------------------------------------------------------------- #
# Cyclical phase features fed to the network (spec 3.1 uses S1, M2, S2; K1 and
# O1 are added because diurnal tides are large in the SE-Asian seas).
TIDE_FEATURE_PERIODS_H = {
    "S1": 24.0,
    "M2": 12.4206012,
    "S2": 12.0,
    "K1": 23.93447213,
    "O1": 25.81933871,
}

# Constituents for the per-station least-squares harmonic analysis used to
# split total water level into astronomical tide + non-tidal residual (surge).
# name -> (period [h], nodal recipe). The nodal recipe says how the 18.61-year
# lunar-node modulation (amplitude factor f, phase shift u) is built from the
# base constituents in tides.py: {base: k} gives f = prod f_base^|k| and
# u = sum k*u_base; {} = no nodal modulation (solar constituents).
# Order = priority: when a station's record is too short to separate two
# constituents (Rayleigh criterion), the later one is dropped.
# v2.1 adds nodal corrections and the shallow-water / minor constituents that
# matter at large-tide shallow stations (e.g. Kelang, residual std 0.19 m with
# the 17-constituent fit: mostly tide leaking into the "surge").
HARMONIC_CONSTITUENTS = {
    # major
    "M2": (12.4206012, {"M2": 1}), "S2": (12.0, {}), "K1": (23.93447213, {"K1": 1}),
    "O1": (25.81933871, {"O1": 1}), "N2": (12.65834751, {"M2": 1}), "K2": (11.96723606, {"K2": 1}),
    "P1": (24.06589016, {}), "Q1": (26.86835667, {"O1": 1}),
    # long period
    "Sa": (8766.15265, {}), "Ssa": (4382.905209, {}), "Mm": (661.3111655, {"Mm": 1}),
    "Mf": (327.8599387, {"Mf": 1}), "MSf": (354.3670666, {"M2": -1}),
    # minor diurnal / semidiurnal
    "S1": (24.0, {}), "J1": (23.09848146, {"J1": 1}), "OO1": (22.30608083, {"OO1": 1}),
    "2Q1": (28.00621204, {"O1": 1}), "2N2": (12.90537297, {"M2": 1}), "MU2": (12.87175763, {"M2": 1}),
    "NU2": (12.62600205, {"M2": 1}), "L2": (12.19162085, {"M2": 1}), "T2": (12.01644934, {}),
    "2SM2": (11.60695156, {"M2": -1}),
    # shallow water / overtides
    "M3": (8.280400802, {"M2": 1.5}), "MO3": (8.38630265, {"M2": 1, "O1": 1}),
    "MK3": (8.177140247, {"M2": 1, "K1": 1}), "SK3": (7.992705055, {"K1": 1}),
    "M4": (6.210300601, {"M2": 2}), "MS4": (6.103339275, {"M2": 1}), "MN4": (6.269173724, {"M2": 2}),
    "MK4": (6.094852323, {"M2": 1, "K2": 1}), "S4": (6.0, {}),
    "M6": (4.140200401, {"M2": 3}), "2MS6": (4.092383475, {"M2": 2}), "2MN6": (4.166283604, {"M2": 3}),
    "M8": (3.105150301, {"M2": 4}),
}
TIDE_NODAL = True
# Kept for reference / old code paths: the v2.0 17-constituent set.
HARMONIC_PERIODS_H = {k: HARMONIC_CONSTITUENTS[k][0] for k in (
    "M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "S1", "M4", "MS4", "MN4", "M6", "Mf", "Mm", "Ssa", "Sa")}

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
ERA5_COLS = ["u10", "v10", "msl", "tp", "wind_speed"]

# canonical name -> accepted aliases (first match wins). Legacy names from the
# older 9-station files are kept so old CSVs still load.
CYCLONE_ALIASES = {
    "cyclone_wind_kts": ["cyclone_wind_kts", "storm_wind", "STORM_WIND"],
    "cyclone_pressure_hpa": ["cyclone_pressure_hpa", "storm_pressure", "STORM_PRES"],
    "cyclone_distance_km": ["cyclone_distance_km", "storm_distance_km"],
    "cyclone_active": ["cyclone_active", "storm_present"],
    "cyclone_latitude": ["cyclone_latitude", "storm_lat", "LAT"],
    "cyclone_longitude": ["cyclone_longitude", "storm_lon", "LON"],
}

# Physical "no cyclone" values. Lat/lon default to the station position at
# load time so the relative offset is zero.
CYCLONE_DEFAULTS = {
    "cyclone_wind_kts": 0.0,
    "cyclone_pressure_hpa": 1013.25,
    "cyclone_distance_km": 10000.0,
    "cyclone_active": 0.0,
}

ELEVATION_ALIASES = ["gebco_mean_5km_m", "gebco_nearest_m", "gebco_min_5km_m", "elevation_m"]
GEBCO_COLS = ["gebco_mean_5km_m", "gebco_nearest_m", "gebco_min_5km_m"]

# GESLA uses -99.9999 as a missing value sentinel.
SEA_LEVEL_MISSING_BELOW = -50.0
