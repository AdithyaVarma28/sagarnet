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
# All are resolvable (Rayleigh criterion) with >= 1 year of hourly data.
HARMONIC_PERIODS_H = {
    "M2": 12.4206012, "S2": 12.0, "N2": 12.65834751, "K2": 11.96723606,
    "K1": 23.93447213, "O1": 25.81933871, "P1": 24.06589016, "Q1": 26.86835667,
    "S1": 24.0,
    "M4": 6.210300601, "MS4": 6.103339275, "MN4": 6.269173724, "M6": 4.140200401,
    "Mf": 327.8599387, "Mm": 661.3111655, "Ssa": 4382.905209, "Sa": 8766.15265,
}

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
