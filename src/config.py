#!/usr/bin/env python3
"""
config.py

Central configuration for the tropical-Pacific equatorial shallow-water PINN
study using NASA satellite products.

All physical constants are SI. All geographic constants are degrees.
Nothing here is secret; NASA Earthdata credentials are NEVER stored in this
repository. They must come from ~/.netrc or the environment variables
EARTHDATA_USERNAME / EARTHDATA_PASSWORD at run time.
"""

import math
from pathlib import Path

# -------------------------------------------------------------------------
# Repository layout
# -------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data_raw"
DATA_PROC = ROOT / "data_processed"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"

for _d in (DATA_RAW, DATA_PROC, FIGURES, MODELS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Physical constants (SI)
# -------------------------------------------------------------------------
OMEGA = 7.292115e-5          # Earth rotation rate, rad/s
R_E = 6.371e6                # Earth radius, m
G = 9.81                     # gravity, m/s^2
BETA = 2.0 * OMEGA / R_E     # equatorial beta, 1/(m s)

# metres per degree (equatorial-plane approximation used for the beta-plane)
DEG2M_LAT = math.pi / 180.0 * R_E              # ~111 km per deg latitude
def deg2m_lon(lat_deg=0.0):
    """metres per degree longitude at a given latitude."""
    return math.pi / 180.0 * R_E * math.cos(math.radians(lat_deg))

# -------------------------------------------------------------------------
# NASA / PO.DAAC product short names (CMR collection short_name)
# -------------------------------------------------------------------------
PRODUCTS = {
    "ssh":   "NASA_SSH_REF_SIMPLE_GRID_V1",           # gridded SSHA, ~0.5deg weekly
    "oscar": "OSCAR_L4_OC_FINAL_V2.0",                # L4 surface currents (u,v,ug,vg)
    "sst":   "MUR-JPL-L4-GLOB-v4.1",                  # analysed_sst
    "sss":   "SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5",  # smap_sss
    "swot":  "SWOT_L2_LR_SSH_BASIC_D",               # high-res / coastal (later)
}

PRODUCT_VARIABLES = {
    "ssh":   ["ssha"],
    "oscar": ["u", "v", "ug", "vg"],
    "sst":   ["analysed_sst"],
    "sss":   ["smap_sss"],
    "swot":  ["ssh_karin", "ssha_karin", "ssh_karin_qual", "distance_to_coast"],
}

# -------------------------------------------------------------------------
# Tropical-Pacific target region (Stage 2)
# CMR bounding box order: (west_lon, south_lat, east_lon, north_lat)
# -------------------------------------------------------------------------
# Longitudes in -180..180 convention (west negative).
LON_MIN = -170.0
LON_MAX = -100.0
LAT_MIN = -8.0
LAT_MAX = 8.0
BBOX = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)

# Equatorial band used for Hovmoller / equatorial averages.
EQ_BAND = (-2.0, 2.0)

# Reference longitude used to centre the beta-plane x coordinate (deg East).
LON_REF = 0.5 * (LON_MIN + LON_MAX)     # -135 deg
LAT_REF = 0.0

# -------------------------------------------------------------------------
# Time windows
# -------------------------------------------------------------------------
# Short smoke-test window (Stage 1).
SMOKE_START = "2023-11-01"
SMOKE_STOP = "2023-12-31"

# Recommended first real experiment window (Stage 2).
# 2015-2016 covers the very strong 2015/16 El Nino: strong downwelling Kelvin
# pulses in boreal spring 2015 and reflected Rossby waves -- ideal for showing
# propagating equatorial anomalies. Adjust after inspecting real data quality.
EXP_START = "2015-01-01"
EXP_STOP = "2016-06-30"

# -------------------------------------------------------------------------
# PINN output / residual conditioning scales (see equatorial_kelvin_pinn_test)
# These are recomputed from the actual data in the real-data driver, but these
# defaults are reasonable for tropical-Pacific baroclinic-mode scales.
# -------------------------------------------------------------------------
# Typical first-baroclinic-mode gravity-wave speed in the equatorial Pacific is
# ~2.5-3.0 m/s (equivalent depth ~0.6-0.9 m). Initialise away from truth.
C_INIT = 1.0                 # m/s
U_SCALE = 0.5                # m/s, typical equatorial current amplitude
PHI_SCALE = G * 0.1          # m^2/s^2 for ~0.1 m SSHA amplitude

# Residual balance scales (order-of-magnitude of each PDE term).
RU_SCALE = 1.0e-6            # m/s^2
RV_SCALE = 1.0e-6            # m/s^2
RPHI_SCALE = 1.0e-6          # m^2/s^3


def summary():
    import json
    d = {
        "region_bbox_WSEN": BBOX,
        "lon_ref_deg": LON_REF,
        "equatorial_band_deg": EQ_BAND,
        "smoke_window": [SMOKE_START, SMOKE_STOP],
        "experiment_window": [EXP_START, EXP_STOP],
        "products": PRODUCTS,
        "beta": BETA,
    }
    return json.dumps(d, indent=2)


if __name__ == "__main__":
    print(summary())
