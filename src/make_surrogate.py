#!/usr/bin/env python3
"""
make_surrogate.py  --  Observing System Simulation Experiment (OSSE) data.

!!! SYNTHETIC DATA -- NOT NASA OBSERVATIONS !!!

This builds a *known-truth* tropical-Pacific dataset by superposing the
VERIFIED analytic equatorial waves (eastward Kelvin + westward Matsuno Rossby
n=1) from physics.py onto the real target grid and region. It exists solely to
exercise and verify the downstream software (Hovmoller analysis, conventional
propagation-speed estimation, the NetCDF-driven PINN, and velocity validation)
in this sandbox, where the live NASA Earthdata endpoints are unreachable.

Because the truth (u, v, c, wave identities) is known exactly, it is the
correct scientific way to validate the *method* before applying it to real
data. Every output file and variable is tagged synthetic=1 so it can never be
mistaken for a NASA product.

Outputs (data_processed/):
    surrogate_ssh.nc     ssha(time, latitude, longitude)   [m]   "observations"
    surrogate_truth.nc   u,v,eta true fields + metadata     [m/s, m]
    surrogate_oscar.nc   u,v on an OSCAR-like grid/time (held-out)  [m/s]

When real NASA data are available, src/build_dataset.py produces files with the
identical schema, so the entire downstream pipeline runs unchanged.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C                                        # noqa: E402
from physics import kelvin_solution, matsuno_mode, G, BETA  # noqa: E402

# ----- known truth -----
C_TRUE = 2.8                      # m/s  (equatorial Pacific first baroclinic mode)
# Wavelengths chosen so ~1+ full wave fits the 70-deg (~7,780 km) sector, giving
# a clear phase tilt in the Hovmoller (basin-crossing in ~1 month => fast
# intraseasonal Kelvin, period ~25 d). This keeps the wave well sampled in
# space for both conventional estimation and the PINN.
KELVIN_LAMBDA = 6.0e6             # m   (~6,000 km; ~25-day period at 2.8 m/s)
KELVIN_ETA_AMP = 0.12             # m   SSH amplitude of Kelvin pulse
ROSSBY_LAMBDA = 4.0e6             # m
ROSSBY_ETA_AMP = 0.06             # m
OBS_NOISE_SSH = 0.01              # m   (1 cm, ~ gridded altimetry noise)
OBS_NOISE_VEL = 0.03              # m/s (OSCAR-like)

SEED = 20260101


def latlon_to_xy(lon, lat):
    x = (lon - C.LON_REF) * C.deg2m_lon(0.0)
    y = (lat - C.LAT_REF) * C.DEG2M_LAT
    return x, y


def fields_on_grid(lon, lat, t_sec, c=C_TRUE):
    """Evaluate true (u,v,eta) analytic fields on a (lon,lat,time) grid."""
    LON, LAT = np.meshgrid(lon, lat)          # (nlat, nlon)
    x, y = latlon_to_xy(LON.ravel(), LAT.ravel())
    nt = len(t_sec)
    U = np.empty((nt, LAT.shape[0], LAT.shape[1]), dtype=np.float64)
    V = np.empty_like(U)
    ETA = np.empty_like(U)

    kK = 2 * math.pi / KELVIN_LAMBDA
    kR = 2 * math.pi / ROSSBY_LAMBDA
    AK = G * KELVIN_ETA_AMP          # Phi amplitude
    AR = G * ROSSBY_ETA_AMP

    xt = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)

    for i, ts in enumerate(t_sec):
        tt = torch.full_like(xt, float(ts))
        X = torch.cat([xt, yt, tt], dim=1)
        uK, vK, pK = kelvin_solution(X, c, AK, kK)
        uR, vR, pR, _ = matsuno_mode(X, c, AR, kR, n=1, branch="rossby")
        u = (uK + uR).detach().numpy().ravel()
        v = (vK + vR).detach().numpy().ravel()
        eta = ((pK + pR) / G).detach().numpy().ravel()
        U[i] = u.reshape(LAT.shape)
        V[i] = v.reshape(LAT.shape)
        ETA[i] = eta.reshape(LAT.shape)
    return U, V, ETA


def _common_attrs(extra):
    a = {
        "title": "SYNTHETIC OSSE surrogate -- NOT NASA DATA",
        "synthetic": 1,
        "source": "src/make_surrogate.py analytic Kelvin+Rossby superposition",
        "warning": "Known-truth test data for software verification only.",
        "c_true_m_s": C_TRUE,
        "equivalent_depth_true_m": C_TRUE**2 / G,
        "beta": BETA,
        "region_WSEN": list(C.BBOX),
    }
    a.update(extra)
    return a


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ----- "NASA-SSH-like" grid: 0.5 deg, weekly -----
    lon = np.arange(C.LON_MIN, C.LON_MAX + 1e-6, 0.5)
    lat = np.arange(C.LAT_MIN, C.LAT_MAX + 1e-6, 0.5)
    nweeks = 78                                   # ~1.5 yr
    days = np.arange(nweeks) * 7.0
    t0 = np.datetime64(C.EXP_START)
    time = t0 + (days * np.timedelta64(1, "D")).astype("timedelta64[ns]")
    t_sec = days * 86400.0

    U, V, ETA = fields_on_grid(lon, lat, t_sec)

    # observation = truth + noise (SSH only is "observed")
    ssha = ETA + OBS_NOISE_SSH * np.random.randn(*ETA.shape)

    ds_obs = xr.Dataset(
        {"ssha": (("time", "latitude", "longitude"), ssha.astype("float32"),
                  {"units": "m", "long_name": "sea surface height anomaly (synthetic)"})},
        coords={"time": time,
                "latitude": ("latitude", lat.astype("float32"), {"units": "degrees_north"}),
                "longitude": ("longitude", lon.astype("float32"), {"units": "degrees_east"})},
        attrs=_common_attrs({"mimics_product": C.PRODUCTS["ssh"],
                             "spatial_resolution_deg": 0.5,
                             "temporal_resolution": "7D"}),
    )
    ds_obs.to_netcdf(C.DATA_PROC / "surrogate_ssh.nc")

    ds_truth = xr.Dataset(
        {"u": (("time", "latitude", "longitude"), U.astype("float32"), {"units": "m/s"}),
         "v": (("time", "latitude", "longitude"), V.astype("float32"), {"units": "m/s"}),
         "eta": (("time", "latitude", "longitude"), ETA.astype("float32"), {"units": "m"})},
        coords={"time": time, "latitude": lat.astype("float32"),
                "longitude": lon.astype("float32")},
        attrs=_common_attrs({"content": "noise-free true fields"}),
    )
    ds_truth.to_netcdf(C.DATA_PROC / "surrogate_truth.nc")

    # ----- "OSCAR-like" independent grid: 0.25 deg, 5-day, offset phase -----
    lon_o = np.arange(C.LON_MIN, C.LON_MAX + 1e-6, 0.25)
    lat_o = np.arange(C.LAT_MIN, C.LAT_MAX + 1e-6, 0.25)
    days_o = np.arange(0, nweeks * 7, 5.0)
    time_o = t0 + (days_o * np.timedelta64(1, "D")).astype("timedelta64[ns]")
    Uo, Vo, _ = fields_on_grid(lon_o, lat_o, days_o * 86400.0)
    Uo = Uo + OBS_NOISE_VEL * np.random.randn(*Uo.shape)
    Vo = Vo + OBS_NOISE_VEL * np.random.randn(*Vo.shape)
    ds_osc = xr.Dataset(
        {"u": (("time", "latitude", "longitude"), Uo.astype("float32"), {"units": "m/s"}),
         "v": (("time", "latitude", "longitude"), Vo.astype("float32"), {"units": "m/s"})},
        coords={"time": time_o, "latitude": lat_o.astype("float32"),
                "longitude": lon_o.astype("float32")},
        attrs=_common_attrs({"mimics_product": C.PRODUCTS["oscar"],
                             "note": "held-out velocity for PINN validation"}),
    )
    ds_osc.to_netcdf(C.DATA_PROC / "surrogate_oscar.nc")

    print("Wrote synthetic OSSE surrogate (NOT NASA data):")
    for f in ["surrogate_ssh.nc", "surrogate_truth.nc", "surrogate_oscar.nc"]:
        p = C.DATA_PROC / f
        print(f"  {p}  ({p.stat().st_size/1e3:.0f} kB)")
    print(f"\nTrue c = {C_TRUE} m/s, equivalent depth = {C_TRUE**2/G:.4f} m")
    print(f"SSH grid: {ssha.shape} (time,lat,lon); "
          f"lon {lon.min()}..{lon.max()}, lat {lat.min()}..{lat.max()}")
    print(f"SSHA range: [{np.nanmin(ssha):.3f}, {np.nanmax(ssha):.3f}] m")


if __name__ == "__main__":
    main()
