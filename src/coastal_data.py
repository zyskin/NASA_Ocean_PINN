#!/usr/bin/env python3
"""
coastal_data.py  --  NOAA/IOOS coastal data ingestion for the New York Bight.

REAL public data sources (NO NASA Earthdata auth needed). These run only where
the hosts are reachable; in a sandbox that blocks *.noaa.gov they raise a clear
error and the synthetic coastal PINN path is used instead.

Sources:
  * HF-radar total surface currents (THREDDS/OPeNDAP):
        https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_usegc_1km   (1 km hourly)
    vars: u, v [m/s], hdop, dopx, dopy, number_of_sites, number_of_radials.
    HF radar = near-surface (~upper 0.5 m) velocity, NOT depth-averaged.
  * NOAA CO-OPS water level (REST API): api.tidesandcurrents.noaa.gov
        stations: Battery NY 8518750, Sandy Hook NJ 8531680
  * NOAA NCEI ETOPO bathymetry (depth H>0 + land mask).

Do NOT download the whole East Coast array: always subset space+time first.
Do NOT mix vertical datums silently (CO-OPS datum is recorded in the output).
"""

import io
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HFRADAR_1KM = "https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_usegc_1km"
HFRADAR_6KM = "https://dods.ndbc.noaa.gov/thredds/dodsC/hfradar_usegc_6km"
COOPS_API = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# New York Bight / harbour-entrance search box (tighten after inspection)
NYB_BBOX = dict(lon_min=-74.6, lon_max=-72.5, lat_min=39.0, lat_max=41.2)

# decision subregions for coverage (lon_min,lon_max,lat_min,lat_max)
SUBREGIONS = {
    "new_york_bight":   (-74.3, -73.2, 39.6, 40.5),
    "harbour_entrance": (-74.10, -73.85, 40.40, 40.55),   # Sandy Hook / Ambrose
    "lower_ny_bay":     (-74.10, -73.95, 40.50, 40.62),
    "inner_harbour":    (-74.05, -73.95, 40.60, 40.72),
}

COOPS_STATIONS = {"battery_ny": "8518750", "sandy_hook_nj": "8531680"}


def host_reachable(host, port=443, timeout=8):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except Exception:
        return False


def coastal_access_ok():
    return host_reachable("dods.ndbc.noaa.gov")


def open_hfradar(url=HFRADAR_1KM, bbox=NYB_BBOX, time_slice=None):
    """Open HF-radar OPeNDAP and subset to the NY box (server-side where possible)."""
    import xarray as xr
    ds = xr.open_dataset(url)
    # coordinate names in the national product are lon/lat/time
    lon = "lon" if "lon" in ds.coords else "longitude"
    lat = "lat" if "lat" in ds.coords else "latitude"
    if float(ds[lon].max()) > 180:
        ds = ds.assign_coords({lon: (((ds[lon] + 180) % 360) - 180)}).sortby(lon)
    sel = {lon: slice(bbox["lon_min"], bbox["lon_max"]),
           lat: slice(bbox["lat_min"], bbox["lat_max"])}
    ds = ds.sel(**sel)
    if time_slice is not None:
        ds = ds.sel(time=slice(*time_slice))
    return ds.rename({lon: "lon", lat: "lat"})


def coverage_fraction(ds, subregions=SUBREGIONS):
    """Fraction of finite (u,v) through time within each subregion."""
    import numpy as np
    out = {}
    for name, (lo0, lo1, la0, la1) in subregions.items():
        sub = ds.sel(lon=slice(lo0, lo1), lat=slice(la0, la1))
        if sub["u"].size == 0:
            out[name] = 0.0
            continue
        valid = np.isfinite(sub["u"].values) & np.isfinite(sub["v"].values)
        out[name] = float(valid.mean())
    return out


def save_subset(ds, path, varlist=("u", "v", "hdop", "number_of_sites",
                                   "number_of_radials")):
    keep = [v for v in varlist if v in ds]
    ds[keep].to_netcdf(path)
    return path


def coops_water_level(station, begin_date, end_date, datum="MSL",
                      product="water_level", interval=None):
    """
    Fetch NOAA CO-OPS water level (or 'currents'/'predictions'). Returns a
    pandas DataFrame with the datum recorded. begin/end as 'YYYYMMDD'.
    """
    import pandas as pd
    import requests
    params = {
        "station": station, "begin_date": begin_date, "end_date": end_date,
        "product": product, "datum": datum, "units": "metric",
        "time_zone": "gmt", "format": "csv", "application": "nasa_ocean_pinn",
    }
    if interval:
        params["interval"] = interval
    r = requests.get(COOPS_API, params=params, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.attrs["datum"] = datum
    df.attrs["station"] = station
    return df


def etopo_bathymetry(bbox=NYB_BBOX,
                     url=("https://www.ngdc.noaa.gov/thredds/dodsC/global/"
                          "ETOPO2022/60s/60s_surface_elev_netcdf/"
                          "ETOPO_2022_v1_60s_N90W180_surface.nc")):
    """
    Open ETOPO elevation (m, positive up). Returns (H, mask) where H = -elev
    is still-water depth in metres (>0 ocean) and mask marks wet cells.
    Depth is taken from the authoritative product, never inferred from the
    coastline.
    """
    import numpy as np
    import xarray as xr
    ds = xr.open_dataset(url)
    latn = "lat" if "lat" in ds.coords else "latitude"
    lonn = "lon" if "lon" in ds.coords else "longitude"
    elevname = [v for v in ds.data_vars if "elev" in v.lower() or "z" == v.lower()][0]
    sub = ds.sel({lonn: slice(bbox["lon_min"], bbox["lon_max"]),
                  latn: slice(bbox["lat_min"], bbox["lat_max"])})
    elev = sub[elevname]
    H = (-elev).where(elev < 0)        # depth>0 only in ocean
    mask = (elev < 0)
    return H.rename({lonn: "lon", latn: "lat"}), mask.rename({lonn: "lon", latn: "lat"})


if __name__ == "__main__":
    print("Coastal data access preflight")
    ok = coastal_access_ok()
    print(f"  dods.ndbc.noaa.gov reachable: {ok}")
    if not ok:
        print("  -> NOAA hosts blocked by egress policy in this environment.")
        print("     Run where *.noaa.gov is allowed; see README. The synthetic")
        print("     coastal PINN (coastal_pinn.py) runs fully offline.")
        sys.exit(2)
    ds = open_hfradar(time_slice=None)
    print("  HF-radar dataset opened:", dict(ds.sizes))
    print("  coverage:", coverage_fraction(ds))
