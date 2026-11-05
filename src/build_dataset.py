#!/usr/bin/env python3
"""
build_dataset.py  --  Stage 2: assemble the real tropical-Pacific dataset.

Runs ONLY where NASA Earthdata is reachable and credentials are configured.
It searches, downloads, subsets, and writes processed NetCDF files whose schema
is IDENTICAL to the OSSE surrogate, so analysis.py / pinn_ssh.py /
validate_velocity.py run unchanged on real data. It also writes
data_manifest.csv recording exactly what was fetched.

Usage:
    python src/build_dataset.py --products ssh oscar --start 2015-01-01 \
        --stop 2016-06-30 [--smoke]

Outputs:
    data_processed/nasa_ssh.nc     ssha(time,latitude,longitude)  [m]
    data_processed/nasa_oscar.nc   u,v,ug,vg(time,latitude,longitude) [m/s]
    data_manifest.csv
"""

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                      # noqa: E402
import data_access as da                # noqa: E402

MANIFEST = C.ROOT / "data_manifest.csv"


def _norm_coords(ds):
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"),
                 ("Time", "time"), ("lon", "longitude"), ("lat", "latitude")]:
        if a in ds.coords or a in ds.dims:
            ren[a] = b
    ds = ds.rename(ren) if ren else ds
    if "longitude" in ds.coords and float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    if "latitude" in ds.coords and ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")
    return ds


def _subset(ds):
    return ds.sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                  latitude=slice(C.LAT_MIN, C.LAT_MAX))


def _res(ds):
    def dd(name):
        v = ds[name].values
        return float(np.median(np.abs(np.diff(v)))) if len(v) > 1 else None
    dx = dd("longitude") if "longitude" in ds.coords else None
    dy = dd("latitude") if "latitude" in ds.coords else None
    if "time" in ds.coords and ds.time.size > 1:
        dts = np.median(np.diff(ds.time.values)) / np.timedelta64(1, "D")
        dtime = f"{dts:.1f}D"
    else:
        dtime = None
    return dx, dy, dtime


def manifest_rows(product_key, short, files, ds, vars_kept, temporal):
    dx, dy, dtime = _res(ds)
    today = dt.date.today().isoformat()
    rows = []
    for v in vars_kept:
        if v not in ds:
            continue
        rows.append({
            "product_key": product_key, "short_name": short, "variable": v,
            "units": ds[v].attrs.get("units", ""),
            "bbox_WSEN": str(list(C.BBOX)),
            "start": temporal[0], "stop": temporal[1],
            "n_granules": len(files),
            "spatial_res_lon_deg": dx, "spatial_res_lat_deg": dy,
            "temporal_res": dtime,
            "n_time": int(ds.sizes.get("time", 0)),
            "query_date": today,
            "files": ";".join(Path(f).name for f in files),
        })
    return rows


def write_manifest(all_rows):
    if not all_rows:
        return
    cols = list(all_rows[0].keys())
    with open(MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {MANIFEST} ({len(all_rows)} rows)")


def build_ssh(temporal, n_max, rows):
    res = da.search("ssh", temporal=temporal, count=500)
    da.preview(res, 2)
    if not res:
        raise RuntimeError("No SSH granules found.")
    files = da.download(res, "ssh", n_max=n_max)
    ds = _subset(_norm_coords(da.open_many(files)))
    var = "ssha" if "ssha" in ds else [v for v in ds.data_vars if "ssh" in v.lower()][0]
    out = xr.Dataset(
        {"ssha": (("time", "latitude", "longitude"),
                  ds[var].transpose("time", "latitude", "longitude").values,
                  {"units": ds[var].attrs.get("units", "m")})},
        coords={"time": ds.time.values, "latitude": ds.latitude.values,
                "longitude": ds.longitude.values},
        attrs={"source_product": C.PRODUCTS["ssh"], "synthetic": 0},
    )
    out.to_netcdf(C.DATA_PROC / "nasa_ssh.nc")
    print(f"Saved {C.DATA_PROC/'nasa_ssh.nc'}  dims={dict(out.sizes)}")
    rows += manifest_rows("ssh", C.PRODUCTS["ssh"], files, ds, ["ssha"], temporal)


def build_oscar(temporal, n_max, rows):
    res = da.search("oscar", temporal=temporal, count=500)
    da.preview(res, 2)
    if not res:
        print("No OSCAR granules found; skipping.")
        return
    files = da.download(res, "oscar", n_max=n_max)
    ds = _subset(_norm_coords(da.open_many(files)))
    keep = [v for v in ["u", "v", "ug", "vg"] if v in ds]
    data = {v: (("time", "latitude", "longitude"),
                ds[v].transpose("time", "latitude", "longitude").values,
                {"units": ds[v].attrs.get("units", "m/s")}) for v in keep}
    out = xr.Dataset(data, coords={"time": ds.time.values,
                                   "latitude": ds.latitude.values,
                                   "longitude": ds.longitude.values},
                     attrs={"source_product": C.PRODUCTS["oscar"], "synthetic": 0})
    out.to_netcdf(C.DATA_PROC / "nasa_oscar.nc")
    print(f"Saved {C.DATA_PROC/'nasa_oscar.nc'}  vars={keep}")
    rows += manifest_rows("oscar", C.PRODUCTS["oscar"], files, ds, keep, temporal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="+", default=["ssh", "oscar"])
    ap.add_argument("--start", default=C.EXP_START)
    ap.add_argument("--stop", default=C.EXP_STOP)
    ap.add_argument("--smoke", action="store_true",
                    help="use the short smoke window and cap downloads at 3")
    ap.add_argument("--n-max", type=int, default=None)
    args = ap.parse_args()

    temporal = (C.SMOKE_START, C.SMOKE_STOP) if args.smoke else (args.start, args.stop)
    n_max = 3 if args.smoke else args.n_max

    da.login()
    rows = []
    if "ssh" in args.products:
        build_ssh(temporal, n_max, rows)
    if "oscar" in args.products:
        build_oscar(temporal, n_max, rows)
    write_manifest(rows)
    print("\nDone. Now run:  python src/analysis.py --ssh data_processed/nasa_ssh.nc")
    print("           and:  python src/pinn_ssh.py --ssh data_processed/nasa_ssh.nc "
          "--oscar data_processed/nasa_oscar.nc --tag nasa")


if __name__ == "__main__":
    main()
