#!/usr/bin/env python3
"""
build_oscar_weekly.py  --  cost-controlled OSCAR currents builder.

OSCAR_L4_OC_FINAL_V2.0 is DAILY and global (~32 MB/granule). For an 18-month
window that is ~547 granules / ~17 GB -- far more than the independent
velocity comparison needs, since the SSH product is weekly. This script
subsamples OSCAR to ~weekly spacing (nearest granule to each requested date),
downloads only those, subsets to the tropical-Pacific study region, and writes
data_processed/nasa_oscar.nc with the SAME schema the surrogate/validation code
expects (u,v,ug,vg on time,latitude,longitude; time as datetime64).

OSCAR is NEVER used to train the PINN; it is an independent comparison product.

Usage:
    python src/build_oscar_weekly.py --start 2015-01-01 --stop 2016-06-30 --every 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                 # noqa: E402
import data_access as da           # noqa: E402


def granule_date(g):
    """Best-effort granule start date as pandas Timestamp."""
    try:
        t = g["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"]
        return pd.Timestamp(t).tz_localize(None)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=C.EXP_START)
    ap.add_argument("--stop", default=C.EXP_STOP)
    ap.add_argument("--every", type=int, default=7,
                    help="target spacing in days (subsample daily OSCAR)")
    args = ap.parse_args()

    da.login()
    res = da.search("oscar", temporal=(args.start, args.stop), count=2000)
    if not res:
        raise RuntimeError("No OSCAR granules found.")

    dated = [(granule_date(g), g) for g in res]
    dated = [(d, g) for d, g in dated if d is not None]
    dated.sort(key=lambda x: x[0])

    # Greedy weekly subsample: keep a granule if >= every days since last kept.
    kept = []
    last = None
    for d, g in dated:
        if last is None or (d - last).days >= args.every:
            kept.append(g)
            last = d
    print(f"OSCAR: {len(res)} daily granules -> {len(kept)} subsampled "
          f"(~every {args.every} d)")

    files = da.download(kept, "oscar", n_max=None)

    # open individually (OSCAR time is cftime Julian; concat after conversion)
    dss = []
    for f in sorted(files):
        if not str(f).endswith((".nc", ".nc4")):
            continue
        d = xr.open_dataset(f)
        ren = {}
        if "lon" in d.coords:
            ren["lon"] = "longitude"
        if "lat" in d.coords:
            ren["lat"] = "latitude"
        d = d.rename(ren) if ren else d
        if float(d.longitude.max()) > 180.0:
            lon = ((d.longitude + 180.0) % 360.0) - 180.0
            d = d.assign_coords(longitude=lon).sortby("longitude")
        if d.latitude.values[0] > d.latitude.values[-1]:
            d = d.sortby("latitude")
        d = d.sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                  latitude=slice(C.LAT_MIN, C.LAT_MAX))
        # normalise time to numpy datetime64 (from cftime)
        tv = d.time.values
        try:
            tnew = np.array([np.datetime64(pd.Timestamp(
                t.isoformat()).to_datetime64()) for t in tv])
        except Exception:
            tnew = np.array([np.datetime64(str(t)[:10]) for t in tv])
        d = d.assign_coords(time=tnew)
        dss.append(d[["u", "v", "ug", "vg"]])

    ds = xr.concat(dss, dim="time").sortby("time")
    ds = ds.transpose("time", "latitude", "longitude")
    out = xr.Dataset(
        {v: (("time", "latitude", "longitude"), ds[v].values,
             {"units": ds[v].attrs.get("units", "m/s")})
         for v in ["u", "v", "ug", "vg"]},
        coords={"time": ds.time.values, "latitude": ds.latitude.values,
                "longitude": ds.longitude.values},
        attrs={"source_product": C.PRODUCTS["oscar"], "synthetic": 0,
               "note": "weekly-subsampled OSCAR total (u,v) & geostrophic (ug,vg); "
                       "independent comparison only, NOT used in PINN training"},
    )
    out.to_netcdf(C.DATA_PROC / "nasa_oscar.nc")
    print(f"Saved {C.DATA_PROC/'nasa_oscar.nc'}  dims={dict(out.sizes)} "
          f"vars={list(out.data_vars)}")
    print(f"  time {str(out.time.values[0])[:10]} .. {str(out.time.values[-1])[:10]}")


if __name__ == "__main__":
    main()
