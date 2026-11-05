#!/usr/bin/env python3
"""Write data_manifest.csv documenting the produced OSSE surrogate files and
the real NASA products targeted for the experiment. For REAL downloads,
build_dataset.py overwrites data_manifest.csv with the actual granule record."""
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C      # noqa: E402

COLS = ["product_key", "short_name", "variable", "units", "status",
        "bbox_WSEN", "start", "stop", "spatial_res_lon_deg",
        "spatial_res_lat_deg", "temporal_res", "n_time", "query_date", "files"]


def res(ds, name):
    v = ds[name].values
    return round(float(np.median(np.abs(np.diff(v)))), 4) if len(v) > 1 else ""


def surrogate_rows():
    rows = []
    today = dt.date.today().isoformat()
    specs = [
        ("surrogate_ssh.nc", "ssh", C.PRODUCTS["ssh"], ["ssha"], "m"),
        ("surrogate_oscar.nc", "oscar", C.PRODUCTS["oscar"], ["u", "v"], "m/s"),
        ("surrogate_truth.nc", "truth", "(analytic)", ["u", "v", "eta"], "m/s,m"),
    ]
    for fname, key, short, vars_, units in specs:
        p = C.DATA_PROC / fname
        if not p.exists():
            continue
        ds = xr.open_dataset(p)
        for v in vars_:
            rows.append({
                "product_key": key, "short_name": short, "variable": v,
                "units": ds[v].attrs.get("units", units),
                "status": "SYNTHETIC OSSE surrogate (NOT NASA data)",
                "bbox_WSEN": str(list(C.BBOX)),
                "start": str(np.datetime_as_string(ds.time.values[0], unit="D")),
                "stop": str(np.datetime_as_string(ds.time.values[-1], unit="D")),
                "spatial_res_lon_deg": res(ds, "longitude"),
                "spatial_res_lat_deg": res(ds, "latitude"),
                "temporal_res": f"{np.median(np.diff(ds.time.values))/np.timedelta64(1,'D'):.0f}D",
                "n_time": int(ds.sizes["time"]), "query_date": today, "files": fname,
            })
    return rows


def planned_rows():
    """Real NASA products targeted (to be fetched where Earthdata is reachable)."""
    today = dt.date.today().isoformat()
    plan = [
        ("ssh", C.PRODUCTS["ssh"], "ssha", "m", "~0.5", "~0.5", "7D"),
        ("oscar", C.PRODUCTS["oscar"], "u;v;ug;vg", "m/s", "0.25", "0.25", "5D"),
        ("sst", C.PRODUCTS["sst"], "analysed_sst", "K", "0.01", "0.01", "1D"),
        ("sss", C.PRODUCTS["sss"], "smap_sss", "PSU", "0.25", "0.25", "8D"),
        ("swot", C.PRODUCTS["swot"], "ssha_karin;ssh_karin_qual", "m",
         "swath", "swath", "~21D repeat"),
    ]
    rows = []
    for key, short, var, units, dx, dy, dtr in plan:
        rows.append({
            "product_key": key, "short_name": short, "variable": var, "units": units,
            "status": "REAL NASA product -- pending Earthdata access",
            "bbox_WSEN": str(list(C.BBOX)), "start": C.EXP_START, "stop": C.EXP_STOP,
            "spatial_res_lon_deg": dx, "spatial_res_lat_deg": dy, "temporal_res": dtr,
            "n_time": "", "query_date": today, "files": "(run src/build_dataset.py)",
        })
    return rows


def main():
    rows = surrogate_rows() + planned_rows()
    with open(C.ROOT / "data_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {C.ROOT/'data_manifest.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
