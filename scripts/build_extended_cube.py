"""
build_extended_cube.py

Historical extension of the SSH+OSCAR weekly cube for statistical power
(Experiment A: long SSH+OSCAR, NO atmospheric forcing).

  * SSH : NASA_SSH_REF_SIMPLE_GRID_V11 (Version 1.1), weekly, ssha [m].
  * OSCAR: OSCAR_L4_OC_FINAL_V2.0 for nominal dates <= 2026-01-17 (best quality),
           OSCAR_L4_OC_INTERIM_V2.0 afterwards. The FINAL->INTERIM transition
           date is recorded per time step in the output (oscar_product).

Each granule is downloaded to a temp dir, subset to the tropical-Pacific grid,
and deleted immediately to keep disk usage low. Output grid matches the original
cube: lon -170..-90, lat -12..12, 0.5 deg. No wind variables are written.

Usage: python scripts/build_extended_cube.py --start 2023-01-01 --stop 2026-11-01
"""
import os
import re
import glob
import time
import argparse
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess

LONS = np.arange(-170.0, -90.0 + 0.01, 0.5)
LATS = np.arange(-12.0, 12.0 + 0.01, 0.5)
BBOX = (-170.0, -12.0, -90.0, 12.0)
TMP = "data_raw/_ext_tmp"
OUT = "data_processed/forecast_v2/ssh_oscar_weekly_ext.nc"
OSCAR_FINAL_CUTOFF = dt.date(2026, 1, 17)


def login():
    for a in range(5):
        try:
            earthaccess.login(strategy="environment")
            return
        except Exception:
            time.sleep(2 ** a)
    raise RuntimeError("earthdata login failed")


def normalize_lon(lon):
    lon = np.asarray(lon, float)
    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def interp_field(values, lon, lat):
    lon = normalize_lon(lon); lat = np.asarray(lat, float)
    io = np.argsort(lon); jo = np.argsort(lat)
    values = values[np.ix_(jo, io)]
    da = xr.DataArray(values, dims=("lat", "lon"),
                      coords={"lat": lat[jo], "lon": lon[io]})
    return da.interp(lat=LATS, lon=LONS).values


def dl(results, subdir):
    os.makedirs(os.path.join(TMP, subdir), exist_ok=True)
    for a in range(4):
        try:
            return earthaccess.download(results, local_path=os.path.join(TMP, subdir),
                                        threads=1)
        except Exception:
            time.sleep(2 ** a)
    return []


def read_ssh(path):
    ds = xr.open_dataset(path)
    q = ds["ssha"]
    if "time" in q.dims:
        q = q.isel(time=-1)
    q = q.squeeze(drop=True)
    lonn = "longitude" if "longitude" in q.coords else "lon"
    latn = "latitude" if "latitude" in q.coords else "lat"
    q = q.transpose(q[latn].dims[0], q[lonn].dims[0])
    f = interp_field(q.values, q[lonn].values, q[latn].values)
    ds.close()
    return f


def read_oscar(path):
    ds = xr.open_dataset(path, decode_times=False)
    lon = ds["lon"].values; lat = ds["lat"].values
    out = {}
    for name in ["u", "v", "ug", "vg"]:
        q = ds[name]
        if "time" in q.dims:
            q = q.isel(time=0)
        q = q.transpose("latitude", "longitude")
        out[name] = interp_field(q.values, lon, lat)
    ds.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--stop", default="2026-11-01")
    args = ap.parse_args()
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    login()

    # 1. SSH granules (one per nominal week)
    print("searching SSH granules...")
    ssh_res = earthaccess.search_data(short_name="NASA_SSH_REF_SIMPLE_GRID_V11",
                                      bounding_box=BBOX, temporal=(args.start, args.stop),
                                      count=3000)
    # nominal date from filename NASA-SSH_..._YYYYMMDD.nc
    recs = []
    for g in ssh_res:
        fn = None
        for rel in g["umm"].get("RelatedUrls", []):
            if rel.get("Type") == "GET DATA" and rel["URL"].endswith(".nc"):
                fn = rel["URL"].split("/")[-1]
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", fn or "")
        if not m:
            continue
        d = dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        recs.append((d, g))
    recs = sorted({d: g for d, g in recs}.items())
    print(f"SSH weekly granules: {len(recs)}  {recs[0][0]} .. {recs[-1][0]}")

    times, ETA, U, V, UG, VG, OPROD = [], [], [], [], [], [], []
    for n, (date, g) in enumerate(recs, 1):
        # SSH
        f = dl([g], "ssh")
        if not f:
            print(f"  [{n}] SSH dl failed {date}"); continue
        try:
            eta = read_ssh(str(f[0]))
        except Exception as e:
            print(f"  [{n}] SSH read failed {date}: {e}"); os.remove(f[0]); continue
        os.remove(f[0])
        # OSCAR (FINAL or INTERIM)
        prod = "OSCAR_L4_OC_FINAL_V2.0" if date <= OSCAR_FINAL_CUTOFF else "OSCAR_L4_OC_INTERIM_V2.0"
        start = date.strftime("%Y-%m-%d") + "T00:00:00"
        stop = date.strftime("%Y-%m-%d") + "T23:59:59"
        ores = earthaccess.search_data(short_name=prod, bounding_box=BBOX,
                                       temporal=(start, stop), count=2)
        if not ores:
            print(f"  [{n}] no OSCAR {prod} for {date}"); continue
        of = dl(ores[:1], "oscar")
        if not of:
            print(f"  [{n}] OSCAR dl failed {date}"); continue
        try:
            cur = read_oscar(str(of[0]))
        except Exception as e:
            print(f"  [{n}] OSCAR read failed {date}: {e}"); os.remove(of[0]); continue
        os.remove(of[0])
        times.append(np.datetime64(date.isoformat()))
        ETA.append(eta); U.append(cur["u"]); V.append(cur["v"])
        UG.append(cur["ug"]); VG.append(cur["vg"]); OPROD.append(prod.split("_")[3])
        if n % 20 == 0:
            print(f"  [{n}/{len(recs)}] {date} ok ({prod.split('_')[3]})")

    print(f"matched dates: {len(times)}")
    ds = xr.Dataset(
        {"ssha": (("time", "lat", "lon"), np.asarray(ETA)),
         "u": (("time", "lat", "lon"), np.asarray(U)),
         "v": (("time", "lat", "lon"), np.asarray(V)),
         "ug": (("time", "lat", "lon"), np.asarray(UG)),
         "vg": (("time", "lat", "lon"), np.asarray(VG)),
         "oscar_product": (("time",), np.asarray(OPROD))},
        coords={"time": np.asarray(times), "lat": LATS, "lon": LONS},
        attrs={"description": "Extended weekly SSH(V1.1)+OSCAR cube, no wind (Experiment A)",
               "ssh_collection": "NASA_SSH_REF_SIMPLE_GRID_V11",
               "oscar_final_cutoff": str(OSCAR_FINAL_CUTOFF),
               "note": "OSCAR FINAL for dates<=cutoff else INTERIM; see oscar_product"})
    for v in ["ssha"]:
        ds[v].attrs["units"] = "m"
    for v in ["u", "v", "ug", "vg"]:
        ds[v].attrs["units"] = "m s-1"
    ds.to_netcdf(OUT)
    print("SAVED", OUT, "n_time", len(times))
    # cleanup temp
    for p in glob.glob(os.path.join(TMP, "**", "*"), recursive=True):
        try:
            os.remove(p)
        except Exception:
            pass


if __name__ == "__main__":
    main()
