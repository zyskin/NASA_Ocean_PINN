import os
import re
import glob
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
import earthaccess


SSH_DIR = "data_raw/nasa_ssh_2026"
OSCAR_DIR = "data_raw/oscar_weekly"
OUT = "data_processed/ssh_oscar_weekly_pacific.nc"

LONS = np.arange(-170.0, -90.0 + 0.01, 0.5)
LATS = np.arange(-12.0,   12.0 + 0.01, 0.5)

os.makedirs(OSCAR_DIR, exist_ok=True)
os.makedirs("data_processed", exist_ok=True)


# ============================================================
# DATE UTILITIES
# ============================================================

def extract_date(x):
    s = str(x)

    m = re.search(
        r"(20\d{2})[-_/]([01]\d)[-_/]([0-3]\d)",
        s
    )
    if m:
        return dt.date(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3))
        )

    m = re.search(
        r"(20\d{2})([01]\d)([0-3]\d)",
        s
    )
    if m:
        return dt.date(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3))
        )

    return None


def infer_date(ds, filename):

    for key in [
        "time_coverage_start",
        "RangeBeginningDate",
        "range_beginning_date",
        "start_date",
        "date",
    ]:
        if key in ds.attrs:
            d = extract_date(ds.attrs[key])
            if d:
                return d

    if "time" in ds.coords:
        vals = np.atleast_1d(ds["time"].values)

        if len(vals):
            x = vals[-1]

            if hasattr(x, "year"):
                return dt.date(
                    int(x.year),
                    int(x.month),
                    int(x.day)
                )

            d = extract_date(x)
            if d:
                return d

    return extract_date(
        os.path.basename(filename)
    )


# ============================================================
# GENERIC GRID EXTRACTION
# ============================================================

def normalize_lon(lon):
    lon = np.asarray(lon, dtype=float)

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180

    return lon


def interp_field(
    values,
    lon,
    lat,
    target_lon=LONS,
    target_lat=LATS
):
    """
    values must be latitude x longitude.
    """

    lon = normalize_lon(lon)
    lat = np.asarray(lat, dtype=float)

    io = np.argsort(lon)
    jo = np.argsort(lat)

    lon = lon[io]
    lat = lat[jo]

    values = values[np.ix_(jo, io)]

    da = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={
            "lat": lat,
            "lon": lon,
        }
    )

    out = da.interp(
        lat=target_lat,
        lon=target_lon
    )

    return out.values


# ============================================================
# LOAD ONE SSH MAP
# ============================================================

def read_ssh(filename):

    ds = xr.open_dataset(filename)

    date = infer_date(ds, filename)

    if date is None:
        ds.close()
        raise RuntimeError(
            f"Cannot infer date from SSH file {filename}"
        )

    if "ssha" in ds.data_vars:
        name = "ssha"
    else:
        candidates = [
            v for v in ds.data_vars
            if "ssh" in v.lower()
            and "qual" not in v.lower()
            and "uncert" not in v.lower()
        ]

        if not candidates:
            ds.close()
            raise RuntimeError(
                f"No SSH field in {filename}"
            )

        name = candidates[0]

    q = ds[name]

    if "time" in q.dims:
        q = q.isel(time=-1)

    q = q.squeeze(drop=True)

    # coordinate names
    lon_name = (
        "longitude"
        if "longitude" in q.coords
        else "lon"
    )

    lat_name = (
        "latitude"
        if "latitude" in q.coords
        else "lat"
    )

    # force lat x lon
    lat_dim = q[lat_name].dims[0]
    lon_dim = q[lon_name].dims[0]

    q = q.transpose(
        lat_dim,
        lon_dim
    )

    field = interp_field(
        q.values,
        q[lon_name].values,
        q[lat_name].values
    )

    ds.close()

    return date, field


# ============================================================
# OSCAR NOMINAL DATE
# ============================================================

def oscar_nominal_date(ds, filename):

    if "time_coverage_start" in ds.attrs:
        d = extract_date(
            ds.attrs["time_coverage_start"]
        )
        if d:
            return d

    d = extract_date(
        os.path.basename(filename)
    )

    return d


# ============================================================
# FIND / DOWNLOAD OSCAR FOR ONE TARGET DATE
# ============================================================

def get_oscar_file(target):

    stamp = target.strftime("%Y%m%d")

    # First look in cache by filename.
    local = sorted(
        glob.glob(
            os.path.join(
                OSCAR_DIR,
                f"*{stamp}*.nc*"
            )
        )
    )

    if local:
        return local[0]

    print(
        " Earthdata search:",
        target
    )

    start = (
        target.strftime("%Y-%m-%d")
        + "T00:00:00"
    )

    stop = (
        target.strftime("%Y-%m-%d")
        + "T23:59:59"
    )

    results = earthaccess.search_data(
        short_name="OSCAR_L4_OC_NRT_V2.0",
        temporal=(start, stop),
        count=5
    )

    if not results:
        print(
            "  no exact-day OSCAR result"
        )
        return None

    files = earthaccess.download(
        results[:1],
        local_path=OSCAR_DIR,
        threads=1
    )

    if not files:
        return None

    return str(files[0])


# ============================================================
# READ ONE OSCAR FILE
# ============================================================

def read_oscar(filename, target_date):

    #
    # decode_times=False deliberately avoids the Julian
    # calendar issue we encountered.
    #
    ds = xr.open_dataset(
        filename,
        decode_times=False
    )

    nominal = oscar_nominal_date(
        ds,
        filename
    )

    print(
        "  OSCAR nominal:",
        nominal,
        "target:",
        target_date
    )

    if nominal is not None:
        dd = abs(
            (nominal - target_date).days
        )

        if dd > 1:
            ds.close()
            raise RuntimeError(
                f"OSCAR file nominal date {nominal} is "
                f"{dd} days from target {target_date}"
            )

    # v2 structure:
    #
    # u(time, longitude, latitude)
    # lon(longitude)
    # lat(latitude)

    lon = ds["lon"].values
    lat = ds["lat"].values

    output = {}

    for name in ["u", "v", "ug", "vg"]:

        q = ds[name]

        if "time" in q.dims:
            q = q.isel(time=0)

        q = q.transpose(
            "latitude",
            "longitude"
        )

        output[name] = interp_field(
            q.values,
            lon,
            lat
        )

    ds.close()

    return output


# ============================================================
# MAIN
# ============================================================

earthaccess.login(
    strategy="environment"
)

ssh_files = sorted(
    glob.glob(
        os.path.join(SSH_DIR, "*.nc")
    )
    +
    glob.glob(
        os.path.join(SSH_DIR, "*.nc4")
    )
)

if not ssh_files:
    raise RuntimeError(
        f"No SSH files in {SSH_DIR}"
    )


# First determine all SSH dates.
ssh_records = []

for f in ssh_files:

    try:
        date, eta = read_ssh(f)

        ssh_records.append(
            (date, f, eta)
        )

    except Exception as e:
        print(
            "SKIP SSH:",
            f,
            repr(e)
        )


# one SSH file per date
unique = {}

for date, f, eta in ssh_records:
    unique[date] = (f, eta)

ssh_records = [
    (date, unique[date][0], unique[date][1])
    for date in sorted(unique)
]


print()
print(
    "SSH maps with usable dates:",
    len(ssh_records)
)

if len(ssh_records) < 10:
    raise RuntimeError(
        "Too few dated SSH maps."
    )


times = []

ETAS = []
US = []
VS = []
UGS = []
VGS = []


for n, (date, sshfile, eta) in enumerate(
    ssh_records,
    start=1
):

    print()
    print(
        f"[{n}/{len(ssh_records)}]",
        date
    )

    ofile = get_oscar_file(date)

    if ofile is None:
        print(
            "  no OSCAR; skipping this week"
        )
        continue

    try:
        currents = read_oscar(
            ofile,
            date
        )

    except Exception as e:
        print(
            "  OSCAR read failed:",
            repr(e)
        )
        continue

    times.append(
        np.datetime64(date.isoformat())
    )

    ETAS.append(eta)
    US.append(currents["u"])
    VS.append(currents["v"])
    UGS.append(currents["ug"])
    VGS.append(currents["vg"])


if len(times) < 12:
    raise RuntimeError(
        "Too few matched SSH/OSCAR dates."
    )


dsout = xr.Dataset(
    data_vars={
        "ssha": (
            ("time", "lat", "lon"),
            np.asarray(ETAS)
        ),
        "u": (
            ("time", "lat", "lon"),
            np.asarray(US)
        ),
        "v": (
            ("time", "lat", "lon"),
            np.asarray(VS)
        ),
        "ug": (
            ("time", "lat", "lon"),
            np.asarray(UGS)
        ),
        "vg": (
            ("time", "lat", "lon"),
            np.asarray(VGS)
        ),
    },
    coords={
        "time": np.asarray(times),
        "lat": LATS,
        "lon": LONS,
    }
)

dsout["ssha"].attrs["units"] = "m"
dsout["u"].attrs["units"] = "m s-1"
dsout["v"].attrs["units"] = "m s-1"
dsout["ug"].attrs["units"] = "m s-1"
dsout["vg"].attrs["units"] = "m s-1"

dsout.attrs[
    "description"
] = (
    "Weekly matched NASA SSH and OSCAR current fields "
    "for equatorial Pacific Rossby-mode analysis."
)

dsout.to_netcdf(OUT)

print()
print("SAVED:", OUT)
print("matched dates:", len(times))
print(
    "range:",
    str(times[0]),
    "to",
    str(times[-1])
)
