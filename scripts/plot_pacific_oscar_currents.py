import os
import re
import glob
import datetime as dt

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import earthaccess


# ============================================================
# SETTINGS
# ============================================================

SSH_DIR = "data_raw/nasa_ssh_current"
FIG_DIR = "figures"
OSCAR_DIR = "data_raw/oscar_current_fixed"

X0, X1 = -170.0, -90.0
Y0, Y1 =  -20.0,  20.0

SSH_DATE_FALLBACK = "2026-11-02"

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OSCAR_DIR, exist_ok=True)


# ============================================================
# DATE HELPERS
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


def object_to_date(x):
    x = np.atleast_1d(x)[0]

    if (
        hasattr(x, "year")
        and hasattr(x, "month")
        and hasattr(x, "day")
    ):
        return dt.date(
            int(x.year),
            int(x.month),
            int(x.day)
        )

    return extract_date(x)


def infer_date(ds, filename, fallback=None):

    # Prefer coverage metadata.
    for key in [
        "time_coverage_start",
        "RangeBeginningDate",
        "range_beginning_date",
        "start_date",
    ]:
        if key in ds.attrs:
            d = extract_date(ds.attrs[key])
            if d:
                return d, f"attribute {key}"

    # Then time coordinate.
    if "time" in ds.coords:
        d = object_to_date(ds["time"].values)
        if d:
            return d, "time coordinate"

    # Then filename.
    d = extract_date(os.path.basename(filename))
    if d:
        return d, "filename"

    if fallback:
        return dt.date.fromisoformat(fallback), "fallback"

    return None, "unknown"


# ============================================================
# FIND PHYSICAL COORDINATE VALUES
# ============================================================

def find_dim_names(da):
    """
    Find spatial DIMENSION names.
    """
    lon_dim = None
    lat_dim = None

    for d in da.dims:
        dl = d.lower()

        if "lon" in dl:
            lon_dim = d

        if "lat" in dl:
            lat_dim = d

    if lon_dim is None or lat_dim is None:
        raise RuntimeError(
            f"Cannot identify spatial dimensions for {da.name}: "
            f"{da.dims}"
        )

    return lon_dim, lat_dim


def plausible_lon(v):
    v = np.asarray(v)

    return (
        v.ndim == 1
        and v.size > 10
        and np.isfinite(v).any()
        and np.nanmin(v) >= -360.1
        and np.nanmax(v) <= 360.1
        and np.nanmax(v) - np.nanmin(v) > 100
    )


def plausible_lat(v):
    v = np.asarray(v)

    return (
        v.ndim == 1
        and v.size > 10
        and np.isfinite(v).any()
        and np.nanmin(v) >= -90.1
        and np.nanmax(v) <= 90.1
        and np.nanmax(v) - np.nanmin(v) > 20
    )


def physical_coord(da, dim, kind):
    """
    Find actual degree values corresponding to dimension `dim`.

    Works whether:
      longitude itself contains degree values, or
      auxiliary coord `lon` has dimensions ('longitude',).
    """

    test = plausible_lon if kind == "lon" else plausible_lat

    # First try coordinate with same name as dimension.
    if dim in da.coords:
        c = da.coords[dim]

        if c.ndim == 1 and c.dims == (dim,):
            vals = np.asarray(c.values)

            if test(vals):
                return dim, vals

    # Search all 1-D coordinates attached to this dimension.
    for name, c in da.coords.items():

        if c.ndim != 1:
            continue

        if c.dims != (dim,):
            continue

        vals = np.asarray(c.values)

        name_low = name.lower()

        if kind == "lon":
            name_match = "lon" in name_low
        else:
            name_match = "lat" in name_low

        if name_match and test(vals):
            return name, vals

    # Last resort: look for any plausible 1-D coordinate
    # living on this dimension.
    for name, c in da.coords.items():

        if c.ndim == 1 and c.dims == (dim,):
            vals = np.asarray(c.values)

            if test(vals):
                return name, vals

    raise RuntimeError(
        f"Could not find physical {kind} coordinate for dimension "
        f"{dim}. Available coords: "
        f"{[(k, v.dims, v.shape) for k,v in da.coords.items()]}"
    )


def lon180(values):
    values = np.asarray(values, dtype=float)

    if np.nanmax(values) > 180:
        return ((values + 180.0) % 360.0) - 180.0

    return values


# ============================================================
# CROP BY INTEGER INDICES, NOT .sel()
# ============================================================

def crop_numeric(
    da,
    lon_dim,
    lat_dim,
    lon_values,
    lat_values,
    x0,
    x1,
    y0,
    y1
):
    """
    Crop using Boolean masks and integer isel(), avoiding
    xarray label-index assumptions entirely.

    Returns:
       cropped DataArray in [lat,lon] order,
       physical longitude array,
       physical latitude array
    """

    LON = lon180(lon_values)
    LAT = np.asarray(lat_values, dtype=float)

    im_lon = np.where(
        np.isfinite(LON)
        &
        (LON >= x0)
        &
        (LON <= x1)
    )[0]

    im_lat = np.where(
        np.isfinite(LAT)
        &
        (LAT >= y0)
        &
        (LAT <= y1)
    )[0]

    if len(im_lon) == 0:
        raise RuntimeError(
            f"No longitude points in {x0}..{x1}; "
            f"range is {np.nanmin(LON)}..{np.nanmax(LON)}"
        )

    if len(im_lat) == 0:
        raise RuntimeError(
            f"No latitude points in {y0}..{y1}; "
            f"range is {np.nanmin(LAT)}..{np.nanmax(LAT)}"
        )

    # Sort by physical longitude/latitude.
    lon_order = np.argsort(LON[im_lon])
    lat_order = np.argsort(LAT[im_lat])

    ilon = im_lon[lon_order]
    ilat = im_lat[lat_order]

    q = da.isel({
        lon_dim: ilon,
        lat_dim: ilat
    })

    q = q.transpose(
        lat_dim,
        lon_dim
    )

    lon_plot = LON[ilon]
    lat_plot = LAT[ilat]

    return q, lon_plot, lat_plot


# ============================================================
# 1. OPEN NASA SSH
# ============================================================

ssh_files = sorted(
    glob.glob(os.path.join(SSH_DIR, "*.nc"))
    +
    glob.glob(os.path.join(SSH_DIR, "*.nc4"))
)

if not ssh_files:
    raise RuntimeError(
        f"No SSH files found in {SSH_DIR}"
    )

ssh_file = ssh_files[-1]

print()
print("NASA SSH FILE")
print("-------------")
print(ssh_file)

ssh = xr.open_dataset(ssh_file)

print()
print("dims:", ssh.sizes)
print("coords:", list(ssh.coords))
print("vars:", list(ssh.data_vars))


if "ssha" in ssh.data_vars:
    ssh_var = "ssha"
else:
    candidates = [
        v for v in ssh.data_vars
        if "ssh" in v.lower()
        and "qual" not in v.lower()
        and "uncert" not in v.lower()
    ]

    if not candidates:
        raise RuntimeError(
            "No SSH-like variable found."
        )

    ssh_var = candidates[0]


eta = ssh[ssh_var]

if "time" in eta.dims:
    eta = eta.isel(time=-1)

eta = eta.squeeze(drop=True)

ssh_lon_dim, ssh_lat_dim = find_dim_names(eta)

ssh_lon_name, ssh_lon_values = physical_coord(
    eta,
    ssh_lon_dim,
    "lon"
)

ssh_lat_name, ssh_lat_values = physical_coord(
    eta,
    ssh_lat_dim,
    "lat"
)

print()
print("SSH variable:", ssh_var)
print("SSH longitude dimension:", ssh_lon_dim)
print("SSH physical lon coordinate:", ssh_lon_name)
print("SSH latitude dimension:", ssh_lat_dim)
print("SSH physical lat coordinate:", ssh_lat_name)


ssh_date, ssh_date_source = infer_date(
    ssh,
    ssh_file,
    SSH_DATE_FALLBACK
)

print()
print("SSH date:", ssh_date)
print("SSH date source:", ssh_date_source)


eta_p, eta_lon, eta_lat = crop_numeric(
    eta,
    ssh_lon_dim,
    ssh_lat_dim,
    ssh_lon_values,
    ssh_lat_values,
    X0,
    X1,
    Y0,
    Y1
)


# ============================================================
# 2. GET OSCAR NEAR SAME DATE
# ============================================================

print()
print("EARTHDATA / OSCAR")
print("-----------------")

earthaccess.login(
    strategy="environment"
)

search_start = ssh_date - dt.timedelta(days=7)
search_end   = ssh_date + dt.timedelta(days=7)

print(
    "search interval:",
    search_start,
    "to",
    search_end
)

results = earthaccess.search_data(
    short_name="OSCAR_L4_OC_NRT_V2.0",
    temporal=(
        str(search_start),
        str(search_end)
    ),
    count=50
)

print("search results:", len(results))

if not results:
    raise RuntimeError(
        "No OSCAR search results."
    )


downloaded = earthaccess.download(
    results,
    local_path=OSCAR_DIR,
    threads=4
)

files = sorted(
    set(
        [str(x) for x in downloaded]
        +
        glob.glob(OSCAR_DIR + "/*.nc")
        +
        glob.glob(OSCAR_DIR + "/*.nc4")
    )
)

files = [
    f for f in files
    if os.path.isfile(f)
]

if not files:
    raise RuntimeError(
        "No OSCAR NetCDF files found after download."
    )


# ============================================================
# 3. PICK CLOSEST OSCAR FILE
#
# Prefer time_coverage_start metadata if present.
# Also print the internal time coordinate, because the current
# file showed a discrepancy we want to see explicitly.
# ============================================================

candidates = []

print()
print("OSCAR CANDIDATE DATES")
print("---------------------")

for f in files:

    try:
        ds = xr.open_dataset(
            f,
            decode_times=True
        )

        metadata_date = None
        internal_date = None

        if "time_coverage_start" in ds.attrs:
            metadata_date = extract_date(
                ds.attrs["time_coverage_start"]
            )

        if "time" in ds.coords:
            internal_date = object_to_date(
                ds["time"].values
            )

        choose_date = (
            metadata_date
            if metadata_date is not None
            else internal_date
        )

        print()
        print(os.path.basename(f))
        print(" metadata coverage date:", metadata_date)
        print(" internal time date    :", internal_date)

        if choose_date is not None:
            delta = abs(
                (choose_date - ssh_date).days
            )

            candidates.append(
                (
                    delta,
                    choose_date,
                    internal_date,
                    f
                )
            )

        ds.close()

    except Exception as e:
        print(
            "could not inspect:",
            f,
            repr(e)
        )


if not candidates:
    raise RuntimeError(
        "No OSCAR candidate had a usable date."
    )

candidates.sort(
    key=lambda x: x[0]
)

delta_days, oscar_nominal_date, oscar_internal_date, oscar_file = candidates[0]

print()
print("CHOSEN OSCAR FILE")
print("-----------------")
print(oscar_file)
print("nominal/coverage date:", oscar_nominal_date)
print("internal time date   :", oscar_internal_date)
print("difference from SSH  :", delta_days, "days")


# ============================================================
# 4. LOAD CHOSEN OSCAR FILE
# ============================================================

oc = xr.open_dataset(
    oscar_file,
    decode_times=True
)

print()
print("OSCAR DATASET")
print("-------------")
print("dims:", oc.sizes)
print("coords:", list(oc.coords))

for name in ["u", "v", "ug", "vg"]:
    if name not in oc.data_vars:
        raise RuntimeError(
            f"OSCAR missing {name}; vars={list(oc.data_vars)}"
        )

    print(
        name,
        "dims=",
        oc[name].dims,
        "shape=",
        oc[name].shape
    )


# Work from u to identify actual spatial dimensions/coords.
probe = oc["u"]

if "time" in probe.dims:
    probe = probe.isel(time=0)

probe = probe.squeeze(drop=True)

oscar_lon_dim, oscar_lat_dim = find_dim_names(probe)

oscar_lon_name, oscar_lon_values = physical_coord(
    probe,
    oscar_lon_dim,
    "lon"
)

oscar_lat_name, oscar_lat_values = physical_coord(
    probe,
    oscar_lat_dim,
    "lat"
)

print()
print("Detected OSCAR spatial dimensions:")
print(" longitude dimension:", oscar_lon_dim)
print(" physical lon coord  :", oscar_lon_name)
print(" latitude dimension  :", oscar_lat_dim)
print(" physical lat coord  :", oscar_lat_name)

print()
print(
    "OSCAR longitude physical range:",
    np.nanmin(oscar_lon_values),
    "to",
    np.nanmax(oscar_lon_values)
)

print(
    "OSCAR latitude physical range:",
    np.nanmin(oscar_lat_values),
    "to",
    np.nanmax(oscar_lat_values)
)


# ============================================================
# 5. EXTRACT A CURRENT FIELD
# ============================================================

def get_current(name):

    q = oc[name]

    if "time" in q.dims:
        q = q.isel(time=0)

    q = q.squeeze(drop=True)

    # We now know spatial dims exactly.
    extra = [
        d for d in q.dims
        if d not in (
            oscar_lon_dim,
            oscar_lat_dim
        )
    ]

    for d in extra:
        print(
            name,
            ": selecting first index of genuine extra dimension",
            d
        )
        q = q.isel({d: 0})

    q, LON, LAT = crop_numeric(
        q,
        oscar_lon_dim,
        oscar_lat_dim,
        oscar_lon_values,
        oscar_lat_values,
        X0,
        X1,
        Y0,
        Y1
    )

    return q, LON, LAT


u, oscar_lon, oscar_lat = get_current("u")
v, _, _ = get_current("v")

ug, _, _ = get_current("ug")
vg, _, _ = get_current("vg")


print()
print("FINAL CROPPED SHAPES")
print("--------------------")
print("SSH:", eta_p.shape)
print("u :", u.shape)
print("v :", v.shape)
print("ug:", ug.shape)
print("vg:", vg.shape)


# ============================================================
# 6. VELOCITY STATISTICS
# ============================================================

def stats(label, U, V):

    speed = np.sqrt(
        U.values**2
        +
        V.values**2
    )

    speed = speed[
        np.isfinite(speed)
    ]

    print()
    print(label)

    if speed.size == 0:
        print(" no finite data")
        return

    print(
        " median:",
        float(np.median(speed)),
        "m/s"
    )

    print(
        " 95th percentile:",
        float(np.percentile(speed, 95)),
        "m/s"
    )

    print(
        " max:",
        float(np.max(speed)),
        "m/s"
    )


stats(
    "TOTAL OSCAR WATER SPEED",
    u,
    v
)

stats(
    "GEOSTROPHIC OSCAR WATER SPEED",
    ug,
    vg
)


# ============================================================
# 7. PLOT
# ============================================================

def plot_field(U, V, title, filename):

    fig, ax = plt.subplots(
        figsize=(15, 7)
    )

    finite_eta = eta_p.values[
        np.isfinite(eta_p.values)
    ]

    vmax = max(
        0.05,
        float(
            np.percentile(
                np.abs(finite_eta),
                98
            )
        )
    )

    p = ax.pcolormesh(
        eta_lon,
        eta_lat,
        eta_p.values,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    cb = fig.colorbar(
        p,
        ax=ax,
        pad=0.015
    )

    cb.set_label(
        "Sea-surface height anomaly [m]"
    )


    LON2, LAT2 = np.meshgrid(
        oscar_lon,
        oscar_lat
    )

    skip = 8

    q = ax.quiver(
        LON2[::skip, ::skip],
        LAT2[::skip, ::skip],
        U.values[::skip, ::skip],
        V.values[::skip, ::skip],
        color="black",
        scale=12,
        width=0.0017
    )

    ax.quiverkey(
        q,
        0.79,
        1.035,
        0.5,
        "0.5 m/s water current",
        coordinates="axes",
        labelpos="E"
    )


    # Equator
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )


    # Niño 3.4
    ax.plot(
        [-170, -120, -120, -170, -170],
        [-5, -5, 5, 5, -5],
        color="black",
        linestyle=":",
        linewidth=1.3
    )

    ax.text(
        -168,
        6.2,
        "Niño-3.4",
        fontsize=9,
        weight="bold"
    )


    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    title2 = (
        title
        + "\n"
        + f"SSH nominal date: {ssh_date}; "
        + f"OSCAR nominal date: {oscar_nominal_date}"
    )

    # Explicitly disclose internal OSCAR time discrepancy.
    if (
        oscar_internal_date is not None
        and oscar_internal_date != oscar_nominal_date
    ):
        title2 += (
            f"  [OSCAR internal time={oscar_internal_date}]"
        )

    ax.set_title(title2)

    plt.tight_layout()

    plt.savefig(
        filename,
        dpi=240
    )

    plt.close()

    print("SAVED:", filename)


plot_field(
    u,
    v,
    "NASA SSH with OSCAR total surface-water velocity",
    "figures/Pacific_SSH_OSCAR_total_currents.png"
)

plot_field(
    ug,
    vg,
    "NASA SSH with OSCAR geostrophic surface-water velocity",
    "figures/Pacific_SSH_OSCAR_geostrophic_currents.png"
)

plot_field(
    u - ug,
    v - vg,
    "NASA SSH with OSCAR total minus geostrophic velocity",
    "figures/Pacific_SSH_OSCAR_nongeostrophic_currents.png"
)


print()
print("DONE")
print("====")
print(
    "figures/Pacific_SSH_OSCAR_total_currents.png"
)
print(
    "figures/Pacific_SSH_OSCAR_geostrophic_currents.png"
)
print(
    "figures/Pacific_SSH_OSCAR_nongeostrophic_currents.png"
)
