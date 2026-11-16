import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


INFILE = "data_processed/ssh_oscar_weekly_pacific.nc"
OUTFILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

# GEOS-FP analysis OPeNDAP.
HOST = "opendap.nccs.nasa.gov"
URL = (
    "https://" + HOST
    + "/dods/GEOS-5/fp/0.25_deg/assim/tavg1_2d_slv_Nx"
)

RHO_AIR = 1.225       # kg/m^3
CD_AIR = 1.3e-3       # first-pass bulk drag coefficient

os.makedirs("figures", exist_ok=True)

local = xr.open_dataset(INFILE)

print("Local cube:")
print(local)

lon_target = local.lon
lat_target = local.lat

print()
print("Opening NASA GEOS-FP hourly analysis...")
geos = xr.open_dataset(
    URL,
    engine="netcdf4",
    decode_times=True
)

print("GEOS dimensions:", geos.sizes)
print("GEOS available vars include:",
      [x for x in ["u10m","v10m","slp"] if x in geos])

needed = ["u10m", "v10m"]

for v in needed:
    if v not in geos:
        raise RuntimeError(f"GEOS missing {v}")

have_slp = "slp" in geos


# ------------------------------------------------------------
# Build mapping from calendar day to hourly time indices.
# Avoid calendar-class comparison problems entirely.
# ------------------------------------------------------------

gt = np.atleast_1d(geos.time.values)

calendar_keys = []

for x in gt:
    # GEOS time may decode either as numpy.datetime64,
    # pandas Timestamp, Python datetime, or cftime.
    if hasattr(x, "year"):
        year  = int(x.year)
        month = int(x.month)
        day   = int(x.day)
    else:
        ts = pd.Timestamp(x)
        year  = int(ts.year)
        month = int(ts.month)
        day   = int(ts.day)

    calendar_keys.append(
        (year, month, day)
    )

calendar_keys = np.asarray(calendar_keys, dtype=int)


WU = []
WV = []
TAUX = []
TAUY = []
SLP = []

actual_hours = []


for t in local.time.values:

    target = pd.Timestamp(t)
    key = (
        target.year,
        target.month,
        target.day
    )

    ids = np.where(
        (calendar_keys[:,0] == key[0])
        &
        (calendar_keys[:,1] == key[1])
        &
        (calendar_keys[:,2] == key[2])
    )[0]

    print()
    print(
        target.date(),
        "GEOS hourly fields:",
        len(ids)
    )

    if len(ids) == 0:
        raise RuntimeError(
            f"No GEOS-FP data for {target.date()}"
        )

    vars_to_read = ["u10m","v10m"]

    if have_slp:
        vars_to_read.append("slp")

    # OPeNDAP only transfers this time/space subset.
    sub = (
        geos[vars_to_read]
        .isel(time=ids)
        .sel(
            lon=slice(float(lon_target.min()),
                      float(lon_target.max())),
            lat=slice(float(lat_target.min()),
                      float(lat_target.max()))
        )
        .load()
    )

    U = sub.u10m
    V = sub.v10m

    # Daily mean wind.
    Um = U.mean("time")
    Vm = V.mean("time")

    # IMPORTANT:
    # mean wind stress != stress of mean wind.
    # Compute hourly stress first.
    speed = np.sqrt(U**2 + V**2)

    tx = (
        RHO_AIR
        * CD_AIR
        * speed
        * U
    ).mean("time")

    ty = (
        RHO_AIR
        * CD_AIR
        * speed
        * V
    ).mean("time")

    # Interpolate onto our common SSH/OSCAR grid.
    Um = Um.interp(
        lon=lon_target,
        lat=lat_target
    )

    Vm = Vm.interp(
        lon=lon_target,
        lat=lat_target
    )

    tx = tx.interp(
        lon=lon_target,
        lat=lat_target
    )

    ty = ty.interp(
        lon=lon_target,
        lat=lat_target
    )

    WU.append(Um.values)
    WV.append(Vm.values)
    TAUX.append(tx.values)
    TAUY.append(ty.values)

    if have_slp:
        pp = (
            sub.slp
            .mean("time")
            .interp(
                lon=lon_target,
                lat=lat_target
            )
        )

        SLP.append(pp.values)

    actual_hours.append(len(ids))


# ------------------------------------------------------------
# Append fields
# ------------------------------------------------------------

out = local.copy()

out["wind_u10m"] = (
    ("time","lat","lon"),
    np.asarray(WU)
)

out["wind_v10m"] = (
    ("time","lat","lon"),
    np.asarray(WV)
)

out["wind_stress_x"] = (
    ("time","lat","lon"),
    np.asarray(TAUX)
)

out["wind_stress_y"] = (
    ("time","lat","lon"),
    np.asarray(TAUY)
)

if have_slp:
    out["sea_level_pressure"] = (
        ("time","lat","lon"),
        np.asarray(SLP)
    )

out["wind_hours_used"] = (
    ("time",),
    np.asarray(actual_hours)
)


out.wind_u10m.attrs["units"] = "m s-1"
out.wind_v10m.attrs["units"] = "m s-1"

out.wind_stress_x.attrs["units"] = "N m-2"
out.wind_stress_y.attrs["units"] = "N m-2"

out.attrs["wind_source"] = (
    "NASA GEOS-FP hourly analysis; daily means sampled "
    "on matched SSH/OSCAR dates"
)

out.attrs["wind_drag_coefficient"] = CD_AIR
out.attrs["air_density_kg_m3"] = RHO_AIR

out.to_netcdf(OUTFILE)

print()
print("SAVED:", OUTFILE)


# ------------------------------------------------------------
# Plot latest matched snapshot: SSH + wind vectors
# ------------------------------------------------------------

it = -1

fig, ax = plt.subplots(
    figsize=(14,6)
)

eta = out.ssha.isel(time=it)

vmax = np.nanpercentile(
    np.abs(eta.values),
    98
)

p = ax.pcolormesh(
    out.lon,
    out.lat,
    eta,
    shading="auto",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax
)

fig.colorbar(
    p,
    ax=ax,
    label="SSH anomaly [m]"
)

LON,LAT = np.meshgrid(
    out.lon.values,
    out.lat.values
)

skipx = 5
skipy = 2

q = ax.quiver(
    LON[::skipy,::skipx],
    LAT[::skipy,::skipx],
    out.wind_u10m.isel(time=it).values[::skipy,::skipx],
    out.wind_v10m.isel(time=it).values[::skipy,::skipx],
    color="black",
    width=0.0017
)

ax.quiverkey(
    q,
    0.82,
    1.035,
    10,
    "10 m/s wind",
    coordinates="axes",
    labelpos="E"
)

ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=0.8
)

date = pd.Timestamp(
    out.time.values[it]
).date()

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title(
    f"NASA SSH + GEOS-FP 10-m wind — {date}"
)

plt.tight_layout()

plt.savefig(
    "figures/Pacific_SSH_GEOS_wind.png",
    dpi=240
)

print(
    "SAVED: figures/Pacific_SSH_GEOS_wind.png"
)
