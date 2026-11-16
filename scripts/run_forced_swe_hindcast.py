import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

OUTDIR = "data_processed/forced_swe_hindcast"
FIGDIR = "figures/forced_swe_hindcast"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ---------------- Physical constants ----------------

G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0


# ---------------- Model parameters ----------------

# Effective first-baroclinic gravity-wave speed.
C = 2.45                  # m/s

# Active upper-layer depth used for wind-stress acceleration.
H_ACTIVE = 100.0          # m

# Corresponding reduced gravity.
G_PRIME = C*C / H_ACTIVE

# Equivalent depth appearing in SSH continuity equation.
H_EQ = C*C / G

# Linear damping.
DAMPING_DAYS = 20.0
R_DAMP = 1.0 / (DAMPING_DAYS * 86400.0)

# Weak horizontal viscosity to suppress grid-scale noise.
NU = 1000.0               # m^2/s

# Numerical step.
DT = 1800.0               # seconds = 30 min

# Forecast interval.
HORIZON_DAYS = 14.0

# We want the newest starting point having about 14 days
# of later observations.  With weekly maps, -3 normally means
# start two maps before the end.
START_INDEX = -3


# ---------------- Spatial experiment ----------------
#
# Stay away from the American land boundary for this first test.
#

LON0 = -170.0
LON1 = -100.0

LAT0 = -10.0
LAT1 =  10.0


# ---------------- Boundary sponge ----------------

SPONGE_CELLS = 5

# Maximum edge damping time ~2 days.
SPONGE_MAX = 1.0 / (2.0 * 86400.0)


# ============================================================
# LOAD DATA
# ============================================================

ds = xr.open_dataset(FILE)

required = [
    "ssha",
    "u",
    "v",
]

for name in required:
    if name not in ds:
        raise RuntimeError(
            f"Input file missing {name}"
        )


# Wind stress should already have been calculated.
if (
    "wind_stress_x" not in ds
    or
    "wind_stress_y" not in ds
):
    raise RuntimeError(
        "Input cube does not contain wind_stress_x/y. "
        "Run scripts/add_geos_wind_to_weekly_cube.py first."
    )


# Crop.
ds = ds.sel(
    lon=slice(LON0, LON1),
    lat=slice(LAT0, LAT1)
)

# Ensure standard ordering.
ds = ds.transpose(
    "time",
    "lat",
    "lon"
)

dates = pd.DatetimeIndex(
    ds.time.values
)

if len(dates) < 4:
    raise RuntimeError(
        "Need at least four matched observation times."
    )


i0 = START_INDEX

if i0 < 0:
    i0 = len(dates) + i0

if i0 < 0 or i0 >= len(dates):
    raise RuntimeError(
        "Bad START_INDEX"
    )


t0 = dates[i0]
t_end_requested = (
    t0
    + pd.Timedelta(days=HORIZON_DAYS)
)


# Available held-out observations.
target_ids = [
    j
    for j in range(i0 + 1, len(dates))
    if dates[j] <= t_end_requested
]


if not target_ids:
    raise RuntimeError(
        "No later observations inside forecast horizon."
    )


print()
print("============================================")
print("FORCED 1.5-LAYER SHALLOW-WATER HINDCAST")
print("============================================")
print()
print("Initial date:", t0)
print("Held-out target dates:")

for j in target_ids:
    print(
        " ",
        dates[j],
        " lead =",
        (dates[j] - t0).total_seconds()/86400.0,
        "days"
    )


print()
print("MODEL PARAMETERS")
print("----------------")
print("c =", C, "m/s")
print("active layer H =", H_ACTIVE, "m")
print("reduced gravity g' =", G_PRIME, "m/s^2")
print("equivalent depth c^2/g =", H_EQ, "m")
print("damping timescale =", DAMPING_DAYS, "days")
print("viscosity =", NU, "m^2/s")
print("time step =", DT, "s")


# ============================================================
# GRID GEOMETRY
# ============================================================

lon = ds.lon.values.astype(float)
lat = ds.lat.values.astype(float)

ny = len(lat)
nx = len(lon)


dlon_rad = np.deg2rad(
    float(np.median(np.diff(lon)))
)

dlat_rad = np.deg2rad(
    float(np.median(np.diff(lat)))
)

dy = R_EARTH * dlat_rad

dx = (
    R_EARTH
    * np.cos(np.deg2rad(lat))
    * dlon_rad
)

# Coriolis.
f = (
    2.0
    * OMEGA
    * np.sin(np.deg2rad(lat))
)

F = f[:, None]


print()
print("GRID")
print("----")
print("nx =", nx, "ny =", ny)
print(
    "dx at equator ~",
    R_EARTH*dlon_rad/1000,
    "km"
)
print("dy =", dy/1000, "km")


# ============================================================
# NUMERICAL DERIVATIVES
# ============================================================

def ddx(A):
    """
    East-west derivative on lat x lon field.
    """
    out = np.empty_like(A)

    out[:, 1:-1] = (
        A[:, 2:] - A[:, :-2]
    ) / (
        2.0 * dx[:, None]
    )

    out[:, 0] = (
        A[:, 1] - A[:, 0]
    ) / dx

    out[:, -1] = (
        A[:, -1] - A[:, -2]
    ) / dx

    return out


def ddy(A):
    """
    North-south derivative.
    """
    out = np.empty_like(A)

    out[1:-1, :] = (
        A[2:, :] - A[:-2, :]
    ) / (
        2.0 * dy
    )

    out[0, :] = (
        A[1, :] - A[0, :]
    ) / dy

    out[-1, :] = (
        A[-1, :] - A[-2, :]
    ) / dy

    return out


def laplacian(A):
    """
    Approximate physical horizontal Laplacian.
    """

    xx = np.empty_like(A)

    xx[:, 1:-1] = (
        A[:, 2:]
        - 2*A[:, 1:-1]
        + A[:, :-2]
    ) / (
        dx[:, None]**2
    )

    xx[:, 0] = xx[:, 1]
    xx[:, -1] = xx[:, -2]


    yy = np.empty_like(A)

    yy[1:-1, :] = (
        A[2:, :]
        - 2*A[1:-1, :]
        + A[:-2, :]
    ) / (
        dy**2
    )

    yy[0, :] = yy[1, :]
    yy[-1, :] = yy[-2, :]

    return xx + yy


# ============================================================
# FILL SMALL MISSING VALUES
# ============================================================

def fill2d(A):

    A = np.asarray(
        A,
        dtype=float
    ).copy()

    # Longitude interpolation.
    xx = np.arange(nx)

    for j in range(ny):

        row = A[j, :]
        good = np.isfinite(row)

        if good.sum() >= 2:
            A[j, :] = np.interp(
                xx,
                xx[good],
                row[good]
            )


    # Latitude interpolation.
    yy = np.arange(ny)

    for i in range(nx):

        col = A[:, i]
        good = np.isfinite(col)

        if good.sum() >= 2:
            A[:, i] = np.interp(
                yy,
                yy[good],
                col[good]
            )


    # Any isolated leftovers.
    if np.isnan(A).any():

        mean = np.nanmean(A)

        if not np.isfinite(mean):
            mean = 0.0

        A = np.nan_to_num(
            A,
            nan=mean
        )

    return A


# ============================================================
# BACKGROUND STATE
#
# IMPORTANT:
# Only observations available at or before t0 are used here.
# No future ocean observations leak into the forecast.
# ============================================================

past = ds.isel(
    time=slice(0, i0+1)
)


eta_bg = fill2d(
    past.ssha.mean("time").values
)

u_bg = fill2d(
    past.u.mean("time").values
)

v_bg = fill2d(
    past.v.mean("time").values
)


# Initial total observations.
eta_obs0 = fill2d(
    ds.ssha.isel(time=i0).values
)

u_obs0 = fill2d(
    ds.u.isel(time=i0).values
)

v_obs0 = fill2d(
    ds.v.isel(time=i0).values
)


# Model evolves perturbations around the pre-t0 background.
eta0 = eta_obs0 - eta_bg
u0 = u_obs0 - u_bg
v0 = v_obs0 - v_bg


# ============================================================
# WIND-STRESS ANOMALIES
#
# We model perturbations around an existing background state,
# so use anomalous wind stress rather than repeatedly applying
# the climatological trade winds to an unbalanced background.
# ============================================================

tx_all = np.asarray(
    ds.wind_stress_x.values,
    dtype=float
)

ty_all = np.asarray(
    ds.wind_stress_y.values,
    dtype=float
)


# Fill each forcing map.
for k in range(len(dates)):
    tx_all[k] = fill2d(tx_all[k])
    ty_all[k] = fill2d(ty_all[k])


tx_bg = np.mean(
    tx_all[:i0+1],
    axis=0
)

ty_bg = np.mean(
    ty_all[:i0+1],
    axis=0
)

tx_all = tx_all - tx_bg
ty_all = ty_all - ty_bg


# Observation times relative to t0, seconds.
forcing_times = np.array([
    (x - t0).total_seconds()
    for x in dates
])


def interp_forcing(A, tsec):
    """
    Linear interpolation between available GEOS daily-mean
    forcing maps associated with our weekly observation dates.
    """

    if tsec <= forcing_times[0]:
        return A[0]

    if tsec >= forcing_times[-1]:
        return A[-1]

    j = np.searchsorted(
        forcing_times,
        tsec
    )

    j0 = j - 1
    j1 = j

    a = (
        (tsec - forcing_times[j0])
        /
        (forcing_times[j1] - forcing_times[j0])
    )

    return (
        (1-a)*A[j0]
        +
        a*A[j1]
    )


# ============================================================
# OPEN-BOUNDARY SPONGE
# ============================================================

jj, ii = np.meshgrid(
    np.arange(nx),
    np.arange(ny)
)

distance_cells = np.minimum.reduce([
    ii,
    ny-1-ii,
    jj,
    nx-1-jj,
])


sponge = np.zeros(
    (ny, nx),
    dtype=float
)

m = distance_cells < SPONGE_CELLS

sponge[m] = (
    SPONGE_MAX
    *
    (
        (SPONGE_CELLS-distance_cells[m])
        /
        SPONGE_CELLS
    )**2
)


# ============================================================
# PDE RIGHT-HAND SIDE
# ============================================================

def rhs(
    u,
    v,
    eta,
    tsec,
    use_wind=True
):

    eta_x = ddx(eta)
    eta_y = ddy(eta)

    div = (
        ddx(u)
        +
        ddy(v)
    )


    # ---------------- Momentum ----------------

    du = (
        F*v
        -
        G*eta_x
        -
        R_DAMP*u
        +
        NU*laplacian(u)
    )

    dv = (
        -F*u
        -
        G*eta_y
        -
        R_DAMP*v
        +
        NU*laplacian(v)
    )


    # ---------------- Wind forcing ----------------

    if use_wind:

        tx = interp_forcing(
            tx_all,
            tsec
        )

        ty = interp_forcing(
            ty_all,
            tsec
        )

        du += (
            tx
            /
            (RHO_WATER*H_ACTIVE)
        )

        dv += (
            ty
            /
            (RHO_WATER*H_ACTIVE)
        )


    # ---------------- Continuity ----------------
    #
    # eta_t + (c^2/g) div(u) = 0
    #

    deta = (
        -H_EQ*div
    )


    # ---------------- Boundary sponge ----------------

    du -= sponge*u
    dv -= sponge*v
    deta -= sponge*eta


    return du, dv, deta


# ============================================================
# RK4 STEP
# ============================================================

def rk4(
    u,
    v,
    eta,
    tsec,
    dtsec,
    use_wind
):

    k1 = rhs(
        u,
        v,
        eta,
        tsec,
        use_wind
    )

    k2 = rhs(
        u + 0.5*dtsec*k1[0],
        v + 0.5*dtsec*k1[1],
        eta + 0.5*dtsec*k1[2],
        tsec + 0.5*dtsec,
        use_wind
    )

    k3 = rhs(
        u + 0.5*dtsec*k2[0],
        v + 0.5*dtsec*k2[1],
        eta + 0.5*dtsec*k2[2],
        tsec + 0.5*dtsec,
        use_wind
    )

    k4 = rhs(
        u + dtsec*k3[0],
        v + dtsec*k3[1],
        eta + dtsec*k3[2],
        tsec + dtsec,
        use_wind
    )

    un = u + (
        dtsec/6.0
        *
        (
            k1[0]
            + 2*k2[0]
            + 2*k3[0]
            + k4[0]
        )
    )

    vn = v + (
        dtsec/6.0
        *
        (
            k1[1]
            + 2*k2[1]
            + 2*k3[1]
            + k4[1]
        )
    )

    en = eta + (
        dtsec/6.0
        *
        (
            k1[2]
            + 2*k2[2]
            + 2*k3[2]
            + k4[2]
        )
    )

    return un, vn, en


# ============================================================
# INTEGRATE TO EACH HELD-OUT OBSERVATION
# ============================================================

def run_model(use_wind):

    u = u0.copy()
    v = v0.copy()
    eta = eta0.copy()

    current_time = 0.0

    predictions = {}


    for target_id in target_ids:

        target_seconds = (
            dates[target_id] - t0
        ).total_seconds()


        while current_time < target_seconds - 1e-6:

            this_dt = min(
                DT,
                target_seconds-current_time
            )

            u, v, eta = rk4(
                u,
                v,
                eta,
                current_time,
                this_dt,
                use_wind
            )

            current_time += this_dt


            # Blow-up protection.
            max_eta = np.nanmax(
                np.abs(eta)
            )

            max_speed = np.nanmax(
                np.sqrt(u*u+v*v)
            )

            if (
                not np.isfinite(max_eta)
                or
                not np.isfinite(max_speed)
                or
                max_eta > 5.0
                or
                max_speed > 8.0
            ):
                raise RuntimeError(
                    f"Model unstable at "
                    f"{current_time/86400:.2f} days: "
                    f"max eta={max_eta}, "
                    f"max speed={max_speed}"
                )


        predictions[target_id] = (
            eta.copy(),
            u.copy(),
            v.copy()
        )


    return predictions


print()
print("Running FORCED model...")
forced = run_model(
    use_wind=True
)

print("Running UNFORCED model...")
unforced = run_model(
    use_wind=False
)


# ============================================================
# METRICS
# ============================================================

def rmse(a, b):
    return float(
        np.sqrt(
            np.mean(
                (a-b)**2
            )
        )
    )


def corr(a, b):

    a = a.ravel()
    b = b.ravel()

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if good.sum() < 3:
        return np.nan

    return float(
        np.corrcoef(
            a[good],
            b[good]
        )[0,1]
    )


rows = []


# ============================================================
# PLOTS + SCORES
# ============================================================

LON, LAT = np.meshgrid(
    lon,
    lat
)


for target_id in target_ids:

    target_date = dates[target_id]

    lead_days = (
        target_date - t0
    ).total_seconds()/86400.0


    # --------------------------------------------------------
    # Observed target
    # --------------------------------------------------------

    eta_obs = fill2d(
        ds.ssha.isel(
            time=target_id
        ).values
    )

    u_obs = fill2d(
        ds.u.isel(
            time=target_id
        ).values
    )

    v_obs = fill2d(
        ds.v.isel(
            time=target_id
        ).values
    )


    # --------------------------------------------------------
    # Forced forecast total state
    # --------------------------------------------------------

    ep, up, vp = forced[
        target_id
    ]

    eta_forced = eta_bg + ep
    u_forced = u_bg + up
    v_forced = v_bg + vp


    # --------------------------------------------------------
    # Unforced forecast
    # --------------------------------------------------------

    eu, uu, vu = unforced[
        target_id
    ]

    eta_unforced = eta_bg + eu
    u_unforced = u_bg + uu
    v_unforced = v_bg + vu


    # --------------------------------------------------------
    # Persistence forecast
    # --------------------------------------------------------

    eta_persist = eta_obs0
    u_persist = u_obs0
    v_persist = v_obs0


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    row = {
        "target_date": str(
            target_date.date()
        ),
        "lead_days": lead_days,

        "ssh_rmse_forced_m":
            rmse(
                eta_forced,
                eta_obs
            ),

        "ssh_rmse_unforced_m":
            rmse(
                eta_unforced,
                eta_obs
            ),

        "ssh_rmse_persistence_m":
            rmse(
                eta_persist,
                eta_obs
            ),

        "ssh_corr_forced":
            corr(
                eta_forced,
                eta_obs
            ),

        "ssh_corr_unforced":
            corr(
                eta_unforced,
                eta_obs
            ),

        "ssh_corr_persistence":
            corr(
                eta_persist,
                eta_obs
            ),

        "u_rmse_forced_m_s":
            rmse(
                u_forced,
                u_obs
            ),

        "v_rmse_forced_m_s":
            rmse(
                v_forced,
                v_obs
            ),

        "vector_rmse_forced_m_s":
            float(
                np.sqrt(
                    np.mean(
                        (u_forced-u_obs)**2
                        +
                        (v_forced-v_obs)**2
                    )
                )
            ),

        "vector_rmse_persistence_m_s":
            float(
                np.sqrt(
                    np.mean(
                        (u_persist-u_obs)**2
                        +
                        (v_persist-v_obs)**2
                    )
                )
            ),
    }

    rows.append(row)


    # --------------------------------------------------------
    # Plot observed / forecast / error
    # --------------------------------------------------------

    vmax = max(
        np.nanpercentile(
            np.abs(eta_obs),
            98
        ),
        np.nanpercentile(
            np.abs(eta_forced),
            98
        )
    )

    err = (
        eta_forced
        -
        eta_obs
    )

    emax = np.nanpercentile(
        np.abs(err),
        98
    )


    fig, ax = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )


    # ---------- observed ----------

    p0 = ax[0].pcolormesh(
        lon,
        lat,
        eta_obs,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    skip = 8

    q0 = ax[0].quiver(
        LON[::skip, ::skip],
        LAT[::skip, ::skip],
        u_obs[::skip, ::skip],
        v_obs[::skip, ::skip],
        color="black",
        scale=10
    )

    ax[0].set_title(
        "Held-out observation"
    )


    # ---------- predicted ----------

    p1 = ax[1].pcolormesh(
        lon,
        lat,
        eta_forced,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    q1 = ax[1].quiver(
        LON[::skip, ::skip],
        LAT[::skip, ::skip],
        u_forced[::skip, ::skip],
        v_forced[::skip, ::skip],
        color="black",
        scale=10
    )

    ax[1].set_title(
        "Forced shallow-water forecast"
    )


    # ---------- error ----------

    p2 = ax[2].pcolormesh(
        lon,
        lat,
        err,
        cmap="RdBu_r",
        vmin=-emax,
        vmax=emax,
        shading="auto"
    )

    ax[2].set_title(
        "Forecast minus observation"
    )


    for a in ax:

        a.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.7
        )

        a.set_xlabel(
            "Longitude"
        )

        a.set_ylabel(
            "Latitude"
        )


    fig.colorbar(
        p1,
        ax=ax[:2],
        shrink=0.85,
        label="SSH anomaly [m]"
    )

    fig.colorbar(
        p2,
        ax=ax[2],
        shrink=0.85,
        label="SSH forecast error [m]"
    )


    fig.suptitle(
        f"{lead_days:.0f}-day blind hindcast "
        f"from {t0.date()} to {target_date.date()}\n"
        f"SSH RMSE: model={row['ssh_rmse_forced_m']:.3f} m, "
        f"persistence={row['ssh_rmse_persistence_m']:.3f} m; "
        f"corr={row['ssh_corr_forced']:.3f}"
    )


    # Avoid tight_layout conflict with shared colorbar.
    fig.subplots_adjust(
        left=0.055,
        right=0.92,
        bottom=0.10,
        top=0.84,
        wspace=0.25
    )


    filename = (
        f"{FIGDIR}/hindcast_"
        f"{int(round(lead_days)):02d}day.png"
    )

    plt.savefig(
        filename,
        dpi=220
    )

    plt.close()

    print(
        "SAVED:",
        filename
    )


# ============================================================
# SAVE METRICS
# ============================================================

scores = pd.DataFrame(rows)

score_file = (
    OUTDIR
    + "/hindcast_scores.csv"
)

scores.to_csv(
    score_file,
    index=False
)


print()
print("============================================")
print("HINDCAST SCORES")
print("============================================")
print()
print(
    scores.to_string(index=False)
)


print()
print("SAVED:", score_file)


# ============================================================
# SAVE LAST FORECAST FIELD
# ============================================================

last_id = target_ids[-1]

ep, up, vp = forced[last_id]

result = xr.Dataset(
    data_vars={
        "ssha_predicted": (
            ("lat","lon"),
            eta_bg + ep
        ),
        "u_predicted": (
            ("lat","lon"),
            u_bg + up
        ),
        "v_predicted": (
            ("lat","lon"),
            v_bg + vp
        ),
        "ssha_observed": (
            ("lat","lon"),
            fill2d(
                ds.ssha.isel(
                    time=last_id
                ).values
            )
        ),
        "u_observed": (
            ("lat","lon"),
            fill2d(
                ds.u.isel(
                    time=last_id
                ).values
            )
        ),
        "v_observed": (
            ("lat","lon"),
            fill2d(
                ds.v.isel(
                    time=last_id
                ).values
            )
        ),
    },
    coords={
        "lat": lat,
        "lon": lon
    }
)

result.attrs[
    "initial_date"
] = str(t0.date())

result.attrs[
    "target_date"
] = str(dates[last_id].date())

result.attrs[
    "gravity_wave_speed_m_s"
] = C

result.attrs[
    "active_layer_depth_m"
] = H_ACTIVE

result.attrs[
    "reduced_gravity_m_s2"
] = G_PRIME

result.attrs[
    "damping_days"
] = DAMPING_DAYS

result.to_netcdf(
    OUTDIR
    + "/last_hindcast_field.nc"
)

print(
    "SAVED:",
    OUTDIR
    + "/last_hindcast_field.nc"
)
