import os
import json
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# FILES / DOMAIN
# ============================================================

FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

OUTDIR = "data_processed/balanced_swe"
FIGDIR = "figures/balanced_swe"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LON0 = -170.0
LON1 = -100.0
LAT0 =  -10.0
LAT1 =   10.0


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0


# ============================================================
# NUMERICAL SETTINGS
# ============================================================

# Two hours is safe at ~55 km resolution and c <= 3 m/s.
DT = 7200.0

NU = 1000.0                    # m^2/s

SPONGE_CELLS = 5
SPONGE_MAX = 1.0/(2.0*86400.0)

# Assimilate one week before forecast start.
ASSIM_WEEKS = 1

# Relaxation timescale during assimilation.
NUDGE_ETA_DAYS = 1.5
NUDGE_UV_DAYS  = 1.5

TAU_ETA = NUDGE_ETA_DAYS * 86400.0
TAU_UV  = NUDGE_UV_DAYS  * 86400.0


# ============================================================
# PARAMETER GRID
# ============================================================

# c controls equivalent depth c^2/g.
C_VALUES = [
    1.50,
    2.00,
    2.45,
    3.00,
]

# Effective depth over which wind stress accelerates water.
HWIND_VALUES = [
    30.0,
    60.0,
    100.0,
]

# Linear damping timescale.
DAMP_VALUES = [
    10.0,
    20.0,
    40.0,
    80.0,
]


# ============================================================
# LOAD DATA
# ============================================================

ds = xr.open_dataset(FILE)

for name in [
    "ssha",
    "u",
    "v",
    "wind_stress_x",
    "wind_stress_y",
]:
    if name not in ds:
        raise RuntimeError(
            f"Input file missing {name}"
        )


ds = ds.sel(
    lon=slice(LON0, LON1),
    lat=slice(LAT0, LAT1)
)

dates = pd.DatetimeIndex(
    ds.time.values
)

nt = len(dates)

if nt < 8:
    raise RuntimeError(
        "Need more weekly observations."
    )


lon = ds.lon.values.astype(float)
lat = ds.lat.values.astype(float)

nx = len(lon)
ny = len(lat)


print()
print("==========================================")
print("BALANCED FORCED SHALLOW-WATER EXPERIMENT")
print("==========================================")
print()
print("Number of observation dates:", nt)
print("Range:", dates[0], "to", dates[-1])
print("Grid:", nx, "x", ny)


# ============================================================
# EXTRACT ARRAYS WITH EXPLICIT DIMENSION ORDER
# ============================================================

ETA_OBS = np.asarray(
    ds["ssha"]
    .transpose("time", "lat", "lon")
    .values,
    dtype=float
)

U_OBS = np.asarray(
    ds["u"]
    .transpose("time", "lat", "lon")
    .values,
    dtype=float
)

V_OBS = np.asarray(
    ds["v"]
    .transpose("time", "lat", "lon")
    .values,
    dtype=float
)

TX = np.asarray(
    ds["wind_stress_x"]
    .transpose("time", "lat", "lon")
    .values,
    dtype=float
)

TY = np.asarray(
    ds["wind_stress_y"]
    .transpose("time", "lat", "lon")
    .values,
    dtype=float
)


# ============================================================
# GRID GEOMETRY
# ============================================================

dlon = np.deg2rad(
    float(np.median(np.diff(lon)))
)

dlat = np.deg2rad(
    float(np.median(np.diff(lat)))
)

dy = R_EARTH*dlat

dx = (
    R_EARTH
    * np.cos(np.deg2rad(lat))
    * dlon
)

f = (
    2.0
    * OMEGA
    * np.sin(np.deg2rad(lat))
)

F = f[:, None]


print(
    "dx equator:",
    R_EARTH*dlon/1000,
    "km"
)

print(
    "dy:",
    dy/1000,
    "km"
)


# ============================================================
# OBSERVATION TIMES
# ============================================================

OBS_SEC = np.array([
    (d - dates[0]).total_seconds()
    for d in dates
])


# ============================================================
# FILL NaNs IN EACH 2-D MAP
# ============================================================

def fill2d(A):

    B = np.asarray(
        A,
        dtype=float
    ).copy()

    xx = np.arange(nx)

    for j in range(ny):

        row = B[j]
        good = np.isfinite(row)

        if good.sum() >= 2:
            B[j] = np.interp(
                xx,
                xx[good],
                row[good]
            )


    yy = np.arange(ny)

    for i in range(nx):

        col = B[:, i]
        good = np.isfinite(col)

        if good.sum() >= 2:
            B[:, i] = np.interp(
                yy,
                yy[good],
                col[good]
            )


    if np.isnan(B).any():

        m = np.nanmean(B)

        if not np.isfinite(m):
            m = 0.0

        B = np.nan_to_num(
            B,
            nan=m
        )

    return B


def fill_cube(A):

    B = A.copy()

    for k in range(B.shape[0]):
        B[k] = fill2d(B[k])

    return B


print()
print("Filling small spatial gaps...")

ETA_OBS = fill_cube(ETA_OBS)
U_OBS   = fill_cube(U_OBS)
V_OBS   = fill_cube(V_OBS)
TX      = fill_cube(TX)
TY      = fill_cube(TY)


# ============================================================
# DERIVATIVES
# ============================================================

def ddx(A):

    D = np.empty_like(A)

    D[:, 1:-1] = (
        A[:, 2:] - A[:, :-2]
    ) / (
        2.0*dx[:, None]
    )

    D[:, 0] = (
        A[:, 1] - A[:, 0]
    ) / dx

    D[:, -1] = (
        A[:, -1] - A[:, -2]
    ) / dx

    return D


def ddy(A):

    D = np.empty_like(A)

    D[1:-1] = (
        A[2:] - A[:-2]
    ) / (
        2.0*dy
    )

    D[0] = (
        A[1] - A[0]
    ) / dy

    D[-1] = (
        A[-1] - A[-2]
    ) / dy

    return D


def laplacian(A):

    Dxx = np.empty_like(A)

    Dxx[:, 1:-1] = (
        A[:, 2:]
        - 2.0*A[:, 1:-1]
        + A[:, :-2]
    ) / (
        dx[:, None]**2
    )

    Dxx[:, 0] = Dxx[:, 1]
    Dxx[:, -1] = Dxx[:, -2]


    Dyy = np.empty_like(A)

    Dyy[1:-1] = (
        A[2:]
        - 2.0*A[1:-1]
        + A[:-2]
    ) / (
        dy**2
    )

    Dyy[0] = Dyy[1]
    Dyy[-1] = Dyy[-2]

    return Dxx + Dyy


# ============================================================
# OPEN-BOUNDARY SPONGE
# ============================================================

jj, ii = np.meshgrid(
    np.arange(nx),
    np.arange(ny)
)

dist = np.minimum.reduce([
    ii,
    ny-1-ii,
    jj,
    nx-1-jj,
])

SPONGE = np.zeros(
    (ny, nx),
    dtype=float
)

mask = dist < SPONGE_CELLS

SPONGE[mask] = (
    SPONGE_MAX
    *
    (
        (SPONGE_CELLS-dist[mask])
        /
        SPONGE_CELLS
    )**2
)


# ============================================================
# TIME INTERPOLATION
# ============================================================

def interp_series(A, tsec):

    if tsec <= OBS_SEC[0]:
        return A[0]

    if tsec >= OBS_SEC[-1]:
        return A[-1]

    j = np.searchsorted(
        OBS_SEC,
        tsec
    )

    j0 = j - 1
    j1 = j

    alpha = (
        (tsec-OBS_SEC[j0])
        /
        (OBS_SEC[j1]-OBS_SEC[j0])
    )

    return (
        (1.0-alpha)*A[j0]
        +
        alpha*A[j1]
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


def vector_rmse(
    u1,
    v1,
    u2,
    v2
):

    return float(
        np.sqrt(
            np.mean(
                (u1-u2)**2
                +
                (v1-v2)**2
            )
        )
    )


def corr(a, b):

    x = np.asarray(a).ravel()
    y = np.asarray(b).ravel()

    good = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if good.sum() < 3:
        return np.nan

    return float(
        np.corrcoef(
            x[good],
            y[good]
        )[0,1]
    )


# ============================================================
# BACKGROUND FOR ONE FORECAST WINDOW
# ============================================================

def background_for_window(
    assimil_start_index
):

    #
    # CRITICAL:
    # only observations available at or BEFORE assimilation
    # start enter the slowly varying background.
    #

    ids = np.arange(
        0,
        assimil_start_index + 1
    )

    eta_bg = np.mean(
        ETA_OBS[ids],
        axis=0
    )

    u_bg = np.mean(
        U_OBS[ids],
        axis=0
    )

    v_bg = np.mean(
        V_OBS[ids],
        axis=0
    )

    tx_bg = np.mean(
        TX[ids],
        axis=0
    )

    ty_bg = np.mean(
        TY[ids],
        axis=0
    )

    return (
        eta_bg,
        u_bg,
        v_bg,
        tx_bg,
        ty_bg
    )


# ============================================================
# RHS
# ============================================================

def rhs(
    u,
    v,
    eta,
    tabs,
    C,
    Hwind,
    damping_days,
    tx_bg,
    ty_bg,
    nudging=False,
    eta_bg=None,
    u_bg=None,
    v_bg=None,
    use_wind=True,
):

    He = C*C/G

    r = (
        1.0
        /
        (damping_days*86400.0)
    )


    # --------------------------------------------------------
    # Basic shallow-water dynamics
    # --------------------------------------------------------

    du = (
        F*v
        -
        G*ddx(eta)
        -
        r*u
        +
        NU*laplacian(u)
    )

    dv = (
        -F*u
        -
        G*ddy(eta)
        -
        r*v
        +
        NU*laplacian(v)
    )

    deta = (
        -He
        *
        (
            ddx(u)
            +
            ddy(v)
        )
    )


    # --------------------------------------------------------
    # Anomalous wind forcing
    # --------------------------------------------------------

    if use_wind:

        tx = (
            interp_series(
                TX,
                tabs
            )
            -
            tx_bg
        )

        ty = (
            interp_series(
                TY,
                tabs
            )
            -
            ty_bg
        )

        du += (
            tx
            /
            (RHO_WATER*Hwind)
        )

        dv += (
            ty
            /
            (RHO_WATER*Hwind)
        )


    # --------------------------------------------------------
    # Assimilation nudging ONLY before forecast start
    # --------------------------------------------------------

    if nudging:

        eta_target = (
            interp_series(
                ETA_OBS,
                tabs
            )
            -
            eta_bg
        )

        u_target = (
            interp_series(
                U_OBS,
                tabs
            )
            -
            u_bg
        )

        v_target = (
            interp_series(
                V_OBS,
                tabs
            )
            -
            v_bg
        )


        du += (
            u_target-u
        ) / TAU_UV

        dv += (
            v_target-v
        ) / TAU_UV

        deta += (
            eta_target-eta
        ) / TAU_ETA


    # --------------------------------------------------------
    # Sponge
    # --------------------------------------------------------

    du -= SPONGE*u
    dv -= SPONGE*v
    deta -= SPONGE*eta


    return du, dv, deta


# ============================================================
# RK4
# ============================================================

def rk4(
    u,
    v,
    eta,
    tabs,
    dtsec,
    params,
    tx_bg,
    ty_bg,
    nudging,
    eta_bg,
    u_bg,
    v_bg,
    use_wind=True,
):

    C, Hwind, damping_days = params


    def R(
        U,
        V,
        E,
        T
    ):
        return rhs(
            U,
            V,
            E,
            T,
            C,
            Hwind,
            damping_days,
            tx_bg,
            ty_bg,
            nudging=nudging,
            eta_bg=eta_bg,
            u_bg=u_bg,
            v_bg=v_bg,
            use_wind=use_wind,
        )


    k1 = R(
        u,
        v,
        eta,
        tabs
    )

    k2 = R(
        u + 0.5*dtsec*k1[0],
        v + 0.5*dtsec*k1[1],
        eta + 0.5*dtsec*k1[2],
        tabs + 0.5*dtsec
    )

    k3 = R(
        u + 0.5*dtsec*k2[0],
        v + 0.5*dtsec*k2[1],
        eta + 0.5*dtsec*k2[2],
        tabs + 0.5*dtsec
    )

    k4 = R(
        u + dtsec*k3[0],
        v + dtsec*k3[1],
        eta + dtsec*k3[2],
        tabs + dtsec
    )


    un = (
        u
        +
        dtsec/6.0
        *
        (
            k1[0]
            + 2*k2[0]
            + 2*k3[0]
            + k4[0]
        )
    )

    vn = (
        v
        +
        dtsec/6.0
        *
        (
            k1[1]
            + 2*k2[1]
            + 2*k3[1]
            + k4[1]
        )
    )

    en = (
        eta
        +
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
# INTEGRATE FROM tA TO tB
# ============================================================

def integrate(
    u,
    v,
    eta,
    tA,
    tB,
    params,
    tx_bg,
    ty_bg,
    nudging,
    eta_bg,
    u_bg,
    v_bg,
    use_wind=True,
):

    t = float(tA)

    while t < tB - 1e-6:

        dti = min(
            DT,
            tB-t
        )

        u, v, eta = rk4(
            u,
            v,
            eta,
            t,
            dti,
            params,
            tx_bg,
            ty_bg,
            nudging,
            eta_bg,
            u_bg,
            v_bg,
            use_wind=use_wind,
        )

        t += dti


        maxeta = np.max(
            np.abs(eta)
        )

        maxuv = np.max(
            np.sqrt(
                u*u+v*v
            )
        )

        if (
            not np.isfinite(maxeta)
            or
            not np.isfinite(maxuv)
            or
            maxeta > 5.0
            or
            maxuv > 8.0
        ):
            raise RuntimeError(
                "Numerical instability: "
                f"eta={maxeta}, speed={maxuv}"
            )


    return u, v, eta


# ============================================================
# CREATE BALANCED INITIAL STATE
# ============================================================

def balanced_initial_state(
    forecast_start_index,
    params,
    use_wind=True,
):

    ia = (
        forecast_start_index
        -
        ASSIM_WEEKS
    )

    if ia < 0:
        raise RuntimeError(
            "Not enough pre-forecast observations."
        )


    (
        eta_bg,
        u_bg,
        v_bg,
        tx_bg,
        ty_bg
    ) = background_for_window(
        ia
    )


    # Start assimilation from actual observation at ia.
    eta = (
        ETA_OBS[ia]
        -
        eta_bg
    )

    u = (
        U_OBS[ia]
        -
        u_bg
    )

    v = (
        V_OBS[ia]
        -
        v_bg
    )


    tA = OBS_SEC[ia]
    t0 = OBS_SEC[
        forecast_start_index
    ]


    u, v, eta = integrate(
        u,
        v,
        eta,
        tA,
        t0,
        params,
        tx_bg,
        ty_bg,
        nudging=True,
        eta_bg=eta_bg,
        u_bg=u_bg,
        v_bg=v_bg,
        use_wind=use_wind,
    )


    return {
        "u": u,
        "v": v,
        "eta": eta,

        "eta_bg": eta_bg,
        "u_bg": u_bg,
        "v_bg": v_bg,

        "tx_bg": tx_bg,
        "ty_bg": ty_bg,

        "assim_start_index": ia,
    }


# ============================================================
# FORECAST ONE WINDOW
# ============================================================

def forecast_window(
    forecast_start_index,
    target_index,
    params,
    balanced=True,
    use_wind=True,
):

    if balanced:

        state = balanced_initial_state(
            forecast_start_index,
            params,
            use_wind=use_wind,
        )

        u = state["u"]
        v = state["v"]
        eta = state["eta"]

        eta_bg = state["eta_bg"]
        u_bg = state["u_bg"]
        v_bg = state["v_bg"]

        tx_bg = state["tx_bg"]
        ty_bg = state["ty_bg"]


    else:

        # Same background definition as balanced case,
        # but start directly from raw observation at t0.

        ia = (
            forecast_start_index
            -
            ASSIM_WEEKS
        )

        (
            eta_bg,
            u_bg,
            v_bg,
            tx_bg,
            ty_bg
        ) = background_for_window(
            ia
        )

        eta = (
            ETA_OBS[forecast_start_index]
            -
            eta_bg
        )

        u = (
            U_OBS[forecast_start_index]
            -
            u_bg
        )

        v = (
            V_OBS[forecast_start_index]
            -
            v_bg
        )


    t0 = OBS_SEC[
        forecast_start_index
    ]

    t1 = OBS_SEC[
        target_index
    ]


    u, v, eta = integrate(
        u,
        v,
        eta,
        t0,
        t1,
        params,
        tx_bg,
        ty_bg,
        nudging=False,
        eta_bg=eta_bg,
        u_bg=u_bg,
        v_bg=v_bg,
        use_wind=use_wind,
    )


    return (
        eta_bg + eta,
        u_bg + u,
        v_bg + v
    )


# ============================================================
# SCORE ONE FORECAST
# ============================================================

def score_window(
    start,
    target,
    params,
):

    eta_pred, u_pred, v_pred = (
        forecast_window(
            start,
            target,
            params,
            balanced=True,
            use_wind=True,
        )
    )


    eta_true = ETA_OBS[target]
    u_true = U_OBS[target]
    v_true = V_OBS[target]


    # Persistence from forecast start.
    eta_persist = ETA_OBS[start]
    u_persist = U_OBS[start]
    v_persist = V_OBS[start]


    re = rmse(
        eta_pred,
        eta_true
    )

    rp = rmse(
        eta_persist,
        eta_true
    )


    rv = vector_rmse(
        u_pred,
        v_pred,
        u_true,
        v_true
    )

    rvp = vector_rmse(
        u_persist,
        v_persist,
        u_true,
        v_true
    )


    eta_ratio = (
        re/rp
        if rp > 1e-10
        else 1e6
    )

    uv_ratio = (
        rv/rvp
        if rvp > 1e-10
        else 1e6
    )


    # Put SSH slightly higher priority.
    objective = (
        0.65*eta_ratio
        +
        0.35*uv_ratio
    )


    return {
        "objective": objective,

        "eta_ratio": eta_ratio,
        "uv_ratio": uv_ratio,

        "ssh_rmse": re,
        "ssh_persist": rp,

        "vector_rmse": rv,
        "vector_persist": rvp,

        "ssh_corr": corr(
            eta_pred,
            eta_true
        ),
    }


# ============================================================
# RESERVE FINAL OCTOBER-LIKE WINDOW
#
# Last three dates:
#
#   start = -3
#   +7 day target = -2
#   +14 day target = -1
#
# Nothing from those forecasts is used for parameter tuning.
# ============================================================

FINAL_START = nt - 3

if FINAL_START < 3:
    raise RuntimeError(
        "Not enough observations for train/test split."
    )


print()
print("FINAL HOLDOUT")
print("-------------")
print(
    "Forecast start:",
    dates[FINAL_START]
)
print(
    "7-day target:",
    dates[FINAL_START+1]
)
print(
    "14-day target:",
    dates[FINAL_START+2]
)


# ============================================================
# CHOOSE EARLIER 7-DAY TUNING WINDOWS
# ============================================================

possible = np.arange(
    ASSIM_WEEKS + 1,
    FINAL_START - 1
)


# Use up to five approximately equally spaced training windows.
NTRAIN = min(
    5,
    len(possible)
)

pick = np.linspace(
    0,
    len(possible)-1,
    NTRAIN
)

TRAIN_STARTS = np.unique(
    possible[
        np.round(pick).astype(int)
    ]
)


print()
print("PARAMETER-TUNING WINDOWS")
print("------------------------")

for s in TRAIN_STARTS:
    print(
        dates[s].date(),
        "->",
        dates[s+1].date()
    )


# ============================================================
# GRID SEARCH
# ============================================================

param_grid = list(
    itertools.product(
        C_VALUES,
        HWIND_VALUES,
        DAMP_VALUES,
    )
)


print()
print(
    "Number of parameter combinations:",
    len(param_grid)
)

print(
    "Each is scored on",
    len(TRAIN_STARTS),
    "earlier 7-day windows."
)


rows = []


for ip, params in enumerate(
    param_grid,
    start=1
):

    C, Hw, Tdamp = params

    scores = []

    failed = False


    for s in TRAIN_STARTS:

        try:

            sc = score_window(
                s,
                s+1,
                params
            )

            scores.append(sc)

        except Exception as e:

            print(
                "FAILED:",
                params,
                dates[s],
                repr(e)
            )

            failed = True
            break


    if failed or not scores:

        row = {
            "C_m_s": C,
            "Hwind_m": Hw,
            "damping_days": Tdamp,
            "objective": np.inf,
        }

    else:

        row = {
            "C_m_s": C,
            "Hwind_m": Hw,
            "damping_days": Tdamp,

            "objective":
                np.mean([
                    x["objective"]
                    for x in scores
                ]),

            "mean_eta_ratio":
                np.mean([
                    x["eta_ratio"]
                    for x in scores
                ]),

            "mean_uv_ratio":
                np.mean([
                    x["uv_ratio"]
                    for x in scores
                ]),

            "mean_ssh_rmse_m":
                np.mean([
                    x["ssh_rmse"]
                    for x in scores
                ]),

            "mean_ssh_corr":
                np.mean([
                    x["ssh_corr"]
                    for x in scores
                ]),
        }


    rows.append(row)


    print(
        f"[{ip:02d}/{len(param_grid)}]",
        f"c={C:.2f}",
        f"Hw={Hw:.0f}",
        f"Tdamp={Tdamp:.0f}",
        "objective=",
        row["objective"]
    )


tuning = pd.DataFrame(rows)

tuning = tuning.sort_values(
    "objective"
)

tuning.to_csv(
    OUTDIR
    + "/parameter_tuning.csv",
    index=False
)


print()
print("==========================================")
print("BEST PARAMETER SETS")
print("==========================================")
print(
    tuning.head(10).to_string(
        index=False
    )
)


best = tuning.iloc[0]

BEST = (
    float(best["C_m_s"]),
    float(best["Hwind_m"]),
    float(best["damping_days"]),
)


print()
print("SELECTED:")
print(" c =", BEST[0], "m/s")
print(" Hwind =", BEST[1], "m")
print(
    " damping =",
    BEST[2],
    "days"
)


with open(
    OUTDIR
    + "/best_parameters.json",
    "w"
) as f:

    json.dump(
        {
            "c_m_s": BEST[0],
            "Hwind_m": BEST[1],
            "damping_days": BEST[2],
            "nu_m2_s": NU,
            "assimilation_weeks": ASSIM_WEEKS,
            "nudge_eta_days": NUDGE_ETA_DAYS,
            "nudge_uv_days": NUDGE_UV_DAYS,
        },
        f,
        indent=2
    )


# ============================================================
# FINAL BLIND HOLDOUT TEST
# ============================================================

final_rows = []

predictions = {}


for lead_weeks in [
    1,
    2,
]:

    target = (
        FINAL_START
        +
        lead_weeks
    )

    lead_days = (
        dates[target]
        -
        dates[FINAL_START]
    ).total_seconds()/86400.0


    # --------------------------------------------------------
    # Balanced + wind
    # --------------------------------------------------------

    eb, ub, vb = forecast_window(
        FINAL_START,
        target,
        BEST,
        balanced=True,
        use_wind=True,
    )


    # --------------------------------------------------------
    # Balanced but NO wind
    # --------------------------------------------------------

    en, un, vn = forecast_window(
        FINAL_START,
        target,
        BEST,
        balanced=True,
        use_wind=False,
    )


    # --------------------------------------------------------
    # Raw initialization + wind
    # --------------------------------------------------------

    er, ur, vr = forecast_window(
        FINAL_START,
        target,
        BEST,
        balanced=False,
        use_wind=True,
    )


    et = ETA_OBS[target]
    ut = U_OBS[target]
    vt = V_OBS[target]


    ep = ETA_OBS[FINAL_START]
    up = U_OBS[FINAL_START]
    vp = V_OBS[FINAL_START]


    row = {
        "target_date":
            str(dates[target].date()),

        "lead_days":
            lead_days,

        "ssh_rmse_balanced_wind":
            rmse(eb, et),

        "ssh_rmse_balanced_nowind":
            rmse(en, et),

        "ssh_rmse_raw_wind":
            rmse(er, et),

        "ssh_rmse_persistence":
            rmse(ep, et),

        "ssh_corr_balanced_wind":
            corr(eb, et),

        "ssh_corr_persistence":
            corr(ep, et),

        "vector_rmse_balanced_wind":
            vector_rmse(
                ub,
                vb,
                ut,
                vt
            ),

        "vector_rmse_balanced_nowind":
            vector_rmse(
                un,
                vn,
                ut,
                vt
            ),

        "vector_rmse_raw_wind":
            vector_rmse(
                ur,
                vr,
                ut,
                vt
            ),

        "vector_rmse_persistence":
            vector_rmse(
                up,
                vp,
                ut,
                vt
            ),
    }


    final_rows.append(row)

    predictions[target] = {
        "balanced_wind":
            (eb, ub, vb),

        "balanced_nowind":
            (en, un, vn),

        "raw_wind":
            (er, ur, vr),
    }


final = pd.DataFrame(
    final_rows
)

final.to_csv(
    OUTDIR
    + "/final_holdout_scores.csv",
    index=False
)


print()
print("==========================================")
print("FINAL BLIND HOLDOUT SCORES")
print("==========================================")
print()
print(
    final.to_string(
        index=False
    )
)


# ============================================================
# INITIAL BALANCE DIAGNOSTIC
# ============================================================

state0 = balanced_initial_state(
    FINAL_START,
    BEST,
    use_wind=True,
)


eta_bal0 = (
    state0["eta_bg"]
    +
    state0["eta"]
)

u_bal0 = (
    state0["u_bg"]
    +
    state0["u"]
)

v_bal0 = (
    state0["v_bg"]
    +
    state0["v"]
)


eta_raw0 = ETA_OBS[
    FINAL_START
]

u_raw0 = U_OBS[
    FINAL_START
]

v_raw0 = V_OBS[
    FINAL_START
]


print()
print("BALANCED INITIAL-STATE CHANGES")
print("------------------------------")
print(
    "SSH RMS adjustment:",
    rmse(
        eta_bal0,
        eta_raw0
    ),
    "m"
)

print(
    "velocity RMS adjustment:",
    vector_rmse(
        u_bal0,
        v_bal0,
        u_raw0,
        v_raw0
    ),
    "m/s"
)


# ============================================================
# PLOT INITIAL OBSERVATION VS BALANCED STATE
# ============================================================

LON, LAT = np.meshgrid(
    lon,
    lat
)

vmax0 = max(
    np.percentile(
        np.abs(eta_raw0),
        98
    ),
    np.percentile(
        np.abs(eta_bal0),
        98
    ),
)


fig, ax = plt.subplots(
    1,
    3,
    figsize=(18,6)
)


p0 = ax[0].pcolormesh(
    lon,
    lat,
    eta_raw0,
    cmap="RdBu_r",
    vmin=-vmax0,
    vmax=vmax0,
    shading="auto"
)

skip = 8

ax[0].quiver(
    LON[::skip,::skip],
    LAT[::skip,::skip],
    u_raw0[::skip,::skip],
    v_raw0[::skip,::skip],
    color="black",
    scale=10
)

ax[0].set_title(
    "Observed initial state"
)


p1 = ax[1].pcolormesh(
    lon,
    lat,
    eta_bal0,
    cmap="RdBu_r",
    vmin=-vmax0,
    vmax=vmax0,
    shading="auto"
)

ax[1].quiver(
    LON[::skip,::skip],
    LAT[::skip,::skip],
    u_bal0[::skip,::skip],
    v_bal0[::skip,::skip],
    color="black",
    scale=10
)

ax[1].set_title(
    "Dynamically spun-up initial state"
)


diff = (
    eta_bal0
    -
    eta_raw0
)

dmax = max(
    np.percentile(
        np.abs(diff),
        98
    ),
    1e-4
)

p2 = ax[2].pcolormesh(
    lon,
    lat,
    diff,
    cmap="RdBu_r",
    vmin=-dmax,
    vmax=dmax,
    shading="auto"
)

ax[2].set_title(
    "Balanced minus observed SSH"
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
    shrink=0.8,
    label="SSH anomaly [m]"
)

fig.colorbar(
    p2,
    ax=ax[2],
    shrink=0.8,
    label="SSH adjustment [m]"
)


fig.suptitle(
    "Balanced initialization — "
    + str(
        dates[FINAL_START].date()
    )
)

fig.subplots_adjust(
    left=0.055,
    right=0.92,
    bottom=0.10,
    top=0.85,
    wspace=0.25
)

plt.savefig(
    FIGDIR
    + "/balanced_initial_state.png",
    dpi=220
)

plt.close()


# ============================================================
# FINAL FORECAST PLOTS
# ============================================================

for target in predictions:

    lead_days = (
        dates[target]
        -
        dates[FINAL_START]
    ).total_seconds()/86400.0


    eb, ub, vb = (
        predictions[target][
            "balanced_wind"
        ]
    )

    et = ETA_OBS[target]
    ut = U_OBS[target]
    vt = V_OBS[target]

    ep = ETA_OBS[
        FINAL_START
    ]


    vmax = max(
        np.percentile(
            np.abs(et),
            98
        ),
        np.percentile(
            np.abs(eb),
            98
        )
    )


    err = eb-et

    emax = max(
        np.percentile(
            np.abs(err),
            98
        ),
        1e-4
    )


    fig, ax = plt.subplots(
        1,
        4,
        figsize=(22,6)
    )


    # Observation
    p0 = ax[0].pcolormesh(
        lon,
        lat,
        et,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    ax[0].quiver(
        LON[::skip,::skip],
        LAT[::skip,::skip],
        ut[::skip,::skip],
        vt[::skip,::skip],
        color="black",
        scale=10
    )

    ax[0].set_title(
        "Held-out observation"
    )


    # Model
    p1 = ax[1].pcolormesh(
        lon,
        lat,
        eb,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    ax[1].quiver(
        LON[::skip,::skip],
        LAT[::skip,::skip],
        ub[::skip,::skip],
        vb[::skip,::skip],
        color="black",
        scale=10
    )

    ax[1].set_title(
        "Balanced + wind PDE"
    )


    # Persistence
    p2 = ax[2].pcolormesh(
        lon,
        lat,
        ep,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        shading="auto"
    )

    ax[2].set_title(
        "Persistence"
    )


    # Error
    p3 = ax[3].pcolormesh(
        lon,
        lat,
        err,
        cmap="RdBu_r",
        vmin=-emax,
        vmax=emax,
        shading="auto"
    )

    ax[3].set_title(
        "PDE minus observation"
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


    row = final[
        final["lead_days"]
        ==
        lead_days
    ].iloc[0]


    fig.suptitle(
        f"{lead_days:.0f}-day blind forecast from "
        f"{dates[FINAL_START].date()}\n"
        f"SSH RMSE: balanced PDE "
        f"{row['ssh_rmse_balanced_wind']:.3f} m; "
        f"persistence "
        f"{row['ssh_rmse_persistence']:.3f} m"
    )


    fig.colorbar(
        p1,
        ax=ax[:3],
        shrink=0.80,
        label="SSH anomaly [m]"
    )

    fig.colorbar(
        p3,
        ax=ax[3],
        shrink=0.80,
        label="SSH forecast error [m]"
    )


    fig.subplots_adjust(
        left=0.04,
        right=0.93,
        bottom=0.10,
        top=0.82,
        wspace=0.28
    )


    fn = (
        FIGDIR
        +
        f"/balanced_forecast_{int(round(lead_days)):02d}day.png"
    )

    plt.savefig(
        fn,
        dpi=220
    )

    plt.close()

    print(
        "SAVED:",
        fn
    )


print()
print("FILES")
print("-----")
print(
    OUTDIR
    + "/parameter_tuning.csv"
)
print(
    OUTDIR
    + "/best_parameters.json"
)
print(
    OUTDIR
    + "/final_holdout_scores.csv"
)
print(
    FIGDIR
    + "/balanced_initial_state.png"
)
print(
    FIGDIR
    + "/balanced_forecast_07day.png"
)
print(
    FIGDIR
    + "/balanced_forecast_14day.png"
)
