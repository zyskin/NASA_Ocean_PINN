import os
import itertools
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

OUTDIR = "data_processed/residual_nonlinear_tuning"
FIGDIR = "figures/residual_nonlinear_tuning"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LON0, LON1 = -170.0, -100.0
LAT0, LAT1 =  -10.0,   10.0


# ============================================================
# PARAMETER SEARCH
# ============================================================

NBG_VALUES = [1, 2, 3]

C_VALUES = [
    1.0,
    1.5,
    2.0,
    2.45,
    3.0,
]

HWIND_VALUES = [
    30.0,
    60.0,
    100.0,
    150.0,
]

DAMP_VALUES = [
    10.0,
    20.0,
    40.0,
]


# All parameter sets first tested on this many representative
# earlier forecast starts.
N_STAGE1_WINDOWS = 6

# Best Stage-1 parameter sets to test on ALL earlier starts.
N_STAGE2_KEEP = 12


# ============================================================
# PHYSICAL / NUMERICAL CONSTANTS
# ============================================================

G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0

NU = 1000.0            # m^2/s

# Stable nonlinear step after previous centered-scheme failure.
DT = 1800.0            # 30 min

SPONGE_CELLS = 5
SPONGE_MAX = 1.0/(2.0*86400.0)


# ============================================================
# OBJECTIVE
#
# Ratio = model RMSE / persistence RMSE.
#
# J < 1 means overall improvement over persistence.
#
# Weight SSH more strongly, but retain velocity because it is
# an important state variable and showed encouraging skill.
# ============================================================

W_SSH_7  = 0.35
W_SSH_14 = 0.35
W_UV_7   = 0.15
W_UV_14  = 0.15


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
        raise RuntimeError(f"Missing {name}")


ds = ds.sel(
    lon=slice(LON0, LON1),
    lat=slice(LAT0, LAT1)
)

dates = pd.DatetimeIndex(ds.time.values)

lon = ds.lon.values.astype(float)
lat = ds.lat.values.astype(float)

nx = len(lon)
ny = len(lat)
nt = len(dates)


def get(name):
    return np.asarray(
        ds[name]
        .transpose("time", "lat", "lon")
        .values,
        dtype=float
    )


ETA = get("ssha")
U   = get("u")
V   = get("v")
TX  = get("wind_stress_x")
TY  = get("wind_stress_y")


print()
print("============================================")
print("RESIDUAL NONLINEAR MODEL PARAMETER TUNING")
print("============================================")
print()
print("Observations:", nt)
print("Range:", dates[0], "to", dates[-1])
print("Grid:", nx, "x", ny)


# ============================================================
# FILL MISSING VALUES
# ============================================================

def fill2d(A):

    B = np.asarray(A, dtype=float).copy()

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


def fillcube(A):

    B = A.copy()

    for k in range(len(B)):
        B[k] = fill2d(B[k])

    return B


print("Filling spatial gaps...")

ETA = fillcube(ETA)
U   = fillcube(U)
V   = fillcube(V)
TX  = fillcube(TX)
TY  = fillcube(TY)


# ============================================================
# GRID
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


OBS_SEC = np.array([
    (d-dates[0]).total_seconds()
    for d in dates
])


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


def lap(A):

    Dxx = np.empty_like(A)

    Dxx[:, 1:-1] = (
        A[:, 2:]
        - 2.0*A[:, 1:-1]
        + A[:, :-2]
    ) / (
        dx[:, None]**2
    )

    Dxx[:, 0]  = Dxx[:, 1]
    Dxx[:, -1] = Dxx[:, -2]


    Dyy = np.empty_like(A)

    Dyy[1:-1] = (
        A[2:]
        - 2.0*A[1:-1]
        + A[:-2]
    ) / (
        dy**2
    )

    Dyy[0]  = Dyy[1]
    Dyy[-1] = Dyy[-2]

    return Dxx + Dyy


# ============================================================
# UPWIND ADVECTION
#
# u A_x + v A_y
# ============================================================

def advect_upwind(u, v, A):

    bx = np.empty_like(A)
    fx = np.empty_like(A)

    bx[:, 1:] = (
        A[:, 1:] - A[:, :-1]
    ) / dx[:, None]

    bx[:, 0] = (
        A[:, 1] - A[:, 0]
    ) / dx

    fx[:, :-1] = (
        A[:, 1:] - A[:, :-1]
    ) / dx[:, None]

    fx[:, -1] = (
        A[:, -1] - A[:, -2]
    ) / dx

    Ax = np.where(
        u >= 0,
        bx,
        fx
    )


    by = np.empty_like(A)
    fy = np.empty_like(A)

    by[1:] = (
        A[1:] - A[:-1]
    ) / dy

    by[0] = (
        A[1] - A[0]
    ) / dy

    fy[:-1] = (
        A[1:] - A[:-1]
    ) / dy

    fy[-1] = (
        A[-1] - A[-2]
    ) / dy

    Ay = np.where(
        v >= 0,
        by,
        fy
    )


    return (
        u*Ax
        +
        v*Ay
    )


# ============================================================
# SPONGE
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

SPONGE = np.zeros((ny, nx))

m = dist < SPONGE_CELLS

SPONGE[m] = (
    SPONGE_MAX
    *
    (
        (SPONGE_CELLS-dist[m])
        /
        SPONGE_CELLS
    )**2
)


# ============================================================
# TIME INTERPOLATION OF WIND
# ============================================================

def interp_time(A, tsec):

    if tsec <= OBS_SEC[0]:
        return A[0]

    if tsec >= OBS_SEC[-1]:
        return A[-1]

    j = np.searchsorted(
        OBS_SEC,
        tsec
    )

    j0 = j-1
    j1 = j

    a = (
        (tsec-OBS_SEC[j0])
        /
        (OBS_SEC[j1]-OBS_SEC[j0])
    )

    return (
        (1-a)*A[j0]
        +
        a*A[j1]
    )


# ============================================================
# PHYSICAL NONLINEAR OPERATOR
# ============================================================

def physical_rhs(
    u,
    v,
    eta,
    tx,
    ty,
    C,
    Hwind,
    damping_days
):

    He = C*C/G

    r = (
        1.0
        /
        (damping_days*86400.0)
    )


    ux = ddx(u)
    vy = ddy(v)

    ex = ddx(eta)
    ey = ddy(eta)


    adv_u = advect_upwind(
        u, v, u
    )

    adv_v = advect_upwind(
        u, v, v
    )

    adv_eta = advect_upwind(
        u, v, eta
    )


    du = (
        -adv_u
        +F*v
        -G*ex
        +tx/(RHO_WATER*Hwind)
        -r*u
        +NU*lap(u)
    )


    dv = (
        -adv_v
        -F*u
        -G*ey
        +ty/(RHO_WATER*Hwind)
        -r*v
        +NU*lap(v)
    )


    deta = (
        -adv_eta
        -He*(ux+vy)
    )


    return du, dv, deta


# ============================================================
# BUILD RESIDUAL-SUBTRACTED MODEL
# ============================================================

def make_model(
    start,
    Nbg,
    C,
    Hwind,
    damping_days
):

    j0 = start-Nbg+1

    if j0 < 0:
        raise RuntimeError(
            "Not enough observations for background."
        )

    ids = np.arange(
        j0,
        start+1
    )


    # Frozen background.
    ebar = np.mean(
        ETA[ids],
        axis=0
    )

    ubar = np.mean(
        U[ids],
        axis=0
    )

    vbar = np.mean(
        V[ids],
        axis=0
    )

    txbar = np.mean(
        TX[ids],
        axis=0
    )

    tybar = np.mean(
        TY[ids],
        axis=0
    )


    # Background PDE tendency that will be removed.
    base = physical_rhs(
        ubar,
        vbar,
        ebar,
        txbar,
        tybar,
        C,
        Hwind,
        damping_days
    )


    # Observed perturbation at forecast start.
    q0 = (
        U[start]   - ubar,
        V[start]   - vbar,
        ETA[start] - ebar
    )


    def rhs(
        up,
        vp,
        ep,
        tabs
    ):

        u = ubar + up
        v = vbar + vp
        eta = ebar + ep


        tx = interp_time(
            TX,
            tabs
        )

        ty = interp_time(
            TY,
            tabs
        )


        du, dv, de = physical_rhs(
            u,
            v,
            eta,
            tx,
            ty,
            C,
            Hwind,
            damping_days
        )


        # Remove frozen-background tendency.
        du -= base[0]
        dv -= base[1]
        de -= base[2]


        # Boundary damping acts only on perturbation.
        du -= SPONGE*up
        dv -= SPONGE*vp
        de -= SPONGE*ep


        return du, dv, de


    return (
        q0,
        rhs,
        (ubar, vbar, ebar)
    )


# ============================================================
# RK4
# ============================================================

def rk4(
    q,
    t,
    dt,
    rhs
):

    u, v, e = q

    k1 = rhs(
        u, v, e,
        t
    )

    k2 = rhs(
        u + 0.5*dt*k1[0],
        v + 0.5*dt*k1[1],
        e + 0.5*dt*k1[2],
        t + 0.5*dt
    )

    k3 = rhs(
        u + 0.5*dt*k2[0],
        v + 0.5*dt*k2[1],
        e + 0.5*dt*k2[2],
        t + 0.5*dt
    )

    k4 = rhs(
        u + dt*k3[0],
        v + dt*k3[1],
        e + dt*k3[2],
        t + dt
    )


    return (
        u
        + dt/6.0
        * (
            k1[0]
            +2*k2[0]
            +2*k3[0]
            +k4[0]
        ),

        v
        + dt/6.0
        * (
            k1[1]
            +2*k2[1]
            +2*k3[1]
            +k4[1]
        ),

        e
        + dt/6.0
        * (
            k1[2]
            +2*k2[2]
            +2*k3[2]
            +k4[2]
        ),
    )


# ============================================================
# INTEGRATE TO 7 AND 14 DAYS IN ONE RUN
# ============================================================

def integrate(
    q0,
    rhs,
    t0,
    target_times
):

    q = tuple(
        x.copy()
        for x in q0
    )

    t = float(t0)

    out = {}


    for target in sorted(target_times):

        while t < target-1e-6:

            dti = min(
                DT,
                target-t
            )

            q = rk4(
                q,
                t,
                dti,
                rhs
            )

            t += dti


            maxe = np.max(
                np.abs(q[2])
            )

            maxuv = np.max(
                np.sqrt(
                    q[0]**2
                    +
                    q[1]**2
                )
            )


            if (
                not np.isfinite(maxe)
                or
                not np.isfinite(maxuv)
                or
                maxe > 5.0
                or
                maxuv > 8.0
            ):
                raise RuntimeError(
                    f"instability: "
                    f"eta={maxe:.3g}, "
                    f"speed={maxuv:.3g}"
                )


        out[target] = tuple(
            x.copy()
            for x in q
        )


    return out


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
    u1, v1,
    u2, v2
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

    x = a.ravel()
    y = b.ravel()

    good = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    return float(
        np.corrcoef(
            x[good],
            y[good]
        )[0,1]
    )


# ============================================================
# RUN ONE 7/14-DAY FORECAST WINDOW
# ============================================================

def score_window(
    start,
    params,
    save_fields=False
):

    Nbg, C, Hwind, Tdamp = params


    q0, rhs, bg = make_model(
        start,
        Nbg,
        C,
        Hwind,
        Tdamp
    )


    targets = [
        start+1,
        start+2
    ]


    pred = integrate(
        q0,
        rhs,
        OBS_SEC[start],
        [
            OBS_SEC[targets[0]],
            OBS_SEC[targets[1]]
        ]
    )


    ubar, vbar, ebar = bg

    results = []
    field_output = {}


    for lead_index, target in enumerate(
        targets,
        start=1
    ):

        up, vp, ep = pred[
            OBS_SEC[target]
        ]

        u_model = ubar + up
        v_model = vbar + vp
        e_model = ebar + ep


        u_true = U[target]
        v_true = V[target]
        e_true = ETA[target]


        u_p = U[start]
        v_p = V[start]
        e_p = ETA[start]


        ssh_model = rmse(
            e_model,
            e_true
        )

        ssh_p = rmse(
            e_p,
            e_true
        )


        uv_model = vector_rmse(
            u_model,
            v_model,
            u_true,
            v_true
        )

        uv_p = vector_rmse(
            u_p,
            v_p,
            u_true,
            v_true
        )


        results.append({
            "lead_days":
                7*lead_index,

            "ssh_model":
                ssh_model,

            "ssh_persistence":
                ssh_p,

            "ssh_ratio":
                ssh_model/ssh_p,

            "ssh_skill":
                1.0-ssh_model/ssh_p,

            "uv_model":
                uv_model,

            "uv_persistence":
                uv_p,

            "uv_ratio":
                uv_model/uv_p,

            "uv_skill":
                1.0-uv_model/uv_p,

            "ssh_corr":
                corr(
                    e_model,
                    e_true
                ),
        })


        if save_fields:
            field_output[target] = {
                "eta": e_model,
                "u": u_model,
                "v": v_model,
            }


    # Combined objective.
    r7 = results[0]
    r14 = results[1]

    J = (
        W_SSH_7  * r7["ssh_ratio"]
        +
        W_SSH_14 * r14["ssh_ratio"]
        +
        W_UV_7   * r7["uv_ratio"]
        +
        W_UV_14  * r14["uv_ratio"]
    )


    return J, results, field_output


# ============================================================
# HOLDOUT DEFINITION
# ============================================================

# 42 = Oct 14 for current 45-date dataset.
FINAL_START = nt-3


print()
print("FINAL HOLDOUT")
print("-------------")
print(
    dates[FINAL_START].date(),
    "->",
    dates[FINAL_START+1].date(),
    "and",
    dates[FINAL_START+2].date()
)


# Training windows must have BOTH 7d and 14d target strictly
# before the final Oct-14 forecast start.
MIN_START = max(NBG_VALUES)-1

ALL_TRAIN_STARTS = np.arange(
    MIN_START,
    FINAL_START-2
)


print()
print(
    "Available earlier rolling windows:",
    len(ALL_TRAIN_STARTS)
)

print(
    "First:",
    dates[ALL_TRAIN_STARTS[0]].date()
)

print(
    "Last:",
    dates[ALL_TRAIN_STARTS[-1]].date(),
    "->",
    dates[ALL_TRAIN_STARTS[-1]+2].date()
)


# Representative Stage-1 windows.
idx = np.linspace(
    0,
    len(ALL_TRAIN_STARTS)-1,
    N_STAGE1_WINDOWS
)

STAGE1_STARTS = np.unique(
    ALL_TRAIN_STARTS[
        np.round(idx).astype(int)
    ]
)


print()
print("STAGE-1 WINDOWS")
print("---------------")

for s in STAGE1_STARTS:
    print(
        dates[s].date(),
        "->",
        dates[s+2].date()
    )


# ============================================================
# PARAMETER GRID
# ============================================================

PARAMS = list(
    itertools.product(
        NBG_VALUES,
        C_VALUES,
        HWIND_VALUES,
        DAMP_VALUES
    )
)


print()
print(
    "Parameter combinations:",
    len(PARAMS)
)


# ============================================================
# EVALUATE PARAMETER SET ON A SET OF WINDOWS
# ============================================================

def evaluate_params(
    params,
    starts
):

    objs = []

    ssh7 = []
    ssh14 = []

    uv7 = []
    uv14 = []

    wins_ssh7 = 0
    wins_ssh14 = 0

    failed = False


    for start in starts:

        try:

            J, res, _ = score_window(
                start,
                params,
                save_fields=False
            )

        except Exception:
            failed = True
            break


        objs.append(J)

        ssh7.append(
            res[0]["ssh_ratio"]
        )

        ssh14.append(
            res[1]["ssh_ratio"]
        )

        uv7.append(
            res[0]["uv_ratio"]
        )

        uv14.append(
            res[1]["uv_ratio"]
        )


        wins_ssh7 += int(
            res[0]["ssh_ratio"] < 1
        )

        wins_ssh14 += int(
            res[1]["ssh_ratio"] < 1
        )


    if failed or len(objs) == 0:

        return {
            "objective": 999.0,
            "failed": True,
        }


    return {
        "objective":
            float(np.mean(objs)),

        "failed":
            False,

        "ssh_ratio_7":
            float(np.mean(ssh7)),

        "ssh_ratio_14":
            float(np.mean(ssh14)),

        "uv_ratio_7":
            float(np.mean(uv7)),

        "uv_ratio_14":
            float(np.mean(uv14)),

        "ssh_wins_7":
            wins_ssh7,

        "ssh_wins_14":
            wins_ssh14,

        "n_windows":
            len(objs),
    }


# ============================================================
# STAGE 1
# ============================================================

print()
print("============================================")
print("STAGE 1 -- COARSE ROLLING SEARCH")
print("============================================")


stage1_rows = []


for i, params in enumerate(
    PARAMS,
    start=1
):

    Nbg, C, Hw, Tdamp = params


    ev = evaluate_params(
        params,
        STAGE1_STARTS
    )


    row = {
        "Nbg": Nbg,
        "C_m_s": C,
        "Hwind_m": Hw,
        "damping_days": Tdamp,
        **ev,
    }

    stage1_rows.append(row)


    print(
        f"[{i:03d}/{len(PARAMS)}]",
        f"N={Nbg}",
        f"c={C:g}",
        f"Hw={Hw:g}",
        f"Td={Tdamp:g}",
        "J=",
        f"{ev['objective']:.4f}"
    )


stage1 = pd.DataFrame(
    stage1_rows
).sort_values(
    "objective"
)

stage1.to_csv(
    OUTDIR
    + "/stage1_all_parameters.csv",
    index=False
)


print()
print("STAGE-1 TOP PARAMETERS")
print("----------------------")
print(
    stage1.head(
        N_STAGE2_KEEP
    ).to_string(
        index=False
    )
)


# ============================================================
# STAGE 2 -- ALL EARLIER ROLLING WINDOWS
# ============================================================

print()
print("============================================")
print("STAGE 2 -- ALL EARLIER WINDOWS")
print("============================================")


top = stage1.head(
    N_STAGE2_KEEP
)


stage2_rows = []


for i, (_, row1) in enumerate(
    top.iterrows(),
    start=1
):

    params = (
        int(row1.Nbg),
        float(row1.C_m_s),
        float(row1.Hwind_m),
        float(row1.damping_days)
    )


    ev = evaluate_params(
        params,
        ALL_TRAIN_STARTS
    )


    row = {
        "Nbg": params[0],
        "C_m_s": params[1],
        "Hwind_m": params[2],
        "damping_days": params[3],
        **ev,
    }


    stage2_rows.append(row)


    print(
        f"[{i:02d}/{len(top)}]",
        params,
        "J=",
        f"{ev['objective']:.4f}",
        "SSH ratios:",
        f"7d={ev.get('ssh_ratio_7', np.nan):.4f}",
        f"14d={ev.get('ssh_ratio_14', np.nan):.4f}"
    )


stage2 = pd.DataFrame(
    stage2_rows
).sort_values(
    "objective"
)

stage2.to_csv(
    OUTDIR
    + "/stage2_all_rolling_windows.csv",
    index=False
)


print()
print("============================================")
print("FINAL PARAMETER RANKING")
print("============================================")
print()

print(
    stage2.to_string(
        index=False
    )
)


bestrow = stage2.iloc[0]

BEST = (
    int(bestrow.Nbg),
    float(bestrow.C_m_s),
    float(bestrow.Hwind_m),
    float(bestrow.damping_days)
)


print()
print("SELECTED PARAMETERS")
print("-------------------")
print("Nbg =", BEST[0])
print("c =", BEST[1], "m/s")
print("Hwind =", BEST[2], "m")
print("damping =", BEST[3], "days")
print(
    "training objective =",
    float(bestrow.objective)
)


with open(
    OUTDIR
    + "/best_parameters.json",
    "w"
) as f:

    json.dump(
        {
            "background_nobs": BEST[0],
            "c_m_s": BEST[1],
            "Hwind_m": BEST[2],
            "damping_days": BEST[3],
            "nu_m2_s": NU,
            "dt_seconds": DT,
            "training_objective":
                float(bestrow.objective),
        },
        f,
        indent=2
    )


# ============================================================
# DETAILED ROLLING RESULTS FOR BEST PARAMETER SET
# ============================================================

rolling_rows = []


for start in ALL_TRAIN_STARTS:

    J, res, _ = score_window(
        start,
        BEST
    )


    for r in res:

        rolling_rows.append({
            "start_date":
                str(dates[start].date()),

            "lead_days":
                r["lead_days"],

            "ssh_ratio":
                r["ssh_ratio"],

            "ssh_skill":
                r["ssh_skill"],

            "uv_ratio":
                r["uv_ratio"],

            "uv_skill":
                r["uv_skill"],

            "ssh_corr":
                r["ssh_corr"],
        })


rolling = pd.DataFrame(
    rolling_rows
)

rolling.to_csv(
    OUTDIR
    + "/best_model_rolling_hindcasts.csv",
    index=False
)


print()
print("BEST MODEL ROLLING SUMMARY")
print("--------------------------")


for lead in [7, 14]:

    q = rolling[
        rolling.lead_days == lead
    ]

    print()
    print(
        f"{lead}-day:"
    )

    print(
        " mean SSH skill =",
        q.ssh_skill.mean()
    )

    print(
        " median SSH skill =",
        q.ssh_skill.median()
    )

    print(
        " SSH wins =",
        int(
            np.sum(
                q.ssh_skill > 0
            )
        ),
        "/",
        len(q)
    )

    print(
        " mean velocity skill =",
        q.uv_skill.mean()
    )


# ============================================================
# FINAL OCTOBER HOLDOUT
# ============================================================

print()
print("============================================")
print("FINAL UNTOUCHED OCTOBER HOLDOUT")
print("============================================")


J_final, final_res, final_fields = score_window(
    FINAL_START,
    BEST,
    save_fields=True
)


final_rows = []


for target, r in zip(
    [
        FINAL_START+1,
        FINAL_START+2
    ],
    final_res
):

    final_rows.append({
        "start_date":
            str(
                dates[FINAL_START].date()
            ),

        "target_date":
            str(
                dates[target].date()
            ),

        "lead_days":
            r["lead_days"],

        "ssh_rmse_model_m":
            r["ssh_model"],

        "ssh_rmse_persistence_m":
            r["ssh_persistence"],

        "ssh_ratio":
            r["ssh_ratio"],

        "ssh_skill":
            r["ssh_skill"],

        "ssh_corr_model":
            r["ssh_corr"],

        "vector_rmse_model_m_s":
            r["uv_model"],

        "vector_rmse_persistence_m_s":
            r["uv_persistence"],

        "vector_ratio":
            r["uv_ratio"],

        "vector_skill":
            r["uv_skill"],
    })


final = pd.DataFrame(
    final_rows
)

final.to_csv(
    OUTDIR
    + "/final_october_holdout.csv",
    index=False
)


print()
print(
    final.to_string(
        index=False
    )
)


# ============================================================
# PLOT PARAMETER RANKING
# ============================================================

plot_top = stage2.head(12).copy()

labels = [
    (
        f"N{int(r.Nbg)} "
        f"c={r.C_m_s:g} "
        f"H={r.Hwind_m:g} "
        f"T={r.damping_days:g}"
    )
    for _,r in plot_top.iterrows()
]


fig, ax = plt.subplots(
    figsize=(11,7)
)

y = np.arange(
    len(plot_top)
)

ax.barh(
    y,
    plot_top.objective
)

ax.axvline(
    1,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Persistence-equivalent objective"
)

ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()

ax.set_xlabel(
    "Combined rolling-hindcast objective"
)

ax.set_title(
    "Residual nonlinear model parameter search\n"
    "lower is better; below 1 beats persistence overall"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIGDIR
    + "/parameter_ranking.png",
    dpi=220
)

plt.close()


# ============================================================
# PLOT BEST MODEL ROLLING SKILL
# ============================================================

fig, ax = plt.subplots(
    figsize=(12,6)
)


for lead in [7,14]:

    q = rolling[
        rolling.lead_days == lead
    ]

    ax.plot(
        pd.to_datetime(
            q.start_date
        ),
        q.ssh_skill,
        marker="o",
        label=f"{lead}-day SSH"
    )


ax.axhline(
    0,
    color="black",
    linestyle="--",
    linewidth=1
)

ax.set_ylabel(
    "Skill vs persistence"
)

ax.set_xlabel(
    "Forecast start date"
)

ax.set_title(
    "Best residual nonlinear model:\n"
    "rolling SSH forecast skill"
)

ax.legend()

fig.autofmt_xdate()

plt.tight_layout()

plt.savefig(
    FIGDIR
    + "/best_model_rolling_skill.png",
    dpi=220
)

plt.close()


# ============================================================
# OCTOBER HOLDOUT MAPS
# ============================================================

for target in [
    FINAL_START+1,
    FINAL_START+2
]:

    field = final_fields[target]

    model_eta = field["eta"]
    true_eta = ETA[target]
    persist_eta = ETA[FINAL_START]

    err = (
        model_eta
        -
        true_eta
    )

    lead = int(
        round(
            (
                dates[target]
                -
                dates[FINAL_START]
            ).total_seconds()
            /
            86400
        )
    )


    vmax = max(
        np.percentile(
            np.abs(true_eta),
            98
        ),
        np.percentile(
            np.abs(model_eta),
            98
        ),
        np.percentile(
            np.abs(persist_eta),
            98
        )
    )


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


    fields = [
        (
            true_eta,
            "Held-out observation",
            vmax
        ),
        (
            model_eta,
            "Residual nonlinear forecast",
            vmax
        ),
        (
            persist_eta,
            "Persistence",
            vmax
        ),
        (
            err,
            "Forecast - observation",
            emax
        ),
    ]


    for i, (
        arr,
        title,
        vm
    ) in enumerate(fields):

        p = ax[i].pcolormesh(
            lon,
            lat,
            arr,
            cmap="RdBu_r",
            vmin=-vm,
            vmax=vm,
            shading="auto"
        )

        ax[i].axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.7
        )

        ax[i].set_title(title)
        ax[i].set_xlabel("Longitude")
        ax[i].set_ylabel("Latitude")


    rr = final[
        final.lead_days == lead
    ].iloc[0]


    fig.suptitle(
        f"{lead}-day untouched October forecast | "
        f"N={BEST[0]}, c={BEST[1]:g} m/s, "
        f"Hw={BEST[2]:g} m, Td={BEST[3]:g} d\n"
        f"SSH RMSE model={rr.ssh_rmse_model_m:.3f} m, "
        f"persistence={rr.ssh_rmse_persistence_m:.3f} m, "
        f"skill={rr.ssh_skill:+.3f}"
    )


    fig.subplots_adjust(
        left=0.04,
        right=0.95,
        bottom=0.10,
        top=0.80,
        wspace=0.26
    )


    plt.savefig(
        FIGDIR
        + f"/october_holdout_{lead:02d}day.png",
        dpi=220
    )

    plt.close()


print()
print("============================================")
print("OUTPUTS")
print("============================================")
print(
    OUTDIR
    + "/stage1_all_parameters.csv"
)
print(
    OUTDIR
    + "/stage2_all_rolling_windows.csv"
)
print(
    OUTDIR
    + "/best_parameters.json"
)
print(
    OUTDIR
    + "/best_model_rolling_hindcasts.csv"
)
print(
    OUTDIR
    + "/final_october_holdout.csv"
)
print(
    FIGDIR
    + "/parameter_ranking.png"
)
print(
    FIGDIR
    + "/best_model_rolling_skill.png"
)
