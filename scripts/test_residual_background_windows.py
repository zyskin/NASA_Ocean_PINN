import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

OUTDIR = "data_processed/background_window_test"
FIGDIR = "figures/background_window_test"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LON0, LON1 = -170.0, -100.0
LAT0, LAT1 =  -10.0,   10.0

BACKGROUND_WINDOWS = [1, 2, 3]


# ============================================================
# PHYSICS -- KEEP IDENTICAL TO PREVIOUS TEST
# ============================================================

G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0

C = 1.50                  # m/s
H_WIND = 60.0             # m
DAMPING_DAYS = 20.0
NU = 1000.0               # m^2/s

H_EQ = C*C/G
R_DAMP = 1.0/(DAMPING_DAYS*86400.0)

DT = 1800.0               # 30 min; nonlinear advection

SPONGE_CELLS = 5
SPONGE_MAX = 1.0/(2.0*86400.0)


# ============================================================
# LOAD
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
print("========================================")
print("RESIDUAL NONLINEAR BACKGROUND TEST")
print("========================================")
print()
print("Dates:", nt)
print("Range:", dates[0], "to", dates[-1])
print("Grid:", nx, "x", ny)
print()
print("c =", C, "m/s")
print("H_eq =", H_EQ, "m")
print("H_wind =", H_WIND, "m")
print("damping =", DAMPING_DAYS, "days")
print("background windows =", BACKGROUND_WINDOWS)


# ============================================================
# FILL MISSING SPATIAL VALUES
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

        B = np.nan_to_num(B, nan=m)

    return B


def fillcube(A):

    B = A.copy()

    for k in range(len(B)):
        B[k] = fill2d(B[k])

    return B


print("Filling gaps...")

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
# TIME INTERPOLATION OF WIND STRESS
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
# UPWIND NONLINEAR ADVECTION
#
# Computes
#
#       u A_x + v A_y
#
# using first-order donor-cell/upwind derivatives.
#
# We keep centered derivatives for pressure gradients,
# Coriolis-wave dynamics, divergence, etc.; only nonlinear
# material advection is changed.
# ============================================================

def advect_upwind(u, v, A):

    # --------------------------------------------------------
    # x derivative selected according to sign of u
    # --------------------------------------------------------

    dback_x = np.empty_like(A)
    dforw_x = np.empty_like(A)

    dback_x[:, 1:] = (
        A[:, 1:] - A[:, :-1]
    ) / dx[:, None]

    dback_x[:, 0] = (
        A[:, 1] - A[:, 0]
    ) / dx

    dforw_x[:, :-1] = (
        A[:, 1:] - A[:, :-1]
    ) / dx[:, None]

    dforw_x[:, -1] = (
        A[:, -1] - A[:, -2]
    ) / dx

    Ax = np.where(
        u >= 0.0,
        dback_x,
        dforw_x
    )


    # --------------------------------------------------------
    # y derivative selected according to sign of v
    # --------------------------------------------------------

    dback_y = np.empty_like(A)
    dforw_y = np.empty_like(A)

    dback_y[1:, :] = (
        A[1:, :] - A[:-1, :]
    ) / dy

    dback_y[0, :] = (
        A[1, :] - A[0, :]
    ) / dy

    dforw_y[:-1, :] = (
        A[1:, :] - A[:-1, :]
    ) / dy

    dforw_y[-1, :] = (
        A[-1, :] - A[-2, :]
    ) / dy

    Ay = np.where(
        v >= 0.0,
        dback_y,
        dforw_y
    )


    return (
        u*Ax
        +
        v*Ay
    )


# ============================================================
# FULL NONLINEAR MODAL OPERATOR F(q,tau)
# ============================================================

def physical_rhs(
    u,
    v,
    eta,
    tx,
    ty
):

    ux = ddx(u)
    uy = ddy(u)

    vx = ddx(v)
    vy = ddy(v)

    ex = ddx(eta)
    ey = ddy(eta)


    # Nonlinear material advection is evaluated with
    # an upwind operator for numerical stability.
    adv_u   = advect_upwind(u, v, u)
    adv_v   = advect_upwind(u, v, v)
    adv_eta = advect_upwind(u, v, eta)


    du = (
        -adv_u
        +F*v
        -G*ex
        +tx/(RHO_WATER*H_WIND)
        -R_DAMP*u
        +NU*lap(u)
    )


    dv = (
        -adv_v
        -F*u
        -G*ey
        +ty/(RHO_WATER*H_WIND)
        -R_DAMP*v
        +NU*lap(v)
    )


    deta = (
        -adv_eta
        -H_EQ*(ux+vy)
    )


    return du, dv, deta


# ============================================================
# BACKGROUND AND RESIDUAL MODEL
# ============================================================

def make_model(
    start,
    nbg
):

    j0 = max(
        0,
        start-nbg+1
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


    # Spurious tendency of frozen background under
    # simplified PDE. Subtract this exactly.
    base = physical_rhs(
        ubar,
        vbar,
        ebar,
        txbar,
        tybar
    )


    # Actual observed perturbation at forecast start.
    up0 = U[start]   - ubar
    vp0 = V[start]   - vbar
    ep0 = ETA[start] - ebar


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
            ty
        )


        # Residual subtraction.
        du -= base[0]
        dv -= base[1]
        de -= base[2]


        # Open-boundary damping only on perturbation.
        du -= SPONGE*up
        dv -= SPONGE*vp
        de -= SPONGE*ep


        return du, dv, de


    return (
        (up0, vp0, ep0),
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
# INTEGRATE
# ============================================================

def integrate(
    q0,
    rhs,
    t0,
    targets
):

    q = tuple(
        z.copy()
        for z in q0
    )

    t = float(t0)

    out = {}


    for target in sorted(targets):

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


            max_eta = np.max(
                np.abs(q[2])
            )

            max_uv = np.max(
                np.sqrt(
                    q[0]**2
                    +
                    q[1]**2
                )
            )


            if (
                not np.isfinite(max_eta)
                or
                not np.isfinite(max_uv)
                or
                max_eta > 5.0
                or
                max_uv > 8.0
            ):

                raise RuntimeError(
                    "Instability at "
                    f"{(t-t0)/86400:.2f} days: "
                    f"eta={max_eta}, "
                    f"speed={max_uv}"
                )


        out[target] = tuple(
            z.copy()
            for z in q
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

    a = a.ravel()
    b = b.ravel()

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    return float(
        np.corrcoef(
            a[good],
            b[good]
        )[0,1]
    )


# ============================================================
# TEST WINDOWS
# ============================================================

FINAL_START = nt-3

possible = np.arange(
    2,
    FINAL_START-1
)

N_EARLY = min(
    5,
    len(possible)
)

pick = np.linspace(
    0,
    len(possible)-1,
    N_EARLY
)

EARLY_STARTS = np.unique(
    possible[
        np.round(pick).astype(int)
    ]
)

STARTS = list(
    EARLY_STARTS
) + [
    FINAL_START
]


print()
print("TEST STARTS")
print("-----------")

for s in STARTS:
    print(dates[s].date())


# ============================================================
# RUN ALL BACKGROUNDS
# ============================================================

rows = []

final_fields = {}


for istart, start in enumerate(
    STARTS,
    start=1
):

    print()
    print(
        f"[{istart}/{len(STARTS)}]",
        "start",
        dates[start].date()
    )


    target_ids = [
        j
        for j in [
            start+1,
            start+2
        ]
        if j < nt
    ]

    target_times = [
        OBS_SEC[j]
        for j in target_ids
    ]


    for nbg in BACKGROUND_WINDOWS:

        q0, rhs, bg = make_model(
            start,
            nbg
        )

        ubar, vbar, ebar = bg


        init_eta_rms = float(
            np.sqrt(
                np.mean(
                    q0[2]**2
                )
            )
        )

        init_uv_rms = float(
            np.sqrt(
                np.mean(
                    q0[0]**2
                    +
                    q0[1]**2
                )
            )
        )


        print(
            f"  Nbg={nbg}: "
            f"initial perturbation "
            f"eta={init_eta_rms:.4f} m, "
            f"uv={init_uv_rms:.4f} m/s"
        )


        pred = integrate(
            q0,
            rhs,
            OBS_SEC[start],
            target_times
        )


        for target in target_ids:

            ts = OBS_SEC[target]

            up, vp, ep = pred[ts]

            e_pred = ebar + ep
            u_pred = ubar + up
            v_pred = vbar + vp


            e_true = ETA[target]
            u_true = U[target]
            v_true = V[target]


            # Persistence.
            e_per = ETA[start]
            u_per = U[start]
            v_per = V[start]


            ssh_model = rmse(
                e_pred,
                e_true
            )

            ssh_per = rmse(
                e_per,
                e_true
            )


            uv_model = vector_rmse(
                u_pred,
                v_pred,
                u_true,
                v_true
            )

            uv_per = vector_rmse(
                u_per,
                v_per,
                u_true,
                v_true
            )


            lead_days = (
                dates[target]
                -
                dates[start]
            ).total_seconds()/86400.0


            ssh_skill = (
                1.0
                -
                ssh_model/ssh_per
            )

            uv_skill = (
                1.0
                -
                uv_model/uv_per
            )


            rows.append({
                "start_date":
                    str(dates[start].date()),

                "target_date":
                    str(dates[target].date()),

                "lead_days":
                    lead_days,

                "background_nobs":
                    nbg,

                "initial_eta_pert_rms_m":
                    init_eta_rms,

                "initial_vector_pert_rms_m_s":
                    init_uv_rms,

                "ssh_rmse_model_m":
                    ssh_model,

                "ssh_rmse_persistence_m":
                    ssh_per,

                "ssh_skill":
                    ssh_skill,

                "ssh_corr_model":
                    corr(
                        e_pred,
                        e_true
                    ),

                "ssh_corr_persistence":
                    corr(
                        e_per,
                        e_true
                    ),

                "vector_rmse_model_m_s":
                    uv_model,

                "vector_rmse_persistence_m_s":
                    uv_per,

                "vector_skill":
                    uv_skill,
            })


            print(
                f"    {lead_days:.0f}d:",
                f"SSH={ssh_model:.4f}",
                f"P={ssh_per:.4f}",
                f"skill={ssh_skill:+.3f}",
                "|",
                f"UV skill={uv_skill:+.3f}"
            )


            if start == FINAL_START:

                final_fields[
                    (nbg, target)
                ] = (
                    e_pred,
                    u_pred,
                    v_pred
                )


# ============================================================
# SAVE SCORES
# ============================================================

scores = pd.DataFrame(rows)

scores.to_csv(
    OUTDIR
    + "/background_window_scores.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("MEAN RESULTS")
print("========================================")


summary_rows = []


for lead in sorted(
    scores.lead_days.unique()
):

    print()
    print(
        f"{lead:.0f}-DAY"
    )
    print(
        "-------"
    )


    for nbg in BACKGROUND_WINDOWS:

        q = scores[
            (scores.lead_days == lead)
            &
            (scores.background_nobs == nbg)
        ]


        row = {
            "lead_days":
                lead,

            "background_nobs":
                nbg,

            "mean_ssh_rmse_m":
                q.ssh_rmse_model_m.mean(),

            "mean_persistence_rmse_m":
                q.ssh_rmse_persistence_m.mean(),

            "mean_ssh_skill":
                q.ssh_skill.mean(),

            "median_ssh_skill":
                q.ssh_skill.median(),

            "number_ssh_wins":
                int(
                    np.sum(
                        q.ssh_skill > 0
                    )
                ),

            "number_cases":
                len(q),

            "mean_vector_skill":
                q.vector_skill.mean(),
        }

        summary_rows.append(row)


        print(
            f"N={nbg}:",
            f"SSH RMSE={row['mean_ssh_rmse_m']:.5f}",
            f"skill={row['mean_ssh_skill']:+.4f}",
            f"wins={row['number_ssh_wins']}/{row['number_cases']}",
            f"UV skill={row['mean_vector_skill']:+.4f}"
        )


summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    OUTDIR
    + "/background_window_summary.csv",
    index=False
)


# ============================================================
# FINAL OCTOBER HOLDOUT TABLE
# ============================================================

final = scores[
    scores.start_date
    ==
    str(
        dates[FINAL_START].date()
    )
]


print()
print("========================================")
print("FINAL OCTOBER HOLDOUT")
print("========================================")
print()
print(
    final.to_string(
        index=False
    )
)


# ============================================================
# PLOT: RMSE / PERSISTENCE FOR EVERY CASE
# ============================================================

fig, ax = plt.subplots(
    figsize=(11,6)
)


for nbg in BACKGROUND_WINDOWS:

    q = scores[
        scores.background_nobs
        ==
        nbg
    ]

    ratio = (
        q.ssh_rmse_model_m
        /
        q.ssh_rmse_persistence_m
    )


    ax.plot(
        np.arange(len(q)),
        ratio,
        marker="o",
        label=f"{nbg} observation background"
    )


ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Persistence"
)


qref = scores[
    scores.background_nobs
    ==
    BACKGROUND_WINDOWS[0]
]

labels = [
    row.start_date[5:]
    +
    f"\n{row.lead_days:.0f}d"
    for _,row in qref.iterrows()
]


ax.set_xticks(
    np.arange(len(qref))
)

ax.set_xticklabels(
    labels,
    rotation=45,
    ha="right"
)

ax.set_ylabel(
    "SSH RMSE / persistence RMSE"
)

ax.set_title(
    "Residual nonlinear model:\n"
    "effect of frozen-background averaging"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIGDIR
    + "/background_window_rmse_ratio.png",
    dpi=220
)

plt.close()


# ============================================================
# BAR PLOT OF MEAN SKILL
# ============================================================

for lead in sorted(
    summary.lead_days.unique()
):

    q = summary[
        summary.lead_days
        ==
        lead
    ]


    fig, ax = plt.subplots(
        figsize=(7,5)
    )


    ax.bar(
        [
            str(int(x))
            for x in q.background_nobs
        ],
        q.mean_ssh_skill
    )


    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "Number of weekly observations in frozen background"
    )

    ax.set_ylabel(
        "Mean SSH skill vs persistence"
    )

    ax.set_title(
        f"{lead:.0f}-day forecast"
    )


    plt.tight_layout()

    plt.savefig(
        FIGDIR
        +
        f"/mean_skill_{int(round(lead)):02d}day.png",
        dpi=220
    )

    plt.close()


# ============================================================
# FINAL OCTOBER MAPS
# ============================================================

for target in [
    FINAL_START+1,
    FINAL_START+2
]:

    if target >= nt:
        continue


    lead_days = (
        dates[target]
        -
        dates[FINAL_START]
    ).total_seconds()/86400.0


    truth = ETA[target]
    persistence = ETA[FINAL_START]


    fields = [
        (
            truth,
            "Held-out observation"
        ),
        (
            persistence,
            "Persistence"
        ),
    ]


    for nbg in BACKGROUND_WINDOWS:

        ep, up, vp = final_fields[
            (nbg, target)
        ]

        fields.append(
            (
                ep,
                f"Residual nonlinear N={nbg}"
            )
        )


    vmax = max(
        np.percentile(
            np.abs(x[0]),
            98
        )
        for x in fields
    )


    fig, ax = plt.subplots(
        1,
        len(fields),
        figsize=(5*len(fields), 6)
    )


    for a, (
        field,
        title
    ) in zip(
        ax,
        fields
    ):

        p = a.pcolormesh(
            lon,
            lat,
            field,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            shading="auto"
        )

        a.axhline(
            0,
            color="black",
            linestyle="--",
            linewidth=0.7
        )

        a.set_title(title)
        a.set_xlabel("Longitude")
        a.set_ylabel("Latitude")


    fig.suptitle(
        f"{lead_days:.0f}-day forecast from "
        f"{dates[FINAL_START].date()}"
    )


    fig.colorbar(
        p,
        ax=ax,
        shrink=0.8,
        label="SSH anomaly [m]"
    )


    fig.subplots_adjust(
        left=0.035,
        right=0.94,
        bottom=0.10,
        top=0.82,
        wspace=0.27
    )


    fn = (
        FIGDIR
        +
        f"/final_background_compare_{int(round(lead_days)):02d}day.png"
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
print("========================================")
print("FILES")
print("========================================")
print(
    OUTDIR
    + "/background_window_scores.csv"
)
print(
    OUTDIR
    + "/background_window_summary.csv"
)
print(
    FIGDIR
    + "/background_window_rmse_ratio.png"
)
print(
    FIGDIR
    + "/mean_skill_07day.png"
)
print(
    FIGDIR
    + "/mean_skill_14day.png"
)
