import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"

OUTDIR = "data_processed/nonlinear_comparison"
FIGDIR = "figures/nonlinear_comparison"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


LON0, LON1 = -170.0, -100.0
LAT0, LAT1 =  -10.0,   10.0


# ============================================================
# PHYSICS
# ============================================================

G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0

# Keep previous best parameters fixed for fair model comparison.
C = 1.50                  # m/s
H_WIND = 60.0             # m
DAMPING_DAYS = 20.0
NU = 1000.0               # m^2/s

H_EQ = C*C/G
R_DAMP = 1.0/(DAMPING_DAYS*86400.0)

# 1 hour nonlinear RK4 step.
DT = 3600.0


# ============================================================
# MODEL B BACKGROUND
# ============================================================
#
# 1 = literal last observation as frozen background.
#
# Then q'(t0)=0 and model B changes only because future forcing
# differs from forcing at t0.
#
# Later try 2 or 3 to use a short smooth background:
#
#     BACKGROUND_NOBS = 2
#
# then q'(t0) is nonzero and existing observed anomalies can
# propagate.
# ============================================================

BACKGROUND_NOBS = 1


# ============================================================
# BOUNDARY SPONGE
# ============================================================

SPONGE_CELLS = 5
SPONGE_MAX = 1.0/(2.0*86400.0)


# ============================================================
# LOAD
# ============================================================

ds = xr.open_dataset(FILE)

for var in [
    "ssha",
    "u",
    "v",
    "wind_stress_x",
    "wind_stress_y",
]:
    if var not in ds:
        raise RuntimeError(
            f"Missing variable: {var}"
        )


ds = ds.sel(
    lon=slice(LON0, LON1),
    lat=slice(LAT0, LAT1)
)


dates = pd.DatetimeIndex(
    ds.time.values
)

lon = ds.lon.values.astype(float)
lat = ds.lat.values.astype(float)

nx = len(lon)
ny = len(lat)
nt = len(dates)


def get(name):

    return np.asarray(
        ds[name]
        .transpose("time","lat","lon")
        .values,
        dtype=float
    )


ETA = get("ssha")
U = get("u")
V = get("v")
TX = get("wind_stress_x")
TY = get("wind_stress_y")


print()
print("========================================")
print("NONLINEAR FORECAST MODEL COMPARISON")
print("========================================")
print()
print("Dates:", nt)
print("Range:", dates[0], "to", dates[-1])
print("Grid:", nx, "x", ny)

print()
print("Parameters")
print("----------")
print("c =", C, "m/s")
print("H_eq =", H_EQ, "m")
print("H_wind =", H_WIND, "m")
print("damping =", DAMPING_DAYS, "days")
print("nu =", NU, "m2/s")
print("background observations =", BACKGROUND_NOBS)


# ============================================================
# FILL MISSING VALUES
# ============================================================

def fill2d(A):

    A = np.asarray(
        A,
        dtype=float
    ).copy()

    xx = np.arange(nx)

    for j in range(ny):

        row = A[j]
        good = np.isfinite(row)

        if good.sum() >= 2:
            A[j] = np.interp(
                xx,
                xx[good],
                row[good]
            )


    yy = np.arange(ny)

    for i in range(nx):

        col = A[:,i]
        good = np.isfinite(col)

        if good.sum() >= 2:
            A[:,i] = np.interp(
                yy,
                yy[good],
                col[good]
            )


    if np.isnan(A).any():

        m = np.nanmean(A)

        if not np.isfinite(m):
            m = 0.0

        A = np.nan_to_num(
            A,
            nan=m
        )

    return A


def fillcube(A):

    B = A.copy()

    for k in range(B.shape[0]):
        B[k] = fill2d(B[k])

    return B


print("Filling spatial gaps...")

ETA = fillcube(ETA)
U = fillcube(U)
V = fillcube(V)
TX = fillcube(TX)
TY = fillcube(TY)


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

F = f[:,None]


OBS_SEC = np.array([
    (d-dates[0]).total_seconds()
    for d in dates
])


# ============================================================
# DERIVATIVES
# ============================================================

def ddx(A):

    out = np.empty_like(A)

    out[:,1:-1] = (
        A[:,2:] - A[:,:-2]
    ) / (
        2.0*dx[:,None]
    )

    out[:,0] = (
        A[:,1] - A[:,0]
    ) / dx

    out[:,-1] = (
        A[:,-1] - A[:,-2]
    ) / dx

    return out


def ddy(A):

    out = np.empty_like(A)

    out[1:-1,:] = (
        A[2:,:] - A[:-2,:]
    ) / (
        2.0*dy
    )

    out[0,:] = (
        A[1,:] - A[0,:]
    ) / dy

    out[-1,:] = (
        A[-1,:] - A[-2,:]
    ) / dy

    return out


def lap(A):

    Dxx = np.empty_like(A)

    Dxx[:,1:-1] = (
        A[:,2:]
        - 2.0*A[:,1:-1]
        + A[:,:-2]
    ) / (
        dx[:,None]**2
    )

    Dxx[:,0] = Dxx[:,1]
    Dxx[:,-1] = Dxx[:,-2]


    Dyy = np.empty_like(A)

    Dyy[1:-1,:] = (
        A[2:,:]
        - 2.0*A[1:-1,:]
        + A[:-2,:]
    ) / (
        dy**2
    )

    Dyy[0,:] = Dyy[1,:]
    Dyy[-1,:] = Dyy[-2,:]

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

SPONGE = np.zeros(
    (ny,nx)
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
# INTERPOLATE WIND IN TIME
# ============================================================

def interp_time(A, tabs):

    if tabs <= OBS_SEC[0]:
        return A[0]

    if tabs >= OBS_SEC[-1]:
        return A[-1]

    j = np.searchsorted(
        OBS_SEC,
        tabs
    )

    j0 = j-1
    j1 = j

    a = (
        (tabs-OBS_SEC[j0])
        /
        (OBS_SEC[j1]-OBS_SEC[j0])
    )

    return (
        (1-a)*A[j0]
        +
        a*A[j1]
    )


# ============================================================
# FULL NONLINEAR MODAL RHS
#
# This is the common physical operator F(q,tau).
#
# q = (u,v,eta)
#
# u_t + u u_x + v u_y - fv = ...
#
# eta_t + u eta_x + v eta_y
#       + H_eq div(u) = 0
#
# H_eq is modal/equivalent depth, not physical ocean depth.
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


    du = (
        -u*ux
        -v*uy
        +F*v
        -G*ex
        +tx/(RHO_WATER*H_WIND)
        -R_DAMP*u
        +NU*lap(u)
    )


    dv = (
        -u*vx
        -v*vy
        -F*u
        -G*ey
        +ty/(RHO_WATER*H_WIND)
        -R_DAMP*v
        +NU*lap(v)
    )


    deta = (
        -u*ex
        -v*ey
        -H_EQ*(ux+vy)
    )


    return du,dv,deta


# ============================================================
# MODEL A
#
# TOTAL-STATE NONLINEAR MODEL
#
# Directly evolve observed u,v,eta.
#
# The edge sponge relaxes toward the initial observed state
# instead of toward zero.
# ============================================================

def rhs_A(
    u,
    v,
    eta,
    tabs,
    q0
):

    tx = interp_time(
        TX,
        tabs
    )

    ty = interp_time(
        TY,
        tabs
    )


    du,dv,de = physical_rhs(
        u,v,eta,
        tx,ty
    )


    u0,v0,e0 = q0

    du -= SPONGE*(u-u0)
    dv -= SPONGE*(v-v0)
    de -= SPONGE*(eta-e0)

    return du,dv,de


# ============================================================
# MODEL B
#
# RESIDUAL-SUBTRACTED NONLINEAR PERTURBATION MODEL
#
# q = qbar + q'
#
# q'_t =
#
#      F(qbar + q', tau(t))
#      -
#      F(qbar, tau_bar)
#
# Therefore if
#
#      q' = 0
#      tau = tau_bar
#
# then q'_t = 0 exactly.
# ============================================================

def make_background(start):

    j0 = max(
        0,
        start-BACKGROUND_NOBS+1
    )

    ids = np.arange(
        j0,
        start+1
    )

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

    return (
        ubar,
        vbar,
        ebar,
        txbar,
        tybar
    )


def make_model_B(
    start
):

    (
        ubar,
        vbar,
        ebar,
        txbar,
        tybar
    ) = make_background(start)


    base_rhs = physical_rhs(
        ubar,
        vbar,
        ebar,
        txbar,
        tybar
    )


    # Initial perturbation is chosen so total state exactly
    # equals observation at forecast start.
    up0 = U[start] - ubar
    vp0 = V[start] - vbar
    ep0 = ETA[start] - ebar


    def rhs_B(
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


        du,dv,de = physical_rhs(
            u,v,eta,
            tx,ty
        )


        du -= base_rhs[0]
        dv -= base_rhs[1]
        de -= base_rhs[2]


        # Sponge only perturbations.
        du -= SPONGE*up
        dv -= SPONGE*vp
        de -= SPONGE*ep


        return du,dv,de


    return (
        (up0,vp0,ep0),
        rhs_B,
        (ubar,vbar,ebar)
    )


# ============================================================
# GENERIC RK4
# ============================================================

def rk4(
    q,
    tabs,
    dtsec,
    rhs
):

    u,v,e = q

    k1 = rhs(
        u,v,e,
        tabs
    )

    k2 = rhs(
        u + 0.5*dtsec*k1[0],
        v + 0.5*dtsec*k1[1],
        e + 0.5*dtsec*k1[2],
        tabs + 0.5*dtsec
    )

    k3 = rhs(
        u + 0.5*dtsec*k2[0],
        v + 0.5*dtsec*k2[1],
        e + 0.5*dtsec*k2[2],
        tabs + 0.5*dtsec
    )

    k4 = rhs(
        u + dtsec*k3[0],
        v + dtsec*k3[1],
        e + dtsec*k3[2],
        tabs + dtsec
    )


    un = (
        u
        +
        dtsec/6.0
        *
        (
            k1[0]
            +2*k2[0]
            +2*k3[0]
            +k4[0]
        )
    )

    vn = (
        v
        +
        dtsec/6.0
        *
        (
            k1[1]
            +2*k2[1]
            +2*k3[1]
            +k4[1]
        )
    )

    en = (
        e
        +
        dtsec/6.0
        *
        (
            k1[2]
            +2*k2[2]
            +2*k3[2]
            +k4[2]
        )
    )

    return un,vn,en


# ============================================================
# INTEGRATOR WITH 7/14-DAY OUTPUTS
# ============================================================

def integrate_targets(
    q0,
    rhs,
    t0,
    target_times
):

    q = tuple(
        x.copy()
        for x in q0
    )

    current = float(t0)

    result = {}


    for target in sorted(target_times):

        while current < target-1e-6:

            dti = min(
                DT,
                target-current
            )

            q = rk4(
                q,
                current,
                dti,
                rhs
            )

            current += dti


            u,v,e = q

            max_eta = np.max(
                np.abs(e)
            )

            max_uv = np.max(
                np.sqrt(u*u+v*v)
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
                    "Model instability at "
                    f"{(current-t0)/86400:.2f} days: "
                    f"eta={max_eta}, "
                    f"speed={max_uv}"
                )


        result[target] = tuple(
            x.copy()
            for x in q
        )


    return result


# ============================================================
# METRICS
# ============================================================

def rmse(a,b):

    return float(
        np.sqrt(
            np.mean(
                (a-b)**2
            )
        )
    )


def vector_rmse(
    u1,v1,
    u2,v2
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


def corr(a,b):

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
# RUN ONE START DATE
# ============================================================

def run_start(
    start,
    leads=(1,2)
):

    targets = [
        start + n
        for n in leads
        if start+n < nt
    ]

    t0 = OBS_SEC[start]

    target_seconds = [
        OBS_SEC[t]
        for t in targets
    ]


    # --------------------------------------------------------
    # MODEL A
    # --------------------------------------------------------

    qA0 = (
        U[start].copy(),
        V[start].copy(),
        ETA[start].copy()
    )


    def wrapped_A(
        u,v,e,tabs
    ):

        return rhs_A(
            u,v,e,tabs,
            qA0
        )


    outA = integrate_targets(
        qA0,
        wrapped_A,
        t0,
        target_seconds
    )


    # --------------------------------------------------------
    # MODEL B
    # --------------------------------------------------------

    (
        qB0,
        rhsB,
        bgB
    ) = make_model_B(
        start
    )


    outBprime = integrate_targets(
        qB0,
        rhsB,
        t0,
        target_seconds
    )


    ubar,vbar,ebar = bgB


    rows = []

    fields = {}


    for target in targets:

        ts = OBS_SEC[target]


        # ---------------- Model A ----------------

        uA,vA,eA = outA[ts]


        # ---------------- Model B ----------------

        up,vp,ep = outBprime[ts]

        uB = ubar + up
        vB = vbar + vp
        eB = ebar + ep


        # ---------------- Truth ----------------

        et = ETA[target]
        ut = U[target]
        vt = V[target]


        # ---------------- Persistence ----------------

        eP = ETA[start]
        uP = U[start]
        vP = V[start]


        lead_days = (
            dates[target]
            -
            dates[start]
        ).total_seconds()/86400.0


        sshP = rmse(
            eP,
            et
        )

        uvP = vector_rmse(
            uP,vP,
            ut,vt
        )


        sshA = rmse(
            eA,
            et
        )

        sshB = rmse(
            eB,
            et
        )


        uvA = vector_rmse(
            uA,vA,
            ut,vt
        )

        uvB = vector_rmse(
            uB,vB,
            ut,vt
        )


        # Increment forecast:
        #
        # observed change from start.
        #
        de_true = (
            et - ETA[start]
        )

        de_A = (
            eA - ETA[start]
        )

        de_B = (
            eB - ETA[start]
        )


        row = {
            "start_date":
                str(dates[start].date()),

            "target_date":
                str(dates[target].date()),

            "lead_days":
                lead_days,

            "ssh_rmse_A_total_nonlinear":
                sshA,

            "ssh_rmse_B_residual_nonlinear":
                sshB,

            "ssh_rmse_persistence":
                sshP,

            "ssh_skill_A":
                1.0-sshA/sshP,

            "ssh_skill_B":
                1.0-sshB/sshP,

            "ssh_corr_A":
                corr(eA,et),

            "ssh_corr_B":
                corr(eB,et),

            "ssh_corr_persistence":
                corr(eP,et),

            "vector_rmse_A":
                uvA,

            "vector_rmse_B":
                uvB,

            "vector_rmse_persistence":
                uvP,

            "vector_skill_A":
                1.0-uvA/uvP,

            "vector_skill_B":
                1.0-uvB/uvP,

            "increment_rmse_A":
                rmse(
                    de_A,
                    de_true
                ),

            "increment_rmse_B":
                rmse(
                    de_B,
                    de_true
                ),

            "increment_rmse_zero":
                rmse(
                    np.zeros_like(
                        de_true
                    ),
                    de_true
                ),
        }


        rows.append(row)


        fields[target] = {
            "A": (
                eA,uA,vA
            ),
            "B": (
                eB,uB,vB
            ),
            "truth": (
                et,ut,vt
            ),
            "persistence": (
                eP,uP,vP
            ),
        }


    return rows,fields


# ============================================================
# TEST WINDOWS
#
# Use the same style as previous experiment:
# five distributed earlier windows plus final Oct 14 holdout.
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
print("TEST START DATES")
print("----------------")

for s in STARTS:
    print(dates[s].date())


# ============================================================
# RUN
# ============================================================

all_rows = []
final_fields = None


for n,start in enumerate(
    STARTS,
    start=1
):

    print()
    print(
        f"[{n}/{len(STARTS)}]",
        "start",
        dates[start].date()
    )


    try:

        rows,fields = run_start(
            start,
            leads=(1,2)
        )


        all_rows.extend(
            rows
        )


        if start == FINAL_START:
            final_fields = fields


        for row in rows:

            print(
                " ",
                f"{row['lead_days']:.0f}d",
                "SSH RMSE:",
                f"A={row['ssh_rmse_A_total_nonlinear']:.4f}",
                f"B={row['ssh_rmse_B_residual_nonlinear']:.4f}",
                f"P={row['ssh_rmse_persistence']:.4f}",
                "| skill:",
                f"A={row['ssh_skill_A']:+.3f}",
                f"B={row['ssh_skill_B']:+.3f}"
            )


    except Exception as e:

        print(
            "FAILED:",
            repr(e)
        )


scores = pd.DataFrame(
    all_rows
)

scores.to_csv(
    OUTDIR
    + "/nonlinear_model_scores.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("========================================")
print("MEAN PERFORMANCE")
print("========================================")


for lead in sorted(
    scores.lead_days.unique()
):

    q = scores[
        scores.lead_days == lead
    ]

    print()
    print(
        f"{lead:.0f}-DAY"
    )

    print(
        " mean SSH RMSE:"
    )

    print(
        "   A total nonlinear       =",
        q.ssh_rmse_A_total_nonlinear.mean()
    )

    print(
        "   B residual nonlinear    =",
        q.ssh_rmse_B_residual_nonlinear.mean()
    )

    print(
        "   persistence             =",
        q.ssh_rmse_persistence.mean()
    )

    print(
        " mean SSH skill vs persistence:"
    )

    print(
        "   A =",
        q.ssh_skill_A.mean()
    )

    print(
        "   B =",
        q.ssh_skill_B.mean()
    )

    print(
        " mean vector skill:"
    )

    print(
        "   A =",
        q.vector_skill_A.mean()
    )

    print(
        "   B =",
        q.vector_skill_B.mean()
    )


# ============================================================
# FINAL OCTOBER HOLDOUT
# ============================================================

final_rows = scores[
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
    final_rows.to_string(
        index=False
    )
)


# ============================================================
# SIMPLE RMSE-RATIO SUMMARY PLOT
#
# Ratio < 1 means beats persistence.
# ============================================================

fig, ax = plt.subplots(
    figsize=(10,6)
)


for model,label in [
    (
        "ssh_rmse_A_total_nonlinear",
        "A: total nonlinear"
    ),
    (
        "ssh_rmse_B_residual_nonlinear",
        "B: residual nonlinear"
    ),
]:

    ratios = (
        scores[model]
        /
        scores["ssh_rmse_persistence"]
    )

    x = np.arange(
        len(scores)
    )

    ax.plot(
        x,
        ratios,
        marker="o",
        label=label
    )


ax.axhline(
    1.0,
    color="black",
    linestyle="--",
    linewidth=1.2,
    label="Persistence"
)

labels = [
    x.start_date[5:]
    + f"\n{x.lead_days:.0f}d"
    for _,x in scores.iterrows()
]

ax.set_xticks(
    np.arange(len(scores))
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
    "Nonlinear forecast skill\n"
    "below 1 = better than persistence"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    FIGDIR
    + "/nonlinear_rmse_ratio.png",
    dpi=220
)

plt.close()


# ============================================================
# FINAL HOLDOUT MAPS
# ============================================================

if final_fields is not None:

    for target,ff in final_fields.items():

        lead_days = (
            dates[target]
            -
            dates[FINAL_START]
        ).total_seconds()/86400.0


        eA,uA,vA = ff["A"]
        eB,uB,vB = ff["B"]
        et,ut,vt = ff["truth"]
        ep,up,vp = ff["persistence"]


        vmax = max(
            np.percentile(
                np.abs(et),
                98
            ),
            np.percentile(
                np.abs(eA),
                98
            ),
            np.percentile(
                np.abs(eB),
                98
            )
        )


        fig,ax = plt.subplots(
            1,
            4,
            figsize=(22,6)
        )


        fields_plot = [
            (
                et,
                "Held-out observation"
            ),
            (
                eA,
                "A: total nonlinear"
            ),
            (
                eB,
                "B: residual nonlinear"
            ),
            (
                ep,
                "Persistence"
            ),
        ]


        for a,(field,title) in zip(
            ax,
            fields_plot
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

            a.set_xlabel(
                "Longitude"
            )

            a.set_ylabel(
                "Latitude"
            )

            a.set_title(
                title
            )


        row = final_rows[
            final_rows.lead_days
            ==
            lead_days
        ].iloc[0]


        fig.suptitle(
            f"{lead_days:.0f}-day forecast from "
            f"{dates[FINAL_START].date()}\n"
            f"SSH RMSE: "
            f"A={row.ssh_rmse_A_total_nonlinear:.3f} m, "
            f"B={row.ssh_rmse_B_residual_nonlinear:.3f} m, "
            f"persistence={row.ssh_rmse_persistence:.3f} m"
        )


        fig.colorbar(
            p,
            ax=ax,
            shrink=0.80,
            label="SSH anomaly [m]"
        )


        fig.subplots_adjust(
            left=0.04,
            right=0.93,
            bottom=0.10,
            top=0.82,
            wspace=0.25
        )


        fn = (
            FIGDIR
            +
            f"/final_compare_{int(round(lead_days)):02d}day.png"
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
print("SAVED:")
print(
    OUTDIR
    + "/nonlinear_model_scores.csv"
)
print(
    FIGDIR
    + "/nonlinear_rmse_ratio.png"
)
