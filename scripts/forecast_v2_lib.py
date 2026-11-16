"""
forecast_v2_lib.py  --  reusable engine for the residual nonlinear equatorial
shallow-water forecast and its diagnostics.

Faithfully reproduces the baseline model in
scripts/tune_residual_nonlinear_model.py (residual-subtracted weakly nonlinear
modal SWE, RK4, dt=1800s, first-order upwind advection, NU=1000, sponge),
and adds:
  * an optional second-order MUSCL/minmod advection scheme,
  * selectable velocity source (total OSCAR u,v vs geostrophic ug,vg),
  * a clean run_forecast() returning model / persistence / truth / increments,
  * an on-disk cache of raw PDE forecast fields so calibration experiments do
    not re-run the integrator.

Physics (modal / equivalent-depth interpretation; eta is satellite SSHA, He is
modal equivalent depth, r is phenomenological damping -- NOT literal depths):
    u_t = -(u.grad)u + f v - g eta_x + tau_x/(rho*Hw) - r u + nu lap(u)
    v_t = -(u.grad)v - f u - g eta_y + tau_y/(rho*Hw) - r v + nu lap(v)
    eta_t = -(u.grad)eta - He (u_x+v_y),   He = c^2/g
evolved as a perturbation about a frozen observed background, with the frozen
background tendency subtracted so persistence (q'=0, tau=tau_bar) is an exact
steady state.
"""
import os
import hashlib
import numpy as np
import pandas as pd
import xarray as xr

# ---- constants (identical to baseline) ----
G = 9.81
OMEGA = 7.2921159e-5
R_EARTH = 6371e3
RHO_WATER = 1025.0
NU = 1000.0
DT = 1800.0
SPONGE_CELLS = 5
SPONGE_MAX = 1.0 / (2.0 * 86400.0)

DEFAULT_FILE = "data_processed/ssh_oscar_wind_weekly_pacific.nc"
DOMAIN = dict(LON0=-170.0, LON1=-100.0, LAT0=-10.0, LAT1=10.0)
CACHE_DIR = "data_processed/forecast_v2/_pde_cache"


class Cube:
    """Loads + gap-fills the cropped cube and holds grid operators."""

    def __init__(self, file=DEFAULT_FILE, vel_source="total", domain=DOMAIN,
                 wind=True):
        ds = xr.open_dataset(file).sel(
            lon=slice(domain["LON0"], domain["LON1"]),
            lat=slice(domain["LAT0"], domain["LAT1"]))
        self.ds = ds
        self.dates = pd.DatetimeIndex(ds.time.values)
        self.lon = ds.lon.values.astype(float)
        self.lat = ds.lat.values.astype(float)
        self.nx = len(self.lon)
        self.ny = len(self.lat)
        self.nt = len(self.dates)
        self.vel_source = vel_source
        self.wind = wind

        def get(name):
            return np.asarray(ds[name].transpose("time", "lat", "lon").values,
                              dtype=float)

        uname, vname = ("u", "v") if vel_source == "total" else ("ug", "vg")
        self.ETA = self._fillcube(get("ssha"))
        self.U = self._fillcube(get(uname))
        self.V = self._fillcube(get(vname))
        if wind and "wind_stress_x" in ds and "wind_stress_y" in ds:
            self.TX = self._fillcube(get("wind_stress_x"))
            self.TY = self._fillcube(get("wind_stress_y"))
            self.has_wind = True
        else:
            self.TX = np.zeros_like(self.ETA)
            self.TY = np.zeros_like(self.ETA)
            self.has_wind = False

        # grid metric
        dlon = np.deg2rad(float(np.median(np.diff(self.lon))))
        dlat = np.deg2rad(float(np.median(np.diff(self.lat))))
        self.dy = R_EARTH * dlat
        self.dx = R_EARTH * np.cos(np.deg2rad(self.lat)) * dlon  # (ny,)
        self.dxc = self.dx[:, None]
        f = 2.0 * OMEGA * np.sin(np.deg2rad(self.lat))
        self.F = f[:, None]
        self.OBS_SEC = np.array([(d - self.dates[0]).total_seconds()
                                 for d in self.dates])
        self._build_sponge()

    # ---- gap fill (identical to baseline) ----
    def _fill2d(self, A):
        B = np.asarray(A, float).copy()
        xx = np.arange(self.nx)
        for j in range(self.ny):
            row = B[j]; good = np.isfinite(row)
            if good.sum() >= 2:
                B[j] = np.interp(xx, xx[good], row[good])
        yy = np.arange(self.ny)
        for i in range(self.nx):
            col = B[:, i]; good = np.isfinite(col)
            if good.sum() >= 2:
                B[:, i] = np.interp(yy, yy[good], col[good])
        if np.isnan(B).any():
            m = np.nanmean(B)
            B = np.nan_to_num(B, nan=(m if np.isfinite(m) else 0.0))
        return B

    def _fillcube(self, A):
        B = A.copy()
        for k in range(len(B)):
            B[k] = self._fill2d(B[k])
        return B

    def _build_sponge(self):
        jj, ii = np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        dist = np.minimum.reduce([ii, self.ny - 1 - ii, jj, self.nx - 1 - jj])
        S = np.zeros((self.ny, self.nx))
        m = dist < SPONGE_CELLS
        S[m] = SPONGE_MAX * ((SPONGE_CELLS - dist[m]) / SPONGE_CELLS) ** 2
        self.SPONGE = S

    # ---- derivative operators (centered, identical to baseline) ----
    def ddx(self, A):
        D = np.empty_like(A)
        D[:, 1:-1] = (A[:, 2:] - A[:, :-2]) / (2.0 * self.dxc)
        D[:, 0] = (A[:, 1] - A[:, 0]) / self.dx
        D[:, -1] = (A[:, -1] - A[:, -2]) / self.dx
        return D

    def ddy(self, A):
        D = np.empty_like(A)
        D[1:-1] = (A[2:] - A[:-2]) / (2.0 * self.dy)
        D[0] = (A[1] - A[0]) / self.dy
        D[-1] = (A[-1] - A[-2]) / self.dy
        return D

    def lap(self, A):
        Dxx = np.empty_like(A)
        Dxx[:, 1:-1] = (A[:, 2:] - 2.0 * A[:, 1:-1] + A[:, :-2]) / (self.dxc ** 2)
        Dxx[:, 0] = Dxx[:, 1]; Dxx[:, -1] = Dxx[:, -2]
        Dyy = np.empty_like(A)
        Dyy[1:-1] = (A[2:] - 2.0 * A[1:-1] + A[:-2]) / (self.dy ** 2)
        Dyy[0] = Dyy[1]; Dyy[-1] = Dyy[-2]
        return Dxx + Dyy

    # ---- advection: first-order upwind (identical to baseline) ----
    def advect_upwind(self, u, v, A):
        dx = self.dx
        bx = np.empty_like(A); fx = np.empty_like(A)
        bx[:, 1:] = (A[:, 1:] - A[:, :-1]) / self.dxc
        bx[:, 0] = (A[:, 1] - A[:, 0]) / dx
        fx[:, :-1] = (A[:, 1:] - A[:, :-1]) / self.dxc
        fx[:, -1] = (A[:, -1] - A[:, -2]) / dx
        Ax = np.where(u >= 0, bx, fx)
        by = np.empty_like(A); fy = np.empty_like(A)
        by[1:] = (A[1:] - A[:-1]) / self.dy
        by[0] = (A[1] - A[0]) / self.dy
        fy[:-1] = (A[1:] - A[:-1]) / self.dy
        fy[-1] = (A[-1] - A[-2]) / self.dy
        Ay = np.where(v >= 0, by, fy)
        return u * Ax + v * Ay

    # ---- advection: MUSCL / minmod (second-order, conservative flux form) ----
    # u.grad A = div(u A) - A div(u). The flux divergence uses minmod-limited
    # face reconstruction + upwind; div(u) is centered. Reduces to 1st order at
    # extrema and near boundaries.
    def advect_muscl(self, u, v, A):
        ny, nx = A.shape

        def minmod(a, b):
            return np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)

        # --- x flux ---
        uf = 0.5 * (u[:, :-1] + u[:, 1:])           # face velocity (ny, nx-1)
        dA = np.zeros_like(A)
        dA[:, 1:-1] = A[:, 2:] - A[:, :-2]
        # limited slopes per cell
        sL = np.zeros_like(A); sR = np.zeros_like(A)
        sL[:, 1:-1] = minmod(A[:, 1:-1] - A[:, :-2], A[:, 2:] - A[:, 1:-1])
        AL = A[:, :-1] + 0.5 * sL[:, :-1]            # left state at face i+1/2
        AR = A[:, 1:] - 0.5 * sL[:, 1:]             # right state
        Aface = np.where(uf >= 0, AL, AR)
        Fx = uf * Aface                              # (ny, nx-1)
        divFx = np.zeros_like(A)
        divFx[:, 1:-1] = (Fx[:, 1:] - Fx[:, :-1]) / self.dxc
        # 1st-order upwind at the two edge columns
        divFx[:, 0] = u[:, 0] * (A[:, 1] - A[:, 0]) / self.dx + A[:, 0] * 0  # placeholder
        # --- y flux ---
        vf = 0.5 * (v[:-1, :] + v[1:, :])
        sLy = np.zeros_like(A)
        sLy[1:-1, :] = minmod(A[1:-1, :] - A[:-2, :], A[2:, :] - A[1:-1, :])
        ALy = A[:-1, :] + 0.5 * sLy[:-1, :]
        ARy = A[1:, :] - 0.5 * sLy[1:, :]
        Afacey = np.where(vf >= 0, ALy, ARy)
        Fy = vf * Afacey
        divFy = np.zeros_like(A)
        divFy[1:-1, :] = (Fy[1:, :] - Fy[:-1, :]) / self.dy

        divu = self.ddx(u) + self.ddy(v)
        adv = (divFx + divFy) - A * divu

        # edge rows/cols: fall back to first-order upwind to stay bounded
        up = self.advect_upwind(u, v, A)
        adv[:, 0] = up[:, 0]; adv[:, -1] = up[:, -1]
        adv[0, :] = up[0, :]; adv[-1, :] = up[-1, :]
        return adv


def physical_rhs(cube, u, v, eta, tx, ty, C, Hwind, damping_days, scheme):
    He = C * C / G
    r = 1.0 / (damping_days * 86400.0)
    ux = cube.ddx(u); vy = cube.ddy(v)
    ex = cube.ddx(eta); ey = cube.ddy(eta)
    adv = cube.advect_muscl if scheme == "muscl" else cube.advect_upwind
    adv_u = adv(u, v, u); adv_v = adv(u, v, v); adv_eta = adv(u, v, eta)
    du = -adv_u + cube.F * v - G * ex + tx / (RHO_WATER * Hwind) - r * u + NU * cube.lap(u)
    dv = -adv_v - cube.F * u - G * ey + ty / (RHO_WATER * Hwind) - r * v + NU * cube.lap(v)
    deta = -adv_eta - He * (ux + vy)
    return du, dv, deta


def interp_time(cube, A, tsec):
    OBS = cube.OBS_SEC
    if tsec <= OBS[0]:
        return A[0]
    if tsec >= OBS[-1]:
        return A[-1]
    j = np.searchsorted(OBS, tsec)
    j0, j1 = j - 1, j
    a = (tsec - OBS[j0]) / (OBS[j1] - OBS[j0])
    return (1 - a) * A[j0] + a * A[j1]


def make_model(cube, start, Nbg, C, Hwind, damping_days, scheme):
    j0 = start - Nbg + 1
    if j0 < 0:
        raise RuntimeError("Not enough observations for background.")
    ids = np.arange(j0, start + 1)
    ebar = np.mean(cube.ETA[ids], axis=0)
    ubar = np.mean(cube.U[ids], axis=0)
    vbar = np.mean(cube.V[ids], axis=0)
    txbar = np.mean(cube.TX[ids], axis=0)
    tybar = np.mean(cube.TY[ids], axis=0)
    base = physical_rhs(cube, ubar, vbar, ebar, txbar, tybar, C, Hwind,
                        damping_days, scheme)
    q0 = (cube.U[start] - ubar, cube.V[start] - vbar, cube.ETA[start] - ebar)

    def rhs(up, vp, ep, tabs):
        u = ubar + up; v = vbar + vp; eta = ebar + ep
        tx = interp_time(cube, cube.TX, tabs)
        ty = interp_time(cube, cube.TY, tabs)
        du, dv, de = physical_rhs(cube, u, v, eta, tx, ty, C, Hwind,
                                  damping_days, scheme)
        du -= base[0]; dv -= base[1]; de -= base[2]
        du -= cube.SPONGE * up; dv -= cube.SPONGE * vp; de -= cube.SPONGE * ep
        return du, dv, de

    return q0, rhs, (ubar, vbar, ebar)


def rk4(q, t, dt, rhs):
    u, v, e = q
    k1 = rhs(u, v, e, t)
    k2 = rhs(u + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1], e + 0.5 * dt * k1[2], t + 0.5 * dt)
    k3 = rhs(u + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1], e + 0.5 * dt * k2[2], t + 0.5 * dt)
    k4 = rhs(u + dt * k3[0], v + dt * k3[1], e + dt * k3[2], t + dt)
    return (u + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            v + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
            e + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))


def integrate(q0, rhs, t0, target_times):
    q = tuple(x.copy() for x in q0)
    t = float(t0); out = {}
    for target in sorted(target_times):
        while t < target - 1e-6:
            dti = min(DT, target - t)
            q = rk4(q, t, dti, rhs)
            t += dti
            maxe = np.max(np.abs(q[2]))
            maxuv = np.max(np.sqrt(q[0] ** 2 + q[1] ** 2))
            if not np.isfinite(maxe) or not np.isfinite(maxuv) or maxe > 5.0 or maxuv > 8.0:
                raise RuntimeError(f"instability: eta={maxe:.3g}, speed={maxuv:.3g}")
        out[target] = tuple(x.copy() for x in q)
    return out


# ---- metrics ----
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def vector_rmse(u1, v1, u2, v2):
    return float(np.sqrt(np.mean((u1 - u2) ** 2 + (v1 - v2) ** 2)))


def corr(a, b):
    x = a.ravel(); y = b.ravel()
    g = np.isfinite(x) & np.isfinite(y)
    if x[g].std() == 0 or y[g].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[g], y[g])[0, 1])


def _cache_key(cube, start, params, scheme, vel_source):
    Nbg, C, Hwind, Tdamp = params
    s = (f"{cube.nt}_{cube.nx}x{cube.ny}_{start}_{Nbg}_{C}_{Hwind}_{Tdamp}_"
         f"{scheme}_{vel_source}_wind{int(getattr(cube, 'wind', True))}")
    return hashlib.md5(s.encode()).hexdigest()[:16]


def run_forecast(cube, start, params, scheme="upwind", leads=(1, 2), use_cache=True):
    """Integrate once to the requested lead indices. Returns a dict with, per
    lead: model (u,v,eta), persistence (u,v,eta)=start state, truth, and the
    PDE increment deta_pde = eta_model - eta_start (and current increments)."""
    Nbg, C, Hwind, Tdamp = params
    key = _cache_key(cube, start, params, scheme, cube.vel_source)
    cpath = os.path.join(CACHE_DIR, key + ".npz")
    if use_cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return z["data"].item()

    q0, rhs, bg = make_model(cube, start, Nbg, C, Hwind, Tdamp, scheme)
    ubar, vbar, ebar = bg
    targets = [start + L for L in leads]
    pred = integrate(q0, rhs, cube.OBS_SEC[start],
                     [cube.OBS_SEC[t] for t in targets])
    out = {"start": start, "leads": {}}
    for L, target in zip(leads, targets):
        up, vp, ep = pred[cube.OBS_SEC[target]]
        u_m = ubar + up; v_m = vbar + vp; e_m = ebar + ep
        out["leads"][L] = dict(
            lead_days=int(round((cube.OBS_SEC[target] - cube.OBS_SEC[start]) / 86400)),
            target=int(target),
            eta_model=e_m, u_model=u_m, v_model=v_m,
            eta_start=cube.ETA[start].copy(), u_start=cube.U[start].copy(),
            v_start=cube.V[start].copy(),
            eta_true=cube.ETA[target].copy(), u_true=cube.U[target].copy(),
            v_true=cube.V[target].copy(),
            deta_pde=e_m - cube.ETA[start], du_pde=u_m - cube.U[start],
            dv_pde=v_m - cube.V[start],
            deta_true=cube.ETA[target] - cube.ETA[start],
            du_true=cube.U[target] - cube.U[start],
            dv_true=cube.V[target] - cube.V[start],
        )
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez(cpath, data=np.array(out, dtype=object))
    return out


def score_window(cube, start, params, scheme="upwind"):
    """Baseline-identical scoring (ratios/skill vs persistence) for leads 7/14."""
    fc = run_forecast(cube, start, params, scheme=scheme, leads=(1, 2))
    res = []
    for L in (1, 2):
        d = fc["leads"][L]
        ssh_m = rmse(d["eta_model"], d["eta_true"])
        ssh_p = rmse(d["eta_start"], d["eta_true"])
        uv_m = vector_rmse(d["u_model"], d["v_model"], d["u_true"], d["v_true"])
        uv_p = vector_rmse(d["u_start"], d["v_start"], d["u_true"], d["v_true"])
        res.append(dict(lead_days=d["lead_days"], ssh_model=ssh_m, ssh_persistence=ssh_p,
                        ssh_ratio=ssh_m / ssh_p, ssh_skill=1 - ssh_m / ssh_p,
                        uv_model=uv_m, uv_persistence=uv_p, uv_ratio=uv_m / uv_p,
                        uv_skill=1 - uv_m / uv_p, ssh_corr=corr(d["eta_model"], d["eta_true"])))
    return res
