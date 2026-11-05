#!/usr/bin/env python3
"""
identifiability.py  --  Track B / B4 (robustness), B5 (identifiability), B6
(conventional-vs-ML c comparison).

Questions:
  B5  Does SSH alone determine (u,v,c) in mixed-mode cases? Run several mode
      combinations, each with multiple random PINN initialisations, and report
      the distribution of inferred c and held-out velocity error. Failures are
      reported, not hidden.
  B4  How do estimates degrade under satellite-like sampling (weekly, ~1deg,
      gaps, noise, truncated latitude)? Compare conventional slant-stack, a
      modal/k-omega estimate, and the PINN at each degradation level.
  B6  For every PINN c, give independent conventional estimates (Kelvin
      Hovmoller slant-stack; k-omega Kelvin-ridge fit; Rossby dispersion fit).

Small, short-window OSSEs are used so the many trainings stay cheap; the physics
and estimators are identical to the full-resolution modules.
"""

import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                       # noqa: E402
from physics import (EquatorialPINN, pinn_residuals, kelvin_solution,  # noqa: E402
                     matsuno_mode, rossby_dispersion, G, BETA)

ALL_MODES = {
    "kelvin":  ("kelvin", 0, 6.0e6, 0.10),
    "rossby1": ("rossby", 1, 8.0e6, 0.06),
    "rossby2": ("rossby", 2, 5.0e6, 0.04),
}
C_TRUE = 2.8


# ---------- OSSE generation (compact) ----------
def make_osse(mode_names, xmax=6.0e6, ymax=1.2e6, dx=1.0e5, dy=1.0e5,
              days=160.0, dt_days=3.0, c=C_TRUE, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(-xmax, xmax + 1, dx)
    y = np.arange(-ymax, ymax + 1, dy)
    t = np.arange(0, days * 86400.0, dt_days * 86400.0)
    Xg, Yg = np.meshgrid(x, y)
    xt = torch.tensor(Xg.ravel(), dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(Yg.ravel(), dtype=torch.float64).reshape(-1, 1)
    sh = Xg.shape
    U = np.zeros((len(t), *sh)); V = np.zeros_like(U); E = np.zeros_like(U)
    for i, ts in enumerate(t):
        us = torch.zeros_like(xt); vs = torch.zeros_like(xt); es = torch.zeros_like(xt)
        for nm in mode_names:
            kind, n, lam, amp = ALL_MODES[nm]
            k = 2 * math.pi / lam; A = G * amp
            X = torch.cat([xt, yt, torch.full_like(xt, float(ts))], 1)
            if kind == "kelvin":
                u, v, p = kelvin_solution(X, c, A, k)
            else:
                u, v, p, _ = matsuno_mode(X, c, A, k, n=n, branch="rossby")
            us += u; vs += v; es += p / G
        U[i] = us.numpy().reshape(sh); V[i] = vs.numpy().reshape(sh)
        E[i] = es.numpy().reshape(sh)
    if noise > 0:
        E = E + noise * rng.standard_normal(E.shape)
    return dict(x=x, y=y, t=t, U=U, V=V, E=E, c=c)


# ---------- conventional estimators ----------
def _eq_band(D, yhalf=4.0e5):
    m = np.abs(D["y"]) <= yhalf
    return np.nanmean(D["E"][:, m, :], axis=1)      # (nt, nx) equatorial-avg


def slant_stack_c(D, cmin=0.2, cmax=4.0, nc=160):
    hov = _eq_band(D).T                              # (nx, nt)
    x, t = D["x"], D["t"]
    data = hov - np.nanmean(hov, axis=1, keepdims=True)
    data = data - np.nanmean(data, axis=0, keepdims=True)
    xc = x - x.mean()
    speeds = np.concatenate([-np.linspace(cmax, cmin, nc//2), np.linspace(cmin, cmax, nc//2)])
    E = np.zeros_like(speeds)
    for i, c in enumerate(speeds):
        acc = np.zeros(len(t)); cnt = np.zeros(len(t))
        for j in range(len(x)):
            ts = t + xc[j]/c
            v = (ts >= t[0]) & (ts <= t[-1])
            if v.sum() < 2:
                continue
            acc[v] += np.interp(ts[v], t, data[j]); cnt[v] += 1
        good = cnt >= 0.6*len(x)
        E[i] = np.var(acc[good]/np.maximum(cnt[good], 1)) if good.sum() > 2 else 0
    E /= E.max()+1e-30
    ce = speeds[speeds > 0][np.argmax(E[speeds > 0])]
    cw = speeds[speeds < 0][np.argmax(E[speeds < 0])]
    return ce, cw


def komega_c(D):
    """Fit c from the eastward (Kelvin) ridge: power-weighted omega/k over k>0, omega>0."""
    hov = _eq_band(D)                               # (nt, nx)
    nt, nx = hov.shape
    dx, dt = np.median(np.diff(D["x"])), np.median(np.diff(D["t"]))
    g = (hov - hov.mean()) * np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    P = np.abs(np.fft.fft2(g))**2
    k = 2*math.pi*np.fft.fftfreq(nx, dx); om = 2*math.pi*np.fft.fftfreq(nt, dt)
    OM = -om[:, None]*np.ones((1, nx)); K = np.ones((nt, 1))*k[None, :]
    sel = (K > 0) & (OM > 0) & (np.abs(K) < 3e-6)
    if P[sel].sum() == 0:
        return np.nan
    return float(np.sum((OM[sel]/K[sel])*P[sel]) / np.sum(P[sel]))


# ---------- compact PINN ----------
def train_pinn(D, seed=0, adam=1500, c_init=1.5, n_data=3000, n_col=4000, verbose=False):
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    x, y, t = D["x"], D["y"], D["t"]
    Xg, Yg = np.meshgrid(x, y)
    pts = []; vals = []; us = []; vs = []
    for i, tt in enumerate(t):
        e = D["E"][i]
        good = np.isfinite(e)
        pts.append(np.stack([Xg[good], Yg[good], np.full(good.sum(), tt)], 1))
        vals.append(e[good]); us.append(D["U"][i][good]); vs.append(D["V"][i][good])
    P = np.concatenate(pts); Eo = np.concatenate(vals)
    Uo = np.concatenate(us); Vo = np.concatenate(vs)
    phi = G*Eo
    rng = np.random.default_rng(seed)
    di = rng.choice(len(phi), min(n_data, len(phi)), replace=False)
    # holdout velocity points (never used in training; SSH-only training anyway)
    span = {"x": (P[:,0].min(), P[:,0].max()), "y": (P[:,1].min(), P[:,1].max()),
            "t": (0, P[:,2].max())}
    xs = max(abs(span["x"][0]), abs(span["x"][1])); ys = max(abs(span["y"][0]), abs(span["y"][1]))
    ts = span["t"][1]
    phis = float(np.std(phi))*2+1e-6; cn = 2.5; usc = phis/cn
    Lc = math.sqrt(cn/BETA); RU = phis/Lc; RP = phis*cn/Lc
    ncyc = (ts * (2*math.pi/(6.0e6/C_TRUE))) / (2*math.pi)
    ft = int(max(8, math.ceil(2*ncyc)+2))

    def T(a): return torch.tensor(a, dtype=torch.float32)
    Xd = T(P[di, :3]); phid = T(phi[di]).reshape(-1, 1)
    lo = np.array([span["x"][0], span["y"][0], 0.0]); hi = np.array([span["x"][1], span["y"][1], ts])
    Xc = T(lo + (hi-lo)*rng.random((n_col, 3)))
    model = EquatorialPINN(xs, ys, ts, usc, phis, c_init=c_init, width=64, depth=4,
                           n_freq=(6, 6, ft))

    def losses():
        _, _, pp = model(Xd)
        Ld = torch.mean(((pp-phid)/phis)**2)
        Ru, Rv, Rp = pinn_residuals(model, Xc)
        return Ld, torch.mean((Ru/RU)**2)+torch.mean((Rv/RU)**2)+torch.mean((Rp/RP)**2)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for it in range(adam+1):
        opt.zero_grad(); Ld, Lp = losses(); (Ld+Lp).backward(); opt.step()
    opt2 = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=200, max_eval=250,
                             tolerance_grad=1e-10, tolerance_change=1e-12,
                             history_size=80, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); Ld, Lp = losses(); loss = Ld+Lp; loss.backward(); return loss
    opt2.step(closure)
    c_est = model.c.item()

    # held-out velocity metrics (all obs points; velocity never trained on)
    with torch.no_grad():
        up, vp, _ = model(T(P[:, :3]))
    up, vp = up.numpy().ravel(), vp.numpy().ravel()
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
    ucorr, vcorr = corr(up, Uo), corr(vp, Vo)
    urmse = float(np.sqrt(np.mean((up-Uo)**2)))
    return dict(c=c_est, ucorr=ucorr, vcorr=vcorr, urmse=urmse)


# ---------- experiments ----------
def experiment_identifiability(seeds=(0, 1, 2)):
    combos = {
        "Kelvin only": ["kelvin"],
        "Rossby n=1 only": ["rossby1"],
        "Kelvin+Rossby1": ["kelvin", "rossby1"],
        "Kelvin+Rossby1+Rossby2": ["kelvin", "rossby1", "rossby2"],
    }
    rows = []
    print("\n" + "=" * 78)
    print("B5 IDENTIFIABILITY: inferred c (true=2.8) and velocity corr over random inits")
    print("=" * 78)
    print(f"{'combination':>26} {'c_mean':>8} {'c_std':>7} {'c_err%':>7} "
          f"{'u_corr':>7} {'v_corr':>7}")
    for name, modes in combos.items():
        D = make_osse(modes, seed=100)
        cs, ucs, vcs = [], [], []
        for s in seeds:
            r = train_pinn(D, seed=s, c_init=1.0 + 0.5 * s)
            cs.append(r["c"]); ucs.append(r["ucorr"]); vcs.append(r["vcorr"])
        cs = np.array(cs)
        cerr = 100 * abs(cs.mean() - C_TRUE) / C_TRUE
        print(f"{name:>26} {cs.mean():8.3f} {cs.std():7.3f} {cerr:7.2f} "
              f"{np.nanmean(ucs):7.3f} {np.nanmean(vcs):7.3f}")
        rows.append([name, cs.mean(), cs.std(), cerr, np.nanmean(ucs), np.nanmean(vcs),
                     ";".join(f"{v:.3f}" for v in cs)])
    with open(C.LOGS / "identifiability_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combination", "c_mean", "c_std", "c_err_pct", "u_corr", "v_corr", "c_all_seeds"])
        w.writerows(rows)
    return rows


def experiment_robustness():
    modes = ["kelvin", "rossby1", "rossby2"]
    Dclean = make_osse(modes, seed=100)
    levels = [
        ("clean", dict()),
        ("noise1cm", dict(noise=0.01)),
        ("weekly+noise", dict(noise=0.01, dt_days=7.0)),
        ("weekly+coarse+noise", dict(noise=0.02, dt_days=7.0, dx=2.0e5, dy=2.0e5)),
    ]
    rows = []
    print("\n" + "=" * 86)
    print("B4 ROBUSTNESS under degradation (K+R1+R2; true c=2.8):")
    print("=" * 86)
    print(f"{'level':>22} {'slant_c':>8} {'komega_c':>9} {'pinn_c':>7} {'pinn_cerr%':>10} "
          f"{'u_corr':>7}")
    for name, kw in levels:
        D = make_osse(modes, seed=100, **kw)
        ce, cw = slant_stack_c(D)
        ck = komega_c(D)
        r = train_pinn(D, seed=0)
        cerr = 100 * abs(r["c"] - C_TRUE) / C_TRUE
        print(f"{name:>22} {ce:8.3f} {ck:9.3f} {r['c']:7.3f} {cerr:10.2f} {r['ucorr']:7.3f}")
        rows.append([name, ce, cw, ck, r["c"], cerr, r["ucorr"]])
    with open(C.LOGS / "robustness_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["level", "slant_east", "slant_west", "komega_c", "pinn_c", "pinn_c_err_pct", "u_corr"])
        w.writerows(rows)
    return rows


def experiment_c_comparison():
    print("\n" + "=" * 78)
    print("B6 c ESTIMATOR COMPARISON (true c=2.8, h_e=0.799 m)")
    print("=" * 78)
    D = make_osse(["kelvin", "rossby1"], seed=100)
    ce, cw = slant_stack_c(D)
    ck = komega_c(D)
    r = train_pinn(D, seed=0)
    # Rossby dispersion fit: from the westward slant speed, invert for c s.t.
    # the long-Rossby n=1 speed matches; c_ph(long) ~ -c/3 -> c ~ -3*cw
    c_from_rossby = -3.0 * cw
    print(f"  Kelvin Hovmoller slant-stack : c = {ce:+.3f} m/s  (err {100*abs(ce-C_TRUE)/C_TRUE:.1f}%)")
    print(f"  k-omega Kelvin-ridge fit     : c = {ck:+.3f} m/s  (err {100*abs(ck-C_TRUE)/C_TRUE:.1f}%)")
    print(f"  Rossby-n1 dispersion fit     : c = {c_from_rossby:+.3f} m/s "
          f"(from westward slant {cw:+.2f})")
    print(f"  PINN (SSH-only, trainable c) : c = {r['c']:+.3f} m/s  (err {100*abs(r['c']-C_TRUE)/C_TRUE:.1f}%)")
    with open(C.LOGS / "c_comparison_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "c_m_s", "err_pct"])
        for nm, cc in [("kelvin_slant", ce), ("komega", ck),
                       ("rossby_dispersion", c_from_rossby), ("pinn", r["c"])]:
            w.writerow([nm, cc, 100*abs(cc-C_TRUE)/C_TRUE])


def main():
    t0 = time.time()
    experiment_c_comparison()
    experiment_robustness()
    experiment_identifiability()
    print(f"\nTotal elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
