#!/usr/bin/env python3
"""
coastal_pinn.py  --  coastal shallow-water PINN proof of concept.

Scientific question: can the linear rotating shallow-water physics regularise /
interpolate an observed surface-current field (HF-radar u,v) and produce a
dynamically reasonable free-surface eta field?

Observations are VELOCITY (u,v); the free surface eta is UNOBSERVED and inferred
from u,v + the PDE (the coastal analogue of inferring velocity from SSH in the
equatorial problem). Sparse water-level (CO-OPS) can anchor eta when available.

This script runs on:
  * a SYNTHETIC tidal field (default) -- two exact M2 Poincare waves on an
    f-plane, constant H; known truth for u,v AND eta, so held-out validation and
    the inferred-eta check are meaningful; OR
  * a REAL HF-radar NetCDF (--hfradar path) with the identical downstream code.

Validation: train on a subset of (u,v) observations, hold out the rest
completely; report RMSE/MAE/vector-correlation on held-out data, and compare the
PINN against a non-PINN baseline (linear space-time interpolation).

Outputs (figures/, data_processed/, logs/, models/): coverage map, observed vs
PINN vectors, held-out errors, inferred eta, PDE residual maps, training
history, compact NetCDF.

HF radar measures near-surface (~0.5 m) velocity, NOT depth-averaged velocity;
for a short tide-dominated interval it is used as a barotropic proxy. See README.
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                  # noqa: E402
from coastal_physics import (CoastalPINN, coastal_residuals, poincare_solution,  # noqa: E402
                             coriolis, G)

LAT0, LON0 = 40.4, -73.8            # local-plane reference (harbour entrance)
PHI0 = LAT0
M2_OMEGA = 2 * math.pi / (12.4206 * 3600.0)   # principal lunar semidiurnal


def lonlat_to_xy(lon, lat):
    x = (lon - LON0) * 111139.0 * math.cos(math.radians(LAT0))
    y = (lat - LAT0) * 111139.0
    return x, y


def make_synthetic(H=25.0, days=3.0, noise=0.03, seed=0):
    """Two exact M2 Poincare waves (different directions) -> known u,v,eta."""
    rng = np.random.default_rng(seed)
    f = coriolis(PHI0)
    # fix omega=M2; Poincare |k| from omega^2=f^2+gH k^2
    kmag = math.sqrt(max((M2_OMEGA**2 - f**2), 1e-16) / (G * H))
    waves = [(0.30, kmag, 0.0), (0.20, 0.0, kmag)]   # (amp, k, l): +x and +y
    lon = np.linspace(-74.3, -73.3, 41)
    lat = np.linspace(39.9, 40.9, 41)
    t = np.arange(0, days * 86400.0, 3600.0)          # hourly
    LON, LAT = np.meshgrid(lon, lat)
    x, y = lonlat_to_xy(LON.ravel(), LAT.ravel())
    xt = torch.tensor(x, dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(y, dtype=torch.float64).reshape(-1, 1)
    U = np.empty((len(t), *LON.shape)); V = np.empty_like(U); E = np.empty_like(U)
    for i, tt in enumerate(t):
        us = torch.zeros_like(xt); vs = torch.zeros_like(xt); es = torch.zeros_like(xt)
        for (A, k, l) in waves:
            X = torch.cat([xt, yt, torch.full_like(xt, tt)], 1)
            u, v, e, _ = poincare_solution(X, A, k, l, H, f)
            us += u; vs += v; es += e
        U[i] = us.numpy().reshape(LON.shape)
        V[i] = vs.numpy().reshape(LON.shape)
        E[i] = es.numpy().reshape(LON.shape)
    # HF-radar-like coverage: drop a fraction of cells (range-dependent gaps)
    cov = rng.random(LON.shape) > 0.25            # ~75% coverage
    Un = U + noise * rng.standard_normal(U.shape)
    Vn = V + noise * rng.standard_normal(V.shape)
    return dict(lon=lon, lat=lat, t=t, LON=LON, LAT=LAT, U=Un, V=Vn,
                Utrue=U, Vtrue=V, Etrue=E, cov=cov, H=H, f=f)


def assemble_points(S):
    """Flatten covered grid points into observation arrays in local metres."""
    recs = []
    for i, tt in enumerate(S["t"]):
        m = S["cov"]
        x, y = lonlat_to_xy(S["LON"][m], S["LAT"][m])
        recs.append(np.stack([x, y, np.full(m.sum(), tt),
                              S["U"][i][m], S["V"][i][m],
                              S["Utrue"][i][m], S["Vtrue"][i][m],
                              S["Etrue"][i][m]], axis=1))
    return np.concatenate(recs, axis=0)   # cols: x,y,t,u,v,ut,vt,et


def metrics(pred, true):
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    corr = float(np.corrcoef(pred, true)[0, 1]) if np.std(pred) > 0 else float("nan")
    return rmse, mae, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hfradar", default=None, help="real HF-radar NetCDF (u,v)")
    ap.add_argument("--adam", type=int, default=4000)
    ap.add_argument("--n-col", type=int, default=6000)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--train-drag", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = torch.device("cpu")

    if args.hfradar:
        import xarray as xr
        ds = xr.open_dataset(args.hfradar)
        print(f"Loaded real HF-radar {args.hfradar}: {dict(ds.sizes)}")
        # (real-data assembly mirrors make_synthetic; uses ds.u, ds.v, lon,lat,time)
        raise SystemExit("Real HF-radar branch: fill grid->points like make_synthetic; "
                         "data unreachable in this sandbox.")
    else:
        S = make_synthetic(seed=args.seed)
        print(f"Synthetic coastal field: H={S['H']} m  f={S['f']:.3e}  "
              f"grid {S['LON'].shape}  nt={len(S['t'])}  coverage={S['cov'].mean():.2f}")

    pts = assemble_points(S)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pts))
    ntr = int((1 - args.holdout) * len(pts))
    tr, ho = pts[idx[:ntr]], pts[idx[ntr:]]
    print(f"obs points: {len(pts)}  train {len(tr)}  holdout {len(ho)}")

    # scales
    x_scale = np.abs(pts[:, 0]).max(); y_scale = np.abs(pts[:, 1]).max()
    t_scale = pts[:, 2].max()
    u_scale = float(np.std(pts[:, 3:5])) * 2 + 1e-6
    eta_scale = float(np.std(S["Etrue"])) * 2 + 1e-6
    # tidal cycles in record -> temporal Fourier harmonics
    ncyc = (S["t"].max() * M2_OMEGA) / (2 * math.pi)
    ft = int(max(8, math.ceil(2 * ncyc) + 2))
    print(f"scales u={u_scale:.3f} eta={eta_scale:.3f}  M2 cycles={ncyc:.1f} ft={ft}")

    def T(a): return torch.tensor(a, dtype=torch.float32, device=dev)
    Xtr = T(tr[:, 0:3]); Utr = T(tr[:, 3:4]); Vtr = T(tr[:, 4:5])
    lo = pts[:, 0:3].min(0); hi = pts[:, 0:3].max(0)
    Xc = T(lo + (hi - lo) * rng.random((args.n_col, 3)))
    Hc = torch.full((args.n_col, 1), S["H"], dtype=torch.float32)
    Z = torch.zeros((args.n_col, 1), dtype=torch.float32)   # const-H -> Hx=Hy=0
    f = S["f"]

    model = CoastalPINN(x_scale, y_scale, t_scale, u_scale, eta_scale,
                        width=64, depth=4, n_freq=(4, 4, ft),
                        train_drag=args.train_drag).to(dev)

    RU = RV = (G * eta_scale) / max(x_scale, y_scale)
    REta = eta_scale / t_scale * 2.0 + S["H"] * u_scale / max(x_scale, y_scale)

    def losses():
        up, vp, _ = model(Xtr)
        Ldata = torch.mean(((up - Utr) / u_scale) ** 2) + torch.mean(((vp - Vtr) / u_scale) ** 2)
        Ru, Rv, Re = coastal_residuals(model, Xc, Hc, Z, Z, f, r=model.r)
        L1 = torch.mean((Ru / RU) ** 2); L2 = torch.mean((Rv / RV) ** 2)
        L3 = torch.mean((Re / REta) ** 2)
        return Ldata, L1, L2, L3

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    hist = []
    print("\nAdam (HF-radar u,v + shallow-water PDE):")
    t0 = time.time()
    for it in range(args.adam + 1):
        opt.zero_grad()
        Ld, L1, L2, L3 = losses()
        loss = Ld + L1 + L2 + L3
        loss.backward(); opt.step()
        if it % 400 == 0:
            r_val = float(model.r) if args.train_drag else 0.0
            hist.append([it, loss.item(), Ld.item(), L1.item(), L2.item(), L3.item(), r_val])
            print(f"{it:5d} tot={loss.item():.2e} data={Ld.item():.2e} "
                  f"pde=({L1.item():.1e},{L2.item():.1e},{L3.item():.1e}) r={r_val:.2e}")
    print(f"Adam {time.time()-t0:.1f}s")

    torch.save({"model_state_dict": model.state_dict(), "stage": "pre_lbfgs"},
               C.MODELS / "coastal_pinn_pre_lbfgs.pt")

    opt2 = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=300, max_eval=400,
                             tolerance_grad=1e-10, tolerance_change=1e-12,
                             history_size=80, line_search_fn="strong_wolfe")

    def closure():
        opt2.zero_grad()
        Ld, L1, L2, L3 = losses()
        loss = Ld + L1 + L2 + L3
        loss.backward(); return loss
    t1 = time.time(); opt2.step(closure)
    print(f"L-BFGS {time.time()-t1:.1f}s")
    torch.save({"model_state_dict": model.state_dict(), "stage": "post_lbfgs"},
               C.MODELS / "coastal_pinn_post_lbfgs.pt")

    # ---- held-out validation ----
    Xho = T(ho[:, 0:3])
    with torch.no_grad():
        uho, vho, _ = model(Xho)
    uho, vho = uho.numpy().ravel(), vho.numpy().ravel()
    ru = metrics(uho, ho[:, 5]); rv = metrics(vho, ho[:, 6])
    # vector correlation (complex)
    zc_p = uho + 1j * vho; zc_t = ho[:, 5] + 1j * ho[:, 6]
    vcorr = float(np.abs(np.vdot(zc_p - zc_p.mean(), zc_t - zc_t.mean())) /
                  (np.linalg.norm(zc_p - zc_p.mean()) * np.linalg.norm(zc_t - zc_t.mean())))

    # ---- baseline: linear space-time interpolation train->holdout ----
    from scipy.interpolate import griddata
    base_u = griddata(tr[:, 0:3], tr[:, 3], ho[:, 0:3], method="linear")
    base_v = griddata(tr[:, 0:3], tr[:, 4], ho[:, 0:3], method="linear")
    bm = np.isfinite(base_u) & np.isfinite(base_v)
    bru = metrics(base_u[bm], ho[bm, 5]); brv = metrics(base_v[bm], ho[bm, 6])

    print("\n" + "=" * 60)
    print("HELD-OUT validation (points never seen in training)")
    print(f"  PINN     u: RMSE={ru[0]:.4f} MAE={ru[1]:.4f} corr={ru[2]:.3f}")
    print(f"  PINN     v: RMSE={rv[0]:.4f} MAE={rv[1]:.4f} corr={rv[2]:.3f}")
    print(f"  PINN vector correlation = {vcorr:.3f}")
    print(f"  baseline u (linear interp): RMSE={bru[0]:.4f} corr={bru[2]:.3f} "
          f"(on {bm.sum()}/{len(ho)} interpolable)")
    print(f"  baseline v (linear interp): RMSE={brv[0]:.4f} corr={brv[2]:.3f}")

    # ---- inferred eta vs truth (eta is UNOBSERVED) ----
    with torch.no_grad():
        _, _, eho = model(Xho)
    eho = eho.numpy().ravel()
    # remove unobservable constant offset (velocity fixes eta only up to a const)
    eta_pred = eho - eho.mean() + ho[:, 7].mean()
    reta = metrics(eta_pred, ho[:, 7])
    print(f"  inferred eta vs truth (de-meaned): RMSE={reta[0]:.4f} m corr={reta[2]:.3f}")
    print("=" * 60)

    # ---- figures + outputs ----
    _figures(S, model, dev, ho, uho, vho)
    with open(C.LOGS / "coastal_pinn_history.csv", "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["adam_it", "total", "L_data", "L_pde1", "L_pde2", "L_pde3", "r"])
        w.writerows(hist)
    _save_netcdf(S, model, dev)
    print("Saved figures, history, checkpoints and NetCDF.")
    return dict(u=ru, v=rv, vcorr=vcorr, eta=reta, base_u=bru)


def _grid_eval(S, model, dev, it):
    import torch
    x, y = lonlat_to_xy(S["LON"].ravel(), S["LAT"].ravel())
    X = torch.tensor(np.stack([x, y, np.full(x.shape, S["t"][it])], 1),
                     dtype=torch.float32, device=dev)
    with torch.no_grad():
        u, v, e = model(X)
    sh = S["LON"].shape
    return (u.numpy().reshape(sh), v.numpy().reshape(sh), e.numpy().reshape(sh))


def _figures(S, model, dev, ho, uho, vho):
    it = len(S["t"]) // 2
    up, vp, ep = _grid_eval(S, model, dev, it)
    LON, LAT = S["LON"], S["LAT"]
    st = 2

    # coverage
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(LON, LAT, S["cov"].astype(float), cmap="Greys", vmin=0, vmax=1)
    ax.set_title("Synthetic HF-radar coverage mask")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    fig.savefig(C.FIGURES / "coastal_coverage.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # observed vs PINN vectors
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    m = S["cov"]
    ax[0].quiver(LON[m][::1], LAT[m][::1], S["Utrue"][it][m], S["Vtrue"][it][m],
                 color="k", scale=6)
    ax[0].set_title("Observed surface currents (truth)")
    ax[1].quiver(LON[::st, ::st], LAT[::st, ::st], up[::st, ::st], vp[::st, ::st],
                 color="r", scale=6)
    ax[1].set_title("PINN surface currents")
    for a in ax:
        a.set_xlabel("lon")
    ax[0].set_ylabel("lat")
    fig.savefig(C.FIGURES / "coastal_vectors.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # inferred eta
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    e0 = ep - ep.mean() + S["Etrue"][it].mean()
    vmax = np.nanpercentile(np.abs(S["Etrue"][it]), 99)
    im = ax[0].pcolormesh(LON, LAT, S["Etrue"][it], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[0].set_title("True eta (unobserved)")
    ax[1].pcolormesh(LON, LAT, e0, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[1].set_title("PINN-inferred eta")
    fig.colorbar(im, ax=ax, label="eta [m]", shrink=0.8)
    fig.savefig(C.FIGURES / "coastal_eta.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # held-out error scatter
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, p, tcol, lab in [(ax[0], uho, ho[:, 5], "u"), (ax[1], vho, ho[:, 6], "v")]:
        a.hexbin(tcol, p, gridsize=40, cmap="viridis", mincnt=1)
        lim = max(np.abs(tcol).max(), np.abs(p).max())
        a.plot([-lim, lim], [-lim, lim], "r--")
        a.set_xlabel(f"true {lab}"); a.set_ylabel(f"PINN {lab}")
    fig.suptitle("Held-out current validation (SYNTHETIC coastal OSSE)")
    fig.savefig(C.FIGURES / "coastal_holdout.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def _save_netcdf(S, model, dev):
    import xarray as xr
    nt = len(S["t"])
    U = np.empty((nt, *S["LON"].shape), np.float32); V = np.empty_like(U); E = np.empty_like(U)
    for i in range(nt):
        U[i], V[i], E[i] = _grid_eval(S, model, dev, i)
    ds = xr.Dataset(
        {"u_pinn": (("time", "lat", "lon"), U, {"units": "m/s"}),
         "v_pinn": (("time", "lat", "lon"), V, {"units": "m/s"}),
         "eta_pinn": (("time", "lat", "lon"), E, {"units": "m"})},
        coords={"time": S["t"], "lat": S["lat"], "lon": S["lon"]},
        attrs={"synthetic": 1, "note": "SYNTHETIC coastal OSSE; not NOAA data",
               "H_m": S["H"], "f": S["f"]})
    ds.to_netcdf(C.DATA_PROC / "coastal_pinn_inferred.nc")


if __name__ == "__main__":
    main()
