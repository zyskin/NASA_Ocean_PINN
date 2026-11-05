#!/usr/bin/env python3
"""
pinn_ssh.py  --  Stage 4: real-data-style SSH PINN  (+ Stage 5 velocity check)

Trains EquatorialPINN from gridded SSH anomaly ONLY (Phi = g*eta), with the
equatorial shallow-water PDE residuals and c>0 as a trainable parameter. No u
or v labels are used in training. The same code runs on the synthetic OSSE
surrogate (default) and on real NASA_SSH_REF_SIMPLE_GRID_V1 output (identical
schema) -- pass --ssh path.

Design choices for real data:
  * All PDE derivatives are in physical SI coordinates (metres, seconds),
    so autograd returns correct dimensional metric factors. Input
    normalisation happens inside the model only.
  * Residual conditioning scales are derived from the data and the equatorial
    deformation radius L=sqrt(c_nom/beta), not hand-tuned per dataset:
        RU=RV = phi_scale / L ,  RPHI = phi_scale * c_nom / L .
  * Each loss is logged separately: L_SSH, L_pde1, L_pde2, L_pde3, L_velreg.
  * Checkpoints saved before and after L-BFGS.

Stage 5: inferred (u,v) are compared to held-out OSCAR-like velocities
(surrogate_oscar.nc; real OSCAR_L4_OC_FINAL_V2.0 has u,v,ug,vg). OSCAR is NOT
used in training, so this is genuine independent validation.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                     # noqa: E402
from physics import EquatorialPINN, pinn_residuals, BETA, G  # noqa: E402

M_PER_DEG_LON = C.deg2m_lon(0.0)
M_PER_DEG_LAT = C.DEG2M_LAT


def _norm_coords(ds):
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"), ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    return ds.rename(ren) if ren else ds


def latlon_time_to_xyt(lon, lat, time, t0):
    x = (lon - C.LON_REF) * M_PER_DEG_LON
    y = (lat - C.LAT_REF) * M_PER_DEG_LAT
    t = (time - t0) / np.timedelta64(1, "s")
    return x, y, t


def load_ssh_points(path, var="ssha"):
    ds = _norm_coords(xr.open_dataset(path))
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    da = ds[var].sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                     latitude=slice(C.LAT_MIN, C.LAT_MAX))
    LON, LAT = np.meshgrid(da.longitude.values, da.latitude.values)
    t0 = da.time.values[0]
    pts = []
    vals = []
    for i, tt in enumerate(da.time.values):
        sl = da.isel(time=i).values            # (nlat, nlon)
        good = np.isfinite(sl)
        x, y, _ = latlon_time_to_xyt(LON[good], LAT[good], tt, t0)
        tsec = np.full(good.sum(), (tt - t0) / np.timedelta64(1, "s"))
        pts.append(np.stack([x, y, tsec], axis=1))
        vals.append(sl[good])
    X = np.concatenate(pts, axis=0)
    eta = np.concatenate(vals, axis=0)
    span = {
        "x": (X[:, 0].min(), X[:, 0].max()),
        "y": (X[:, 1].min(), X[:, 1].max()),
        "t": (X[:, 2].min(), X[:, 2].max()),
        "t0": t0,
        "synthetic": ds.attrs.get("synthetic", 0),
        "c_true": ds.attrs.get("c_true_m_s", None),
    }
    return X, eta, span


def load_vel_points(path, t0):
    ds = _norm_coords(xr.open_dataset(path))
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    ds = ds.sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                latitude=slice(C.LAT_MIN, C.LAT_MAX))
    LON, LAT = np.meshgrid(ds.longitude.values, ds.latitude.values)
    out = {"X": [], "u": [], "v": []}
    for i, tt in enumerate(ds.time.values):
        u = ds["u"].isel(time=i).values
        v = ds["v"].isel(time=i).values
        good = np.isfinite(u) & np.isfinite(v)
        x, y, _ = latlon_time_to_xyt(LON[good], LAT[good], tt, t0)
        tsec = np.full(good.sum(), (tt - t0) / np.timedelta64(1, "s"))
        out["X"].append(np.stack([x, y, tsec], axis=1))
        out["u"].append(u[good])
        out["v"].append(v[good])
    return (np.concatenate(out["X"]), np.concatenate(out["u"]),
            np.concatenate(out["v"]))


def metrics(pred, true):
    pred, true = np.asarray(pred), np.asarray(true)
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    if np.std(pred) > 0 and np.std(true) > 0:
        corr = float(np.corrcoef(pred, true)[0, 1])
    else:
        corr = float("nan")
    return rmse, mae, corr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", default=str(C.DATA_PROC / "surrogate_ssh.nc"))
    ap.add_argument("--var", default="ssha")
    ap.add_argument("--oscar", default=str(C.DATA_PROC / "surrogate_oscar.nc"))
    ap.add_argument("--truth", default=str(C.DATA_PROC / "surrogate_truth.nc"))
    ap.add_argument("--adam", type=int, default=4000)
    ap.add_argument("--n-data", type=int, default=5000)
    ap.add_argument("--n-col", type=int, default=7000)
    ap.add_argument("--c-init", type=float, default=1.5)
    ap.add_argument("--fx", type=int, default=8)
    ap.add_argument("--fy", type=int, default=6)
    ap.add_argument("--ft", type=int, default=48,
                    help="temporal Fourier harmonics (>= 2x number of cycles in record)")
    ap.add_argument("--tag", default="ssh")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    # ---- load SSH observations ----
    Xall, eta_all, span = load_ssh_points(args.ssh, args.var)
    phi_all = G * eta_all
    print(f"SSH points: {len(eta_all)}  eta range [{eta_all.min():.3f},{eta_all.max():.3f}] m")
    print(f"x span {span['x'][0]:.2e}..{span['x'][1]:.2e} m | "
          f"y {span['y'][0]:.2e}..{span['y'][1]:.2e} | "
          f"t 0..{span['t'][1]:.2e} s")

    # ---- scales ----
    x_scale = max(abs(span["x"][0]), abs(span["x"][1]))
    y_scale = max(abs(span["y"][0]), abs(span["y"][1]))
    t_scale = span["t"][1]
    phi_scale = float(np.std(phi_all)) * 2.0 + 1e-6
    c_nom = 2.5
    u_scale = phi_scale / c_nom
    L_char = np.sqrt(c_nom / BETA)
    RU = RV = phi_scale / L_char
    RPHI = phi_scale * c_nom / L_char
    print(f"scales: phi={phi_scale:.3f} u={u_scale:.3f} L={L_char:.3e} "
          f"RU={RU:.2e} RPHI={RPHI:.2e}")

    # ---- tensors ----
    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    idx_d = np.random.choice(len(eta_all), min(args.n_data, len(eta_all)), replace=False)
    Xd = T(Xall[idx_d])
    phi_d = T(phi_all[idx_d]).reshape(-1, 1)

    # collocation points: random within the data span
    lo = np.array([span["x"][0], span["y"][0], span["t"][0]])
    hi = np.array([span["x"][1], span["y"][1], span["t"][1]])
    Xc = T(lo + (hi - lo) * np.random.rand(args.n_col, 3))

    model = EquatorialPINN(x_scale, y_scale, t_scale, u_scale, phi_scale,
                           c_init=args.c_init, width=64, depth=4,
                           n_freq=(args.fx, args.fy, args.ft)).to(device)
    print(f"Fourier features: fx={args.fx} fy={args.fy} ft={args.ft}")

    def losses():
        _, _, phi_pred = model(Xd)
        L_ssh = torch.mean(((phi_pred - phi_d) / phi_scale) ** 2)
        Ru, Rv, Rphi = pinn_residuals(model, Xc)
        L1 = torch.mean((Ru / RU) ** 2)
        L2 = torch.mean((Rv / RV) ** 2)
        L3 = torch.mean((Rphi / RPHI) ** 2)
        u_c, v_c, _ = model(Xc)
        L_vel = 1e-4 * (torch.mean((u_c / u_scale) ** 2)
                        + torch.mean((v_c / u_scale) ** 2))
        return L_ssh, L1, L2, L3, L_vel

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    hist = []
    print("\nAdam (SSH observations only):")
    t0 = time.time()
    for it in range(args.adam + 1):
        opt.zero_grad()
        L_ssh, L1, L2, L3, L_vel = losses()
        loss = L_ssh + L1 + L2 + L3 + L_vel
        loss.backward()
        opt.step()
        if it % 300 == 0:
            row = [it, loss.item(), L_ssh.item(), L1.item(), L2.item(),
                   L3.item(), L_vel.item(), model.c.item()]
            hist.append(row)
            print(f"{it:5d} tot={row[1]:.2e} SSH={row[2]:.2e} "
                  f"pde=({L1.item():.2e},{L2.item():.2e},{L3.item():.2e}) "
                  f"vel={L_vel.item():.1e} c={model.c.item():.4f}")
    print(f"Adam elapsed {time.time()-t0:.1f}s")

    torch.save({"model_state_dict": model.state_dict(), "c": model.c.item(),
                "scales": dict(x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
                               u_scale=u_scale, phi_scale=phi_scale),
                "stage": "pre_lbfgs"},
               C.MODELS / f"pinn_{args.tag}_pre_lbfgs.pt")

    print("\nL-BFGS refinement:")
    opt2 = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=500, max_eval=600,
                             tolerance_grad=1e-10, tolerance_change=1e-12,
                             history_size=100, line_search_fn="strong_wolfe")
    calls = [0]

    def closure():
        opt2.zero_grad()
        L_ssh, L1, L2, L3, L_vel = losses()
        loss = L_ssh + L1 + L2 + L3 + L_vel
        loss.backward()
        calls[0] += 1
        return loss

    t1 = time.time()
    opt2.step(closure)
    print(f"L-BFGS calls={calls[0]} elapsed {time.time()-t1:.1f}s  c={model.c.item():.4f}")

    torch.save({"model_state_dict": model.state_dict(), "c": model.c.item(),
                "scales": dict(x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
                               u_scale=u_scale, phi_scale=phi_scale),
                "stage": "post_lbfgs"},
               C.MODELS / f"pinn_{args.tag}_post_lbfgs.pt")

    c_est = model.c.item()
    print("\n" + "=" * 60)
    print("INFERRED PARAMETERS")
    print(f"  c inferred        = {c_est:.4f} m/s")
    print(f"  equiv depth h_e   = {c_est**2/G:.4f} m")
    if span["c_true"] is not None:
        ct = float(span["c_true"])
        print(f"  c true (surrogate)= {ct:.4f} m/s   error {100*abs(c_est-ct)/ct:.2f}%")
        print(f"  h_e true          = {ct**2/G:.4f} m")

    # ---- Stage 5: velocity validation vs held-out OSCAR-like ----
    with open(C.LOGS / f"pinn_{args.tag}_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adam_it", "total", "L_SSH", "L_pde1", "L_pde2", "L_pde3",
                    "L_velreg", "c_m_s"])
        w.writerows(hist)

    val_report = {}
    if Path(args.oscar).exists():
        Xo, uo, vo = load_vel_points(args.oscar, span["t0"])
        with torch.no_grad():
            up, vp, _ = model(T(Xo))
        up, vp = up.cpu().numpy().ravel(), vp.cpu().numpy().ravel()
        ru = metrics(up, uo)
        rv = metrics(vp, vo)
        sp_p = np.sqrt(up**2 + vp**2)
        sp_o = np.sqrt(uo**2 + vo**2)
        spd_rmse = float(np.sqrt(np.mean((sp_p - sp_o) ** 2)))
        # direction error where speed is meaningful
        mask = sp_o > 0.05
        ang = np.arctan2(vp[mask], up[mask]) - np.arctan2(vo[mask], uo[mask])
        ang = (ang + np.pi) % (2 * np.pi) - np.pi
        dir_mae = float(np.degrees(np.mean(np.abs(ang))))
        val_report = {"u": ru, "v": rv, "speed_rmse": spd_rmse, "dir_mae_deg": dir_mae}
        print("\nStage 5: inferred velocity vs HELD-OUT OSCAR-like (not in training)")
        print(f"  u: RMSE={ru[0]:.4f} MAE={ru[1]:.4f} corr={ru[2]:.3f} m/s")
        print(f"  v: RMSE={rv[0]:.4f} MAE={rv[1]:.4f} corr={rv[2]:.3f} m/s")
        print(f"  speed RMSE={spd_rmse:.4f} m/s | direction MAE={dir_mae:.1f} deg")

    # ---- save inferred fields on the SSH grid as NetCDF ----
    ds = _norm_coords(xr.open_dataset(args.ssh))
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    ds = ds.sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                latitude=slice(C.LAT_MIN, C.LAT_MAX))
    lonv, latv, timev = ds.longitude.values, ds.latitude.values, ds.time.values
    LON, LAT = np.meshgrid(lonv, latv)
    U = np.empty((len(timev), len(latv), len(lonv)), np.float32)
    V = np.empty_like(U)
    PHI = np.empty_like(U)
    for i, tt in enumerate(timev):
        x, y, _ = latlon_time_to_xyt(LON.ravel(), LAT.ravel(), tt, span["t0"])
        tsec = np.full(x.shape, (tt - span["t0"]) / np.timedelta64(1, "s"))
        Xg = T(np.stack([x, y, tsec], axis=1))
        with torch.no_grad():
            u, v, phi = model(Xg)
        U[i] = u.cpu().numpy().reshape(LAT.shape)
        V[i] = v.cpu().numpy().reshape(LAT.shape)
        PHI[i] = phi.cpu().numpy().reshape(LAT.shape)
    out = xr.Dataset(
        {"u_pinn": (("time", "latitude", "longitude"), U, {"units": "m/s"}),
         "v_pinn": (("time", "latitude", "longitude"), V, {"units": "m/s"}),
         "eta_pinn": (("time", "latitude", "longitude"), (PHI / G).astype(np.float32),
                      {"units": "m"})},
        coords={"time": timev, "latitude": latv, "longitude": lonv},
        attrs={"c_inferred_m_s": c_est, "equivalent_depth_m": c_est**2 / G,
               "source_ssh": str(args.ssh),
               "note": "PINN-inferred fields; velocity from SSH+shallow-water PDE"},
    )
    out.to_netcdf(C.DATA_PROC / f"pinn_{args.tag}_inferred.nc")
    print(f"\nSaved inferred fields -> {C.DATA_PROC/f'pinn_{args.tag}_inferred.nc'}")
    print(f"Saved checkpoints -> models/pinn_{args.tag}_pre/post_lbfgs.pt")
    print("=" * 60)
    return c_est, val_report


if __name__ == "__main__":
    main()
