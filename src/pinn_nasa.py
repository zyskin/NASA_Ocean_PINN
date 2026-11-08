#!/usr/bin/env python3
"""
pinn_nasa.py  --  REAL-data SSH-only equatorial PINN (Stage 4/5 on NASA data).

Reuses the VERIFIED architecture and physics (physics.EquatorialPINN,
physics.pinn_residuals) unchanged, and adds the real-data machinery the
synthetic driver did not need:

  * SSHA anomaly preprocessing: the per-pixel temporal mean is removed so the
    network fits the *propagating* anomaly the linear shallow-water system
    describes. (A steady mean SSH gradient is NOT a free wave solution: with
    f = beta*y -> 0 at the equator it would drive runaway acceleration, so
    feeding the stationary mean would corrupt the effective c.)
  * A genuine HELD-OUT SSH test: entire time snapshots are withheld from the
    data loss; predicted eta at those unseen times is scored (RMSE/MAE/corr).
  * Independent velocity comparison against OSCAR TOTAL (u,v) AND GEOSTROPHIC
    (ug,vg) separately. OSCAR is never used in training.

The learned c is an EFFECTIVE single-baroclinic-mode gravity-wave speed, not a
claim that the stratified equatorial Pacific is one homogeneous layer. Real SSH
superposes Kelvin + Rossby + several vertical modes + ENSO evolution + forcing +
eddies + product error; one c summarises their dominant linear propagation.

Usage:
    python src/pinn_nasa.py --ssh data_processed/nasa_ssh.nc \
        --oscar data_processed/nasa_oscar.nc --tag nasa \
        --adam 5000 --n-data 20000 --n-col 15000
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
import config as C                                      # noqa: E402
from physics import EquatorialPINN, pinn_residuals, BETA, G   # noqa: E402

M_PER_DEG_LON = C.deg2m_lon(0.0)
M_PER_DEG_LAT = C.DEG2M_LAT


def _norm(ds):
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"),
                 ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren) if ren else ds
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    return ds.sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                  latitude=slice(C.LAT_MIN, C.LAT_MAX))


def latlon_time_to_xyt(lon, lat, tt, t0):
    x = (lon - C.LON_REF) * M_PER_DEG_LON
    y = (lat - C.LAT_REF) * M_PER_DEG_LAT
    t = (tt - t0) / np.timedelta64(1, "s")
    return x, y, t


def metrics(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    corr = (float(np.corrcoef(pred, true)[0, 1])
            if np.std(pred) > 0 and np.std(true) > 0 else float("nan"))
    return rmse, mae, corr


def build_points(da2d, t0):
    """Return stacked (x,y,t) and anomaly value arrays per time index list."""
    lonv, latv, timev = da2d.longitude.values, da2d.latitude.values, da2d.time.values
    LON, LAT = np.meshgrid(lonv, latv)
    pts, vals, tidx = [], [], []
    for i, tt in enumerate(timev):
        sl = da2d.isel(time=i).values
        good = np.isfinite(sl)
        x, y, _ = latlon_time_to_xyt(LON[good], LAT[good], tt, t0)
        tsec = np.full(good.sum(), (tt - t0) / np.timedelta64(1, "s"))
        pts.append(np.stack([x, y, tsec], axis=1))
        vals.append(sl[good])
        tidx.append(np.full(good.sum(), i))
    return (np.concatenate(pts), np.concatenate(vals), np.concatenate(tidx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", default=str(C.DATA_PROC / "nasa_ssh.nc"))
    ap.add_argument("--var", default="ssha")
    ap.add_argument("--oscar", default=str(C.DATA_PROC / "nasa_oscar.nc"))
    ap.add_argument("--start", default=None, help="optional sub-window start YYYY-MM-DD")
    ap.add_argument("--stop", default=None, help="optional sub-window stop YYYY-MM-DD")
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--lbfgs", type=int, default=300)
    ap.add_argument("--n-data", type=int, default=20000)
    ap.add_argument("--n-col", type=int, default=15000)
    ap.add_argument("--c-init", type=float, default=1.5)
    ap.add_argument("--fx", type=int, default=8)
    ap.add_argument("--fy", type=int, default=6)
    ap.add_argument("--ft", type=int, default=48)
    ap.add_argument("--holdout-every", type=int, default=5,
                    help="withhold every k-th time snapshot for held-out SSH test")
    ap.add_argument("--tag", default="nasa")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    ds = _norm(xr.open_dataset(args.ssh))
    is_synth = bool(ds.attrs.get("synthetic", 1))
    print(f"data category: {'SYNTHETIC' if is_synth else 'REAL NASA'} "
          f"({ds.attrs.get('source_product','')})")
    da = ds[args.var]
    if args.start or args.stop:
        da = da.sel(time=slice(args.start, args.stop))
    timev = da.time.values
    print(f"window {str(timev[0])[:10]} .. {str(timev[-1])[:10]}  n_time={len(timev)}")

    # --- SSHA anomaly: remove per-pixel temporal mean (propagating signal) ---
    da_anom = da - da.mean("time")
    da_anom.attrs["units"] = "m"

    t0 = timev[0]
    # held-out time snapshots (temporal interpolation test)
    n_time = len(timev)
    held_idx = set(range(0, n_time, args.holdout_every)) if args.holdout_every > 0 else set()
    held_idx.discard(0)                          # keep the first snapshot in training
    held_idx.discard(n_time - 1)                 # keep the last snapshot in training
    train_mask_times = np.array([i not in held_idx for i in range(n_time)])
    print(f"held-out snapshots: {sorted(held_idx)}  "
          f"({len(held_idx)}/{n_time} times withheld from SSH data loss)")

    Xall, eta_all, tidx_all = build_points(da_anom, t0)
    phi_all = G * eta_all
    train_pt = np.array([train_mask_times[i] for i in tidx_all])
    Xtr, phitr = Xall[train_pt], phi_all[train_pt]
    Xte, phite = Xall[~train_pt], phi_all[~train_pt]
    print(f"SSH points: train {len(phitr)}  held-out {len(phite)}  "
          f"eta_anom range [{eta_all.min():.3f},{eta_all.max():.3f}] m")

    # --- scales (from training anomaly) ---
    span = {"x": (Xall[:, 0].min(), Xall[:, 0].max()),
            "y": (Xall[:, 1].min(), Xall[:, 1].max()),
            "t": (Xall[:, 2].min(), Xall[:, 2].max())}
    x_scale = max(abs(span["x"][0]), abs(span["x"][1]))
    y_scale = max(abs(span["y"][0]), abs(span["y"][1]))
    t_scale = span["t"][1]
    phi_scale = float(np.std(G * eta_all)) * 2.0 + 1e-6
    c_nom = 2.5
    u_scale = phi_scale / c_nom
    L_char = np.sqrt(c_nom / BETA)
    RU = RV = phi_scale / L_char
    RPHI = phi_scale * c_nom / L_char
    print(f"scales: phi={phi_scale:.3f} u={u_scale:.3f} L={L_char:.3e} "
          f"RU={RU:.2e} RPHI={RPHI:.2e}")

    def T(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    idx_d = np.random.choice(len(phitr), min(args.n_data, len(phitr)), replace=False)
    Xd = T(Xtr[idx_d]); phi_d = T(phitr[idx_d]).reshape(-1, 1)
    lo = np.array([span["x"][0], span["y"][0], span["t"][0]])
    hi = np.array([span["x"][1], span["y"][1], span["t"][1]])
    Xc = T(lo + (hi - lo) * np.random.rand(args.n_col, 3))

    model = EquatorialPINN(x_scale, y_scale, t_scale, u_scale, phi_scale,
                           c_init=args.c_init, width=64, depth=4,
                           n_freq=(args.fx, args.fy, args.ft)).to(device)
    print(f"Fourier features fx={args.fx} fy={args.fy} ft={args.ft}; "
          f"params={sum(p.numel() for p in model.parameters())}")

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
    t_start = time.time()
    for it in range(args.adam + 1):
        opt.zero_grad()
        L_ssh, L1, L2, L3, L_vel = losses()
        loss = L_ssh + L1 + L2 + L3 + L_vel
        loss.backward()
        opt.step()
        if it % 500 == 0:
            row = [it, loss.item(), L_ssh.item(), L1.item(), L2.item(),
                   L3.item(), L_vel.item(), model.c.item()]
            hist.append(row)
            print(f"{it:5d} tot={row[1]:.2e} SSH={row[2]:.2e} "
                  f"pde=({L1.item():.1e},{L2.item():.1e},{L3.item():.1e}) "
                  f"vel={L_vel.item():.1e} c={model.c.item():.4f}")
    print(f"Adam elapsed {time.time()-t_start:.1f}s")

    torch.save({"model_state_dict": model.state_dict(), "c": model.c.item(),
                "scales": dict(x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
                               u_scale=u_scale, phi_scale=phi_scale),
                "stage": "pre_lbfgs"}, C.MODELS / f"pinn_{args.tag}_pre_lbfgs.pt")

    if args.lbfgs > 0:
        print("\nL-BFGS refinement:")
        opt2 = torch.optim.LBFGS(model.parameters(), lr=0.8, max_iter=args.lbfgs,
                                 max_eval=int(args.lbfgs * 1.25),
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
        print(f"L-BFGS calls={calls[0]} elapsed {time.time()-t1:.1f}s "
              f"c={model.c.item():.4f}")

    torch.save({"model_state_dict": model.state_dict(), "c": model.c.item(),
                "scales": dict(x_scale=x_scale, y_scale=y_scale, t_scale=t_scale,
                               u_scale=u_scale, phi_scale=phi_scale),
                "stage": "post_lbfgs"}, C.MODELS / f"pinn_{args.tag}_post_lbfgs.pt")

    c_est = model.c.item()
    print("\n" + "=" * 62)
    print("INFERRED EFFECTIVE PARAMETERS (real NASA SSHA)")
    print(f"  c (effective)     = {c_est:.4f} m/s")
    print(f"  equiv depth h_e   = {c_est**2/G:.4f} m   (h_e = c^2/g)")

    # --- held-out + in-sample SSH accuracy ---
    def eta_pred(Xarr):
        with torch.no_grad():
            _, _, phi = model(T(Xarr))
        return (phi.cpu().numpy().ravel()) / G
    report = {"c": c_est, "h_e": c_est**2 / G}
    tr_rmse, tr_mae, tr_corr = metrics(eta_pred(Xtr), phitr / G)
    report["train_ssh"] = (tr_rmse, tr_mae, tr_corr)
    print(f"\nin-sample SSHA : RMSE={tr_rmse:.4f} MAE={tr_mae:.4f} "
          f"corr={tr_corr:.3f} m")
    if len(phite) > 0:
        te_rmse, te_mae, te_corr = metrics(eta_pred(Xte), phite / G)
        report["heldout_ssh"] = (te_rmse, te_mae, te_corr)
        # skill vs persistence-of-zero (anomaly) baseline = std of held-out field
        base = float(np.sqrt(np.mean((phite / G) ** 2)))
        print(f"HELD-OUT SSHA  : RMSE={te_rmse:.4f} MAE={te_mae:.4f} "
              f"corr={te_corr:.3f} m   (field RMS={base:.4f}; "
              f"skill={1 - te_rmse/base:+.2f})")

    # --- velocity comparison vs OSCAR (independent; not in training) ---
    val = {}
    if Path(args.oscar).exists():
        osc = _norm(xr.open_dataset(args.oscar))
        osc_i = osc.interp(longitude=da.longitude, latitude=da.latitude,
                           time=da.time, method="linear")
        # The PINN is trained on SSHA ANOMALY, so its velocity is the anomalous
        # (wave-associated) current. Compare against OSCAR velocity ANOMALY
        # (per-pixel temporal mean removed) so the OSCAR time-mean SEC/EUC does
        # not unfairly penalise the comparison. Both total and geostrophic are
        # scored; a '_full' comparison against the un-demeaned OSCAR is also kept.
        osc_a = osc_i - osc_i.mean("time")
        lonv, latv = da.longitude.values, da.latitude.values
        LON, LAT = np.meshgrid(lonv, latv)
        comparisons = [("total_anom", ("u", "v"), osc_a),
                       ("geostrophic_anom", ("ug", "vg"), osc_a),
                       ("total_full", ("u", "v"), osc_i)]
        for comp, (uname, vname), src in comparisons:
            UP, VP, UO, VO = [], [], [], []
            for i, tt in enumerate(da.time.values):
                x, y, _ = latlon_time_to_xyt(LON.ravel(), LAT.ravel(), tt, t0)
                tsec = np.full(x.shape, (tt - t0) / np.timedelta64(1, "s"))
                with torch.no_grad():
                    up, vp, _ = model(T(np.stack([x, y, tsec], axis=1)))
                UP.append(up.cpu().numpy().ravel()); VP.append(vp.cpu().numpy().ravel())
                UO.append(src[uname].isel(time=i).values.ravel())
                VO.append(src[vname].isel(time=i).values.ravel())
            up = np.concatenate(UP); vp = np.concatenate(VP)
            uo = np.concatenate(UO); vo = np.concatenate(VO)
            m = np.isfinite(up) & np.isfinite(vp) & np.isfinite(uo) & np.isfinite(vo)
            up, vp, uo, vo = up[m], vp[m], uo[m], vo[m]
            ru, rv = metrics(up, uo), metrics(vp, vo)
            sp_p, sp_o = np.hypot(up, vp), np.hypot(uo, vo)
            spd_rmse = float(np.sqrt(np.mean((sp_p - sp_o) ** 2)))
            mask = sp_o > 0.1
            ang = np.arctan2(vp[mask], up[mask]) - np.arctan2(vo[mask], uo[mask])
            ang = (ang + np.pi) % (2 * np.pi) - np.pi
            dir_mae = float(np.degrees(np.mean(np.abs(ang))))
            val[comp] = {"u": ru, "v": rv, "speed_rmse": spd_rmse, "dir_mae": dir_mae}
            print(f"\nPINN velocity vs OSCAR {comp.upper()} (independent):")
            print(f"  u: RMSE={ru[0]:.4f} MAE={ru[1]:.4f} corr={ru[2]:.3f} m/s")
            print(f"  v: RMSE={rv[0]:.4f} MAE={rv[1]:.4f} corr={rv[2]:.3f} m/s")
            print(f"  speed RMSE={spd_rmse:.4f} m/s  dir MAE={dir_mae:.1f} deg")

    # --- save inferred fields and loss history ---
    with open(C.LOGS / f"pinn_{args.tag}_history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["adam_it", "total", "L_SSH", "L_pde1", "L_pde2", "L_pde3",
                    "L_velreg", "c_m_s"])
        w.writerows(hist)

    lonv, latv, tv = da.longitude.values, da.latitude.values, da.time.values
    LON, LAT = np.meshgrid(lonv, latv)
    U = np.empty((len(tv), len(latv), len(lonv)), np.float32)
    V = np.empty_like(U); ETA = np.empty_like(U)
    for i, tt in enumerate(tv):
        x, y, _ = latlon_time_to_xyt(LON.ravel(), LAT.ravel(), tt, t0)
        tsec = np.full(x.shape, (tt - t0) / np.timedelta64(1, "s"))
        with torch.no_grad():
            u, v, phi = model(T(np.stack([x, y, tsec], axis=1)))
        U[i] = u.cpu().numpy().reshape(LAT.shape)
        V[i] = v.cpu().numpy().reshape(LAT.shape)
        ETA[i] = (phi.cpu().numpy().reshape(LAT.shape)) / G
    out = xr.Dataset(
        {"u_pinn": (("time", "latitude", "longitude"), U, {"units": "m/s"}),
         "v_pinn": (("time", "latitude", "longitude"), V, {"units": "m/s"}),
         "eta_pinn": (("time", "latitude", "longitude"), ETA, {"units": "m"})},
        coords={"time": tv, "latitude": latv, "longitude": lonv},
        attrs={"c_inferred_m_s": c_est, "equivalent_depth_m": c_est**2 / G,
               "synthetic": 0, "source_ssh": str(args.ssh),
               "note": "PINN-inferred fields from REAL NASA SSHA anomaly "
                       "(per-pixel temporal mean removed); velocity from "
                       "SSH + equatorial shallow-water PDE; OSCAR NOT used in training"},
    )
    out.to_netcdf(C.DATA_PROC / f"pinn_{args.tag}_inferred.nc")
    print(f"\nSaved inferred fields -> {C.DATA_PROC/f'pinn_{args.tag}_inferred.nc'}")

    # machine-readable summary
    with open(C.LOGS / f"pinn_{args.tag}_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "unit", "note"])
        w.writerow(["c_effective", c_est, "m/s", "PINN inferred gravity-wave speed"])
        w.writerow(["equivalent_depth", c_est**2 / G, "m", "h_e = c^2/g"])
        w.writerow(["ssh_train_rmse", report["train_ssh"][0], "m", "in-sample"])
        w.writerow(["ssh_train_corr", report["train_ssh"][2], "", "in-sample"])
        if "heldout_ssh" in report:
            w.writerow(["ssh_heldout_rmse", report["heldout_ssh"][0], "m",
                        "held-out time snapshots"])
            w.writerow(["ssh_heldout_mae", report["heldout_ssh"][1], "m", "held-out"])
            w.writerow(["ssh_heldout_corr", report["heldout_ssh"][2], "", "held-out"])
        for comp, d in val.items():
            w.writerow([f"u_corr_vs_{comp}", d["u"][2], "", f"OSCAR {comp}"])
            w.writerow([f"v_corr_vs_{comp}", d["v"][2], "", f"OSCAR {comp}"])
            w.writerow([f"u_rmse_vs_{comp}", d["u"][0], "m/s", f"OSCAR {comp}"])
            w.writerow([f"v_rmse_vs_{comp}", d["v"][0], "m/s", f"OSCAR {comp}"])
            w.writerow([f"speed_rmse_vs_{comp}", d["speed_rmse"], "m/s", f"OSCAR {comp}"])
            w.writerow([f"dir_mae_vs_{comp}", d["dir_mae"], "deg", f"OSCAR {comp}"])
    print(f"Saved summary -> {C.LOGS/f'pinn_{args.tag}_summary.csv'}")
    print("=" * 62)
    return c_est, report, val


if __name__ == "__main__":
    main()
