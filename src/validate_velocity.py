#!/usr/bin/env python3
"""
validate_velocity.py  --  Stage 5 figures.

Compares PINN-inferred velocity (pinn_*_inferred.nc) against an independent
velocity product (surrogate_oscar.nc here; real OSCAR_L4_OC_FINAL_V2.0 has
u,v,ug,vg). Produces:
  * scatter of inferred vs reference u and v,
  * a time-mean surface-current quiver overlay,
  * a difference map of inferred-minus-reference speed.

Why total vs geostrophic (ug,vg) differ from PINN velocity:
  The linear shallow-water PINN velocity is the full (unsteady + pressure)
  balance of one baroclinic mode. Near the equator f->0, so OSCAR's midlatitude
  geostrophic (ug,vg) is replaced by a beta/ageostrophic estimate there; the
  PINN's momentum balance u_t - beta y v + Phi_x = 0 is the correct equatorial
  analogue. Expect better agreement with OSCAR TOTAL (u,v) than with (ug,vg) at
  the equator, and closer agreement with geostrophic off-equator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                # noqa: E402


def _norm(ds):
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"), ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    return ds.rename(ren) if ren else ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinn", default=str(C.DATA_PROC / "pinn_ssh_inferred.nc"))
    ap.add_argument("--oscar", default=str(C.DATA_PROC / "surrogate_oscar.nc"))
    args = ap.parse_args()

    pin = _norm(xr.open_dataset(args.pinn))
    osc = _norm(xr.open_dataset(args.oscar))

    # interpolate OSCAR onto the PINN grid/time for collocated comparison
    osc_i = osc.interp(longitude=pin.longitude, latitude=pin.latitude,
                       time=pin.time, method="linear")

    up = pin["u_pinn"].values.ravel()
    vp = pin["v_pinn"].values.ravel()
    uo = osc_i["u"].values.ravel()
    vo = osc_i["v"].values.ravel()
    m = np.isfinite(up) & np.isfinite(uo) & np.isfinite(vp) & np.isfinite(vo)
    up, vp, uo, vo = up[m], vp[m], uo[m], vo[m]

    # --- scatter ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, p, o, lab in [(ax[0], up, uo, "u"), (ax[1], vp, vo, "v")]:
        a.hexbin(o, p, gridsize=50, cmap="viridis", mincnt=1)
        lim = max(np.abs(o).max(), np.abs(p).max())
        a.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        cc = np.corrcoef(o, p)[0, 1]
        rmse = np.sqrt(np.mean((p - o) ** 2))
        a.set_xlabel(f"OSCAR {lab} [m/s]")
        a.set_ylabel(f"PINN {lab} [m/s]")
        a.set_title(f"{lab}: corr={cc:.3f}  RMSE={rmse:.3f} m/s")
    fig.suptitle("Stage 5: PINN velocity vs held-out OSCAR-like (SYNTHETIC OSSE)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "velocity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- time-mean quiver overlay ---
    um = pin["u_pinn"].mean("time")
    vm = pin["v_pinn"].mean("time")
    uom = osc_i["u"].mean("time")
    vom = osc_i["v"].mean("time")
    LON, LAT = np.meshgrid(pin.longitude.values, pin.latitude.values)
    step = 4
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.quiver(LON[::step, ::step], LAT[::step, ::step],
              uom.values[::step, ::step], vom.values[::step, ::step],
              color="k", scale=5, label="OSCAR-like (held out)")
    ax.quiver(LON[::step, ::step], LAT[::step, ::step],
              um.values[::step, ::step], vm.values[::step, ::step],
              color="r", scale=5, alpha=0.7, label="PINN inferred")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("Longitude [deg E]")
    ax.set_ylabel("Latitude [deg N]")
    ax.set_title("Time-mean surface currents: PINN (red) vs OSCAR-like (black)")
    ax.legend(loc="upper right")
    fig.savefig(C.FIGURES / "velocity_quiver.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved figures/velocity_scatter.png and figures/velocity_quiver.png")
    print(f"u corr={np.corrcoef(uo,up)[0,1]:.3f}  v corr={np.corrcoef(vo,vp)[0,1]:.3f}")


if __name__ == "__main__":
    main()
