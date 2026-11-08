#!/usr/bin/env python3
"""
validate_nasa.py  --  REAL-data velocity validation figures + wave-speed table.

Compares the PINN-inferred surface velocity (from SSH only) against the
INDEPENDENT OSCAR product (never used in training), using velocity ANOMALIES
(per-pixel temporal mean removed) because the PINN is trained on SSHA anomaly
and therefore reconstructs the anomalous wave-associated current, not the mean
SEC/EUC. Produces:
  * figures/nasa_velocity_scatter.png  -- PINN vs OSCAR-geostrophic-anom u,v
  * figures/nasa_velocity_hovmoller.png -- equatorial zonal-current anomaly
  * figures/nasa_wave_speed_comparison.png -- bar chart of independent c estimates
  * logs/nasa_wave_speed_comparison.csv -- the numeric table

OSCAR is an independent satellite-derived surface-current analysis (altimetric
geostrophy + wind-driven Ekman + buoyancy terms); it is NOT direct truth.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                  # noqa: E402


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
    return ds


def velocity_figures(pin, osc):
    osc_i = osc.interp(longitude=pin.longitude, latitude=pin.latitude,
                       time=pin.time, method="linear")
    osc_a = osc_i - osc_i.mean("time")          # OSCAR anomalies

    up = pin["u_pinn"]; vp = pin["v_pinn"]
    ug = osc_a["ug"]; vg = osc_a["vg"]

    # --- scatter vs geostrophic anomaly ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for a, p, o, lab in [(ax[0], up.values.ravel(), ug.values.ravel(), "u"),
                         (ax[1], vp.values.ravel(), vg.values.ravel(), "v")]:
        m = np.isfinite(p) & np.isfinite(o)
        p, o = p[m], o[m]
        a.hexbin(o, p, gridsize=50, cmap="viridis", mincnt=1)
        lim = max(np.abs(o).max(), np.abs(p).max())
        a.plot([-lim, lim], [-lim, lim], "r--", lw=1)
        cc = np.corrcoef(o, p)[0, 1]
        rmse = np.sqrt(np.mean((p - o) ** 2))
        a.set_xlabel(f"OSCAR geostrophic-anomaly {lab} [m/s]")
        a.set_ylabel(f"PINN {lab} [m/s]")
        a.set_title(f"{lab}: corr={cc:.3f}  RMSE={rmse:.3f} m/s")
    fig.suptitle("REAL NASA: PINN velocity (SSH-only) vs INDEPENDENT OSCAR "
                 "geostrophic anomaly")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "nasa_velocity_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- equatorial zonal-current anomaly Hovmoller: PINN vs OSCAR ---
    eq = slice(-2, 2)
    up_eq = up.sel(latitude=eq).mean("latitude")
    uo_eq = osc_a["u"].sel(latitude=eq).mean("latitude")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    vmax = float(np.nanpercentile(np.abs(uo_eq.values), 98))
    for ax, dat, title in [(axes[0], up_eq, "PINN u-anomaly (from SSH only)"),
                           (axes[1], uo_eq, "OSCAR u-anomaly (independent)")]:
        im = ax.pcolormesh(dat.longitude.values, dat.time.values, dat.values,
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Longitude [deg E]")
    fig.colorbar(im, ax=axes, label="zonal current anomaly [m/s]", shrink=0.8)
    fig.suptitle("Equatorial (2S-2N) zonal-current anomaly: PINN reconstruction "
                 "vs independent OSCAR")
    fig.savefig(C.FIGURES / "nasa_velocity_hovmoller.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/nasa_velocity_scatter.png, nasa_velocity_hovmoller.png")


def speed_table(pin):
    """Collect independent wave-speed estimates into one table + bar chart."""
    rows = []

    def grab(path, keymap):
        p = C.LOGS / path
        if not p.exists():
            return
        with open(p) as f:
            for r in csv.DictReader(f):
                for k, (label, note) in keymap.items():
                    if r.get("method") == k and r.get("speed_m_per_s"):
                        rows.append((label, float(r["speed_m_per_s"]), note))

    grab("nasa_propagation_speeds.csv", {
        "slant_stack_east": ("slant-stack (east)", "Radon eastward peak"),
        "slant_stack_west": ("slant-stack (west)", "Radon westward peak")})
    grab("nasa_komega_speeds.csv", {
        "komega_east": ("k-omega (east)", "power-weighted eastward"),
        "komega_west": ("k-omega (west)", "power-weighted westward")})

    c_pinn = float(pin.attrs.get("c_inferred_m_s", np.nan))
    rows.append(("PINN effective c", c_pinn, "single effective gravity-wave speed"))

    with open(C.LOGS / "nasa_wave_speed_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimate", "speed_m_per_s", "equivalent_depth_m", "note"])
        for label, spd, note in rows:
            he = spd**2 / C.G if spd > 0 else ""
            w.writerow([label, f"{spd:.3f}", he if he == "" else f"{he:.3f}", note])
    print("Saved logs/nasa_wave_speed_comparison.csv")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]
    colors[-1] = "#196f3d"
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="k", lw=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("phase / effective speed [m/s]  (>0 eastward, <0 westward)")
    ax.set_title("Independent equatorial-Pacific wave-speed estimates\n"
                 "(REAL NASA SSHA, 2015-2016)")
    fig.tight_layout()
    fig.savefig(C.FIGURES / "nasa_wave_speed_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/nasa_wave_speed_comparison.png")
    return rows


def main():
    pin = _norm(xr.open_dataset(C.DATA_PROC / "pinn_nasa_inferred.nc"))
    osc = _norm(xr.open_dataset(C.DATA_PROC / "nasa_oscar.nc"))
    velocity_figures(pin, osc)
    rows = speed_table(pin)
    print("\nWave-speed comparison:")
    for label, spd, note in rows:
        print(f"  {label:22s} {spd:+.3f} m/s   {note}")


if __name__ == "__main__":
    main()
