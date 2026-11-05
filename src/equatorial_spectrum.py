#!/usr/bin/env python3
"""
equatorial_spectrum.py  --  Track B / B1 wavenumber-frequency analysis.

Computes the zonal-wavenumber / frequency (k-omega) power spectrum of equatorial
SSHA and overlays the EXACT dimensional shallow-water dispersion curves for the
inferred c (Kelvin; Rossby n=1,2,3; eastward/westward inertia-gravity). This is
stronger evidence of mode identity than a Hovmoller slope alone.

Method (Wheeler & Kiladis 1999 style):
  * split SSHA into parts symmetric / antisymmetric about the equator
    (Kelvin & odd-n Rossby are symmetric; even-n Rossby & MRG antisymmetric);
  * for each latitude FFT over (x,t), take |.|^2, then average power over the
    latitude band (power averaging never cancels);
  * plot log power with theoretical curves overlaid.

Sign convention is VERIFIED at run time with a pure eastward Kelvin wave:
displayed OMEGA = -omega_fft so that eastward (c_ph>0) sits at k>0, OMEGA>0.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                  # noqa: E402
from physics import rossby_dispersion, dispersion_roots, kelvin_solution  # noqa: E402
import torch                        # noqa: E402

G = 9.81


def load_xyt(path, var="ssha"):
    ds = xr.open_dataset(path)
    if "x" in ds.coords:                       # physical-coordinate OSSE
        return ds[var].transpose("t", "y", "x").values, ds.x.values, ds.y.values, ds.t.values
    # lon/lat/time product -> convert to metres about the domain centre
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"), ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren) if ren else ds
    lon, lat = ds.longitude.values, ds.latitude.values
    x = (lon - lon.mean()) * C.deg2m_lon(0.0)
    y = (lat - lat.mean()) * C.DEG2M_LAT
    t = (ds.time.values - ds.time.values[0]) / np.timedelta64(1, "s")
    E = ds[var].transpose("time", "latitude", "longitude").values
    return E, x, y, t


def power_spectrum(field, x, t, yband_mask):
    """Average |FFT_{x,t}|^2 over selected latitudes. Returns k, OMEGA, P."""
    nt, ny, nx = field.shape
    dx = np.median(np.diff(x))
    dt = np.median(np.diff(t))
    win_t = np.hanning(nt)[:, None]
    win_x = np.hanning(nx)[None, :]
    P = np.zeros((nt, nx))
    nsel = 0
    for iy in np.where(yband_mask)[0]:
        g = field[:, iy, :]
        if not np.all(np.isfinite(g)):
            g = np.nan_to_num(g - np.nanmean(g))
        else:
            g = g - g.mean()
        g = g * win_t * win_x
        F = np.fft.fft2(g)
        P += np.abs(F) ** 2
        nsel += 1
    P /= max(nsel, 1)
    k = 2 * math.pi * np.fft.fftfreq(nx, d=dx)        # rad/m
    om = 2 * math.pi * np.fft.fftfreq(nt, d=dt)        # rad/s (fft convention)
    OMEGA = -om                                        # physical: eastward -> +
    # shift for plotting
    k_s = np.fft.fftshift(k)
    O_s = np.fft.fftshift(OMEGA)
    P_s = np.fft.fftshift(P)
    return k_s, O_s, P_s


def theoretical_curves(kpos, c):
    curves = {}
    curves["Kelvin"] = ("k", c * kpos)                       # eastward, omega=ck
    for n in (1, 2, 3):
        curves[f"Rossby n={n}"] = ("r", np.array([rossby_dispersion(kk, n, c) for kk in kpos]))
    curves["IG n=1 (E)"] = ("g", np.array([max(dispersion_roots(kk, 1, c)) for kk in kpos]))
    curves["IG n=1 (W)"] = ("m", np.array([min(dispersion_roots(kk, 1, c)) for kk in kpos]))
    return curves


def convention_selfcheck(c):
    """Pure eastward Kelvin -> peak must lie on OMEGA=+c k (OMEGA/k≈+c)."""
    x = np.arange(-1e7, 1e7, 5e4)
    t = np.arange(0, 400 * 86400.0, 3e5)
    k0 = 2 * math.pi / 6.0e6
    xt = torch.tensor(np.meshgrid(x, [0.0])[0].ravel(), dtype=torch.float64).reshape(-1, 1)
    field = np.zeros((len(t), 1, len(x)))
    for i, tt in enumerate(t):
        X = torch.cat([xt, torch.zeros_like(xt), torch.full_like(xt, tt)], 1)
        _, _, p = kelvin_solution(X, c, G * 0.1, k0)
        field[i, 0, :] = (p / G).numpy().ravel()
    ks, Os, P = power_spectrum(field, x, t, np.array([True]))
    j, i = np.unravel_index(np.argmax(P), P.shape)
    kp, Op = ks[i], Os[j]
    cph = Op / kp if kp != 0 else float("nan")
    return kp, Op, cph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_PROC / "osse_multimode_clean.nc"))
    ap.add_argument("--var", default="ssha")
    ap.add_argument("--c", type=float, default=2.8)
    ap.add_argument("--tag", default="multimode")
    args = ap.parse_args()

    # 0. convention self-check
    kp, Op, cph = convention_selfcheck(args.c)
    print(f"Convention self-check (pure eastward Kelvin, c={args.c}):")
    print(f"  spectral peak at k={kp:.3e}, OMEGA={Op:.3e} -> c_ph={cph:+.3f} m/s "
          f"(expect +{args.c})")
    assert cph > 0, "sign convention check failed (eastward should be +)"

    E, x, y, t = load_xyt(args.data, args.var)
    print(f"\nLoaded {args.data}: field {E.shape} (t,y,x)")

    sym = 0.5 * (E + E[:, ::-1, :])
    anti = 0.5 * (E - E[:, ::-1, :])
    L = math.sqrt(args.c / (2 * 7.292115e-5 / 6.371e6))  # deformation radius
    band = np.abs(y) <= 3.0 * L
    print(f"  deformation radius L={L/1e3:.0f} km; latitude band |y|<={3*L/1e3:.0f} km "
          f"({band.sum()} rows)")

    ks, Os, Psym = power_spectrum(sym, x, t, band)
    _, _, Panti = power_spectrum(anti, x, t, band)

    kpos = np.linspace(1e-7, ks.max(), 200)
    curves = theoretical_curves(kpos, args.c)

    for which, P, title in [("sym", Psym, "symmetric (Kelvin, Rossby n=1,3)"),
                            ("anti", Panti, "antisymmetric (Rossby n=2, MRG)")]:
        fig, ax = plt.subplots(figsize=(8, 7))
        # restrict to low frequencies / wavenumbers of interest
        kmax = 3e-6
        omax = 6e-6
        logP = np.log10(P + P.max() * 1e-6)
        im = ax.pcolormesh(ks, Os, logP, cmap="magma", shading="auto")
        for name, (col, w) in curves.items():
            ax.plot(kpos, w, col, lw=1.2, label=name)
            ax.plot(-kpos, -w, col, lw=1.2)        # (k,om)->(-k,-om) symmetry
        ax.set_xlim(-kmax, kmax); ax.set_ylim(-omax, omax)
        ax.axhline(0, color="w", lw=0.3); ax.axvline(0, color="w", lw=0.3)
        ax.set_xlabel("zonal wavenumber k [rad/m]  (>0 eastward)")
        ax.set_ylabel("frequency OMEGA [rad/s]  (>0 eastward)")
        ax.set_title(f"k-OMEGA power ({title})\nSYNTHETIC multimode OSSE, curves for c={args.c}")
        ax.legend(fontsize=7, loc="upper left")
        fig.colorbar(im, ax=ax, label="log10 power")
        fig.savefig(C.FIGURES / f"komega_{which}_{args.tag}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # verify spectral peaks lie near theoretical branches
    def nearest_branch(kp, Op):
        best, bd = None, 1e99
        for name, (_, w) in theoretical_curves(np.array([abs(kp)]), args.c).items():
            wt = w[0] if kp > 0 else -w[0]
            if abs(wt - Op) < bd:
                bd, best = abs(wt - Op), name
        return best, bd

    print("\nDominant symmetric peaks vs theoretical branches:")
    Pm = Psym.copy()
    kmasksel = (np.abs(ks) < 3e-6)
    for _ in range(3):
        j, i = np.unravel_index(np.argmax(Pm * kmasksel[None, :]), Pm.shape)
        kp, Op = ks[i], Os[j]
        if abs(kp) < 1e-8:
            Pm[j, i] = 0; continue
        br, bd = nearest_branch(kp, Op)
        print(f"  k={kp:+.3e} OMEGA={Op:+.3e} c_ph={Op/kp:+.2f} -> {br} (|dOMEGA|={bd:.2e})")
        # zero a neighbourhood
        Pm[max(0, j-2):j+3, max(0, i-2):i+3] = 0

    print(f"\nSaved figures/komega_sym_{args.tag}.png and komega_anti_{args.tag}.png")


if __name__ == "__main__":
    main()
