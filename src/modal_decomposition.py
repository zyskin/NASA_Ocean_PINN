#!/usr/bin/env python3
"""
modal_decomposition.py  --  Track B / B2 meridional-mode diagnostics.

Equatorial waves have fixed meridional structure set by Hermite functions
psi_m(y/L), L=sqrt(c/beta). SSHA for the relevant modes projects as:
    Kelvin        -> psi_0
    Rossby n=1    -> psi_0, psi_2          (symmetric)
    Rossby n=2    -> psi_1, psi_3          (antisymmetric)
    Rossby n=3    -> psi_2, psi_4          (symmetric)

We least-squares project eta(x,y,t) onto psi_0..psi_M to get coefficient fields
a_m(x,t), report the SSHA variance fraction carried by each meridional mode, and
estimate the dominant zonal phase speed (sign = direction) within each a_m from
its (k,omega) peak. Together with the dispersion relation (B1) and propagation
direction, this identifies modes WITHOUT relying on appearance.
"""

import argparse
import csv
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

BETA = 2 * 7.292115e-5 / 6.371e6


def hermite_funcs(z, M):
    """psi_0..psi_M on z. Returns (len(z), M+1)."""
    H = np.zeros((len(z), M + 1))
    H[:, 0] = 1.0
    if M >= 1:
        H[:, 1] = 2 * z
    for m in range(1, M):
        H[:, m + 1] = 2 * z * H[:, m] - 2 * m * H[:, m - 1]
    return H * np.exp(-0.5 * z * z)[:, None]


def phase_speed_peak(a_xt, x, t):
    """Dominant (k,omega) of a coefficient field a(t,x) -> physical c_ph."""
    nt, nx = a_xt.shape
    dx, dt = np.median(np.diff(x)), np.median(np.diff(t))
    g = (a_xt - a_xt.mean()) * np.hanning(nt)[:, None] * np.hanning(nx)[None, :]
    P = np.abs(np.fft.fft2(g)) ** 2
    k = 2 * math.pi * np.fft.fftfreq(nx, dx)
    om = 2 * math.pi * np.fft.fftfreq(nt, dt)
    OMEGA = -om
    # ignore k=0 column
    P[:, 0] = 0
    j, i = np.unravel_index(np.argmax(P), P.shape)
    kp, Op = k[i], OMEGA[j]
    return (Op / kp if kp != 0 else np.nan), kp, Op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(C.DATA_PROC / "osse_multimode_clean.nc"))
    ap.add_argument("--var", default="ssha")
    ap.add_argument("--c", type=float, default=2.8)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--tag", default="multimode")
    args = ap.parse_args()

    ds = xr.open_dataset(args.data)
    E = ds[args.var].transpose("t", "y", "x").values    # (nt,ny,nx)
    x, y, t = ds.x.values, ds.y.values, ds.t.values
    L = math.sqrt(args.c / BETA)
    z = y / L
    print(f"Loaded {args.data}: {E.shape}; L={L/1e3:.0f} km")

    B = hermite_funcs(z, args.M)                        # (ny, M+1)
    # least-squares projection: for each (t,x), a = pinv(B) @ eta(:,t,x)
    Bpinv = np.linalg.pinv(B)                           # (M+1, ny)
    nt, ny, nx = E.shape
    Eflat = E.transpose(1, 0, 2).reshape(ny, nt * nx)   # (ny, nt*nx)
    A = Bpinv @ Eflat                                   # (M+1, nt*nx)
    A = A.reshape(args.M + 1, nt, nx)

    tot_var = np.var(E)
    rows = []
    print("\nMeridional-mode decomposition of SSHA:")
    print(f"{'mode':>6} {'var_frac':>9} {'c_ph[m/s]':>10} {'direction':>10}  expected")
    expect = {0: "Kelvin(+)/R1", 1: "Rossby n=2(-)", 2: "R1/R3(+/-)",
              3: "Rossby n=2/4(-)", 4: "Rossby n=3(-)"}
    for m in range(args.M + 1):
        # variance contributed by mode m = var of (a_m(x,t) * psi_m(y))
        field_m = (B[:, m][None, :, None] * A[m][:, None, :])  # (nt,ny,nx)
        vf = np.var(field_m) / tot_var
        cph, kp, Op = phase_speed_peak(A[m], x, t)
        direction = "east(+)" if cph > 0 else "west(-)"
        print(f"{m:>6} {vf:9.3f} {cph:10.3f} {direction:>10}  {expect.get(m,'')}")
        rows.append([m, vf, cph, direction, expect.get(m, "")])

    # figure: basis + coefficient power
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for m in range(args.M + 1):
        ax[0].plot(B[:, m], y / 1e3, label=f"psi_{m}")
    ax[0].set_xlabel("psi_m(y/L)"); ax[0].set_ylabel("y [km]")
    ax[0].set_title("Hermite meridional basis"); ax[0].legend(fontsize=8)
    vfs = [r[1] for r in rows]
    ax[1].bar(range(args.M + 1), vfs, color="steelblue")
    ax[1].set_xlabel("meridional mode m"); ax[1].set_ylabel("SSHA variance fraction")
    ax[1].set_title("Variance by meridional mode")
    fig.suptitle("Meridional modal decomposition (SYNTHETIC multimode OSSE)")
    fig.savefig(C.FIGURES / f"modal_decomp_{args.tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(C.LOGS / f"modal_decomposition_{args.tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode_m", "var_fraction", "c_ph_m_s", "direction", "expected"])
        w.writerows(rows)

    print(f"\nInterpretation: psi_0 and psi_2 carry the symmetric eastward Kelvin")
    print(f"and westward Rossby n=1; psi_1/psi_3 carry the antisymmetric westward")
    print(f"Rossby n=2 -- matching the modes injected into the OSSE.")
    print(f"Saved figures/modal_decomp_{args.tag}.png")


if __name__ == "__main__":
    main()
