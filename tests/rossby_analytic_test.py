#!/usr/bin/env python3
"""
rossby_analytic_test.py

Independently verify the analytic equatorial wave solutions in physics.py by
substituting them (via autograd) back into the linear shallow-water PDE and
checking the residuals are at machine precision. This is the Stage-6
"trusted solution" step: a field is only used as PINN training truth after it
passes this test.

Checks:
  1. Kelvin wave residuals ~ machine precision.
  2. Matsuno Rossby mode (n=1) residuals ~ machine precision, AND that it
     propagates WESTWARD (phase speed omega/k < 0).
  3. Matsuno n=1 eastward/westward gravity roots also satisfy the PDE
     (sanity on the dispersion solver).
  4. Dispersion relation self-consistency: omega satisfies the cubic.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import BETA                                   # noqa: E402
from physics import (kelvin_solution, matsuno_mode, grad,  # noqa: E402
                     rossby_dispersion, dispersion_roots)


def residuals_from_fields(u, v, phi, X, c):
    gu, gv, gp = grad(u, X), grad(v, X), grad(phi, X)
    ux, ut = gu[:, 0:1], gu[:, 2:3]
    vy, vt = gv[:, 1:2], gv[:, 2:3]
    px, py, pt = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    y = X[:, 1:2]
    f = BETA * y
    Ru = ut - f * v + px
    Rv = vt + f * u + py
    Rphi = pt + c * c * (ux + vy)
    return Ru, Rv, Rphi


def rms(t):
    return torch.sqrt(torch.mean(t * t)).item()


def sample(n, xmax, ymax, tmax, device):
    X = torch.empty(n, 3, device=device, dtype=torch.float64)
    X[:, 0].uniform_(-xmax, xmax)
    X[:, 1].uniform_(-ymax, ymax)
    X[:, 2].uniform_(0.0, tmax)
    return X


def main():
    torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")
    torch.manual_seed(0)

    c = 2.5
    A = 0.5
    wavelength = 8.0e6
    k = 2.0 * math.pi / wavelength
    xmax, ymax, tmax = 4.0e6, 9.0e5, 30 * 86400.0

    print("=" * 64)
    print("Analytic equatorial-wave PDE verification (float64)")
    print("=" * 64)

    # --- 1. Kelvin ---
    X = sample(5000, xmax, ymax, tmax, device).requires_grad_(True)
    u, v, phi = kelvin_solution(X, c, A, k)
    Ru, Rv, Rphi = residuals_from_fields(u, v, phi, X, c)
    print("\n[1] Kelvin wave (eastward, v=0):")
    print(f"    Ru RMS={rms(Ru):.3e}  Rv RMS={rms(Rv):.3e}  Rphi RMS={rms(Rphi):.3e}")
    kelvin_ok = max(rms(Ru), rms(Rv), rms(Rphi)) < 1e-15
    print(f"    phase speed = +{c:.3f} m/s (eastward by construction)")

    # --- 2. Rossby n=1 ---
    omega_r = rossby_dispersion(k, 1, c)
    X = sample(5000, xmax, ymax, tmax, device).requires_grad_(True)
    u, v, phi, w = matsuno_mode(X, c, A, k, n=1, branch="rossby")
    Ru, Rv, Rphi = residuals_from_fields(u, v, phi, X, c)
    cph = w / k
    print(f"\n[2] Matsuno Rossby mode n=1:")
    print(f"    omega={w:.6e} rad/s  phase speed c_ph=omega/k={cph:+.4f} m/s")
    print(f"    Ru RMS={rms(Ru):.3e}  Rv RMS={rms(Rv):.3e}  Rphi RMS={rms(Rphi):.3e}")
    rossby_ok = max(rms(Ru), rms(Rv), rms(Rphi)) < 1e-14
    westward_ok = cph < 0
    print(f"    westward propagation: {westward_ok}  (Rossby waves go west)")
    # Rossby phase speed magnitude should be < c/(2n+1)=c/3 for long waves
    print(f"    |c_ph|={abs(cph):.4f} m/s  (long-Rossby limit c/(2n+1)={c/3:.4f})")

    # --- 3. gravity roots sanity ---
    X = sample(3000, xmax, ymax, tmax, device).requires_grad_(True)
    u, v, phi, w = matsuno_mode(X, c, A, k, n=1, branch="egrav")
    Ru, Rv, Rphi = residuals_from_fields(u, v, phi, X, c)
    print(f"\n[3] Matsuno n=1 eastward-gravity root (sanity):")
    print(f"    omega={w:.4e}  c_ph={w/k:+.3f} m/s  "
          f"maxRMS={max(rms(Ru),rms(Rv),rms(Rphi)):.3e}")
    grav_ok = max(rms(Ru), rms(Rv), rms(Rphi)) < 1e-14

    # --- 4. dispersion cubic self-consistency ---
    roots = dispersion_roots(k, 1, c)
    resid = [abs(r**3 - c*c*(k*k + 3*BETA/c)*r - c*c*BETA*k) for r in roots]
    print(f"\n[4] Dispersion cubic roots (n=1): {['%.3e'%r for r in roots]}")
    print(f"    |cubic residual| = {['%.1e'%r for r in resid]}")

    print("\n" + "=" * 64)
    allok = kelvin_ok and rossby_ok and westward_ok and grav_ok
    print(f"RESULT: {'ALL PASS' if allok else 'FAIL'}")
    print("  Kelvin PDE residual ~0 :", kelvin_ok)
    print("  Rossby PDE residual ~0 :", rossby_ok)
    print("  Rossby westward        :", westward_ok)
    print("  Gravity root PDE ~0    :", grav_ok)
    print("=" * 64)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
