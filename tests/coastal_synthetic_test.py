#!/usr/bin/env python3
"""
coastal_synthetic_test.py  --  verify the coastal shallow-water operator.

1. Exact Poincare (inertia-gravity) plane wave on an f-plane with constant H:
   substitute into all three residuals via autograd; expect ~machine precision
   with r=0.
2. Drag term: with r>0 the exact (undamped) wave no longer satisfies momentum;
   verify the residual equals exactly r*u, r*v (confirms the drag term wiring).
3. Variable-H continuity: verify d/dx(Hu)+d/dy(Hv) is computed correctly by
   comparing the operator against an independent analytic divergence for a
   chosen smooth H(x,y) and fields u,v.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coastal_physics import (poincare_solution, residuals_from_fields,  # noqa: E402
                             coriolis, grad, G)


def rms(t):
    return torch.sqrt(torch.mean(t * t)).item()


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    phi0 = 40.0
    f = coriolis(phi0)
    H = 30.0                       # m, typical shelf depth
    A = 0.5                        # m, tidal-ish amplitude
    Lx, Ly = 1.5e5, 1.5e5          # ~150 km domain
    k = 2 * math.pi / 8.0e4        # 80 km wavelength
    l = 2 * math.pi / 1.2e5
    Tmax = 3 * 86400.0

    N = 6000
    X = torch.empty(N, 3, dtype=torch.float64)
    X[:, 0].uniform_(-Lx, Lx)
    X[:, 1].uniform_(-Ly, Ly)
    X[:, 2].uniform_(0.0, Tmax)
    X.requires_grad_(True)

    Hc = torch.full((N, 1), H, dtype=torch.float64)
    Z = torch.zeros((N, 1), dtype=torch.float64)

    u, v, eta, w = poincare_solution(X, A, k, l, H, f)

    print("=" * 64)
    print("Coastal shallow-water operator verification (float64)")
    print("=" * 64)
    print(f"phi0={phi0} deg  f={f:.3e} 1/s  H={H} m  omega={w:.3e} rad/s")
    print(f"inertial period 2pi/f = {2*math.pi/f/3600:.2f} h  "
          f"wave period 2pi/omega = {2*math.pi/w/3600:.2f} h")

    # --- 1. r=0 exact ---
    Ru, Rv, Re = residuals_from_fields(u, v, eta, X, Hc, Z, Z, f, r=0.0)
    print(f"\n[1] Poincare plane wave, r=0:")
    print(f"    Ru RMS={rms(Ru):.3e}  Rv RMS={rms(Rv):.3e}  Reta RMS={rms(Re):.3e}")
    ok1 = max(rms(Ru), rms(Rv), rms(Re)) < 1e-6 * (G * A * max(k, l))

    # --- 2. drag wiring ---
    r = 3.0e-5
    Ru2, Rv2, _ = residuals_from_fields(u, v, eta, X, Hc, Z, Z, f, r=r)
    err_u = rms(Ru2 - r * u)
    err_v = rms(Rv2 - r * v)
    print(f"\n[2] Drag term r={r:.1e}: residual - r*u RMS={err_u:.3e}, "
          f"- r*v RMS={err_v:.3e}")
    ok2 = max(err_u, err_v) < 1e-9 * rms(r * u)

    # --- 3. variable-H continuity operator ---
    # choose smooth H(x,y)=H0(1+a sin(kx x)cos(ky y)); fields u=U0 cos, v=V0 sin
    H0, a = 40.0, 0.3
    kx, ky = 2 * math.pi / 2.0e5, 2 * math.pi / 2.0e5
    xx, yy = X[:, 0:1], X[:, 1:2]
    Hf = H0 * (1 + a * torch.sin(kx * xx) * torch.cos(ky * yy))
    Hx = H0 * a * kx * torch.cos(kx * xx) * torch.cos(ky * yy)
    Hy = -H0 * a * ky * torch.sin(kx * xx) * torch.sin(ky * yy)
    U0, V0 = 0.4, 0.3
    uu = U0 * torch.cos(kx * xx)
    vv = V0 * torch.sin(ky * yy)
    # operator value via residuals_from_fields with eta=0 (et=0): Reta = div(Hu)
    zero_eta = torch.zeros_like(xx)
    _, _, div_op = residuals_from_fields(uu, vv, zero_eta, X, Hf, Hx, Hy, f, r=0.0)
    # independent analytic divergence d/dx(H u)+d/dy(H v)
    Hu_x = Hx * uu + Hf * (-U0 * kx * torch.sin(kx * xx))
    Hv_y = Hy * vv + Hf * (V0 * ky * torch.cos(ky * yy))
    div_true = Hu_x + Hv_y
    err_div = rms(div_op - div_true)
    print(f"\n[3] Variable-H continuity operator vs analytic divergence:")
    print(f"    RMS error = {err_div:.3e}  (scale {rms(div_true):.3e})")
    ok3 = err_div < 1e-9 * rms(div_true)

    print("\n" + "=" * 64)
    allok = ok1 and ok2 and ok3
    print(f"RESULT: {'ALL PASS' if allok else 'FAIL'}")
    print(f"  [1] Poincare residuals ~0 : {ok1}")
    print(f"  [2] drag term wiring      : {ok2}")
    print(f"  [3] variable-H divergence : {ok3}")
    print("=" * 64)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
