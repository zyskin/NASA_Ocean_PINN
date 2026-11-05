#!/usr/bin/env python3
"""
matsuno_multimode_test.py  --  Track B / B3.

Generalise the analytic verification beyond Rossby n=1:
  * verify Rossby modes n=1,2 and representative eastward/westward
    inertia-gravity (IG) roots by autograd substitution into ALL THREE
    equatorial shallow-water PDEs (expect ~machine precision in float64);
  * build a table of  n, k, omega, phase velocity c_ph=omega/k,
    group velocity c_g=domega/dk, and max/RMS PDE residual;
  * compute c_g from the dispersion relation by finite differences in k and
    confirm each mode's residuals vanish.

Branch identity is fixed by the root ordering of the cubic
    omega^3 - c^2 (k^2 + (2n+1) beta/c) omega - c^2 beta k = 0 :
Rossby = smallest |omega|; eastward IG = largest omega; westward IG = smallest.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BETA, G                                   # noqa: E402
from physics import matsuno_mode, grad, dispersion_roots      # noqa: E402


def omega_branch(k, n, c, branch):
    roots = dispersion_roots(k, n, c)
    if branch == "rossby":
        return sorted(roots, key=abs)[0]
    if branch == "egrav":
        return max(roots)
    if branch == "wgrav":
        return min(roots)
    raise ValueError(branch)


def residuals(u, v, phi, X, c):
    gu, gv, gp = grad(u, X), grad(v, X), grad(phi, X)
    ux, ut = gu[:, 0:1], gu[:, 2:3]
    vy, vt = gv[:, 1:2], gv[:, 2:3]
    px, py, pt = gp[:, 0:1], gp[:, 1:2], gp[:, 2:3]
    y = X[:, 1:2]
    f = BETA * y
    return (ut - f * v + px, vt + f * u + py, pt + c * c * (ux + vy))


def rms(t):
    return torch.sqrt(torch.mean(t * t)).item()


def group_velocity(k, n, c, branch, dk_frac=1e-3):
    dk = dk_frac * k
    wp = omega_branch(k + dk, n, c, branch)
    wm = omega_branch(k - dk, n, c, branch)
    return (wp - wm) / (2 * dk)


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    c = 2.8
    wavelength = 6.0e6
    k = 2 * math.pi / wavelength
    xmax, ymax, tmax = 4.0e6, 1.2e6, 40 * 86400.0

    def sample(N):
        X = torch.empty(N, 3, dtype=torch.float64)
        X[:, 0].uniform_(-xmax, xmax)
        X[:, 1].uniform_(-ymax, ymax)
        X[:, 2].uniform_(0.0, tmax)
        return X.requires_grad_(True)

    cases = [(1, "rossby"), (2, "rossby"), (1, "egrav"), (1, "wgrav"),
             (2, "egrav"), (2, "wgrav")]

    print("=" * 94)
    print(f"Matsuno multimode verification (float64)   c={c} m/s  "
          f"lambda={wavelength/1e3:.0f} km  k={k:.3e} 1/m")
    print("=" * 94)
    hdr = f"{'mode':>12} {'omega[rad/s]':>14} {'c_ph[m/s]':>11} {'c_g[m/s]':>11} " \
          f"{'maxR':>10} {'rmsR':>10}  ok"
    print(hdr)
    print("-" * 94)

    rows = []
    allok = True
    for n, br in cases:
        w = omega_branch(k, n, c, br)
        X = sample(5000)
        u, v, phi, _ = matsuno_mode(X, c, 0.5, k, n=n, omega=w)
        Ru, Rv, Rp = residuals(u, v, phi, X, c)
        maxR = max(Ru.abs().max().item(), Rv.abs().max().item(), Rp.abs().max().item())
        rmsR = max(rms(Ru), rms(Rv), rms(Rp))
        cph = w / k
        cg = group_velocity(k, n, c, br)
        ok = rmsR < 1e-13
        allok &= ok
        name = f"n={n} {br}"
        print(f"{name:>12} {w:14.4e} {cph:11.4f} {cg:11.4f} "
              f"{maxR:10.2e} {rmsR:10.2e}  {ok}")
        rows.append((name, w, cph, cg, maxR, rmsR))

    print("-" * 94)
    # Physics sanity checks
    rossby1 = [r for r in rows if r[0] == "n=1 rossby"][0]
    print(f"\nPhysics checks:")
    print(f"  Rossby n=1 westward (c_ph<0): {rossby1[2] < 0}  "
          f"c_ph={rossby1[2]:.3f}, long-wave limit c/(2n+1)={c/3:.3f}")
    print(f"  Rossby n=1 group velocity {'westward' if rossby1[3] < 0 else 'eastward'} "
          f"(c_g={rossby1[3]:.3f})")
    eg1 = [r for r in rows if r[0] == "n=1 egrav"][0]
    print(f"  Eastward IG n=1 c_ph>0: {eg1[2] > 0} (c_ph={eg1[2]:.3f} m/s)")
    print(f"  equivalent depth h_e=c^2/g = {c*c/G:.4f} m")

    print("\n" + "=" * 94)
    print(f"RESULT: {'ALL PASS' if allok else 'FAIL'} "
          f"(all six modes satisfy the 3 PDEs at machine precision)")
    print("=" * 94)

    # save table
    out = Path(__file__).resolve().parent.parent / "logs" / "matsuno_multimode_table.csv"
    import csv
    with open(out, "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["mode", "omega_rad_s", "c_ph_m_s", "c_g_m_s", "maxR", "rmsR"])
        w_.writerows(rows)
    print(f"Wrote {out}")
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
