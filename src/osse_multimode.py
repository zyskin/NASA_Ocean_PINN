#!/usr/bin/env python3
"""
osse_multimode.py  --  Track B / B4 controlled multi-mode OSSE.

!!! SYNTHETIC -- NOT NASA DATA !!!

Builds a known-truth equatorial SSHA field by superposing several VERIFIED
analytic modes that share a common equivalent gravity-wave speed c:
    * Kelvin (eastward),
    * Rossby n=1 (westward),
    * Rossby n=2 (westward).
Stored in PHYSICAL coordinates (x,y in m, t in s) on a wide, well-sampled
equatorial domain so the wavenumber-frequency spectrum is well resolved.

Also provides degrade() to emulate satellite sampling: weekly time sampling,
0.5-deg grid coarsening, Gaussian SSH noise, random spatial gaps, and truncated
latitude range. Used by equatorial_spectrum.py, modal_decomposition.py and the
identifiability experiments.

Truth is tagged on every file; u,v,eta true fields are saved for validation.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                                       # noqa: E402
from physics import kelvin_solution, matsuno_mode, G, BETA  # noqa: E402

C_TRUE = 2.8                      # m/s (common gravity-wave speed)
DEG = 111139.0                    # m per degree

# mode table: (name, kind, n, wavelength_m, eta_amp_m)
MODES = [
    ("kelvin",   "kelvin", 0, 6.0e6, 0.10),
    ("rossby1",  "rossby", 1, 8.0e6, 0.06),
    ("rossby2",  "rossby", 2, 5.0e6, 0.04),
]


def _eval_modes(x, y, t_sec, modes, c=C_TRUE):
    """Sum selected analytic modes on a (x,y) grid for each time -> U,V,ETA."""
    LON = None
    X2 = np.meshgrid(x, y)               # (ny,nx)
    xg, yg = X2[0].ravel(), X2[1].ravel()
    xt = torch.tensor(xg, dtype=torch.float64).reshape(-1, 1)
    yt = torch.tensor(yg, dtype=torch.float64).reshape(-1, 1)
    sh = X2[0].shape
    nt = len(t_sec)
    U = np.zeros((nt, *sh)); V = np.zeros_like(U); E = np.zeros_like(U)
    for i, ts in enumerate(t_sec):
        tt = torch.full_like(xt, float(ts))
        X = torch.cat([xt, yt, tt], 1)
        us = torch.zeros_like(xt); vs = torch.zeros_like(xt); es = torch.zeros_like(xt)
        for (name, kind, n, lam, amp) in modes:
            k = 2 * math.pi / lam
            A = G * amp
            if kind == "kelvin":
                u, v, p = kelvin_solution(X, c, A, k)
            else:
                u, v, p, _ = matsuno_mode(X, c, A, k, n=n, branch="rossby")
            us += u; vs += v; es += p / G
        U[i] = us.detach().numpy().reshape(sh)
        V[i] = vs.detach().numpy().reshape(sh)
        E[i] = es.detach().numpy().reshape(sh)
    return U, V, E


def generate_clean(xmax=1.0e7, ymax=1.5e6, dx=5.0e4, dy=5.0e4,
                   days=540.0, dt_days=3.0, modes=MODES, c=C_TRUE):
    x = np.arange(-xmax, xmax + 1e-6, dx)
    y = np.arange(-ymax, ymax + 1e-6, dy)
    t = np.arange(0, days * 86400.0, dt_days * 86400.0)
    U, V, E = _eval_modes(x, y, t, modes, c)
    return dict(x=x, y=y, t=t, U=U, V=V, E=E, c=c, modes=modes)


def to_netcdf(D, path, extra=None):
    attrs = {"title": "SYNTHETIC multimode OSSE -- NOT NASA DATA", "synthetic": 1,
             "c_true_m_s": D["c"], "equivalent_depth_true_m": D["c"]**2 / G,
             "beta": BETA, "modes": ";".join(m[0] for m in D["modes"])}
    if extra:
        attrs.update(extra)
    ds = xr.Dataset(
        {"ssha": (("t", "y", "x"), D["E"].astype("float32"), {"units": "m"}),
         "u": (("t", "y", "x"), D["U"].astype("float32"), {"units": "m/s"}),
         "v": (("t", "y", "x"), D["V"].astype("float32"), {"units": "m/s"})},
        coords={"t": D["t"].astype("float64"), "y": D["y"].astype("float64"),
                "x": D["x"].astype("float64")},
        attrs=attrs)
    ds.to_netcdf(path)
    return path


def degrade(D, dt_days=7.0, coarsen=2, noise=0.01, gap_frac=0.2,
            ymax_trunc=8.9e5, seed=0):
    """Emulate satellite sampling. Returns a new dict with degraded ssha only."""
    rng = np.random.default_rng(seed)
    # temporal subsample to weekly
    dt0 = (D["t"][1] - D["t"][0]) / 86400.0
    step_t = max(1, int(round(dt_days / dt0)))
    ti = np.arange(0, len(D["t"]), step_t)
    # spatial coarsen + latitude truncation
    ymask = np.abs(D["y"]) <= ymax_trunc
    xi = np.arange(0, len(D["x"]), coarsen)
    yi = np.where(ymask)[0][::coarsen]
    E = D["E"][np.ix_(ti, yi, xi)].copy()
    U = D["U"][np.ix_(ti, yi, xi)].copy()
    V = D["V"][np.ix_(ti, yi, xi)].copy()
    # noise + random gaps
    E = E + noise * rng.standard_normal(E.shape)
    gaps = rng.random(E.shape) < gap_frac
    E[gaps] = np.nan
    return dict(x=D["x"][xi], y=D["y"][yi], t=D["t"][ti], U=U, V=V, E=E,
                c=D["c"], modes=D["modes"])


def main():
    D = generate_clean()
    to_netcdf(D, C.DATA_PROC / "osse_multimode_clean.nc")
    print(f"Clean multimode OSSE: ssha {D['E'].shape} (t,y,x)")
    print(f"  x {D['x'].min():.2e}..{D['x'].max():.2e} m ({len(D['x'])} pts, "
          f"dx={D['x'][1]-D['x'][0]:.1e})")
    print(f"  y {D['y'].min():.2e}..{D['y'].max():.2e} m ({len(D['y'])} pts)")
    print(f"  t 0..{D['t'].max()/86400:.0f} d ({len(D['t'])} pts, "
          f"dt={ (D['t'][1]-D['t'][0])/86400:.1f} d)")
    print(f"  ssha range [{np.nanmin(D['E']):.3f},{np.nanmax(D['E']):.3f}] m")
    print(f"  c_true={D['c']} m/s  h_e={D['c']**2/G:.4f} m")
    Dg = degrade(D)
    to_netcdf({**Dg}, C.DATA_PROC / "osse_multimode_degraded.nc",
              extra={"degraded": 1, "sampling": "weekly, ~1deg, 20% gaps, +1cm noise, |lat|<8deg"})
    print(f"Degraded OSSE: ssha {Dg['E'].shape}  "
          f"valid frac={np.isfinite(Dg['E']).mean():.2f}")


if __name__ == "__main__":
    main()
