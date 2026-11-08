#!/usr/bin/env python3
"""
komega.py  --  zonal wavenumber-frequency (k-omega) spectrum of equatorial SSHA.

Computes the 2-D Fourier power spectrum of the equatorial-band (2S-2N) SSHA
Hovmoller and overlays theoretical equatorial-wave dispersion branches (Kelvin
and Matsuno n=1,2 Rossby) from the shared physics module, so that observed
spectral power can be tested against physically plausible wave branches.

Conventions (documented and self-tested):
  * Plane-wave convention  eta ~ Re{ exp[i(k x - omega t)] }.
      - eastward phase propagation: c = omega/k > 0.
      - westward phase propagation: c = omega/k < 0.
  * Distance: x = (lon - lon_mean) * metres-per-degree-longitude at the equator,
    increasing eastward; dx is the median grid spacing.
  * Time: t in seconds; dt the median sampling interval (weekly => ~7 d).
  * Detrending: per-longitude linear-in-time trend removed (kills the ENSO
    warming/cooling ramp), then the zonal mean at each time removed (kills the
    k=0 basin-wide oscillation).
  * Windowing: separable Hann window in x and t to limit spectral leakage.
  * Sign: a 2-D FFT is arranged (frequency axis negated) so that a synthetic
    eastward test wave lands at (k>0, omega>0). This is asserted at run time.

The plotted spectrum is symmetric under (k,omega)->(-k,-omega) for a real field,
so the convention "eastward = positive k half, upper (omega>0)" is used: the
right half-plane (k>0) carries eastward power, the left half (k<0) westward.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C                  # noqa: E402
from physics import dispersion_roots, BETA  # noqa: E402

M_PER_DEG_LON = C.deg2m_lon(0.0)
DAY = 86400.0


def load_equatorial_hov(path, var="ssha"):
    ds = xr.open_dataset(path)
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"),
                 ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren) if ren else ds
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    da = ds[var].sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                     latitude=slice(C.LAT_MIN, C.LAT_MAX))
    hov = da.sel(latitude=slice(*C.EQ_BAND)).mean("latitude", skipna=True)
    hov = hov.transpose("time", "longitude")
    return ds, hov


def komega_spectrum(hov_vals, lon, time):
    """Return k [rad/m], omega [rad/s], power[nomega, nk] with the convention
    that eastward (c>0) power is at k>0, omega>0."""
    nt, nx = hov_vals.shape
    x = (lon - lon.mean()) * M_PER_DEG_LON
    dx = float(np.median(np.diff(x)))
    t_sec = (time - time[0]) / np.timedelta64(1, "s")
    dt = float(np.median(np.diff(t_sec)))

    data = hov_vals.astype(float).copy()
    # per-longitude linear detrend in time (remove ENSO ramp)
    tt = (t_sec - t_sec.mean())
    for j in range(nx):
        col = data[:, j]
        a1 = np.polyfit(tt, col, 1)
        data[:, j] = col - np.polyval(a1, tt)
    # remove zonal mean at each time (k=0)
    data -= data.mean(axis=1, keepdims=True)

    # separable Hann window
    wt = np.hanning(nt)[:, None]
    wx = np.hanning(nx)[None, :]
    win = wt * wx
    dw = data * win

    Z = np.fft.fft2(dw)                       # axes (time, x)
    P = (np.abs(Z) ** 2)

    k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)       # rad/m
    # negate the frequency axis so eastward (k>0) sits at omega>0 (see module docstring)
    omega = -2.0 * np.pi * np.fft.fftfreq(nt, d=dt)  # rad/s

    k = np.fft.fftshift(k)
    omega = np.fft.fftshift(omega)
    P = np.fft.fftshift(P)
    # after fftshift, omega axis follows fftshift(-freq): sort ascending to be safe
    oi = np.argsort(omega)
    ki = np.argsort(k)
    omega = omega[oi]
    k = k[ki]
    P = P[np.ix_(oi, ki)]
    return k, omega, P, dx, dt


def _selftest_convention():
    """Inject an eastward wave and confirm the peak is at k>0, omega>0."""
    nx, nt = 140, 80
    dx, dt = 55_000.0, 7 * DAY
    lon = np.linspace(-170, -100, nx)
    time = np.arange(nt) * np.timedelta64(7, "D") + np.datetime64("2015-01-01")
    x = (lon - lon.mean()) * M_PER_DEG_LON
    t = np.arange(nt) * dt
    k0 = 2 * np.pi / 6.0e6         # ~6000 km wavelength
    c0 = 2.5
    w0 = c0 * k0                   # eastward
    field = np.cos(k0 * x[None, :] - w0 * t[:, None])
    k, omega, P, _, _ = komega_spectrum(field, lon, time)
    iw, ik = np.unravel_index(np.argmax(P), P.shape)
    assert k[ik] > 0 and omega[iw] > 0, (
        f"convention self-test FAILED: peak at k={k[ik]:.2e}, omega={omega[iw]:.2e}")
    cpeak = omega[iw] / k[ik]
    return cpeak, c0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", default=str(C.DATA_PROC / "nasa_ssh.nc"))
    ap.add_argument("--var", default="ssha")
    ap.add_argument("--cvals", type=float, nargs="+", default=[2.3, 2.8],
                    help="gravity-wave speeds (m/s) for dispersion overlays")
    args = ap.parse_args()

    cpk, c0 = _selftest_convention()
    print(f"convention self-test: eastward wave recovered c={cpk:+.2f} m/s "
          f"(true {c0:+.2f}) -> OK")

    ds, hov = load_equatorial_hov(args.ssh, args.var)
    is_synth = bool(ds.attrs.get("synthetic", 1))
    cat = "SYNTHETIC OSSE" if is_synth else f"REAL NASA: {ds.attrs.get('source_product','')}"
    prefix = "" if is_synth else "nasa_"

    lon = hov.longitude.values.astype(float)
    time = hov.time.values
    k, omega, P, dx, dt = komega_spectrum(hov.values, lon, time)
    print(f"grid: nx={len(lon)} nt={len(time)}  dx={dx/1e3:.1f} km  dt={dt/DAY:.1f} d")
    print(f"Nyquist: k={np.pi/dx:.2e} rad/m (lambda={2*dx/1e3:.0f} km), "
          f"period={2*dt/DAY:.0f} d")

    # cycles-per-day and 10^-6 rad/m for display
    omega_cpd = omega / (2 * np.pi) * DAY          # cycles/day
    k_disp = k * 1e6                                # 1e-6 rad/m

    # restrict display to resolved, meaningful band
    fmax = 0.06                                     # cpd (period >= ~17 d)
    kmax = 3.0                                      # 1e-6 rad/m
    fig, ax = plt.subplots(figsize=(9, 6))
    Plog = np.log10(P / P.max() + 1e-6)
    pm = ax.pcolormesh(k_disp, omega_cpd, Plog, cmap="magma",
                       vmin=-3, vmax=0, shading="auto")
    fig.colorbar(pm, ax=ax, label="log10 normalised power")

    # dispersion overlays (positive-omega branches)
    kk = np.linspace(1e-7, kmax * 1e-6, 200)       # rad/m, k>0
    for c in args.cvals:
        wk = c * kk / (2 * np.pi) * DAY            # Kelvin omega=ck (cpd)
        ax.plot(kk * 1e6, wk, "c-", lw=1.3,
                label=f"Kelvin c={c:.1f} m/s (h_e={c**2/C.G:.2f} m)")
        # Rossby n=1 and n=2 (westward -> negative omega for k>0; plot on k<0 side
        # as the mirror eastward-k representation is standard for display)
        for n, style in [(1, "--"), (2, ":")]:
            wr = []
            for kv in kk:
                roots = dispersion_roots(kv, n, c)
                roots = sorted(roots, key=lambda w: abs(w))
                wr.append(roots[0])               # low-freq Rossby root (<0)
            wr = np.array(wr) / (2 * np.pi) * DAY
            # Rossby phase is westward: appears at k<0, omega>0 half after folding;
            # plot magnitude on negative-k side for a conventional display.
            ax.plot(-kk * 1e6, -wr, "w" + style, lw=1.1,
                    label=f"Rossby n={n} c={c:.1f}" if c == args.cvals[-1] else None)

    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-kmax, kmax)
    ax.set_ylim(0, fmax)
    ax.set_xlabel("zonal wavenumber k [10$^{-6}$ rad/m]   (k>0 eastward, k<0 westward)")
    ax.set_ylabel("frequency [cycles/day]")
    ax.set_title(f"Equatorial (2S-2N) SSHA wavenumber-frequency spectrum\n({cat})")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    fig.savefig(C.FIGURES / f"{prefix}komega_spectrum.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {C.FIGURES/(prefix+'komega_spectrum.png')}")

    # Quantify: in the eastward half (k>0), find the power-weighted mean phase
    # speed over the resolved band, as a k-omega Kelvin-speed estimate.
    kg, og = np.meshgrid(k, omega)
    band = (og > 0) & (kg > 0.2e-6) & (np.abs(og) < 2 * np.pi * fmax / DAY) \
        & (kg < kmax * 1e-6)
    w = P[band]
    cph = og[band] / kg[band]
    # keep physically plausible gravity-wave speeds
    sel = (cph > 0.5) & (cph < 4.0)
    c_komega_east = float(np.sum(w[sel] * cph[sel]) / np.sum(w[sel]))
    # westward half
    bandw = (og > 0) & (kg < -0.2e-6) & (np.abs(og) < 2 * np.pi * fmax / DAY) \
        & (kg > -kmax * 1e-6)
    ww = P[bandw]
    cphw = og[bandw] / kg[bandw]
    selw = (cphw < -0.3) & (cphw > -3.0)
    c_komega_west = float(np.sum(ww[selw] * cphw[selw]) / np.sum(ww[selw]))
    print(f"k-omega power-weighted phase speed: east {c_komega_east:+.2f} m/s "
          f"(h_e={c_komega_east**2/C.G:.3f} m), west {c_komega_west:+.2f} m/s")

    with open(C.LOGS / f"{prefix}komega_speeds.csv", "w") as f:
        f.write("method,speed_m_per_s,equivalent_depth_m,note\n")
        f.write(f"komega_east,{c_komega_east},{c_komega_east**2/C.G},"
                "power-weighted eastward phase speed\n")
        f.write(f"komega_west,{c_komega_west},,power-weighted westward phase speed\n")
    print(f"Saved {C.LOGS/(prefix+'komega_speeds.csv')}")


if __name__ == "__main__":
    main()
