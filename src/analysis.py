#!/usr/bin/env python3
"""
analysis.py  --  Stage 3: ordinary analysis BEFORE any neural network.

Reads a gridded SSHA NetCDF (surrogate_ssh.nc here; identical schema for real
NASA_SSH_REF_SIMPLE_GRID_V1 output) and produces:

  1. SSHA snapshot maps.
  2. Equatorial-average (2S-2N) longitude-time Hovmoller diagram.
  3. Latitude-time diagram at a central longitude.
  4. Conventional propagation-speed estimates (NO neural network):
       (a) slant-stack / Radon velocity spectrum  -- separates eastward and
           westward propagation and gives a quantitative dx/dt for each;
       (b) lagged cross-correlation of the equatorial band -- dominant speed.
  5. figures/*.png and logs/propagation_speed_estimates.csv

These conventional estimates are the non-neural baseline against which the
PINN-inferred c is compared.
"""

import argparse
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

M_PER_DEG_LON = C.deg2m_lon(0.0)


def load_ssh(path, var="ssha"):
    ds = xr.open_dataset(path)
    # normalise coord names
    ren = {}
    for a, b in [("Longitude", "longitude"), ("Latitude", "latitude"), ("Time", "time")]:
        if a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren) if ren else ds
    if float(ds.longitude.max()) > 180.0:
        lon = ((ds.longitude + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords(longitude=lon).sortby("longitude")
    da = ds[var].sel(longitude=slice(C.LON_MIN, C.LON_MAX),
                     latitude=slice(C.LAT_MIN, C.LAT_MAX))
    return ds, da


def equatorial_hovmoller(da):
    return da.sel(latitude=slice(*C.EQ_BAND)).mean("latitude", skipna=True)


def slant_stack_speed(hov, cmin=0.2, cmax=4.0, nc=160, min_cov=0.6):
    """
    Velocity spectrum via slant stacking (Radon / tau-p), WITHOUT circular
    wrap. For each trial phase speed c a coherent propagating signal
    eta(x,t)=F(x - c t) aligns under tau = t - x/c. We build

        stack(tau) = mean_j F_j(tau + x_j/c)      (linear interp in time)

    counting only samples whose shifted time falls inside the record. Energy
    E(c)=var(stack) peaks at each true phase speed. Eastward (c>0) and westward
    (c<0) branches are scanned separately so both a Kelvin and a Rossby signal
    appear as distinct peaks. Returns (speeds, energy, c_east, c_west).
    """
    lon = hov.longitude.values.astype(float)
    time = hov.time.values
    t_sec = (time - time[0]) / np.timedelta64(1, "s")
    data = hov.transpose("longitude", "time").values  # (nlon, nt)
    data = data - np.nanmean(data, axis=1, keepdims=True)   # remove time-mean per lon
    # Remove the zonal mean (kx=0, direction-less large-scale oscillation) so a
    # dominant long wave does not leak into the opposite-direction branch.
    data = data - np.nanmean(data, axis=0, keepdims=True)

    x = (lon - lon.mean()) * M_PER_DEG_LON          # m, centred
    nt = len(t_sec)
    # common reference time axis = the record's own times
    tau = t_sec

    speeds = np.concatenate([-np.linspace(cmax, cmin, nc // 2),
                             np.linspace(cmin, cmax, nc // 2)])
    energy = np.zeros_like(speeds)
    for i, c in enumerate(speeds):
        acc = np.zeros(nt)
        cnt = np.zeros(nt)
        for j in range(len(x)):
            tshift = tau + x[j] / c              # time to sample at longitude j
            valid = (tshift >= t_sec[0]) & (tshift <= t_sec[-1])
            if valid.sum() < 2:
                continue
            vals = np.interp(tshift[valid], t_sec, data[j])
            acc[valid] += vals
            cnt[valid] += 1.0
        good = cnt >= (min_cov * len(x))
        if good.sum() < 3:
            energy[i] = 0.0
            continue
        stack = acc[good] / cnt[good]
        energy[i] = np.var(stack)
    energy = energy / (energy.max() + 1e-30)

    east = speeds > 0
    west = speeds < 0
    c_east = speeds[east][np.argmax(energy[east])]
    c_west = speeds[west][np.argmax(energy[west])]
    return speeds, energy, c_east, c_west


def xcorr_speed(hov):
    """
    Dominant propagation speed from lagged cross-correlation between the
    westmost quarter and each longitude: fit lag(peak-corr) vs distance.
    Sign of slope gives direction; 1/slope gives speed.
    """
    lon = hov.longitude.values.astype(float)
    time = hov.time.values
    t_sec = (time - time[0]) / np.timedelta64(1, "s")
    dt = np.median(np.diff(t_sec))
    data = hov.transpose("longitude", "time").values
    data = data - data.mean(axis=1, keepdims=True)

    ref = len(lon) // 4
    ref_sig = data[ref]
    nt = data.shape[1]
    lags, dists = [], []
    for j in range(len(lon)):
        cc = np.correlate(data[j], ref_sig, mode="full")
        lag = (np.argmax(cc) - (nt - 1)) * dt
        lags.append(lag)
        dists.append((lon[j] - lon[ref]) * M_PER_DEG_LON)
    lags = np.array(lags)
    dists = np.array(dists)
    # robust linear fit distance = speed * lag  -> speed = slope of dist vs lag
    A = np.vstack([lags, np.ones_like(lags)]).T
    slope, _ = np.linalg.lstsq(A, dists, rcond=None)[0]
    return slope, lags, dists


def plot_maps(da, outdir):
    times = da.time.values
    idx = [0, len(times) // 3, 2 * len(times) // 3, len(times) - 1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    vmax = float(np.nanpercentile(np.abs(da.values), 99))
    for ax, i in zip(axes.ravel(), idx):
        im = da.isel(time=i).plot(ax=ax, x="longitude", y="latitude",
                                  vmin=-vmax, vmax=vmax, cmap="RdBu_r",
                                  add_colorbar=False)
        ax.set_title(str(np.datetime_as_string(times[i], unit="D")))
    fig.colorbar(im, ax=axes, label="SSHA [m]", shrink=0.8)
    fig.suptitle("SSHA snapshots (SYNTHETIC OSSE surrogate)")
    fig.savefig(outdir / "ssha_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hovmoller(hov, outdir):
    fig, ax = plt.subplots(figsize=(8, 7))
    vmax = float(np.nanpercentile(np.abs(hov.values), 99))
    hov.plot(ax=ax, x="longitude", y="time", vmin=-vmax, vmax=vmax,
             cmap="RdBu_r", cbar_kwargs={"label": "SSHA [m]"})
    ax.set_title("Equatorial (2S-2N) SSHA Hovmoller\n(SYNTHETIC OSSE surrogate)")
    ax.set_xlabel("Longitude [deg E]")
    fig.savefig(outdir / "hovmoller_equatorial.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_speed_spectrum(speeds, energy, c_east, c_west, outdir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(speeds, energy, "k-")
    ax.axvline(c_east, color="r", ls="--", label=f"east peak {c_east:+.2f} m/s")
    ax.axvline(c_west, color="b", ls="--", label=f"west peak {c_west:+.2f} m/s")
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel("trial phase speed [m/s]  (>0 eastward, <0 westward)")
    ax.set_ylabel("normalised stack energy")
    ax.set_title("Slant-stack velocity spectrum (conventional, non-neural)")
    ax.legend()
    fig.savefig(outdir / "speed_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssh", default=str(C.DATA_PROC / "surrogate_ssh.nc"))
    ap.add_argument("--var", default="ssha")
    args = ap.parse_args()

    ds, da = load_ssh(args.ssh, args.var)
    print(f"Loaded {args.ssh}")
    print(f"  dims={dict(da.sizes)}  var={args.var}  units={da.attrs.get('units','?')}")
    print(f"  lon {float(da.longitude.min()):.2f}..{float(da.longitude.max()):.2f}  "
          f"lat {float(da.latitude.min()):.2f}..{float(da.latitude.max()):.2f}")
    print(f"  time {np.datetime_as_string(da.time.values[0],unit='D')} .. "
          f"{np.datetime_as_string(da.time.values[-1],unit='D')}  "
          f"n={da.sizes['time']}")

    hov = equatorial_hovmoller(da)

    plot_maps(da, C.FIGURES)
    plot_hovmoller(hov, C.FIGURES)

    speeds, energy, c_east, c_west = slant_stack_speed(hov)
    plot_speed_spectrum(speeds, energy, c_east, c_west, C.FIGURES)
    c_xc, lags, dists = xcorr_speed(hov)

    print("\nConventional propagation-speed estimates (non-neural baseline):")
    print(f"  slant-stack eastward peak : {c_east:+.3f} m/s  "
          f"(h_e={c_east**2/C.G:.3f} m)")
    print(f"  slant-stack westward peak : {c_west:+.3f} m/s")
    print(f"  dominant cross-corr speed : {c_xc:+.3f} m/s")

    if hasattr(ds, "attrs") and ds.attrs.get("synthetic"):
        ct = ds.attrs.get("c_true_m_s")
        print(f"\n  [surrogate truth c = {ct} m/s eastward Kelvin; "
              f"Rossby westward ~ -{float(ct)/3:.2f} m/s]")

    with open(C.LOGS / "propagation_speed_estimates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "speed_m_per_s", "equivalent_depth_m", "note"])
        w.writerow(["slant_stack_east", c_east, c_east**2 / C.G, "eastward Kelvin"])
        w.writerow(["slant_stack_west", c_west, "", "westward Rossby"])
        w.writerow(["xcorr_dominant", c_xc, "", "dominant signal"])
    print(f"\nSaved figures to {C.FIGURES} and speeds to "
          f"{C.LOGS/'propagation_speed_estimates.csv'}")


if __name__ == "__main__":
    main()
