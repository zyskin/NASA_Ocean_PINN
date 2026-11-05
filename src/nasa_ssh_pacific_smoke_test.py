#!/usr/bin/env python3
"""
nasa_ssh_pacific_smoke_test.py

Minimal NASA Earthdata -> xarray test for the first real-data PINN experiment.

Product:
    NASA_SSH_REF_SIMPLE_GRID_V1
Variable:
    ssha   [m]

Region:
    tropical Pacific, 170 W to 100 W, 8 S to 8 N

The script:
  1. logs into NASA Earthdata,
  2. searches only a short time window,
  3. prints granule metadata before downloading,
  4. downloads at most three granules,
  5. opens them with xarray,
  6. makes a small equatorial Hovmoller plot.

Requirements:
    pip install -U earthaccess xarray netCDF4 dask matplotlib numpy
"""

from pathlib import Path

import earthaccess
import matplotlib.pyplot as plt
import xarray as xr


SHORT_NAME = "NASA_SSH_REF_SIMPLE_GRID_V1"
VARIABLE = "ssha"

# CMR bounding-box order:
# (west longitude, south latitude, east longitude, north latitude)
BBOX = (-170.0, -8.0, -100.0, 8.0)

# Deliberately short smoke-test interval.
START = "2023-11-01"
STOP = "2023-12-31"

OUTDIR = Path("data_raw/nasa_ssh_test")
OUTDIR.mkdir(parents=True, exist_ok=True)


def main():
    # This may open a browser or use already configured Earthdata credentials.
    earthaccess.login()

    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        bounding_box=BBOX,
        temporal=(START, STOP),
        count=20,
    )

    print(f"\nCollection: {SHORT_NAME}")
    print(f"Requested region: {BBOX}")
    print(f"Requested interval: {START} to {STOP}")
    print(f"Matching granules returned: {len(results)}")

    if not results:
        raise RuntimeError("No matching granules found.")

    print("\nFirst few granules:")
    for i, result in enumerate(results[:5]):
        print(f"\n--- granule {i} ---")
        print(result)

    # Safety rule: never bulk-download before a tiny test succeeds.
    n_download = min(3, len(results))
    print(f"\nDownloading only {n_download} granule(s) for the smoke test.")

    files = earthaccess.download(
        results[:n_download],
        local_path=str(OUTDIR),
    )

    print("\nDownloaded files:")
    for f in files:
        print(f)

    nc_files = [str(Path(f)) for f in files if str(f).endswith(".nc")]
    if not nc_files:
        nc_files = [str(p) for p in OUTDIR.glob("*.nc")]

    if not nc_files:
        raise RuntimeError("No NetCDF files found after download.")

    # Engine fallback: netcdf4 then h5netcdf (some PO.DAAC files need h5netcdf).
    ds = None
    for engine in ("netcdf4", "h5netcdf"):
        try:
            ds = xr.open_mfdataset(nc_files, combine="by_coords",
                                   decode_times=True, engine=engine)
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    if ds is None:
        raise last_err

    print("\nDataset:")
    print(ds)

    if VARIABLE not in ds:
        raise KeyError(
            f"Expected variable {VARIABLE!r} not present. "
            f"Variables found: {list(ds.data_vars)}"
        )

    print(f"\nVariable {VARIABLE}:")
    print(ds[VARIABLE])

    # Normalize longitude if a future file uses 0..360 convention.
    lon_name = "longitude"
    lat_name = "latitude"

    if float(ds[lon_name].max()) > 180.0:
        lon = ((ds[lon_name] + 180.0) % 360.0) - 180.0
        ds = ds.assign_coords({lon_name: lon}).sortby(lon_name)

    # xarray slice() needs monotonically INCREASING coordinates; some grids
    # store latitude north-to-south. Sort so the slices below never return empty.
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        ds = ds.sortby(lat_name)

    # Keep the requested tropical-Pacific region.
    ssh = ds[VARIABLE].sel(
        longitude=slice(BBOX[0], BBOX[2]),
        latitude=slice(BBOX[1], BBOX[3]),
    )

    # Equatorial average over 2 S to 2 N.
    ssh_eq = ssh.sel(
        latitude=slice(-2.0, 2.0)
    ).mean("latitude", skipna=True)

    print("\nEquatorial-average SSHA:")
    print(ssh_eq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ssh_eq.plot(
        ax=ax,
        x="longitude",
        y="time",
        cbar_kwargs={"label": "SSHA [m]"},
    )
    ax.set_title("NASA SSH: equatorial Pacific, mean over 2°S–2°N")
    ax.set_xlabel("Longitude [degrees east]")
    ax.set_ylabel("Time")
    fig.tight_layout()

    outpng = Path("nasa_ssh_equatorial_hovmoller_smoke_test.png")
    fig.savefig(outpng, dpi=180)
    print(f"\nSaved {outpng}")

    # Save just the small selected field for subsequent PINN development.
    outnc = Path("nasa_ssh_equatorial_smoke_test.nc")
    ssh.to_netcdf(outnc)
    print(f"Saved {outnc}")


if __name__ == "__main__":
    main()
