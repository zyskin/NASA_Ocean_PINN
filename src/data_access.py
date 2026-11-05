#!/usr/bin/env python3
"""
data_access.py  --  NASA Earthdata search / download helpers (Stage 1 & 2).

Credentials are NEVER stored in the repository. earthaccess.login() uses, in
order: environment variables EARTHDATA_USERNAME / EARTHDATA_PASSWORD, then
~/.netrc (machine urs.earthdata.nasa.gov), then an interactive prompt. Set up
once with:   python -c "import earthaccess; earthaccess.login(persist=True)"

This module is import-safe even where NASA hosts are unreachable: it only
touches the network inside the functions. The tropical-Pacific experiment uses
ssh + oscar; sst/sss/swot are provided for later stages.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C      # noqa: E402


def login(strategy="environment"):
    import earthaccess
    try:
        auth = earthaccess.login(strategy=strategy)
    except Exception:
        auth = earthaccess.login()        # fall back to netrc / interactive
    if not getattr(auth, "authenticated", False):
        raise RuntimeError(
            "Earthdata login failed. Provide credentials via EARTHDATA_USERNAME/"
            "EARTHDATA_PASSWORD env vars or ~/.netrc (urs.earthdata.nasa.gov). "
            "Never hard-code them.")
    return auth


def search(product_key, bbox=C.BBOX, temporal=(C.SMOKE_START, C.SMOKE_STOP),
           count=50):
    """Search CMR for a product. Returns the list of granule results."""
    import earthaccess
    short = C.PRODUCTS[product_key]
    results = earthaccess.search_data(
        short_name=short, bounding_box=bbox, temporal=temporal, count=count)
    print(f"[{product_key}] {short}: {len(results)} granules in "
          f"{temporal[0]}..{temporal[1]} over {bbox}")
    return results


def preview(results, n=3):
    for i, r in enumerate(results[:n]):
        print(f"--- granule {i} ---")
        print(r)


def download(results, product_key, n_max=None):
    """Download granules into data_raw/<product>/ . n_max caps the count."""
    import earthaccess
    outdir = C.DATA_RAW / product_key
    outdir.mkdir(parents=True, exist_ok=True)
    sel = results if n_max is None else results[:n_max]
    print(f"[{product_key}] downloading {len(sel)} granule(s) -> {outdir}")
    files = earthaccess.download(sel, local_path=str(outdir))
    return [str(f) for f in files]


def open_many(files, **kw):
    """Open a set of local NetCDF files robustly with engine fallbacks."""
    import xarray as xr
    files = [f for f in files if str(f).endswith((".nc", ".nc4", ".h5"))]
    if not files:
        raise RuntimeError("No NetCDF files to open.")
    for engine in ("netcdf4", "h5netcdf"):
        try:
            return xr.open_mfdataset(files, combine="by_coords",
                                     engine=engine, **kw)
        except Exception as e:  # noqa: BLE001
            last = e
    # final fallback: open individually and concat on time
    try:
        dss = [xr.open_dataset(f) for f in files]
        return xr.concat(dss, dim="time")
    except Exception:
        raise last
