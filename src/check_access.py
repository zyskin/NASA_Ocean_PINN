#!/usr/bin/env python3
"""
check_access.py  --  non-hanging preflight for NASA Earthdata access.

Reports, without ever blocking on an interactive prompt:
  * whether Earthdata credentials are configured (env vars or ~/.netrc),
  * whether the CMR search host is reachable,
  * a single unauthenticated CMR search (counts granules) if reachable.

Use this before build_dataset.py to see immediately whether the real-data
pipeline can run in the current environment.
"""

import os
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C          # noqa: E402


def creds_present():
    if os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD"):
        return "env vars"
    netrc = Path.home() / ".netrc"
    if netrc.exists() and "urs.earthdata.nasa.gov" in netrc.read_text(errors="ignore"):
        return "~/.netrc"
    return None


def host_reachable(host, port=443, timeout=8):
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True, ""
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main():
    print("NASA Earthdata access preflight")
    print("-" * 40)
    c = creds_present()
    print(f"credentials: {'FOUND via ' + c if c else 'NOT CONFIGURED'}")

    ok, why = host_reachable("cmr.earthdata.nasa.gov")
    print(f"CMR host reachable: {ok}  {why}")
    if not ok:
        print("\n-> This environment cannot reach NASA Earthdata (egress policy).")
        print("   Run build_dataset.py where *.earthdata.nasa.gov is allowed and")
        print("   credentials are set. The surrogate pipeline runs fully offline.")
        return 2

    try:
        import earthaccess
        r = earthaccess.search_data(short_name=C.PRODUCTS["ssh"],
                                    bounding_box=C.BBOX,
                                    temporal=(C.SMOKE_START, C.SMOKE_STOP),
                                    count=5)
        print(f"CMR search OK: {len(r)} SSH granules in smoke window")
        return 0 if c else 1
    except Exception as e:  # noqa: BLE001
        print(f"CMR search failed: {type(e).__name__}: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
