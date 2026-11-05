#!/usr/bin/env bash
# Reproduce the full study. Steps 1-2 and 4-8 run fully OFFLINE on the OSSE
# surrogate. Step 3 (real NASA data) runs only where Earthdata is reachable.
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

echo "== [1] Synthetic Kelvin PINN test (math + inverse verification) =="
python3 tests/equatorial_kelvin_pinn_test.py

echo "== [2] Analytic Rossby/Matsuno PDE verification =="
python3 tests/rossby_analytic_test.py

echo "== [3] NASA access preflight (expected to fail where egress is blocked) =="
python3 src/check_access.py || echo "  (NASA unreachable here -- using surrogate)"
# When reachable + credentials set, build the REAL dataset instead:
#   python3 src/build_dataset.py --products ssh oscar --start 2015-01-01 --stop 2016-06-30
#   (add --smoke for a 3-granule test first)

echo "== [4] Build OSSE surrogate (SYNTHETIC; Kelvin+Rossby on real grid) =="
python3 src/make_surrogate.py
python3 src/write_surrogate_manifest.py

echo "== [5] Stage 3: Hovmoller + conventional propagation speed =="
python3 src/analysis.py

echo "== [6] Stage 4: SSH-only PINN (infer c, u, v) =="
python3 src/pinn_ssh.py --adam 5000

echo "== [7] Stage 5: velocity validation vs held-out OSCAR-like =="
python3 src/validate_velocity.py

echo "== DONE. See figures/, models/, logs/, data_processed/ =="
# To run the identical pipeline on REAL NASA output:
#   python3 src/analysis.py   --ssh data_processed/nasa_ssh.nc
#   python3 src/pinn_ssh.py   --ssh data_processed/nasa_ssh.nc \
#                             --oscar data_processed/nasa_oscar.nc --tag nasa
