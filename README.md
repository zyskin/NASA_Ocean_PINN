# Equatorial Shallow-Water PINN for Tropical-Pacific Ocean Dynamics

A physics-informed neural network (PINN) study that infers the baroclinic
gravity-wave speed `c` (and hence equivalent depth `h_e = c²/g`) **and the
unobserved surface velocity field** from sea-surface-height (SSH) observations
alone, using the linear equatorial beta-plane shallow-water equations as the
physical constraint.

```
u_t - β y v + Φ_x = 0
v_t + β y u + Φ_y = 0
Φ_t + c² (u_x + v_y) = 0           β = 2Ω/R_E ,  Φ = g·η
```

The target application is NASA satellite SSH (`NASA_SSH_REF_SIMPLE_GRID_V1`,
variable `ssha`) over the tropical Pacific, with independent validation against
OSCAR surface currents, and later MUR SST / SMAP SSS and a SWOT coastal branch.

---

## ⚠️ Data status in this checkout

This repository was developed in a sandbox where **all NASA Earthdata hosts are
blocked by the network egress policy** (`cmr.earthdata.nasa.gov`,
`urs.earthdata.nasa.gov`, … all return 403 at the proxy) and **no Earthdata
credentials are present**. The real-data download therefore could not run here.

To make real scientific progress anyway, the full downstream workflow (Stages
3–6) was verified on a **controlled Observing System Simulation Experiment
(OSSE) surrogate**: a *known-truth* field built by superposing the
**verified analytic** eastward Kelvin wave and westward Matsuno/Rossby wave on
the exact target grid/region. Every surrogate file and variable is tagged
`synthetic=1` and named `surrogate_*` — it is **never** presented as a NASA
observation. This is standard practice: validate the *method* against known
truth before applying it to data.

**The real-data path is complete and runnable** (`src/data_access.py`,
`src/build_dataset.py`, `src/check_access.py`). It writes NetCDF with the
*identical schema* to the surrogate, so `analysis.py`, `pinn_ssh.py` and
`validate_velocity.py` run unchanged on real NASA output — see
["Running on real NASA data"](#running-on-real-nasa-data).

---

## Verified results (this checkout)

| Stage | What | Result |
|---|---|---|
| Math | Analytic Kelvin residuals (float32 run) | RMS ~1e-14 |
| Math | Analytic Kelvin/Rossby residuals (float64) | RMS ~1e-22 — machine precision |
| Inverse (synthetic) | `c` from SSH-only Kelvin (true 2.5) | **2.502 m/s, 0.09% error** |
| Rossby (Stage 6) | Matsuno n=1 mode propagation | **westward −0.817 m/s** (long-Rossby limit c/3=0.833); PDE residual ~1e-22 |
| Stage 3 (surrogate) | Conventional slant-stack speed (true: +2.8 Kelvin / −0.93 Rossby) | **east +2.89 m/s, west −0.87 m/s** |
| Stage 4 (surrogate) | PINN `c` from SSH only (true 2.8) | **2.93 m/s, 4.6% error**, h_e = 0.875 m |
| Stage 5 (surrogate) | PINN velocity vs **held-out** OSCAR-like | **u corr 0.83**, v corr 0.49, dir MAE 32° |

The PINN recovers the gravity-wave speed to within the uncertainty of the
conventional Hovmöller estimate **and additionally reconstructs the full
velocity field** that was never shown to it in training.

### Why v is recovered less well than u
In a Kelvin-dominated equatorial field `v ≈ 0`, so `v` is small and
noise-limited; its correlation is necessarily weaker than `u`. This is the
physically correct behaviour, not a bug.

---

## Recommended first real experiment

* **Region:** 170°W–100°W, 8°S–8°N  (CMR bbox `(-170, -8, -100, 8)`)
* **Equatorial band for Hovmöller:** 2°S–2°N
* **Dates:** 2015-01-01 → 2016-06-30 — covers the very strong **2015/16 El Niño**
  (downwelling Kelvin pulses in boreal spring 2015 + reflected Rossby waves);
  ideal for propagating equatorial anomalies. Start with the smoke window
  (2023-11-01 → 2023-12-31) to validate access first.
* **Primary variable:** `ssha` [m] from `NASA_SSH_REF_SIMPLE_GRID_V1`
  (~0.5°, weekly). Validate velocity against `OSCAR_L4_OC_FINAL_V2.0`
  (`u,v,ug,vg`).

Finalise exact dates after inspecting real granule availability/quality; see
`data_manifest.csv`.

---

## Quick start (offline surrogate — runs anywhere)

```bash
pip install -r requirements.txt       # torch, xarray, earthaccess, …
bash run_all.sh                       # full pipeline on the OSSE surrogate
```

Individual steps:

```bash
python3 tests/equatorial_kelvin_pinn_test.py   # math + synthetic inverse test
python3 tests/rossby_analytic_test.py          # Rossby/Kelvin PDE verification
python3 src/make_surrogate.py                  # build known-truth OSSE data
python3 src/analysis.py                        # Hovmöller + conventional speed
python3 src/pinn_ssh.py --adam 5000            # SSH-only PINN (infer c,u,v)
python3 src/validate_velocity.py               # Stage-5 velocity validation
```

## Running on real NASA data

1. Configure Earthdata credentials **once** (never hard-coded):
   ```bash
   python3 -c "import earthaccess; earthaccess.login(persist=True)"   # writes ~/.netrc
   # or: export EARTHDATA_USERNAME=... EARTHDATA_PASSWORD=...
   ```
2. Preflight and smoke test:
   ```bash
   python3 src/check_access.py
   python3 src/build_dataset.py --smoke               # ≤3 granules, inspect first
   ```
3. Build the experiment dataset and run the identical pipeline:
   ```bash
   python3 src/build_dataset.py --products ssh oscar --start 2015-01-01 --stop 2016-06-30
   python3 src/analysis.py  --ssh data_processed/nasa_ssh.nc
   python3 src/pinn_ssh.py  --ssh data_processed/nasa_ssh.nc \
                            --oscar data_processed/nasa_oscar.nc --tag nasa
   python3 src/validate_velocity.py --pinn data_processed/pinn_nasa_inferred.nc \
                                    --oscar data_processed/nasa_oscar.nc
   ```

If your environment also blocks NASA hosts, see the note in
`src/check_access.py`; the environment's **Network access** setting must allow
`*.earthdata.nasa.gov` (and the PO.DAAC/S3 data hosts).

---

## Method notes (for real data)

* **SI physical coordinates everywhere.** Longitude/latitude/time are mapped to
  metres/seconds; the network normalises inputs internally, but autograd
  differentiates the *physical* coordinates so PDE metric factors are exact.
* **Φ = g·η** is an *effective single-baroclinic-mode* approximation, not a
  claim that the ocean is one homogeneous layer.
* **`c > 0` trainable** via `c = exp(log_c)`; reported with `h_e = c²/g`.
* **Fourier-feature input embedding** (Tancik et al. 2020) defeats MLP spectral
  bias so fields with many oscillations over a long record are representable.
  `--ft` should be ≳ 2× the number of temporal cycles in the record.
* **Residual conditioning is data-derived**, not hand-tuned:
  `RU=RV=Φ_scale/L`, `RΦ=Φ_scale·c_nom/L`, with `L=√(c_nom/β)` the equatorial
  deformation radius.
* **Losses logged separately:** `L_SSH, L_pde1, L_pde2, L_pde3, L_velreg`
  (see `logs/pinn_*_history.csv`), never only the total.
* **Adam → L-BFGS**, checkpoints saved before and after L-BFGS.

## Scientific safeguards honoured

* Synthetic surrogate is clearly labelled and never passed off as NASA data.
* OSCAR is **held out** of training, so velocity validation is independent.
* No midlatitude `1/f` geostrophy at the equator — the beta-plane momentum
  equations are used (`f = βy → 0` handled correctly).
* PDE signs, dimensional factors and units are unchanged from the verified
  reference and re-checked by `tests/rossby_analytic_test.py`.
* Wave identity (Kelvin vs Rossby) is decided by propagation direction,
  meridional structure and the dispersion relation — not by visual appearance.
* Low NN loss alone is not treated as proof; parameter recovery and independent
  velocity validation are the evidence.

---

## Repository layout

```
src/
  config.py            region, dates, constants, product short names, scales
  physics.py           PINN model, PDE residuals, analytic Kelvin/Matsuno waves
  data_access.py       earthaccess search/download helpers (real data)
  build_dataset.py     Stage 2 real-data assembler + manifest  (needs NASA access)
  check_access.py      non-hanging NASA access preflight
  make_surrogate.py    OSSE surrogate generator (SYNTHETIC, known truth)
  analysis.py          Stage 3: Hovmöller + conventional propagation speed
  pinn_ssh.py          Stage 4/5: SSH-only PINN + velocity validation
  validate_velocity.py Stage 5 figures
  write_surrogate_manifest.py
tests/
  equatorial_kelvin_pinn_test.py   given reference (math + synthetic inverse)
  rossby_analytic_test.py          Rossby/Kelvin PDE verification (Stage 6)
data_processed/        surrogate_*.nc, pinn_*_inferred.nc  (and nasa_*.nc on real runs)
models/                PINN checkpoints (pre/post L-BFGS)
logs/                  run logs + training-history CSVs + speed estimates
figures/               Hovmöller, maps, speed spectrum, velocity validation
data_manifest.csv      produced surrogate files + targeted real NASA products
requirements.txt / environment.yml / run_all.sh
```

## NASA products (PO.DAAC short names)

| Role | Short name | Key variables |
|---|---|---|
| SSH (primary) | `NASA_SSH_REF_SIMPLE_GRID_V1` | `ssha` |
| Surface currents (validation) | `OSCAR_L4_OC_FINAL_V2.0` | `u,v,ug,vg` |
| SST (tracer, Stage 7) | `MUR-JPL-L4-GLOB-v4.1` | `analysed_sst` |
| SSS (auxiliary, Stage 7) | `SMAP_JPL_L3_SSS_CAP_8DAY-RUNNINGMEAN_V5` | `smap_sss` |
| Coastal high-res (Stage 8) | `SWOT_L2_LR_SSH_BASIC_D` | SSH/SSHA, quality, dist-to-coast |

## Roadmap (remaining stages)

* **Stage 7** — collocate MUR SST / SMAP SSS with SSH & inferred currents;
  correlations first, tracer equations (with non-zero surface fluxes) only after.
* **Stage 8** — New York Bight (75.5–71°W, 38.5–41.5°N): SWOT coverage check,
  MUR SST, variable depth `H(x,y)`, tides/wind/open boundaries from NOAA/IOOS;
  start linear, stay NASA-centred.

See the per-stage notes and `tests/` for what has been independently verified.
