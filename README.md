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

## Data provenance — read this first

Every result below is tagged by how it was obtained. The categories are kept
strictly separate and are **never** blurred:

| Category | Meaning | Files |
|---|---|---|
| **REAL NASA DATA** | actually downloaded from NASA Earthdata and analysed in this session | `data_processed/nasa_ssh.nc`, `data_processed/nasa_oscar.nc` |
| **REAL NASA (derived)** | computed by the PINN **from** real NASA SSHA | `data_processed/pinn_nasa_inferred.nc` |
| **SYNTHETIC / OSSE** | known-truth surrogate for *method* validation; tagged `synthetic=1`, named `surrogate_*` | `data_processed/surrogate_*.nc`, `data_processed/pinn_ssh_inferred.nc` |
| **REAL NOAA DATA** | — | none yet (NASA ran, so the NOAA HF-radar fallback was not needed) |
| **IMPLEMENTED BUT BLOCKED** | — | none in this session |
| **PLANNED** | MUR SST / SMAP SSS collocation; New York coastal SWOT/HF-radar branch | see roadmap |

**NASA Earthdata access worked in this environment** (`EARTHDATA_TOKEN` present;
`earthaccess.login(strategy="environment")` authenticated; PO.DAAC Cumulus OPS
downloads succeeded). The 2015–16 El-Niño tropical-Pacific experiment below is
on **real satellite observations**, cross-checked against the OSSE surrogate
that the previous session used to validate the method.

---

## REAL NASA DATA results (2015–16 El Niño, tropical Pacific)

* **Region** 170°W–100°W, 8°S–8°N · **SSH** `NASA_SSH_REF_SIMPLE_GRID_V1`
  (`ssha`, 80 weekly granules, 2014-12-29 → 2016-07-04) ·
  **Currents** `OSCAR_L4_OC_FINAL_V2.0` (`u,v,ug,vg`, 79 weekly, independent).

| Quantity | Method | Real-data result |
|---|---|---|
| Eastward (Kelvin-like) speed | slant-stack / Radon | **+1.50 m/s** |
| Eastward (Kelvin-like) speed | k–ω power-weighted | **+1.62 m/s** |
| Westward (Rossby-like) speed | slant-stack / Radon | **−0.68 m/s** |
| Westward (Rossby-like) speed | k–ω power-weighted | **−0.78 m/s** |
| **PINN effective `c`** | SSH-only inverse PINN | **0.834 m/s** (h_e = 0.071 m) |
| Held-out SSHA (unseen snapshots) | PINN | **corr 0.971, RMSE 0.021 m, skill +0.76** |
| Inferred `u` vs OSCAR geostrophic-anom | **independent** (OSCAR not trained on) | **corr 0.841**, RMSE 0.156 m/s |
| Inferred `u` vs OSCAR total-anom | independent | corr 0.781, RMSE 0.161 m/s |
| Inferred `v` vs OSCAR | independent | corr 0.08–0.19 (equatorial `v≈0`, noise-limited) |

**Interpretation — the estimates disagree, and that is physical, not a bug.**
The conventional methods isolate the *fastest coherent eastward* ridge
(≈ +1.5–1.6 m/s) and a *westward* Rossby band (≈ −0.7–0.8 m/s) separately. The
PINN instead fits **one effective `c`** to the whole anomaly field, which over
this weekly-sampled 2015–16 record is dominated by the slowly-evolving,
large-scale ENSO signal and long (westward) Rossby waves; the fast eastward
Kelvin pulses are a weaker, near-Nyquist spectral feature. The single effective
`c ≈ 0.83 m/s` (h_e ≈ 0.07 m) is therefore **lower than the canonical
first-baroclinic Kelvin speed (~2.5 m/s, h_e ~ 0.6 m)** — it is a blended
single-mode summary, *not* a clean first-mode gravity-wave speed. The honest,
strong result is the **independent velocity reconstruction**: the surface
zonal-current anomaly recovered from SSH alone tracks OSCAR at corr ≈ 0.84
(see `figures/nasa_velocity_hovmoller.png`) — the El-Niño eastward current
anomaly of 2015 and the La-Niña westward anomaly of early 2016 are both
reproduced without ever showing the PINN any velocity data.

Figures: `figures/nasa_ssha_maps.png`, `nasa_hovmoller_equatorial.png`,
`nasa_speed_spectrum.png`, `nasa_komega_spectrum.png`,
`nasa_velocity_scatter.png`, `nasa_velocity_hovmoller.png`,
`nasa_wave_speed_comparison.png`. Tables: `logs/nasa_propagation_speeds.csv`,
`nasa_komega_speeds.csv`, `nasa_wave_speed_comparison.csv`,
`pinn_nasa_summary.csv`.

---

## SYNTHETIC / OSSE method-validation results

These validate the *method* against known truth (they are **not** NASA
observations); the real-data results are in the section above.

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

This is exactly the pipeline that produced the real-data results above.

1. Credentials (never hard-coded). In this cloud environment an
   `EARTHDATA_TOKEN` is injected and `earthaccess` picks it up automatically:
   ```bash
   python3 -c "import earthaccess; print(earthaccess.login(strategy='environment').authenticated)"
   # locally instead: python3 -c "import earthaccess; earthaccess.login(persist=True)"  # ~/.netrc
   ```
2. Smoke test (≤3 granules, inspect first):
   ```bash
   python3 src/build_dataset.py --smoke               # real SSH, 3 granules
   ```
3. Build the real datasets and run the full real-data pipeline:
   ```bash
   python3 src/build_dataset.py --products ssh --start 2015-01-01 --stop 2016-06-30
   python3 src/build_oscar_weekly.py --start 2015-01-01 --stop 2016-06-30 --every 7
   python3 src/analysis.py     --ssh data_processed/nasa_ssh.nc        # maps, Hovmöller, slant-stack
   python3 src/komega.py       --ssh data_processed/nasa_ssh.nc        # k–ω spectrum + dispersion
   python3 src/pinn_nasa.py    --ssh data_processed/nasa_ssh.nc \
                               --oscar data_processed/nasa_oscar.nc --tag nasa
   python3 src/validate_nasa.py                                        # velocity figs + speed table
   ```

Real-data specifics handled by `pinn_nasa.py` (vs the synthetic `pinn_ssh.py`):
the per-pixel temporal mean is removed so the network fits the *propagating*
SSHA anomaly; entire time snapshots are **held out** for an honest SSH test; and
velocity is scored against OSCAR **anomalies** (total and geostrophic
separately), since the PINN reconstructs anomalous — not mean — currents.

If an environment blocks NASA hosts, `src/check_access.py` reports it without
hanging; the **Network access** setting must allow `*.earthdata.nasa.gov` and
the PO.DAAC data hosts, and `EARTHDATA_TOKEN` (or `~/.netrc`) must be set.

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
  build_dataset.py     Stage 2 real SSH assembler + manifest  (ran: nasa_ssh.nc)
  build_oscar_weekly.py cost-controlled weekly OSCAR subset   (ran: nasa_oscar.nc)
  check_access.py      non-hanging NASA access preflight
  make_surrogate.py    OSSE surrogate generator (SYNTHETIC, known truth)
  analysis.py          Hovmöller + conventional slant-stack speed (real & synthetic)
  komega.py            wavenumber-frequency spectrum + equatorial dispersion overlays
  pinn_ssh.py          SSH-only PINN on the SYNTHETIC surrogate
  pinn_nasa.py         SSH-only PINN on REAL NASA SSHA (held-out test + OSCAR compare)
  validate_velocity.py Stage-5 figures (synthetic)
  validate_nasa.py     real-data velocity figures + wave-speed comparison table
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
