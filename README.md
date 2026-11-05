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
OSCAR surface currents, and later MUR SST / SMAP SSS. A second **coastal branch**
(New York Bight) targets NOAA/IOOS HF-radar currents + CO-OPS water levels +
ETOPO bathymetry. A third strand **deepens the equatorial wave physics**
(k-ω spectra, multi-mode decomposition, identifiability).

---

## STATUS TABLE

| Component | Status | Evidence |
|---|---|---|
| Equatorial PDE signs/scales + Kelvin analytic | **VERIFIED (math)** | residuals ~1e-22 (float64) |
| Kelvin inverse PINN (synthetic) | **VERIFIED SYNTHETIC/OSSE** | c err 0.09% |
| Matsuno Rossby n=1,2 + IG roots | **VERIFIED (math)** | residuals ~1e-21; `tests/matsuno_multimode_test.py` |
| Equatorial SSH→(u,v,c) PINN | **VERIFIED SYNTHETIC/OSSE** | c err 4.6%, u corr 0.83 |
| Conventional slant-stack / k-ω / modal c | **VERIFIED SYNTHETIC/OSSE** | east +2.8–2.9, west Rossby resolved |
| k-ω spectrum + dispersion overlay (B1) | **VERIFIED SYNTHETIC/OSSE** | peaks on Kelvin/Rossby branches |
| Meridional modal decomposition (B2) | **VERIFIED SYNTHETIC/OSSE** | modes separated by structure+direction |
| Degradation robustness / identifiability (B4/B5) | **VERIFIED SYNTHETIC/OSSE** | `logs/robustness_table.csv`, `logs/identifiability_table.csv` |
| Coastal rotating-SWE operator (var. H, drag) | **VERIFIED (math)** | residuals ~1e-20; `tests/coastal_synthetic_test.py` |
| Coastal PINN (HF-radar→u,v,η) | **VERIFIED SYNTHETIC/OSSE** | held-out vec corr 1.0; 15× better than interp |
| NASA SSH/OSCAR/MUR/SMAP/SWOT ingestion | **IMPLEMENTED — DATA ACCESS BLOCKED** | `src/build_dataset.py`; egress blocks `*.earthdata.nasa.gov` |
| NOAA HF-radar / CO-OPS / ETOPO ingestion | **IMPLEMENTED — DATA ACCESS BLOCKED** | `src/coastal_data.py`; egress blocks `*.noaa.gov` |
| **Any real-data result** | **NONE YET** | all real endpoints blocked in this sandbox |
| Tracer coupling (SST/SSS), nonlinear coastal | **PLANNED** | see roadmap |

> **No result in this repository uses real observations.** Every quantitative
> result is from controlled known-truth synthetic data and validates the
> *method*. Real NASA/NOAA endpoints are unreachable from this sandbox
> (egress policy), so the ingestion code is implemented and runnable but has
> not executed here.

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

## Track A — New York Bight coastal PINN (synthetic PoC; real data blocked)

Real NOAA/IOOS endpoints (HF-radar THREDDS `dods.ndbc.noaa.gov`, CO-OPS
`api.tidesandcurrents.noaa.gov`, NCEI/ETOPO, CoastWatch ERDDAP) are **all blocked
by the same egress policy** as NASA in this sandbox. The ingestion code is written
and runnable where those hosts are reachable (`src/coastal_data.py`,
`python src/coastal_data.py` for the preflight). Offline, the method is verified:

* **Coastal operator** (`src/coastal_physics.py`): linear rotating shallow-water
  `u_t - f v + g η_x + r u = 0`, `v_t + f u + g η_y + r v = 0`,
  `η_t + ∂_x(Hu) + ∂_y(Hv) = 0`, with variable `H(x,y)`, f-plane `f=2Ω sinφ₀`,
  optional trainable linear drag `r≥0`. `tests/coastal_synthetic_test.py`
  verifies the Poincaré plane-wave residuals (~1e-20), the drag-term wiring, and
  the variable-H flux divergence — all at machine precision.
* **Coastal PINN PoC** (`src/coastal_pinn.py`): observe velocity `(u,v)` only;
  infer the **unobserved** free surface `η` from velocity + physics (the coastal
  analogue of inferring velocity from SSH). On a synthetic two-wave M2 tidal
  field with HF-radar-like coverage + noise, spatial/temporal holdout gives PINN
  `u,v` RMSE ~0.0015 m/s and vector correlation ~1.0, **~15× better than linear
  space-time interpolation**, and inferred η RMSE ~0.008 m.

**Observation-model caveat (important):** HF radar measures **near-surface**
(~upper 0.5 m) velocity; the shallow-water equations describe **depth-averaged**
velocity. These are not the same quantity. For a short, tide-dominated interval
the near-surface current is used as a barotropic proxy; stronger claims are not
made. The synthetic PoC is also idealized (exact PDE solution, smooth tidal
field), so near-perfect recovery is expected — real HF-radar (non-tidal flow,
baroclinicity, gaps) will be harder. NOAA data must never be labelled NASA data.

Region (search box, tighten after coverage inspection): −74.6…−72.5°E,
39.0…41.2°N; decision subregions (Bight / Sandy Hook entrance / Lower Bay / inner
harbour) and coverage fractions are computed in `src/coastal_data.py`.

## Track B — deeper equatorial wave physics (synthetic/OSSE)

* **B1 k-ω spectrum** (`src/equatorial_spectrum.py`): Wheeler-Kiladis
  symmetric/antisymmetric wavenumber-frequency power with the exact Matsuno
  dispersion curves overlaid and a runtime sign-convention self-check. The
  Kelvin power lies on `ω=ck`; Rossby power on the westward branches
  (`figures/komega_sym_multimode.png`, `komega_anti_multimode.png`).
* **B2 modal decomposition** (`src/modal_decomposition.py`): least-squares
  projection onto Hermite meridional modes `ψ_m(y/L)`. ψ₀ → eastward Kelvin
  (77% variance), ψ₂ → westward Rossby n=1, ψ₁/ψ₃ → antisymmetric westward
  Rossby n=2 — modes separated by **structure + direction**, not appearance.
* **B3 multimode verification** (`tests/matsuno_multimode_test.py`): Rossby
  n=1,2 and eastward/westward inertia-gravity roots satisfy all three PDEs at
  ~1e-21; table of `n,k,ω,c_ph,c_g` with group velocity from finite differences
  in k (`logs/matsuno_multimode_table.csv`).
* **B4 robustness** (`src/identifiability.py`): slant-stack, k-ω, and PINN `c`
  under clean → noise → weekly → weekly+coarse sampling (`logs/robustness_table.csv`).
* **B5 identifiability** (`src/identifiability.py`): inferred `c` distribution
  over random inits for Kelvin-only / Rossby-only / K+R1 / K+R1+R2
  (`logs/identifiability_table.csv`); failures reported, not hidden.
* **B6 c comparison**: `c_PINN` vs Kelvin Hovmöller slant-stack vs k-ω ridge vs
  Rossby-dispersion inversion (`logs/c_comparison_table.csv`). The slant-stack
  and PINN agree to ~2-5%; the single-bin k-ω peak is coarser (~10%).

Run all Track B: `python src/osse_multimode.py && python src/equatorial_spectrum.py
&& python src/modal_decomposition.py && python tests/matsuno_multimode_test.py
&& python src/identifiability.py`.

### Track B key findings (synthetic)

**B6 — c estimators agree (true c = 2.8, K+R1):**

| method | c [m/s] | error |
|---|---|---|
| Kelvin Hovmöller slant-stack | 2.797 | 0.1% |
| PINN (SSH-only, trainable c) | 2.808 | 0.3% |
| Rossby-n1 dispersion inversion | 2.620 | 6.4% |
| k-ω single-bin Kelvin peak | 3.110 | 11% (coarse: FFT grid) |

**B4 — robustness under satellite-like degradation (K+R1+R2):** PINN c stays
within ~3.4% from clean → +1 cm noise → weekly → weekly+coarse; held-out
velocity correlation degrades gracefully 0.94 → 0.83. Slant-stack c 2.80 → 2.85.

**B5 — identifiability (inferred c over random inits, true 2.8):**

| modes present | c mean ± std | error | u corr | v corr |
|---|---|---|---|---|
| Kelvin only | 2.809 ± 0.003 | 0.3% | 1.000 | — (v≡0) |
| **Rossby n=1 only** | **0.503 ± 0.159** | **82%** | 0.970 | 0.963 |
| Kelvin + Rossby n=1 | 2.793 ± 0.026 | 0.3% | 0.966 | 0.905 |
| Kelvin + Rossby n=1 + n=2 | 2.738 ± 0.027 | 2.2% | 0.944 | 0.817 |

**Interpretation:** the gravity-wave speed `c` is well-determined from SSH alone
**only when a Kelvin component is present** (Kelvin propagates at exactly `c`).
A pure Rossby field does **not** constrain `c` (82% error, large init-variance),
because Rossby phase speed depends on `c` only weakly — yet the PINN still
reconstructs the Rossby velocity field well (corr ~0.96). This non-identifiable
case is reported, not hidden.

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
  # Track B (equatorial wave physics)
  osse_multimode.py    multi-mode OSSE (Kelvin+Rossby n1,n2) + degradation
  equatorial_spectrum.py  B1 k-ω spectrum + dispersion overlay
  modal_decomposition.py  B2 Hermite meridional-mode decomposition
  identifiability.py   B4/B5/B6 robustness, identifiability, c comparison
  # Track A (coastal)
  coastal_physics.py   rotating SWE operator + CoastalPINN (var H, drag)
  coastal_data.py      NOAA HF-radar/CO-OPS/ETOPO ingestion (needs NOAA access)
  coastal_pinn.py      synthetic coastal PINN PoC (HF-radar→u,v,η)
tests/
  equatorial_kelvin_pinn_test.py   given reference (math + synthetic inverse)
  rossby_analytic_test.py          Rossby/Kelvin PDE verification
  matsuno_multimode_test.py        B3 Rossby n1,2 + IG verification
  coastal_synthetic_test.py        coastal operator verification
data_processed/        surrogate_*.nc, osse_multimode_*.nc, pinn/coastal_*_inferred.nc
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

## Coastal (NOAA/IOOS) data sources — Track A

| Role | Source | Key variables |
|---|---|---|
| Surface currents | HF-radar THREDDS `hfradar_usegc_1km` (1 km hourly) | `u,v,hdop,number_of_sites` |
| Water level | NOAA CO-OPS (Battery 8518750, Sandy Hook 8531680) | water_level (datum recorded) |
| Bathymetry | NOAA NCEI ETOPO 2022 | elevation → `H(x,y)>0`, land mask |
| Coastal SSH (later) | NASA `SWOT_L2_LR_SSH_BASIC_D` | SSH/SSHA, quality, dist-to-coast |

## Roadmap (remaining)

* **Coastal, real data** — run `src/coastal_data.py` + `src/coastal_pinn.py --hfradar`
  where NOAA is reachable; inspect actual NY coverage, pick a high-coverage
  interval, add CO-OPS η anchors, ETOPO `H(x,y)`, and a no-normal-flow coastline
  condition. Then add SWOT/MUR where NASA is reachable.
* **Equatorial, real data** — run `src/build_dataset.py` where NASA is reachable;
  apply the verified k-ω / modal / PINN pipeline to real SSH + OSCAR.
* **Tracers (Stage 7)** — collocate MUR SST / SMAP SSS with SSH & inferred
  currents; correlations first, tracer equations (non-zero surface fluxes) later.
* **Nonlinear / open boundaries** — only after the linear coastal PoC on real data.

See the STATUS TABLE, the per-stage notes and `tests/` for what is independently
verified. **No real-data science has been done yet** — all endpoints are blocked
in this sandbox.
