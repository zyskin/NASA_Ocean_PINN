# Forecast v2 — does physically-informed ocean dynamics predict tropical-Pacific SSH/current evolution beyond persistence?

Branch `claude/ocean-forecast-v2` (from `tacc/local-real-nasa` @ `a84920d`).

**Central question.** Not "can a PDE be tuned until its RMSE is just below
persistence?", but: *does physically-informed ocean dynamics contain
reproducible information about the future SSH / current increment beyond the
last observed state?* We answer it with strictly chronological out-of-sample
validation.

**Short answer. Yes — demonstrated on a 3.8-year, regime-diverse record with
strictly chronological validation and no oracle wind.**
- The **raw dynamical model overshoots** (α≈1 is ~2–3× too much amplitude), so
  on a short/quiet record it can look worse than persistence — exactly why the
  previous tuning fled to a near-persistence optimum (c=1 m/s, large wind depth,
  strong damping). That optimum *suppressed the very signal the dynamics carry.*
- The **PDE-predicted increment carries a reproducible pattern** of the true
  future change — pooled pattern correlation **≈0.4 at 14 days for both SSH and
  currents** on the extended record — once real dynamics are admitted
  (background window N_bg ≥ 2, realistic wave speed c ≈ 2.45 m/s).
- After a one-coefficient **chronological amplitude calibration** (fit on past
  windows only), the calibrated-PDE and a PDE+tendency **hybrid beat persistence
  out-of-sample with significance at both leads**. On the **201-date** record
  (182 chronological starts, **no wind**) the clean **14-day SSH** hybrid skill
  is **+0.123, 95% CI [0.103, 0.143]** (92% of windows beat persistence);
  current skill **+0.094 [0.080, 0.109]**.
- The hybrid **crushes a pure recent-tendency baseline** (≈0 at 14 d), so the
  gain is **dynamical**, not temporal extrapolation.
- On the **45-date** cube the clean 14-day SSH gain was only +0.061 and **not**
  significant — a **statistical-power artefact** of the short, quiescent record
  (confirmed by the extension). The advection scheme is *not* limiting
  (second-order MUSCL ≈ first-order upwind).

---

## 1. Data sources actually used (verified against NASA Earthdata, 2026-11-16)

| Field | Collection | Version | Native | Notes |
|---|---|---|---|---|
| SSH anomaly `ssha` [m] | `NASA_SSH_REF_SIMPLE_GRID_V11` | **1.1** | 0.5°, weekly | V1 has **no** 2026 granules → the cube uses **V1.1**. File `NASA-SSH_alt_ref_simple_grid_v1_1_YYYYMMDD.nc`. |
| Surface currents `u,v,ug,vg` [m/s] | `OSCAR_L4_OC_NRT_V2.0` | 2.0 (NRT) | 0.25°, daily | Analyzed product (uses SSH gradients+wind+SST) — **not** an independent current measurement. |
| 10-m wind / stress | NASA GEOS-FP `tavg1_2d_slv_Nx` | — | ~0.25°, hourly | Daily-mean sampled on weekly dates; stress computed hourly-then-averaged. **Analyzed** (oracle) winds. |

- **Cube used for v2 core analysis**: `data_processed/ssh_oscar_wind_weekly_pacific.nc`,
  45 weekly dates **2025-12-24 → 2026-10-28**, grid lon −170…−90°, lat −12…12°,
  0.5°. Forecasts crop to lon −170…−100°, lat −10…10°.
- Full provenance + leakage + serial-dependence table: `data_processed/forecast_v2/data_audit.csv`.

## 2. Leakage / product-construction audit (critical)

Each NASA-SSH weekly map is a **10-day composite** (nominal date ± 5 days;
confirmed from CMR granule `RangeDateTime`, span = 10.0 d). Consequences:

- **7-day lead is contaminated.** Init window [t₀−5, t₀+5] overlaps the 7-day
  target window [t₀+2, t₀+12] by ~3 days of shared altimeter observations. A
  "7-day forecast" is partly verified against data already inside its own
  initialization.
- **14-day lead is clean.** Init [t₀−5, t₀+5] vs target [t₀+9, t₀+19] do not
  overlap (4-day gap). **We therefore treat the 14-day lead as the honest
  horizon** and flag 7-day numbers.
- The 10-day compositing also **temporally low-pass filters** the maps, which
  inflates persistence skill and depresses any model's apparent gain.

**Serial dependence** (local computation): consecutive weekly SSHA maps have
pattern correlation **0.931** (7 d), 0.850 (14 d); weekly *increments* are
nearly white (lag-1 0.105). A 45-week record thus contains only a handful of
effectively independent forecast opportunities — small percentage RMSE
differences are not resolvable here. We use a **moving-block bootstrap** (block =
4 starts) for all confidence intervals.

## 3. Forecast model (unchanged physics, modal interpretation)

Residual-subtracted weakly nonlinear equatorial shallow-water system, evolving a
perturbation q′=(u′,v′,η′) about a frozen observed background q̄ so that
persistence (q′=0, τ=τ̄) is an exact steady state:

```
u_t = -(u·grad)u + f v - g η_x + τ_x/(ρ Hw) - r u + ν ∇²u
v_t = -(u·grad)v - f u - g η_y + τ_y/(ρ Hw) - r v + ν ∇²v
η_t = -(u·grad)η - He (u_x+v_y),      He = c²/g
         F(q̄+q′, τ(t)) - F(q̄, τ̄)   (frozen-background tendency removed)
```

η is satellite SSHA, He is a **modal equivalent depth** (not physical depth), Hw
an effective wind-coupling depth, r phenomenological damping. Numerics: RK4,
Δt=1800 s, first-order upwind advection (and an added second-order MUSCL/minmod
option), ν=1000 m²/s, boundary sponge. The engine
(`scripts/forecast_v2_lib.py`) **reproduces the baseline rolling hindcasts to
machine precision** (max |SSH-skill diff| ≈ 1×10⁻¹⁶ vs the committed baseline
CSV).

## 4. Validation protocol (no leakage of the forecast kind)

Rolling-origin: a forecast initialized at tᵢ uses, for **all** fitted quantities
(shrinkage α, tendency coefficient, hybrid coefficients), **only** windows
strictly before tᵢ. Coefficients are ridge least-squares fits pooling all grid
cells of all past windows for that lead. The PDE configuration (N_bg=3, c=2.45,
Hw=60, Td=20) is fixed **a priori** on physical grounds (realistic equatorial
wave speed; N_bg≥2 so an observed transient can propagate), not chosen by test
skill. Analyzed future winds make the wind-forced runs **oracle hindcasts**, not
causal forecasts (Section 8).

## 5. PRIMARY diagnostic — PDE increment information content

`data_processed/forecast_v2/increment_information_content.csv`. Pooled over the
38 rolling starts, pattern correlation of δη_PDE = η_model(t₀+Δt) − η(t₀) with
the observed δη_true:

| config | SSH corr 7d | SSH corr 14d | α (shrink) | current corr 14d |
|---|---|---|---|---|
| baseline N1, c=1.0 (tuned) | +0.060 | +0.075 | 0.99 | +0.171 |
| phys N1, c=2.45 | +0.081 | +0.109 | 0.19 | +0.139 |
| phys N2, c=2.45 | +0.227 | +0.269 | 0.39 | +0.291 |
| **phys N3, c=2.45** | **+0.283** | **+0.356** | 0.44 | **+0.390** |

The tuned baseline has **almost no increment information** (its score comes from
≈persistence). Admitting dynamics (N_bg, realistic c) exposes a reproducible
predictable pattern; the optimal shrink α≈0.3–0.44 says the PDE gets the
**direction/pattern right but ~2–3× too much amplitude**.

## 6. Chronological out-of-sample results (45-date cube, 30 scored starts)

`model_comparison.csv`, `bootstrap_ci.csv`. SSH skill vs persistence:

| lead | raw PDE (α=1) | calibrated PDE | tendency | **hybrid PDE+tendency** |
|---|---|---|---|---|
| 7 d | −0.118 | +0.032 [0.010, 0.051] ✓ | +0.012 | **+0.069 [0.026, 0.115] ✓** |
| 14 d (clean) | −0.082 | +0.025 [−0.020, 0.067] | +0.006 | **+0.061 [−0.003, 0.127]** |

Current (vector) skill vs persistence:

| lead | calibrated PDE | hybrid |
|---|---|---|
| 7 d | +0.062 [0.042, 0.078] ✓ | +0.067 [0.049, 0.086] ✓ |
| 14 d | +0.068 [0.050, 0.086] ✓ | +0.069 [0.053, 0.088] ✓ |

(✓ = 95% block-bootstrap CI above zero.) Wins: hybrid SSH 22/30 at both leads.
Regional (`regional_skill.csv`): hybrid positive in every subregion; strongest
in the **equatorial core at 7 d (+0.101)** and off-equatorial / eastern at 14 d
(+0.06–0.07).

**Reading.** The dynamics help **currents** significantly at both leads and
**7-day SSH** significantly. For the honest **14-day SSH** metric the hybrid is
positive (+0.061) and beats pure tendency, but its CI just includes zero — a
*suggestive, not significant* result on 45 weeks. Raw (uncalibrated) dynamics
hurt — consistent with the previous near-persistence optimum.

## 7. Numerical advection (MUSCL vs upwind)

`physics_variant_tests.csv`. Second-order MUSCL/minmod gives essentially the
same increment information and skill as first-order upwind (14-d SSH
increment-corr 0.361 vs 0.356; hybrid SSH skill +0.058 vs +0.062). **Numerical
diffusion from first-order upwind is not the limiting factor** at these scales
and lead times; we keep upwind as the stable default and report MUSCL parity.

## 8. Wind: oracle vs ocean-only, and causality

- **No-wind (pure ocean dynamics) retains most of the SSH signal** (14-d
  increment-corr 0.276 vs 0.356 with wind; hybrid SSH skill +0.056 vs +0.062).
  The ocean-state dynamics, not the wind, carry the bulk of SSH predictability.
- **Oracle wind helps currents** (14-d current increment-corr 0.39 vs 0.32).
  But these winds are GEOS-FP **analysis** for t>t₀ — future information. A
  genuine causal forecast would use forecast winds and lose part of this gain;
  we label the wind-forced runs **oracle hindcasts**. The no-wind experiment is
  the honest lower bound and motivates the no-wind historical extension.
- **Total vs geostrophic OSCAR init**: total (u,v) initialization gives better
  current skill than geostrophic (ug,vg) (hybrid UV +0.069 vs +0.023); SSH
  comparable. We use total OSCAR, noting OSCAR's partial dependence on SSH/wind
  (so current "skill" is state-estimate agreement, not independent truth).

## 9. Historical extension (Experiment A: long SSH+OSCAR, no wind)

Motivation: 45 weeks give too few independent samples to resolve the 14-day SSH
question. `NASA_SSH_REF_SIMPLE_GRID_V11` extends to **1992-10-21** (weekly) and
OSCAR FINAL to 1993; GEOS-FP (wind) only to ~2017, so the extension is run
**without wind**. We rebuilt an SSH+OSCAR cube **2023-01-02 → 2026-11-02** (`ssh_oscar_weekly_ext.nc`,
**201 weekly dates**; OSCAR FINAL for dates ≤2026-01-17 else INTERIM, transition
recorded per time step in `oscar_product`) and repeated the increment diagnostic
+ chronological calibration with **no wind forcing** (pure ocean dynamics;
`scripts/forecast_v2_extended_analysis.py`, `extended_model_comparison.csv`).
This spans the strong **2023-24 El Niño** and **2024-25 La Niña** — far more
regime diversity than the 45-week cube.

**The result is decisive (182 chronological out-of-sample starts).** Every
calibrated/hybrid skill is now statistically significant, and the clean 14-day
SSH gain — borderline on 45 weeks — is large and significant on the long record:

PDE-increment pattern correlation (no wind): SSH +0.383 (7 d), **+0.408 (14 d)**;
currents +0.359 (7 d), +0.410 (14 d).

SSH skill vs persistence (95% block-bootstrap CI):

| lead | raw PDE (α=1) | calibrated PDE | tendency | **hybrid** |
|---|---|---|---|---|
| 7 d | −0.024 (n.s.) | +0.062 [0.049,0.076] ✓ | +0.008 ✓ | **+0.120 [0.104,0.136] ✓ (96% wins)** |
| 14 d (clean) | +0.048 [0.019,0.076] ✓ | +0.069 [0.049,0.090] ✓ | −0.000 (n.s.) | **+0.123 [0.103,0.143] ✓ (92% wins)** |

Current (vector) skill: calibrated/hybrid +0.057…+0.094, all 95% CIs above zero
at both leads.

**Interpretation.**
- The borderline 14-day SSH result on 45 weeks was a **statistical-power
  artefact**: with 4× the data and more regimes the same clean metric becomes
  **+0.123 [0.103, 0.143]** — unambiguously significant and *larger*.
- The hybrid (+0.12) crushes **pure tendency (≈0 at 14 d)**, so the gain is
  **dynamical**, not temporal extrapolation.
- This is the **no-wind (Experiment A)** result, so it does **not** rely on
  oracle winds — it is a genuine ocean-state forecast (initial OSCAR state is
  available at t₀). On the long, regime-diverse record even the **raw**
  dynamics beat persistence at 14 days.
- Predictability is **regime-dependent**: the quiescent 2025-26 window (the
  45-date cube) was less predictable than the 2023-25 ENSO-active years, which
  is why the short record understated the dynamical skill.

Figure `figures/forecast_v2/fig7_extended_power.png` shows the 45→201-date
power/significance gain.

**Bottom line on the central question:** on a 3.8-year, regime-diverse NASA
SSH+OSCAR record with strictly chronological validation, physically-informed
ocean dynamics carry **reproducible, statistically significant** information
about the future SSH and surface-current increment beyond both persistence and a
recent-tendency baseline, at 7- and 14-day leads, **without** oracle wind —
after a simple chronological amplitude calibration that fixes the raw model's
~2–3× over-amplitude.

## 10. Models tested (summary) and negative results

- Persistence (reference). **Very strong** here due to 0.93 weekly
  autocorrelation + 10-day compositing.
- Raw dynamical PDE (baseline family): **negative skill** at α=1 — a clear
  negative result; uncalibrated dynamics overshoot.
- Recent-tendency extrapolation: small, mostly non-significant positive skill.
- Calibrated PDE and PDE+tendency hybrid: the only models that beat persistence
  with significance (currents both leads; SSH 7 d; SSH 14 d suggestive).
- MUSCL advection, geostrophic init, no-wind: parity or modest changes as above.

## 11. Metrics reported

SSH RMSE (m) & skill, SSH pattern correlation, vector-current RMSE & skill,
mean/median skill, win fraction, worst-case skill, regional breakdown, and
block-bootstrap 95% CIs — all in `data_processed/forecast_v2/*.csv`.

## 12. Remaining limitations

1. **Record length / independence (addressed).** 45 weeks gave too few
   independent samples (clean 14-day SSH not significant); the 201-week
   extension resolves this and makes the dynamical skill significant. Longer
   still (V1.1 reaches 1992, OSCAR FINAL to 1993) would enable per-year
   train/validation/test splits and season/regime stratification.
2. **Product smoothing & 7-day leakage.** 10-day composites inflate persistence
   and contaminate the 7-day lead.
3. **Oracle winds.** Wind benefit is not available to a causal forecast.
4. **OSCAR circularity.** OSCAR uses SSH/wind, so current verification is not
   fully independent.
5. **Single modal layer.** One equivalent depth; multi-mode physics untested
   (deferred until one-mode skill is firmly established).

## Reproduce

```bash
python3 scripts/forecast_v2_increment_diagnostic.py   # primary diagnostic
python3 scripts/forecast_v2_calibration.py            # chronological calibration
python3 scripts/forecast_v2_significance.py           # bootstrap CIs, regional, velocity
python3 scripts/forecast_v2_physics_tests.py          # MUSCL, no-wind, geostrophic
python3 scripts/build_extended_cube.py                # historical SSH+OSCAR 2023-26 (no wind)
python3 scripts/forecast_v2_extended_analysis.py      # extended chronological analysis
python3 scripts/forecast_v2_figures.py                # figures 1-6
python3 scripts/forecast_v2_fig_extended.py           # figure 7 (power gain)
```

Outputs: `data_processed/forecast_v2/*.csv` (+ `ssh_oscar_weekly_ext.nc`),
`figures/forecast_v2/fig1..7*.png`. The reproducible PDE-forecast cache
`data_processed/forecast_v2/_pde_cache/` is git-ignored.
