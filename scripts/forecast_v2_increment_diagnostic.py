"""
forecast_v2_increment_diagnostic.py

PRIMARY diagnostic (central question): does the PDE-predicted increment
  deta_pde = eta_model(t0+dt) - eta(t0)
carry information about the observed increment
  deta_true = eta_obs(t0+dt) - eta(t0)
beyond the last observed state (persistence)?

For several model configurations and both leads, pooled over all rolling
out-of-sample starts, we report:
  * pooled pattern correlation (all grid cells x all windows),
  * mean per-window spatial pattern correlation,
  * regression slope alpha = <deta_pde . deta_true> / <deta_pde^2>,
  * explained variance R^2 of alpha*deta_pde vs deta_true,
  * amplitude ratio std(deta_pde)/std(deta_true),
  * whether the (in-sample optimal) calibrated increment alpha*deta_pde beats
    persistence in pooled RMSE.
Same for the current-vector increments (u,v stacked).

This is an in-sample (pooled-alpha) diagnostic of information content; the
strictly chronological calibration is in forecast_v2_calibration.py.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

OUT = "data_processed/forecast_v2"
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    "baseline_N1_c1.0":    (1, 1.0, 150.0, 10.0),
    "phys_N1_c2.45":       (1, 2.45, 60.0, 20.0),
    "phys_N2_c2.45":       (2, 2.45, 60.0, 20.0),
    "phys_N3_c2.45":       (3, 2.45, 60.0, 20.0),
    "fast_N2_c2.8":        (2, 2.8, 60.0, 20.0),
}


def pooled_stats(pde_list, true_list):
    """pde_list/true_list: lists of flattened increment arrays per window."""
    percorr = [np.corrcoef(p, t)[0, 1] for p, t in zip(pde_list, true_list)
               if p.std() > 0 and t.std() > 0]
    P = np.concatenate(pde_list)
    T = np.concatenate(true_list)
    pooled_corr = np.corrcoef(P, T)[0, 1]
    denom = np.sum(P * P)
    alpha = float(np.sum(P * T) / denom) if denom > 0 else 0.0
    # explained variance of alpha*P as predictor of T
    ss_tot = np.sum((T - T.mean()) ** 2)
    ss_res = np.sum((T - alpha * P) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    amp = float(P.std() / T.std()) if T.std() > 0 else np.nan
    # persistence RMSE on increments is just RMSE(0 vs T) = std-like
    rmse_persist = float(np.sqrt(np.mean(T ** 2)))
    rmse_raw = float(np.sqrt(np.mean((T - P) ** 2)))          # raw PDE increment
    rmse_cal = float(np.sqrt(np.mean((T - alpha * P) ** 2)))  # calibrated
    return dict(pooled_corr=pooled_corr, mean_window_corr=float(np.mean(percorr)),
                alpha=alpha, r2=r2, amp_ratio=amp,
                rmse_persist_inc=rmse_persist, rmse_rawpde_inc=rmse_raw,
                rmse_calpde_inc=rmse_cal,
                skill_raw=1 - rmse_raw / rmse_persist,
                skill_cal=1 - rmse_cal / rmse_persist)


def main():
    cube = fx.Cube(vel_source="total")
    FINAL_START = cube.nt - 3
    starts = np.arange(2, FINAL_START - 2)   # 38 rolling starts (same as baseline)
    rows = []
    for name, params in CONFIGS.items():
        for L, lead in [(1, 7), (2, 14)]:
            eta_pde, eta_true = [], []
            u_pde, u_true = [], []
            failed = 0
            for s in starts:
                try:
                    fc = fx.run_forecast(cube, s, params, scheme="upwind", leads=(1, 2))
                except Exception:
                    failed += 1
                    continue
                d = fc["leads"][L]
                eta_pde.append(d["deta_pde"].ravel())
                eta_true.append(d["deta_true"].ravel())
                # vector increment: stack u and v
                u_pde.append(np.concatenate([d["du_pde"].ravel(), d["dv_pde"].ravel()]))
                u_true.append(np.concatenate([d["du_true"].ravel(), d["dv_true"].ravel()]))
            if not eta_pde:
                rows.append(dict(config=name, params=str(params), lead_days=lead,
                                 field="ssh", n_windows=0, failed=failed))
                continue
            s_ssh = pooled_stats(eta_pde, eta_true)
            s_uv = pooled_stats(u_pde, u_true)
            rows.append(dict(config=name, params=str(params), lead_days=lead,
                             field="ssh", n_windows=len(eta_pde), failed=failed, **s_ssh))
            rows.append(dict(config=name, params=str(params), lead_days=lead,
                             field="uv", n_windows=len(u_pde), failed=failed, **s_uv))
            print(f"{name:18s} lead={lead:2d}  SSH: poolcorr={s_ssh['pooled_corr']:+.3f} "
                  f"wincorr={s_ssh['mean_window_corr']:+.3f} alpha={s_ssh['alpha']:.3f} "
                  f"R2={s_ssh['r2']:+.4f} amp={s_ssh['amp_ratio']:.2f} "
                  f"skill_cal={s_ssh['skill_cal']:+.4f} | UV poolcorr={s_uv['pooled_corr']:+.3f} "
                  f"alpha={s_uv['alpha']:.3f} skill_cal={s_uv['skill_cal']:+.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT + "/increment_information_content.csv", index=False)
    print("\nSaved", OUT + "/increment_information_content.csv")


if __name__ == "__main__":
    main()
