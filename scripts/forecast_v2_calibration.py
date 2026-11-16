"""
forecast_v2_calibration.py

Strictly chronological (rolling-origin) out-of-sample test of whether the PDE
increment carries USABLE forecast skill once calibrated using ONLY past data.

For each forecast start t_i (chronological), using ONLY windows strictly before
t_i, we fit per-lead coefficients and then forecast t_i and score vs persistence:

  M0 persistence        : eta_f = eta_0
  M1 raw PDE (alpha=1)  : eta_f = eta_0 + deta_pde                (the dynamical model)
  M2 calibrated PDE     : eta_f = eta_0 + a * deta_pde            (a from past, ridge)
  M3 tendency           : eta_f = eta_0 + b * (eta_0 - eta_m1)    (b from past)
  M4 hybrid PDE+tendency: eta_f = eta_0 + a*deta_pde + b*dtend    (a,b from past, ridge)

Coefficients are fit by (ridge-regularized) least squares pooling all grid cells
of all PAST windows for the given lead. The PDE increment source is a fixed,
physically-motivated configuration chosen a priori (realistic equatorial wave
speed, N_bg>=2 so an observed transient can propagate), NOT chosen by test skill.

Outputs rolling_predictions.csv (per start/lead/model) and model_comparison.csv
(aggregate skill, wins, medians) and calibration_coefficients.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

OUT = "data_processed/forecast_v2"
os.makedirs(OUT, exist_ok=True)

PDE_CONFIG = (3, 2.45, 60.0, 20.0)   # physically motivated, fixed a priori
PDE_CONFIG_NAME = "N3_c2.45_Hw60_Td20"
RIDGE = 1e-3                          # ridge on normalized increments
MIN_TRAIN = 8                        # need this many past windows before scoring


def fit_coef(X_list, y_list, ridge=RIDGE):
    """Ridge LS for y ~ sum_k a_k X_k (no intercept) pooling all cells/windows.
    X_list: list of (ncols,) feature stacks per window? Here we pass arrays:
    X is (N, k), y is (N,)."""
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    # normalize columns to make ridge scale-fair; return coefs in original units
    scales = X.std(axis=0)
    scales[scales == 0] = 1.0
    Xs = X / scales
    A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1]) * len(y)
    b = Xs.T @ y
    coef_s = np.linalg.solve(A, b)
    return coef_s / scales


def main():
    cube = fx.Cube(vel_source="total")
    FINAL_START = cube.nt - 3
    starts = list(range(2, FINAL_START - 2))   # 38 rolling starts

    # Precompute per-start increment fields (cached integrator).
    feat = {}   # start -> lead -> dict of flattened arrays
    for s in starts:
        fc = fx.run_forecast(cube, s, PDE_CONFIG, scheme="upwind", leads=(1, 2))
        feat[s] = {}
        for L in (1, 2):
            d = fc["leads"][L]
            dtend = (cube.ETA[s] - cube.ETA[s - 1]).ravel()
            feat[s][L] = dict(
                deta_pde=d["deta_pde"].ravel(),
                dtend=dtend,
                deta_true=d["deta_true"].ravel(),
                eta0=cube.ETA[s].ravel(),
                eta_true=cube.ETA[s].ravel() + d["deta_true"].ravel(),
            )

    rows = []
    coef_rows = []
    for L, lead in [(1, 7), (2, 14)]:
        for i, s in enumerate(starts):
            past = starts[:i]
            if len(past) < MIN_TRAIN:
                continue
            # assemble training features from past windows
            Xp = [feat[p][L]["deta_pde"][:, None] for p in past]
            Xt = [feat[p][L]["dtend"][:, None] for p in past]
            Xh = [np.column_stack([feat[p][L]["deta_pde"], feat[p][L]["dtend"]]) for p in past]
            y = [feat[p][L]["deta_true"] for p in past]
            a_pde = float(fit_coef(Xp, y)[0])
            b_tend = float(fit_coef(Xt, y)[0])
            ch = fit_coef(Xh, y)
            coef_rows.append(dict(start_date=str(cube.dates[s].date()), lead_days=lead,
                                  a_pde=a_pde, b_tend=b_tend, hyb_a=float(ch[0]),
                                  hyb_b=float(ch[1]), n_train=len(past)))
            # forecast target
            d = feat[s][L]
            eta_true = d["eta_true"]
            eta0 = d["eta0"]
            preds = {
                "persistence": eta0,
                "raw_pde": eta0 + d["deta_pde"],
                "cal_pde": eta0 + a_pde * d["deta_pde"],
                "tendency": eta0 + b_tend * d["dtend"],
                "hybrid": eta0 + ch[0] * d["deta_pde"] + ch[1] * d["dtend"],
            }
            rp = np.sqrt(np.mean((eta0 - eta_true) ** 2))
            for name, p in preds.items():
                rm = np.sqrt(np.mean((p - eta_true) ** 2))
                rows.append(dict(start_date=str(cube.dates[s].date()), lead_days=lead,
                                 model=name, ssh_rmse=rm, ssh_rmse_persist=rp,
                                 ssh_skill=1 - rm / rp,
                                 ssh_corr=np.corrcoef(p, eta_true)[0, 1]))

    pred = pd.DataFrame(rows)
    pred.to_csv(OUT + "/rolling_predictions.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT + "/calibration_coefficients.csv", index=False)

    # aggregate
    comp = []
    for lead in (7, 14):
        for name in ["persistence", "raw_pde", "cal_pde", "tendency", "hybrid"]:
            q = pred[(pred.lead_days == lead) & (pred.model == name)]
            comp.append(dict(
                lead_days=lead, model=name, n=len(q),
                mean_skill=q.ssh_skill.mean(), median_skill=q.ssh_skill.median(),
                wins=int((q.ssh_skill > 0).sum()), frac_wins=float((q.ssh_skill > 0).mean()),
                mean_rmse_m=q.ssh_rmse.mean(), worst_skill=q.ssh_skill.min()))
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT + "/model_comparison.csv", index=False)
    print("Chronological out-of-sample SSH skill vs persistence")
    print(f"PDE config = {PDE_CONFIG_NAME}; MIN_TRAIN={MIN_TRAIN}; "
          f"scored starts/lead = {len(pred[(pred.lead_days==7)&(pred.model=='persistence')])}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
