"""
forecast_v2_extended_analysis.py

Repeat the PDE-increment information-content + strictly-chronological
calibration on the EXTENDED SSH+OSCAR record (2023-01..2026-10, 201 weekly
dates, Experiment A: NO wind forcing), to test whether the clean 14-day SSH
signal strengthens / becomes significant with ~4.4x more data and more ENSO
regimes than the 45-date cube.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

OUT = "data_processed/forecast_v2"
EXT = OUT + "/ssh_oscar_weekly_ext.nc"
CFG = (3, 2.45, 60.0, 20.0)
RIDGE = 1e-3
MIN_TRAIN = 12
rng = np.random.default_rng(7)


def fit_coef(X, y, ridge=RIDGE):
    scales = X.std(axis=0); scales[scales == 0] = 1.0
    Xs = X / scales
    A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1]) * len(y)
    return np.linalg.solve(A, Xs.T @ y) / scales


def block_ci(v, block=4, nboot=3000):
    v = np.asarray(v, float); n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    nb = int(np.ceil(n / block)); means = np.empty(nboot)
    for b in range(nboot):
        s = rng.integers(0, n, size=nb)
        idx = np.concatenate([np.arange(k, k + block) % n for k in s])[:n]
        means[b] = v[idx].mean()
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    cube = fx.Cube(file=EXT, vel_source="total", wind=False)
    print(f"extended cube: {cube.nt} dates {cube.dates[0].date()}..{cube.dates[-1].date()} "
          f"(has_wind={cube.has_wind})")
    starts = list(range(2, cube.nt - 3 - 2))
    feat = {}
    for s in starts:
        try:
            fc = fx.run_forecast(cube, s, CFG, scheme="upwind", leads=(1, 2))
        except Exception:
            continue
        feat[s] = {}
        for L in (1, 2):
            d = fc["leads"][L]
            feat[s][L] = dict(
                deta_pde=d["deta_pde"].ravel(), deta_true=d["deta_true"].ravel(),
                dtend=(cube.ETA[s] - cube.ETA[s - 1]).ravel(), eta0=cube.ETA[s].ravel(),
                duv_pde=np.concatenate([d["du_pde"].ravel(), d["dv_pde"].ravel()]),
                duv_true=np.concatenate([d["du_true"].ravel(), d["dv_true"].ravel()]),
                dtend_uv=np.concatenate([(cube.U[s] - cube.U[s - 1]).ravel(),
                                         (cube.V[s] - cube.V[s - 1]).ravel()]),
                uv0=np.concatenate([cube.U[s].ravel(), cube.V[s].ravel()]))
    svalid = sorted(feat)
    print(f"valid forecast starts: {len(svalid)}")

    # increment pooled corr
    inc_rows = []
    for L, lead in [(1, 7), (2, 14)]:
        P = np.concatenate([feat[s][L]["deta_pde"] for s in svalid])
        T = np.concatenate([feat[s][L]["deta_true"] for s in svalid])
        Pu = np.concatenate([feat[s][L]["duv_pde"] for s in svalid])
        Tu = np.concatenate([feat[s][L]["duv_true"] for s in svalid])
        inc_rows.append(dict(lead_days=lead, ssh_inc_corr=float(np.corrcoef(P, T)[0, 1]),
                             uv_inc_corr=float(np.corrcoef(Pu, Tu)[0, 1]),
                             n_windows=len(svalid)))
    pd.DataFrame(inc_rows).to_csv(OUT + "/extended_increment_info.csv", index=False)

    # chronological skill series
    skill = {m: {7: [], 14: []} for m in ["raw_pde", "cal_pde", "tendency", "hybrid"]}
    uvskill = {m: {7: [], 14: []} for m in ["cal_pde", "hybrid"]}
    for L, lead in [(1, 7), (2, 14)]:
        for i, s in enumerate(svalid):
            past = svalid[:i]
            if len(past) < MIN_TRAIN:
                continue
            Xp = np.vstack([feat[p][L]["deta_pde"][:, None] for p in past])
            Xt = np.vstack([feat[p][L]["dtend"][:, None] for p in past])
            Xh = np.vstack([np.column_stack([feat[p][L]["deta_pde"], feat[p][L]["dtend"]]) for p in past])
            y = np.concatenate([feat[p][L]["deta_true"] for p in past])
            a = float(fit_coef(Xp, y)[0]); b = float(fit_coef(Xt, y)[0]); ch = fit_coef(Xh, y)
            d = feat[s][L]; true = d["eta0"] + d["deta_true"]; e0 = d["eta0"]
            rp = np.sqrt(np.mean((e0 - true) ** 2))
            for m, pr in [("raw_pde", e0 + d["deta_pde"]), ("cal_pde", e0 + a * d["deta_pde"]),
                          ("tendency", e0 + b * d["dtend"]),
                          ("hybrid", e0 + ch[0] * d["deta_pde"] + ch[1] * d["dtend"])]:
                skill[m][lead].append(1 - np.sqrt(np.mean((pr - true) ** 2)) / rp)
            # velocity
            Xpu = np.vstack([feat[p][L]["duv_pde"][:, None] for p in past])
            Xhu = np.vstack([np.column_stack([feat[p][L]["duv_pde"], feat[p][L]["dtend_uv"]]) for p in past])
            yu = np.concatenate([feat[p][L]["duv_true"] for p in past])
            au = float(fit_coef(Xpu, yu)[0]); chu = fit_coef(Xhu, yu)
            du = feat[s][L]; tru = du["uv0"] + du["duv_true"]; uv0 = du["uv0"]
            rpu = np.sqrt(np.mean((uv0 - tru) ** 2))
            uvskill["cal_pde"][lead].append(1 - np.sqrt(np.mean((uv0 + au * du["duv_pde"] - tru) ** 2)) / rpu)
            uvskill["hybrid"][lead].append(1 - np.sqrt(np.mean((uv0 + chu[0] * du["duv_pde"] + chu[1] * du["dtend_uv"] - tru) ** 2)) / rpu)

    rows = []
    for lead in (7, 14):
        for m in ["raw_pde", "cal_pde", "tendency", "hybrid"]:
            mean, lo, hi = block_ci(skill[m][lead])
            v = np.array(skill[m][lead])
            rows.append(dict(field="ssh", lead_days=lead, model=m, n=len(v),
                             mean_skill=mean, median_skill=float(np.median(v)),
                             wins=int((v > 0).sum()), frac_wins=float((v > 0).mean()),
                             ci95_lo=lo, ci95_hi=hi, significant=bool(lo > 0)))
        for m in ["cal_pde", "hybrid"]:
            mean, lo, hi = block_ci(uvskill[m][lead])
            v = np.array(uvskill[m][lead])
            rows.append(dict(field="uv", lead_days=lead, model=m, n=len(v),
                             mean_skill=mean, median_skill=float(np.median(v)),
                             wins=int((v > 0).sum()), frac_wins=float((v > 0).mean()),
                             ci95_lo=lo, ci95_hi=hi, significant=bool(lo > 0)))
    comp = pd.DataFrame(rows)
    comp.to_csv(OUT + "/extended_model_comparison.csv", index=False)
    print("\nEXTENDED (201-date, no-wind) increment pattern corr:")
    print(pd.DataFrame(inc_rows).to_string(index=False))
    print("\nEXTENDED chronological skill vs persistence (block-bootstrap 95% CI):")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
