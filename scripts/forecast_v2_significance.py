"""
forecast_v2_significance.py

Uncertainty + regional + velocity analysis for the chronological forecast
comparison. Because adjacent weekly SSH maps are ~0.93 correlated, naive
per-window counting overstates independence; we use a MOVING-BLOCK BOOTSTRAP
over forecast starts (block length ~ decorrelation scale) for confidence
intervals on mean skill.

Also:
  * velocity (vector) skill for persistence / raw-PDE / calibrated-PDE / hybrid,
  * regional SSH skill (equatorial |lat|<=2, off-equatorial 2-8, and
    west/central/east thirds) for the hybrid and calibrated-PDE models.
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

OUT = "data_processed/forecast_v2"
PDE_CONFIG = (3, 2.45, 60.0, 20.0)
RIDGE = 1e-3
MIN_TRAIN = 8
rng = np.random.default_rng(12345)


def fit_coef(X, y, ridge=RIDGE):
    scales = X.std(axis=0); scales[scales == 0] = 1.0
    Xs = X / scales
    A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1]) * len(y)
    return np.linalg.solve(A, Xs.T @ y) / scales


def block_bootstrap_ci(values, block=4, nboot=3000):
    v = np.asarray(values, float)
    n = len(v)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    means = np.empty(nboot)
    nblocks = int(np.ceil(n / block))
    for b in range(nboot):
        starts = rng.integers(0, n, size=nblocks)
        idx = np.concatenate([np.arange(s, s + block) % n for s in starts])[:n]
        means[b] = v[idx].mean()
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    cube = fx.Cube(vel_source="total")
    FINAL_START = cube.nt - 3
    starts = list(range(2, FINAL_START - 2))
    ny, nx = cube.ny, cube.nx
    lat2d = np.broadcast_to(cube.lat[:, None], (ny, nx))
    lon2d = np.broadcast_to(cube.lon[None, :], (ny, nx))
    regions = {
        "equatorial_|lat|<=2": (np.abs(lat2d) <= 2).ravel(),
        "offeq_2to8": ((np.abs(lat2d) > 2) & (np.abs(lat2d) <= 8)).ravel(),
        "west_-170to-147": (lon2d <= -147).ravel(),
        "central_-147to-123": ((lon2d > -147) & (lon2d <= -123)).ravel(),
        "east_-123to-100": (lon2d > -123).ravel(),
    }

    feat = {}
    for s in starts:
        fc = fx.run_forecast(cube, s, PDE_CONFIG, scheme="upwind", leads=(1, 2))
        feat[s] = {}
        for L in (1, 2):
            d = fc["leads"][L]
            feat[s][L] = dict(
                deta_pde=d["deta_pde"].ravel(), deta_true=d["deta_true"].ravel(),
                dtend=(cube.ETA[s] - cube.ETA[s - 1]).ravel(),
                eta0=cube.ETA[s].ravel(),
                # velocity increments stacked [u;v]
                duv_pde=np.concatenate([d["du_pde"].ravel(), d["dv_pde"].ravel()]),
                duv_true=np.concatenate([d["du_true"].ravel(), d["dv_true"].ravel()]),
                dtend_uv=np.concatenate([(cube.U[s] - cube.U[s - 1]).ravel(),
                                         (cube.V[s] - cube.V[s - 1]).ravel()]),
                uv0=np.concatenate([cube.U[s].ravel(), cube.V[s].ravel()]),
            )

    # ---- per-window skill series (SSH + velocity) ----
    ssh_skill = {m: {7: [], 14: []} for m in
                 ["raw_pde", "cal_pde", "tendency", "hybrid"]}
    uv_skill = {m: {7: [], 14: []} for m in ["raw_pde", "cal_pde", "hybrid"]}
    region_skill = {r: {"cal_pde": {7: [], 14: []}, "hybrid": {7: [], 14: []}}
                    for r in regions}

    for L, lead in [(1, 7), (2, 14)]:
        for i, s in enumerate(starts):
            past = starts[:i]
            if len(past) < MIN_TRAIN:
                continue
            # SSH coefficients
            Xp = np.vstack([feat[p][L]["deta_pde"][:, None] for p in past])
            Xt = np.vstack([feat[p][L]["dtend"][:, None] for p in past])
            Xh = np.vstack([np.column_stack([feat[p][L]["deta_pde"], feat[p][L]["dtend"]]) for p in past])
            y = np.concatenate([feat[p][L]["deta_true"] for p in past])
            a_pde = float(fit_coef(Xp, y)[0])
            b_tend = float(fit_coef(Xt, y)[0])
            ch = fit_coef(Xh, y)
            # velocity coefficients
            Xpu = np.vstack([feat[p][L]["duv_pde"][:, None] for p in past])
            Xhu = np.vstack([np.column_stack([feat[p][L]["duv_pde"], feat[p][L]["dtend_uv"]]) for p in past])
            yu = np.concatenate([feat[p][L]["duv_true"] for p in past])
            a_uv = float(fit_coef(Xpu, yu)[0])
            chu = fit_coef(Xhu, yu)

            d = feat[s][L]
            etrue = d["eta0"] + d["deta_true"]
            e0 = d["eta0"]
            rp = np.sqrt(np.mean((e0 - etrue) ** 2))
            fcs = {
                "raw_pde": e0 + d["deta_pde"],
                "cal_pde": e0 + a_pde * d["deta_pde"],
                "tendency": e0 + b_tend * d["dtend"],
                "hybrid": e0 + ch[0] * d["deta_pde"] + ch[1] * d["dtend"],
            }
            for m, p in fcs.items():
                ssh_skill[m][lead].append(1 - np.sqrt(np.mean((p - etrue) ** 2)) / rp)
            # regional (cal_pde, hybrid)
            for rname, mask in regions.items():
                rpr = np.sqrt(np.mean((e0[mask] - etrue[mask]) ** 2))
                for m in ("cal_pde", "hybrid"):
                    pm = fcs[m]
                    region_skill[rname][m][lead].append(
                        1 - np.sqrt(np.mean((pm[mask] - etrue[mask]) ** 2)) / rpr)
            # velocity skill
            uvtrue = d["uv0"] + d["duv_true"]
            uv0 = d["uv0"]
            rpu = np.sqrt(np.mean((uv0 - uvtrue) ** 2))
            uv_fcs = {
                "raw_pde": uv0 + d["duv_pde"],
                "cal_pde": uv0 + a_uv * d["duv_pde"],
                "hybrid": uv0 + chu[0] * d["duv_pde"] + chu[1] * d["dtend_uv"],
            }
            for m, p in uv_fcs.items():
                uv_skill[m][lead].append(1 - np.sqrt(np.mean((p - uvtrue) ** 2)) / rpu)

    # ---- bootstrap CIs ----
    ci_rows = []
    for lead in (7, 14):
        for m in ["raw_pde", "cal_pde", "tendency", "hybrid"]:
            mean, lo, hi = block_bootstrap_ci(ssh_skill[m][lead])
            ci_rows.append(dict(field="ssh", lead_days=lead, model=m, n=len(ssh_skill[m][lead]),
                                mean_skill=mean, ci95_lo=lo, ci95_hi=hi,
                                significant=bool(lo > 0)))
        for m in ["raw_pde", "cal_pde", "hybrid"]:
            mean, lo, hi = block_bootstrap_ci(uv_skill[m][lead])
            ci_rows.append(dict(field="uv", lead_days=lead, model=m, n=len(uv_skill[m][lead]),
                                mean_skill=mean, ci95_lo=lo, ci95_hi=hi,
                                significant=bool(lo > 0)))
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT + "/bootstrap_ci.csv", index=False)
    print("Block-bootstrap 95% CIs on mean skill (block=4 starts):")
    print(ci.to_string(index=False))

    reg_rows = []
    for rname in regions:
        for m in ("cal_pde", "hybrid"):
            for lead in (7, 14):
                vals = region_skill[rname][m][lead]
                reg_rows.append(dict(region=rname, model=m, lead_days=lead,
                                     mean_skill=float(np.mean(vals)),
                                     median_skill=float(np.median(vals))))
    pd.DataFrame(reg_rows).to_csv(OUT + "/regional_skill.csv", index=False)
    print("\nSaved bootstrap_ci.csv and regional_skill.csv")


if __name__ == "__main__":
    main()
