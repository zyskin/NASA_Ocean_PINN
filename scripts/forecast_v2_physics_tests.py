"""
forecast_v2_physics_tests.py

Targeted, cheap physics/numerics tests on the 45-date cube, all using the fixed
physically-motivated config (N_bg=3, c=2.45, Hw=60, Td=20). For each variant we
report the PDE-increment information content (pooled pattern corr vs the true
increment) and the strictly-chronological hybrid out-of-sample SSH & UV skill at
14 days (the clean, non-leaking lead). Variants:
  * advection: first-order upwind  vs  second-order MUSCL/minmod
  * wind forcing: ON (oracle GEOS-FP)  vs  OFF (pure ocean dynamics, Exp. A)
  * velocity source: OSCAR total (u,v)  vs  OSCAR geostrophic (ug,vg)
"""
import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

OUT = "data_processed/forecast_v2"
CFG = (3, 2.45, 60.0, 20.0)
RIDGE = 1e-3
MIN_TRAIN = 8


def fit_coef(X, y, ridge=RIDGE):
    scales = X.std(axis=0); scales[scales == 0] = 1.0
    Xs = X / scales
    A = Xs.T @ Xs + ridge * np.eye(Xs.shape[1]) * len(y)
    return np.linalg.solve(A, Xs.T @ y) / scales


def eval_variant(cube, scheme):
    starts = list(range(2, cube.nt - 3 - 2))
    # gather increments
    feat = {}
    ok = 0
    for s in starts:
        try:
            fc = fx.run_forecast(cube, s, CFG, scheme=scheme, leads=(1, 2))
        except Exception:
            continue
        ok += 1
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
    res = {}
    for L, lead in [(1, 7), (2, 14)]:
        P = np.concatenate([feat[s][L]["deta_pde"] for s in svalid])
        T = np.concatenate([feat[s][L]["deta_true"] for s in svalid])
        res[f"ssh_inc_corr_{lead}"] = float(np.corrcoef(P, T)[0, 1])
        Pu = np.concatenate([feat[s][L]["duv_pde"] for s in svalid])
        Tu = np.concatenate([feat[s][L]["duv_true"] for s in svalid])
        res[f"uv_inc_corr_{lead}"] = float(np.corrcoef(Pu, Tu)[0, 1])
    # chronological hybrid skill at 14d (ssh and uv)
    for field, lead in [("ssh", 2), ("uv", 2)]:
        skills = []
        for i, s in enumerate(svalid):
            past = svalid[:i]
            if len(past) < MIN_TRAIN:
                continue
            if field == "ssh":
                Xh = np.vstack([np.column_stack([feat[p][lead]["deta_pde"], feat[p][lead]["dtend"]]) for p in past])
                y = np.concatenate([feat[p][lead]["deta_true"] for p in past])
                ch = fit_coef(Xh, y)
                d = feat[s][lead]
                true = d["eta0"] + d["deta_true"]; base = d["eta0"]
                pred = base + ch[0] * d["deta_pde"] + ch[1] * d["dtend"]
            else:
                Xh = np.vstack([np.column_stack([feat[p][lead]["duv_pde"], feat[p][lead]["dtend_uv"]]) for p in past])
                y = np.concatenate([feat[p][lead]["duv_true"] for p in past])
                ch = fit_coef(Xh, y)
                d = feat[s][lead]
                true = d["uv0"] + d["duv_true"]; base = d["uv0"]
                pred = base + ch[0] * d["duv_pde"] + ch[1] * d["dtend_uv"]
            rp = np.sqrt(np.mean((base - true) ** 2))
            skills.append(1 - np.sqrt(np.mean((pred - true) ** 2)) / rp)
        res[f"hybrid_{field}_skill_14"] = float(np.mean(skills))
    res["n_starts"] = ok
    return res


def main():
    variants = [
        ("upwind_total_wind", dict(vel_source="total", wind=True), "upwind"),
        ("muscl_total_wind", dict(vel_source="total", wind=True), "muscl"),
        ("upwind_total_NOwind", dict(vel_source="total", wind=False), "upwind"),
        ("upwind_geostrophic_wind", dict(vel_source="geostrophic", wind=True), "upwind"),
    ]
    rows = []
    for label, kw, scheme in variants:
        cube = fx.Cube(**kw)
        r = eval_variant(cube, scheme)
        r["variant"] = label
        rows.append(r)
        print(f"{label:26s} SSHinc(7/14)={r['ssh_inc_corr_7']:+.3f}/{r['ssh_inc_corr_14']:+.3f} "
              f"UVinc(7/14)={r['uv_inc_corr_7']:+.3f}/{r['uv_inc_corr_14']:+.3f} "
              f"hybrid14 ssh={r['hybrid_ssh_skill_14']:+.4f} uv={r['hybrid_uv_skill_14']:+.4f}")
    df = pd.DataFrame(rows)[["variant", "n_starts", "ssh_inc_corr_7", "ssh_inc_corr_14",
                             "uv_inc_corr_7", "uv_inc_corr_14",
                             "hybrid_ssh_skill_14", "hybrid_uv_skill_14"]]
    df.to_csv(OUT + "/physics_variant_tests.csv", index=False)
    print("\nSaved", OUT + "/physics_variant_tests.csv")


if __name__ == "__main__":
    main()
