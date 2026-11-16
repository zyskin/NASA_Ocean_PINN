"""
forecast_v2_figures.py  --  presentation figures for the v2 forecast study.
Reads the saved CSVs in data_processed/forecast_v2/ and re-runs a couple of
cached forecasts for the example maps. Figures -> figures/forecast_v2/.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import forecast_v2_lib as fx

D = "data_processed/forecast_v2"
F = "figures/forecast_v2"
os.makedirs(F, exist_ok=True)
PDE_CONFIG = (3, 2.45, 60.0, 20.0)


def fig_rolling_skill():
    pred = pd.read_csv(D + "/rolling_predictions.csv")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, lead in zip(axes, (7, 14)):
        for m, c in [("persistence", "k"), ("tendency", "tab:gray"),
                     ("cal_pde", "tab:blue"), ("hybrid", "tab:red")]:
            q = pred[(pred.lead_days == lead) & (pred.model == m)].copy()
            q["start_date"] = pd.to_datetime(q.start_date)
            ax.plot(q.start_date, q.ssh_skill, marker="o", ms=4, color=c, label=m)
        ax.axhline(0, color="k", lw=0.8, ls="--")
        ax.set_ylabel("SSH skill vs persistence")
        ax.set_title(f"{lead}-day lead" + ("  (clean: no obs-window overlap)" if lead == 14
                     else "  (note: init/target obs windows overlap ~3 d)"))
        ax.legend(ncol=4, fontsize=8)
    axes[-1].set_xlabel("Forecast start date")
    fig.suptitle("Chronological out-of-sample SSH forecast skill (45-date cube)")
    fig.tight_layout()
    fig.savefig(F + "/fig1_rolling_skill.png", dpi=160)
    plt.close(fig)


def fig_model_ci():
    ci = pd.read_csv(D + "/bootstrap_ci.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, field in zip(axes, ("ssh", "uv")):
        sub = ci[ci.field == field]
        models = ["raw_pde", "cal_pde", "tendency", "hybrid"] if field == "ssh" \
            else ["raw_pde", "cal_pde", "hybrid"]
        x = np.arange(len(models)); w = 0.38
        for k, lead in enumerate((7, 14)):
            vals, los, his = [], [], []
            for m in models:
                r = sub[(sub.model == m) & (sub.lead_days == lead)]
                if len(r):
                    vals.append(r.mean_skill.values[0]); los.append(r.ci95_lo.values[0]); his.append(r.ci95_hi.values[0])
                else:
                    vals.append(np.nan); los.append(np.nan); his.append(np.nan)
            vals = np.array(vals); los = np.array(los); his = np.array(his)
            ax.bar(x + (k - 0.5) * w, vals, w, label=f"{lead}-day",
                   yerr=[vals - los, his - vals], capsize=3,
                   color=("tab:blue" if lead == 7 else "tab:red"), alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=20)
        ax.set_title(f"{field.upper()} skill vs persistence (95% block-bootstrap CI)")
        ax.set_ylabel("mean skill")
        ax.legend()
    fig.suptitle("Out-of-sample skill with uncertainty — dynamics help currents (sig.) "
                 "and 7-day SSH; 14-day SSH positive but not significant")
    fig.tight_layout()
    fig.savefig(F + "/fig2_model_comparison_ci.png", dpi=160)
    plt.close(fig)


def fig_increment_info():
    inc = pd.read_csv(D + "/increment_information_content.csv")
    order = ["baseline_N1_c1.0", "phys_N1_c2.45", "phys_N2_c2.45", "fast_N2_c2.8", "phys_N3_c2.45"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, fld in zip(axes, ("ssh", "uv")):
        sub = inc[inc.field == fld]
        x = np.arange(len(order)); w = 0.38
        for k, lead in enumerate((7, 14)):
            vals = [sub[(sub.config == c) & (sub.lead_days == lead)].pooled_corr.values[0]
                    if len(sub[(sub.config == c) & (sub.lead_days == lead)]) else np.nan for c in order]
            ax.bar(x + (k - 0.5) * w, vals, w, label=f"{lead}-day",
                   color=("tab:blue" if lead == 7 else "tab:red"), alpha=0.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=8)
        ax.set_title(f"{fld.upper()} increment pattern corr (PDE vs observed)")
        ax.set_ylabel("pooled pattern correlation")
        ax.legend()
    fig.suptitle("PDE-increment information content rises with admitted dynamics "
                 "(N_bg, wave speed); the tuned baseline (N1,c=1) suppresses it")
    fig.tight_layout()
    fig.savefig(F + "/fig3_increment_information.png", dpi=160)
    plt.close(fig)


def _hybrid_field(cube, start, lead_L):
    coef = pd.read_csv(D + "/calibration_coefficients.csv")
    lead = 7 * lead_L
    r = coef[(coef.start_date == str(cube.dates[start].date())) & (coef.lead_days == lead)]
    fc = fx.run_forecast(cube, start, PDE_CONFIG, scheme="upwind", leads=(1, 2))
    d = fc["leads"][lead_L]
    dtend = cube.ETA[start] - cube.ETA[start - 1]
    if len(r):
        a, b = float(r.hyb_a.values[0]), float(r.hyb_b.values[0])
    else:
        a, b = 0.4, 0.3
    hyb = cube.ETA[start] + a * d["deta_pde"] + b * dtend
    return d, hyb


def fig_example_maps():
    cube = fx.Cube(vel_source="total")
    pred = pd.read_csv(D + "/rolling_predictions.csv")
    hyb14 = pred[(pred.lead_days == 14) & (pred.model == "hybrid")].copy()
    hyb14["start_date"] = pd.to_datetime(hyb14.start_date)
    best = hyb14.loc[hyb14.ssh_skill.idxmax()]
    worst = hyb14.loc[hyb14.ssh_skill.idxmin()]
    date2idx = {str(d.date()): i for i, d in enumerate(cube.dates)}
    for tag, row in [("success", best), ("failure", worst)]:
        s = date2idx[str(row.start_date.date())]
        d, hyb = _hybrid_field(cube, s, 2)
        obs = d["eta_true"]; persist = d["eta_start"]; err = hyb - obs
        vmax = np.percentile(np.abs(obs), 98)
        emax = max(np.percentile(np.abs(err), 98), 1e-4)
        fig, ax = plt.subplots(1, 4, figsize=(20, 4.2))
        panels = [(obs, "Observed (t0+14d)", vmax, "RdBu_r"),
                  (persist, "Persistence (t0)", vmax, "RdBu_r"),
                  (hyb, "Hybrid forecast", vmax, "RdBu_r"),
                  (err, "Forecast - obs", emax, "RdBu_r")]
        for a, (arr, t, vm, cm) in zip(ax, panels):
            p = a.pcolormesh(cube.lon, cube.lat, arr, cmap=cm, vmin=-vm, vmax=vm, shading="auto")
            a.axhline(0, color="k", lw=0.5, ls="--"); a.set_title(t, fontsize=10)
            a.set_xlabel("lon"); plt.colorbar(p, ax=a, fraction=0.025)
        fig.suptitle(f"Representative 14-day {tag}: start {row.start_date.date()}  "
                     f"hybrid SSH skill={row.ssh_skill:+.3f}")
        fig.tight_layout()
        fig.savefig(F + f"/fig{'4' if tag=='success' else '5'}_example_{tag}.png", dpi=150)
        plt.close(fig)


def fig_regional():
    reg = pd.read_csv(D + "/regional_skill.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    regions = reg.region.unique()
    x = np.arange(len(regions)); w = 0.2
    for k, (m, lead, c) in enumerate([("cal_pde", 7, "tab:cyan"), ("cal_pde", 14, "tab:blue"),
                                      ("hybrid", 7, "tab:orange"), ("hybrid", 14, "tab:red")]):
        vals = [reg[(reg.region == r) & (reg.model == m) & (reg.lead_days == lead)].mean_skill.values[0]
                for r in regions]
        ax.bar(x + (k - 1.5) * w, vals, w, label=f"{m} {lead}d", color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([r.replace("_", "\n") for r in regions], fontsize=8)
    ax.set_ylabel("SSH skill vs persistence")
    ax.set_title("Regional out-of-sample SSH skill (hybrid & calibrated-PDE)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(F + "/fig6_regional_skill.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_rolling_skill(); print("fig1 done")
    fig_model_ci(); print("fig2 done")
    fig_increment_info(); print("fig3 done")
    fig_example_maps(); print("fig4/5 done")
    fig_regional(); print("fig6 done")
    print("figures in", F)
