"""fig7: 45-date vs 201-date out-of-sample skill (power gain from the extension)."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = "data_processed/forecast_v2"
F = "figures/forecast_v2"
ci45 = pd.read_csv(D + "/bootstrap_ci.csv")
ci201 = pd.read_csv(D + "/extended_model_comparison.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, field in zip(axes, ("ssh", "uv")):
    models = ["cal_pde", "hybrid"]
    x = np.arange(len(models)); w = 0.2
    series = [("45-date 7d", ci45, 7, "tab:cyan"), ("45-date 14d", ci45, 14, "tab:blue"),
              ("201-date 7d", ci201, 7, "tab:orange"), ("201-date 14d", ci201, 14, "tab:red")]
    for k, (lab, df, lead, col) in enumerate(series):
        vals, los, his = [], [], []
        for m in models:
            r = df[(df.field == field) & (df.model == m) & (df.lead_days == lead)]
            if len(r):
                vals.append(r.mean_skill.values[0]); los.append(r.ci95_lo.values[0]); his.append(r.ci95_hi.values[0])
            else:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
        vals = np.array(vals); los = np.array(los); his = np.array(his)
        ax.bar(x + (k - 1.5) * w, vals, w, label=lab, color=col,
               yerr=[vals - los, his - vals], capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_title(f"{field.upper()} skill vs persistence")
    ax.set_ylabel("mean skill (95% block-bootstrap CI)")
    ax.legend(fontsize=8)
fig.suptitle("Historical extension (45 -> 201 weekly dates, no wind): the clean 14-day SSH\n"
             "skill becomes large and significant — the short record was power-limited")
fig.tight_layout()
fig.savefig(F + "/fig7_extended_power.png", dpi=160)
print("saved", F + "/fig7_extended_power.png")
