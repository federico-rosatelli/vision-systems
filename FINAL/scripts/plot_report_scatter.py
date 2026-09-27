"""Predicted vs. expert score scatter panels for the report (Figure: generalization)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRED = Path("outputs/tables/generalization/per_image_predictions.csv")
OUT = Path("report/figures/pred_vs_true.pdf")
MODEL = "pred_aggregation_weighted_seed42"
SERIES, INK, MUTED, GRID = "#2a78d6", "#1a1a19", "#6b6a64", "#e4e3dc"
PANELS = [
    ("GG1_calibration_test", "GG test (in-domain)"),
    ("GG1_heldout_disagreement", "GG, raters disagree"),
    ("WG1_Weilburger_Grenze", "Weilburger Grenze"),
    ("DSV_Trial_1", "DSV Asendorf"),
]

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
df = pd.read_csv(PRED)
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.05), sharex=True, sharey=True)
for ax, (key, title) in zip(axes, PANELS):
    d = df[df.dataset == key]
    y, p = d.target.to_numpy(), d[MODEL].to_numpy()
    ax.plot([0, 50], [0, 50], color=MUTED, lw=0.8, ls="--", zorder=1)
    ax.scatter(y, p, s=9, color=SERIES, alpha=0.55, linewidths=0.3, edgecolors="white", zorder=2)
    ax.set_title(title, fontsize=8, color=INK, pad=3)
    ax.text(0.04, 0.96, f"n={len(y)}\nMAE={np.abs(y - p).mean():.2f}\nρ={spearmanr(y, p)[0]:.2f}",
            transform=ax.transAxes, va="top", fontsize=7, color=INK)
    ax.set_xlim(-1, 50); ax.set_ylim(-1, 50)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Expert score (%)")
axes[0].set_ylabel("Predicted (%)")
fig.tight_layout(pad=0.3, w_pad=0.6)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
fig.savefig(OUT.with_suffix(".png"), dpi=150)
print("saved", OUT)
