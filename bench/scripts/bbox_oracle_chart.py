"""One-off pre-study analysis script (not part of the package); see bench/reports/prestudy-lean/review.md."""

# ruff: noqa: N806, N816, E741, E402, B905, B007, E501
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
out = sys.argv[2]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
GEO, VLM, BASE = "#2a78d6", "#eb6834", "#8a8984"
rows = [
    (
        "B0: fruit-type mean (no image)",
        d["rows"]["B0 type mean"]["mape"],
        d["rows"]["B0 type mean"]["mape_ci"],
        BASE,
    ),
    ("claude-sonnet-4.6, S1", 0.277, None, VLM),
    ("gemini-3.5-flash, S1", 0.199, None, VLM),
    (
        "bbox geometry, side view",
        d["rows"]["bbox side-only"]["mape"],
        d["rows"]["bbox side-only"]["mape_ci"],
        GEO,
    ),
    (
        "bbox geometry, top view",
        d["rows"]["bbox top-only"]["mape"],
        d["rows"]["bbox top-only"]["mape_ci"],
        GEO,
    ),
    (
        "bbox geometry, top + side",
        d["rows"]["bbox top+side"]["mape"],
        d["rows"]["bbox top+side"]["mape_ci"],
        GEO,
    ),
]
plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
    }
)
fig, ax = plt.subplots(figsize=(8, 3.9), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
for i, (lab, v, ci, c) in enumerate(rows):
    if ci:
        ax.plot(
            [ci[0] * 100, ci[1] * 100], [i, i], color=c, lw=2, solid_capstyle="round", alpha=0.55
        )
    ax.plot(v * 100, i, "o", ms=8, color=c, mec=SURF, mew=2, zorder=3)
    ax.text(v * 100 + 1.2, i + 0.22, f"{v * 100:.1f}%", color=INK2, fontsize=9)
ax.axvline(rows[0][1] * 100, color=BASE, lw=1, ls=(0, (3, 3)))
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Hold-out MAPE, whole-fruit weight (%) · bars: 95% bootstrap CI over objects")
ax.set_xlim(0, 40)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0)
fig.suptitle(
    "Geometry from coin-scaled boxes beats both VLMs and the type-mean baseline",
    x=0.02,
    ha="left",
    fontsize=11,
    color=INK,
)
from matplotlib.lines import Line2D

fig.legend(
    handles=[
        Line2D([], [], marker="o", ls="", color=c, label=l)
        for c, l in (
            (GEO, "geometry (ground-truth boxes)"),
            (VLM, "VLM direct estimate"),
            (BASE, "baseline"),
        )
    ],
    loc="upper left",
    bbox_to_anchor=(0.01, 0.94),
    ncol=3,
    frameon=False,
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(out, facecolor=SURF)
