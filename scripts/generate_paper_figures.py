"""Generate publication-quality figures for the ablation study paper"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

EM_COLOR  = "#2166ac"   # blue
F1_COLOR  = "#d6604d"   # red-orange
REF_COLOR = "#888888"   # grey for reference lines

# ── Figure 1: ablation bar chart ──────────────────────────────────────────────
variants = [
    "baseline",
    "no-decomp",
    "no-reranker",
    "steps-1",
    "steps-2",
    "steps-3",
    "dense-only",
    "sparse-only",
    "hybrid-only",
    "agentic\n(full)",
]
em = [43.1, 51.9, 51.5, 46.1, 52.9, 53.2, 53.0, 53.0, 55.0, 53.2]
f1 = [54.0, 60.1, 59.4, 53.9, 61.3, 61.6, 61.9, 61.2, 63.5, 61.6]

x      = np.arange(len(variants))
width  = 0.35

fig, ax = plt.subplots(figsize=(7.5, 3.8))

bars_em = ax.bar(x - width / 2, em, width, label="EM",
                 color=EM_COLOR, alpha=0.88, zorder=3)
bars_f1 = ax.bar(x + width / 2, f1, width, label="F1",
                 color=F1_COLOR, alpha=0.88, zorder=3)

# reference lines: full agentic pipeline
ax.axhline(53.2, color=EM_COLOR, linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
ax.axhline(61.6, color=F1_COLOR, linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)

# annotate hybrid-only bars (the standout result)
for bar, val in [(bars_em[8], 55.0), (bars_f1[8], 63.5)]:
    ax.annotate(
        f"{val:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, val),
        xytext=(0, 3), textcoords="offset points",
        ha="center", va="bottom", fontsize=7.5, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(variants, rotation=30, ha="right")
ax.set_ylabel("Score (%)")
ax.set_ylim(38, 68)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right")

fig.tight_layout()
out1 = OUT_DIR / "fig_ablation_bar.pdf"
fig.savefig(out1)
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: retrieval step depth curve ─────────────────────────────────────
steps     = [1, 2, 3, 5]
em_steps  = [46.1, 52.9, 53.2, 53.2]
f1_steps  = [53.9, 61.3, 61.6, 61.6]

fig, ax = plt.subplots(figsize=(4.2, 3.0))

ax.plot(steps, em_steps, marker="o", color=EM_COLOR, linewidth=1.8,
        markersize=6, label="EM", zorder=3)
ax.plot(steps, f1_steps, marker="s", color=F1_COLOR, linewidth=1.8,
        markersize=6, label="F1", zorder=3)

# shade the "plateau" region (steps 2-5)
ax.axvspan(2, 5.3, alpha=0.06, color="green", zorder=1)
ax.text(3.4, 54.8, "plateau", fontsize=8, color="green", alpha=0.7, style="italic")

ax.set_xlabel("Retrieval Loop Depth (steps)")
ax.set_ylabel("Score (%)")
ax.set_xticks(steps)
ax.set_xticklabels(["1", "2", "3", "5 (full)"])
ax.set_ylim(44, 66)
ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="lower right")

fig.tight_layout()
out2 = OUT_DIR / "fig_steps_curve.pdf"
fig.savefig(out2)
plt.close(fig)
print(f"Saved: {out2}")

# ── Figure 3: routing strategy distribution (horizontal bar) ─────────────────
strategies = ["Dense", "Hybrid\n(RRF)", "Sparse\n(BM25)"]
pcts       = [4.5, 16.3, 79.2]
colors_bar = ["#74add1", "#2166ac", "#d6604d"]

fig, ax = plt.subplots(figsize=(4.2, 2.2))
bars = ax.barh(strategies, pcts, color=colors_bar, height=0.5,
               edgecolor="white", linewidth=0.6)

for bar, pct in zip(bars, pcts):
    ax.text(pct + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", ha="left", fontsize=9)

ax.set_xlabel("Percentage of queries (%)", fontsize=9)
ax.set_xlim(0, 95)
ax.set_title("Routing strategy distribution\n(agentic full, n=5,000)", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.grid(True, linestyle=":", alpha=0.5)
ax.set_axisbelow(True)

fig.tight_layout()
out3 = OUT_DIR / "fig_routing_pie.pdf"
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {out3}")

print("\nDone. Upload PDFs from results/figures/ to Overleaf.")
