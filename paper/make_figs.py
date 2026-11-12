"""Figures for the paper. Numbers in fig_failures come from the pooled result files named beside them.
usage: python paper/make_figs.py   (from the repository root) -> paper/figs/*.pdf"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
plt.rcParams.update({"font.size": 8, "font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
                     "axes.titlesize": 8, "legend.frameon": False, "pdf.fonttype": 42})
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#e6e5e0"
ADD, CUR, BOTH = "#8a8984", "#2a78d6", "#eb6834"  # add the failures (baseline) / curate / curate then add

# mean crash % and s.e. over seeds with 1 and 3 failure runs (None = not shown). Curation uses the sign rule (no
# removal count) everywhere; "both" = sign rule, then add the same failure runs at W=20.
PANELS = [
    # work/mis_prop50j8_s*.json via pool_quadrotor.py, 5 seeds
    ("(a) Propeller fault",
     dict(add=[(94.7, 3.6), (69.4, 3.8)], cur=[(3.1, 1.2), (2.5, 0.5)], both=[(2.5, 0.5), (2.8, 1.2)], none=98.9, oracle=1.9)),
    ("(b) Thrust-gain fault",
     dict(add=[(74.7, 2.9), (46.7, 3.0)], cur=[(1.9, 0.7), (1.9, 0.7)], both=[(1.9, 0.3), (1.7, 0.8)], none=91.1, oracle=1.9)),
    # work/mis_prop50j8m_s*.json, 3 seeds
    ("(c) Propeller fault, look-alikes",
     dict(add=[(81.0, 5.2), (26.4, 0.8)], cur=[(1.4, 0.8), (0.5, 0.5)], both=[(0.9, 0.9), (1.4, 0.8)], none=94.4, oracle=0.0)),
    # results/crazyflie/cf_3.1_eq_s*.json (equal gradient steps), 5 seeds
    ("(d) Crazyflie battery sag (real logs)",
     dict(add=[(43.1, 5.5), (10.0, 3.0)], cur=[(50.0, 11.5), (51.9, 5.5)], both=[(27.2, 9.2), (5.6, 1.8)], none=73.6,
          oracle=12.5)),
]


SHORT = {"(c) Propeller fault, look-alikes": "(c) Propeller, with look-alikes", "(d) Crazyflie battery sag (real logs)": "(d) Crazyflie, real logs",
         "(a) Propeller fault": "(a) Propeller", "(b) Thrust-gain fault": "(b) Thrust gain"}
ROWS = [("add", ADD, "Add the failures"), ("cur", CUR, "Curate (ours)"), ("both", BOTH, "Curate, then add (ours)")]


def fig_failures():
    """One row per fix and one panel per benchmark, on a linear axis: 1 failure run (open) to 3 (filled)."""
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.55), sharey=True)
    for ax, (title, d) in zip(axs, PANELS):
        n = len(ROWS)
        # reference lines: no fix and oracle
        ax.axvline(d["none"], color=INK, lw=0.6, ls=(0, (4, 2)), zorder=1)
        ax.axvline(d["oracle"], color=INK, lw=0.6, ls=(0, (1, 1.2)), zorder=1)
        for r, (key, col, lab) in enumerate(ROWS):
            (m1, s1), (m3, s3) = d[key]
            ax.plot([m1, m3], [r, r], color=col, lw=1.6, alpha=0.45, solid_capstyle="round", zorder=2)
            for m, se, filled in ((m1, s1, False), (m3, s3, True)):
                ax.errorbar(m, r, xerr=se, fmt="none", ecolor=col, elinewidth=0.6, capsize=0, zorder=3)
                ax.plot(m, r, "o", ms=4.6, mfc=col if filled else "white", mec=col, mew=1.0, zorder=4)
        ax.set_title(SHORT.get(title, title), pad=3, loc="left")
        ax.set_xlim(-3, 103); ax.set_xticks([0, 50, 100])
        ax.set_ylim(n - 0.5, -0.6)
        ax.set_yticks(range(n), [lab for *_, lab in ROWS])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.4); ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Crash rate (%)", labelpad=1)
    h = [plt.Line2D([], [], ls="none", marker="o", ms=4.6, mfc="white", mec=MUTED, mew=1.0),
         plt.Line2D([], [], ls="none", marker="o", ms=4.6, mfc=MUTED, mec=MUTED, mew=1.0),
         plt.Line2D([], [], color=INK, lw=0.6, ls=(0, (4, 2))), plt.Line2D([], [], color=INK, lw=0.6, ls=(0, (1, 1.2)))]
    l = ["1 failure run", "3 failure runs", "No fix (all logs)", "Oracle (bad logs removed)"]
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.6,
               handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.88), w_pad=0.6)
    fig.savefig(os.path.join(HERE, "figs", "fig_failures.pdf"), metadata={"CreationDate": None}); fig.savefig(os.path.join(HERE, "figs", "fig_failures.png"), dpi=300)



def fig_scores():
    plt.rcParams.update({"font.size": 7, "axes.titlesize": 7})
    z = json.load(open(os.path.join(ROOT, "results", "drone", "fig_scores_prop50j8m_s1.json")))
    bad, look = np.array(z["bad"]), np.array(z["lookalike"])
    other = ~bad & ~look
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.5), sharey=True)
    rng = np.random.default_rng(0)
    groups = [("Faulty (32)", bad, "#e34948"), ("Look-alike (16)", look, CUR), ("Other (47)", other, MUTED)]
    for ax, (key, title) in zip(axs, [("train_loss", "(a) Training loss (worst-fit)"), ("tracin", "(b) FACT, one crash dive")]):
        v = np.array(z[key]); v = (v - v.mean()) / v.std()
        cut = np.sort(v)[-40]
        for row, (lab, m, col) in enumerate(groups):
            ax.scatter(v[m], row + rng.uniform(-0.18, 0.18, m.sum()), s=9, color=col, edgecolors="white", linewidths=0.4, zorder=3)
        ax.axvline(cut, color=INK, lw=0.6, ls=(0, (3, 2)))
        ax.text(cut, 2.55, " top 40", fontsize=6, va="center", ha="left", color=INK)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xlabel("Standardised value", labelpad=1)
        ax.set_yticks(range(3), [g[0] for g in groups]); ax.set_ylim(2.8, -0.6)
        ax.grid(axis="x", color="#e6e5e0", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.6)
    fig.savefig(os.path.join(HERE, "figs", "fig_scores.pdf"), metadata={"CreationDate": None}); fig.savefig(os.path.join(HERE, "figs", "fig_scores.png"), dpi=300)


if __name__ == "__main__":
    os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
    fig_failures(); fig_scores()
    print("figures written")
