"""Regenerate the article's two figures from results/revision.json.

fig_substitution.pdf -- static vs adaptive success under substitution constraints (Fig. 1)
fig_optimizer.pdf    -- greedy vs exhaustive set-level selection at k=3 (Fig. 2)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def bars(ax, labels, vals, colors, ylabel, title, ymax):
    b = ax.bar(labels, vals, color=colors)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + ymax * 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=11); ax.set_ylim(0, ymax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--out", default=str(HERE / "figures"))
    a = ap.parse_args()
    rev = json.loads((Path(a.results) / "revision.json").read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ab = rev["A_substitution_ablation"]
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    bars(ax, ["static\n(BAS)", "adaptive\ncorpus", "adaptive\nplatform", "adaptive\nco-occur."],
         [ab["static_success"], ab["corpus"]["adaptive_success"], ab["platform"]["adaptive_success"],
          ab["cooccur"]["adaptive_success"]],
         ["#888888", "#1f77b4", "#2a9d8f", "#e76f51"], "Mean campaign success",
         "Adaptive reachability under substitution constraints", 0.44)
    fig.tight_layout(); fig.savefig(out / "fig_substitution.pdf"); plt.close(fig)

    op = rev["C_optimizer_k3"]
    fig, ax = plt.subplots(figsize=(5.0, 3.2), dpi=300)
    bars(ax, ["no\ncontrols", "greedy\n(per-control)", "exhaustive\noptimal (set)"],
         [op["no_controls_adaptive_breach"], op["greedy_adaptive_breach"], op["exhaustive_optimal_adaptive_breach"]],
         ["#999999", "#e76f51", "#2a9d8f"], "Adaptive breach, budget k=3",
         "Greedy stalls; set-level optimization closes the gap", 0.8)
    fig.tight_layout(); fig.savefig(out / "fig_optimizer.pdf"); plt.close(fig)
    print("wrote", out / "fig_substitution.pdf", "and", out / "fig_optimizer.pdf")


if __name__ == "__main__":
    main()
