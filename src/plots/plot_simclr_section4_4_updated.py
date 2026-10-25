"""Figure 4.3 (updated): SimCLR raw ASR over five downstream classifier seeds.

Uses the four final leakage-corrected SimCLR aggregate JSONs supplied by
user, reproduced below at full precision (five per-seed raw ASRs).
No JSON files are required to run this self-contained script.

Install: pip install numpy matplotlib
Run:     python plot_simclr_section4_4_updated.py
Produces: simclr_section4_4_asr_updated.png and .svg beside this script.

This figure follows the MoCo Figure 4.4 plotting style: same axes,
individual seed point offsets, mean diamond, and sample SD error bars.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Seeds appear in this exact order: 42, 43, 44, 45, 46.
# Original sources by folder/uploaded filename:
#  CIFAR→CIFAR: inactive_downstream_aggregate(1).json
#  CIFAR→STL:   inactive_downstream_aggregate.json
#  STL→CIFAR:   inactive_downstream_aggregate(3).json
#  STL→STL:     inactive_downstream_aggregate(2).json
ASR_BY_DIRECTION = {
    "CIFAR → CIFAR": [
        93.87777777777778, 94.1, 94.16666666666667,
        94.12222222222222, 94.54444444444444,
    ],
    "CIFAR → STL": [
        97.52777777777779, 97.375, 97.34722222222221,
        97.34722222222221, 98.15277777777777,
    ],
    "STL → CIFAR": [
        95.71111111111111, 96.21111111111111, 93.28888888888889,
        93.4, 93.64444444444445,
    ],
    "STL → STL": [
        99.3888888888889, 99.31944444444444, 99.40277777777777,
        99.41666666666666, 99.13888888888889,
    ],
}


def main():
    groups = list(ASR_BY_DIRECTION)
    data = np.array(list(ASR_BY_DIRECTION.values()), dtype=float)
    means = data.mean(axis=1)
    stds = data.std(axis=1, ddof=1)
    x = np.arange(len(groups), dtype=float)
    seed_offsets = np.array([-0.12, -0.06, 0.00, 0.06, 0.12])

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=150)
    for i in range(5):
        ax.scatter(x + seed_offsets[i], data[:, i], s=36, alpha=0.75,
                   zorder=3, label="Individual seeds" if i == 0 else None)

    ax.errorbar(x, means, yerr=stds, fmt="D", markersize=7, capsize=5,
                elinewidth=1.5, linewidth=0, zorder=4,
                label="Mean ± sample SD")

    for xi, m in zip(x, means):
        offset = -7.0 if m > 92 else 5.8
        va = "top" if m > 92 else "bottom"
        ax.annotate(f"{m:.2f}%", (xi, m), xytext=(0, offset),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x, groups)
    ax.set_xlim(-0.45, len(groups) - 0.55)
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Raw attack success rate (ASR, %)", fontsize=11)
    ax.set_xlabel("Upstream → downstream dataset", fontsize=11)
    ax.set_title("SimCLR raw ASR across dataset-transfer settings", fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.54), frameon=True, fontsize=9)
    fig.tight_layout(pad=1.6)

    outdir = Path(__file__).resolve().parent
    png = outdir / "simclr_section4_4_asr_updated.png"
    svg = outdir / "simclr_section4_4_asr_updated.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {svg}")
    for group, mean, sd in zip(groups, means, stds):
        print(f"{group}: {mean:.2f} ± {sd:.2f}%")


if __name__ == "__main__":
    main()
