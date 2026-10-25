"""Figure 4.4: MoCo raw ASR over five downstream-classifier seeds.

The numbers below are transcribed from the final 4 October 2026 MoCo
experiment log, Section 5 (five-seed raw ASR results). No JSON paths
or external files are required.

Usage:
    python plot_moco_section4_5.py
Requires:
    pip install numpy matplotlib
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Classifier-training seeds: 42, 43, 44, 45, 46.
ASR_BY_DIRECTION = {
    "CIFAR → CIFAR": [61.17, 70.34, 78.00, 1.02, 93.29],
    "CIFAR → STL":   [99.11, 99.24, 99.21, 99.35, 99.14],
    "STL → CIFAR":   [0.74, 3.32, 2.21, 2.02, 2.01],
    "STL → STL":     [99.71, 99.49, 99.60, 99.57, 99.65],
}


def main():
    groups = list(ASR_BY_DIRECTION)
    data = np.array(list(ASR_BY_DIRECTION.values()), dtype=float)
    means = np.mean(data, axis=1)
    stds = np.std(data, axis=1, ddof=1)
    x = np.arange(len(groups), dtype=float)
    # Constant offset is deliberate and deterministic: each point is one seed.
    seed_offsets = np.array([-0.12, -0.06, 0.00, 0.06, 0.12])

    fig, ax = plt.subplots(figsize=(8.6, 4.7), dpi=150)
    for i in range(5):
        ax.scatter(x + seed_offsets[i], data[:, i], s=36, alpha=0.75,
                   zorder=3, label="Individual seeds" if i == 0 else None)
    ax.errorbar(x, means, yerr=stds, fmt="D", markersize=7, capsize=5,
                elinewidth=1.5, linewidth=0, zorder=4,
                label="Mean ± sample SD")
    for xi, m in zip(x, means):
        # Place numeric means next to their markers; avoid chart edge at ~100%.
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
    ax.set_title("MoCo raw ASR across dataset-transfer settings", fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center", bbox_to_anchor=(0.52, 0.54), frameon=True, fontsize=9)
    fig.tight_layout(pad=1.6)

    outdir = Path(__file__).resolve().parent
    png = outdir / "moco_section4_5_asr.png"
    svg = outdir / "moco_section4_5_asr.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {svg}")
    for group, mean, sd in zip(groups, means, stds):
        print(f"{group}: {mean:.2f} +/- {sd:.2f}%")


if __name__ == "__main__":
    main()
