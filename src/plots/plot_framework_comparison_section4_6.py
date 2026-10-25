"""Generate thesis Figure 4.5: SimCLR versus MoCo raw-ASR comparison.
All values below come from the final leakage-corrected SimCLR JSON aggregates
and the final 4 October 2026 MoCo experiment log (seeds 42–46).

Usage:
    pip install numpy matplotlib
    python plot_framework_comparison_section4_6.py

Outputs PNG (300 DPI) and editable/scalable SVG in this script's folder.
No JSON input files or command-line arguments required.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

TRANSFER_DIRECTIONS = ["CIFAR → CIFAR", "CIFAR → STL", "STL → CIFAR", "STL → STL"]
# Columns ordered by downstream classifier-training seed 42, 43, 44, 45, 46
SIMCLR_RAW_ASR = np.array([[93.87777777777778, 94.1, 94.16666666666667, 94.12222222222222, 94.54444444444444], [97.52777777777779, 97.375, 97.34722222222221, 97.34722222222221, 98.15277777777777], [95.71111111111111, 96.21111111111111, 93.28888888888889, 93.4, 93.64444444444445], [99.3888888888889, 99.31944444444444, 99.40277777777777, 99.41666666666666, 99.13888888888889]], dtype=float)
MOCO_RAW_ASR = np.array([[61.17, 70.34, 78.0, 1.02, 93.29], [99.11, 99.24, 99.21, 99.35, 99.14], [0.74, 3.32, 2.21, 2.02, 2.01], [99.71, 99.49, 99.6, 99.57, 99.65]], dtype=float)

def main():
    sim_mean = SIMCLR_RAW_ASR.mean(axis=1)
    sim_sd = SIMCLR_RAW_ASR.std(axis=1, ddof=1)
    moco_mean = MOCO_RAW_ASR.mean(axis=1)
    moco_sd = MOCO_RAW_ASR.std(axis=1, ddof=1)

    x = np.arange(len(TRANSFER_DIRECTIONS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9, 4.7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    ax.bar(x - width / 2, sim_mean, width, yerr=sim_sd,
           capsize=5, color=colors[0], label="SimCLR", zorder=3)
    ax.bar(x + width / 2, moco_mean, width, yerr=moco_sd,
           capsize=5, color=colors[1], label="MoCo", zorder=3)

    for x0, val in zip(x - width/2, sim_mean):
        ax.annotate(f"{val:.2f}%", (x0, val), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=8.5)
    for i, (x0, val) in enumerate(zip(x + width/2, moco_mean)):
        # Shift the highly variable CIFAR→CIFAR label away from its tall error bar.
        offset = (17, 4) if i == 0 else (0, 4)
        ax.annotate(f"{val:.2f}%", (x0, val), xytext=offset,
                    textcoords="offset points", ha="left" if i == 0 else "center",
                    fontsize=8.5)

    ax.set_xticks(x, TRANSFER_DIRECTIONS)
    ax.set_ylabel("Raw attack success rate (ASR, %)")
    ax.set_xlabel("Upstream → downstream dataset")
    ax.set_title("Raw ASR: SimCLR vs MoCo")
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlim(-0.58, len(x)-0.42)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(pad=1.3)

    folder = Path(__file__).resolve().parent
    for ext in ("png", "svg"):
        p = folder / f"framework_comparison_section4_6.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)
    for label, a, s, b, t in zip(TRANSFER_DIRECTIONS, sim_mean, sim_sd,
                                 moco_mean, moco_sd):
        print(f"{label}: SimCLR {a:.2f} ± {s:.2f}%; MoCo {b:.2f} ± {t:.2f}%")

if __name__ == "__main__":
    main()
