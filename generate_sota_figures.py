#!/usr/bin/env python3
"""
Generate two-panel SOTA bar chart (reviewer C15): panel A = prior methods on
their public benchmarks; panel B = proposed method on proprietary corpus only.
Run from repo root:  python generate_sota_figures.py

Output: fig9_sota_two_panel.png (and fig9_sota_map_comparison.png as symlink copy optional)
Values are those cited in the manuscript table (approximate for prior work).
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # Prior methods: (label, mAP %, dataset tag, color)
    prior = [
        ("Hasan et al.", 72, "ShanghaiTech", "#7eb0d5"),
        ("Sultani et al.", 81.1, "UCF-Crime", "#fd7f6f"),
        ("Ristea et al.", 83, "ShanghaiTech", "#b2e061"),
        ("Goswami &\nMandal", 88, "UHCTD", "#bd7ebe"),
        ("Wu et al.", 85, "UCF-Crime", "#ffb55a"),
        ("Zanella et al.", 86, "CVPR-set", "#ff9ff3"),
        ("Karim et al.", 85, "UCF-Crime", "#c5b0d5"),
    ]
    labels_a = [p[0] for p in prior]
    vals_a = [p[1] for p in prior]
    colors_a = [p[3] for p in prior]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [2.1, 1.0]})

    x1 = np.arange(len(labels_a))
    ax1.bar(x1, vals_a, color=colors_a, edgecolor="0.35", linewidth=0.6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_a, rotation=28, ha="right", fontsize=8.5)
    ax1.set_ylabel(r"Reported mAP / AP (\%) — frame or protocol of original paper")
    ax1.set_title("(A) Prior methods (each bar: source benchmark only)")
    ax1.set_ylim(0, 105)
    ax1.axhline(0, color="0.5", linewidth=0.5)

    # Panel B: proposed alone
    ax2.bar(
        [0],
        [97.2],
        width=0.45,
        color="#1e3a5f",
        edgecolor="0.2",
        linewidth=0.8,
        label="This work",
    )
    ax2.set_xticks([0])
    ax2.set_xticklabels(["Proposed\n(project corpus)"], fontsize=9)
    ax2.set_ylabel(r"$\mathrm{mAP}_{\mathrm{frame}}$ (\%)")
    ax2.set_title("(B) Proposed pipeline (proprietary test set only)")
    ax2.set_ylim(0, 105)
    ax2.text(
        0.5,
        0.02,
        "Do not compare (B) numerically to (A):\ndifferent data and protocols.",
        transform=ax2.transAxes,
        fontsize=8,
        va="bottom",
        ha="center",
        color="0.25",
        linespacing=1.2,
    )

    fig.suptitle(
        "Capability-oriented positioning (not a single ranking axis)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    path_two = os.path.join(out_dir, "fig9_sota_two_panel.png")
    fig.savefig(path_two, dpi=220, bbox_inches="tight")
    print(f"Wrote {path_two}")

    # Optional: overwrite legacy single filename expected by some drafts
    legacy = os.path.join(out_dir, "fig9_sota_map_comparison.png")
    fig.savefig(legacy, dpi=220, bbox_inches="tight")
    print(f"Wrote {legacy} (legacy name)")


if __name__ == "__main__":
    main()
