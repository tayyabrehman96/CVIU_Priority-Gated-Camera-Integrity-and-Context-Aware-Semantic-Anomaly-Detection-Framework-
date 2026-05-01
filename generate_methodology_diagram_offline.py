"""
Generate a publication-quality methodology diagram for the
offline surveillance analyser used in the MDPI Sensors paper.

Architecture:
- Input Layer (CCTV video clips)
- Stage I: 3D Spatio–Temporal Autoencoder & Camera-Health Analysis
- Stage II: Object & Safety Detection (YOLO + Fire/Smoke)
- Stage III: Multimodal Reasoning (VLM + Text Classifier)
- Decision & Logging Layer (incidents, exports, metrics)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths / Colors / Global style
# -----------------------------------------------------------------------------

OUT = Path("Images")
OUT.mkdir(exist_ok=True)

BLUE = "#2563eb"
DBLUE = "#1e40af"
GREEN = "#16a34a"
DGREEN = "#166534"
ORANGE = "#ea580c"
AMBER = "#f59e0b"
PURPLE = "#7c3aed"
DPURPLE = "#5b21b6"
GRAY = "#6b7280"
DGRAY = "#374151"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "mathtext.fontset": "cm",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def rbox(ax, x, y, w, h, fc, ec=DGRAY, lw=1.2, alpha=1.0, zorder=2):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.12",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)


def arr(ax, x1, y1, x2, y2, color=DGRAY, lw=1.6, zorder=5):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
        zorder=zorder,
    )


def camera_icon(ax, cx, cy, s=0.28):
    body = FancyBboxPatch(
        (cx - s * 0.6, cy - s * 0.35),
        s * 1.2,
        s * 0.7,
        boxstyle="round,pad=0.03",
        facecolor="#e5e7eb",
        edgecolor=DGRAY,
        linewidth=0.8,
        zorder=6,
    )
    ax.add_patch(body)
    ax.add_patch(Circle((cx, cy), s * 0.2, fc="#1f2937", ec=DGRAY, lw=0.6, zorder=7))
    ax.add_patch(Circle((cx, cy), s * 0.1, fc="#60a5fa", ec="none", zorder=8))


def eye_icon(ax, cx, cy, s=0.22):
    pts = [
        [cx - s, cy],
        [cx - s * 0.3, cy + s * 0.5],
        [cx + s * 0.3, cy + s * 0.5],
        [cx + s, cy],
        [cx + s * 0.3, cy - s * 0.5],
        [cx - s * 0.3, cy - s * 0.5],
    ]
    ax.add_patch(
        plt.Polygon(pts, closed=True, fc="#fce7f3", ec="#9d174d", lw=0.8, zorder=7)
    )
    ax.add_patch(Circle((cx, cy), s * 0.25, fc="#9d174d", ec="none", zorder=8))
    ax.add_patch(Circle((cx + s * 0.08, cy + s * 0.08), s * 0.08, fc="white", ec="none", zorder=9))


def brain_icon(ax, cx, cy, s=0.18):
    colors = ["#c4b5fd", "#a78bfa", "#8b5cf6", "#7c3aed"]
    offsets = [(-0.12, 0.08), (0.12, 0.08), (-0.08, -0.08), (0.08, -0.08)]
    for (dx, dy), c in zip(offsets, colors):
        ax.add_patch(
            Circle((cx + dx * s * 2, cy + dy * s * 2), s * 0.35, fc=c, ec="white", lw=0.4, alpha=0.8, zorder=7)
        )
    ax.text(
        cx,
        cy,
        "AI",
        ha="center",
        va="center",
        fontsize=5.5,
        color="white",
        weight="bold",
        zorder=9,
    )


def main():
    fig, ax = plt.subplots(figsize=(15.5, 8.0))
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(-2.3, 8.2)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Title
    # ------------------------------------------------------------------
    ax.text(
        9.5,
        7.9,
        "Offline Autoencoder–YOLO–Multimodal Anomaly Detection Pipeline",
        ha="center",
        fontsize=13,
        weight="bold",
        color=DGRAY,
    )

    # ------------------------------------------------------------------
    # Input Layer
    # ------------------------------------------------------------------
    rbox(ax, -0.2, 1.4, 2.4, 5.2, "#f8fafc", ec="#94a3b8", lw=1.5)
    ax.text(1.0, 6.2, "Input Layer", ha="center", fontsize=10, weight="bold", color=DGRAY)

    for i, yy in enumerate([5.0, 4.0, 3.0]):
        camera_icon(ax, 0.6, yy)
        ax.text(1.4, yy, f"Camera {i+1}", fontsize=7.5, va="center", color=DGRAY)

    rbox(ax, 0.2, 1.6, 1.6, 0.7, "#f1f5f9", ec="#64748b", lw=0.9)
    ax.text(1.0, 2.1, "Video File Reader", ha="center", fontsize=7, color="#475569", weight="bold")
    ax.text(1.0, 1.85, "(offline clips)", ha="center", fontsize=6, color="#9ca3af")

    arr(ax, 2.2, 3.8, 3.1, 3.8, color=DGRAY, lw=2.0)
    ax.text(2.65, 4.05, r"$X_t \in \mathbb{R}^{3\times T\times H\times W}$", fontsize=8.5, ha="center", color=DGRAY)

    # ------------------------------------------------------------------
    # Stage I: Autoencoder & Camera Health
    # ------------------------------------------------------------------
    rbox(ax, 3.1, 0.5, 4.1, 6.0, "#eff6ff", ec=DBLUE, lw=1.8)
    ax.text(
        5.15,
        6.15,
        "Stage I: 3D Autoencoder & Camera-Health Analysis",
        ha="center",
        fontsize=9.6,
        weight="bold",
        color=DBLUE,
    )

    # Encoder
    rbox(ax, 3.3, 3.7, 1.7, 2.1, "#dbeafe", ec="#93c5fd", lw=0.9)
    ax.text(4.15, 5.65, "Encoder", ha="center", fontsize=8, weight="bold", color=DBLUE)
    enc_y = [5.1, 4.55, 4.0]
    enc_lbl = ["Conv3D k×k×k", "Conv3D k×k×k", "Conv3D k×k×k"]
    enc_ch = ["3→32", "32→64", "64→128"]
    for y, l, ch in zip(enc_y, enc_lbl, enc_ch):
        rbox(ax, 3.45, y, 1.4, 0.4, BLUE, ec="white", lw=0.4)
        ax.text(4.15, y + 0.25, l, ha="center", fontsize=6.4, color="white", weight="bold")
        ax.text(4.15, y + 0.08, ch, ha="center", fontsize=5.6, color="#bfdbfe")

    # Bottleneck
    rbox(ax, 3.9, 3.1, 2.5, 0.55, "#1d4ed8", ec="white", lw=0.5)
    ax.text(5.15, 3.45, r"Latent Code $Z$ (motion, health)", ha="center", fontsize=7.3, color="white", weight="bold")

    # Decoder
    rbox(ax, 5.25, 3.7, 1.7, 2.1, "#dbeafe", ec="#93c5fd", lw=0.9)
    ax.text(6.1, 5.65, "Decoder", ha="center", fontsize=8, weight="bold", color=DBLUE)
    dec_y = [4.0, 4.55, 5.1]
    dec_lbl = ["DeConv3D", "DeConv3D", "DeConv3D"]
    dec_ch = ["128→64", "64→32", "32→3"]
    for y, l, ch in zip(dec_y, dec_lbl, dec_ch):
        rbox(ax, 5.4, y, 1.4, 0.4, BLUE, ec="white", lw=0.4)
        ax.text(6.1, y + 0.25, l, ha="center", fontsize=6.4, color="white", weight="bold")
        ax.text(6.1, y + 0.08, ch, ha="center", fontsize=5.6, color="#bfdbfe")

    arr(ax, 4.15, 3.7, 4.4, 3.45, color=DBLUE, lw=0.9)
    arr(ax, 5.9, 3.45, 6.15, 3.7, color=DBLUE, lw=0.9)

    # Reconstruction error + camera health
    rbox(ax, 3.4, 0.85, 3.5, 2.0, "#f0f9ff", ec="#60a5fa", lw=0.9)
    ax.text(5.15, 2.6, "Reconstruction & Camera-Health Metrics", ha="center", fontsize=7.6, weight="bold", color=DBLUE)
    ax.text(5.15, 2.2, r"recon. error  $e_t$", ha="center", fontsize=7, color=DBLUE)
    ax.text(5.15, 1.9, "blur score, occlusion score", ha="center", fontsize=6.4, color="#3b82f6")
    ax.text(5.15, 1.55, "camera_moved flag, corrupted_frame", ha="center", fontsize=6.4, color="#3b82f6")

    arr(ax, 5.15, 3.7, 5.15, 2.9, color=DBLUE, lw=1.1)

    # Early normal exit
    rbox(ax, 3.8, -0.9, 2.7, 0.9, "#dcfce7", ec=GREEN, lw=1.1)
    ax.text(5.15, -0.35, "Normal (Early Exit)", ha="center", fontsize=8.8, weight="bold", color=DGREEN)
    ax.text(5.15, -0.7, r"$e_t \leq \tau_{AE}$", ha="center", fontsize=7.2, color=GREEN)
    arr(ax, 5.15, 0.85, 5.15, -0.1, color=GREEN, lw=1.8)

    # Arrow to Stage II when suspicious
    arr(ax, 7.2, 1.8, 8.1, 1.8, color=ORANGE, lw=2.0)
    ax.text(7.65, 2.15, r"$e_t > \tau_{AE}$  or camera issue", ha="center", fontsize=7.4, color=ORANGE)

    # ------------------------------------------------------------------
    # Stage II: YOLO + Fire/Smoke
    # ------------------------------------------------------------------
    rbox(ax, 8.1, 0.5, 3.6, 6.0, "#fefce8", ec="#a16207", lw=1.8)
    ax.text(9.9, 6.15, "Stage II: Object & Safety Detection", ha="center", fontsize=9.6, weight="bold", color="#92400e")
    ax.text(9.9, 5.85, "(YOLOv8 + Weapon + Fire/Smoke CV)", ha="center", fontsize=7, color="#b45309", style="italic")

    # YOLO backbone
    rbox(ax, 8.3, 4.4, 3.2, 1.0, "#fef9c3", ec="#facc15", lw=0.9)
    ax.text(9.9, 5.25, "YOLOv8 Backbone", ha="center", fontsize=7.3, weight="bold", color="#713f12")
    for i, w in enumerate([0.45, 0.9, 1.35, 1.8, 2.25]):
        rbox(ax, 8.45 + w * 0.0 + 0.1 * i, 4.5, 0.35, 0.5, ["#eab308", "#ca8a04", "#a16207", "#854d0e", "#713f12"][i], ec="white", lw=0.4)

    # Detection head
    rbox(ax, 8.3, 3.25, 3.2, 0.9, "#fef3c7", ec="#facc15", lw=0.9)
    ax.text(9.9, 3.9, "Detection Heads", ha="center", fontsize=7.5, weight="bold", color="#713f12")
    ax.text(9.9, 3.55, "persons, vehicles, weapon candidates", ha="center", fontsize=6.6, color="#b45309")

    arr(ax, 9.9, 4.4, 9.9, 4.15, color="#a16207", lw=1.0)

    # Fire / smoke module
    rbox(ax, 8.3, 2.1, 3.2, 0.9, "#fffbeb", ec="#fbbf24", lw=0.9)
    ax.text(9.9, 2.75, "Fire/Smoke CV (HSV + contours)", ha="center", fontsize=7.1, weight="bold", color="#92400e")
    ax.text(9.9, 2.4, "fire_mask, smoke_mask, scores", ha="center", fontsize=6.4, color="#b45309")

    arr(ax, 9.9, 3.25, 9.9, 3.0, color="#a16207", lw=1.0)

    # Scene feature summary
    rbox(ax, 8.4, 0.85, 3.0, 1.0, "#fef9c3", ec="#facc15", lw=0.9)
    ax.text(9.9, 1.55, "Scene Features", ha="center", fontsize=7.5, weight="bold", color="#713f12")
    ax.text(9.9, 1.25, "counts, boxes, fire/smoke regions", ha="center", fontsize=6.4, color="#b45309")

    arr(ax, 9.9, 2.1, 9.9, 1.85, color="#a16207", lw=1.0)

    # Arrow to Stage III
    arr(ax, 11.7, 3.0, 12.5, 3.0, color=DGRAY, lw=2.0)

    # ------------------------------------------------------------------
    # Stage III: Multimodal Reasoning
    # ------------------------------------------------------------------
    rbox(ax, 12.5, 0.5, 3.8, 6.0, "#fdf2f8", ec="#9d174d", lw=1.8)
    ax.text(14.4, 6.15, "Stage III: Multimodal Reasoning", ha="center", fontsize=9.6, weight="bold", color="#9d174d")

    # Fusion block
    rbox(ax, 12.8, 4.4, 3.2, 1.0, "#fce7f3", ec="#f9a8d4", lw=0.9)
    ax.text(14.4, 5.2, "Feature Fusion + Anomaly Score", ha="center", fontsize=7.5, weight="bold", color="#9d174d")
    ax.text(13.2, 4.75, "AE features Z, e_t", ha="left", fontsize=6.4, color="#be185d")
    ax.text(13.2, 4.5, "Scene features (detections, fire/smoke)", ha="left", fontsize=6.4, color="#be185d")

    # arrows from Stage I & II to fusion
    arr(ax, 7.2, 4.4, 12.8, 4.9, color=DBLUE, lw=1.4)
    arr(ax, 11.7, 4.0, 12.8, 4.7, color="#a16207", lw=1.4)

    # VLM + text classifier
    rbox(ax, 12.8, 2.7, 1.5, 1.2, "#fce7f3", ec="#f9a8d4", lw=0.9)
    eye_icon(ax, 13.55, 3.45)
    ax.text(13.55, 2.95, "VLM", ha="center", fontsize=7, weight="bold", color="#9d174d")

    rbox(ax, 14.5, 2.7, 1.5, 1.2, "#fce7f3", ec="#f9a8d4", lw=0.9)
    brain_icon(ax, 15.25, 3.45)
    ax.text(15.25, 2.95, "Regolo", ha="center", fontsize=7, weight="bold", color="#9d174d")

    arr(ax, 14.4, 4.4, 14.4, 3.9, color="#be185d", lw=1.1)
    arr(ax, 13.55, 2.7, 13.55, 2.4, color="#be185d", lw=1.0)
    arr(ax, 14.5, 3.3, 14.5, 3.0, color="#be185d", lw=1.0)
    arr(ax, 14.3, 3.3, 14.5, 3.3, color="#be185d", lw=1.0)

    rbox(ax, 12.9, 1.1, 3.0, 1.0, "#fdf2f8", ec="#f472b6", lw=0.9)
    ax.text(14.4, 1.75, "Final Anomaly Label", ha="center", fontsize=7.8, weight="bold", color="#9d174d")
    ax.text(
        14.4,
        1.4,
        "camera_blocked, camera_covered, blur_frame,\n"
        "camera_moved, fire_detected, smoke_detected,\n"
        "weapon_detected, theft_suspect, Benign",
        ha="center",
        fontsize=5.7,
        color="#9d174d",
    )

    arr(ax, 14.4, 2.7, 14.4, 2.2, color="#be185d", lw=1.1)

    # ------------------------------------------------------------------
    # Decision & Logging Layer
    # ------------------------------------------------------------------
    arr(ax, 16.3, 3.0, 17.0, 3.0, color=DGRAY, lw=2.0)

    rbox(ax, 17.0, 0.5, 2.3, 6.0, "#fef2f2", ec="#991b1b", lw=1.8)
    ax.text(18.15, 6.15, "Decision & Logging Layer", ha="center", fontsize=9.4, weight="bold", color="#991b1b")

    # Incidents table
    rbox(ax, 17.2, 4.4, 1.9, 1.0, "#fee2e2", ec="#fecaca", lw=0.8)
    ax.text(18.15, 5.0, "incidents.csv", ha="center", fontsize=7.2, weight="bold", color="#991b1b")
    ax.text(18.15, 4.65, "day, time, camera, label, caption, score", ha="center", fontsize=5.9, color="#b91c1c")

    # Exported frames
    rbox(ax, 17.2, 3.2, 1.9, 0.8, "#fee2e2", ec="#fecaca", lw=0.8)
    ax.text(18.15, 3.6, "Exported frames/", ha="center", fontsize=6.6, color="#b91c1c")
    ax.text(18.15, 3.35, "reports/", ha="center", fontsize=6.2, color="#b91c1c")

    # Metrics
    rbox(ax, 17.2, 2.0, 1.9, 0.8, "#fee2e2", ec="#fecaca", lw=0.8)
    ax.text(18.15, 2.4, "Metrics & Benchmarks", ha="center", fontsize=6.8, color="#b91c1c")
    ax.text(18.15, 2.15, "per-label F1, confusion matrix", ha="center", fontsize=6.0, color="#b91c1c")

    # ------------------------------------------------------------------
    # Bottom legend
    # ------------------------------------------------------------------
    ly = -1.9
    rbox(ax, 0.5, ly - 0.15, 18.0, 0.55, "#f8fafc", ec="#e2e8f0", lw=0.8)
    ax.text(1.0, ly + 0.12, "Blue: Autoencoder / camera-health modules", fontsize=7.0, color=DBLUE)
    ax.text(6.4, ly + 0.12, "Yellow: Object & safety detectors (YOLO, fire/smoke)", fontsize=7.0, color="#92400e")
    ax.text(12.0, ly + 0.12, "Purple: Multimodal reasoning (VLM + Regolo)", fontsize=7.0, color="#9d174d")
    ax.text(17.2, ly + 0.12, "Red: Decision & logging", fontsize=7.0, color="#b91c1c")

    out_path = OUT / "Offline_Surveillance_AE_YOLO_Pipeline.png"
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

