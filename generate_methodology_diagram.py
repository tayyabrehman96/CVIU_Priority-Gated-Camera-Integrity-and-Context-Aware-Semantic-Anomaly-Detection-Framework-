"""
Generate a complex, publication-quality methodology diagram with icons,
layered architecture, and detailed sub-components.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path("Images")
OUT.mkdir(exist_ok=True)

BLUE    = '#2563eb'
DBLUE   = '#1e40af'
GREEN   = '#16a34a'
DGREEN  = '#166534'
RED     = '#dc2626'
ORANGE  = '#ea580c'
AMBER   = '#f59e0b'
PURPLE  = '#7c3aed'
DPURPLE = '#5b21b6'
GRAY    = '#6b7280'
DGRAY   = '#374151'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'mathtext.fontset': 'cm',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
})


def rbox(ax, x, y, w, h, fc, ec='#374151', lw=1.2, alpha=1.0, zorder=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                         facecolor=fc, edgecolor=ec, linewidth=lw,
                         alpha=alpha, zorder=zorder)
    ax.add_patch(box)


def arr(ax, x1, y1, x2, y2, color='#374151', lw=1.8, zorder=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                zorder=zorder)


def camera_icon(ax, cx, cy, s=0.28):
    body = FancyBboxPatch((cx - s*0.6, cy - s*0.35), s*1.2, s*0.7,
                          boxstyle="round,pad=0.03", facecolor='#e5e7eb',
                          edgecolor=DGRAY, linewidth=0.8, zorder=6)
    ax.add_patch(body)
    ax.add_patch(Circle((cx, cy), s*0.2, fc='#1f2937', ec=DGRAY, lw=0.6, zorder=7))
    ax.add_patch(Circle((cx, cy), s*0.1, fc='#60a5fa', ec='none', zorder=8))
    flash = FancyBboxPatch((cx - s*0.15, cy + s*0.35), s*0.3, s*0.12,
                           boxstyle="round,pad=0.02", facecolor='#d1d5db',
                           edgecolor=DGRAY, linewidth=0.5, zorder=6)
    ax.add_patch(flash)


def eye_icon(ax, cx, cy, s=0.22):
    pts = [[cx-s, cy], [cx-s*0.3, cy+s*0.5], [cx+s*0.3, cy+s*0.5],
           [cx+s, cy], [cx+s*0.3, cy-s*0.5], [cx-s*0.3, cy-s*0.5]]
    ax.add_patch(plt.Polygon(pts, closed=True, fc='#fce7f3', ec='#9d174d', lw=0.8, zorder=7))
    ax.add_patch(Circle((cx, cy), s*0.25, fc='#9d174d', ec='none', zorder=8))
    ax.add_patch(Circle((cx+s*0.08, cy+s*0.08), s*0.08, fc='white', ec='none', zorder=9))


def brain_icon(ax, cx, cy, s=0.18):
    colors = ['#c4b5fd', '#a78bfa', '#8b5cf6', '#7c3aed']
    offsets = [(-0.12, 0.08), (0.12, 0.08), (-0.08, -0.08), (0.08, -0.08)]
    for (dx, dy), c in zip(offsets, colors):
        ax.add_patch(Circle((cx+dx*s*2, cy+dy*s*2), s*0.35,
                            fc=c, ec='white', lw=0.4, alpha=0.8, zorder=7))
    ax.text(cx, cy, 'AI', ha='center', va='center', fontsize=5.5,
            color='white', weight='bold', zorder=9)


def alert_icon(ax, cx, cy, s=0.25):
    pts = [[cx-s*0.4, cy-s*0.1], [cx-s*0.5, cy-s*0.5],
           [cx+s*0.5, cy-s*0.5], [cx+s*0.4, cy-s*0.1]]
    ax.add_patch(plt.Polygon(pts, closed=True, fc='#fbbf24', ec='#92400e', lw=0.8, zorder=7))
    ax.add_patch(Circle((cx, cy+s*0.05), s*0.12, fc='#fbbf24', ec='#92400e', lw=0.6, zorder=7))
    ax.text(cx, cy-s*0.25, '!', ha='center', va='center',
            fontsize=7, color='#92400e', weight='bold', zorder=8)


# ================================================================
fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(-0.5, 19)
ax.set_ylim(-2.8, 8.2)
ax.axis('off')

# ── Title ──
ax.text(9.25, 7.85, 'Cascaded Multi-Agent Anomaly Detection Pipeline',
        ha='center', fontsize=14, weight='bold', color=DGRAY)

# ════════════════════════════════════════════════════════
# INPUT SECTION  (x: -0.3 to 2.1)
# ════════════════════════════════════════════════════════

rbox(ax, -0.3, 1.2, 2.4, 5.6, '#f8fafc', ec='#94a3b8', lw=1.5)
ax.text(0.9, 6.35, 'Input Layer', ha='center', fontsize=10, weight='bold', color=DGRAY)

for i, yy in enumerate([4.8, 3.8, 2.8]):
    camera_icon(ax, 0.5, yy)
    ax.text(1.3, yy, f'$c_{{{i+1}}}$', fontsize=8, va='center', color=DGRAY)

# Agents
rbox(ax, 0.0, 1.55, 0.9, 0.55, '#fff7ed', ec=ORANGE, lw=1.0)
ax.text(0.45, 1.9, r'$\mathcal{A}_e$', ha='center', fontsize=9, color=ORANGE, weight='bold')
ax.text(0.45, 1.67, 'Event', ha='center', fontsize=5.5, color=ORANGE)

rbox(ax, 1.15, 1.55, 0.9, 0.55, '#eff6ff', ec=BLUE, lw=1.0)
ax.text(1.6, 1.9, r'$\mathcal{A}_m$', ha='center', fontsize=9, color=BLUE, weight='bold')
ax.text(1.6, 1.67, 'Cyclical', ha='center', fontsize=5.5, color=BLUE)

# Pub/Sub
rbox(ax, 0.15, 0.55, 1.6, 0.6, '#f1f5f9', ec='#64748b', lw=0.8)
ax.text(0.95, 0.92, 'Pub/Sub Broker', ha='center', fontsize=6.5, color='#475569', weight='bold')
ax.text(0.95, 0.7, '(Redis)', ha='center', fontsize=5, color='#94a3b8')

arr(ax, 0.45, 1.55, 0.45, 1.2, color=ORANGE, lw=1.0)
arr(ax, 1.6, 1.55, 1.6, 1.2, color=BLUE, lw=1.0)

# Main output arrow
arr(ax, 2.1, 3.8, 3.0, 3.8, color=DGRAY, lw=2.2)
ax.text(2.55, 4.1, r'$x_t$', fontsize=10, ha='center', color=DGRAY, weight='bold')

# ════════════════════════════════════════════════════════
# STAGE I: AUTOENCODER  (x: 3.0 to 7.0)
# ════════════════════════════════════════════════════════

rbox(ax, 3.0, 0.5, 4.0, 6.3, '#eff6ff', ec=DBLUE, lw=1.8)
ax.text(5.0, 6.35, 'Stage I: Reconstruction Screening',
        ha='center', fontsize=10, weight='bold', color=DBLUE)

# Encoder sub-box
rbox(ax, 3.2, 3.7, 1.6, 2.2, '#dbeafe', ec='#93c5fd', lw=0.8)
ax.text(4.0, 5.65, 'Encoder', ha='center', fontsize=8, weight='bold', color=DBLUE)

enc_data = [
    (3.35, 5.1, 'Conv2d 3x3', '3 \u2192 32',  '#3b82f6'),
    (3.35, 4.55, 'Conv2d 3x3', '32 \u2192 64', '#2563eb'),
    (3.35, 4.0,  'Conv2d 3x3', '64 \u2192 128', '#1d4ed8'),
]
for (ex, ey, lbl, ch, clr) in enc_data:
    rbox(ax, ex, ey, 1.3, 0.4, clr, ec='white', lw=0.5, zorder=4)
    ax.text(ex+0.65, ey+0.25, lbl, ha='center', fontsize=6, color='white', weight='bold', zorder=6)
    ax.text(ex+0.65, ey+0.08, ch, ha='center', fontsize=5.5, color='#bfdbfe', zorder=6)

# Bottleneck
rbox(ax, 3.8, 3.0, 2.4, 0.55, '#1e3a8a', ec='white', lw=0.5, zorder=4)
ax.text(5.0, 3.35, 'Bottleneck  16 x 16 x 128', ha='center', fontsize=6.5,
        color='white', weight='bold', zorder=6)
ax.text(5.0, 3.1, '32,768 dims', ha='center', fontsize=5.5, color='#93c5fd', zorder=6)

# Decoder sub-box
rbox(ax, 5.2, 3.7, 1.6, 2.2, '#dbeafe', ec='#93c5fd', lw=0.8)
ax.text(6.0, 5.65, 'Decoder', ha='center', fontsize=8, weight='bold', color=DBLUE)

dec_data = [
    (5.35, 4.0,  'DeConv 3x3', '128 \u2192 64', '#1d4ed8'),
    (5.35, 4.55, 'DeConv 3x3', '64 \u2192 32',  '#2563eb'),
    (5.35, 5.1,  'DeConv 3x3', '32 \u2192 3',   '#3b82f6'),
]
for (dx, dy, lbl, ch, clr) in dec_data:
    rbox(ax, dx, dy, 1.3, 0.4, clr, ec='white', lw=0.5, zorder=4)
    ax.text(dx+0.65, dy+0.25, lbl, ha='center', fontsize=6, color='white', weight='bold', zorder=6)
    ax.text(dx+0.65, dy+0.08, ch, ha='center', fontsize=5.5, color='#bfdbfe', zorder=6)

# Arrows Enc → Bottleneck → Dec
arr(ax, 4.0, 3.7, 4.3, 3.55, color=DBLUE, lw=1.0)
arr(ax, 5.7, 3.55, 6.0, 3.7, color=DBLUE, lw=1.0)

# Reconstruction error box
rbox(ax, 3.3, 0.7, 3.4, 1.9, '#f0f9ff', ec='#60a5fa', lw=0.8)
ax.text(5.0, 2.35, 'Reconstruction Error', ha='center', fontsize=8,
        weight='bold', color=DBLUE, zorder=6)
ax.text(5.0, 1.85, r'$e(x_t) = \frac{1}{3HW}\|x_t - \hat{x}_t\|_2^2$',
        ha='center', fontsize=9, color=DBLUE, zorder=6)
ax.text(5.0, 1.3, 'PSNR = 38.3 dB', ha='center', fontsize=6.5,
        color='#3b82f6', style='italic', zorder=6)
ax.text(5.0, 0.95, 'SSIM = 0.965', ha='center', fontsize=6.5,
        color='#3b82f6', style='italic', zorder=6)

arr(ax, 5.0, 3.7, 5.0, 2.65, color=DBLUE, lw=1.2)

# ── Early exit from Stage I ──
rbox(ax, 3.6, -1.0, 2.6, 0.9, '#dcfce7', ec=GREEN, lw=1.2)
ax.text(4.9, -0.35, 'Normal (Early Exit)', ha='center', fontsize=9,
        weight='bold', color=DGREEN, zorder=6)
ax.text(4.9, -0.75, r'$\sim$72% frames  |  $\sim$6 ms/frame', ha='center',
        fontsize=7, color=GREEN, zorder=6)

arr(ax, 5.0, 0.7, 4.9, -0.1, color=GREEN, lw=1.8)
ax.text(5.5, 0.2, r'$e(x_t) \leq \tau_1$', fontsize=8, color=GREEN, weight='bold')

# ════════════════════════════════════════════════════════
# Transition arrow Stage I → Stage II
# ════════════════════════════════════════════════════════

arr(ax, 7.0, 1.65, 7.8, 1.65, color=RED, lw=2.2)
ax.text(7.4, 1.95, r'$e(x_t) > \tau_1$', fontsize=8, color=RED, weight='bold', ha='center')

# ════════════════════════════════════════════════════════
# STAGE II: YOLO  (x: 7.8 to 11.2)
# ════════════════════════════════════════════════════════

rbox(ax, 7.8, 0.5, 3.4, 6.3, '#fefce8', ec='#a16207', lw=1.8)
ax.text(9.5, 6.35, 'Stage II: Object Detection',
        ha='center', fontsize=10, weight='bold', color='#92400e')
ax.text(9.5, 6.0, '(YOLOv8-nano, ~7M params)',
        ha='center', fontsize=7, color='#b45309', style='italic')

# Backbone
rbox(ax, 8.0, 4.3, 2.9, 1.05, '#fef9c3', ec='#facc15', lw=0.8)
ax.text(9.45, 5.1, 'Feature Backbone (CSPDarknet)', ha='center', fontsize=7,
        weight='bold', color='#713f12', zorder=6)

bcolors = ['#eab308', '#ca8a04', '#a16207', '#854d0e', '#713f12']
bx = 8.15
for bc in bcolors:
    rbox(ax, bx, 4.4, 0.4, 0.55, bc, ec='white', lw=0.4, zorder=4)
    bx += 0.5

# Detection head
rbox(ax, 8.0, 3.1, 2.9, 0.95, '#fef08a', ec='#facc15', lw=0.8)
ax.text(9.45, 3.8, 'Detection Head', ha='center', fontsize=8,
        weight='bold', color='#713f12', zorder=6)
ax.text(9.45, 3.4, r'$P_2(y\,|\,x_t)$ over object categories', ha='center', fontsize=7,
        color='#92400e', zorder=6)

arr(ax, 9.45, 4.3, 9.45, 4.1, color='#a16207', lw=1.0)

# Detection output
rbox(ax, 8.0, 1.6, 2.9, 1.2, '#fef3c7', ec='#f59e0b', lw=0.8)
ax.text(9.45, 2.55, 'Detection Output', ha='center', fontsize=7.5,
        weight='bold', color='#92400e', zorder=6)

# Bounding box mini-icons
for bxi in [8.3, 9.0, 9.7]:
    rect = Rectangle((bxi, 1.75), 0.4, 0.45, linewidth=1.0,
                      edgecolor=AMBER, facecolor='#fef9c3', zorder=6)
    ax.add_patch(rect)
    ax.text(bxi+0.2, 1.98, '?', ha='center', va='center',
            fontsize=7, color='#92400e', weight='bold', zorder=7)

ax.text(9.45, 2.3, r'~8 ms/frame', ha='center', fontsize=6,
        color='#b45309', style='italic', zorder=6)

arr(ax, 9.45, 3.1, 9.45, 2.85, color='#a16207', lw=1.0)

# ── Exit from Stage II ──
rbox(ax, 8.1, -1.0, 2.6, 0.9, '#fef9c3', ec=AMBER, lw=1.2)
ax.text(9.4, -0.35, 'Object Event (Resolved)', ha='center', fontsize=9,
        weight='bold', color='#92400e', zorder=6)
ax.text(9.4, -0.75, r'$\sim$18% frames', ha='center',
        fontsize=7, color='#b45309', zorder=6)

arr(ax, 9.45, 1.6, 9.4, -0.1, color=AMBER, lw=1.8)
ax.text(10.05, 0.6, r'$\max_k P_2 \geq \tau_2$', fontsize=8,
        color=AMBER, weight='bold')

# ════════════════════════════════════════════════════════
# Transition arrow Stage II → Stage III
# ════════════════════════════════════════════════════════

arr(ax, 11.2, 3.5, 12.0, 3.5, color=DGRAY, lw=2.2)
ax.text(11.6, 3.8, 'Uncertain', fontsize=7, color=GRAY, style='italic', ha='center')

# ════════════════════════════════════════════════════════
# STAGE III: VLM  (x: 12.0 to 15.6)
# ════════════════════════════════════════════════════════

rbox(ax, 12.0, 0.5, 3.6, 6.3, '#fdf2f8', ec='#9d174d', lw=1.8)
ax.text(13.8, 6.35, 'Stage III: Semantic Reasoning',
        ha='center', fontsize=10, weight='bold', color='#9d174d')
ax.text(13.8, 6.0, '(Vision-Language Model)',
        ha='center', fontsize=7, color='#be185d', style='italic')

# Vision encoder
rbox(ax, 12.2, 4.35, 1.4, 1.05, '#fce7f3', ec='#f9a8d4', lw=0.8)
eye_icon(ax, 12.9, 5.05)
ax.text(12.9, 4.65, 'Vision', ha='center', fontsize=7, weight='bold', color='#9d174d', zorder=6)
ax.text(12.9, 4.45, 'Encoder', ha='center', fontsize=6.5, color='#be185d', zorder=6)

# Language model
rbox(ax, 14.0, 4.35, 1.4, 1.05, '#fce7f3', ec='#f9a8d4', lw=0.8)
brain_icon(ax, 14.7, 5.05)
ax.text(14.7, 4.65, 'LLaVA-Next', ha='center', fontsize=7, weight='bold', color='#9d174d', zorder=6)
ax.text(14.7, 4.45, 'LLM Decoder', ha='center', fontsize=6.5, color='#be185d', zorder=6)

arr(ax, 13.6, 4.85, 14.0, 4.85, color='#be185d', lw=1.2)

# Embedding + prototype matching
rbox(ax, 12.2, 2.7, 3.2, 1.35, '#fbcfe8', ec='#ec4899', lw=0.8)
ax.text(13.8, 3.8, 'Embedding + Prototype Matching', ha='center', fontsize=7.5,
        weight='bold', color='#9d174d', zorder=6)
ax.text(13.8, 3.4, r'$z_t = E(T_t) \in \mathbb{R}^d$', ha='center',
        fontsize=9, color='#9d174d', zorder=6)
ax.text(13.8, 2.98, r'$y^* = \arg\max_k \langle z_t, \mu_k \rangle$',
        ha='center', fontsize=9, color='#9d174d', zorder=6)

arr(ax, 13.8, 4.35, 13.8, 4.1, color='#be185d', lw=1.2)

# Semantic labels
rbox(ax, 12.2, 0.7, 3.2, 1.7, '#fdf2f8', ec='#f472b6', lw=0.8)
ax.text(13.8, 2.15, r'Similarity $> \tau_c$ ?', ha='center', fontsize=7.5,
        color='#9d174d', weight='bold', zorder=6)

labels_data = [
    (12.4, 1.55, 'camera_blocked', RED),
    (13.9, 1.55, 'person_detected', GREEN),
    (12.4, 1.05, 'fire / explosion', ORANGE),
    (13.9, 1.05, 'abstain', GRAY),
]
for (lx, ly, lt, lc) in labels_data:
    rbox(ax, lx, ly, 1.35, 0.3, lc, ec='white', lw=0.3, alpha=0.8, zorder=4)
    ax.text(lx+0.675, ly+0.15, lt, ha='center', va='center',
            fontsize=5.5, color='white', weight='bold', zorder=7)

arr(ax, 13.8, 2.7, 13.8, 2.45, color='#be185d', lw=1.0)

# ════════════════════════════════════════════════════════
# OUTPUT / DECISION  (x: 16.0 to 18.7)
# ════════════════════════════════════════════════════════

arr(ax, 15.6, 3.5, 16.3, 3.5, color=DGRAY, lw=2.2)

rbox(ax, 16.3, 0.5, 2.5, 6.3, '#fef2f2', ec='#991b1b', lw=1.8)
ax.text(17.55, 6.35, 'Decision Layer', ha='center', fontsize=10,
        weight='bold', color='#991b1b')

alert_icon(ax, 17.55, 5.2)

# Severity score
rbox(ax, 16.5, 3.6, 2.1, 1.2, '#fee2e2', ec='#fca5a5', lw=0.8)
ax.text(17.55, 4.55, 'Severity Score', ha='center', fontsize=8,
        weight='bold', color='#991b1b', zorder=6)
ax.text(17.55, 4.15, r'$S = \lambda_1 c_{vis} + \lambda_2 c_{ctx}$',
        ha='center', fontsize=8.5, color='#991b1b', zorder=6)
ax.text(17.55, 3.78, r'$S \geq \tau_S \Rightarrow$ Alert', ha='center',
        fontsize=7, color=RED, zorder=6)

# Action outputs
actions = [
    ('Security Alert', '#ef4444', 2.85),
    ('Event Log', '#f59e0b', 2.15),
    ('Operator Report', '#3b82f6', 1.45),
    ('Dashboard Update', '#8b5cf6', 0.75),
]
for label, color, yy in actions:
    rbox(ax, 16.5, yy, 2.1, 0.5, color, ec='white', lw=0.5, alpha=0.85, zorder=4)
    ax.text(17.55, yy+0.25, label, ha='center', va='center',
            fontsize=7.5, color='white', weight='bold', zorder=6)

# ════════════════════════════════════════════════════════
# BOTTOM LEGEND
# ════════════════════════════════════════════════════════

ly = -2.3
rbox(ax, 0.5, ly-0.15, 18.0, 0.55, '#f8fafc', ec='#e2e8f0', lw=0.8)
items = [
    (0.8, r'$\tau_1$', 'AE Recon. Threshold', DBLUE),
    (5.5, r'$\tau_2$', 'YOLO Confidence Threshold', '#92400e'),
    (10.5, r'$\tau_c$', 'Cosine Similarity Threshold', '#9d174d'),
    (15.5, r'$\tau_S$', 'Alert Severity Threshold', '#991b1b'),
]
for (xp, sym, desc, clr) in items:
    ax.text(xp, ly+0.12, f'{sym} = {desc}', fontsize=7.5, color=clr, weight='bold')


fig.savefig(str(OUT / 'Dual-Stage Perception Module.png'), dpi=300)
plt.close(fig)
print("Saved: Dual-Stage Perception Module.png")
