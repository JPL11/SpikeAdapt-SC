#!/usr/bin/env python3
"""Regenerate Fig 1 architecture diagram — match original style closely.

Style preserved from original fig1_architecture.png:
  - Image thumbnail in top-left
  - 3-column layout with dashed vertical separators
  - Compact, tightly-packed boxes
  - Dashed group containers (orange for encoder, green for scorer/decoder)
  - Same color palette
  - Spikes S_1 intermediate shown in encoder
  - BatchNorm as separate box, MPBN as side annotation
  - Italic side labels (Frozen backbone, membrane feedback)
  - Classification with checkmark

Corrections from original:
  1. IF Neuron theta=1.0  ->  LIF Neuron (beta, v_th learnable)
  2. IHF Neuron theta_hat (learned)  ->  LIF Neuron
  3. CHANNEL block: BSC/AWGN/Rayleigh  ->  BSC only (matches removed cross-channel section)

Output: paper/figures/fig1_architecture_v2.{pdf,png}
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.image import imread

os.makedirs('paper/figures', exist_ok=True)

# ===== Style =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
})

# Colors from the original figure
COLOR_BACKBONE = '#cfe2f3'   # ResNet blue
COLOR_FEATURE = '#fff2cc'    # feature map yellow
COLOR_ENCODER = '#fce5cd'    # SNN encoder orange
COLOR_SCORER = '#d9ead3'     # scorer green
COLOR_CHANNEL = '#f4cccc'    # channel red
COLOR_MASK = '#ffe599'       # mask tan
COLOR_DECODER = '#d9ead3'    # decoder green
COLOR_CONVERTER = '#d5a6bd'  # converter purple
COLOR_BORDER = '#666666'

EDGE_DASH = (0, (4, 2))


def box(ax, x, y, w, h, text, color, fontsize=7.5, fontweight='normal'):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle='round,pad=0.01,rounding_size=0.04',
        linewidth=0.7, edgecolor=COLOR_BORDER,
        facecolor=color, zorder=2,
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text,
            ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, zorder=3)


def arrow(ax, x1, y1, x2, y2, color='black', lw=0.9):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>', mutation_scale=8,
        lw=lw, color=color, zorder=4,
    )
    ax.add_patch(a)


def dashed_group(ax, x, y, w, h, label, color):
    rect = Rectangle(
        (x, y), w, h,
        linewidth=0.9, edgecolor=color, facecolor='none',
        linestyle='--', zorder=1,
    )
    ax.add_patch(rect)
    # Label sits inside the top-left corner, on white background to avoid overlap
    ax.text(x + 0.08, y + h - 0.09, label,
            ha='left', va='top', fontsize=7.5, fontweight='bold',
            color=color, zorder=2,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.5))


# Canvas
fig, ax = plt.subplots(figsize=(8, 9))
ax.set_xlim(0, 8)
ax.set_ylim(0, 10)
ax.axis('off')

# ===== Section headers =====
ax.text(1.65, 9.7, 'TRANSMITTER SIDE',
        ha='center', fontsize=10, fontweight='bold')
ax.text(4.0, 9.7, 'SpikeAdapt-SC',
        ha='center', fontsize=12, fontweight='bold')
ax.text(6.45, 9.7, 'RECEIVER SIDE',
        ha='center', fontsize=10, fontweight='bold')

# Vertical separators (dashed)
ax.plot([2.85, 2.85], [0.2, 9.4], color='#999', linestyle=EDGE_DASH, lw=0.7, zorder=0)
ax.plot([5.18, 5.18], [0.2, 9.4], color='#999', linestyle=EDGE_DASH, lw=0.7, zorder=0)


# ============================================================
# TRANSMITTER (x range: 0.2 - 2.85)
# ============================================================

# 1. Image thumbnail (small box) + label
THUMB_W, THUMB_H = 0.45, 0.45
# Try to load an actual aerial image; fall back to a synthetic one
thumb_path = None
for candidate in ['data/AID/Airport/airport_1.jpg',
                  'paper/figures/dota_v6d2_viz_1.png']:
    if os.path.exists(candidate):
        thumb_path = candidate; break

ax_thumb = ax.inset_axes([0.36/8, 9.0/10, THUMB_W/8, THUMB_H/10])
if thumb_path:
    try:
        ax_thumb.imshow(imread(thumb_path))
    except Exception:
        rng = np.random.default_rng(7)
        ax_thumb.imshow(rng.random((20, 20, 3)))
else:
    rng = np.random.default_rng(7)
    ax_thumb.imshow(rng.random((20, 20, 3)))
ax_thumb.axis('off')
for spine in ax_thumb.spines.values():
    spine.set_visible(False)

# "Input Image" box (right of thumbnail)
box(ax, 1.05, 9.05, 1.55, 0.40,
    'Input Image\n3$\\times$224$\\times$224', 'white', fontsize=8)
arrow(ax, 1.83, 9.05, 1.83, 8.80)

# 2. ResNet50 Front
box(ax, 1.0, 8.30, 1.65, 0.50,
    'ResNet-50 Front\n(Layers 1-3)', COLOR_BACKBONE, fontsize=8)
ax.text(2.72, 8.55, 'Frozen\nbackbone',
        ha='left', va='center', fontsize=7, style='italic')
arrow(ax, 1.83, 8.30, 1.83, 8.10)

# 3. Feature Map F
box(ax, 0.95, 7.45, 1.75, 0.60,
    'Feature Map $\\mathbf{F}$\n1024$\\times$14$\\times$14\n196 spatial blocks',
    COLOR_FEATURE, fontsize=7.5)
arrow(ax, 1.83, 7.45, 1.83, 7.20)

# 4. SNN Encoder dashed group
dashed_group(ax, 0.45, 4.40, 2.30, 2.85,
             'SNN Encoder ($T{=}8$ timesteps)', '#cc7a00')

box(ax, 0.65, 6.65, 1.95, 0.30,
    'Conv2d 3$\\times$3, $1024{\\to}256$', COLOR_ENCODER, fontsize=7.5)
arrow(ax, 1.65, 6.65, 1.65, 6.50)
box(ax, 0.65, 6.20, 1.95, 0.30, 'BatchNorm', COLOR_ENCODER, fontsize=7.5)
arrow(ax, 1.65, 6.20, 1.65, 6.05)
box(ax, 0.65, 5.75, 1.95, 0.30,
    'Spikes $\\mathbf{S}_1$ $(256{\\times}14{\\times}14)$',
    COLOR_ENCODER, fontsize=7.5)
# Dotted divider inside encoder
ax.plot([0.65, 2.60], [5.62, 5.62], color='#999',
        linestyle=(0, (2, 2)), lw=0.6, zorder=2)
arrow(ax, 1.65, 5.62, 1.65, 5.50)
box(ax, 0.65, 5.20, 1.95, 0.30,
    'Conv2d 3$\\times$3, $256{\\to}36$', COLOR_ENCODER, fontsize=7.5)
arrow(ax, 1.65, 5.20, 1.65, 5.05)
box(ax, 0.65, 4.75, 1.95, 0.30, 'BatchNorm', COLOR_ENCODER, fontsize=7.5)
arrow(ax, 1.65, 4.75, 1.65, 4.60)
box(ax, 0.65, 4.30, 1.95, 0.30,
    'LIF Neuron ($\\beta$, $v_\\text{th}$ learnable)',
    COLOR_ENCODER, fontsize=7.5)
arrow(ax, 1.65, 4.30, 1.65, 4.15)
box(ax, 0.65, 3.85, 1.95, 0.30,
    'Spikes $\\mathbf{S}_2$ $(36{\\times}14{\\times}14)$',
    COLOR_ENCODER, fontsize=7.5)

# Membrane feedback annotation (italic, right side)
ax.text(2.78, 5.40, 'membrane\nfeedback\n$\\mathbf{m}(t)$',
        ha='left', va='center', fontsize=7, style='italic',
        color='#666')

# MPBN annotation (right of encoder)
ax.text(2.78, 4.0, 'MPBN:\nFR=0.167',
        ha='left', va='center', fontsize=7.5, fontweight='bold',
        color='#cc7a00')

# 5. Noise-Aware Scorer dashed group
dashed_group(ax, 0.45, 1.05, 2.30, 2.55,
             'Noise-Aware Scorer', '#3d7a3d')

box(ax, 0.65, 3.05, 1.95, 0.30,
    'Time-avg $\\bar{\\mathbf{S}}=\\frac{1}{T}\\sum_t \\mathbf{S}_t$',
    COLOR_SCORER, fontsize=7.5)
arrow(ax, 1.65, 3.05, 1.65, 2.95)
ax.text(1.65, 2.76, 'BER-dependent channel reweighting',
        ha='center', va='center', fontsize=6.8, style='italic',
        color='#3d7a3d')
box(ax, 0.65, 2.20, 1.95, 0.30,
    'Conv1$\\times$1 $36{\\to}32$, ReLU', COLOR_SCORER, fontsize=7.5)
arrow(ax, 1.65, 2.20, 1.65, 2.05)
box(ax, 0.65, 1.70, 1.95, 0.30,
    'Conv1$\\times$1 $32{\\to}1$, Sigmoid', COLOR_SCORER, fontsize=7.5)
arrow(ax, 1.65, 1.70, 1.65, 1.55)
box(ax, 0.65, 1.20, 1.95, 0.30,
    'Scores $\\mathbf{I} \\in (0,1)^{14{\\times}14}$',
    COLOR_SCORER, fontsize=7.5, fontweight='bold')

# Spikes S_2 -> scorer (left side feedback)
arrow(ax, 0.65, 4.00, 0.40, 3.60, lw=0.8)
arrow(ax, 0.40, 3.60, 0.40, 3.20, lw=0.8)
arrow(ax, 0.40, 3.20, 0.65, 3.20, lw=0.8)


# ============================================================
# CENTER (x range: 2.85 - 5.18)
# ============================================================

# Bits transmitted info
box(ax, 3.05, 8.30, 2.00, 0.95,
    'Bits transmitted\n$=\\rho \\cdot T \\cdot C_2 \\cdot H \\cdot W$\n'
    '$=0.75 \\times 8 \\times 36 \\times 14 \\times 14$\n$=$ 42,336 bits',
    'white', fontsize=7.5)

# CHANNEL header
ax.text(4.0, 7.55, 'CHANNEL', ha='center', fontsize=10, fontweight='bold')
# Channel red box - BSC only
box(ax, 3.10, 6.30, 1.85, 1.10,
    'BSC: $\\mathbf{y}=\\mathbf{x}\\oplus \\mathbf{e}$\n'
    '$\\mathbf{e} \\sim \\mathrm{Bern}(\\text{BER})$\n'
    '\nTrained on uniform BER\nwith bias toward [0.15, 0.40]',
    COLOR_CHANNEL, fontsize=7.5)

# Block Mask
box(ax, 3.10, 5.10, 1.85, 0.45,
    'Block Mask\nTop-$k$ selection',
    COLOR_MASK, fontsize=7.5)

# Binary Mask annotation
ax.text(4.03, 4.77, 'Binary Mask',
        ha='center', va='center', fontsize=7.5)
ax.text(4.03, 4.60, '$\\mathbf{M}, \\{0,1\\}^{14\\times14}$',
        ha='center', va='center', fontsize=7.5)

# S2 ⊙ M with circled times
ax.text(3.55, 4.10, '$\\mathbf{S}_2 \\odot \\mathbf{M}$',
        ha='center', va='center', fontsize=10)
ax.text(4.10, 4.10, '$\\otimes$',
        ha='center', va='center', fontsize=14)
ax.text(4.55, 4.10, 'Masked',
        ha='center', va='center', fontsize=7.5)
ax.text(4.55, 3.92, 'Spikes',
        ha='center', va='center', fontsize=7.5)

# Arrows: scorer scores -> block mask
arrow(ax, 2.60, 1.35, 4.03, 5.10, lw=0.8)
# encoder spikes -> mask multiplication
arrow(ax, 2.60, 4.00, 3.20, 4.10, lw=0.8)
# block mask -> mask multiplication
arrow(ax, 4.03, 5.10, 4.03, 4.30, lw=0.8)
# masked spikes -> channel
arrow(ax, 4.55, 4.20, 4.55, 6.30, lw=0.8)
# channel out -> decoder
arrow(ax, 4.95, 6.85, 5.40, 7.95, lw=0.8)


# ============================================================
# RECEIVER (x range: 5.18 - 7.85)
# ============================================================

# SNN Decoder dashed group
dashed_group(ax, 5.30, 3.65, 2.45, 4.85,
             'SNN Decoder ($T{=}8$ timesteps)', '#3d7a3d')

box(ax, 5.45, 7.95, 2.20, 0.30,
    'Conv2d 3$\\times$3, $36{\\to}256$', COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 7.95, 6.55, 7.85)
box(ax, 5.45, 7.50, 2.20, 0.30, 'BatchNorm', COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 7.50, 6.55, 7.40)
box(ax, 5.45, 7.05, 2.20, 0.30,
    'LIF Neuron ($\\beta$, $v_\\text{th}$ learnable)',
    COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 7.05, 6.55, 6.95)
ax.text(6.55, 6.78, '$\\mathbf{S}_3$ label',
        ha='center', va='center', fontsize=7, style='italic')
arrow(ax, 6.55, 6.65, 6.55, 6.55)
box(ax, 5.45, 6.20, 2.20, 0.30,
    'Conv2d 3$\\times$3, $256{\\to}1024$', COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 6.20, 6.55, 6.10)
box(ax, 5.45, 5.75, 2.20, 0.30, 'BatchNorm', COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 5.75, 6.55, 5.65)
box(ax, 5.45, 5.30, 2.20, 0.30,
    'LIF Neuron ($\\beta$, $v_\\text{th}$ learnable)',
    COLOR_DECODER, fontsize=7.5)
arrow(ax, 6.55, 5.30, 6.55, 5.20)
ax.text(6.55, 5.02, '$\\mathbf{S}_4$, $\\mathbf{m}_4$ labels',
        ha='center', va='center', fontsize=7, style='italic')
arrow(ax, 6.55, 4.85, 6.55, 4.65)

# Spike-to-Feature Converter
box(ax, 5.40, 3.80, 2.30, 0.80,
    'Spike-to-Feature Converter\n'
    'Stack $[\\mathbf{S}_4, \\mathbf{m}_4]$ over $T$\n'
    '$\\to$ Gated Linear\n$\\to$ Sum',
    COLOR_CONVERTER, fontsize=7.5)
arrow(ax, 6.55, 3.80, 6.55, 3.55)

# Reconstructed Features
box(ax, 5.45, 2.85, 2.20, 0.55,
    'Reconstructed Features\n$\\hat{\\mathbf{F}}$, 1024$\\times$14$\\times$14',
    COLOR_FEATURE, fontsize=7.5)
arrow(ax, 6.55, 2.85, 6.55, 2.55)

# ResNet50 Back
box(ax, 5.45, 1.85, 2.20, 0.55,
    'ResNet-50 Back\n(Layer 4 + FC)',
    COLOR_BACKBONE, fontsize=7.5)
ax.text(7.75, 1.55, 'Fine-tuned',
        ha='right', va='center', fontsize=7, style='italic')
arrow(ax, 6.55, 1.85, 6.55, 1.30)

# Classification box
box(ax, 5.65, 0.65, 1.80, 0.50,
    'Classification\n$\\hat{y} \\in \\mathbb{R}^C$',
    'white', fontsize=8)
ax.text(7.55, 0.90, 'Airport',
        ha='left', va='center', fontsize=8.5, fontweight='bold')
ax.text(7.55, 0.65, '$\\checkmark$',
        ha='left', va='center', fontsize=12, color='#2a8a2a')


plt.tight_layout()
fig.savefig('paper/figures/fig1_architecture_v2.pdf', dpi=300,
            bbox_inches='tight', pad_inches=0.05)
fig.savefig('paper/figures/fig1_architecture_v2.png', dpi=200,
            bbox_inches='tight', pad_inches=0.05)
plt.close()
print('Saved: paper/figures/fig1_architecture_v2.pdf')
print('Saved: paper/figures/fig1_architecture_v2.png')
