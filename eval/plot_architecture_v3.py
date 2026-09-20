#!/usr/bin/env python3
"""Fig 1 architecture, composed edition (v3): three tinted system bands
(UAV transmitter / A2G channel / ground station) in Times serif, colored
role chips, aligned typography, and the closed rate-adaptation loop drawn
as an explicit right-side feedback rail (the paper's contribution).
Real AID airport image as input; spike/mask rasters derived from its
pooled saliency as illustration. Output: paper/figures/fig1_architecture_v3.{pdf,png}
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.image import imread

BLUE, ORANGE, GREEN = "#2470a8", "#d1611e", "#2c8a5a"
RED, INK, ARROW, PURPLE = "#b03a3a", "#1b1b1b", "#3a3a3a", "#7d5185"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                   "DejaVu Serif"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.0})

FW, FH = 4.8, 5.9
AR = FW / FH
fig = plt.figure(figsize=(FW, FH))
bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 1); bg.set_ylim(0, 1)
bg.axis("off")


def band(y0, y1, fill, edge, label, lcolor):
    bg.add_patch(FancyBboxPatch((0.015, y0), 0.905, y1 - y0,
                 boxstyle="round,pad=0.004,rounding_size=0.012",
                 facecolor=fill, edgecolor=edge, lw=0.9, zorder=0))
    bg.text(0.045, y1 - 0.016, label, fontsize=6.4, fontweight="bold",
            color=lcolor, ha="left", va="center")


def rbox(x0, y0, w, h, text, fc, ec, fs=6.0, tc=INK, lw=0.9, z=2):
    bg.add_patch(FancyBboxPatch((x0, y0), w, h,
                 boxstyle="round,pad=0.004,rounding_size=0.010",
                 facecolor=fc, edgecolor=ec, lw=lw, zorder=z))
    bg.text(x0 + w / 2, y0 + h / 2, text, fontsize=fs, color=tc,
            ha="center", va="center", linespacing=1.45, zorder=z + 1)


def chip(xc, yc, text, color, fs=4.6):
    w = 0.014 + 0.0088 * len(text)
    bg.add_patch(FancyBboxPatch((xc - w / 2, yc - 0.011), w, 0.022,
                 boxstyle="round,pad=0.002,rounding_size=0.009",
                 facecolor=color, edgecolor="none", zorder=8))
    bg.text(xc, yc, text, fontsize=fs, color="white", fontweight="bold",
            ha="center", va="center", zorder=9)


def harrow(x0, x1, y, color=ARROW, lw=1.1):
    bg.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def varrow(x, y0, y1, color=ARROW, lw=1.1):
    bg.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                shrinkA=0, shrinkB=0))


def raster(x0, yc, w, img, edge, cmap=None, h=None):
    hh = h if h else w * AR
    ax = fig.add_axes([x0, yc - hh / 2, w, hh])
    ax.imshow(img, interpolation="nearest", aspect="auto", cmap=cmap)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.8); s.set_color(edge)
    return ax, hh


def iso(x0, w, h, yc, label, edge, front, top, side, sub=""):
    dx = 0.014
    dy = dx * AR * 0.7
    fx0, fx1 = x0, x0 + w
    fy0, fy1 = yc - h / 2, yc + h / 2
    bg.add_patch(Polygon([(fx0, fy1), (fx1, fy1), (fx1 + dx, fy1 + dy),
                          (fx0 + dx, fy1 + dy)], closed=True, facecolor=top,
                 edgecolor=edge, lw=0.9, zorder=2))
    bg.add_patch(Polygon([(fx1, fy0), (fx1 + dx, fy0 + dy),
                          (fx1 + dx, fy1 + dy), (fx1, fy1)], closed=True,
                 facecolor=side, edgecolor=edge, lw=0.9, zorder=2))
    bg.add_patch(Polygon([(fx0, fy0), (fx1, fy0), (fx1, fy1), (fx0, fy1)],
                 closed=True, facecolor=front, edgecolor=edge, lw=0.9,
                 zorder=3))
    bg.text((fx0 + fx1) / 2, yc + (0.012 if sub else 0), label, ha="center",
            va="center", fontsize=5.9, color=INK, zorder=4, linespacing=1.35)
    if sub:
        bg.text((fx0 + fx1) / 2, yc - 0.020, sub, ha="center", va="center",
                fontsize=5.0, color="0.38", zorder=4)


# ---- real image + derived rasters ----
img = None
for cand in ["data/AID/Airport/airport_1.jpg",
             "paper/figures/dota_v6d2_viz_1.png"]:
    if os.path.exists(cand):
        img = imread(cand); break
lum = img[..., :3].mean(-1).astype(float)
H, W = lum.shape
ph, pw = H // 14, W // 14
pool = lum[:ph * 14, :pw * 14].reshape(14, ph, 14, pw).mean((1, 3))
sal = np.abs(pool - pool.mean())
mask = (sal >= np.quantile(sal, 0.25)).astype(float)      # rho = 0.75 keep
rng = np.random.default_rng(0)
spk = (rng.random((14, 14)) < 0.10 + 0.25 * sal / sal.max()).astype(float)
score = (sal / sal.max()) ** 0.7

# =================== BANDS ===================
band(0.575, 0.975, "#f4f8fc", "#ccdcea",
     "U A V   T R A N S M I T T E R", BLUE)
band(0.460, 0.560, "#fdf5f3", "#eccfc9", "A 2 G   C H A N N E L", RED)
band(0.128, 0.450, "#f3f9f5", "#cbe3d3", "G R O U N D   S T A T I O N", GREEN)

# =================== TRANSMITTER ===================
rowA = 0.868
ax_t, ht = raster(0.045, rowA, 0.100, img, "0.55")
bg.text(0.095, rowA - ht / 2 - 0.012, "AID input $3{\\times}224^2$",
        fontsize=5.0, color="0.38", ha="center", va="top")
harrow(0.150, 0.196, rowA)
iso(0.200, 0.100, 0.072, rowA, "ResNet-50\nfront (1\u20133)", BLUE, "#eaf2f8",
    "#c3daed", "#9cc4e2")
chip(0.256, rowA + 0.055, "FROZEN", BLUE)
harrow(0.318, 0.358, rowA)
bg.text(0.338, rowA + 0.016, r"$F$", fontsize=5.8, color="0.30",
        ha="center", style="italic")
rbox(0.362, rowA - 0.034, 0.150, 0.068,
     "feature map\n$1024{\\times}14{\\times}14$", "#fdf9ec", "#d8c88f", 5.5)
harrow(0.516, 0.556, rowA)
rbox(0.560, rowA - 0.050, 0.225, 0.100,
     "SNN encoder\nConv\u2013BN\u2013LIF ($1024{\\to}256$)\nConv\u2013BN\u2013LIF ($256{\\to}36$)",
     "#fdf0e6", "#e3b48e", 5.3)
chip(0.6725, rowA + 0.068, "TRAINABLE \u00b7 T = 8", ORANGE)
bg.text(0.795, rowA - 0.062, "membrane feedback $m(t)$", fontsize=4.7,
        color="0.42", ha="left", va="top", style="italic")

rowB = 0.722
varrow(0.6725, rowA - 0.054, rowB + 0.052)
bg.text(0.684, (rowA + rowB) / 2 - 0.002, r"$S_2$: $36{\times}14{\times}14$",
        fontsize=5.0, color="0.30", ha="left", style="italic")
rbox(0.040, rowB - 0.052, 0.355, 0.104,
     "noise-aware scorer\ntime-avg $\\bar{S}$ \u00b7 reweight $(1{+}\\tanh$ MLP$(\\hat{p}))$\nconv $1{\\times}1 \\to$ scores $I \\in (0,1)^{14\\times 14}$",
     "#eef6f0", "#a9cdb4", 5.2)
chip(0.2175, rowB + 0.068, "SCORER", GREEN)
harrow(0.397, 0.427, rowB)
ax_sc, hs = raster(0.430, rowB, 0.080, score, "#a9cdb4", cmap="viridis")
bg.text(0.470, rowB + hs / 2 + 0.010, "scores $I$", fontsize=4.9,
        color="0.38", ha="center")
harrow(0.513, 0.543, rowB)
rbox(0.547, rowB - 0.038, 0.118, 0.076,
     "top-$k$ mask\n$\\rho = 0.75$", "#fdf0e6", "#e3b48e", 5.3)
harrow(0.668, 0.696, rowB)
bg.text(0.682, rowB + 0.015, r"$\odot$", fontsize=6.5, color="0.30",
        ha="center")
ax_m, hm = raster(0.700, rowB, 0.080, mask * spk, "#e3b48e", cmap="gray_r")
bg.text(0.740, rowB + hm / 2 + 0.010, "masked spikes", fontsize=4.9,
        color="0.38", ha="center")

# rate controller (the closed loop) fills the TX band lower-right
ctl_y = 0.615
bg.add_patch(FancyBboxPatch((0.335, ctl_y - 0.036), 0.275, 0.072,
             boxstyle="round,pad=0.004,rounding_size=0.010",
             facecolor="white", edgecolor=RED, lw=1.0, zorder=5))
bg.text(0.4725, ctl_y + 0.013, "rate controller (pilot-driven)",
        fontsize=5.5, color=INK, ha="center", va="center", zorder=6)
bg.text(0.4725, ctl_y - 0.016,
        r"$\rho^{*}(\hat{p}),\; T'^{*}(\hat{p})$ = cheapest meeting $\varepsilon$",
        fontsize=5.1, color=RED, ha="center", zorder=6)
chip(0.4725, ctl_y + 0.052, "CLOSED LOOP", RED)
varrow(0.582, ctl_y + 0.036, rowB - 0.040, color=RED)
bg.text(0.594, (ctl_y + rowB) / 2 - 0.002, r"$\rho^{*},\, T'^{*}$",
        fontsize=5.0, color=RED, ha="left", style="italic")
bg.text(0.045, 0.622,
        "payload $= \\rho\\, T'\\, C_2 H W$\n$= 42{,}336$ bits / image",
        fontsize=5.2, color="0.35", ha="left", style="italic",
        linespacing=1.35)

# =================== CHANNEL ===================
chy = 0.503
varrow(0.740, rowB - hm / 2 - 0.006, chy + 0.032)
rbox(0.420, chy - 0.032, 0.360, 0.064,
     "BSC$(p)$:  $y = x \\oplus e$,  $e \\sim$ Bern$(p)$  \u00b7  A2G BPSK, Rician $K(\\theta)$",
     "#fbeae6", "#dba79e", 5.2)
# pilot tap up the right rail into the controller
railx = 0.878
bg.plot([0.782, railx], [chy, chy], color=RED, lw=1.1, solid_capstyle="butt")
bg.plot([railx, railx], [chy, ctl_y], color=RED, lw=1.1,
        solid_capstyle="butt")
harrow(railx, 0.614, y=ctl_y, color=RED)
bg.text(0.886, (chy + ctl_y) / 2 + 0.020, r"pilot $\hat{p}$", fontsize=5.0,
        color=RED, ha="left", style="italic", rotation=90, va="center")

# =================== GROUND STATION ===================
rowD = 0.352
varrow(0.55, chy - 0.036, rowD + 0.056)
bg.text(0.562, 0.440, "received spikes", fontsize=4.9, color="0.30",
        ha="left", style="italic")
rbox(0.4375, rowD - 0.052, 0.225, 0.104,
     "SNN decoder\nConv\u2013BN\u2013LIF ($36{\\to}256$)\nConv\u2013BN\u2013LIF ($256{\\to}1024$)",
     "#eef6f0", "#a9cdb4", 5.3)
chip(0.468, rowD + 0.068, "TRAINABLE", GREEN)
rowE = 0.195
bg.plot([0.55, 0.55], [rowD - 0.052, 0.268], color=ARROW, lw=1.1,
        solid_capstyle="butt")
bg.plot([0.55, 0.220], [0.268, 0.268], color=ARROW, lw=1.1,
        solid_capstyle="butt")
varrow(0.220, 0.268, rowE + 0.042)
rbox(0.075, rowE - 0.040, 0.290, 0.080,
     "spike-to-feature converter\nstack $[S_4, m_4]$ over $T \\to$ gated linear",
     "#f4ecf3", "#c8a9c4", 5.2)
chip(0.220, rowE + 0.058, "CONVERTER", PURPLE)
harrow(0.369, 0.401, rowE)
bg.text(0.385, rowE + 0.015, r"$\hat{F}$", fontsize=5.8, color="0.30",
        ha="center", style="italic")
iso(0.405, 0.100, 0.072, rowE, "ResNet-50\nback (4+FC)", GREEN, "#eef6f0",
    "#cfe6d6", "#aed3ba")
chip(0.459, rowE + 0.055, "FINE-TUNED", GREEN)
harrow(0.523, 0.555, rowE)
rbox(0.559, rowE - 0.032, 0.215, 0.064,
     "classification\n$\\hat{y}$ = Airport $\\checkmark$", "#eaf5ec",
     GREEN, 5.7)

from matplotlib.transforms import Bbox
crop = Bbox([[0.0, 0.112 * FH], [FW, 0.985 * FH]])
out = "paper/figures/fig1_architecture_v3"
fig.savefig(out + ".pdf", bbox_inches=crop)
fig.savefig(out + ".png", dpi=220, bbox_inches=crop)
print("wrote", out)
