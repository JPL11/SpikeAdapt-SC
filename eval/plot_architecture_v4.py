#!/usr/bin/env python3
"""Fig 1 architecture v4 (review round 6): UAV-side controller with p-hat
feedback, three explicit control arrows (p-hat->scorer, T'*->encoder,
rho*->mask), encoder->scorer spike input, ground-side pilot estimator,
adaptive labels, enlarged small text. Based on v3: three tinted system bands
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
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 7.6})

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
bg.text(0.095, rowA - ht / 2 - 0.012, "AID input $3{\\times}224{\\times}224$",
        fontsize=5.0, color="0.38", ha="center", va="top")
harrow(0.150, 0.196, rowA)
iso(0.200, 0.100, 0.072, rowA, "ResNet-50\nfront (1\u20133)", BLUE, "#eaf2f8",
    "#c3daed", "#9cc4e2")
chip(0.256, rowA + 0.055, "FROZEN", BLUE)
harrow(0.318, 0.358, rowA)
bg.text(0.338, rowA + 0.016, r"$F$", fontsize=6.4, color="0.30",
        ha="center", style="italic")
rbox(0.362, rowA - 0.034, 0.150, 0.068,
     "feature map\n$1024{\\times}14{\\times}14$", "#fdf9ec", "#d8c88f", 5.5)
harrow(0.516, 0.556, rowA)
rbox(0.560, rowA - 0.050, 0.225, 0.100,
     "SNN encoder\nConv\u2013BN\u2013LIF ($1024{\\to}256$)\nConv\u2013BN\u2013LIF ($256{\\to}36$)",
     "#fdf0e6", "#e3b48e", 5.3)
chip(0.6725, rowA + 0.068, "TRAINED T=8 \u00b7 INFER T' \u2264 8", ORANGE, fs=4.9)
bg.text(0.6725, rowA + 0.090, "optional $+T'$aug fine-tune", fontsize=5.6,
        color="0.35", ha="center", style="italic")

rowB = 0.722
varrow(0.6725, rowA - 0.054, rowB + 0.052)
bg.text(0.684, (rowA + rowB) / 2 - 0.002,
        r"$\{S^{(2)}_t\}_{t\leq T'}$: $T'{\times}36{\times}14{\times}14$",
        fontsize=6.2, color="0.15", ha="left", style="italic")
bg.plot([0.640, 0.640, 0.2175], [rowA - 0.054, rowB + 0.076, rowB + 0.076],
        color=ARROW, lw=1.0, solid_capstyle="butt")
varrow(0.2175, rowB + 0.076, rowB + 0.054)
bg.text(0.300, rowB + 0.085, r"time-avg $\bar{S}$", fontsize=6.4,
        color="0.25", ha="center", style="italic")
rbox(0.040, rowB - 0.052, 0.355, 0.104,
     "noise-aware scorer\ntime-avg $\\bar{S}$ \u00b7 reweight $(1{+}\\tanh$ MLP$(\\tilde{p}))$\nconv $1{\\times}1 \\to$ scores $I \\in (0,1)^{14\\times 14}$",
     "#eef6f0", "#a9cdb4", 5.2)
chip(0.2175, rowB + 0.068, "SCORER", GREEN)
harrow(0.397, 0.427, rowB)
ax_sc, hs = raster(0.430, rowB, 0.080, score, "#a9cdb4", cmap="viridis")
bg.text(0.470, rowB + hs / 2 + 0.010, "scores $I$", fontsize=6.2,
        color="0.25", ha="center")
harrow(0.513, 0.543, rowB)
rbox(0.547, rowB - 0.038, 0.118, 0.076,
     "top-$k$ mask\n$k=\\lfloor\\rho^{*}HW\\rfloor$", "#fdf0e6", "#e3b48e", 5.4)
harrow(0.668, 0.696, rowB)
bg.text(0.682, rowB + 0.015, r"$\odot$", fontsize=6.5, color="0.30",
        ha="center")
ax_m, hm = raster(0.700, rowB, 0.080, mask * spk, "#e3b48e", cmap="gray_r")
bg.text(0.740, rowB + hm / 2 + 0.010, "sparse spikes", fontsize=6.3,
        color="0.20", ha="center")
bg.text(0.740, rowB - hm / 2 - 0.014, "+ 196-bit mask", fontsize=6.2,
        color="0.30", ha="center")

# rate controller (the closed loop) fills the TX band lower-right
ctl_y = 0.615
bg.add_patch(FancyBboxPatch((0.335, ctl_y - 0.036), 0.275, 0.072,
             boxstyle="round,pad=0.004,rounding_size=0.010",
             facecolor="white", edgecolor=RED, lw=1.0, zorder=5))
bg.text(0.4725, ctl_y + 0.013, "rate controller (pilot-driven)",
        fontsize=6.3, color=INK, ha="center", va="center", zorder=6)
bg.text(0.4725, ctl_y - 0.016,
        r"$\rho^{*}(\tilde{p}),\; T'^{*}(\tilde{p})$ = cheapest meeting $\varepsilon$",
        fontsize=5.9, color=RED, ha="center", zorder=6)
chip(0.4725, ctl_y + 0.052, "CLOSED LOOP", RED)
bg.annotate("", xy=(0.606, rowB - 0.040), xytext=(0.606, ctl_y + 0.006),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                            shrinkA=0, shrinkB=0, zorder=6))
bg.text(0.612, (ctl_y + rowB) / 2 - 0.002, r"$\rho^{*}$",
        fontsize=6.2, color=RED, ha="left", style="italic")
bg.plot([0.610, 0.845], [ctl_y + 0.022, ctl_y + 0.022], color=RED, lw=1.1,
        solid_capstyle="butt")
bg.plot([0.845, 0.845], [ctl_y + 0.022, rowA], color=RED, lw=1.1,
        solid_capstyle="butt")
harrow(0.845, 0.790, rowA, color=RED)
bg.text(0.853, (ctl_y + rowA) / 2, r"$T'^{*}$", fontsize=6.2,
        color=RED, ha="left", style="italic", rotation=90, va="center")
bg.plot([0.340, 0.275], [ctl_y, ctl_y], color=RED, lw=1.1,
        ls=(0, (4, 2)), solid_capstyle="butt", zorder=6)
bg.plot([0.335 - 0.06, 0.335 - 0.06], [ctl_y, rowB - 0.058], color=RED,
        lw=0)  # spacer no-op
bg.plot([0.275, 0.275], [ctl_y, rowB - 0.056], color=RED, lw=1.1,
        ls=(0, (4, 2)), solid_capstyle="butt", zorder=6)
varrow(0.275, rowB - 0.056, rowB - 0.052, color=RED)
bg.text(0.267, (ctl_y + rowB - 0.05) / 2, r"$\tilde{p}$", fontsize=6.4,
        color=RED, ha="right", style="italic")
bg.text(0.045, 0.596,
        "semantic payload $= T'^{*} C_2 \\lfloor\\rho^{*} H W\\rfloor$\n$\\leq 42{,}336$ channel uses ($=$bits, BPSK)",
        fontsize=5.4, color="0.35", ha="left", style="italic",
        linespacing=1.35)

# =================== CHANNEL ===================
chy = 0.503
varrow(0.740, rowB - hm / 2 - 0.006, chy + 0.032)
rbox(0.420, chy - 0.032, 0.330, 0.064,
     "BPSK / Rician A2G $\\rightarrow$ hard-decision BSC$(p)$",
     "#fbeae6", "#dba79e", 5.8)
# pilot tap up the right rail into the controller
railx = 0.878
rbox(0.760, 0.400, 0.150, 0.046, "pilot detector /\nBER estimator",
     "#fbeae6", RED, 5.2)
bg.plot([0.752, 0.835], [chy, chy], color=RED, lw=1.1,
        solid_capstyle="butt", zorder=6)
bg.annotate("", xy=(0.835, 0.448), xytext=(0.835, chy),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                            shrinkA=0, shrinkB=0, zorder=6))
bg.plot([railx, railx], [0.444, ctl_y], color=RED, lw=1.1, ls=(0, (4, 2)),
        solid_capstyle="butt", zorder=6)
bg.annotate("", xy=(0.614, ctl_y), xytext=(railx, ctl_y),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                            shrinkA=0, shrinkB=0, zorder=6,
                            linestyle=(0, (4, 2))))
bg.text(0.886, 0.52, r"feedback $\hat{p}$", fontsize=6.6,
        color=RED, ha="left", style="italic", rotation=90, va="center")

# =================== GROUND STATION ===================
rowD = 0.352
varrow(0.55, chy - 0.036, rowD + 0.056)
bg.text(0.562, 0.440, "received spikes", fontsize=5.6, color="0.30",
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
     "spike-to-feature converter\nstack over $T' \\to$ zero-pad to $T$ $\\to$ gated linear",
     "#f4ecf3", "#c8a9c4", 5.2)
chip(0.220, rowE + 0.058, "CONVERTER", PURPLE)
harrow(0.369, 0.401, rowE)
bg.text(0.385, rowE + 0.015, r"$\hat{F}$", fontsize=6.4, color="0.30",
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
out = "paper/figures/fig1_architecture_v4"
fig.savefig(out + ".pdf", bbox_inches=crop)
fig.savefig(out + ".png", dpi=220, bbox_inches=crop)
print("wrote", out)
