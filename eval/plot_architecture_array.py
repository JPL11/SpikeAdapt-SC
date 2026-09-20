#!/usr/bin/env python3
"""Fig 1 for main_array.tex, in the ICC paper's v3 visual language
(eval/plot_architecture_v3.py): three tinted system bands, Times serif,
colored role chips, and the closed rate-adaptation loop as a right-side
feedback rail. Radar edition: real RADIal range-Doppler input, frozen
FFTRadNet pyramid, spikes-only bottleneck (x3 scales), ground-side
RA decoder + heads. Output: paper/figures/fig1_architecture_array.{pdf,png}
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

BLUE, ORANGE, GREEN = "#2470a8", "#d1611e", "#2c8a5a"
RED, INK, ARROW, PURPLE = "#b03a3a", "#1b1b1b", "#3a3a3a", "#7d5185"
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
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
    # box sized for the post-hoc 1.42x text scaling (see end of file)
    w = 0.020 + 0.0127 * len(text)
    bg.add_patch(FancyBboxPatch((xc - w / 2, yc - 0.016), w, 0.032,
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
                fontsize=5.0, color="0.28", zorder=4)


# ---- real RD frame + derived rasters ----
rd = None
for cand in ["data/RADIal/radar_FFT/fft_000018.npy",
             "data/RADIal/radar_FFT/fft_000037.npy"]:
    if os.path.exists(cand):
        arr = np.load(cand)                     # (512, 256, 16) complex64
        rd = np.log1p(np.abs(arr[..., 0]).astype(float))[:256, :]
        break
if rd is None:  # fallback: synthetic RD texture
    rng0 = np.random.default_rng(1)
    rd = rng0.gamma(2.0, 1.0, (256, 128))
H, W = rd.shape
ph, pw = H // 14, W // 14
pool = rd[:ph * 14, :pw * 14].reshape(14, ph, 14, pw).mean((1, 3))
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
ax_t, ht = raster(0.045, rowA, 0.100, rd, "0.55", cmap="viridis")
bg.text(0.020, rowA - ht / 2 - 0.004, "RD spectrum $32{\\times}512{\\times}256$",
        fontsize=5.0, color="0.28", ha="left", va="top")
harrow(0.150, 0.196, rowA)
iso(0.200, 0.100, 0.072, rowA, "FFTRadNet\npyramid", BLUE, "#eaf2f8",
    "#c3daed", "#9cc4e2")
chip(0.256, rowA + 0.055, "FROZEN", BLUE)
harrow(0.318, 0.358, rowA)
bg.text(0.338, rowA + 0.016, r"$x_i$", fontsize=5.8, color="0.28",
        ha="center", style="italic")
rbox(0.362, rowA - 0.040, 0.150, 0.080,
     "scales $x_2,x_3,x_4$\n$160/192/224$ ch\n$58.2$ Mbit fp32", "#fdf9ec",
     "#d8c88f", 5.2)
harrow(0.516, 0.556, rowA)
rbox(0.560, rowA - 0.050, 0.225, 0.100,
     "spiking encoder $\\times 3$\nConv–MPBN–LIF $\\to C_s{=}36$\nbinary maps, $T{=}8$ steps",
     "#fdf0e6", "#e3b48e", 5.3)
chip(0.6725, rowA + 0.068, "TRAINABLE · T = 8", ORANGE)

rowB = 0.722
# encoder spikes feed the element-wise mask operator (odot), NOT the mask block
varrow(0.682, rowA - 0.054, 0.742)
# spike branch feeding the noise-aware scorer (time-averaged spikes)
bg.plot([0.682, 0.355], [0.800, 0.800], color=ARROW, lw=1.0,
        solid_capstyle="butt")
varrow(0.355, 0.800, rowB + 0.052, color=ARROW, lw=1.0)
bg.text(0.52, 0.808, r"$s_{i,t}$ (time-avg.)", fontsize=4.9, color="0.28",
        ha="center", style="italic")
rbox(0.040, rowB - 0.052, 0.355, 0.104,
     "noise-aware scorer\ntime-avg $\\bar{S}$ · reweight$(\\hat{p})$\n$\\to$ importance $I$ per position",
     "#eef6f0", "#a9cdb4", 5.2)
chip(0.2175, rowB + 0.068, "SCORER", GREEN)
harrow(0.397, 0.427, rowB)
ax_sc, hs = raster(0.430, rowB, 0.080, score, "#a9cdb4", cmap="viridis")
bg.text(0.470, rowB + hs / 2 + 0.010, "scores $I$", fontsize=4.9,
        color="0.28", ha="center")
harrow(0.513, 0.543, rowB)
rbox(0.547, rowB - 0.046, 0.118, 0.092,
     "spatial mask $\\rho$\n$+$ temporal\ngate $T'$", "#fdf0e6", "#e3b48e", 5.0)
harrow(0.668, 0.696, rowB)
bg.text(0.682, rowB + 0.004, r"$\odot$", fontsize=6.5, color="0.28",
        ha="center")
ax_m, hm = raster(0.700, rowB, 0.080, mask * spk, "#e3b48e", cmap="gray_r")
bg.text(0.700, rowB + hm / 2 + 0.010, "binary spike payload", fontsize=5.4,
        color="0.28", ha="left")

# rate controller (the closed loop)
ctl_y = 0.615
bg.add_patch(FancyBboxPatch((0.315, ctl_y - 0.040), 0.325, 0.080,
             boxstyle="round,pad=0.004,rounding_size=0.010",
             facecolor="white", edgecolor=RED, lw=1.0, zorder=5))
bg.text(0.4775, ctl_y + 0.015, "rate controller (pilot-driven)",
        fontsize=5.3, color=INK, ha="center", va="center", zorder=6)
bg.text(0.4775, ctl_y - 0.018,
        r"min-payload $(\rho,T')$ within $\varepsilon$ of best val. $F_H$",
        fontsize=5.0, color=RED, ha="center", zorder=6)
chip(0.4775, ctl_y + 0.058, "CLOSED LOOP", RED)
varrow(0.582, ctl_y + 0.036, rowB - 0.046, color=RED)
bg.text(0.594, (ctl_y + rowB) / 2 - 0.002, r"$\rho^{*},\, T'^{*}$",
        fontsize=5.0, color=RED, ha="left", style="italic")
# pilot estimate conditions the noise-aware scorer (reweight(p-hat))
varrow(0.360, ctl_y + 0.036, rowB - 0.052, color=RED, lw=1.0)
bg.text(0.368, (ctl_y + rowB) / 2 - 0.004, r"$\hat p$", fontsize=5.2,
        color=RED, ha="left", style="italic")
bg.text(0.045, 0.620,
        "spike payload $2.32$ Mbit/frame\n($\\rho{=}0.75$, $T{=}8$)",
        fontsize=5.8, color="0.28", ha="left", style="italic",
        linespacing=1.35)

# =================== CHANNEL ===================
chy = 0.503
varrow(0.740, rowB - hm / 2 - 0.006, chy + 0.032)
bg.text(0.752, (rowB - hm / 2 + chy) / 2 - 0.012,
        "$+$ support bitmap/\npilot ($\\approx1\\%$)", fontsize=5.1,
        color="0.28", ha="left", va="center", style="italic", linespacing=1.3)
rbox(0.420, chy - 0.032, 0.360, 0.064,
     "BPSK / Rician A2G $\\rightarrow$ hard-decision BSC$(p)$",
     "#fbeae6", "#dba79e", 5.8)
railx = 0.878
# RX-side pilot detector / BER estimator (ground station), fed from the channel
rbox(0.755, 0.368, 0.155, 0.046, "pilot detector /\nBER estimator",
     "#fbeae6", RED, 5.0)
bg.plot([0.782, 0.833], [chy, chy], color=RED, lw=1.1, solid_capstyle="butt",
        zorder=6)
bg.annotate("", xy=(0.833, 0.418), xytext=(0.833, chy),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                            shrinkA=0, shrinkB=0, zorder=6))
# low-rate feedback rail: estimator top -> controller (dashed = feedback)
bg.plot([railx, railx], [0.416, ctl_y], color=RED, lw=1.1, ls=(0, (4, 2)),
        solid_capstyle="butt", zorder=6)
bg.annotate("", xy=(0.644, ctl_y), xytext=(railx, ctl_y),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color=RED,
                            shrinkA=0, shrinkB=0, zorder=6,
                            linestyle=(0, (4, 2))))
bg.text(0.886, (0.416 + ctl_y) / 2 + 0.012, r"low-rate feedback $\hat{p}$",
        fontsize=5.2, color=RED, ha="left", style="italic", rotation=90,
        va="center")
# path-style key
bg.text(0.045, 0.492,
        "black: payload $\\cdot$ red solid: control $(\\rho^{*},T'^{*})$\nred dashed: feedback $\\hat p$",
        fontsize=4.9, color="0.30", ha="left", va="center", linespacing=1.5)

# =================== GROUND STATION ===================
rowD = 0.352
varrow(0.55, chy - 0.036, rowD + 0.056)
bg.text(0.562, 0.440, r"corrupted maps $y_{i,t}$", fontsize=4.9, color="0.28",
        ha="left", style="italic")
rbox(0.4375, rowD - 0.052, 0.225, 0.104,
     "spike decoder $\\times 3$\nlearned temporal weights\nConv $\\to \\hat{x}_2,\\hat{x}_3,\\hat{x}_4$",
     "#eef6f0", "#a9cdb4", 5.3)
chip(0.468, rowD + 0.068, "TRAINABLE", GREEN)
rowE = 0.195
bg.plot([0.55, 0.55], [rowD - 0.052, 0.285], color=ARROW, lw=1.1,
        solid_capstyle="butt")
bg.plot([0.55, 0.110], [0.285, 0.285], color=ARROW, lw=1.1,
        solid_capstyle="butt")
varrow(0.110, 0.285, rowE + 0.042)
rbox(0.075, rowE - 0.040, 0.290, 0.080,
     "range–azimuth decoder\nmulti-scale fusion (FFTRadNet)",
     "#f4ecf3", "#c8a9c4", 5.2)
chip(0.220, rowE + 0.058, "FINE-TUNED", PURPLE)
harrow(0.369, 0.401, rowE)
bg.text(0.385, rowE + 0.015, r"RA", fontsize=5.4, color="0.28",
        ha="center", style="italic")
iso(0.405, 0.100, 0.072, rowE, "det. + seg.\nheads", GREEN, "#eef6f0",
    "#cfe6d6", "#aed3ba")
chip(0.459, rowE + 0.055, "FINE-TUNED", GREEN)
harrow(0.523, 0.555, rowE)
rbox(0.559, rowE - 0.032, 0.215, 0.064,
     "vehicles $\\checkmark$\nfree space $\\checkmark$", "#eaf5ec",
     GREEN, 5.7)

from matplotlib.transforms import Bbox
# Enlarge ALL text uniformly for legibility at print size (the figure is
# downscaled to ~column width). Scaling every Text object keeps the layout
# proportions identical while lifting effective point sizes.
for _t in fig.findobj(plt.Text):
    _t.set_fontsize(_t.get_fontsize() * 1.15)
crop = Bbox([[0.0, 0.112 * FH], [FW, 0.985 * FH]])
out = "paper/figures/fig1_architecture_array"
fig.savefig(out + ".pdf", bbox_inches=crop)
fig.savefig(out + ".png", dpi=220, bbox_inches=crop)
print("wrote", out)
