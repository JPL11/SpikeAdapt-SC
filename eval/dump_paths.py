#!/usr/bin/env python3
"""Dump representative per-block channel realizations (Lc=1 and Lc=16) plus
the ergodic BER path for the journal fading-trace figure."""
import sys
import numpy as np
sys.path.insert(0, "eval")
import channel_a2g as ch
from scipy.special import erfc

T = 100
H_M = 60.0
seg = np.concatenate([np.linspace(100, 1000, 45), np.full(10, 1000.0), np.linspace(1000, 150, 45)])
dg = seg

def q_ber(g):
    return float(np.clip(0.5 * erfc(np.sqrt(max(g, 1e-9))), 0, 0.45))

erg = np.array([ch.ber_bpsk_rician(ch.snr_db(H_M, g, "suburban"),
                                   ch.k_factor_db(ch.elevation_deg(H_M, g))) for g in dg])

def geo_params(g_dist):
    th = ch.elevation_deg(H_M, g_dist)
    d3 = ch.distance_3d_m(H_M, g_dist)
    plos = ch.p_los(th, "suburban")
    e = ch.ENVIRONMENTS["suburban"]
    noise_dbm = -174.0 + 10 * np.log10(ch.LINK["bw_mhz"] * 1e6) + ch.LINK["nf_db"]
    def snr_mean(los):
        pl = ch.fspl_db(d3, ch.LINK["fc_ghz"]) + (e["eta_los"] if los else e["eta_nlos"])
        return 10 ** ((ch.LINK["ptx_dbm"] + ch.LINK["gain_db"] - pl - noise_dbm) / 10)
    K = 10 ** (ch.k_factor_db(th) / 10)
    return plos, snr_mean, K

def draw(rng, Lc):
    a = 0.0 if Lc <= 1 else np.exp(-1.0 / Lc)
    stay = a
    p = np.zeros(T); losv = np.zeros(T, bool)
    los = None; hre = him = None
    for t, g_dist in enumerate(dg):
        plos, snr_mean, K = geo_params(g_dist)
        if los is None or rng.random() > stay:
            los = rng.random() < plos
        mu = np.sqrt(K / (K + 1)) if los else 0.0
        sd = np.sqrt(0.5 / (K + 1)) if los else np.sqrt(0.5)
        if hre is None or Lc <= 1:
            hre = mu + rng.normal(0, sd); him = rng.normal(0, sd)
        else:
            hre = mu + a * (hre - mu) + np.sqrt(1 - a * a) * rng.normal(0, sd)
            him = a * him + np.sqrt(1 - a * a) * rng.normal(0, sd)
        p[t] = q_ber(snr_mean(los) * (hre * hre + him * him))
        losv[t] = los
    return p, losv

p1, l1 = draw(np.random.default_rng(7), 1)
p16, l16 = draw(np.random.default_rng(7), 16)
np.savez("eval/seed_results/fading_paths.npz", erg=erg, p_lc1=p1, los_lc1=l1, p_lc16=p16, los_lc16=l16, dist=dg)
print("wrote fading_paths.npz")
