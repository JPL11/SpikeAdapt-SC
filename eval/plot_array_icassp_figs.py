#!/usr/bin/env python3
"""ICASSP-2027 figures for paper/main_array_icassp.tex.

Regenerates fig2 (detection crossover) and fig3 with pdf.fonttype=42 (NO Type-3
fonts) and folds in the leakage-free closed loop + the Ziv-Zakai bound.

fig3 = 3 panels: (a) leakage-free closed-loop trajectory over the A2G profile
(val-selected policy, seed 42 shown); (b) DoA transport RMSE vs channel BER;
(c) DoA estimation bounds vs array SNR -- MUSIC RMSE, sqrt(CRB), sqrt(ZZB) --
showing the threshold region the local CRB misses.

Usage: python eval/plot_array_icassp_figs.py
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- NO Type-3 fonts: force TrueType (Type 42) embedding for text + mathtext.
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'font.size': 10.5, 'axes.labelsize': 11.5,
    'axes.titlesize': 11, 'legend.fontsize': 8.0,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = os.path.join(ROOT, 'eval/seed_results')
OUT = os.path.join(ROOT, 'paper/figures')
import sys
sys.path.insert(0, ROOT)
from eval.channel_a2g import distance_profile  # noqa: E402

BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
GBERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
RHOS, TPRIMES = [0.5, 0.75, 1.0], [4, 6, 8]


def series(d, key='mAP', prefix='ber'):
    return [d[f'{prefix}{b}'][key] for b in BERS]


def fh_series(d, prefix='ber'):
    out = []
    for b in BERS:
        P, R = d[f'{prefix}{b}']['mAP'], d[f'{prefix}{b}']['mAR']
        out.append(2 * P * R / (P + R) if (P + R) > 0 else 0.0)
    return out


def fig2(ylink=False):
    snn = {s: json.load(open(os.path.join(SR, f)))
           for s, f in [(42, 'radial_snn_full_sweep_so.json'),
                        (123, 'radial_snn_full_sweep_s123_so.json'),
                        (456, 'radial_snn_full_sweep_s456_so.json')]}
    arr = np.array([series(v) for v in snn.values()])
    q = json.load(open(os.path.join(SR, 'radial_quant_baseline.json')))
    ldpc = json.load(open(os.path.join(SR, 'radial_ldpc_baseline.json')))
    q1s = [json.load(open(os.path.join(SR, f'radial_quant1_retrained_sweep{t}.json')))
           for t in ['', '_s123', '_s456']]

    fair = [json.load(open(os.path.join(SR, f'radial_adaptive1bit_sweep_{t}.json')))
            for t in ['s42', 's123', 's456']]
    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    if ylink:
        # main SNN curve: physical receiver (y_link) from the grid, test split,
        # rho0.75/T8 (Table I bold row); y_stress as a thin dashed secondary.
        gl = json.load(open(os.path.join(SR, 'radial_ylink_grid_valtest.json')))
        gs = json.load(open(os.path.join(SR, 'radial_grid_valtest.json')))
        cell = 'rho0.75_T8_ber{}'
        def cellser(g):
            return np.array([[f1(g[s]['test'][cell.format(b)]) for b in BERS]
                             for s in ['42', '123', '456']])
        arr = cellser(gl)
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.plot(BERS, m, 'o-', color='#c1272d', lw=2,
                label='SNN, $y^{\mathrm{link}}$ (2.3 Mbit, ours)')
        ax.fill_between(BERS, m - sd, m + sd, color='#c1272d', alpha=0.25)
        ms = cellser(gs).mean(0)
        ax.plot(BERS, ms, '--', color='#c1272d', lw=1.0, alpha=0.75,
                label='SNN, $y^{\mathrm{stress}}$')
    else:
        arr = np.array([fh_series(v) for v in snn.values()])
        m, sd = arr.mean(0), arr.std(0, ddof=1)
        ax.plot(BERS, m, 'o-', color='#c1272d', lw=2,
                label='SNN, stress-grid protocol (2.3 Mbit, ours)')
        ax.fill_between(BERS, m - sd, m + sd, color='#c1272d', alpha=0.25)
    fa = np.array([fh_series(v) for v in fair])
    fm, fsd = fa.mean(0), fa.std(0, ddof=1)
    ax.plot(BERS, fm, 'P-', color='#7b3294', lw=1.6,
            label='1-bit fair, matched+trunc.-tr. (2.3 Mbit)')
    ax.fill_between(BERS, fm - fsd, fm + fsd, color='#7b3294', alpha=0.18)
    q1a = np.array([fh_series(v) for v in q1s])
    q1m, q1sd = q1a.mean(0), q1a.std(0, ddof=1)
    ax.plot(BERS, q1m, 's-', color='#e69f00', lw=1.5, label='1-bit retrained (1.8 Mbit)')
    ax.fill_between(BERS, q1m - q1sd, q1m + q1sd, color='#e69f00', alpha=0.18)
    ax.plot(BERS, fh_series(q['q8']), '^--', color='#4477aa', lw=1.5, label='Quant-8 (14.5 Mbit)')
    ax.plot(BERS, fh_series(ldpc['ldpc_r12']), 'v--', color='#117733', lw=1.5,
            label='Quant-8+LDPC 1/2 (29.1 Mbit)')
    ax.plot(BERS, fh_series(ldpc['ldpc_r13']), 'd--', color='#332288', lw=1.5,
            label='Quant-8+LDPC 1/3 (43.6 Mbit)')
    ax.axhline(0.9498, color='gray', ls=':', lw=1)
    ax.text(0.001, 1.005, 'released anchor', color='gray', fontsize=6.5)
    ax.axhline(0.9631, color='gray', ls='--', lw=0.8)
    ax.text(0.115, 1.005, 'fine-tuned anchor (both: no channel)', color='gray', fontsize=6.5)
    ax.set_xlabel('Channel BER $p$')
    ax.set_ylabel(r'Detection $F_H$')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, framealpha=0.9, fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_array_crossover.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig2_array_crossover.png'), dpi=200, bbox_inches='tight')
    print('fig2 done')


def f1(m):
    p, r = m['mAP'], m['mAR']
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def build_policy(gval, eps=0.01):
    """Min-payload (rho,T') within eps of best F1 per BER, on the VAL grid."""
    pol = {}
    for b in GBERS:
        cells = []
        for r in RHOS:
            for t in TPRIMES:
                c = gval[f'rho{r}_T{t}_ber{b}']
                cells.append((r, t, f1(c), c['payload_bits']))
        best = max(c[2] for c in cells)
        ok = [c for c in cells if c[2] >= best - eps]
        r, t, _, pay = min(ok, key=lambda c: c[3])
        pol[b] = (r, t, pay)
    return pol


def fig3(ylink=False):
    gname = ('radial_ylink_grid_valtest.json' if ylink
             else 'radial_grid_valtest.json')
    grid = json.load(open(os.path.join(SR, gname)))
    # (a) leakage-free trajectory: policy on seed-42 VAL, scored on seed-42 TEST
    pol = build_policy(grid['42']['val'])
    gt = grid['42']['test']
    full_pay = gt['rho1.0_T8_ber0.0']['payload_bits']
    prof = [r for r in distance_profile(100.0, 'suburban')
            if 200 <= r['d_ground_m'] <= 3000 and r['ber'] <= 0.40]
    d_km, m_ad, m_fx, pay_ad = [], [], [], []
    for r in prof:
        nb = min(GBERS, key=lambda b: abs(b - r['ber']))
        rr, tt, pay = pol[nb]
        d_km.append(r['d_ground_m'] / 1000)
        m_ad.append(gt[f'rho{rr}_T{tt}_ber{nb}']['mAP'])
        m_fx.append(gt[f'rho1.0_T8_ber{nb}']['mAP'])
        pay_ad.append(pay / 1e6)
    clname = ('radial_closedloop_valtest_ylink.json' if ylink
              else 'radial_closedloop_valtest.json')
    cl = json.load(open(os.path.join(SR, clname)))
    doa_b = json.load(open(os.path.join(SR, 'doa_sim_baselines.json')))
    doa = {s: json.load(open(os.path.join(SR, f'doa_sim_snn_seed{s}.json')))
           for s in [42, 123, 456]}
    zzb = json.load(open(os.path.join(SR, 'doa_bounds_snr_sweep.json')))['sweep']

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 2.9))

    # (a) closed-loop trajectory
    ax = axes[0]
    if ylink:
        # same comparator as Table II's dagger row: F_H-selected fixed cell
        # (rho=0.5, T'=8), mission-mean mAP on this seed's test surface
        bf = float(np.mean([gt[f"rho0.5_T8_ber{min(GBERS, key=lambda b: abs(b - r['ber']))}"]['mAP']
                            for r in prof]))
        bf_label = f"test-oracle fixed ($\\bar P{{=}}{bf:.3f}$)"
    else:
        bf = cl['best_fixed_below']['mAP']
        bf_label = f"best fixed ($\\bar P{{=}}{bf:.3f}$)"
    ax.plot(d_km, m_ad, '-', color='#c1272d', lw=2, label="closed loop $(\\rho,T')$")
    ax.plot(d_km, m_fx, '--', color='#4477aa', lw=1.5, label="fixed $\\rho{=}1,T'{=}8$")
    ax.axhline(bf, ls='-.', color='#117733', lw=1.3, label=bf_label)
    ax2 = ax.twinx()
    ax2.plot(d_km, pay_ad, ':', color='#c1272d', lw=1.2,
             label='adaptive payload (right axis)')
    ax2.axhline(full_pay / 1e6, ls=':', color='#4477aa', lw=1.2,
                label='full-rate payload (right axis)')
    ax2.set_ylabel('Payload (Mbit, dotted)', fontsize=9)
    ax2.set_ylim(0, 2.6)
    ax.set_title("(a) Representative validation-selected A2G trajectory (seed 42)")
    ax.set_xlabel('Ground distance (km), $h{=}100$ m')
    ax.set_ylabel('$\\bar P$ (mean precision)')
    ax.set_ylim(0.7, 1.0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, -0.34),
              ncol=3, fontsize=5.2, framealpha=0.9)
    ax.grid(alpha=0.3)

    # (b) DoA transport vs BER
    ax = axes[1]
    qc = [doa_b['quantcov_rmse_vs_ber'][str(b)] for b in BERS]
    ax.semilogy(BERS, qc, '^--', color='#4477aa', lw=1.5, label='quant-cov + MUSIC (1536 b)')
    key = 'rho0.75_T8_ber{}'
    arr = np.array([[doa[s]['snn_grid'][key.format(b)]['rmse_deg'] for b in BERS]
                    for s in doa])
    m_, sd = arr.mean(0), arr.std(0, ddof=1)
    ax.semilogy(BERS, m_, 'o-', color='#c1272d', lw=2, label='SNN codec (1536 b, ours)')
    ax.fill_between(BERS, m_ - sd, m_ + sd, color='#c1272d', alpha=0.32)
    ax.axhline(doa_b['rmse_crb'], color='gray', ls=':', lw=1)
    ax.text(0.002, doa_b['rmse_crb'] * 1.3, r'$\sqrt{\mathrm{CRB}}$', color='gray', fontsize=8)
    ax.set_title('(b) DoA transport, matched 1536-bit payload')
    ax.set_xlabel('Channel BER $p$')
    ax.set_ylabel('RMSE (deg)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.34), ncol=3, fontsize=6, framealpha=0.9)
    ax.grid(alpha=0.3, which='both')

    # (c) DoA estimation bounds vs SNR: MUSIC, CRB, ZZB (threshold region)
    ax = axes[2]
    snr = [r['snr_db'] for r in zzb]
    ax.semilogy(snr, [r['rmse_music'] for r in zzb], 'o-', color='#c1272d', lw=2,
                label='MUSIC RMSE')
    ax.semilogy(snr, [r['sqrt_zzb'] for r in zzb], 's-', color='#117733', lw=1.6,
                label=r'relaxed $\sqrt{\mathrm{ZZB}}$')
    ax.semilogy(snr, [r['sqrt_crb'] for r in zzb], '--', color='gray', lw=1.4,
                label=r'$\sqrt{\mathrm{CRB}}$')
    ax.axvspan(-15, -11, color='#c1272d', alpha=0.07)
    ax.text(-10.4, 3.2, 'threshold onset\n(illustrative)', fontsize=6.5,
            color='#c1272d', ha='left')
    ax.set_title('(c) Relaxed ZZB indicates the threshold')
    ax.set_xlabel('Array SNR (dB)')
    ax.set_ylabel('RMSE / bound (deg)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.34), ncol=3, fontsize=6, framealpha=0.9)
    ax.grid(alpha=0.3, which='both')

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.34, top=0.88, wspace=0.34)
    fig.savefig(os.path.join(OUT, 'fig3_array_loop_doa.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig3_array_loop_doa.png'), dpi=200, bbox_inches='tight')
    print('fig3 done')


if __name__ == '__main__':
    apar = argparse.ArgumentParser()
    apar.add_argument('--ylink', action='store_true',
                      help='physical-receiver primary: SNN curves/policy from '
                           'the y_link grid, y_stress secondary in fig2')
    a = apar.parse_args()
    fig2(a.ylink)
    fig3(a.ylink)
