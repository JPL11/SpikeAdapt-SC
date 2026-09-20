#!/usr/bin/env python3
"""Figures for paper/main_array.tex (repo paper-figure conventions).

Fig. 2: crossover — mAP vs BER for SNN (3-seed band) vs LDPC-protected /
        unprotected quantized transport vs 1-bit retrained control.
Fig. 3: (a) closed-loop trajectory over the A2G profile; (b) DoA RMSE vs BER
        (SNN 3-seed vs quantized-covariance MUSIC, CRB line).

Usage: python eval/plot_array_paper_figs.py
Outputs: paper/figures/fig2_array_crossover.{pdf,png},
         paper/figures/fig3_array_loop_doa.{pdf,png}
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = os.path.join(ROOT, 'eval/seed_results')
OUT = os.path.join(ROOT, 'paper/figures')
BERS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

plt.rcParams.update({'font.size': 10.5, 'axes.labelsize': 11.5,
                     'axes.titlesize': 12, 'legend.fontsize': 8.0})


def series(d, key='mAP', prefix='ber'):
    return [d[f'{prefix}{b}'][key] for b in BERS]


def fig2():
    snn = {s: json.load(open(os.path.join(SR, f)))
           for s, f in [(42, 'radial_snn_full_sweep_so.json'),
                        (123, 'radial_snn_full_sweep_s123_so.json'),
                        (456, 'radial_snn_full_sweep_s456_so.json')]}
    arr = np.array([series(v) for v in snn.values()])
    q = json.load(open(os.path.join(SR, 'radial_quant_baseline.json')))
    ldpc = json.load(open(os.path.join(SR, 'radial_ldpc_baseline.json')))
    q1s = [json.load(open(os.path.join(SR, f'radial_quant1_retrained_sweep{t}.json')))
           for t in ['', '_s123', '_s456']]

    fig, ax = plt.subplots(figsize=(4.9, 3.4))
    m, sd = arr.mean(0), arr.std(0, ddof=1)
    ax.plot(BERS, m, 'o-', color='#c1272d', lw=2, label='SNN spikes-only (2.3 Mbit, ours)')
    ax.fill_between(BERS, m - sd, m + sd, color='#c1272d', alpha=0.2)
    q1a = np.array([series(v) for v in q1s])
    q1m, q1sd = q1a.mean(0), q1a.std(0, ddof=1)
    ax.plot(BERS, q1m, 's-', color='#e69f00', lw=1.5,
            label='1-bit retrained (1.8 Mbit)')
    ax.fill_between(BERS, q1m - q1sd, q1m + q1sd, color='#e69f00', alpha=0.18)
    ax.plot(BERS, series(q['q8']), '^--', color='#4477aa', lw=1.5,
            label='Quant-8 (14.5 Mbit)')
    ax.plot(BERS, series(ldpc['ldpc_r12']), 'v--', color='#117733', lw=1.5,
            label='Quant-8+LDPC 1/2 (29.1 Mbit)')
    ax.plot(BERS, series(ldpc['ldpc_r13']), 'd--', color='#332288', lw=1.5,
            label='Quant-8+LDPC 1/3 (43.6 Mbit)')
    ax.axhline(0.9849, color='gray', ls=':', lw=1)
    ax.text(0.002, 0.995, 'released anchor', color='gray', fontsize=7)
    ax.axhline(0.9564, color='gray', ls='--', lw=0.8)
    ax.text(0.002, 0.923, 'fine-tuned anchor', color='gray', fontsize=7)
    ax.set_xlabel('Channel BER $p$')
    ax.set_ylabel('Vehicle detection mAP')
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc='center left', framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_array_crossover.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig2_array_crossover.png'),
                dpi=200, bbox_inches='tight')
    print('fig2 done')


def fig3():
    cl = json.load(open(os.path.join(SR, 'radial_closedloop_v2_so.json')))
    cl = {'trajectory': cl['per_seed']['42']['trajectory'], 'mission': cl['mission_mean']}
    doa_b = json.load(open(os.path.join(SR, 'doa_sim_baselines.json')))
    doa = {s: json.load(open(os.path.join(SR, f'doa_sim_snn_seed{s}.json')))
           for s in [42, 123, 456]}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))

    # (a) closed-loop trajectory
    ax = axes[0]
    t = cl['trajectory']
    d = [r['d_m'] / 1000 for r in t]
    ax.plot(d, [r['mAP_adaptive'] for r in t], '-', color='#c1272d', lw=2,
            label='closed loop $(\\rho,T\')$')
    ax.plot(d, [r['mAP_fixed'] for r in t], '--', color='#4477aa', lw=1.5,
            label="fixed $\\rho{=}1,T'{=}8$")
    ax2 = ax.twinx()
    ax2.plot(d, [r['payload_adaptive'] / 1e6 for r in t], ':',
             color='#c1272d', lw=1.2)
    ax2.plot(d, [r['payload_fixed'] / 1e6 for r in t], ':',
             color='#4477aa', lw=1.2)
    ax2.set_ylabel('Payload (Mbit, dotted)', fontsize=9)
    ax2.set_ylim(0, 2.6)
    m = cl['mission']
    ax.set_title(f"(a) A2G mission: mAP {m['mAP_adaptive']:.3f} vs "
                 f"{m['mAP_fixed']:.3f}, $-${m['payload_saving']*100:.0f}\\% bits",
                 fontsize=10)
    ax.set_xlabel('Ground distance (km), $h{=}100$ m suburban')
    ax.set_ylabel('mAP')
    ax.set_ylim(0.7, 1.0)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # (b) DoA
    ax = axes[1]
    qc = [doa_b['quantcov_rmse_vs_ber'][str(b)] for b in BERS]
    ax.semilogy(BERS, qc, '^--', color='#4477aa', lw=1.5,
                label='quant-cov + MUSIC (1536 b)')
    key = 'rho0.75_T8_ber{}'
    arr = np.array([[doa[s]['snn_grid'][key.format(b)]['rmse_deg']
                     for b in BERS] for s in doa])
    m_, sd = arr.mean(0), arr.std(0, ddof=1)
    ax.semilogy(BERS, m_, 'o-', color='#c1272d', lw=2,
                label='SNN codec (1536 b, ours)')
    ax.fill_between(BERS, m_ - sd, m_ + sd, color='#c1272d', alpha=0.2)
    ax.axhline(doa_b['rmse_crb'], color='gray', ls=':', lw=1)
    ax.text(0.002, doa_b['rmse_crb'] * 1.25, r'$\sqrt{\mathrm{CRB}}$',
            color='gray', fontsize=8)
    ax.set_title('(b) DoA, matched 1536-bit payload', fontsize=10)
    ax.set_xlabel('Channel BER $p$')
    ax.set_ylabel('RMSE (deg)')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_array_loop_doa.pdf'),
                dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUT, 'fig3_array_loop_doa.png'),
                dpi=200, bbox_inches='tight')
    print('fig3 done')


if __name__ == '__main__':
    fig2()
    fig3()
