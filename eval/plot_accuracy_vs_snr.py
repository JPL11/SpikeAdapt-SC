#!/usr/bin/env python3
"""Accuracy vs mean SNR and vs UAV ground distance (communications view).

Maps each method's cached accuracy-vs-BER sweep through the physical A2G
channel model (eval/channel_a2g.py):

  Panel (a): accuracy vs mean SNR for BPSK over Rician fading, K = 10 dB.
  Panel (b): accuracy vs ground distance for the h = 300 m suburban A2G
             link profile (P_t = 10 mW, elevation-dependent K factor).

Curves (post-panel revision): SpikeAdapt+T'aug (3-seed), base SpikeAdapt-SC
(10-seed), fair SNN-SC-style reimplementation (10-seed), Adaptive-1bit
(10-seed), continuous JSCC (10-seed), JPEG+LDPC 1/2 and 1/3 (analytic).
All learned curves share the validation-selected, test-scored-once protocol;
bands = seedwise std for the two SpikeAdapt curves.

Output: paper/figures/fig_acc_vs_snr_distance_{ds}.{pdf,png}
Usage:  python eval/plot_accuracy_vs_snr.py [aid|resisc45]
"""

import json
import sys

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from eval.channel_a2g import ber_bpsk_rician

DS = sys.argv[1] if len(sys.argv) > 1 else 'aid'
DS_LABEL = 'AID' if DS == 'aid' else 'RESISC45'
CHANCE = {'aid': 100 / 30, 'resisc45': 100 / 45}
BKEYS = ['0.0', '0.05', '0.1', '0.15', '0.2', '0.25', '0.3']


def _stats(rows):
    """rows: list over seeds of dict ber->acc  ->  (bers, mean, std)."""
    bers = np.array([float(b) for b in BKEYS])
    arr = np.array([[r[b] for b in BKEYS] for r in rows])
    return bers, arr.mean(0), arr.std(0)


def spikeadapt_10seed(ds, rho=0.75):
    with open('eval/seed_results/per_rho_10seed_valproto.json') as f:
        d = json.load(f)[ds]
    rows = []
    for s in sorted(d, key=int):
        rk = next(k for k in d[s] if float(k) == rho)
        rows.append({b: d[s][rk][next(k for k in d[s][rk]
                                      if float(k) == float(b))]
                     for b in BKEYS})
    return _stats(rows)


def taug_3seed(ds):
    with open(f'eval/seed_results/spikeadapt_taug_joint_grid_{ds}.json') as f:
        d = json.load(f)
    rows = [{b: d[s]['test']['0.75']['8'][b] for b in BKEYS}
            for s in ['42', '123', '456']]
    return _stats(rows)


def snnsc_10seed(ds):
    with open('eval/seed_results/valproto_snnsc.json') as f:
        d = json.load(f)[ds]
    rows = [{b: d[s][b] for b in BKEYS} for s in d]
    return _stats(rows)


def adaptive1bit_10seed(ds):
    with open('eval/seed_results/valproto_adaptive1bit.json') as f:
        d = json.load(f)[ds]
    rows = [{b: d[s]['test']['0.75'][b] for b in BKEYS} for s in d]
    return _stats(rows)


def jscc_10seed(ds):
    with open('eval/seed_results/valproto_jscc.json') as f:
        d = json.load(f)[ds]
    rows = [{b: d[s]['bsc'][b] for b in BKEYS} for s in d]
    return _stats(rows)


def jpeg_ldpc_sweeps(ds):
    with open('eval/seed_results/jpeg_ldpc_results.json') as f:
        d = json.load(f)[ds]
    out = {}
    for key, label in [('r12_n648_dv3dc6_q50', 'JPEG+LDPC (R=1/2)'),
                       ('r13_n648_dv4dc6_q50', 'JPEG+LDPC (R=1/3)')]:
        sw = d['sweeps'][key]
        items = sorted(sw.items(), key=lambda kv: float(kv[0]))
        out[label] = (np.array([float(k) for k, _ in items]),
                      np.array([v for _, v in items]), None)
    return out


def acc_at_ber(bers, accs, ber_query, chance):
    """Piecewise-linear interp; beyond last point, decay toward chance."""
    ber_query = np.asarray(ber_query, float)
    out = np.interp(ber_query, bers, accs)
    beyond = ber_query > bers[-1]
    if beyond.any():
        last_b, last_a = bers[-1], accs[-1]
        frac = np.clip((ber_query - last_b) / (0.5 - last_b), 0, 1)
        out = np.where(beyond, last_a + frac * (chance - last_a), out)
    return out


STYLE = {
    "SpikeAdapt$+T'$aug ($\\rho{=}0.75$, 3-seed)":
        dict(color='#0d47a1', ls='-', lw=2.6, z=7, band=True),
    'SpikeAdapt-SC ($\\rho{=}0.75$, 10-seed)':
        dict(color='#42a5f5', ls='-', lw=2.0, z=6, band=True),
    'SNN-SC-style reimpl. ($\\rho{=}1$, 10-seed)':
        dict(color='#7b1fa2', ls='--', lw=1.8, z=5, band=False),
    'Adaptive-1bit ($\\rho{=}0.75$, 10-seed)':
        dict(color='#ef6c00', ls='-', lw=1.8, z=5, band=False),
    'JSCC (continuous, 10-seed)':
        dict(color='#c62828', ls='-.', lw=1.8, z=4, band=False),
    'JPEG+LDPC (R=1/2)': dict(color='#37474f', ls=':', lw=2, z=4, band=False),
    'JPEG+LDPC (R=1/3)': dict(color='#8d6e63', ls=':', lw=2, z=4, band=False),
}


def main():
    chance = CHANCE[DS]
    methods = {
        "SpikeAdapt$+T'$aug ($\\rho{=}0.75$, 3-seed)": taug_3seed(DS),
        'SpikeAdapt-SC ($\\rho{=}0.75$, 10-seed)': spikeadapt_10seed(DS),
        'SNN-SC-style reimpl. ($\\rho{=}1$, 10-seed)': snnsc_10seed(DS),
        'Adaptive-1bit ($\\rho{=}0.75$, 10-seed)': adaptive1bit_10seed(DS),
        'JSCC (continuous, 10-seed)': jscc_10seed(DS),
    }
    methods.update(jpeg_ldpc_sweeps(DS))

    with open('eval/seed_results/a2g_channel_map.json') as f:
        a2g = json.load(f)
    prof = a2g['distance_profiles']['h300m_suburban']

    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 17,
                         'axes.titlesize': 15.5, 'legend.fontsize': 13,
                         'xtick.labelsize': 14, 'ytick.labelsize': 14})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ===== (a) accuracy vs mean SNR, Rician K=10 dB =====
    ax = axes[0]
    snr_grid = np.linspace(-12, 6, 200)
    ber_grid = np.array([ber_bpsk_rician(s, 10.0) for s in snr_grid])
    for name, (b, m, s) in methods.items():
        st = STYLE[name]
        acc = acc_at_ber(b, m, ber_grid, chance)
        ax.plot(snr_grid, acc, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=name, zorder=st['z'])
        if st['band'] and s is not None:
            std = np.interp(ber_grid, b, s)
            ax.fill_between(snr_grid, acc - std, acc + std,
                            color=st['color'], alpha=0.15, zorder=2)
    for ber_anchor, lbl in [(0.15, 'BER=0.15'), (0.30, 'BER=0.30')]:
        snr_a = a2g['ber_anchors_snr'][str(ber_anchor)]['K10dB']
        ax.axvline(snr_a, color='gray', ls=':', lw=1, alpha=0.6)
        ax.text(snr_a, 8, f'{lbl}\n({snr_a:+.1f} dB)', fontsize=10,
                ha='center', color='gray')
    ax.axhline(chance, color='gray', ls=':', lw=1, alpha=0.4)
    ax.set_xlabel('Mean SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'(a) {DS_LABEL}: BPSK over Rician ($K_{{\\mathrm{{dB}}}}$=10)',
                 fontweight='bold')
    ax.set_xlim(-12, 6)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # ===== (b) accuracy vs ground distance (h=300 m suburban) =====
    ax = axes[1]
    d_km = np.array([r['d_ground_m'] for r in prof]) / 1000
    ber_d = np.array([r['ber'] for r in prof])
    sel = d_km <= 3.0
    for name, (b, m, s) in methods.items():
        st = STYLE[name]
        acc = acc_at_ber(b, m, ber_d[sel], chance)
        ax.plot(d_km[sel], acc, color=st['color'], ls=st['ls'],
                lw=st['lw'], label=name, zorder=st['z'])
        if st['band'] and s is not None:
            std = np.interp(ber_d[sel], b, s)
            ax.fill_between(d_km[sel], acc - std, acc + std,
                            color=st['color'], alpha=0.15, zorder=2)
    ax.axhline(chance, color='gray', ls=':', lw=1, alpha=0.4)
    ax.set_xlabel('UAV ground distance (km)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'(b) {DS_LABEL}: A2G, h=300 m suburban, '
                 '$P_t$=10 mW, $K_{{\\mathrm{{dB}}}}(\\theta)$', fontweight='bold')
    ax.set_xlim(0.4, 3.0)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = f'paper/figures/fig_acc_vs_snr_distance_{DS}'
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=12.5,
               framealpha=0.95, bbox_to_anchor=(0.5, -0.18))
    fig.savefig(out + '.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out + '.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}.{{pdf,png}}')


if __name__ == '__main__':
    main()
