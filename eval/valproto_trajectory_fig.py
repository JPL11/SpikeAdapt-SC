#!/usr/bin/env python3
"""Regenerate fig_trajectory_closedloop from the VAL-protocol policy
(RESISC45 seed 42; panels 1-2 protocol-independent; panel 3 payload traces
from valproto val-surface tables under the 0.75 budget)."""
import sys, os
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '.'); sys.path.insert(0, 'eval')
from eval.valproto_missions import load_grids, acc_at, T_FULL, BUDGET, EPS_PP
from eval.compare_joint_baseline import load_grid
from eval.eval_trajectory_closedloop import mission_profile, N_PILOT

d, snr, kdb, ber = mission_profile()
T = load_grids('resisc45')['42']['val']
Taug = load_grid('eval/seed_results/spikeadapt_taug_joint_grid_resisc45.json')['42']['val']
rng = np.random.default_rng(5)
est = rng.binomial(N_PILOT, np.clip(ber, 0, 1)) / N_PILOT

def pol(table, p, rho_only=False):
    p = float(np.clip(p, 0, 0.30))
    ks = [k for k in table if k[0]*k[1]/T_FULL <= BUDGET + 1e-9
          and (not rho_only or k[1] == T_FULL)]
    accs = {k: acc_at(table, k, p) for k in ks}
    best = max(accs.values())
    return min((k[0]*k[1]/T_FULL, k) for k, v in accs.items()
               if v >= best - EPS_PP)[0]

joint = [pol(T, p) for p in est]
joint_aug = [pol(Taug, p) for p in est]
rho_only = [pol(T, p, rho_only=True) for p in est]
t = np.arange(len(d))
plt.rcParams.update({'font.size': 11.5, 'axes.labelsize': 12.5,
                     'legend.fontsize': 10, 'xtick.labelsize': 10.5,
                     'ytick.labelsize': 10.5})
fig, axes = plt.subplots(3, 1, figsize=(6.4, 4.9), sharex=True,
                         gridspec_kw={'height_ratios': [1, 1, 1.25]})
ax = axes[0]
ax.plot(t, np.array(d)/1000, color='#455a64', lw=2)
ax.set_ylabel('Ground dist. (km)')
ax2 = ax.twinx(); ax2.plot(t, snr, color='#1565c0', lw=1.6, alpha=0.8)
ax2.set_ylabel('SNR (dB)', color='#1565c0'); ax2.tick_params(axis='y', colors='#1565c0')
ax.set_title('UAV out-and-back mission, h=300 m suburban A2G, $P_t$=10 mW',
             fontsize=12, fontweight='bold')
ax = axes[1]
ax.plot(t, ber, color='black', lw=2, label='true BER')
ax.plot(t, est, color='#c62828', lw=1.0, ls='--', label=f'pilot estimate (n={N_PILOT})')
ax.axhline(0.15, color='0.6', ls=':', lw=0.9)
ax.axhline(0.30, color='0.6', ls=':', lw=0.9)
ax.text(1, 0.155, 'BER 0.15', fontsize=7.5, color='0.4', va='bottom')
ax.text(1, 0.305, 'BER 0.30', fontsize=7.5, color='0.4', va='bottom')
ax.set_ylabel('BER'); ax.legend(fontsize=9.5, loc='upper center')
ax = axes[2]
ax.step(t, rho_only, where='mid', color='#1b5e20', lw=1.4, alpha=0.6,
        label=r'$\rho$-only loop (base)')
ax.step(t, joint, where='mid', color='#8d6e63', lw=1.5, ls='--',
        label='joint loop (base)')
ax.step(t, joint_aug, where='mid', color='#e65100', lw=2.4,
        label=r"joint $+T'$aug (flagship)")
ax.axhline(0.75, color='#9e9e9e', ls=':', lw=1.2, label='semantic-payload cap (0.75)')
ax.set_ylabel(r"Payload $T'\lfloor\rho HW\rfloor/(THW)$"); ax.set_xlabel('Mission time step')
ax.set_ylim(0, 1.14); ax.legend(fontsize=8.6, ncol=2, loc='upper center',
          framealpha=0.95)
for a in axes: a.grid(True, alpha=0.3)
plt.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'paper/figures/fig_trajectory_closedloop.{ext}',
                dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
print('base joint range:', round(min(joint),3), '-', round(max(joint),3),
      '| aug range:', round(min(joint_aug),3), '-', round(max(joint_aug),3))
