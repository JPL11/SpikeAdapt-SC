#!/usr/bin/env python3
"""Plot AID alpha with extended BER data (up to 0.4)."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('eval/extended_ber_rho_partial.json') as f:
    data = json.load(f)

aid = data['aid']
rho_vals = sorted([float(r) for r in aid.keys()])
ber_vals = sorted([float(b) for b in aid[list(aid.keys())[0]].keys()])

# A(rho=1) for each BER
a_full = {}
for ber in ber_vals:
    a_full[ber] = aid['1.000'][str(ber)]

# ====== Plot: Accuracy and Alpha side by side ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
cmap = plt.cm.RdYlBu_r
norm = plt.Normalize(0, 0.4)

# Left: Accuracy vs rho
ax = axes[0]
for ber in ber_vals:
    accs = [aid[f'{r:.3f}'][str(ber)] for r in rho_vals]
    color = cmap(norm(ber))
    label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
    ax.plot(rho_vals, accs, 'o-', color=color, linewidth=2, markersize=5, label=label)
ax.set_xlabel('Transmission rate (rho)', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('AID: Accuracy vs rho at different BER', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='lower right')
ax.set_xlim(0.05, 1.05)
ax.grid(True, alpha=0.3)

# Right: Alpha vs rho
ax = axes[1]
for ber in ber_vals:
    alphas = [(aid[f'{r:.3f}'][str(ber)] - a_full[ber]) / a_full[ber]
              if a_full[ber] > 0 else 0 for r in rho_vals]
    color = cmap(norm(ber))
    label = f'BER={ber:.2f}' if ber > 0 else 'Clean'
    ax.plot(rho_vals, alphas, 'o-', color=color, linewidth=2, markersize=5, label=label)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax.set_xlabel('Transmission rate (rho)', fontsize=12)
ax.set_ylabel('alpha = (A(rho) - A(1)) / A(1)', fontsize=13)
ax.set_title('AID: Relative accuracy change (alpha)', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, ncol=2, loc='lower right')
ax.set_xlim(0.05, 1.05)
ax.grid(True, alpha=0.3)

# Shade masking-helps region
for ber in [0.35, 0.3]:
    pos_rhos = [r for r in rho_vals if r < 1.0 and
                (aid[f'{r:.3f}'][str(ber)] - a_full[ber]) / a_full[ber] > 0.001]
    if pos_rhos:
        ax.axvspan(min(pos_rhos)-0.03, max(pos_rhos)+0.03, alpha=0.06, color='green')

plt.tight_layout()
os.makedirs('eval/figures', exist_ok=True)
fig.savefig('eval/figures/alpha_aid_extended.pdf', dpi=300, bbox_inches='tight')
fig.savefig('eval/figures/alpha_aid_extended.png', dpi=150, bbox_inches='tight')
plt.close()

# Print alpha table
print('AID alpha table (BER up to 0.4):')
header = f"{'rho':>8}" + ''.join(f'  BER={b:.2f}' for b in ber_vals)
print(header)
print('-' * len(header))
for rho in rho_vals:
    row = f'{rho:>8.3f}'
    for ber in ber_vals:
        acc = aid[f'{rho:.3f}'][str(ber)]
        alpha = (acc - a_full[ber]) / a_full[ber] if a_full[ber] > 0 else 0
        marker = '+' if alpha > 0.001 else ' '
        row += f'  {marker}{alpha:>7.4f}'
    print(row)

print()
print('Key observations at BER=0.35:')
for rho in rho_vals:
    acc = aid[f'{rho:.3f}']['0.35']
    alpha = (acc - a_full[0.35]) / a_full[0.35]
    print(f'  rho={rho:.3f}: acc={acc:.2f}%, alpha={alpha:+.4f}')

print(f'\nBER=0.40:')
for rho in rho_vals:
    acc = aid[f'{rho:.3f}']['0.4']
    alpha = (acc - a_full[0.4]) / a_full[0.4]
    print(f'  rho={rho:.3f}: acc={acc:.2f}%, alpha={alpha:+.4f}')

print(f'\nSaved: eval/figures/alpha_aid_extended.pdf')
