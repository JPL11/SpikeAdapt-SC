#!/usr/bin/env python3
"""Air-to-ground (A2G) UAV link model: geometry -> SNR/K-factor -> BER.

Maps the abstract BSC BER used in the paper to physical link parameters:

  1. Geometry: UAV altitude h, ground distance d_g -> 3D distance, elevation.
  2. LoS probability: Al-Hourani et al. (2014) sigmoid model,
       P_LoS(theta) = 1 / (1 + a*exp(-b*(theta - a))).
  3. Path loss: FSPL + LoS/NLoS excess loss (probabilistic mean).
  4. Rician K-factor: elevation-dependent, K_dB linear in elevation
     (low-elevation multipath-rich ~5 dB -> near-zenith LoS ~15 dB,
      consistent with 3GPP TR 36.777 A2G assumptions).
  5. BER for BPSK with hard decision over Rician fading (perfect CSI):
       P_b = (1/pi) * int_0^{pi/2} M(-1/sin^2 phi) dphi   (MGF method)
     with M the MGF of the instantaneous SNR under Rician fading.
     Cross-checked by Monte Carlo (matches the torch Rician_Channel).

Outputs (when run as a script):
    eval/seed_results/a2g_channel_map.json
      - snr_to_ber tables for several K factors
      - distance profile: d_g -> (elevation, P_LoS, path loss, SNR, K, BER)

Usage:
    python eval/channel_a2g.py
"""

import os, json, math
import numpy as np

# ---------------------------------------------------------------- constants
# Al-Hourani environment parameters (a, b) and excess losses (dB)
ENVIRONMENTS = {
    'suburban': dict(a=4.88, b=0.43, eta_los=0.1, eta_nlos=21.0),
    'urban':    dict(a=9.61, b=0.16, eta_los=1.0, eta_nlos=20.0),
}

# Default link budget (small UAV, ISM band)
LINK = dict(
    fc_ghz=2.4,        # carrier frequency
    ptx_dbm=10.0,      # transmit power (10 mW, SWaP-constrained UAV)
    gain_db=5.0,       # combined antenna gains
    bw_mhz=10.0,       # signal bandwidth
    nf_db=7.0,         # receiver noise figure
)

# Elevation-dependent Rician K factor (dB): K(0 deg)=5, K(90 deg)=15
K_MIN_DB, K_MAX_DB = 5.0, 15.0


# ---------------------------------------------------------------- geometry
def elevation_deg(h_m, d_ground_m):
    return math.degrees(math.atan2(h_m, max(d_ground_m, 1e-9)))


def distance_3d_m(h_m, d_ground_m):
    return math.hypot(h_m, d_ground_m)


def p_los(elev_deg_, env='suburban'):
    p = ENVIRONMENTS[env]
    return 1.0 / (1.0 + p['a'] * math.exp(-p['b'] * (elev_deg_ - p['a'])))


def fspl_db(d_m, fc_ghz):
    return 20 * math.log10(max(d_m, 1.0)) + 20 * math.log10(fc_ghz * 1e9) - 147.55


def path_loss_db(h_m, d_ground_m, env='suburban', fc_ghz=LINK['fc_ghz']):
    """Probabilistic mean path loss: FSPL + P_LoS-weighted excess."""
    th = elevation_deg(h_m, d_ground_m)
    d3 = distance_3d_m(h_m, d_ground_m)
    pl = p_los(th, env)
    e = ENVIRONMENTS[env]
    excess = pl * e['eta_los'] + (1 - pl) * e['eta_nlos']
    return fspl_db(d3, fc_ghz) + excess


def k_factor_db(elev_deg_):
    return K_MIN_DB + (K_MAX_DB - K_MIN_DB) * min(max(elev_deg_, 0), 90) / 90.0


def snr_db(h_m, d_ground_m, env='suburban', link=LINK):
    noise_dbm = -174.0 + 10 * math.log10(link['bw_mhz'] * 1e6) + link['nf_db']
    prx_dbm = (link['ptx_dbm'] + link['gain_db']
               - path_loss_db(h_m, d_ground_m, env, link['fc_ghz']))
    return prx_dbm - noise_dbm


# ---------------------------------------------------------------- BER models
def ber_bpsk_rician(snr_db_, k_db, n_phi=512):
    """Exact BER of coherent BPSK over Rician fading via the MGF method.

    P_b = (1/pi) * int_0^{pi/2} [(1+K)s / ((1+K)s + g)] *
                       exp(-K*g / ((1+K)s + g)) dphi,   s = sin^2(phi)
    where g is the mean SNR (linear) and K the Rician factor (linear).
    K -> 0 reduces to Rayleigh; K -> inf reduces to AWGN Q(sqrt(2g)).
    """
    g = 10 ** (np.asarray(snr_db_, dtype=float) / 10.0)
    K = 10 ** (k_db / 10.0)
    phi = (np.arange(n_phi) + 0.5) * (math.pi / 2) / n_phi
    s = np.sin(phi) ** 2                                  # (n_phi,)
    g_ = np.atleast_1d(g)[:, None]                        # (N,1)
    denom = (1 + K) * s + g_
    integrand = ((1 + K) * s / denom) * np.exp(-K * g_ / denom)
    pb = integrand.mean(axis=1) * 0.5  # (1/pi)*(pi/2)*mean = mean/2
    return pb if np.ndim(snr_db_) else float(pb[0])


def ber_bpsk_rician_mc(snr_db_, k_db, n=2_000_000, seed=0):
    """Monte Carlo cross-check matching models/snn_modules.py Rician_Channel."""
    rng = np.random.default_rng(seed)
    g = 10 ** (snr_db_ / 10.0)
    K = 10 ** (k_db / 10.0)
    los = math.sqrt(K / (K + 1.0))
    scale = math.sqrt(1.0 / (2.0 * (K + 1.0)))
    h = np.abs(los + scale * rng.standard_normal(n)
               + 1j * scale * rng.standard_normal(n))
    noise = rng.standard_normal(n) / math.sqrt(2 * g)
    # transmit +1 (BPSK), coherent detection with perfect CSI
    received = h * 1.0 + noise
    return float((received / h < 0).mean())


def snr_for_ber(target_ber, k_db, lo=-30.0, hi=40.0):
    """Invert ber_bpsk_rician(snr) = target_ber by bisection."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ber_bpsk_rician(mid, k_db) > target_ber:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- profiles
def distance_profile(h_m=100.0, env='suburban', d_grid=None, link=LINK):
    """Full link profile vs ground distance."""
    if d_grid is None:
        d_grid = np.concatenate([np.arange(100, 2000, 100),
                                 np.arange(2000, 30001, 250)])
    rows = []
    for d in d_grid:
        th = elevation_deg(h_m, d)
        kdb = k_factor_db(th)
        s = snr_db(h_m, d, env, link)
        rows.append(dict(
            d_ground_m=float(d),
            elevation_deg=round(th, 3),
            p_los=round(p_los(th, env), 4),
            path_loss_db=round(path_loss_db(h_m, d, env, link['fc_ghz']), 2),
            snr_db=round(s, 2),
            k_factor_db=round(kdb, 2),
            ber=float(ber_bpsk_rician(s, kdb)),
        ))
    return rows


def main():
    os.makedirs('eval/seed_results', exist_ok=True)

    # --- validation: MGF vs Monte Carlo
    print('MGF vs Monte Carlo validation (K=10 dB):')
    for s in [-10, -5, 0, 5, 10]:
        a = ber_bpsk_rician(s, 10.0)
        m = ber_bpsk_rician_mc(s, 10.0)
        print(f'  SNR={s:+3d} dB: analytic={a:.5f}  MC={m:.5f}  '
              f'diff={abs(a-m):.5f}')

    # --- SNR -> BER tables
    snr_grid = np.arange(-25, 25.01, 0.25)
    snr_to_ber = {}
    for kdb in [0.0, 5.0, 10.0, 15.0]:
        snr_to_ber[f'K{kdb:.0f}dB'] = {
            'snr_db': snr_grid.round(2).tolist(),
            'ber': ber_bpsk_rician(snr_grid, kdb).tolist(),
        }
    # Rayleigh reference (K -> 0) and AWGN reference (K -> inf approx)
    snr_to_ber['rayleigh'] = snr_to_ber['K0dB']
    from math import erfc, sqrt
    snr_to_ber['awgn'] = {
        'snr_db': snr_grid.round(2).tolist(),
        'ber': [0.5 * erfc(sqrt(10 ** (s / 10.0))) for s in snr_grid],
    }

    # --- BER anchors used in the paper -> required SNR
    print('\nPaper BER anchors -> required mean SNR (BPSK, Rician):')
    anchors = {}
    for ber in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        row = {}
        for kdb in [5.0, 10.0, 15.0]:
            row[f'K{kdb:.0f}dB'] = round(snr_for_ber(ber, kdb), 2)
        anchors[str(ber)] = row
        print(f'  BER={ber:.2f}: ' + '  '.join(
            f'K={k[1:]}: {v:+.1f} dB' for k, v in row.items()))

    # --- distance profiles at two altitudes
    profiles = {}
    for h in [100.0, 300.0]:
        for env in ['suburban', 'urban']:
            profiles[f'h{h:.0f}m_{env}'] = distance_profile(h, env)

    out = dict(link_budget=LINK, environments=ENVIRONMENTS,
               k_factor_model=dict(k_min_db=K_MIN_DB, k_max_db=K_MAX_DB),
               snr_to_ber=snr_to_ber, ber_anchors_snr=anchors,
               distance_profiles=profiles)
    path = 'eval/seed_results/a2g_channel_map.json'
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'\nSaved: {path}')

    # quick sanity rows
    print('\nDistance profile h=100 m, suburban:')
    for row in profiles['h100m_suburban']:
        if row['d_ground_m'] in (500, 1000, 2000, 5000, 10000, 20000, 30000):
            print(f"  d={row['d_ground_m']/1000:5.1f} km  elev={row['elevation_deg']:5.1f}  "
                  f"PLoS={row['p_los']:.2f}  SNR={row['snr_db']:+6.1f} dB  "
                  f"K={row['k_factor_db']:4.1f} dB  BER={row['ber']:.4f}")


if __name__ == '__main__':
    main()
