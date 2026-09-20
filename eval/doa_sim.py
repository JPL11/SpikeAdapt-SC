#!/usr/bin/env python3
"""Task-oriented DoA estimation over a noisy binary channel — simulation harness.

Scenario (array/ISAC follow-up paper): a UAV-mounted ULA observes K narrowband
ground emitters and must deliver their directions-of-arrival to a ground
station over a rate-limited A2G link (BSC abstraction; physical grounding via
eval/channel_a2g.py). Three transport schemes are compared at matched payload:

  1. oracle       — full-precision snapshots at the receiver (no channel):
                    MUSIC on the sample covariance. Upper anchor.
  2. quant-cov    — "separation" baseline: quantize the sample covariance,
                    transmit its bits over BSC(p), MUSIC at the receiver.
  3. snn          — task-oriented spiking encoder: snapshots -> binary spike
                    latent (T timesteps x L bits, group-maskable for spatial
                    rate rho) -> BSC(p) -> MLP decoder -> DoA estimates.
                    Trained with a BER curriculum (same recipe as the
                    classification/detection codecs, cap 0.40).

The stochastic CRB (Stoica & Nehorai 1990) is reported as the fundamental
bound for scheme 1's observation model.

Stages:
    python eval/doa_sim.py baselines            # MUSIC vs CRB vs quant-cov BER sweep
    python eval/doa_sim.py train  [--quick]     # train the SNN codec
    python eval/doa_sim.py eval   [--quick]     # (rho, T', BER) sweep of trained codec
Outputs: eval/seed_results/doa_sim_{baselines,snn}.json
Run from the repo root with the semcom conda python.
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.snn_modules import BSC_Channel, LIFNeuron  # noqa: E402

# ---------------------------------------------------------------- scene model
N_ANT = 16          # ULA elements, d = lambda/2
N_SNAP = 64         # snapshots per observation window
K_SRC = 2           # number of sources (fixed in this draft)
FOV_DEG = 60.0      # DoAs drawn from [-FOV, FOV]
MIN_SEP_DEG = 8.0   # minimum angular separation between sources
SNR_DB = 10.0       # per-source array SNR (element level)


def steering(theta_deg, n_ant=N_ANT):
    """ULA steering vectors, d=lambda/2. theta_deg: (...,K) -> (...,N,K)."""
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    n = np.arange(n_ant)[:, None]
    return np.exp(1j * math.pi * n * np.sin(theta)[..., None, :])


def draw_scene(rng, k=K_SRC):
    """Random DoAs with minimum separation, sorted ascending."""
    while True:
        th = np.sort(rng.uniform(-FOV_DEG, FOV_DEG, size=k))
        if k == 1 or np.min(np.diff(th)) >= MIN_SEP_DEG:
            return th


def sim_snapshots(theta_deg, rng, m=N_SNAP, snr_db=SNR_DB):
    """X = A S + W: (N, M) complex. Unconditional model, S ~ CN(0, sigma_s^2 I)."""
    a = steering(theta_deg)                                     # (N, K)
    sigma_s = math.sqrt(10 ** (snr_db / 10.0))
    s = sigma_s * (rng.standard_normal((len(theta_deg), m))
                   + 1j * rng.standard_normal((len(theta_deg), m))) / math.sqrt(2)
    w = (rng.standard_normal((N_ANT, m))
         + 1j * rng.standard_normal((N_ANT, m))) / math.sqrt(2)
    return a @ s + w


# ---------------------------------------------------------------- MUSIC + CRB
def music_spectrum(r_hat, k=K_SRC, grid=None):
    if grid is None:
        grid = np.arange(-FOV_DEG - 5, FOV_DEG + 5.001, 0.1)
    w, v = np.linalg.eigh(r_hat)                                # ascending
    en = v[:, : N_ANT - k]                                      # noise subspace
    a = steering(grid)                                          # (N, G)
    denom = np.sum(np.abs(en.conj().T @ a) ** 2, axis=0)
    return grid, 1.0 / np.maximum(denom, 1e-12)


def music_doa(r_hat, k=K_SRC):
    """Top-k local maxima of the MUSIC spectrum (greedy NMS fallback)."""
    grid, p = music_spectrum(r_hat, k)
    peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]))[0] + 1
    if len(peaks) >= k:
        sel = peaks[np.argsort(p[peaks])[-k:]]
    else:  # degenerate spectrum: greedy pick with 2-deg exclusion
        sel, order = [], np.argsort(p)[::-1]
        for i in order:
            if all(abs(grid[i] - grid[j]) > 2.0 for j in sel):
                sel.append(i)
            if len(sel) == k:
                break
        sel = np.array(sel)
    return np.sort(grid[sel])


def crb_stochastic(theta_deg, m=N_SNAP, snr_db=SNR_DB):
    """Stochastic (unconditional) CRB for ULA DoAs, deg^2 (Stoica-Nehorai 1990).

    CRB = (sigma^2 / 2M) * { Re[ (D^H Pi_A^perp D) o (P A^H R^-1 A P)^T ] }^-1
    with P = sigma_s^2 I, sigma^2 = 1, o = Hadamard product.
    """
    theta = np.deg2rad(np.asarray(theta_deg, dtype=np.float64))
    a = steering(theta_deg)                                     # (N, K)
    n = np.arange(N_ANT)[:, None]
    d = 1j * math.pi * n * np.cos(theta)[None, :] * a           # d a / d theta
    sigma_s2 = 10 ** (snr_db / 10.0)
    p = sigma_s2 * np.eye(len(theta))
    r = a @ p @ a.conj().T + np.eye(N_ANT)
    r_inv = np.linalg.inv(r)
    pi_perp = np.eye(N_ANT) - a @ np.linalg.pinv(a)
    h = (d.conj().T @ pi_perp @ d) * (p @ a.conj().T @ r_inv @ a @ p).T
    crb_rad2 = np.linalg.inv(np.real(h)) / (2 * m)
    return np.diag(crb_rad2) * (180.0 / math.pi) ** 2           # deg^2


# ------------------------------------------------- quantized-cov "separation"
def quantize_cov_bits(r_hat, bits=6):
    """Trace-normalized covariance -> uniform-quantized bit tensor.

    Sends the diagonal (real) and upper-triangle (complex) entries; the clip
    range [-1, 1] covers the normalized entries at these SNRs.
    Returns (bit array {0,1}, metadata for reconstruction, payload_bits).
    """
    scale = np.trace(r_hat).real / N_ANT
    rn = r_hat / max(scale, 1e-12)
    iu = np.triu_indices(N_ANT, 1)
    vals = np.concatenate([np.diag(rn).real, rn[iu].real, rn[iu].imag])
    levels = 2 ** bits
    q = np.clip((vals + 1) / 2, 0, 1)
    idx = np.minimum((q * levels).astype(int), levels - 1)
    bit_arr = ((idx[:, None] >> np.arange(bits)[None, :]) & 1).astype(np.float32)
    return bit_arr.ravel(), (scale, len(vals)), bit_arr.size


def dequantize_cov(bit_flat, meta, bits=6):
    scale, n_vals = meta
    idx = (bit_flat.reshape(n_vals, bits).astype(int)
           * (1 << np.arange(bits))[None, :]).sum(1)
    vals = (idx + 0.5) / (2 ** bits) * 2 - 1
    rn = np.zeros((N_ANT, N_ANT), dtype=complex)
    np.fill_diagonal(rn, vals[:N_ANT])
    iu = np.triu_indices(N_ANT, 1)
    n_off = len(iu[0])
    rn[iu] = vals[N_ANT:N_ANT + n_off] + 1j * vals[N_ANT + n_off:]
    rn = rn + np.triu(rn, 1).conj().T
    return rn * scale


def rmse_deg(est, true):
    return float(np.sqrt(np.mean((np.sort(est) - np.sort(true)) ** 2)))


def run_baselines(args):
    rng = np.random.default_rng(args.seed)
    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    n_trials = 200 if args.quick else 2000
    res = {"config": dict(n_ant=N_ANT, n_snap=N_SNAP, k=K_SRC, snr_db=SNR_DB,
                          fov=FOV_DEG, min_sep=MIN_SEP_DEG, bits=args.bits,
                          n_trials=n_trials, seed=args.seed)}
    err_oracle, err_crb = [], []
    err_q = {b: [] for b in bers}
    payload = None
    for _ in range(n_trials):
        th = draw_scene(rng)
        x = sim_snapshots(th, rng)
        r_hat = x @ x.conj().T / N_SNAP
        err_oracle.append(rmse_deg(music_doa(r_hat), th) ** 2)
        err_crb.append(np.mean(crb_stochastic(th)))
        bit_flat, meta, payload = quantize_cov_bits(r_hat, args.bits)
        for p in bers:
            noisy = bit_flat if p == 0 else np.where(
                rng.random(bit_flat.shape) < p, 1 - bit_flat, bit_flat)
            r_rec = dequantize_cov(noisy, meta, args.bits)
            err_q[p].append(rmse_deg(music_doa(r_rec), th) ** 2)
    res["payload_bits_quantcov"] = int(payload)
    res["rmse_oracle_music"] = float(np.sqrt(np.mean(err_oracle)))
    res["rmse_crb"] = float(np.sqrt(np.mean(err_crb)))
    res["quantcov_rmse_vs_ber"] = {str(p): float(np.sqrt(np.mean(v)))
                                   for p, v in err_q.items()}
    out = "eval/seed_results/doa_sim_baselines.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"oracle MUSIC RMSE = {res['rmse_oracle_music']:.3f} deg | "
          f"sqrt(CRB) = {res['rmse_crb']:.3f} deg | "
          f"quant-cov payload = {payload} bits")
    for p in bers:
        print(f"  quant-cov BER={p:.2f}: RMSE = "
              f"{res['quantcov_rmse_vs_ber'][str(p)]:.3f} deg")
    print(f"Saved: {out}")


# ---------------------------------------------------------------- SNN codec
class DoAEncoderSNN(nn.Module):
    """Spiking encoder: (B, 2, N, M) snapshots -> (B, T, L) binary spikes.

    The latent is organized in n_groups groups along L; a spatial rate rho
    masks the trailing groups (transmit ceil(rho * n_groups) of them), and
    temporal truncation T' uses only the first T' of T timesteps — the same
    two knobs as the image codec.
    """

    def __init__(self, latent_bits=256, timesteps=8, n_groups=8):
        super().__init__()
        self.T, self.L, self.G = timesteps, latent_bits, n_groups
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=(1, 2), padding=1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=(2, 2), padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=(2, 2), padding=1), nn.BatchNorm2d(64), nn.SiLU(),
        )
        with torch.no_grad():
            feat = self.conv(torch.zeros(1, 2, N_ANT, N_SNAP)).flatten(1).shape[1]
        self.proj = nn.Linear(feat, latent_bits)
        self.lif = LIFNeuron(C=latent_bits)

    def forward(self, x, rho=1.0, t_prime=None):
        t_use = self.T if t_prime is None else t_prime
        h = self.proj(self.conv(x).flatten(1))                  # (B, L)
        h4 = h.unsqueeze(-1).unsqueeze(-1)                      # (B, L, 1, 1) for LIF
        mem, spikes = None, []
        for _ in range(t_use):
            sp, mem = self.lif(h4, mem)
            spikes.append(sp.squeeze(-1).squeeze(-1))
        z = torch.stack(spikes, dim=1)                          # (B, T', L)
        n_active = max(1, math.ceil(rho * self.G))
        gsize = self.L // self.G
        mask = torch.zeros(1, 1, self.L, device=z.device)
        mask[..., : n_active * gsize] = 1.0
        return z * mask, mask


class DoADecoder(nn.Module):
    """(B, T, L) noisy spikes -> (B, K) angles in degrees (sorted by training)."""

    def __init__(self, latent_bits=256, timesteps=8, k=K_SRC):
        super().__init__()
        self.T, self.L = timesteps, latent_bits
        self.net = nn.Sequential(
            nn.Linear(latent_bits * timesteps, 512), nn.SiLU(),
            nn.Linear(512, 256), nn.SiLU(),
            nn.Linear(256, k),
        )

    def forward(self, z, t_prime=None):
        if t_prime is not None and t_prime < self.T:            # zero-pad truncated tail
            z = torch.cat([z, z.new_zeros(z.shape[0], self.T - t_prime, self.L)], 1)
        return self.net(z.flatten(1)) * FOV_DEG                 # scale to degrees


def torch_scenes(batch, rng, device):
    th = np.stack([draw_scene(rng) for _ in range(batch)])
    x = np.stack([sim_snapshots(t, rng) for t in th])
    xt = torch.from_numpy(
        np.stack([x.real, x.imag], 1).astype(np.float32)).to(device)
    return xt, torch.from_numpy(th.astype(np.float32)).to(device)


def run_train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    enc = DoAEncoderSNN(args.latent_bits, args.timesteps).to(device)
    dec = DoADecoder(args.latent_bits, args.timesteps).to(device)
    chan = BSC_Channel()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    iters = 200 if args.quick else args.iters
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    rhos = [0.5, 0.75, 1.0]
    enc.train(), dec.train()
    for it in range(iters):
        x, th = torch_scenes(args.batch, rng, device)
        # BER curriculum to 0.40 (image-codec recipe), random rho and T'
        ber = min(0.40, 0.40 * it / max(1, int(iters * 0.6))) * float(torch.rand(1))
        rho = rhos[int(torch.randint(len(rhos), (1,)))]
        tp = int(torch.randint(args.timesteps // 2, args.timesteps + 1, (1,)))
        z, mask = enc(x, rho=rho, t_prime=tp)
        z_noisy = chan(z, ber) * mask                           # receiver knows the mask
        pred = dec(z_noisy, t_prime=tp)
        loss = nn.functional.mse_loss(pred, th)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % max(1, iters // 10) == 0:
            print(f"iter {it:5d}  loss {loss.item():8.3f}  ber {ber:.3f} "
                  f"rho {rho:.2f} T' {tp}", flush=True)
    os.makedirs("snapshots_doa", exist_ok=True)
    ckpt = f"snapshots_doa/doa_snn_seed{args.seed}.pth"
    torch.save({"enc": enc.state_dict(), "dec": dec.state_dict(),
                "args": vars(args)}, ckpt)
    print(f"Saved: {ckpt}")


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"snapshots_doa/doa_snn_seed{args.seed}.pth",
                      map_location=device, weights_only=False)
    ta = ckpt["args"]
    enc = DoAEncoderSNN(ta["latent_bits"], ta["timesteps"]).to(device)
    dec = DoADecoder(ta["latent_bits"], ta["timesteps"]).to(device)
    enc.load_state_dict(ckpt["enc"]), dec.load_state_dict(ckpt["dec"])
    enc.eval(), dec.eval()
    chan = BSC_Channel()
    rng = np.random.default_rng(args.seed + 1)
    bers = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    n_batches = 5 if args.quick else 50
    res = {"config": {**ta, "eval_batches": n_batches, "batch": args.batch}}
    grid = {}
    with torch.no_grad():
        for rho in [0.5, 0.75, 1.0]:
            for tp in range(ta["timesteps"] // 2, ta["timesteps"] + 1):
                for ber in bers:
                    errs = []
                    for _ in range(n_batches):
                        x, th = torch_scenes(args.batch, rng, device)
                        z, mask = enc(x, rho=rho, t_prime=tp)
                        pred = dec(chan(z, ber) * mask, t_prime=tp)
                        errs.append(((pred - th) ** 2).mean().item())
                    payload = int(rho * ta["latent_bits"]) * tp
                    grid[f"rho{rho}_T{tp}_ber{ber}"] = dict(
                        rmse_deg=float(math.sqrt(np.mean(errs))),
                        payload_bits=payload)
    res["snn_grid"] = grid
    out = "eval/seed_results/doa_sim_snn.json"
    json.dump(res, open(out, "w"), indent=1)
    for key in [f"rho1.0_T{ta['timesteps']}_ber{b}" for b in bers]:
        g = grid[key]
        print(f"{key}: RMSE {g['rmse_deg']:.3f} deg @ {g['payload_bits']} bits")
    print(f"Saved: {out}")


# ------------------------------------------------ Bayesian bound (Ziv-Zakai)
# Reviewer fix: the stochastic CRB is a LOCAL bound and does not predict the
# low-SNR threshold region where MUSIC breaks down. We add (i) empirical bias
# and (ii) the extended Ziv-Zakai bound (ZZB; Bell, Steinberg, Ephraim & Van
# Trees, IEEE T-IT 1997), which captures the threshold via a binary-detection
# argument and correctly saturates to the prior variance at low SNR.
# Single-source (K=1) so RMSE/CRB/ZZB share one consistent scenario.

def _snapshot_cov(theta_deg, snr_db):
    """Per-snapshot array covariance R = SNR*a a^H + I (single source, the
    STOCHASTIC model used throughout this paper)."""
    a = steering([theta_deg])                                   # (N,1)
    snr = 10 ** (snr_db / 10.0)
    return snr * (a @ a.conj().T) + np.eye(N_ANT)


def _pmin_pair(th0, th1, snr_db, m):
    """Minimum error probability of the equiprobable binary test H0:theta=th0
    vs H1:theta=th1 for the STOCHASTIC (zero-mean Gaussian) snapshot model.
    `b` is the ONE-SNAPSHOT (per-snapshot) Bhattacharyya distance B_1 between
    the two per-snapshot covariances; the M-snapshot coefficient is
    rho_B = exp(-M * B_1). We then apply the Kailath (1967) LOWER bound on the
    Bayes error, Pe >= 0.5*(1 - sqrt(1 - rho_B^2)). Using a valid LOWER bound
    on Pmin keeps the ZZB a valid (Bhattacharyya-relaxed) lower bound on MSE,
    while B_1 carries the correct (weaker) stochastic per-snapshot gain."""
    r0, r1 = _snapshot_cov(th0, snr_db), _snapshot_cov(th1, snr_db)
    rbar = 0.5 * (r0 + r1)
    _, ld_bar = np.linalg.slogdet(rbar)
    _, ld0 = np.linalg.slogdet(r0)
    _, ld1 = np.linalg.slogdet(r1)
    b = float(np.real(ld_bar - 0.5 * (ld0 + ld1)))             # >= 0
    rho_b2 = math.exp(-2.0 * m * b)                            # rho_B^2
    return 0.5 * (1.0 - math.sqrt(max(0.0, 1.0 - rho_b2)))


def zzb_single(snr_db, m=N_SNAP, fov=FOV_DEG, n_h=600, n_theta=120):
    """Bhattacharyya-relaxed extended Ziv-Zakai bound (deg^2) for single-source
    DoA, uniform prior on [-fov, fov] (width W=2*fov).
    ZZB = INT_0^W h * V{(1-h/W) A(h)} dh, with A(h)=mean_theta Pmin(theta,
    theta+h) the BASE-ANGLE-AVERAGED pairwise min error probability, Pmin from
    _pmin_pair (Kailath lower bound via the ONE-snapshot Bhattacharyya B_1),
    and V{} the valley-filling operator applied to the FULL prior-weighted
    product (1-h/W)A(h). Low-SNR limit -> W^2/12.
    The h-grid is GEOMETRIC (dense near 0) so the near-origin mass that carries
    the high-SNR bound down to the CRB is resolved; a uniform grid misses it."""
    W = 2 * fov
    hs = np.geomspace(1e-3, W, n_h)                            # dense near 0
    g = np.empty_like(hs)
    for i, h in enumerate(hs):
        ths = np.linspace(-fov, fov - h, n_theta)
        pmin = np.array([_pmin_pair(t, t + h, snr_db, m) for t in ths])
        g[i] = (1.0 - h / W) * pmin.mean()
    # valley-filling: non-increasing envelope from the right
    vf = np.maximum.accumulate(g[::-1])[::-1]
    return float(np.trapz(hs * vf, hs))                        # deg^2


def run_bounds(args):
    """SNR sweep: empirical MUSIC RMSE + bias vs sqrt(CRB) vs sqrt(ZZB),
    single-source, to expose the threshold region the CRB misses."""
    rng = np.random.default_rng(args.seed)
    snrs = [-15, -12, -9, -6, -3, 0, 3, 6, 9, 12, 15, 18]
    n_trials = 200 if args.quick else 3000
    W2_12 = (2 * FOV_DEG) ** 2 / 12.0                           # prior variance
    rows = []
    for snr in snrs:
        errs, biases = [], []
        for _ in range(n_trials):
            th = draw_scene(rng, k=1)                           # single source
            x = sim_snapshots(th, rng, snr_db=snr)
            r_hat = x @ x.conj().T / N_SNAP
            est = music_doa(r_hat, k=1)
            errs.append((est[0] - th[0]) ** 2)
            biases.append(est[0] - th[0])
        crb = float(np.mean(crb_stochastic(draw_scene(rng, k=1), snr_db=snr)))
        # CRB averaged over scenes is cleaner via a small inner average:
        crb = float(np.mean([np.mean(crb_stochastic(draw_scene(rng, k=1),
                     snr_db=snr)) for _ in range(200)]))
        zzb = zzb_single(snr)
        rows.append(dict(snr_db=snr,
                         rmse_music=float(np.sqrt(np.mean(errs))),
                         bias_deg=float(np.mean(biases)),
                         sqrt_crb=float(np.sqrt(crb)),
                         sqrt_zzb=float(np.sqrt(zzb))))
        print(f"SNR {snr:+3d} dB | MUSIC {rows[-1]['rmse_music']:6.3f} | "
              f"sqrt(CRB) {rows[-1]['sqrt_crb']:6.3f} | "
              f"sqrt(ZZB) {rows[-1]['sqrt_zzb']:6.3f} | "
              f"bias {rows[-1]['bias_deg']:+.3f} deg")
    # correctness checks (theorems for a valid lower bound + convergence)
    hi = rows[-1]
    zzb_le_music = all(r['sqrt_zzb'] <= r['rmse_music'] + 0.05 for r in rows)
    zzb_monotone = all(rows[i]['sqrt_zzb'] >= rows[i + 1]['sqrt_zzb'] - 1e-6
                       for i in range(len(rows) - 1))
    chk = {
        'zzb_is_lower_bound_on_music': zzb_le_music,        # ZZB <= MUSIC RMSE
        'zzb_monotone_decreasing_in_snr': zzb_monotone,
        'zzb_highSNR_converges_to_crb': 0.3 <= hi['sqrt_zzb'] /
        max(hi['sqrt_crb'], 1e-9) <= 3.0,                    # same order as CRB
        'prior_std_deg': round(math.sqrt(W2_12), 3),
    }
    res = dict(config=dict(n_ant=N_ANT, n_snap=N_SNAP, k=1, fov=FOV_DEG,
                           n_trials=n_trials, seed=args.seed,
                           prior_std_deg=round(math.sqrt(W2_12), 3)),
               sweep=rows, checks=chk)
    out = "eval/seed_results/doa_bounds_snr_sweep.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"checks: {chk}")
    print(f"Saved: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["baselines", "train", "eval", "bounds"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true", help="smoke-test sizes")
    ap.add_argument("--bits", type=int, default=6, help="quant-cov bits/entry")
    ap.add_argument("--latent-bits", dest="latent_bits", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args()
    {"baselines": run_baselines, "train": run_train, "eval": run_eval,
     "bounds": run_bounds}[args.stage](args)


if __name__ == "__main__":
    main()
