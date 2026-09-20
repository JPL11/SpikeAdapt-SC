#!/usr/bin/env python3
"""Regular LDPC codes with vectorized min-sum BP decoding (numpy).

Used to give the separation-based JPEG baseline a *modern* FEC instead of the
rate-1/3 repetition code: a regular (dv=3, dc=6) rate-1/2 LDPC (BP threshold
p* ~= 0.084 over the BSC) and a (dv=4, dc=6) rate-1/3 code. For reference,
BSC capacity limits: rate 1/2 requires p <= 0.110, rate 1/3 requires
p <= 0.174. No code of those rates can communicate reliably beyond that.

Methodology: since the code is linear and the BSC is symmetric, code
performance is independent of the transmitted data (all-zero codeword
assumption). We characterize each code offline -- frame error rate (FER) and
residual BER inside failed frames as functions of channel BER -- and the JPEG
baseline applies frame-level corruption sampled from those statistics.

Run as a script to produce eval/seed_results/ldpc_characterization.json.
"""

import json
import os
import numpy as np


# --------------------------------------------------------------- construction
def make_regular_ldpc(n=648, dv=3, dc=6, seed=0):
    """Gallager-style random regular LDPC parity-check matrix (m x n).

    Column weight dv, row weight dc, m = n*dv/dc. Resamples to avoid
    duplicate edges (which create length-4 cycles).
    """
    assert (n * dv) % dc == 0
    m = n * dv // dc
    rng = np.random.default_rng(seed)

    # Progressive Edge Growth (PEG): for each variable node add dv edges,
    # each time picking the check node farthest from the variable node in
    # the current bipartite graph (maximizes local girth), breaking ties by
    # lowest current check degree, then randomly.
    check_deg = np.zeros(m, dtype=int)
    var_adj = [[] for _ in range(n)]   # variable -> checks
    chk_adj = [[] for _ in range(m)]   # check -> variables

    for v in range(n):
        for _ in range(dv):
            # BFS from v over current graph to find distances to checks
            dist = np.full(m, np.inf)
            seen_v = {v}
            frontier_checks = set(var_adj[v])
            d = 0
            while frontier_checks:
                for c in frontier_checks:
                    if dist[c] == np.inf:
                        dist[c] = d
                next_vars = set()
                for c in frontier_checks:
                    next_vars.update(chk_adj[c])
                next_vars -= seen_v
                seen_v |= next_vars
                nf = set()
                for v2 in next_vars:
                    nf.update(var_adj[v2])
                frontier_checks = {c for c in nf if dist[c] == np.inf}
                d += 1
            # candidates: unreachable checks (inf distance) else max distance;
            # never reuse a check already attached to v
            attached = set(var_adj[v])
            cand = [c for c in range(m) if c not in attached]
            dmax = max(dist[c] for c in cand)
            cand = [c for c in cand if dist[c] == dmax]
            deg_min = min(check_deg[c] for c in cand)
            cand = [c for c in cand if check_deg[c] == deg_min]
            c = int(rng.choice(cand))
            var_adj[v].append(c)
            chk_adj[c].append(v)
            check_deg[c] += 1

    H = np.zeros((m, n), dtype=np.uint8)
    for v in range(n):
        for c in var_adj[v]:
            H[c, v] = 1
    return H


# ------------------------------------------------------------------- decoding
class MinSumDecoder:
    """Vectorized BP over the BSC, batch of codewords.

    method='minsum': normalized min-sum (alpha). method='sumprod': exact
    tanh-rule sum-product (slower, better for dv>3 graphs where min-sum
    oscillates at high LLR magnitudes).
    """

    def __init__(self, H, max_iter=50, alpha=0.8, method='minsum'):
        self.H = H
        self.max_iter = max_iter
        self.alpha = alpha
        self.method = method
        self.m, self.n = H.shape
        self.rows, self.cols = np.nonzero(H)      # edge list (E,)
        self.E = len(self.rows)

    def decode(self, hard_in, p):
        """hard_in: (B, n) received hard bits; p: channel BER.
        Returns (B, n) decoded bits."""
        B = hard_in.shape[0]
        Lch = np.log((1 - p) / max(p, 1e-12))
        llr = (1 - 2 * hard_in.astype(np.float64)) * Lch     # (B, n)

        v2c = llr[:, self.cols].copy()                        # (B, E)
        for _ in range(self.max_iter):
            if self.method == 'sumprod':
                # --- exact tanh rule: c2v = 2*atanh(prod_{e'!=e} tanh(v2c/2))
                t = np.tanh(np.clip(v2c, -30, 30) / 2.0)
                # avoid zeros for stable product-exclusion via division
                t = np.where(np.abs(t) < 1e-12, 1e-12 * np.sign(t + 1e-30), t)
                row_prod = np.ones((B, self.m))
                np.multiply.at(row_prod, (slice(None), self.rows), t)
                ext = row_prod[:, self.rows] / t
                ext = np.clip(ext, -0.999999999999, 0.999999999999)
                c2v = 2.0 * np.arctanh(ext)
            else:
                # --- check node update (normalized min-sum) per row
                sign = np.sign(v2c)
                sign[sign == 0] = 1.0
                mag = np.abs(v2c)
                row_sign = np.ones((B, self.m))
                np.multiply.at(row_sign, (slice(None), self.rows), sign)
                INF = 1e30
                min1 = np.full((B, self.m), INF)
                np.minimum.at(min1, (slice(None), self.rows), mag)
                # second minimum: min over edges excluding position of min1
                is_min = mag == min1[:, self.rows]
                mag2 = np.where(is_min, INF, mag)
                min2 = np.full((B, self.m), INF)
                np.minimum.at(min2, (slice(None), self.rows), mag2)
                ext_mag = np.where(is_min, min2[:, self.rows], min1[:, self.rows])
                ext_sign = row_sign[:, self.rows] * sign
                c2v = self.alpha * ext_sign * ext_mag

            # --- variable node update
            col_sum = llr.copy()
            np.add.at(col_sum, (slice(None), self.cols), c2v)
            v2c = col_sum[:, self.cols] - c2v

            # --- tentative decision + early stop
            hard = (col_sum < 0).astype(np.uint8)
            synd = (hard @ self.H.T) % 2
            if not synd.any():
                return hard
        return (col_sum < 0).astype(np.uint8)


# ------------------------------------------------------------ characterization
def characterize(H, ber_grid, n_frames=2000, max_iter=50, seed=1,
                 method='sumprod'):
    """All-zero-codeword Monte Carlo: FER and residual BER vs channel BER."""
    dec = MinSumDecoder(H, max_iter=max_iter, method=method)
    rng = np.random.default_rng(seed)
    out = {}
    for p in ber_grid:
        if p <= 0:
            out[f'{p:.4f}'] = dict(fer=0.0, res_ber_failed=0.0, res_ber=0.0)
            continue
        rx = (rng.random((n_frames, dec.n)) < p).astype(np.uint8)
        decoded = dec.decode(rx, p)
        frame_err = decoded.any(axis=1)
        fer = float(frame_err.mean())
        res_failed = (float(decoded[frame_err].mean())
                      if frame_err.any() else 0.0)
        out[f'{p:.4f}'] = dict(
            fer=fer,
            res_ber_failed=res_failed,
            res_ber=float(decoded.mean()),
        )
    return out


def main():
    os.makedirs('eval/seed_results', exist_ok=True)
    ber_grid = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                0.09, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    results = {}
    for name, n, dv, dc in [('r12_n648_dv3dc6', 648, 3, 6),
                            ('r13_n648_dv4dc6', 648, 4, 6)]:
        rate = 1 - dv / dc
        print(f'== {name} (rate {rate:.3f}) ==')
        H = make_regular_ldpc(n, dv, dc, seed=0)
        stats = characterize(H, ber_grid)
        for k, v in stats.items():
            print(f'  p={k}: FER={v["fer"]:.4f}  resBER={v["res_ber"]:.4f}')
        results[name] = dict(n=n, dv=dv, dc=dc, rate=rate, stats=stats)

    path = 'eval/seed_results/ldpc_characterization.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'Saved: {path}')


if __name__ == '__main__':
    main()
