#!/usr/bin/env python3
"""On-device transmitter-side benchmark for Jetson Orin Nano.

Measures per-image latency and board power (INA3221, VDD_IN rail) for the
UAV-side encoder stack, batch 1, fp32:

  1. front        : ResNet-50 up to layer3 (shared by all methods)
  2. sasc_T8      : front + SNN encoder (T=8) + noise-aware scorer + mask
  3. sasc_T4      : same with temporal truncation T'=4 (clean-channel op point)
  4. cnn1bit      : front + CNN-1bit encoder (single pass)

Power protocol: sample INA3221 sysfs at ~20 Hz in a background thread;
10 s idle baseline, then >=30 s sustained inference per config.
Energy/image = mean(P_load - P_idle) * latency  (dynamic), plus total power.

Run ON THE JETSON:  ~/torchenv/bin/python bench_jetson_encoder.py
Output: jetson_encoder_bench.json
"""

import json
import os
import sys
import threading
import time

import torch
import torch.nn as nn
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from train_aid_v5 import EncoderV5
from train_aid_v2 import LearnedBlockMask
from noise_aware_scorer import NoiseAwareScorer

DEVICE = torch.device('cuda')
HWMON = None
for h in os.listdir('/sys/class/hwmon'):
    p = f'/sys/class/hwmon/{h}/name'
    if os.path.exists(p) and open(p).read().strip() == 'ina3221':
        HWMON = f'/sys/class/hwmon/{h}'
        break


class ResNet50Front(nn.Module):
    """Local copy (weights=None to avoid download; state loaded from ckpt)."""
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.conv1 = r.conv1; self.bn1 = r.bn1; self.relu = r.relu
        self.maxpool = r.maxpool
        self.layer1 = r.layer1; self.layer2 = r.layer2; self.layer3 = r.layer3
        self.spatial_pool = nn.Identity()

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.spatial_pool(self.layer3(self.layer2(self.layer1(x))))


class SpikeEncoderTx(nn.Module):
    """Transmitter side of SpikeAdaptSC_v5c_NA: encoder + scorer + mask."""
    def __init__(self, C_in=1024, C1=256, C2=36, T=8, target_rate=0.75):
        super().__init__()
        self.T = T
        self.encoder = EncoderV5(C_in, C1, C2, T, use_mpbn=True)
        self.scorer = NoiseAwareScorer(C_spike=C2, hidden=32)
        self.block_mask = LearnedBlockMask(target_rate, 0.5)

    def forward(self, feat, tprime=None, noise_param=0.0):
        tprime = tprime or self.T
        all_S2, m1, m2 = [], None, None
        for t in range(tprime):
            _, s2, m1, m2 = self.encoder(feat, m1, m2, t=t)
            all_S2.append(s2)
        importance = self.scorer(all_S2, noise_param).squeeze(1)
        mask, _ = self.block_mask(importance, training=False)
        return [s * mask for s in all_S2]


class PowerSampler:
    def __init__(self, period=0.05):
        self.period = period
        self.samples = []
        self._stop = threading.Event()
        self._th = None

    def _read_mw(self):
        # channel 1 = VDD_IN on Orin Nano devkit
        v = int(open(f'{HWMON}/in1_input').read())      # mV
        i = int(open(f'{HWMON}/curr1_input').read())    # mA
        return v * i / 1000.0                           # mW

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._read_mw())
            except OSError:
                pass
            time.sleep(self.period)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        self._th.join()
        return sum(self.samples) / max(1, len(self.samples))


def bench(fn, warmup=10, min_s=30.0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    sampler = PowerSampler()
    sampler.start()
    n, t0 = 0, time.time()
    while time.time() - t0 < min_s:
        fn()
        n += 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    p_mw = sampler.stop()
    return dict(latency_ms=1000.0 * dt / n, power_mw=p_mw, iters=n)


def main():
    assert HWMON, 'INA3221 hwmon not found'
    torch.backends.cudnn.benchmark = True

    front = ResNet50Front().to(DEVICE).eval()
    bb = torch.load('backbone_best.pth', map_location=DEVICE, weights_only=False)
    front.load_state_dict({k: v for k, v in bb.items()
                           if not k.startswith(('layer4.', 'fc.', 'avgpool.',
                                                'spatial_pool.'))}, strict=False)

    tx = SpikeEncoderTx().to(DEVICE).eval()
    ck = torch.load('v5cna_best.pth', map_location=DEVICE, weights_only=False)
    missing, unexpected = tx.load_state_dict(
        {k: v for k, v in ck['model'].items()
         if k.startswith(('encoder.', 'scorer.', 'block_mask.'))}, strict=False)
    print('tx load: missing', len(missing), 'unexpected', len(unexpected))

    from train_1bit_baseline import BinaryCNN_SC
    cnn = BinaryCNN_SC(C_in=1024, C1=256, C2=36).to(DEVICE).eval()
    ck1 = torch.load('1bit_best.pth', map_location=DEVICE, weights_only=False)
    cnn.load_state_dict(ck1['model'])

    img = torch.randn(1, 3, 224, 224, device=DEVICE)
    with torch.no_grad():
        feat = front(img)

    results = {'device': torch.cuda.get_device_name(0),
               'power_mode': open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip()}

    # idle baseline
    time.sleep(2)
    s = PowerSampler(); s.start(); time.sleep(10)
    results['idle_power_mw'] = s.stop()
    print(f"idle: {results['idle_power_mw']:.0f} mW")

    with torch.no_grad():
        cases = {
            'front': lambda: front(img),
            'sasc_T8': lambda: tx(front(img), tprime=8),
            'sasc_T4': lambda: tx(front(img), tprime=4),
            'cnn1bit': lambda: cnn.encoder(front(img)),
        }
        for name, fn in cases.items():
            r = bench(fn)
            r['dyn_energy_mj'] = (r['power_mw'] - results['idle_power_mw']) \
                * r['latency_ms'] / 1e3
            results[name] = r
            print(f"{name}: {r['latency_ms']:.2f} ms, {r['power_mw']:.0f} mW "
                  f"({r['dyn_energy_mj']:.1f} mJ dyn/img, {r['iters']} iters)")

    with open('jetson_encoder_bench.json', 'w') as f:
        json.dump(results, f, indent=1)
    print('Saved: jetson_encoder_bench.json')


if __name__ == '__main__':
    main()
