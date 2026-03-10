"""Energy estimation via spike operation counting.

Compares SNN synaptic operations (SynOps) to ANN multiply-accumulate (MAC)
operations using the Horowitz 2014 energy model (45nm CMOS):
  - SynOp: 0.9 pJ per event-driven operation
  - MAC:   4.6 pJ per multiply-accumulate
"""

from collections import defaultdict


class SpikeOpCounter:
    """Counts synaptic operations for energy efficiency estimation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.snn_synops = 0
        self.ann_macs = 0
        self.spike_counts = defaultdict(int)
        self.total_elements = defaultdict(int)

    def count_snn_layer(self, spikes, layer_name, out_channels, kernel_size=3):
        """Count SynOps: each spike triggers fan_out downstream operations."""
        n_spikes = spikes.sum().item()
        fan_out = out_channels * kernel_size * kernel_size
        self.snn_synops += n_spikes * fan_out
        self.spike_counts[layer_name] += n_spikes
        self.total_elements[layer_name] += spikes.numel()

    def count_ann_layer(self, input_tensor, out_channels, kernel_size=3):
        """Count MACs for equivalent dense ANN layer."""
        B, C, H, W = input_tensor.shape
        macs_per_element = C * kernel_size * kernel_size * out_channels
        self.ann_macs += B * H * W * macs_per_element

    def get_energy_ratio(self):
        """Energy ratio: SNN / ANN (lower is better)."""
        E_SNN = self.snn_synops * 0.9  # pJ per SynOp
        E_ANN = self.ann_macs * 4.6    # pJ per MAC
        return E_SNN / max(E_ANN, 1)

    def get_summary(self):
        firing_rates = {}
        for name in self.spike_counts:
            if self.total_elements[name] > 0:
                firing_rates[name] = self.spike_counts[name] / self.total_elements[name]
        return {
            'snn_synops': self.snn_synops,
            'ann_macs': self.ann_macs,
            'energy_ratio': self.get_energy_ratio(),
            'firing_rates': firing_rates,
            'energy_savings_pct': (1 - self.get_energy_ratio()) * 100,
        }
