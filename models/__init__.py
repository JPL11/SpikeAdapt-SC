"""SpikeAdapt-SC Models Package.

Provides clean, documented implementations matching the paper's architecture:
- SNN modules: LIFNeuron, MPBN, spike functions, channel models
- SpikeAdaptSC: full encoder-scorer-mask-channel-decoder pipeline
- NoiseAwareScorer: BER-conditioned importance scoring with diversity loss
"""

from .snn_modules import (
    SpikeFunction,
    SpikeFunction_Learnable,
    IFNeuron,
    LIFNeuron,
    IHFNeuron,
    MPBN,
    BSC_Channel,
    AWGN_Channel,
    Rayleigh_Channel,
    BEC_Channel,
    get_channel,
)

from .spikeadapt_sc import (
    SpikeAdaptSC,
    Encoder,
    Decoder,
    LearnedBlockMask,
)

from .noise_aware_scorer import NoiseAwareScorer

__all__ = [
    'SpikeAdaptSC',
    'Encoder', 'Decoder',
    'LearnedBlockMask',
    'NoiseAwareScorer',
    'SpikeFunction', 'SpikeFunction_Learnable',
    'IFNeuron', 'LIFNeuron', 'IHFNeuron',
    'MPBN',
    'BSC_Channel', 'AWGN_Channel', 'Rayleigh_Channel', 'BEC_Channel',
    'get_channel',
]
