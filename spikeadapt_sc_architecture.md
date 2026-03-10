# SpikeAdapt-SC: Detailed Architecture & Pseudocode

## Table of Contents
1. System Overview
2. Detailed Architecture
3. Component Specifications
4. Pseudocode: All Modules
5. Training Algorithm
6. Inference Algorithm
7. Loss Functions
8. Dimension Walkthrough

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UAV / SATELLITE EDGE DEVICE                     │
│                                                                     │
│  ┌──────────┐    ┌──────────────────────────────────────────────┐   │
│  │  Sensor  │───▶│  Feature Extraction (ResNet50 Layers 1-14)  │   │
│  │ (Camera) │    │  (Frozen after Step 1 training)              │   │
│  └──────────┘    └───────────────────┬──────────────────────────┘   │
│                                      │ F ∈ R^{C×H×W}               │
│                                      ▼                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              SpikeAdapt-SC ENCODER                           │   │
│  │                                                              │   │
│  │  ┌─────────┐   ┌─────┐   ┌─────────┐   ┌─────┐             │   │
│  │  │Conv1-BN │──▶│ IF1 │──▶│Conv2-BN │──▶│ IF2 │──▶ S_t      │   │
│  │  └─────────┘   └─────┘   └─────────┘   └─────┘    ↓        │   │
│  │                              ┌─────────────────────┘        │   │
│  │                              ▼                              │   │
│  │                    ┌───────────────────┐                     │   │
│  │                    │ Spike Rate        │                     │   │
│  │                    │ Entropy Estimator │                     │   │
│  │                    └────────┬──────────┘                     │   │
│  │                             ▼                                │   │
│  │                    ┌───────────────────┐                     │   │
│  │                    │   Block Masking   │──▶ Mask M           │   │
│  │                    └────────┬──────────┘                     │   │
│  │                             ▼                                │   │
│  │                    Masked S_t = S_t ⊙ M                     │   │
│  └─────────────────────┬────────────────────────────────────────┘   │
│                         │                                           │
└─────────────────────────┼───────────────────────────────────────────┘
                          │ Transmit: {Masked S_t for t=1..T} + M
                          │ Total bits per step: nnz(M) × C2
                          ▼
              ╔═══════════════════════════╗
              ║   DIGITAL CHANNEL         ║
              ║   BSC(p) or BEC(p)        ║
              ╚═══════════╤═══════════════╝
                          │ Received: {Ŝ_t for t=1..T} + M
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GROUND STATION / CLOUD                          │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              SpikeAdapt-SC DECODER                           │   │
│  │                                                              │   │
│  │  ┌────────────┐                                              │   │
│  │  │ Zero-Fill  │  (restore full spatial dims using M)         │   │
│  │  └─────┬──────┘                                              │   │
│  │        ▼                                                     │   │
│  │  ┌──────────────────────────────────────────────────────┐    │   │
│  │  │              RECONSTRUCTOR (per time step)           │    │   │
│  │  │  ┌─────────┐   ┌─────┐   ┌─────────┐   ┌─────┐     │    │   │
│  │  │  │Conv3-BN │──▶│ IF3 │──▶│Conv4-BN │──▶│ IHF │     │    │   │
│  │  │  └─────────┘   └─────┘   └─────────┘   └──┬──┘     │    │   │
│  │  │                                     ┌──────┴──────┐  │    │   │
│  │  │                                     │    │        │  │    │   │
│  │  │                                    F_s^t  F_m^t   │  │    │   │
│  │  └────────────────────────────────────────────────────┘  │    │   │
│  │                                                          │    │   │
│  │  ┌──────────────────────────────────────────────────┐    │   │
│  │  │              CONVERTER                           │    │   │
│  │  │  Inputs: [F_s^1, F_m^1, ..., F_s^T, F_m^T]     │    │   │
│  │  │  FCN + Sigmoid → Weighted sum → F'               │    │   │
│  │  └───────────────────┬──────────────────────────────┘    │   │
│  └──────────────────────┼───────────────────────────────────┘   │
│                         ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Task Execution (ResNet50 Layers 15+)                    │   │
│  │  → Classification / Segmentation result                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Architecture (Layer-by-Layer)

### 2.1 Encoder

```
Input: F ∈ R^{C × H × W}           (e.g., 2048 × 4 × 4 for ResNet50)

Stage 1 - Initial Compression (floating-point → spike):
  ┌──────────────────────────────────────────────────────────────┐
  │ Conv1: Conv2d(C, C1, kernel=3, stride=1, padding=1)         │
  │ BN1:   BatchNorm2d(C1)                                      │
  │ Output: I ∈ R^{C1 × H × W}                                 │
  │                                                              │
  │ IF1:   Integrate-and-Fire neuron                             │
  │        m1_t = m1_{t-1} + I          (charge)                 │
  │        S1_t = (m1_t > V_th) ? 1 : 0 (fire)                  │
  │        m1_t = m1_t - S1_t × V_th    (soft reset)             │
  │ Output: S1_t ∈ {0,1}^{C1 × H × W}                          │
  └──────────────────────────────────────────────────────────────┘

Stage 2 - Semantic Extraction (spike → compressed spike):
  ┌──────────────────────────────────────────────────────────────┐
  │ Conv2: Conv2d(C1, C2, kernel=3, stride=1, padding=1)        │
  │ BN2:   BatchNorm2d(C2)                                      │
  │                                                              │
  │ IF2:   Integrate-and-Fire neuron                             │
  │        m2_t = m2_{t-1} + BN2(Conv2(S1_t))                   │
  │        S2_t = (m2_t > V_th) ? 1 : 0                         │
  │        m2_t = m2_t - S2_t × V_th                             │
  │ Output: S2_t ∈ {0,1}^{C2 × H2 × W2}                        │
  │         (this is the semantic information at time step t)    │
  └──────────────────────────────────────────────────────────────┘

  Note: C2 < C1 < C, and H2 ≤ H, W2 ≤ W
  Typical: C=2048, C1=256, C2=128 for ResNet50 split at layer 14
  For spatial downsampling: use stride=2 in Conv2 or keep stride=1
```

### 2.2 Spike Rate Entropy Estimator (NEW - runs after T time steps)

```
Input: {S2_1, S2_2, ..., S2_T}     each ∈ {0,1}^{C2 × H2 × W2}

Step 1 - Compute per-block firing rate:
  ┌──────────────────────────────────────────────────────────────┐
  │ For each spatial position (i,j) where i ∈ [0,H2), j ∈ [0,W2):  │
  │                                                              │
  │   f(i,j) = (1 / (T × C2)) × Σ_t Σ_c S2_t[c, i, j]        │
  │                                                              │
  │   This is the average firing rate across all channels and   │
  │   all time steps for spatial block (i,j)                    │
  │   f(i,j) ∈ [0, 1]                                           │
  └──────────────────────────────────────────────────────────────┘

Step 2 - Compute per-block entropy:
  ┌──────────────────────────────────────────────────────────────┐
  │ For each spatial position (i,j):                             │
  │                                                              │
  │   H(i,j) = -f(i,j) × log2(f(i,j))                         │
  │            -(1-f(i,j)) × log2(1-f(i,j))                    │
  │                                                              │
  │   Clamp f to [ε, 1-ε] to avoid log(0), where ε = 1e-7     │
  │   H(i,j) ∈ [0, 1]                                           │
  │                                                              │
  │ Output: Entropy map E ∈ R^{H2 × W2}                        │
  └──────────────────────────────────────────────────────────────┘

Step 3 - Generate binary mask:
  ┌──────────────────────────────────────────────────────────────┐
  │ M(i,j) = 1  if H(i,j) ≥ η                                  │
  │         = 0  otherwise                                       │
  │                                                              │
  │ M ∈ {0,1}^{H2 × W2}                                        │
  │                                                              │
  │ η is a hyperparameter controlling the target rate            │
  │ Higher η → fewer blocks transmitted → lower bandwidth        │
  │ Lower η → more blocks transmitted → higher bandwidth         │
  │                                                              │
  │ For training: use soft mask with straight-through estimator  │
  │   M_soft(i,j) = σ((H(i,j) - η) / τ)   where τ is temp     │
  │   Forward: use hard mask (round to 0/1)                      │
  │   Backward: use gradient of M_soft                           │
  └──────────────────────────────────────────────────────────────┘
```

### 2.3 Masked Transmission

```
For each time step t = 1, ..., T:
  ┌──────────────────────────────────────────────────────────────┐
  │ Masked_S2_t[c, i, j] = S2_t[c, i, j] × M(i,j)            │
  │                                                              │
  │ The mask M is spatial-only — it applies to ALL channels     │
  │ of each spatial position. If M(i,j) = 0, all C2 channels   │
  │ at position (i,j) are dropped.                               │
  │                                                              │
  │ Bits transmitted per time step:                              │
  │   nnz(M) × C2   (where nnz = number of 1s in M)            │
  │                                                              │
  │ Total bits transmitted:                                      │
  │   T × nnz(M) × C2 + H2 × W2  (mask overhead)              │
  │                                                              │
  │ Compression ratio:                                           │
  │   (C × H × W × 32) / (T × nnz(M) × C2 + H2 × W2)         │
  └──────────────────────────────────────────────────────────────┘
```

### 2.4 Digital Channel

```
Input: Masked_S2_t ∈ {0,1}^{C2 × H2 × W2}   (with zeros at masked positions)
Only the non-zero blocks are actually transmitted.

BSC with bit error rate p:
  ┌──────────────────────────────────────────────────────────────┐
  │ For each transmitted bit b:                                  │
  │   b_received = b ⊕ Bernoulli(p)                             │
  │   (bit is flipped with probability p)                        │
  └──────────────────────────────────────────────────────────────┘

BEC with bit erasure rate p:
  ┌──────────────────────────────────────────────────────────────┐
  │ For each transmitted bit b:                                  │
  │   With probability p: b is erased → replaced by Bernoulli(0.5)  │
  │   With probability 1-p: b is received correctly              │
  └──────────────────────────────────────────────────────────────┘

During backpropagation: channel is treated as identity (gradient = 1)
```

### 2.5 Decoder: Reconstructor

```
Input: Ŝ2_t (received, possibly noisy semantic info)

Step 0 - Zero-fill:
  ┌──────────────────────────────────────────────────────────────┐
  │ Restore full spatial dimensions using mask M:                │
  │ Ŝ2_t_full[c, i, j] = Ŝ2_t[c, i, j]  if M(i,j) = 1       │
  │                      = 0                if M(i,j) = 0       │
  │                                                              │
  │ Output: Ŝ2_t_full ∈ {0,1}^{C2 × H2 × W2}                 │
  └──────────────────────────────────────────────────────────────┘

Stage 3 - First decompression:
  ┌──────────────────────────────────────────────────────────────┐
  │ Conv3: Conv2d(C2, C1, kernel=3, stride=1, padding=1)        │
  │ BN3:   BatchNorm2d(C1)                                      │
  │                                                              │
  │ IF3:   Integrate-and-Fire neuron                             │
  │        m3_t = m3_{t-1} + BN3(Conv3(Ŝ2_t_full))             │
  │        S3_t = (m3_t > V_th) ? 1 : 0                         │
  │        m3_t = m3_t - S3_t × V_th                             │
  │ Output: S3_t ∈ {0,1}^{C1 × H1 × W1}                       │
  └──────────────────────────────────────────────────────────────┘

Stage 4 - Final decompression with IHF:
  ┌──────────────────────────────────────────────────────────────┐
  │ Conv4: Conv2d(C1, C, kernel=3, stride=1, padding=1)         │
  │ BN4:   BatchNorm2d(C)                                       │
  │                                                              │
  │ IHF:   Integrate-and-Hybrid-Fire neuron (LEARNABLE)         │
  │        m4_t = m4_{t-1} + BN4(Conv4(S3_t))   (charge)       │
  │        S4_t = (m4_t > V_th_learn) ? 1 : 0   (fire)         │
  │        m4_t = m4_t - S4_t × V_th_learn       (soft reset)   │
  │        M4_t = m4_t                            (output MP)    │
  │                                                              │
  │ Output: F_s^t = S4_t ∈ {0,1}^{C × H × W}   (spike)        │
  │         F_m^t = M4_t ∈ R^{C × H × W}        (membrane pot) │
  └──────────────────────────────────────────────────────────────┘
```

### 2.6 Decoder: Converter

```
Input: {F_s^1, F_m^1, F_s^2, F_m^2, ..., F_s^T, F_m^T}
       Total: 2T tensors, each ∈ R^{C × H × W}

  ┌──────────────────────────────────────────────────────────────┐
  │ Step 1 - Stack all outputs along a new dimension:            │
  │   X = stack([F_s^1, F_m^1, ..., F_s^T, F_m^T])             │
  │   X ∈ R^{2T × C × H × W}                                   │
  │                                                              │
  │ Step 2 - Learnable weighted combination:                     │
  │   W = Sigmoid(FCN(X))    where FCN: R^{2T} → R^{2T}        │
  │   (FCN is applied per-element across spatial dimensions)     │
  │                                                              │
  │ Step 3 - Weighted sum:                                       │
  │   F' = Σ_{k=1}^{2T} W_k × X_k                              │
  │   F' ∈ R^{C × H × W}                                       │
  │                                                              │
  │ This is the final reconstructed feature                      │
  └──────────────────────────────────────────────────────────────┘

Output: F' ∈ R^{C × H × W}  →  fed into Task Execution network
```

---

## 3. Component Specifications

### 3.1 Parameter Table (ResNet50, split at layer 14)

| Component | Layer | Input Dims | Output Dims | Kernel | Stride | Params |
|-----------|-------|-----------|-------------|--------|--------|--------|
| **Encoder** | Conv1 | (2048, 4, 4) | (256, 4, 4) | 3×3 | 1 | 4,718,592 |
| | BN1 | (256, 4, 4) | (256, 4, 4) | - | - | 512 |
| | IF1 | (256, 4, 4) | (256, 4, 4) | - | - | 0 |
| | Conv2 | (256, 4, 4) | (128, 4, 4) | 3×3 | 1 | 294,912 |
| | BN2 | (128, 4, 4) | (128, 4, 4) | - | - | 256 |
| | IF2 | (128, 4, 4) | (128, 4, 4) | - | - | 0 |
| **Entropy Est.** | - | (128, 4, 4) × T | (4, 4) | - | - | 0 |
| **Block Mask** | - | (4, 4) | (4, 4) | - | - | 0 |
| **Reconstructor** | Conv3 | (128, 4, 4) | (256, 4, 4) | 3×3 | 1 | 294,912 |
| | BN3 | (256, 4, 4) | (256, 4, 4) | - | - | 512 |
| | IF3 | (256, 4, 4) | (256, 4, 4) | - | - | 0 |
| | Conv4 | (256, 4, 4) | (2048, 4, 4) | 3×3 | 1 | 4,718,592 |
| | BN4 | (2048, 4, 4) | (2048, 4, 4) | - | - | 4,096 |
| | IHF | (2048, 4, 4) | 2×(2048, 4, 4) | - | - | 1* |
| **Converter** | FCN | 2T | 2T | - | - | (2T)² |

*IHF has 1 learnable parameter: V_th_learn (firing threshold)

### 3.2 Bits Transmitted

| Scenario | Bits per time step | Total bits (T steps) | Mask overhead |
|----------|-------------------|---------------------|---------------|
| SNN-SC (fixed) | C2 × H2 × W2 = 128×4×4 = 2048 | T × 2048 | 0 |
| SpikeAdapt-SC | nnz(M) × C2 | T × nnz(M) × C2 | H2×W2 = 16 bits |

When T=8, SNN-SC transmits 16,384 bits total.
SpikeAdapt-SC with 75% mask retention: ~12,288 + 16 = 12,304 bits (25% savings).

---

## 4. Pseudocode: All Modules

### 4.1 Spiking Neuron: IF (Integrate-and-Fire)

```python
class IF_Neuron:
    """Standard Integrate-and-Fire neuron (used in encoder and reconstructor)"""

    def __init__(self, V_th=1.0):
        self.V_th = V_th
        self.membrane = None    # membrane potential, shape matches input

    def reset_state(self):
        self.membrane = None

    def forward(self, input_current):
        """
        Args:
            input_current: tensor of shape (batch, C, H, W)
        Returns:
            spikes: binary tensor of shape (batch, C, H, W)
        """
        # Initialize membrane potential on first call
        if self.membrane is None:
            self.membrane = zeros_like(input_current)

        # CHARGE: accumulate input into membrane potential
        self.membrane = self.membrane + input_current

        # FIRE: generate spikes where membrane exceeds threshold
        spikes = heaviside(self.membrane - self.V_th)
        # heaviside(x) = 1 if x > 0, else 0

        # RESET: soft reset — subtract threshold where spike occurred
        self.membrane = self.membrane - spikes * self.V_th

        return spikes    # ∈ {0, 1}

    def backward_surrogate(self, grad_output):
        """Surrogate gradient for fire step (used during training)"""
        # Sigmoid surrogate: σ'(membrane - V_th)
        sigmoid_grad = sigmoid(self.membrane - self.V_th)
        sigmoid_grad = sigmoid_grad * (1 - sigmoid_grad)
        return grad_output * sigmoid_grad
```

### 4.2 Spiking Neuron: IHF (Integrate-and-Hybrid-Fire) — LEARNABLE

```python
class IHF_Neuron:
    """
    Novel neuron: outputs BOTH spike and membrane potential.
    Extended with learnable threshold.
    """

    def __init__(self, V_th_init=1.0):
        # LEARNABLE firing threshold
        self.V_th = Parameter(tensor(V_th_init))  # nn.Parameter, optimized via backprop
        self.membrane = None

    def reset_state(self):
        self.membrane = None

    def forward(self, input_current):
        """
        Args:
            input_current: tensor of shape (batch, C, H, W)
        Returns:
            spikes:   binary tensor (batch, C, H, W)
            mem_out:  float tensor (batch, C, H, W)
        """
        if self.membrane is None:
            self.membrane = zeros_like(input_current)

        # CHARGE
        self.membrane = self.membrane + input_current

        # FIRE
        spikes = heaviside(self.membrane - self.V_th)  # {0, 1}

        # RESET (soft)
        self.membrane = self.membrane - spikes * self.V_th

        # OUTPUT MEMBRANE POTENTIAL (unique to IHF)
        mem_out = self.membrane.clone()

        return spikes, mem_out

    # Same surrogate gradient approach as IF for the fire step
```

### 4.3 Encoder

```python
class SpikeAdaptEncoder:
    """Runs on UAV/Satellite edge device"""

    def __init__(self, C_in, C1, C2):
        # Stage 1: float → spike
        self.conv1 = Conv2d(C_in, C1, kernel_size=3, stride=1, padding=1)
        self.bn1   = BatchNorm2d(C1)
        self.if1   = IF_Neuron(V_th=1.0)

        # Stage 2: spike → compressed spike
        self.conv2 = Conv2d(C1, C2, kernel_size=3, stride=1, padding=1)
        self.bn2   = BatchNorm2d(C2)
        self.if2   = IF_Neuron(V_th=1.0)

    def forward_single_step(self, F):
        """
        Forward pass for ONE time step.
        F is the original feature (constant across time steps).

        Args:
            F: tensor (batch, C_in, H, W) — original feature from backbone
        Returns:
            S2_t: binary tensor (batch, C2, H2, W2) — semantic info at time t
        """
        # Stage 1
        I = self.bn1(self.conv1(F))          # (batch, C1, H, W)
        S1_t = self.if1.forward(I)           # (batch, C1, H, W) binary

        # Stage 2
        Z = self.bn2(self.conv2(S1_t))       # (batch, C2, H2, W2)
        S2_t = self.if2.forward(Z)           # (batch, C2, H2, W2) binary

        return S2_t

    def forward_all_steps(self, F, T):
        """
        Run encoder for T time steps. Collect all outputs.

        Args:
            F: tensor (batch, C_in, H, W)
            T: number of time steps
        Returns:
            all_S2: list of T tensors, each (batch, C2, H2, W2)
        """
        self.if1.reset_state()
        self.if2.reset_state()

        all_S2 = []
        for t in range(T):
            S2_t = self.forward_single_step(F)
            all_S2.append(S2_t)

        return all_S2    # list of length T
```

### 4.4 Spike Rate Entropy Estimator

```python
class SpikeRateEntropyEstimator:
    """
    Computes per-spatial-block entropy from spike firing rates.
    NO learnable parameters — purely statistical.
    """

    def __init__(self, epsilon=1e-7):
        self.eps = epsilon

    def compute_firing_rate(self, all_S2):
        """
        Args:
            all_S2: list of T tensors, each (batch, C2, H2, W2) ∈ {0,1}
        Returns:
            firing_rate: (batch, H2, W2) in [0, 1]
        """
        # Stack: (T, batch, C2, H2, W2)
        stacked = stack(all_S2, dim=0)

        # Average over time steps (dim=0) and channels (dim=2)
        # Result: (batch, H2, W2)
        firing_rate = stacked.mean(dim=0).mean(dim=1)

        return firing_rate

    def compute_entropy(self, firing_rate):
        """
        Binary entropy of each spatial block.

        Args:
            firing_rate: (batch, H2, W2)
        Returns:
            entropy_map: (batch, H2, W2) in [0, 1]
        """
        f = firing_rate.clamp(self.eps, 1.0 - self.eps)
        entropy_map = -f * log2(f) - (1 - f) * log2(1 - f)
        return entropy_map

    def forward(self, all_S2):
        """
        Full pipeline: spike sequences → entropy map.

        Args:
            all_S2: list of T tensors
        Returns:
            entropy_map: (batch, H2, W2)
            firing_rate: (batch, H2, W2)
        """
        firing_rate = self.compute_firing_rate(all_S2)
        entropy_map = self.compute_entropy(firing_rate)
        return entropy_map, firing_rate
```

### 4.5 Block Masking

```python
class BlockMask:
    """
    Generates binary mask based on entropy threshold.
    Uses straight-through estimator for training.
    """

    def __init__(self, eta=0.5, temperature=0.1):
        self.eta = eta              # entropy threshold
        self.temperature = temperature  # for soft approximation

    def forward(self, entropy_map, training=True):
        """
        Args:
            entropy_map: (batch, H2, W2) ∈ [0, 1]
            training: bool
        Returns:
            mask: (batch, 1, H2, W2) ∈ {0, 1}  (1-channel, broadcastable over C2)
            transmission_rate: scalar — fraction of blocks transmitted
        """
        if training:
            # Soft mask for gradient flow
            soft_mask = sigmoid((entropy_map - self.eta) / self.temperature)
            # Straight-through estimator: forward uses hard, backward uses soft
            hard_mask = (entropy_map >= self.eta).float()
            mask = hard_mask + (soft_mask - soft_mask.detach())  # STE trick
        else:
            # Hard mask at inference
            mask = (entropy_map >= self.eta).float()

        # Reshape for broadcasting: (batch, 1, H2, W2)
        mask = mask.unsqueeze(1)

        # Compute transmission rate (for rate loss)
        transmission_rate = mask.mean()

        return mask, transmission_rate

    def apply_mask(self, S2_t, mask):
        """
        Zero out blocks below entropy threshold.

        Args:
            S2_t: (batch, C2, H2, W2)
            mask:  (batch, 1, H2, W2) — broadcasts over channels
        Returns:
            masked_S2_t: (batch, C2, H2, W2)
        """
        return S2_t * mask
```

### 4.6 Digital Channel

```python
class BSC_Channel:
    """Binary Symmetric Channel"""

    def forward(self, x, bit_error_rate):
        """
        Args:
            x: binary tensor ∈ {0, 1}
            bit_error_rate: probability of bit flip
        Returns:
            x_noisy: binary tensor with random flips
        """
        # Generate flip mask
        flip_mask = (rand_like(x.float()) < bit_error_rate).float()
        # XOR: flip where mask is 1
        x_noisy = (x + flip_mask) % 2

        return x_noisy


class BEC_Channel:
    """Binary Erasure Channel"""

    def forward(self, x, erasure_rate):
        """
        Args:
            x: binary tensor ∈ {0, 1}
            erasure_rate: probability of erasure
        Returns:
            x_noisy: binary tensor with random erasures replaced by coin flip
        """
        erase_mask = (rand_like(x.float()) < erasure_rate).float()
        random_bits = (rand_like(x.float()) > 0.5).float()
        x_noisy = x * (1 - erase_mask) + random_bits * erase_mask

        return x_noisy
```

### 4.7 Decoder: Reconstructor + Converter

```python
class SpikeAdaptDecoder:
    """Runs on ground station / cloud"""

    def __init__(self, C_out, C1, C2, T, V_th_init=1.0):
        # Reconstructor Stage 3
        self.conv3 = Conv2d(C2, C1, kernel_size=3, stride=1, padding=1)
        self.bn3   = BatchNorm2d(C1)
        self.if3   = IF_Neuron(V_th=1.0)

        # Reconstructor Stage 4 with learnable IHF
        self.conv4 = Conv2d(C1, C_out, kernel_size=3, stride=1, padding=1)
        self.bn4   = BatchNorm2d(C_out)
        self.ihf   = IHF_Neuron(V_th_init=V_th_init)  # LEARNABLE threshold

        # Converter
        self.converter_fc = Linear(2 * T, 2 * T)  # weights for 2T outputs
        self.T = T

    def zero_fill(self, received_S2_t, mask):
        """
        Restore full spatial dims. Masked positions become 0.

        Args:
            received_S2_t: (batch, C2, H2, W2) — noisy received semantic info
            mask: (batch, 1, H2, W2)
        Returns:
            S2_full: (batch, C2, H2, W2)
        """
        return received_S2_t * mask    # mask already applied; this is identity
        # At receiver, we know which positions are zeros from the received mask

    def reconstruct_single_step(self, S2_t_full):
        """
        One time step of reconstruction.

        Args:
            S2_t_full: (batch, C2, H2, W2)
        Returns:
            F_s_t: (batch, C_out, H, W) — spike output
            F_m_t: (batch, C_out, H, W) — membrane potential output
        """
        # Stage 3
        Z3 = self.bn3(self.conv3(S2_t_full))
        S3_t = self.if3.forward(Z3)

        # Stage 4 with IHF
        Z4 = self.bn4(self.conv4(S3_t))
        F_s_t, F_m_t = self.ihf.forward(Z4)

        return F_s_t, F_m_t

    def convert(self, all_F_s, all_F_m):
        """
        Fuse all time step outputs into final reconstructed feature.

        Args:
            all_F_s: list of T tensors, each (batch, C_out, H, W)
            all_F_m: list of T tensors, each (batch, C_out, H, W)
        Returns:
            F_prime: (batch, C_out, H, W)
        """
        # Interleave: [F_s^1, F_m^1, F_s^2, F_m^2, ..., F_s^T, F_m^T]
        interleaved = []
        for t in range(self.T):
            interleaved.append(all_F_s[t])
            interleaved.append(all_F_m[t])

        # Stack along new dim: (batch, 2T, C_out, H, W)
        stacked = stack(interleaved, dim=1)

        # Compute weights: apply FC across the 2T dimension
        # Permute to (batch, C_out, H, W, 2T), apply FC, permute back
        B, K, C, H, W = stacked.shape    # K = 2T
        x = stacked.permute(0, 2, 3, 4, 1)        # (B, C, H, W, 2T)
        weights = sigmoid(self.converter_fc(x))     # (B, C, H, W, 2T)
        x_weighted = x * weights                    # element-wise
        F_prime = x_weighted.sum(dim=-1)            # (B, C, H, W)

        return F_prime

    def forward(self, received_all_S2, mask):
        """
        Full decoder forward pass.

        Args:
            received_all_S2: list of T tensors (noisy semantic info)
            mask: (batch, 1, H2, W2)
        Returns:
            F_prime: (batch, C_out, H, W) — reconstructed feature
        """
        self.if3.reset_state()
        self.ihf.reset_state()

        all_F_s = []
        all_F_m = []

        for t in range(self.T):
            S2_t_full = self.zero_fill(received_all_S2[t], mask)
            F_s_t, F_m_t = self.reconstruct_single_step(S2_t_full)
            all_F_s.append(F_s_t)
            all_F_m.append(F_m_t)

        F_prime = self.convert(all_F_s, all_F_m)
        return F_prime
```

---

## 5. Training Algorithm

```python
def train_SpikeAdaptSC():
    """
    Three-step training strategy (same structure as SNN-SC).
    """

    # ============================================================
    # STEP 1: Train backbone (ResNet50 or CCNet) standalone
    # ============================================================
    backbone = ResNet50()
    train_backbone(backbone, dataset, epochs=100, lr=1e-4, loss=CrossEntropy)
    # After training: split backbone at layer 14
    vehicle_part = backbone.layers[0:14]    # feature extraction
    cloud_part   = backbone.layers[14:]     # task execution

    # Freeze both parts
    vehicle_part.freeze()
    cloud_part.freeze()

    # ============================================================
    # STEP 2: Train SpikeAdapt-SC with frozen backbone
    # ============================================================
    encoder = SpikeAdaptEncoder(C_in=2048, C1=256, C2=128)
    decoder = SpikeAdaptDecoder(C_out=2048, C1=256, C2=128, T=8)
    entropy_est = SpikeRateEntropyEstimator()
    block_mask = BlockMask(eta=0.5, temperature=0.1)
    channel = BSC_Channel()  # or BEC_Channel()

    optimizer = Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()),    # includes IHF's V_th
        lr=1e-4
    )

    for epoch in range(50):
        for batch in dataloader:
            images, labels = batch

            # --- FORWARD PASS ---

            # 1. Extract features (frozen)
            F = vehicle_part(images)                    # (B, 2048, 4, 4)

            # 2. Encode: run T time steps
            all_S2 = encoder.forward_all_steps(F, T=8)  # list of 8 tensors

            # 3. Compute entropy and mask
            entropy_map, firing_rate = entropy_est.forward(all_S2)
            mask, tx_rate = block_mask.forward(entropy_map, training=True)

            # 4. Apply mask and transmit through channel
            p = uniform(0, 0.3)    # random BER for training robustness
            received_all_S2 = []
            for t in range(T):
                masked = block_mask.apply_mask(all_S2[t], mask)
                noisy = channel.forward(masked, bit_error_rate=p)
                received_all_S2.append(noisy)

            # 5. Decode
            F_prime = decoder.forward(received_all_S2, mask)

            # 6. Task execution (frozen)
            predictions = cloud_part(F_prime)

            # --- COMPUTE LOSS ---
            L_CE = CrossEntropyLoss(predictions, labels)
            L_entropy = entropy_loss(all_S2, alpha=1.0)
            L_rate = rate_loss(tx_rate, target_rate=0.75)

            L_total = L_CE + lambda_1 * L_entropy + lambda_2 * L_rate

            # --- BACKWARD PASS ---
            optimizer.zero_grad()
            L_total.backward()    # surrogate gradients through IF/IHF neurons
            optimizer.step()

    # ============================================================
    # STEP 3: Joint fine-tuning (unfreeze everything)
    # ============================================================
    vehicle_part.unfreeze()
    cloud_part.unfreeze()

    optimizer_finetune = Adam(
        list(vehicle_part.parameters()) +
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(cloud_part.parameters()),
        lr=1e-5    # lower learning rate
    )

    for epoch in range(50):
        # Same forward/backward as Step 2, but with optimizer_finetune
        ...
```

---

## 6. Inference Algorithm

```python
def inference(image, vehicle_part, encoder, entropy_est, block_mask,
              channel, decoder, cloud_part, T, bit_error_rate):
    """
    Full inference pipeline.

    Args:
        image: input image
        T: number of time steps
        bit_error_rate: channel BER (known or estimated)
    Returns:
        prediction: classification/segmentation result
        stats: dict with transmission statistics
    """

    # ========== UAV/SATELLITE SIDE ==========

    # 1. Feature extraction
    F = vehicle_part(image)                          # (1, 2048, 4, 4)

    # 2. Encode for T time steps
    all_S2 = encoder.forward_all_steps(F, T)         # list of T binary tensors

    # 3. Compute entropy and generate mask
    entropy_map, firing_rate = entropy_est.forward(all_S2)
    mask, tx_rate = block_mask.forward(entropy_map, training=False)

    # 4. Apply mask
    masked_all_S2 = []
    for t in range(T):
        masked = block_mask.apply_mask(all_S2[t], mask)
        masked_all_S2.append(masked)

    # 5. Transmit through digital channel
    # In practice: only non-zero blocks are serialized and sent
    # Also transmit mask M (H2 × W2 bits overhead)
    bits_per_step = mask.sum().item() * C2    # nnz(M) × C2
    total_bits = T * bits_per_step + H2 * W2  # + mask overhead

    received_all_S2 = []
    for t in range(T):
        noisy = channel.forward(masked_all_S2[t], bit_error_rate)
        received_all_S2.append(noisy)

    # ========== GROUND STATION SIDE ==========

    # 6. Decode
    F_prime = decoder.forward(received_all_S2, mask)

    # 7. Task execution
    prediction = cloud_part(F_prime)

    # Statistics
    original_bits = C * H * W * 32    # 32-bit float original feature
    stats = {
        'total_bits_transmitted': total_bits,
        'compression_ratio': original_bits / total_bits,
        'transmission_rate': tx_rate.item(),
        'mask_density': mask.mean().item(),
    }

    return prediction, stats
```

---

## 7. Loss Functions

### 7.1 Task Loss (Cross-Entropy)

```python
def task_loss(predictions, labels):
    """Standard cross-entropy for classification, or per-pixel CE for segmentation."""
    return CrossEntropyLoss(predictions, labels)
```

### 7.2 Entropy Maximization Loss (from SNN-SC)

```python
def entropy_loss(all_S2, alpha=1.0):
    """
    Encourages 0s and 1s in semantic info to be equally distributed.
    This maximizes mutual information between channel input and output.

    Args:
        all_S2: list of T tensors, each (batch, C2, H2, W2) ∈ {0,1}
        alpha: target entropy (1.0 = maximum = equal 0s and 1s)
    Returns:
        loss: scalar
    """
    # Concatenate all semantic info
    all_bits = cat([s.flatten() for s in all_S2])

    # Estimate p0, p1 from frequencies
    p1 = all_bits.mean()          # probability of 1
    p0 = 1.0 - p1                 # probability of 0

    # Compute entropy
    eps = 1e-7
    H = -p0 * log2(p0 + eps) - p1 * log2(p1 + eps)

    # Loss: penalize deviation from target entropy
    L_entropy = (alpha - H) ** 2

    return L_entropy
```

### 7.3 Rate Regularization Loss (NEW)

```python
def rate_loss(transmission_rate, target_rate):
    """
    Encourages the average transmission rate to match target.
    Prevents the mask from being all-1s (no savings) or all-0s (no info).

    Args:
        transmission_rate: fraction of blocks transmitted (from block_mask)
        target_rate: desired average fraction (e.g., 0.75 for 25% savings)
    Returns:
        loss: scalar
    """
    L_rate = (transmission_rate - target_rate) ** 2
    return L_rate
```

### 7.4 Total Loss

```python
def total_loss(predictions, labels, all_S2, transmission_rate,
               alpha=1.0, target_rate=0.75, lambda_1=1.0, lambda_2=0.5):
    """
    Combined training objective.

    L_total = L_CE + λ_1 × L_entropy + λ_2 × L_rate
    """
    L_CE      = task_loss(predictions, labels)
    L_ent     = entropy_loss(all_S2, alpha)
    L_rt      = rate_loss(transmission_rate, target_rate)

    L_total = L_CE + lambda_1 * L_ent + lambda_2 * L_rt

    return L_total, {
        'L_CE': L_CE.item(),
        'L_entropy': L_ent.item(),
        'L_rate': L_rt.item(),
        'L_total': L_total.item()
    }
```

---

## 8. Dimension Walkthrough

### ResNet50 on CIFAR-100 (split at layer 14)

```
Input image:       (batch, 3, 32, 32)
After ResNet L1-14: F = (batch, 2048, 4, 4)     ← 2,048 × 4 × 4 = 32,768 floats = 1,048,576 bits

ENCODER:
  Conv1-BN:         (batch, 2048, 4, 4) → (batch, 256, 4, 4)
  IF1:              (batch, 256, 4, 4)  → (batch, 256, 4, 4)   binary
  Conv2-BN:         (batch, 256, 4, 4)  → (batch, 128, 4, 4)
  IF2:              (batch, 128, 4, 4)  → (batch, 128, 4, 4)   binary  ← semantic info S_t

ENTROPY ESTIMATION (after T=8 steps):
  Firing rate:      (batch, 4, 4)       ← mean over T=8 and C=128
  Entropy map:      (batch, 4, 4)       ← binary entropy per block

MASK:
  Block mask:       (batch, 1, 4, 4)    ← 16 spatial positions
  Suppose 12 of 16 blocks pass threshold (75% retention)

TRANSMISSION per time step:
  SNN-SC (fixed):   128 × 16 = 2,048 bits
  SpikeAdapt-SC:    128 × 12 = 1,536 bits

TOTAL TRANSMISSION (T=8):
  SNN-SC:           8 × 2,048 = 16,384 bits
  SpikeAdapt-SC:    8 × 1,536 + 16 = 12,304 bits   (25% savings)
  Compression ratio: 1,048,576 / 12,304 ≈ 85×

DECODER:
  Zero-fill:        (batch, 128, 4, 4)  ← zeros at masked positions
  Conv3-BN:         (batch, 128, 4, 4)  → (batch, 256, 4, 4)
  IF3:              (batch, 256, 4, 4)  → (batch, 256, 4, 4)   binary
  Conv4-BN:         (batch, 256, 4, 4)  → (batch, 2048, 4, 4)
  IHF:              (batch, 2048, 4, 4) → F_s (binary) + F_m (float)

CONVERTER:
  Inputs:           2 × 8 = 16 tensors of (batch, 2048, 4, 4)
  FCN + Sigmoid:    weights (16,) per spatial-channel element
  Output:           F' = (batch, 2048, 4, 4)  ← reconstructed feature

TASK EXECUTION:
  After ResNet L15+: (batch, 100)  ← classification logits
```

### CCNet on FoodSeg103 (split at backbone output)

```
Input image:        (batch, 3, 520, 520)
After backbone:     F = (batch, 2048, 65, 65)    ← much larger spatial dims

ENCODER:
  Conv1-BN:          (batch, 2048, 65, 65) → (batch, 256, 65, 65)
  IF1:               binary (batch, 256, 65, 65)
  Conv2-BN:          (batch, 256, 65, 65)  → (batch, 128, 65, 65)
  IF2:               binary (batch, 128, 65, 65)  ← semantic info

ENTROPY ESTIMATION:
  Firing rate:       (batch, 65, 65)
  Entropy map:       (batch, 65, 65)

MASK:
  Block mask:        (batch, 1, 65, 65)  ← 4,225 spatial positions
  Much more room for content-adaptive savings on large feature maps

TRANSMISSION per step:
  SNN-SC:            128 × 4,225 = 540,800 bits
  SpikeAdapt-SC:     varies per image (content-adaptive)

  Example: urban scene → 90% retention → 486,720 bits/step
  Example: open field → 60% retention → 324,480 bits/step
  Average ~75%: 405,600 bits/step (25% avg savings)

TOTAL (T=8):
  SNN-SC:            4,326,400 bits
  SpikeAdapt-SC:     ~3,249,025 bits avg (including mask overhead 4,225)
```

---

## 9. Key Implementation Notes

### 9.1 Straight-Through Estimator for Masking

The block mask involves a hard threshold which is non-differentiable. During training:
```
Forward:  mask = (entropy >= eta).float()           # hard 0/1
Backward: gradient flows through sigmoid((entropy - eta) / tau)  # soft approximation
```
This is the standard straight-through estimator (STE) trick, same concept used for the spiking neuron fire step.

### 9.2 Hyperparameter Selection for η

η controls the bandwidth-accuracy tradeoff:
- η = 0.0: transmit everything (equivalent to SNN-SC, no savings)
- η = 1.0: transmit nothing (all entropy < 1.0 unless f=0.5 exactly)
- η = 0.3-0.7: typical operating range

**Strategy:** Train multiple models with different η values (e.g., 0.3, 0.5, 0.7) and report the Pareto front of accuracy vs bandwidth.

Alternatively, the rate loss L_rate with target_rate parameter implicitly controls η's effect. You can fix η=0.5 and vary target_rate.

### 9.3 Two-Phase Forward on Encoder

The encoder must run ALL T time steps before the mask can be computed (because firing rate requires statistics over T). This means:

```
Phase 1: Run encoder T steps → collect {S2_1, ..., S2_T}
Phase 2: Compute entropy → generate mask → apply mask to all S2_t
Phase 3: Transmit masked S2_t through channel
```

This adds latency of T forward passes before any transmission begins. For real-time UAV scenarios, you could approximate by:
- Using firing rate from previous frame to predict mask for current frame
- Or using a sliding window of the first few time steps (e.g., T/2) to estimate the mask early

### 9.4 Surrogate Gradient

All IF and IHF neurons use sigmoid surrogate for backpropagation through the fire step:

```python
# Forward: heaviside (non-differentiable)
spikes = (membrane > V_th).float()

# Backward: sigmoid derivative as surrogate
# Implemented via autograd custom function:
class SurrogateSpike(autograd.Function):
    @staticmethod
    def forward(ctx, membrane, V_th):
        ctx.save_for_backward(membrane, V_th)
        return (membrane > V_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, V_th = ctx.saved_tensors
        sigmoid_val = torch.sigmoid(membrane - V_th)
        surrogate_grad = sigmoid_val * (1 - sigmoid_val)
        return grad_output * surrogate_grad, None
```

### 9.5 SpikingJelly Integration

The SNN-SC paper uses SpikingJelly framework. Your implementation should leverage it:

```python
from spikingjelly.activation_based import neuron, layer, functional

# IF neuron from SpikingJelly
if_neuron = neuron.IFNode(v_threshold=1.0, v_reset=None,  # None = soft reset
                           surrogate_function=surrogate.Sigmoid())

# For IHF: subclass IFNode to also output membrane potential
class IHFNode(neuron.IFNode):
    def __init__(self, v_threshold_init=1.0):
        super().__init__(v_threshold=v_threshold_init, v_reset=None,
                         surrogate_function=surrogate.Sigmoid())
        # Make threshold learnable
        self.v_threshold = nn.Parameter(torch.tensor(v_threshold_init))

    def forward(self, x):
        # Standard IF forward (charge, fire, reset)
        spikes = super().forward(x)
        # Also return membrane potential after reset
        mem_out = self.v.clone()
        return spikes, mem_out

# Reset all neurons between samples
functional.reset_net(model)
```
