# SpikeAdapt-SC: System Architecture

## System Block Diagram

```mermaid
graph LR
    subgraph TX["Transmitter"]
        A["Input Image<br/>x ∈ ℝ³ˣ³²ˣ³²"] --> B["ResNet50<br/>Front<br/>(Layers 1-3)"]
        B --> C["Feature Map<br/>F ∈ ℝ¹⁰²⁴ˣ⁸ˣ⁸"]
        C --> D["SNN Encoder<br/>(IF Neurons)"]
        D --> E["Spike Trains<br/>S₂ ∈ {0,1}<br/>T×128×8×8"]
        E --> F["Importance<br/>Scorer"]
        F --> G["Block Mask<br/>M ∈ {0,1}⁸ˣ⁸"]
        G --> H["Masked Spikes<br/>S₂ ⊙ M"]
    end

    subgraph CH["Channel"]
        H --> I["BSC / AWGN /<br/>Rayleigh"]
        I --> J["Received<br/>Spikes"]
    end

    subgraph RX["Receiver"]
        J --> K["SNN Decoder<br/>(IHF Neurons)"]
        K --> L["Reconstructed<br/>Features F̂"]
        L --> M["ResNet50<br/>Back<br/>(Layer 4)"]
        M --> N["Classification<br/>ŷ ∈ ℝ²⁰⁰"]
    end

    style TX fill:#1a1a2e,color:#e0e0e0,stroke:#16213e
    style CH fill:#0f3460,color:#e0e0e0,stroke:#16213e
    style RX fill:#1a1a2e,color:#e0e0e0,stroke:#16213e
```

---

## Detailed Encoder Architecture

```mermaid
graph TD
    subgraph ENC["SNN Encoder (T=8 timesteps)"]
        F["Feature F<br/>1024×8×8"] --> C1["Conv2d 3×3<br/>1024→256"]
        C1 --> BN1["BatchNorm2d"]
        BN1 --> IF1["IF Neuron<br/>θ=1.0"]
        IF1 --> S1["Spikes S₁<br/>256×8×8"]
        S1 --> C2["Conv2d 3×3<br/>256→128"]
        C2 --> BN2["BatchNorm2d"]
        BN2 --> IF2["IF Neuron<br/>θ=1.0"]
        IF2 --> S2["Spikes S₂<br/>128×8×8"]
    end

    subgraph MEM["Membrane State (across T)"]
        IF1 -. "m₁(t) = m₁(t-1) + x - spike·θ" .-> IF1
        IF2 -. "m₂(t) = m₂(t-1) + x - spike·θ" .-> IF2
    end

    style ENC fill:#1e3a5f,color:#e0e0e0,stroke:#16213e
    style MEM fill:#2d2d44,color:#b0b0b0,stroke:#16213e
```

---

## Importance Scoring & Masking

```mermaid
graph LR
    subgraph ENTROPY["Option A: Spike-Rate Entropy"]
        S["S₂ (T×B×128×8×8)"] --> AVG1["Mean over T<br/>→ firing rate r"]
        AVG1 --> H["H(r) = -r·log₂(r)<br/>-(1-r)·log₂(1-r)"]
        H --> TH["Threshold<br/>H(r) ≥ η"]
        TH --> M1["Mask M<br/>8×8 binary"]
    end

    subgraph LEARNED["Option B: Learned Scorer"]
        S2["S₂ (T×B×128×8×8)"] --> AVG2["Mean over T"]
        AVG2 --> CONV1["Conv2d 1×1<br/>128→32 + ReLU"]
        CONV1 --> CONV2["Conv2d 1×1<br/>32→1 + Sigmoid"]
        CONV2 --> IMP["Importance<br/>scores ∈ (0,1)⁸ˣ⁸"]
        IMP --> TOPK["Top-k Selection<br/>(k = rate × 64)"]
        TOPK --> M2["Mask M<br/>8×8 binary"]
    end

    style ENTROPY fill:#2d4a22,color:#e0e0e0,stroke:#16213e
    style LEARNED fill:#4a2d22,color:#e0e0e0,stroke:#16213e
```

---

## Decoder Architecture

```mermaid
graph TD
    subgraph DEC["SNN Decoder (T=8 timesteps)"]
        R["Received S₂·M"] --> C3["Conv2d 3×3<br/>128→256"]
        C3 --> BN3["BatchNorm2d"]
        BN3 --> IF3["IF Neuron<br/>θ=1.0"]
        IF3 --> S3["Spikes S₃"]
        S3 --> C4["Conv2d 3×3<br/>256→1024"]
        C4 --> BN4["BatchNorm2d"]
        BN4 --> IHF["IHF Neuron<br/>θ learned"]
        IHF --> S4["Spikes S₄"]
        IHF --> M4["Membrane m₄"]
    end

    subgraph CONV["Spike-to-Feature Converter"]
        S4 --> STACK["Stack all T:<br/>[S₄(1)..S₄(T), m₄(1)..m₄(T)]"]
        M4 --> STACK
        STACK --> FC["Linear 16→16<br/>+ Sigmoid gate"]
        FC --> SUM["Weighted Sum<br/>→ F̂ ∈ ℝ¹⁰²⁴ˣ⁸ˣ⁸"]
    end

    style DEC fill:#1e3a5f,color:#e0e0e0,stroke:#16213e
    style CONV fill:#3a2d5f,color:#e0e0e0,stroke:#16213e
```

---

## Channel Models

```mermaid
graph TD
    subgraph CHANNELS["Physical Channel Models"]
        direction TB
        subgraph BSC["BSC (Binary Symmetric)"]
            B1["x ∈ {0,1}"] --> B2["Flip with<br/>prob = BER"]
            B2 --> B3["y = x ⊕ flip"]
        end
        subgraph AWGN["AWGN"]
            A1["x ∈ {0,1}"] --> A2["BPSK: s = 2x-1"]
            A2 --> A3["y = s + n<br/>n ~ N(0, σ²)"]
            A3 --> A4["Hard decision<br/>x̂ = (y>0)"]
        end
        subgraph RAY["Rayleigh Fading"]
            R1["x ∈ {0,1}"] --> R2["BPSK: s = 2x-1"]
            R2 --> R3["y = h·s + n<br/>|h| ~ Rayleigh"]
            R3 --> R4["Equalize: y/|h|"]
            R4 --> R5["Hard decision"]
        end
    end

    style BSC fill:#2d3a22,color:#e0e0e0,stroke:#16213e
    style AWGN fill:#3a2d22,color:#e0e0e0,stroke:#16213e
    style RAY fill:#22293a,color:#e0e0e0,stroke:#16213e
```

---

## Training Pipeline

```mermaid
graph TD
    subgraph S1["Step 1: Backbone (100 epochs)"]
        T1["Train ResNet50<br/>end-to-end on<br/>CIFAR-100 / TinyIN"]
        T1 --> T1R["Freeze weights<br/>Split at Layer3"]
    end

    subgraph S2["Step 2: SNN Channel (60 epochs)"]
        T2["Train Encoder +<br/>Importance + Decoder"]
        T2B["BER-weighted sampling<br/>50% from [0.15, 0.4]<br/>50% from [0, 0.15]"]
        T2L["Loss = CE +<br/>λ_rate · |tx - target|"]
        T2 --> T2B
        T2B --> T2L
    end

    subgraph S3["Step 3: Fine-tune (40 epochs)"]
        T3["Unfreeze back +<br/>joint fine-tune"]
        T3G["Gradient accumulation<br/>(2 steps, BS=32)"]
        T3 --> T3G
    end

    S1 --> S2 --> S3

    style S1 fill:#2d2d44,color:#e0e0e0,stroke:#16213e
    style S2 fill:#1e3a5f,color:#e0e0e0,stroke:#16213e
    style S3 fill:#3a1e5f,color:#e0e0e0,stroke:#16213e
```

---

## Adaptive Bandwidth Control

```mermaid
graph LR
    subgraph ADAPT["Runtime Bandwidth Adaptation"]
        IMG["Input Image"] --> SCORE["Importance<br/>Scorer"]
        SCORE --> DECIDE{"Channel<br/>Quality?"}
        DECIDE -->|"Good (low BER)"| HIGH["Rate = 50%<br/>32/64 blocks<br/>~33K bits"]
        DECIDE -->|"Medium"| MED["Rate = 75%<br/>48/64 blocks<br/>~49K bits"]
        DECIDE -->|"Poor (high BER)"| LOW["Rate = 100%<br/>64/64 blocks<br/>~65K bits"]

        HIGH --> ACC1["73.83% acc<br/>50% BW saved"]
        MED --> ACC2["74.68% acc<br/>25% BW saved"]
        LOW --> ACC3["74.68% acc<br/>0% saved"]
    end

    style ADAPT fill:#1a2a3a,color:#e0e0e0,stroke:#16213e
```

---

## Tensor Dimensions Through the Pipeline

```
Input:          (B, 3, 32, 32)     — RGB image
                        ↓ ResNet Front (L1→L2→L3)
Features:       (B, 1024, 8, 8)    — 64 spatial blocks
                        ↓ SNN Encoder (×T timesteps)
Spikes:         (T, B, 128, 8, 8)  — binary spike trains
                        ↓ Importance Scorer
Importance:     (B, 8, 8)          — per-block scores
                        ↓ Block Mask (top-k or threshold)
Mask:           (B, 1, 8, 8)       — binary mask
                        ↓ Mask × Spikes
Masked:         (T, B, 128, 8, 8)  — masked spikes → CHANNEL
                        ↓ BSC / AWGN / Rayleigh
Received:       (T, B, 128, 8, 8)  — noisy spikes
                        ↓ SNN Decoder + Converter
Reconstructed:  (B, 1024, 8, 8)    — recovered features
                        ↓ ResNet Back (L4→FC)
Output:         (B, num_classes)    — classification logits
```

**Bits transmitted** = `T × C₂ × H × W × mask_rate` = `8 × 128 × 8 × 8 × 0.75` ≈ **49,152 bits**
