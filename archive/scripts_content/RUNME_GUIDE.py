# ============================================================================
# QUICK REFERENCE: What to Run and Why
# ============================================================================
#
# PRIORITY ORDER (for GLOBECOM April 1 deadline):
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ WEEK 1 (Mar 5-11): Baselines                                      │
# │                                                                     │
# │ ★ FIRST: Verify backbone accuracy (5 min)                          │
# │   python baselines_and_ablations.py                                │
# │   → Check "BACKBONE ACCURACY (no SC)" output                      │
# │   → If ~73%, your SpikeAdapt-SC is lossless (good)                │
# │   → If ~78%, you're losing 5% somewhere (investigate)             │
# │                                                                     │
# │ Then train these (each ~2-4 hours on V100):                        │
# │   1. SNN-SC (T=8)  — your direct predecessor                      │
# │   2. SNN-SC (T=6)  — bandwidth-fair comparison (key result!)       │
# │   3. CNN-Uni        — DNN with uniform quantization [17]           │
# │   4. CNN-Bern       — DNN with Bernoulli sampling [22]            │
# │   5. Random Mask    — ablation: random vs entropy-guided           │
# │   6. JPEG+Conv      — no training needed, runs at eval time       │
# │                                                                     │
# │ Expected story from BER sweep plot:                                │
# │   BER < 0.1:  All methods ~similar                                 │
# │   BER > 0.15: SpikeAdapt-SC > SNN-SC >> CNN-Uni, CNN-NonUni      │
# │   BER > 0.05: JPEG+Conv cliff effect (drops to random)            │
# │   Same BW:    SpikeAdapt-SC (T=8,η=0.5) > SNN-SC (T=6)          │
# │                                                                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │ WEEK 2 (Mar 12-18): Ablations + Analysis                          │
# │                                                                     │
# │ From the trained models, generate:                                 │
# │   - Table: Accuracy comparison at BER=0, 0.1, 0.2, 0.3           │
# │   - Table: Complexity (params, FLOPs) — already computed          │
# │   - Table: Bandwidth (bits transmitted, compression ratio)         │
# │   - Figure: BER sweep with ALL baselines on same plot             │
# │   - Figure: Ablation bar chart (entropy mask vs random mask)      │
# │                                                                     │
# ├─────────────────────────────────────────────────────────────────────┤
# │ WEEK 3-4 (Mar 19-Apr 1): Writing + Polish                         │
# └─────────────────────────────────────────────────────────────────────┘
#
#
# ============================================================================
# WHAT EACH BASELINE TESTS (for paper narrative)
# ============================================================================
#
# CNN-Uni [17]:
#   WHY: Shows that uniform quantization creates high-weight bits.
#         Bit errors in MSB cause huge reconstruction errors.
#   EXPECT: Good at BER=0, drops sharply at BER>0.1
#   PAPER: "errors in high-weight bits of the quantized information can
#           significantly degrade task performance" (SNN-SC, Section I)
#
# CNN-Bern [22]:
#   WHY: Bernoulli sampling treats all bits equally (like SNN), but
#         loses too much information in the sampling process.
#   EXPECT: Degrades slowly (like SNN-SC) but consistently lower accuracy
#   PAPER: "the Bernoulli sampling-based quantization method loses too
#           much information" (SNN-SC, Section IV-D)
#
# SNN-SC (T=8):
#   WHY: Direct predecessor. Same architecture, no masking.
#   EXPECT: ~same accuracy at BER=0, but uses 33% MORE bandwidth
#   KEY COMPARISON: SpikeAdapt-SC matches SNN-SC quality with less BW
#
# SNN-SC (T=6):
#   WHY: Bandwidth-fair comparison. T=6 sends ~12,288 bits ≈ your 12,304.
#   EXPECT: Lower accuracy than SpikeAdapt-SC at same bandwidth
#   KEY RESULT: Proves entropy-guided masking > naive timestep reduction
#   This is your STRONGEST argument for the paper.
#
# JPEG+Conv:
#   WHY: Traditional separate source & channel coding baseline.
#   EXPECT: Cliff effect at BER≈0.05, collapses at higher BER
#   PAPER: "JPEG+Conv suffers from the 'cliff effect'" (SNN-SC, Section IV-D)
#
# Random Mask (ablation):
#   WHY: Tests if entropy guidance matters vs random spatial dropout
#   EXPECT: Entropy mask > random mask (especially at higher η values)
#   If random ≈ entropy: your entropy estimator isn't adding value
#   If entropy >> random: proves the estimator finds informative blocks
#
#
# ============================================================================
# POTENTIAL ISSUES AND FIXES
# ============================================================================
#
# ISSUE: Your features might be (2048, 1, 1) not (2048, 4, 4)
#   Your CIFAR-100 ResNet50 uses conv1=3×3/s1 + no maxpool.
#   Input 32×32 → after 4 stride-2 layers → 2×2 or 1×1 spatial.
#   Check: print(front(dummy).shape)
#   If (B, 2048, 1, 1): masking on 1×1 is meaningless!
#   FIX: Split ResNet50 EARLIER (e.g., after layer3 instead of layer4)
#        layer3 output: (B, 1024, 4, 4) — gives 16 spatial blocks
#        layer2 output: (B, 512, 8, 8)  — gives 64 spatial blocks
#   This would change C_in but give meaningful spatial masking.
#
# ISSUE: CNN baselines might not train well with frozen backbone
#   If CNN-Uni or CNN-Bern accuracy is very low, try:
#   - Longer training (100 epochs)
#   - Lower learning rate (5e-5)
#   - Unfreezing the classification head too
#
# ISSUE: JPEG+Conv is slow (processes images one by one)
#   It's expected — this baseline is for evaluation only, not training.
#   Reduce test set size for quick debugging.
#
# ============================================================================
# PAPER TABLE TEMPLATE (fill in after experiments)
# ============================================================================
#
# Table: Classification Accuracy (%) on CIFAR-100 under BSC
#
# | Method              | BW (bits) | CR    | BER=0  | BER=0.1 | BER=0.2 | BER=0.3 |
# |---------------------|-----------|-------|--------|---------|---------|---------|
# | ResNet50 (no SC)    |     —     |  —    | XX.XX  |    —    |    —    |    —    |
# | CNN-Uni [17]        |  16,384   |  64×  | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
# | CNN-Bern [22]       |  16,384   |  64×  | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
# | JPEG+Conv           |  ~50,000  |  ~21× | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
# | SNN-SC (T=8)        |  16,384   |  64×  | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
# | SNN-SC (T=6)        |  12,288   |  85×  | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
# | SpikeAdapt-SC       |  12,304   |  85×  | 72.85  |  72.74  |  72.53  |  71.27  |
# |  w/ random mask     |  ~12,300  |  85×  | XX.XX  |  XX.XX  |  XX.XX  |  XX.XX  |
#
# Bold the best result in each BER column.
#
