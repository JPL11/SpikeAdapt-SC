# ============================================================================
# ARCHITECTURAL AUDIT: Is SpikeAdapt-SC a Correct Neuromorphic SC / JSCC?
# ============================================================================
#
# Cross-referencing against:
#   [1] SNN-SC (Wang et al., IEEE TVT 2025) — our base architecture
#   [2] DHF-JSCC (Li et al., IEEE TCCN 2025) — inspiration for adaptive rate
#
# VERDICT: 6 issues found. 2 critical, 2 moderate, 2 minor.
#
# ============================================================================

"""
╔══════════════════════════════════════════════════════════════════════╗
║                    FRAMEWORK CLASSIFICATION                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  SNN-SC [1]:    "Semantic Communication for Collaborative          ║
║                  Intelligence" over DIGITAL channels (BSC/BEC)     ║
║                  - Task-oriented (classification accuracy)         ║
║                  - NOT reconstruction-oriented                     ║
║                  - Loss = CE (task) + entropy maximization         ║
║                                                                    ║
║  DHF-JSCC [2]:  "Joint Source-Channel Coding"                      ║
║                  over ANALOG channels (AWGN)                       ║
║                  - Reconstruction-oriented (PSNR/MS-SSIM)          ║
║                  - Loss = rate-distortion (MSE + entropy)          ║
║                                                                    ║
║  Ours:          "Adaptive Spiking Semantic Communication           ║
║                  for Edge Intelligence"                            ║
║                  - Task-oriented SC (like SNN-SC)                  ║
║                  - Digital channels (like SNN-SC)                  ║
║                  - Adaptive rate (inspired by DHF-JSCC)            ║
║                  - Neuromorphic encoder (SNN)                      ║
║                                                                    ║
║  ★ CORRECT TERM: This is SC (Semantic Communication), not JSCC.   ║
║    But the joint encoder-decoder optimization through the channel  ║
║    IS a form of joint source-channel coding implicitly.            ║
║    Paper framing: "SNN-based Semantic Communication" is safest.    ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║                ISSUE #1 [CRITICAL]: Feature Dimensions             ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Code comment (line 419): "→ (B, 2048, 1, 1) for CIFAR 32x32"    ║
║  Code comment (line 632): "Should be (2, 2048, 1, 1)"            ║
║                                                                    ║
║  BUT: With conv1=3×3/s1 and no maxpool, the actual strides are:   ║
║    conv1(s1) × Identity × layer2(s2) × layer3(s2) × layer4(s2)   ║
║    = 1 × 1 × 2 × 2 × 2 = 8 total stride                        ║
║    32 / 8 = 4 → features are (B, 2048, 4, 4)                     ║
║                                                                    ║
║  Your entropy visualizations CONFIRM 4×4 grids. So the actual     ║
║  dims ARE (2048, 4, 4), matching the SNN-SC paper exactly.        ║
║  The comments are just wrong.                                      ║
║                                                                    ║
║  SNN-SC paper: "the output of the 14th bottleneck block in        ║
║  ResNet50, which has feature dimensions of (2048, 4, 4)"          ║
║                                                                    ║
║  ✓ ACTUAL DIMS ARE CORRECT. Fix the misleading comments.          ║
║                                                                    ║
║  BUT: 4×4 = 16 spatial blocks is still very small for             ║
║  content-adaptive masking. Your compression stats show all         ║
║  images transmit ~75% = 12/16 blocks identically.                 ║
║                                                                    ║
║  FIX: For stronger adaptive results, split at layer3 instead:     ║
║    layer3 output: (B, 1024, 8, 8) → 64 spatial blocks            ║
║    More room for per-image variation in masking                    ║
║    Trade-off: edge device does less computation                    ║
║    (This is actually BETTER for UAV scenario — less edge compute)  ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║          ISSUE #2 [CRITICAL]: Channel Noise During Eval            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Your BSC_Channel.forward():                                       ║
║    if bit_error_rate <= 0 or not self.training:                    ║
║        return x                                                    ║
║                                                                    ║
║  This means: during model.eval(), NO channel noise is applied!    ║
║  The eval script works around this by doing:                       ║
║    spikeadapt.eval()                                              ║
║    spikeadapt.channel.train()  # Force channel noise on           ║
║                                                                    ║
║  This is FRAGILE and INCORRECT per the SNN-SC paper.              ║
║  The paper says: "the semantic communication process were          ║
║  repeated 10 times to mitigate the effect of randomness            ║
║  introduced by the digital channel"                                ║
║                                                                    ║
║  FIX: Channel should ALWAYS apply noise when BER > 0,             ║
║  regardless of training mode. The training flag should only        ║
║  control whether gradients flow through (STE):                     ║
║                                                                    ║
║    class BSC_Channel(nn.Module):                                   ║
║        def forward(self, x, bit_error_rate):                       ║
║            if bit_error_rate <= 0:                                  ║
║                return x                                            ║
║            flip = (torch.rand_like(x) < bit_error_rate).float()   ║
║            x_noisy = (x + flip) % 2                               ║
║            if self.training:                                       ║
║                return x + (x_noisy - x).detach()  # STE           ║
║            else:                                                   ║
║                return x_noisy  # No gradient needed                ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║    ISSUE #3 [MODERATE]: Missing Entropy Loss for SNN-SC Baseline   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  SNN-SC paper Eq 16: L_total = L_CE + L_entropy                   ║
║  Your baselines_and_ablations.py trains SNN-SC with ONLY L_CE.    ║
║                                                                    ║
║  The SNN-SC paper explicitly says: "The loss function is           ║
║  formulated by combining the cloud classification task loss and    ║
║  the entropy maximization loss" for training SNN-SC.               ║
║                                                                    ║
║  Only the CNN baselines use CE-only, because:                      ║
║  "Since the binarization process of quantization is                ║
║  non-differentiable, we cannot constrain the information           ║
║  entropy of the binarized bitstream in baselines"                  ║
║                                                                    ║
║  FIX: Add entropy loss when training SNN-SC baseline:              ║
║    L = L_CE + L_entropy  (for SNN-SC)                             ║
║    L = L_CE              (for CNN-Uni, CNN-Bern)                   ║
║    L = L_CE + L_entropy + L_rate  (for SpikeAdapt-SC)            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║     ISSUE #4 [MODERATE]: IHF Learnable Threshold Gradient          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  Your IHFNeuron:                                                   ║
║    spike = SpikeFunction.apply(membrane, self.threshold.item())    ║
║                                                                    ║
║  .item() converts the Parameter to a Python float, DETACHING       ║
║  it from the computation graph! The threshold gradient is ZERO.    ║
║  The threshold is NOT actually learning through backprop.           ║
║                                                                    ║
║  The fact it moved from 1.0 → 1.067 might be from:               ║
║  - The membrane subtraction: membrane - spike * self.threshold     ║
║    (this DOES have gradient w.r.t. self.threshold)                 ║
║  - But the fire step gradient is disconnected                      ║
║                                                                    ║
║  FIX: Pass the Parameter directly, modify SpikeFunction:           ║
║                                                                    ║
║  class SpikeFunction(torch.autograd.Function):                     ║
║      @staticmethod                                                 ║
║      def forward(ctx, membrane, threshold):                        ║
║          ctx.save_for_backward(membrane, threshold)                ║
║          return (membrane > threshold).float()                     ║
║                                                                    ║
║      @staticmethod                                                 ║
║      def backward(ctx, grad_output):                               ║
║          membrane, threshold = ctx.saved_tensors                   ║
║          scale = 10.0                                              ║
║          sig = torch.sigmoid(scale * (membrane - threshold))       ║
║          grad_membrane = grad_output * sig * (1 - sig) * scale     ║
║          # Gradient w.r.t. threshold: negative of membrane grad    ║
║          grad_threshold = -grad_membrane.sum()                     ║
║          return grad_membrane, grad_threshold                      ║
║                                                                    ║
║  class IHFNeuron(nn.Module):                                       ║
║      def forward(self, x, membrane=None):                          ║
║          ...                                                       ║
║          spike = SpikeFunction.apply(membrane, self.threshold)     ║
║          # NO .item()! Pass the Parameter directly.                ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║        ISSUE #5 [MINOR]: Encoder Input is Floating-Point           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  The SNN-SC paper acknowledges this:                               ║
║  "the inputs to the first layer (Conv1) and the converter (FCN)   ║
║  are floating-point features, so the operation number of these     ║
║  two parts are calculated based on [MAC operations]"               ║
║                                                                    ║
║  This means Conv1 uses full MAC (multiply-accumulate), not the     ║
║  low-power AC (accumulate-only) that SNN provides.                 ║
║                                                                    ║
║  Your code handles this correctly — Conv1 takes float backbone     ║
║  features and Conv2+ takes binary spike inputs.                    ║
║                                                                    ║
║  This is a HYBRID architecture (DNN + SNN), not fully neuromorphic.║
║  The paper should say "SNN-based" not "fully neuromorphic."        ║
║                                                                    ║
║  ✓ NO CODE FIX NEEDED. Just frame correctly in paper.             ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝


╔══════════════════════════════════════════════════════════════════════╗
║      ISSUE #6 [MINOR]: SC vs JSCC Terminology                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  SNN-SC paper: calls it "Semantic Communication" (SC)              ║
║  DHF-JSCC paper: calls it "Joint Source-Channel Coding" (JSCC)     ║
║                                                                    ║
║  The difference:                                                   ║
║  - SC: task-oriented, loss = task performance (CE accuracy)        ║
║  - JSCC: reconstruction-oriented, loss = distortion (MSE/PSNR)    ║
║                                                                    ║
║  Your system uses CE loss (task-oriented) → this is SC, not JSCC.  ║
║                                                                    ║
║  However, the joint optimization of encoder+decoder through        ║
║  a noisy channel IS implicit JSCC — source coding (compression)   ║
║  and channel coding (noise resilience) happen jointly in the       ║
║  same learned encoder/decoder.                                     ║
║                                                                    ║
║  RECOMMENDATION: Call it "Semantic Communication" in the paper.    ║
║  Say "which implicitly performs joint source-channel coding"       ║
║  in the introduction, but don't put JSCC in the title.            ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# EQUATION-BY-EQUATION VERIFICATION
# ============================================================================
"""
SNN-SC Paper Equations vs Your Code
====================================

Eq 4: I = E1(F; θ1)
  Paper: Conv1-BN module compresses feature F
  Code:  x = self.bn1(self.conv1(F))                          ✓ CORRECT

Eq 5: S1_t = IF1(I; m1_{t-1})
  Paper: IF neuron converts float to spikes using membrane state
  Code:  s1, mem_if1 = self.if1(x, mem_if1)                   ✓ CORRECT
  Note:  I is constant across timesteps (same F each step)      ✓ CORRECT per paper

Eq 6: S2_t = IF2(E2(S1_t; θ2); m2_{t-1})
  Paper: Conv2-BN + IF2 further compresses spikes
  Code:  x = self.bn2(self.conv2(s1))
         s2, mem_if2 = self.if2(x, mem_if2)                   ✓ CORRECT

Eq 7: Ŝ2_t = η(S2_t; p)
  Paper: Digital channel (BSC/BEC) corrupts bits
  Code:  noisy = self.channel(masked, bit_error_rate)          ✓ CORRECT
  NEW:   Mask applied BEFORE channel (not in original paper)    ✓ OUR CONTRIBUTION

Eq 8: S3_t = IF3(R1(Ŝ2_t; φ1); m3_{t-1})
  Paper: Conv3-BN + IF3 decompresses
  Code:  x = self.bn3(self.conv3(S2_t))
         s3, mem_if3 = self.if3(x, mem_if3)                   ✓ CORRECT

Eq 9: Fs_t, Fm_t = IHF(R2(S3_t; φ2); m4_{t-1})
  Paper: Conv4-BN + IHF outputs spike AND membrane potential
  Code:  x = self.bn4(self.conv4(s3))
         spike, mem_ihf = self.ihf(x, mem_ihf)                ✓ CORRECT
  IHF outputs: spike (binary) + membrane potential (float)      ✓ CORRECT

Eq 10: F' = FCN([F1s, F1m, ..., FTs, FTm]; ψ)
  Paper: Converter fuses 2T outputs via FC + Sigmoid weights
  Code:  interleaved → stack → FC → sigmoid → weighted sum     ✓ CORRECT

Eq 11a-d: IHF neuron steps (charge, fire, reset, output MP)
  Paper: m_t = m_{t-1} + I  (charge)
         S_t = 1 if m_t > Vth else 0  (fire)
         m_t = m_t - S_t * Vth  (soft reset)
         M_t = m_t  (output membrane potential)
  Code:  membrane = membrane + x
         spike = SpikeFunction.apply(membrane, threshold)
         membrane = membrane - spike * threshold
         return spike, membrane                                ✓ CORRECT

Eq 14: L_entropy = (α - H(p0, p1))^2
  Paper: Maximize entropy of semantic info to reach channel capacity
  Code:  H = -(p0*log2(p0) + p1*log2(p1))
         return (alpha - H) ** 2                               ✓ CORRECT

Eq 16: L_total = L_CE + L_entropy
  Paper: Combined loss for SNN-SC training
  Code:  L_total = L_CE + λ1*L_entropy + λ2*L_rate            ✓ EXTENDED (+ rate loss)

Training Strategy:
  Paper: 3 steps — train backbone, train SC (frozen backbone), fine-tune all
  Code:  Step 1 → Step 2 → Step 3                             ✓ CORRECT

Surrogate Gradient:
  Paper: Sigmoid function σ(x) = (1 + e^{-x})^{-1}
  Code:  sig = torch.sigmoid(scale * (membrane - threshold))   ✓ CORRECT
  Note:  scale=10.0 is a common choice, paper doesn't specify scale


SUMMARY: 12/12 equations correctly implemented.
         + 3 new contributions (entropy masking, rate loss, learnable IHF)
         - 2 bugs found (channel eval mode, IHF threshold gradient)
"""


# ============================================================================
# THE FIXES — COPY THESE INTO YOUR CODE
# ============================================================================

import torch
import torch.nn as nn
import math

# ============================================================================
# FIX 1: Corrected BSC Channel (always apply noise when BER > 0)
# ============================================================================
class BSC_Channel_Fixed(nn.Module):
    """
    Binary Symmetric Channel — FIXED version.
    Always applies noise when BER > 0, regardless of training mode.
    Training mode only controls whether STE gradient flows.
    """
    def forward(self, x, bit_error_rate):
        if bit_error_rate <= 0:
            return x

        # ALWAYS flip bits (even during eval)
        flip_mask = (torch.rand_like(x.float()) < bit_error_rate).float()
        x_noisy = (x + flip_mask) % 2

        if self.training:
            # STE: forward uses noisy, backward pretends identity
            return x + (x_noisy - x).detach()
        else:
            # Eval: just return noisy (no gradient needed)
            return x_noisy


class BEC_Channel_Fixed(nn.Module):
    """Binary Erasure Channel — FIXED version."""
    def forward(self, x, erasure_rate):
        if erasure_rate <= 0:
            return x

        erase_mask = (torch.rand_like(x.float()) < erasure_rate).float()
        random_bits = (torch.rand_like(x.float()) > 0.5).float()
        x_noisy = x * (1 - erase_mask) + random_bits * erase_mask

        if self.training:
            return x + (x_noisy - x).detach()
        else:
            return x_noisy


# ============================================================================
# FIX 2: Corrected Spike Function with threshold gradient
# ============================================================================
class SpikeFunctionFixed(torch.autograd.Function):
    """
    Heaviside forward, sigmoid surrogate backward.
    FIXED: Supports gradient w.r.t. threshold parameter.
    """
    @staticmethod
    def forward(ctx, membrane, threshold):
        # threshold can be a scalar tensor (nn.Parameter) or a float
        if isinstance(threshold, (int, float)):
            threshold = torch.tensor(float(threshold), device=membrane.device)
        ctx.save_for_backward(membrane, threshold)
        return (membrane > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, threshold = ctx.saved_tensors
        scale = 10.0
        sig = torch.sigmoid(scale * (membrane - threshold))
        surrogate_grad = sig * (1 - sig) * scale

        # Gradient w.r.t. membrane
        grad_membrane = grad_output * surrogate_grad

        # Gradient w.r.t. threshold (chain rule: d/dθ H(m-θ) = -d/dm H(m-θ))
        grad_threshold = -(grad_output * surrogate_grad).sum()

        return grad_membrane, grad_threshold


class IFNeuron_Fixed(nn.Module):
    """IF neuron — uses fixed SpikeFunctionFixed."""
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold  # Fixed (not learnable) for IF

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        spike = SpikeFunctionFixed.apply(membrane, self.threshold)
        membrane = membrane - spike * self.threshold
        return spike, membrane


class IHFNeuron_Fixed(nn.Module):
    """
    IHF neuron — FIXED: threshold is nn.Parameter with proper gradient.
    No more .item() detaching the computation graph!
    """
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(threshold)))

    def forward(self, x, membrane=None):
        if membrane is None:
            membrane = torch.zeros_like(x)
        membrane = membrane + x
        # Pass Parameter directly — NO .item()!
        spike = SpikeFunctionFixed.apply(membrane, self.threshold)
        membrane = membrane - spike * self.threshold
        return spike, membrane  # Both outputs


# ============================================================================
# FIX 3: SNN-SC baseline must use entropy loss
# ============================================================================
def train_snnsc_baseline(model, front, back, train_loader, test_loader,
                         model_name, epochs=50, lr=1e-4, ber_max=0.3,
                         lambda_entropy=1.0, alpha=1.0,
                         save_dir="./snapshots_baselines/"):
    """
    Training loop for SNN-SC baseline.
    FIXED: Uses L_CE + L_entropy (per SNN-SC paper Eq 16).
    CNN baselines should use train_cnn_baseline() with L_CE only.
    """
    import os, random
    from tqdm import tqdm

    os.makedirs(save_dir, exist_ok=True)
    front.eval()
    for p in front.parameters():
        p.requires_grad = False
    for p in back.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        pbar = tqdm(train_loader, desc=f"{model_name} E{epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(next(model.parameters()).device), \
                             labels.to(next(model.parameters()).device)
            ber = random.uniform(0, ber_max)

            with torch.no_grad():
                feat = front(images)

            F_prime, all_S2, stats = model(feat, bit_error_rate=ber)
            outputs = back(F_prime)

            # SNN-SC uses BOTH task loss AND entropy loss (Eq 16)
            L_CE = criterion(outputs, labels)
            L_entropy = compute_entropy_loss_fn(all_S2, alpha=alpha)
            loss = L_CE + lambda_entropy * L_entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'acc': f'{100.*pred.eq(labels).sum().item()/labels.size(0):.0f}%'})

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            acc = evaluate_snnsc(model, front, back, test_loader, ber=0.0)
            print(f"  {model_name} Test: {acc:.2f}%", flush=True)
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           os.path.join(save_dir, f"{model_name}_best.pth"))

    return best_acc


def compute_entropy_loss_fn(all_S2, alpha=1.0):
    all_bits = torch.cat([s.flatten() for s in all_S2])
    p1 = all_bits.mean()
    p0 = 1.0 - p1
    eps = 1e-7
    p0c = torch.clamp(p0, eps, 1 - eps)
    p1c = torch.clamp(p1, eps, 1 - eps)
    H = -(p0c * torch.log2(p0c) + p1c * torch.log2(p1c))
    return (alpha - H) ** 2


def evaluate_snnsc(model, front, back, test_loader, ber=0.0):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            dev = next(model.parameters()).device
            images, labels = images.to(dev), labels.to(dev)
            feat = front(images)
            F_prime, _, _ = model(feat, bit_error_rate=ber)
            outputs = back(F_prime)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100. * correct / total


# ============================================================================
# FIX 4: Corrected SpikeAdaptSC with all fixes applied
# ============================================================================

class SpikeRateEntropyEstimator(nn.Module):
    """No changes needed — this is correct."""
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    def forward(self, all_S2):
        stacked = torch.stack(all_S2, dim=0)
        firing_rate = stacked.mean(dim=0).mean(dim=1)
        f = firing_rate.clamp(self.eps, 1.0 - self.eps)
        entropy_map = -f * torch.log2(f) - (1 - f) * torch.log2(1 - f)
        return entropy_map, firing_rate


class BlockMask(nn.Module):
    """No changes needed — STE implementation is correct."""
    def __init__(self, eta=0.5, temperature=0.1):
        super().__init__()
        self.eta = eta
        self.temperature = temperature
    def forward(self, entropy_map, training=True):
        if training:
            soft_mask = torch.sigmoid((entropy_map - self.eta) / self.temperature)
            hard_mask = (entropy_map >= self.eta).float()
            mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            mask = (entropy_map >= self.eta).float()
        mask = mask.unsqueeze(1)
        tx_rate = mask.mean()
        return mask, tx_rate
    def apply_mask(self, S2_t, mask):
        return S2_t * mask


class Encoder_Fixed(nn.Module):
    """Encoder with fixed IF neurons."""
    def __init__(self, C_in=2048, C1=256, C2=128):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C1, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(C1)
        self.if1 = IFNeuron_Fixed()
        self.conv2 = nn.Conv2d(C1, C2, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(C2)
        self.if2 = IFNeuron_Fixed()

    def forward(self, F, mem_if1=None, mem_if2=None):
        x = self.bn1(self.conv1(F))
        s1, mem_if1 = self.if1(x, mem_if1)
        x = self.bn2(self.conv2(s1))
        s2, mem_if2 = self.if2(x, mem_if2)
        return s2, mem_if1, mem_if2


class Decoder_Fixed(nn.Module):
    """Decoder with fixed IHF neuron (proper threshold gradient)."""
    def __init__(self, C_out=2048, C1=256, C2=128, T=8):
        super().__init__()
        self.T = T
        self.conv3 = nn.Conv2d(C2, C1, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(C1)
        self.if3 = IFNeuron_Fixed()
        self.conv4 = nn.Conv2d(C1, C_out, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(C_out)
        self.ihf = IHFNeuron_Fixed()  # FIXED: proper gradient
        self.converter_fc = nn.Linear(2 * T, 2 * T)

    def reconstruct_step(self, S2_t, mem_if3, mem_ihf):
        x = self.bn3(self.conv3(S2_t))
        s3, mem_if3 = self.if3(x, mem_if3)
        x = self.bn4(self.conv4(s3))
        spike, mem_ihf = self.ihf(x, mem_ihf)
        return spike, mem_ihf.clone(), mem_if3, mem_ihf

    def convert(self, all_F_s, all_F_m):
        interleaved = []
        for t in range(self.T):
            interleaved.append(all_F_s[t])
            interleaved.append(all_F_m[t])
        stacked = torch.stack(interleaved, dim=1)
        B, K, C, H, W = stacked.shape
        x = stacked.permute(0, 2, 3, 4, 1)
        weights = torch.sigmoid(self.converter_fc(x))
        F_prime = (x * weights).sum(dim=-1)
        return F_prime

    def forward(self, received_all_S2, mask):
        mem_if3, mem_ihf = None, None
        all_F_s, all_F_m = [], []
        for t in range(self.T):
            S2_t = received_all_S2[t] * mask
            spike, mem_out, mem_if3, mem_ihf = self.reconstruct_step(
                S2_t, mem_if3, mem_ihf)
            all_F_s.append(spike)
            all_F_m.append(mem_out)
        return self.convert(all_F_s, all_F_m)


class SpikeAdaptSC_Fixed(nn.Module):
    """
    Complete SpikeAdapt-SC with ALL fixes applied:
    - Fixed channel (noise in eval mode)
    - Fixed IHF (proper threshold gradient)
    - Fixed spike function (threshold gradient)
    """
    def __init__(self, C_in=2048, C1=256, C2=128, T=8,
                 eta=0.5, temperature=0.1, channel_type='bsc'):
        super().__init__()
        self.T = T
        self.encoder = Encoder_Fixed(C_in, C1, C2)
        self.entropy_est = SpikeRateEntropyEstimator()
        self.block_mask = BlockMask(eta=eta, temperature=temperature)
        self.decoder = Decoder_Fixed(C_in, C1, C2, T)

        if channel_type == 'bsc':
            self.channel = BSC_Channel_Fixed()
        else:
            self.channel = BEC_Channel_Fixed()

    def forward(self, backbone_features, bit_error_rate=0.0, eta_override=None):
        # Phase 1: Encode T timesteps
        all_S2 = []
        mem_if1, mem_if2 = None, None
        for t in range(self.T):
            s2, mem_if1, mem_if2 = self.encoder(backbone_features, mem_if1, mem_if2)
            all_S2.append(s2)

        # Phase 2: Entropy estimation + masking
        entropy_map, firing_rate = self.entropy_est(all_S2)

        if eta_override is not None:
            old_eta = self.block_mask.eta
            self.block_mask.eta = eta_override
            mask, tx_rate = self.block_mask(entropy_map, training=False)
            self.block_mask.eta = old_eta
        else:
            mask, tx_rate = self.block_mask(entropy_map, training=self.training)

        # Phase 3: Mask + channel
        received_all_S2 = []
        for t in range(self.T):
            masked = self.block_mask.apply_mask(all_S2[t], mask)
            noisy = self.channel(masked, bit_error_rate)
            received_all_S2.append(noisy)

        # Phase 4: Decode
        F_prime = self.decoder(received_all_S2, mask)

        # Stats
        B, C2, H2, W2 = all_S2[0].shape
        C_in = backbone_features.shape[1]
        original_bits = C_in * backbone_features.shape[2] * backbone_features.shape[3] * 32
        nnz = mask.sum() / B
        transmitted_bits = self.T * nnz.item() * C2 + H2 * W2
        compression_ratio = original_bits / max(transmitted_bits, 1)

        stats = {
            'tx_rate': tx_rate.item(),
            'compression_ratio': compression_ratio,
            'mask_density': mask.mean().item(),
            'avg_entropy': entropy_map.mean().item(),
        }
        return F_prime, all_S2, tx_rate, stats


# ============================================================================
# VERIFICATION CHECKLIST
# ============================================================================
"""
Run this after applying fixes to verify everything works:

1. Gradient check for IHF threshold:
   model = SpikeAdaptSC_Fixed(...)
   ... forward pass ...
   loss.backward()
   print(f"IHF grad: {model.decoder.ihf.threshold.grad}")
   # Should be NON-ZERO (was zero before fix)

2. Channel noise in eval:
   model.eval()
   x = torch.ones(1, 128, 4, 4)
   y = model.channel(x, bit_error_rate=0.1)
   print(f"Bits flipped: {(x != y).sum().item()}")
   # Should be ~10% of total bits (was 0 before fix)

3. Entropy loss for SNN-SC baseline:
   # Train with: loss = L_CE + L_entropy
   # NOT just: loss = L_CE

4. Feature dimensions:
   dummy = torch.randn(1, 3, 32, 32).to(device)
   feat = front(dummy)
   print(f"Feature shape: {feat.shape}")
   # Should be (1, 2048, 4, 4) for CIFAR-100 with modified ResNet50
   # If (1, 2048, 1, 1), spatial masking is meaningless!
"""


# ============================================================================
# FRAMEWORK DIAGRAM (CORRECTED)
# ============================================================================
"""
CORRECT framework classification:

    ┌──────────────────────────────────────────────────────────────┐
    │              SEMANTIC COMMUNICATION (SC)                     │
    │         for Collaborative Intelligence (CI)                  │
    │                                                              │
    │  NOT "JSCC" — we optimize task performance, not              │
    │  reconstruction quality. But the end-to-end training         │
    │  through noisy channel IS implicit joint source-channel      │
    │  coding.                                                     │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │              HYBRID DNN-SNN ARCHITECTURE                     │
    │                                                              │
    │  NOT "fully neuromorphic" — Conv-BN layers are standard DNN. │
    │  Only IF/IHF neurons are spiking.                            │
    │  Conv1 takes float input (MAC operations).                   │
    │  Conv2-Conv4 take binary spike input (AC operations).        │
    │  Converter takes float input (MAC operations).               │
    │                                                              │
    │  Paper should say: "SNN-based" or "neuromorphic-inspired"    │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │              DIGITAL CHANNEL MODEL                           │
    │                                                              │
    │  BSC (Binary Symmetric Channel): bit flips with prob p       │
    │  BEC (Binary Erasure Channel): bit erasure with prob p       │
    │                                                              │
    │  NOT analog AWGN channel (that's DHF-JSCC).                  │
    │  SNN's binary output maps directly to digital channels.      │
    │  No quantization needed (unlike DNN-based SC).               │
    └──────────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────────┐
    │              WHAT WE ADD OVER SNN-SC                         │
    │                                                              │
    │  1. Spike-rate entropy estimation (zero-parameter)           │
    │  2. Block masking with STE (content-aware bandwidth saving)  │
    │  3. Rate regularization loss (L_rate)                        │
    │  4. Learnable IHF threshold (V_th as nn.Parameter)           │
    │                                                              │
    │  All 4 are compatible with the SNN-SC framework.             │
    │  The entropy estimation is inspired by DHF-JSCC's            │
    │  hyperprior model but uses spike statistics instead of       │
    │  a learned hyperprior encoder/decoder.                       │
    │  → Zero additional parameters                                │
    │  → Zero additional compute on edge device                    │
    └──────────────────────────────────────────────────────────────┘
"""
