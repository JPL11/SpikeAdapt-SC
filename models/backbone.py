"""ResNet50 backbone split for semantic communication.

Splits ResNet50 at Layer3 boundary:
  - Front: Layers 1-3 (feature extractor, frozen at inference)
  - Back:  Layer 4 + FC (classifier head, fine-tuned)
"""

import torch
import torch.nn as nn
import torchvision


class ResNet50Front(nn.Module):
    """ResNet50 front-end: conv1 + layers 1-3.
    
    Output shape: (B, 1024, H, W)
      - CIFAR-100 (32×32 input):  H=W=8  → 64 spatial blocks
      - Tiny-ImageNet (64×64):    H=W=16 → 256 spatial blocks
    
    Args:
        input_size: Expected input spatial dimension. For ≤64, uses
                    smaller conv1 (3×3, stride 1) without maxpool.
        pool_size:  If set, applies AdaptiveAvgPool2d to reduce
                    spatial dims (e.g., 16→8 for Tiny-ImageNet).
    """
    def __init__(self, input_size=32, pool_size=None):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        if input_size <= 64:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = r.conv1
            self.maxpool = r.maxpool
        self.bn1 = r.bn1
        self.relu = r.relu
        self.layer1 = r.layer1
        self.layer2 = r.layer2
        self.layer3 = r.layer3
        self.spatial_pool = nn.AdaptiveAvgPool2d(pool_size) if pool_size else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(self.maxpool(x))))
        return self.spatial_pool(x)


class ResNet50Back(nn.Module):
    """ResNet50 back-end: layer4 + global pool + FC classifier."""
    def __init__(self, num_classes=100):
        super().__init__()
        r = torchvision.models.resnet50(weights=None)
        self.layer4 = r.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.fc(torch.flatten(self.avgpool(self.layer4(x)), 1))
