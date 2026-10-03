"""A compact CNN for 64x64 single-channel wafer maps, and a 1-channel ResNet-18 option."""

from __future__ import annotations

import torch
from torch import nn


class WaferCNN(nn.Module):
    """Four conv blocks (32-64-128-256), batch norm, ReLU, 2x2 max-pool, global average pool,
    one linear layer. About 0.4M parameters: small enough to train a few epochs on a T4 in
    minutes, large enough to separate the nine WM-811K patterns."""

    def __init__(self, n_classes: int = 9):
        super().__init__()
        chans = [1, 32, 64, 128, 256]
        layers: list[nn.Module] = []
        for cin, cout in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(256, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def resnet18_1ch(n_classes: int = 9) -> nn.Module:
    from torchvision.models import resnet18

    m = resnet18(weights=None, num_classes=n_classes)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return m


def build(name: str, n_classes: int = 9) -> nn.Module:
    if name == "cnn":
        return WaferCNN(n_classes)
    if name == "resnet18":
        return resnet18_1ch(n_classes)
    raise ValueError(name)
