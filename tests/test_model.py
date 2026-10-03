import numpy as np
import pytest
import torch

from wafer.model import WaferCNN, resnet18_1ch, build


class TestWaferCNN:
    def test_forward_shape(self):
        """WaferCNN forward on (4,1,64,64) should give (4,9)."""
        model = WaferCNN(n_classes=9)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 9)

    def test_parameter_count(self):
        """Parameter count should be between 300k and 600k."""
        model = WaferCNN(n_classes=9)

        num_params = sum(p.numel() for p in model.parameters())
        assert 300_000 <= num_params <= 600_000


class TestResNet18:
    def test_forward_works(self):
        """resnet18_1ch forward should work."""
        pytest.importorskip("torchvision")

        model = resnet18_1ch(n_classes=9)
        x = torch.randn(4, 1, 64, 64)
        output = model(x)

        assert output.shape == (4, 9)


class TestBuild:
    def test_build_cnn(self):
        """build('cnn') should create WaferCNN."""
        model = build("cnn", n_classes=9)
        assert isinstance(model, WaferCNN)

        x = torch.randn(4, 1, 64, 64)
        output = model(x)
        assert output.shape == (4, 9)
