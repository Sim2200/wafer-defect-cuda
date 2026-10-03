import numpy as np
import pytest
import torch

from wafer.reference import conv3x3, median3x3, preprocess
from wafer.data import resize_nearest


class TestConv3x3:
    def test_constant_image_interior(self):
        """Conv3x3 with Gaussian kernel on constant image should give the constant in interior."""
        # Create a constant image
        x = torch.ones((1, 32, 32), dtype=torch.float32) * 5.0

        # Gaussian kernel (normalized so it sums to 1)
        gaussian = torch.tensor([
            [1., 4., 6., 4., 1.],
            [4., 16., 24., 16., 4.],
            [6., 24., 36., 24., 6.],
            [4., 16., 24., 16., 4.],
            [1., 4., 6., 4., 1.]
        ], dtype=torch.float32) / 256.0

        # For 3x3, use simplified Gaussian
        weight = torch.tensor([
            [1., 2., 1.],
            [2., 4., 2.],
            [1., 2., 1.]
        ], dtype=torch.float32) / 16.0

        result = conv3x3(x, weight)

        # Check interior region is close to 5.0
        interior = result[0, 1:-1, 1:-1]
        np.testing.assert_allclose(interior.numpy(), 5.0, atol=1e-5)


class TestMedian3x3:
    def test_removes_salt_noise(self):
        """Median3x3 should remove single-pixel spike."""
        # Create a constant image with one spike
        x = torch.ones((1, 5, 5), dtype=torch.float32) * 1.0
        x[0, 2, 2] = 100.0  # salt noise

        result = median3x3(x)

        # Check that the spike is removed/averaged
        assert result[0, 2, 2].item() == 1.0 or result[0, 2, 2].item() < 50.0

    def test_keeps_constant_region(self):
        """Median3x3 should keep a constant region unchanged."""
        x = torch.ones((1, 5, 5), dtype=torch.float32) * 3.0
        result = median3x3(x)

        np.testing.assert_allclose(result.numpy(), 3.0, atol=1e-5)


class TestPreprocess:
    def test_output_shape(self, small_wafers):
        """Preprocess should output (N, 64, 64)."""
        wafers_tensor = torch.stack([
            torch.from_numpy(resize_nearest(w, size=64)) for w in small_wafers
        ])
        result = preprocess(wafers_tensor, size=64)

        assert result.shape == (len(small_wafers), 64, 64)

    def test_standardization_per_wafer(self, small_wafers):
        """Preprocess output should have per-wafer mean ~0 and std ~1."""
        wafers_tensor = torch.stack([
            torch.from_numpy(resize_nearest(w, size=64)) for w in small_wafers
        ])
        result = preprocess(wafers_tensor, size=64)

        # Check per-wafer mean and std
        means = result.mean(dim=(1, 2))
        stds = result.std(dim=(1, 2), unbiased=False)

        np.testing.assert_allclose(means.numpy(), 0.0, atol=1e-3)
        np.testing.assert_allclose(stds.numpy(), 1.0, atol=1e-3)
