import numpy as np
import torch
import pytest

from wafer.synth import (
    to_signed, from_signed, nearest_neighbour_distance, near_copy_fraction,
    memorization_report, classifier_features, frechet_distance, sample_grid
)
from wafer.model import WaferCNN

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestToSigned:
    def test_numpy_array_conversion(self):
        x = np.array([0, 1, 2], dtype=np.uint8)
        result = to_signed(x)
        expected = np.array([-1, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_torch_tensor_conversion(self):
        x = torch.tensor([0, 1, 2], dtype=torch.uint8)
        result = to_signed(x)
        expected = torch.tensor([-1, 0, 1], dtype=torch.int64)
        torch.testing.assert_close(result, expected.to(result.dtype))

    def test_2d_array_shape_preserved(self):
        x = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        result = to_signed(x)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, np.array([[-1, 0], [1, 0]]))


class TestFromSigned:
    def test_numpy_exact_values(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = from_signed(x)
        expected = np.array([0, 1, 2], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_torch_exact_values(self):
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float32)
        result = from_signed(x)
        expected = torch.tensor([0, 1, 2], dtype=torch.uint8)
        torch.testing.assert_close(result, expected)

    def test_rounding_behavior(self):
        # -0.6 rounds to -1 -> 0
        # -0.4 rounds to 0 -> 1
        # 0.4 rounds to 0 -> 1
        # 0.6 rounds to 1 -> 2
        x = np.array([-0.6, -0.4, 0.4, 0.6], dtype=np.float32)
        result = from_signed(x)
        expected = np.array([0, 1, 1, 2], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_clipping_out_of_range(self):
        x = np.array([-3.0, 1.7], dtype=np.float32)
        result = from_signed(x)
        expected = np.array([0, 2], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_round_trip_numpy(self):
        original = np.array([0, 1, 2, 1, 0, 2], dtype=np.uint8).reshape(2, 3)
        signed = to_signed(original)
        recovered = from_signed(signed)
        np.testing.assert_array_equal(recovered, original)

    def test_round_trip_torch(self):
        original = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.uint8).reshape(2, 3)
        signed = to_signed(original)
        recovered = from_signed(signed)
        torch.testing.assert_close(recovered, original)


class TestNearestNeighbourDistance:
    def test_identical_query_in_reference(self):
        # Create minimal 64x64 maps with known pattern
        reference = np.zeros((2, 64, 64), dtype=np.uint8)
        reference[0, 0, 0] = 1
        queries = np.zeros((1, 64, 64), dtype=np.uint8)
        queries[0, 0, 0] = 1
        result = nearest_neighbour_distance(queries, reference, batch=512)
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_known_distance(self):
        # Create query with 8 pixel difference from closest reference
        # Reshape for 64x64 context (4096 pixels)
        reference = np.zeros((1, 64, 64), dtype=np.uint8)
        queries = np.zeros((1, 64, 64), dtype=np.uint8)
        queries[0, 0, 0:8] = 1

        result = nearest_neighbour_distance(queries, reference, batch=512)
        expected_distance = 8.0 / 4096
        np.testing.assert_allclose(result[0], expected_distance, rtol=1e-5)

    def test_batch_consistency(self):
        rng = np.random.default_rng(42)
        reference = rng.integers(0, 3, size=(5000, 64, 64), dtype=np.uint8)
        queries = rng.integers(0, 3, size=(10, 64, 64), dtype=np.uint8)

        result_batch1 = nearest_neighbour_distance(queries, reference, batch=1)
        result_batch512 = nearest_neighbour_distance(queries, reference, batch=512)

        np.testing.assert_allclose(result_batch1, result_batch512, rtol=1e-5)

    def test_reference_chunking(self):
        rng = np.random.default_rng(42)
        reference = rng.integers(0, 3, size=(5000, 64, 64), dtype=np.uint8)
        queries = rng.integers(0, 3, size=(2, 64, 64), dtype=np.uint8)

        result = nearest_neighbour_distance(queries, reference, batch=512)
        assert result.shape == (2,)
        assert np.all(result >= 0) and np.all(result <= 1)


class TestNearCopyFraction:
    def test_all_above_threshold(self):
        distances = np.array([0.5, 0.6, 0.7])
        result = near_copy_fraction(distances, 0.4)
        assert result == 0.0

    def test_all_below_threshold(self):
        distances = np.array([0.1, 0.2, 0.3])
        result = near_copy_fraction(distances, 0.4)
        assert result == 1.0

    def test_half_below_threshold(self):
        distances = np.array([0.1, 0.3, 0.5, 0.7])
        result = near_copy_fraction(distances, 0.4)
        assert result == 0.5


class TestMemorizationReport:
    def test_report_structure(self):
        gen_dist = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_dist = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        result = memorization_report(gen_dist, test_dist)

        assert isinstance(result, dict)
        assert "generated" in result
        assert "real_test" in result
        assert "threshold" in result
        assert "near_copy_fraction_generated" in result
        assert "near_copy_fraction_real_test" in result

    def test_threshold_is_1st_percentile_of_test(self):
        gen_dist = np.array([0.5] * 100)
        test_dist = np.arange(0.01, 1.01, 0.01)

        result = memorization_report(gen_dist, test_dist)
        expected_threshold = np.percentile(test_dist, 1)

        assert result["threshold"] == round(expected_threshold, 4)

    def test_rounding_to_4_decimals(self):
        gen_dist = np.array([0.123456, 0.234567])
        test_dist = np.array([0.012345, 0.023456])

        result = memorization_report(gen_dist, test_dist)

        assert result["generated"]["mean"] == 0.179
        assert result["generated"]["median"] == 0.179
        assert all(v == round(float(v), 4) for v in result["generated"].values())
        assert all(v == round(float(v), 4) for v in result["real_test"].values())


class TestClassifierFeatures:
    def test_output_shape(self):
        model = WaferCNN(n_classes=9)
        x = torch.randn(5, 1, 64, 64, dtype=torch.float32)
        device = torch.device("cpu")

        result = classifier_features(model, x, device, batch=1024)

        assert result.shape == (5, 256)
        assert result.dtype == np.float32

    def test_matches_manual_extraction(self):
        model = WaferCNN(n_classes=9)
        x = torch.randn(3, 1, 64, 64, dtype=torch.float32)
        device = torch.device("cpu")

        result = classifier_features(model, x, device, batch=1024)

        model.eval()
        with torch.no_grad():
            feat = model.features(x.to(device))
            feat = model.head[0](feat)
            feat = model.head[1](feat)
            expected = feat.cpu().numpy().astype(np.float32)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_batching_consistency(self):
        model = WaferCNN(n_classes=9)
        x = torch.randn(10, 1, 64, 64, dtype=torch.float32)
        device = torch.device("cpu")

        result_batch2 = classifier_features(model, x, device, batch=2)
        result_batch10 = classifier_features(model, x, device, batch=10)

        np.testing.assert_allclose(result_batch2, result_batch10, rtol=1e-5)


class TestFrechetDistance:
    def test_zero_for_identical_sets(self):
        a = np.random.randn(100, 10)
        b = a.copy()

        result = frechet_distance(a, b)

        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_symmetric(self):
        a = np.random.randn(50, 8)
        b = np.random.randn(50, 8)

        result_ab = frechet_distance(a, b)
        result_ba = frechet_distance(b, a)

        np.testing.assert_allclose(result_ab, result_ba, rtol=1e-5)

    def test_shifted_sets_mean_difference_equals_covariance(self):
        # When covariances are identical, FD = ||mu_a - mu_b||^2
        a = np.random.randn(200, 5)
        b = a + 2.0

        result = frechet_distance(a, b)
        expected = np.sum((a.mean(0) - b.mean(0)) ** 2)

        np.testing.assert_allclose(result, expected, rtol=1e-3)

    def test_positive_for_different_sets(self):
        a = np.random.randn(100, 5)
        b = np.random.randn(100, 5) + 5.0

        result = frechet_distance(a, b)

        assert result > 0


class TestSampleGrid:
    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_creates_png_file(self, tmp_path):
        real = {
            "Center": np.random.randint(0, 3, (3, 64, 64), dtype=np.uint8),
            "Donut": np.random.randint(0, 3, (3, 64, 64), dtype=np.uint8),
        }
        generated = {
            "Center": np.random.randint(0, 3, (3, 64, 64), dtype=np.uint8),
            "Donut": np.random.randint(0, 3, (3, 64, 64), dtype=np.uint8),
        }

        output_path = tmp_path / "sample_grid.png"
        sample_grid(real, generated, ["Center", "Donut"], per_class=3, path=str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
    def test_single_class(self, tmp_path):
        real = {"Class1": np.random.randint(0, 3, (2, 64, 64), dtype=np.uint8)}
        generated = {"Class1": np.random.randint(0, 3, (2, 64, 64), dtype=np.uint8)}

        output_path = tmp_path / "single_class.png"
        sample_grid(real, generated, ["Class1"], per_class=2, path=str(output_path))

        assert output_path.exists()
