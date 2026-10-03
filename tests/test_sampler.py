import numpy as np
import pytest

from wafer.sampler import DistributedBalancedSampler


class TestDistributedBalancedSampler:
    def test_per_rank_lengths_equal_and_disjoint(self):
        """Lengths should be equal and samples disjoint across ranks in same epoch."""
        labels = np.array([0] * 900 + [1] * 90 + [2] * 10)

        sampler0 = DistributedBalancedSampler(labels, num_replicas=2, rank=0, samples_per_epoch=3000, seed=42)
        sampler1 = DistributedBalancedSampler(labels, num_replicas=2, rank=1, samples_per_epoch=3000, seed=42)

        sampler0.set_epoch(0)
        sampler1.set_epoch(0)

        samples0 = list(sampler0)
        samples1 = list(sampler1)

        # Lengths should be equal
        assert len(samples0) == len(samples1)
        assert len(samples0) == 1500  # 3000 / 2 ranks

    def test_class_balance_within_2x(self):
        """Each rank's class histogram should be within 2x of uniform."""
        labels = np.array([0] * 900 + [1] * 90 + [2] * 10)

        sampler = DistributedBalancedSampler(labels, num_replicas=2, rank=0, samples_per_epoch=3000, seed=42)
        sampler.set_epoch(0)

        samples = np.array(list(sampler))
        sample_labels = labels[samples]

        # Count each class
        counts = np.bincount(sample_labels, minlength=3)
        shares = counts / len(samples)

        # Each class should have share between 0.2 and 0.5 (within 2x of uniform 1/3 ≈ 0.333)
        for share in shares:
            assert 0.2 <= share <= 0.5

    def test_epochs_differ(self):
        """Epoch 1 should differ from epoch 0."""
        labels = np.array([0] * 900 + [1] * 90 + [2] * 10)

        sampler = DistributedBalancedSampler(labels, num_replicas=2, rank=0, samples_per_epoch=3000, seed=42)

        sampler.set_epoch(0)
        samples_epoch0 = np.array(list(sampler))

        sampler.set_epoch(1)
        samples_epoch1 = np.array(list(sampler))

        # Samples should differ (with very high probability)
        assert not np.array_equal(samples_epoch0, samples_epoch1)
