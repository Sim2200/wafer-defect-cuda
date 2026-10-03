import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from wafer.data import label_of, resize_nearest, lot_grouped_split


class TestLabelOf:
    def test_string_valid(self):
        assert label_of("Center") == "Center"
        assert label_of("none") == "none"

    def test_string_invalid(self):
        assert label_of("Bogus") is None

    def test_array_nested(self):
        assert label_of(np.array([["Center"]])) == "Center"

    def test_empty_list(self):
        assert label_of([]) is None

    def test_empty_array(self):
        assert label_of(np.array([])) is None


class TestResizeNearest:
    def test_output_shape(self):
        wafer = np.ones((26, 26), dtype=np.uint8)
        resized = resize_nearest(wafer, size=64)
        assert resized.shape == (64, 64)
        assert resized.dtype == np.uint8

    def test_values_in_input(self, small_wafers):
        """Resized values should be a subset of input values."""
        for wafer in small_wafers:
            resized = resize_nearest(wafer, size=64)
            unique_input = set(np.unique(wafer))
            unique_output = set(np.unique(resized))
            assert unique_output.issubset(unique_input)

    def test_matches_torch_interpolate(self, small_wafers):
        """Compare with torch F.interpolate(mode='nearest-exact')."""
        for wafer in small_wafers:
            resized_np = resize_nearest(wafer, size=64)

            # Use torch to compute the reference
            wafer_tensor = torch.from_numpy(wafer).unsqueeze(0).unsqueeze(0).float()
            resized_torch = F.interpolate(wafer_tensor, size=(64, 64), mode="nearest-exact")
            resized_torch = resized_torch.squeeze(0).squeeze(0).numpy().astype(np.uint8)

            np.testing.assert_array_equal(resized_np, resized_torch)


class TestLotGroupedSplit:
    def test_synthetic_split(self):
        """Test lot_grouped_split on a synthetic DataFrame."""
        # Create 40 lots x 5 wafers = 200 wafers, 3 classes
        lots = [f"lot_{i}" for i in range(40)]
        wafers_per_lot = 5

        rows = []
        for lot_idx, lot in enumerate(lots):
            for wafer_idx in range(wafers_per_lot):
                # Distribute classes: lots 0-13 -> class 0, 14-26 -> class 1, 27-39 -> class 2
                if lot_idx < 14:
                    y = 0
                elif lot_idx < 27:
                    y = 1
                else:
                    y = 2

                rows.append({"lotName": lot, "y": y, "waferMap": np.zeros((32, 32), dtype=np.uint8), "label": f"class_{y}"})

        df = pd.DataFrame(rows)
        split = lot_grouped_split(df, val_frac=0.1, test_frac=0.2, seed=0)

        # Check disjoint
        train_set = set(split.train)
        val_set = set(split.val)
        test_set = set(split.test)
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

        # Check coverage
        union = train_set | val_set | test_set
        assert union == set(range(len(df)))

        # Check no lot spans two splits
        train_df = df.iloc[split.train]
        val_df = df.iloc[split.val]
        test_df = df.iloc[split.test]

        train_lots = set(train_df["lotName"])
        val_lots = set(val_df["lotName"])
        test_lots = set(test_df["lotName"])

        assert len(train_lots & val_lots) == 0
        assert len(train_lots & test_lots) == 0
        assert len(val_lots & test_lots) == 0

        # Check every class in train
        assert set(df.iloc[split.train]["y"]) == {0, 1, 2}
