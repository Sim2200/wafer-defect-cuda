import numpy as np
import pytest

from wafer.metrics import confusion, report


class TestConfusion:
    def test_confusion_matrix_hand_computed(self):
        """Confusion matrix for y_true=[0,0,1,1,2,2], y_pred=[0,1,1,1,2,0]."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        cm = confusion(y_true, y_pred, n=3)

        expected = np.array([
            [1, 1, 0],  # true 0: 1 correct, 1 as class 1, 0 as class 2
            [0, 2, 0],  # true 1: 0 as class 0, 2 correct, 0 as class 2
            [1, 0, 1]   # true 2: 1 as class 0, 0 as class 1, 1 correct
        ])

        np.testing.assert_array_equal(cm, expected)


class TestReport:
    def test_report_hand_computed(self):
        """Report on hand-computed 3-class case."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 1, 1, 1, 2, 0])

        result = report(y_true, y_pred)

        # Hand-computed values:
        # Class 0 (Center): tp=1, support=2, predicted_total=2 -> precision=0.5, recall=0.5, f1=0.5
        # Class 1 (Donut): tp=2, support=2, predicted_total=3 -> precision=2/3≈0.6667, recall=1.0, f1=0.8
        # Class 2 (Edge-Loc): tp=1, support=2, predicted_total=1 -> precision=1.0, recall=0.5, f1≈0.6667
        # macro_f1 = (0.5 + 0.8 + 0.6667) / 3 ≈ 0.6556

        assert result["macro_f1"] == pytest.approx(0.6556, abs=0.001)
        assert result["per_class"]["Center"]["precision"] == 0.5
        assert result["per_class"]["Donut"]["recall"] == 1.0

        # Check confusion matrix is correct (9x9 with first 3x3 populated)
        for i in range(3):
            assert result["confusion"][i][:3] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]][i]

    def test_macro_f1_only_counts_present_classes(self):
        """Macro F1 should only average classes with support > 0."""
        # Only classes 0 and 1 are present
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])

        result = report(y_true, y_pred)

        # Both present classes have F1=1.0
        assert result["macro_f1"] == 1.0

        # Class 2 (none) has support 0
        assert result["per_class"]["Random"]["support"] == 0
