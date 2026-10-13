"""The Triton kernels match the PyTorch references on real-shaped inputs (GPU + triton only)."""

import sys
from pathlib import Path

import pytest
import torch

triton = pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
import wafer_triton as T  # noqa: E402

from wafer import reference as R  # noqa: E402


@pytest.fixture(scope="module")
def maps():
    g = torch.Generator(device="cpu").manual_seed(0)
    # three-level wafer maps plus a few random floats, so ties and non-integer values are covered
    x = torch.randint(0, 3, (37, 64, 64), generator=g).to(torch.float32)
    x[:5] = torch.rand(5, 64, 64, generator=g) * 4 - 2
    return x.cuda()


@pytest.mark.parametrize("block_h", [8, 16, 32])
def test_median3x3_matches_reference(maps, block_h):
    assert torch.equal(T.median3x3(maps, block_h), R.median3x3(maps))


@pytest.mark.parametrize("block_h", [8, 16, 32])
def test_conv3x3_matches_reference(maps, block_h):
    w = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device="cuda") / 16
    out = T.conv3x3(maps, w, block_h)
    assert torch.allclose(out, R.conv3x3(maps, w), atol=1e-5)


def test_odd_sizes_and_borders():
    x = torch.rand(3, 50, 37, device="cuda")
    assert torch.equal(T.median3x3(x), R.median3x3(x))
    w = torch.rand(3, 3, device="cuda")
    assert torch.allclose(T.conv3x3(x, w), R.conv3x3(x, w), atol=1e-5)
