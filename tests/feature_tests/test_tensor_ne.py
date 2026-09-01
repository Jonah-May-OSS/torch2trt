import pytest
import torch

from torch2trt import torch2trt, trt

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_tensor_ne():

    class NotEqual(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x != y

    module = NotEqual().cuda().eval()

    x = torch.randn(1, 3, 40, 20).cuda()
    y = torch.randn(1, 3, 1, 20).cuda()

    module_trt = torch2trt(module, [x, y], log_level=trt.Logger.VERBOSE)

    assert torch.all(module_trt(x, y) == module(x, y))


if __name__ == "__main__":
    test_tensor_ne()
