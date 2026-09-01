import pytest
import torch

from torch2trt import torch2trt, trt

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_div_constant_batch():

    class DivConstantBatch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("y", torch.ones((1, 3, 10, 10)))

        def forward(self, x):
            return x / self.y

    module = DivConstantBatch().cuda().eval()

    x = torch.randn(1, 3, 10, 10).cuda()

    module_trt = torch2trt(module, [x], log_level=trt.Logger.VERBOSE)

    assert torch.allclose(module_trt(x), module(x), atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    test_div_constant_batch()
