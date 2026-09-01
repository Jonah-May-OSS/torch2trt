import pytest
import torch
from torch import nn

from torch2trt import torch2trt

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_flatten_nested_tuple_args():

    class TestModule(nn.Module):
        def forward(self, x, yz):
            return torch.cat([x, yz[0], yz[1]], dim=-1)

    module = TestModule().cuda().eval()

    data = (
        torch.randn(1, 3, 32, 32).cuda(),
        (torch.randn(1, 3, 32, 32).cuda(), torch.randn(1, 3, 32, 32).cuda()),
    )

    module_trt = torch2trt(module, data)

    out = module(*data)
    out_trt = module_trt(*data)

    assert torch.allclose(out, out_trt, atol=1e-3, rtol=1e-3)
