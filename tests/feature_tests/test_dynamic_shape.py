import pytest
import tensorrt as trt
import torch
from torch import nn

from torch2trt import torch2trt
from torch2trt.dataset import ListDataset

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_dynamic_shape_conv2d():

    torch.manual_seed(0)

    module = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

    dataset = ListDataset()
    dataset.insert((torch.randn(1, 3, 224, 224).cuda(),))
    dataset.insert((torch.randn(1, 3, 64, 64).cuda(),))
    dataset.insert((torch.randn(1, 3, 128, 128).cuda(),))
    dataset.insert((torch.randn(4, 3, 32, 32).cuda(),))

    module_trt = torch2trt(module, dataset, log_level=trt.Logger.INFO)

    inputs = dataset[0]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[1]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[2]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[3]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)


def test_dynamic_shape_conv2d_onnx():

    torch.manual_seed(0)

    module = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1).cuda().eval()

    dataset = ListDataset()
    dataset.insert((torch.randn(1, 3, 224, 224).cuda(),))
    dataset.insert((torch.randn(1, 3, 64, 64).cuda(),))
    dataset.insert((torch.randn(1, 3, 128, 128).cuda(),))
    dataset.insert((torch.randn(4, 3, 32, 32).cuda(),))

    module_trt = torch2trt(module, dataset, use_onnx=True, log_level=trt.Logger.INFO)

    inputs = dataset[0]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[1]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[2]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)
    inputs = dataset[3]
    assert torch.allclose(module(*inputs), module_trt(*inputs), rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    test_dynamic_shape_conv2d()
