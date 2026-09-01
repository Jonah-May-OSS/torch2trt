import pytest
import torch

import torch2trt

# torchvision is not a torch2trt dependency; it only supplies the models
# under test. importorskip turns a missing optional dep into a skip, rather
# than a collection error -- and a collection error aborts the entire run.
torchvision = pytest.importorskip("torchvision")

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def _cross_validate_module(model, shape=(224, 224)):
    model = model.cuda().eval()
    data = torch.randn(1, 3, *shape).cuda()
    model_trt = torch2trt.torch2trt(model, [data])
    data = torch.randn(1, 3, *shape).cuda()
    out = model(data)
    out_trt = model_trt(data)
    assert torch.allclose(out, out_trt, rtol=1e-2, atol=1e-2)


def test_deeplabv3_resnet50():
    bb = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_deeplabv3_resnet101():
    bb = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_fcn_resnet50():
    bb = torchvision.models.segmentation.fcn_resnet50(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)


def test_fcn_resnet101():
    bb = torchvision.models.segmentation.fcn_resnet101(pretrained=False)
    model = ModelWrapper(bb)
    _cross_validate_module(model)
