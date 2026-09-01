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


def _cross_validate_module(model, shape=(224, 224)):
    model = model.cuda().eval()
    data = torch.randn(1, 3, *shape).cuda()
    model_trt = torch2trt.torch2trt(model, [data])
    data = torch.randn(1, 3, *shape).cuda()
    out = model(data)
    out_trt = model_trt(data)
    assert torch.allclose(out, out_trt, rtol=1e-1, atol=1e-1)


def test_alexnet():
    model = torchvision.models.alexnet(pretrained=False)
    _cross_validate_module(model)


def test_squeezenet1_0():
    model = torchvision.models.squeezenet1_0(pretrained=False)
    _cross_validate_module(model)


def test_squeezenet1_1():
    model = torchvision.models.squeezenet1_1(pretrained=False)
    _cross_validate_module(model)


def test_resnet18():
    model = torchvision.models.resnet18(pretrained=False)
    _cross_validate_module(model)


def test_resnet34():
    model = torchvision.models.resnet34(pretrained=False)
    _cross_validate_module(model)


def test_resnet50():
    model = torchvision.models.resnet50(pretrained=False)
    _cross_validate_module(model)


def test_resnet101():
    model = torchvision.models.resnet101(pretrained=False)
    _cross_validate_module(model)


def test_resnet152():
    model = torchvision.models.resnet152(pretrained=False)
    _cross_validate_module(model)


def test_densenet121():
    model = torchvision.models.densenet121(pretrained=False)
    _cross_validate_module(model)


def test_densenet169():
    model = torchvision.models.densenet169(pretrained=False)
    _cross_validate_module(model)


def test_densenet201():
    model = torchvision.models.densenet201(pretrained=False)
    _cross_validate_module(model)


def test_densenet161():
    model = torchvision.models.densenet161(pretrained=False)
    _cross_validate_module(model)


def test_vgg11():
    model = torchvision.models.vgg11(pretrained=False)
    _cross_validate_module(model)


def test_vgg13():
    model = torchvision.models.vgg13(pretrained=False)
    _cross_validate_module(model)


def test_vgg16():
    model = torchvision.models.vgg16(pretrained=False)
    _cross_validate_module(model)


def test_vgg19():
    model = torchvision.models.vgg19(pretrained=False)
    _cross_validate_module(model)


def test_vgg11_bn():
    model = torchvision.models.vgg11_bn(pretrained=False)
    _cross_validate_module(model)


def test_vgg13_bn():
    model = torchvision.models.vgg13_bn(pretrained=False)
    _cross_validate_module(model)


def test_vgg16_bn():
    model = torchvision.models.vgg16_bn(pretrained=False)
    _cross_validate_module(model)


def test_vgg19_bn():
    model = torchvision.models.vgg19_bn(pretrained=False)
    _cross_validate_module(model)


def mobilenet_v2():
    model = torchvision.models.mobilenet_v2(pretrained=False)
    _cross_validate_module(model)


def test_shufflenet_v2_x0_5():
    model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
    _cross_validate_module(model)


def test_shufflenet_v2_x1_0():
    model = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
    _cross_validate_module(model)


def test_shufflenet_v2_x1_5():
    model = torchvision.models.shufflenet_v2_x1_5(pretrained=False)
    _cross_validate_module(model)


def test_shufflenet_v2_x2_0():
    model = torchvision.models.shufflenet_v2_x2_0(pretrained=False)
    _cross_validate_module(model)


def test_mnasnet0_5():
    model = torchvision.models.mnasnet0_5(pretrained=False)
    _cross_validate_module(model)


def test_mnasnet0_75():
    model = torchvision.models.mnasnet0_75(pretrained=False)
    _cross_validate_module(model)


def test_mnasnet1_0():
    model = torchvision.models.mnasnet1_0(pretrained=False)
    _cross_validate_module(model)


def test_mnasnet1_3():
    model = torchvision.models.mnasnet1_3(pretrained=False)
    _cross_validate_module(model)
