import pytest
import torch

import torch2trt

# timm is not a torch2trt dependency; it only supplies the models under test.
# importorskip turns a missing optional dep into a skip, rather than a
# collection error -- and a collection error aborts the entire run, so one
# absent optional package took all 286 tests down with it.
maxxvit = pytest.importorskip("timm.models.maxxvit")
maxvit_tiny_rw_224 = maxxvit.maxvit_tiny_rw_224
maxvit_rmlp_pico_rw_256 = maxxvit.maxvit_rmlp_pico_rw_256
maxvit_rmlp_small_rw_224 = maxxvit.maxvit_rmlp_small_rw_224

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def _cross_validate_module(model, shape=(224, 224)):
    model = model.cuda()
    data = torch.randn(1, 3, *shape).cuda()
    model_trt = torch2trt.torch2trt(model, [data])
    out = model(data)
    out_trt = model_trt(data)
    assert torch.allclose(out, out_trt, rtol=1e-2, atol=1e-2)


def test_maxvit_tiny_rw_224():
    _cross_validate_module(maxvit_tiny_rw_224().cuda().eval(), (224, 224))


def test_maxvit_rmlp_small_rw_224():
    _cross_validate_module(maxvit_rmlp_small_rw_224().cuda().eval(), (224, 224))


if __name__ == "__main__":
    test_maxvit_tiny_rw_224()
