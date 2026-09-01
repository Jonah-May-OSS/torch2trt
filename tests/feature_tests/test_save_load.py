import pytest
import torch

import torch2trt

# TensorRT conversion needs a device to build and run engines on, so every
# test in this module requires one. Without the skip these fail rather than
# skip, which makes a CPU-only run indistinguishable from a broken one.
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


def test_save_load():
    model = torch.nn.Conv2d(3, 3, 1).cuda().eval().half()
    data = torch.randn((1, 3, 224, 224)).cuda().half()

    print("Running torch2trt...")
    model_trt = torch2trt.torch2trt(
        model, [data], fp16_mode=True, max_workspace_size=1 << 25
    )

    print("Saving model...")
    torch.save(model_trt.state_dict(), ".test_model.pth")

    print("Loading model...")
    model_trt_2 = torch2trt.TRTModule()
    model_trt_2.load_state_dict(torch.load(".test_model.pth"))

    assert model_trt_2.engine is not None

    print(torch.max(torch.abs(model_trt_2(data) - model(data))))
    print(torch.max(torch.abs(model_trt_2(data) - model_trt(data))))
