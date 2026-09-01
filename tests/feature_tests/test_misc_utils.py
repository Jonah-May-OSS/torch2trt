"""Device conversion between torch and TensorRT.

No GPU needed: these are pure lookups over ``torch.device`` and
``trt.TensorLocation``, and the interesting case is the one neither side can
represent.
"""

import pytest
import torch

from torch2trt import trt
from torch2trt.misc_utils import torch_device_from_trt, torch_device_to_trt


def test_cuda_device_maps_to_trt_device():
    assert torch_device_to_trt(torch.device("cuda")) is trt.TensorLocation.DEVICE


def test_cpu_device_maps_to_trt_host():
    assert torch_device_to_trt(torch.device("cpu")) is trt.TensorLocation.HOST


def test_trt_device_maps_to_cuda():
    assert torch_device_from_trt(trt.TensorLocation.DEVICE).type == "cuda"


def test_trt_host_maps_to_cpu():
    assert torch_device_from_trt(trt.TensorLocation.HOST).type == "cpu"


def test_an_unsupported_torch_device_raises():
    """The failure has to be raised, not returned.

    This returned the TypeError rather than raising it, so an unrepresentable
    device produced an exception *object* that the caller went on to assign to
    ``trt_tensor.location``. The diagnostic was destroyed and the failure
    surfaced somewhere unrelated.
    """
    with pytest.raises(TypeError, match="not supported by tensorrt"):
        torch_device_to_trt(torch.device("meta"))


def test_an_unsupported_trt_location_raises():
    """As above, on the way back: the object was passed as ``device=``."""
    with pytest.raises(TypeError, match="not supported by torch"):
        torch_device_from_trt(object())
