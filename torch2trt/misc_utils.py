import torch
import numpy as np
import tensorrt as trt

from .version_utils import (
    trt_version
)


# Cache device objects to avoid repeated construction
_CUDA_DEVICE = torch.device("cuda")
_CPU_DEVICE = torch.device("cpu")


def torch_dtype_to_trt(dtype):
    if trt_version() >= '7.0' and dtype == torch.bool:
        return trt.bool
    elif dtype == torch.int8:
        return trt.int8
    elif dtype == torch.int32:
        return trt.int32
    elif dtype == torch.float16:
        return trt.float16
    elif dtype == torch.float32:
        return trt.float32
    else:
        raise TypeError("%s is not supported by tensorrt" % dtype)


def torch_dtype_from_trt(dtype):
    if dtype == trt.int8:
        return torch.int8
    elif trt_version() >= '7.0' and dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError("%s is not supported by torch" % dtype)


def torch_device_to_trt(device):
    if device.type == _CUDA_DEVICE.type:
        return trt.TensorLocation.DEVICE
    elif device.type == _CPU_DEVICE.type:
        return trt.TensorLocation.HOST
    else:
        raise TypeError("%s is not supported by tensorrt" % device)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return _CUDA_DEVICE
    elif device == trt.TensorLocation.HOST:
        return _CPU_DEVICE
    else:
        raise TypeError("%s is not supported by torch" % device)


def trt_int_dtype():
    if trt_version() >= "10.0":
        return np.int64
    else:
        return np.int32
    
