"""Precision handling across TensorRT's weak-typing and strong-typing eras.

TensorRT 11 removed weak typing. Up to 10.x a network was built in FP32 and the
builder was *asked* for reduced precision with per-precision builder flags
(``BuilderFlag.FP16``, ``BuilderFlag.INT8``), which TensorRT honoured only where
a faster tactic existed. In 11.x every network is strongly typed, those flags
are gone, and precision is whatever the graph handed to the parser says it is:

- FP16 comes from casting the ONNX graph (ModelOpt AutoCast inserts the casts).
- INT8 comes from Quantize/Dequantize pairs in the graph (explicit
  quantization); implicit quantization and the whole ``IInt8Calibrator`` family
  were removed with the flags.

Everything here probes the installed TensorRT rather than branching on the
version number, so one code path covers 8.x through 11.x.
"""

import tensorrt as trt

__all__ = [
    'WEAK_TYPING_AVAILABLE',
    'LEGACY_INT8_CALIBRATION_AVAILABLE',
    'network_creation_flags',
    'network_has_explicit_quantization',
    'autocast_onnx_to_fp16',
]


# True while the per-precision builder flags exist (TensorRT <= 10.x).
WEAK_TYPING_AVAILABLE = hasattr(trt.BuilderFlag, 'FP16')

# True while implicit INT8 quantization exists (TensorRT <= 10.x). Probed as a
# pair because TensorRT 11 removed the calibrator interface and the calibration
# algorithm enum together.
LEGACY_INT8_CALIBRATION_AVAILABLE = (
    hasattr(trt, 'IInt8Calibrator') and hasattr(trt, 'CalibrationAlgoType')
)


def network_creation_flags():
    """Return the ``create_network`` flags for an explicit-batch network.

    ``EXPLICIT_BATCH`` was the only sane mode from 8.x on, became implicit in
    10.x, and no longer exists as a flag once every network is strongly typed.
    """
    flag = getattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH', None)
    return 0 if flag is None else 1 << int(flag)


def builder_flag(name):
    """Return ``trt.BuilderFlag.<name>`` if this TensorRT still has it."""
    return getattr(trt.BuilderFlag, name, None)


def network_has_explicit_quantization(network):
    """Whether ``network`` carries Q/DQ layers, i.e. explicit quantization.

    Used to tell a network that was quantized before the build (Q/DQ baked into
    the ONNX graph) from one expecting the builder to calibrate it.
    """
    quantize = getattr(trt.LayerType, 'QUANTIZE', None)
    if quantize is None:
        return False
    return any(network.get_layer(i).type == quantize for i in range(network.num_layers))


def autocast_onnx_to_fp16(model_proto, op_block_list=None):
    """Cast an FP32 ONNX graph to mixed FP16, keeping graph I/O in FP32.

    This is the strong-typing replacement for ``BuilderFlag.FP16``: ModelOpt's
    AutoCast rewrites the graph's tensor types and inserts the casts, so the
    parsed network is already FP16 where it matters. Graph inputs and outputs
    stay FP32 so callers keep feeding and reading FP32 tensors exactly as they
    did under the builder flag.

    ``op_block_list`` names op types to leave in FP32 — the escape hatch for a
    layer that overflows or loses too much accuracy in FP16.
    """
    try:
        from modelopt.onnx.autocast import convert_to_f16
    except ImportError as exc:
        raise RuntimeError(
            f"fp16_mode needs the precision baked into the ONNX graph on TensorRT "
            f"{trt.__version__} (weak typing and BuilderFlag.FP16 were removed in 11.0), "
            "which requires ModelOpt: pip install nvidia-modelopt"
        ) from exc

    return convert_to_f16(
        model_proto,
        low_precision_type='fp16',
        keep_io_types=True,
        op_block_list=list(op_block_list or []),
    )
