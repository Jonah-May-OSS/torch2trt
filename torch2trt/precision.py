"""Precision handling across TensorRT's weak-typing and strong-typing eras.

TensorRT 11 removed weak typing. Up to 10.x a network was built in FP32 and the
builder was *asked* for reduced precision with per-precision builder flags
(``BuilderFlag.FP16``, ``BuilderFlag.INT8``), which TensorRT honoured only where
a faster tactic existed. In 11.x every network is strongly typed, those flags
are gone, and precision is whatever the graph handed to the parser says it is:

- FP16 alone comes from casting the ONNX graph (ModelOpt AutoCast).
- INT8 comes from Quantize/Dequantize pairs in the graph (explicit
  quantization); implicit quantization and the whole ``IInt8Calibrator`` family
  were removed with the flags.
- INT8 *and* FP16 together come from ModelOpt's ONNX quantizer, which inserts
  the Q/DQ pairs and casts the rest of the graph in one pass. AutoCast cannot
  do this half: it does not support graphs that already carry Q/DQ, and casting
  one anyway yields a QuantizeLinear whose scale type no longer matches its
  input — an invalid graph that ONNX's own type inference rejects.

Everything here probes the installed TensorRT rather than branching on the
version number, so one code path covers 8.x through 11.x.
"""

import tensorrt as trt

__all__ = [
    'LEGACY_INT8_CALIBRATION_AVAILABLE',
    'WEAK_TYPING_AVAILABLE',
    'autocast_onnx_to_fp16',
    'builder_flag',
    'calibration_arrays',
    'graph_has_explicit_quantization',
    'network_creation_flags',
    'network_has_explicit_quantization',
    'quantize_onnx_int8',
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


def graph_has_explicit_quantization(model_proto):
    """Whether an ONNX graph already carries Q/DQ nodes.

    Checked before touching the graph, because a pre-quantized model needs the
    quantizer's FP16 handling rather than AutoCast's.
    """
    return any(
        node.op_type in ('QuantizeLinear', 'DequantizeLinear')
        for node in model_proto.graph.node
    )


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


def calibration_arrays(calib_dataset, flattener, input_names):
    """Turn a torch calibration dataset into ModelOpt's per-input arrays.

    Each input gets its items concatenated along axis 0, which is how the
    quantizer's data reader recovers the iteration count: it divides the array's
    leading dimension by the graph input's own leading dimension, so every input
    has to be stacked the same number of times.
    """
    import numpy as np

    columns = {name: [] for name in input_names}
    for index in range(len(calib_dataset)):
        tensors = flattener.flatten(calib_dataset[index])
        if len(tensors) != len(input_names):
            raise ValueError(
                f"Calibration item {index} has {len(tensors)} tensors but the model "
                f"takes {len(input_names)} inputs."
            )
        for name, tensor in zip(input_names, tensors):
            columns[name].append(tensor.detach().cpu().numpy())
    if not any(columns.values()):
        raise ValueError('Calibration dataset is empty.')
    return {name: np.concatenate(arrays, axis=0) for name, arrays in columns.items()}


def quantize_onnx_int8(
    onnx_path,
    output_path,
    calib_arrays,
    fp16=True,
    int8_op_block_list=None,
    fp16_op_block_list=None,
    calibration_eps=None,
    external_data=False,
):
    """Rewrite an FP32 ONNX model as INT8 Q/DQ, optionally FP16 elsewhere.

    This is the strong-typing replacement for ``BuilderFlag.INT8`` plus a
    calibrator, and it owns the FP16 cast too (``high_precision_dtype``) — the
    combination AutoCast cannot produce. Graph I/O stays FP32.

    Calibration runs the graph through onnxruntime over ``calib_arrays``, so it
    is bounded by the CPU execution provider unless a GPU/TensorRT provider is
    available and named in ``calibration_eps``.

    ``int8_op_block_list`` leaves op types unquantized; ``fp16_op_block_list``
    leaves op types in FP32 — the two escape hatches for a layer that cannot
    take the respective precision.
    """
    try:
        from modelopt.onnx.quantization import quantize
    except ImportError as exc:
        raise RuntimeError(
            f"int8_mode needs Q/DQ in the ONNX graph on TensorRT {trt.__version__} "
            "(implicit quantization was removed in 11.0), which requires ModelOpt: "
            "pip install nvidia-modelopt"
        ) from exc

    quantize(
        onnx_path,
        quantize_mode='int8',
        calibration_data=calib_arrays,
        calibration_eps=list(calibration_eps or ['cpu']),
        high_precision_dtype='fp16' if fp16 else 'fp32',
        op_types_to_exclude=list(int8_op_block_list) if int8_op_block_list else None,
        op_types_to_exclude_fp16=list(fp16_op_block_list) if fp16_op_block_list else None,
        use_external_data_format=external_data,
        output_path=output_path,
        log_level='WARNING',
    )
    return output_path
