import torch
import tensorrt as trt
import numpy as np
import os
from collections import defaultdict
import importlib

from .dataset_calibrator import (
    DatasetCalibrator,
    DEFAULT_CALIBRATION_ALGORITHM,
)

from .precision import (
    WEAK_TYPING_AVAILABLE,
    LEGACY_INT8_CALIBRATION_AVAILABLE,
    autocast_onnx_to_fp16,
    builder_flag,
    calibration_arrays,
    graph_has_explicit_quantization,
    network_creation_flags,
    network_has_explicit_quantization,
    quantize_onnx_int8,
)

from .dataset import (
    Dataset,
    ListDataset
)

from .flattener import Flattener
from .flatten_module import Flatten
from .version_utils import trt_version
from .trt_module import TRTModule
from .misc_utils import (
    torch_device_to_trt,
    torch_dtype_to_trt,
    trt_int_dtype
)
# UTILITY FUNCTIONS

def trt_num_inputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            count += 1
    return count


def trt_num_outputs(engine):
    count = 0
    for i in range(engine.num_bindings):
        if not engine.binding_is_input(i):
            count += 1
    return count


def torch_dim_resolve_negative(dim, ndim):
    if not isinstance(dim, tuple):
        dim = (dim,)
    pos = []
    for d in dim:
        if d < 0:
            d = ndim + d
        pos.append(d)
    return tuple(pos)


def torch_dim_to_trt_axes(dim):
    """Converts torch dim, or tuple of dims to a tensorrt axes bitmask"""
    if not isinstance(dim, tuple):
        dim = (dim,)

    # create axes bitmask for reduce layer
    axes = 0
    for d in dim:
        axes |= 1 << d 

    return axes


def add_trt_constant(network, tensor):
    shape = tuple(tensor.shape)
    array = tensor[0].detach().cpu().numpy()
    layer = network.add_constant(shape, array)
    return layer.get_output(0)


def check_torch_dtype(*tensors):
    dtype = None
    for t in tensors:
        if isinstance(t, torch.Tensor):
            if dtype is None:
                dtype = t.dtype
            else:
                assert dtype == t.dtype  # , 'Tensor data types must match')
    assert (
        dtype is not None
    )  # , 'Data type could not be inferred from any item in list')
    return dtype


def add_missing_trt_tensors(network, tensors):
    """Creates missing TensorRT tensors as constants and attaches them to the Torch Tensors"""
    with use_shape_wrapping(False):
        trt_tensors = [None] * len(tensors)

        dtype = check_torch_dtype(*tensors)

        for i, t in enumerate(tensors):
            trt_tensor = None

            # GET TRT TENSOR (OR CREATE TRT CONSTANT)

            # get tensor w/ _trt
            # or... add constant for scalar primitive
            if hasattr(t, "_trt") or isinstance(t, IntWrapper):
                trt_tensor = t._trt
            elif isinstance(t, float) or isinstance(t, int):
                shape = (1,)
                scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
                trt_tensor = network.add_constant(shape, scalar).get_output(0)

            # or... add constant for leaf tensor w/o _trt
            else:

                # remove all preceding ones, these can be re-inserted later when broadcasting
                num_preceding_ones = 0
                for j in range(len(t.shape)):
                    if int(t.shape[j]) == 1:
                        num_preceding_ones += 1
                    else:
                        break
                shape = tuple(t.shape[num_preceding_ones:])

                weight = t.detach().cpu().numpy()
                t._trt = network.add_constant(shape, weight).get_output(0)
                trt_tensor = t._trt


            assert trt_tensor is not None

            trt_tensors[i] = trt_tensor

        return trt_tensors


def broadcast_trt_tensors(network, trt_tensors, broadcast_ndim):
    """Broadcast TensorRT tensors to the specified dimension by pre-padding shape 1 dims"""
    with use_shape_wrapping(False):
        broadcasted_trt_tensors = [None] * len(trt_tensors)

        for i, t in enumerate(trt_tensors):

            if len(t.shape) < broadcast_ndim:
                # append 1 size dims to front
                diff = broadcast_ndim - len(t.shape)
                shape = tuple([1] * diff + list(t.shape))
                layer = network.add_shuffle(t)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)
            else:
                trt_tensor = t

            broadcasted_trt_tensors[i] = trt_tensor

        return broadcasted_trt_tensors


def trt_(network, *tensors):
    """Creates missing TensorRT tensors and adds shuffle layers to make tensors broadcastable"""
    with use_shape_wrapping(False):
        trt_tensors = [None] * len(tensors)

        dtype = check_torch_dtype(*tensors)

        # get broadcast dimension
        broadcast_num_dim = 0
        for t in tensors:
            if isinstance(t, torch.Tensor):
                if not hasattr(t, "_trt"):
                    num_dim = len(t.shape)  # don't exclude batch for constants
                else:
                    num_dim = len(
                        t._trt.shape
                    )  # non-leaf tensors must already have _trt, get shape from that
                if num_dim > broadcast_num_dim:
                    broadcast_num_dim = num_dim

        for i, t in enumerate(tensors):
            trt_tensor = None

            # GET TRT TENSOR (OR CREATE TRT CONSTANT)

            # get tensor w/ _trt
            if (isinstance(t, torch.Tensor) and hasattr(t, "_trt")) or isinstance(t, IntWrapper):
                trt_tensor = t._trt

            # or... add constant for leaf tensor w/o _trt
            elif isinstance(t, torch.Tensor) and not hasattr(t, "_trt"):
                # add leaf tensor
                shape = tuple(t.shape)  #  don't exclude batch when adding constants...?
                weight = t.detach().cpu().numpy()
                t._trt = network.add_constant(shape, weight).get_output(0)
                trt_tensor = t._trt

            # or... add constant for scalar primitive
            elif isinstance(t, float) or isinstance(t, int):
                shape = (1,) * broadcast_num_dim
                scalar = t * torch.ones(shape, dtype=dtype).cpu().numpy()
                trt_tensor = network.add_constant(shape, scalar).get_output(0)

            assert trt_tensor is not None

            # MAKE TRT TENSOR BROADCASTABLE IF IT IS NOT ALREADY

            if len(trt_tensor.shape) < broadcast_num_dim:
                # append 1 size dims to front
                diff = broadcast_num_dim - len(trt_tensor.shape)
                shape = tuple([1] * diff + list(trt_tensor.shape))
                layer = network.add_shuffle(trt_tensor)
                layer.reshape_dims = shape
                trt_tensor = layer.get_output(0)

            trt_tensors[i] = trt_tensor

        if len(trt_tensors) == 1:
            return trt_tensors[0]
        else:
            return tuple(trt_tensors)


# CONVERSION REGISTRY AND HOOKS


CONVERTERS = {}


def get_arg(ctx, name, pos, default):
    if name in ctx.method_kwargs:
        return ctx.method_kwargs[name]
    elif len(ctx.method_args) > pos:
        return ctx.method_args[pos]
    else:
        return default


def attach_converter(ctx, method, converter, method_str):
    """Gets a function that executes PyTorch method and TensorRT converter"""
    global DUMMY_CONVERTERS

    def wrapper(*args, **kwargs):
        skip = True

        # check if another (parent) converter has lock
        if not ctx.lock:
            if converter["is_real"]:
                ctx.lock = True  # only real converters can acquire lock
            skip = False

        # run original method
        outputs = method(*args, **kwargs)

        if not skip:
            ctx.method_args = args
            ctx.method_kwargs = kwargs
            ctx.method_return = outputs
            ctx.method_str = method_str

            #             print('%s' % (converter.__name__,))
            converter["converter"](ctx)

            # allow overwriting output, for things like shape converter
            outputs = ctx.method_return

            # convert to None so conversion will fail for unsupported layers
            ctx.method_args = None
            ctx.method_kwargs = None
            ctx.method_return = None
            ctx.lock = False

        return outputs

    return wrapper


class ConversionHook(object):
    """Attaches TensorRT converter to PyTorch method call"""

    def __init__(self, ctx, key, converter):
        self.ctx = ctx
        self.key = key
        self.converter = converter

    def _set_method(self, method):
        module = self.converter['module']
        exec('module.%s = method' % self.converter['qual_name'])

    def __enter__(self):
        self._set_method(
            attach_converter(
                self.ctx, self.converter['method_impl'], self.converter, self.converter['method_str']
            )
        )

    def __exit__(self, type, val, tb):
        self._set_method(self.converter['method_impl'])

def default_input_names(num_inputs):
    return ["input_%d" % i for i in range(num_inputs)]

def default_output_names(num_outputs):
    return ["output_%d" % i for i in range(num_outputs)]


def device_type_str(device_type):
    if device_type == trt.DeviceType.GPU:
        return 'GPU'
    elif device_type == trt.DeviceType.DLA:
        return 'DLA'
    

class NetworkWrapper(object):
    def __init__(self, ctx, network):
        self._ctx = ctx
        self._network = network
        self._layer_counts = defaultdict(lambda: 0)

    def _configure_layer(self, layer):
        with use_shape_wrapping(False):
        
            # set layer device type
            device_type = self._ctx.current_device_type()
            self._ctx.builder_config.set_device_type(layer, device_type)
            orig_device_type = device_type
            if device_type == trt.DeviceType.DLA and not self._ctx.builder_config.can_run_on_DLA(layer):
                if self._ctx.torch2trt_kwargs['gpu_fallback']:
                    device_type = trt.DeviceType.GPU  # layer will fall back to GPU
            
            # set layer name
            def arg_str(arg):
                if isinstance(arg, torch.Tensor):
                    return "tensor(shape=%s, dtype=%s)" % (str(list(arg.shape)), str(arg.dtype))
                return str(arg)
            scope_name = self._ctx.current_module_name()# + ':' + layer.type.name
            self._layer_counts[scope_name] += 1
            args = [arg_str(arg) for arg in self._ctx.method_args]
            kwargs = ["%s=%s" % (key, arg_str(arg)) for key, arg in self._ctx.method_kwargs.items()]
            layer.name = scope_name + ':' + str(self._layer_counts[scope_name] - 1) + ':' + layer.type.name + ':' + device_type_str(device_type) 
            
            if orig_device_type != device_type:
                layer.name = layer.name + '(' + device_type_str(orig_device_type) + ')'
    #         "%s [%s #%d, %s] %s(%s)" % (self._ctx.current_module_name(), layer.type.name, self._layer_counts[layer.type.name], device_type_str(device_type),
    #                                           self._ctx.method_str, ", ".join(args + kwargs))
    
        
    def __getattr__(self, name):
        attr = getattr(self._network, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                ret = attr(*args, **kwargs)
                if isinstance(ret, trt.ILayer):
                    self._configure_layer(ret)
                return ret

            return wrapper
        else:
            return attr


_ACTIVE_CONVERSION_CONTEXT = None


def get_conversion_context():
    return _ACTIVE_CONVERSION_CONTEXT


class ConversionContext(object):
    
    def __init__(self, network, converters=CONVERTERS, torch2trt_kwargs=None, builder_config=None, logger=None):
        self.network = NetworkWrapper(self, network)
        self.lock = False
        self.method_args = None
        self.method_kwargs = None
        self.method_return = None
        self.torch2trt_kwargs = torch2trt_kwargs
        self.builder_config = builder_config
        self.hooks = [
            ConversionHook(self, key, converter)
            for key, converter in converters.items()
        ]
        
        self.module_stack = []
        self.module_handles = []
        self.device_type_stack = []
        self.module_name_map = {}
        for name, module in torch2trt_kwargs['module'].named_modules():
            self.module_name_map[module] = name
        self.logger = logger

    def current_module_name(self):
        return self.get_module_name(self.current_module())
    
    def current_module(self):
        return self.module_stack[-1]
    
    def get_module_name(self, module):
        return self.module_name_map[module]
    
    def _module_pre_hook(self, module, input):
        # TODO(@jwelsh): add logging to show module entry / exit
        self.module_stack.append(module)
        
        # hook that is attached to modulee using register_forward_pre_hook, which is called before module is executed
        if module in self.torch2trt_kwargs['device_types']:
            device_type = self.torch2trt_kwargs['device_types'][module]
            self.device_type_stack.append((module, device_type))
        
    def _module_post_hook(self, module, input, output):
        
        # if module was used to set the current device type, pop device type from stack
        if self.current_device_type_module() == module:
            self.device_type_stack.pop()
            
        self.module_stack.pop()
        
    def current_device_type(self):
        """Returns the current device type"""
        if len(self.device_type_stack) > 0:
            return self.device_type_stack[-1][1]
        else:
            return self.torch2trt_kwargs['default_device_type']
        
    def current_device_type_module(self):
        """Returns the module which controls the current device type"""
        if len(self.device_type_stack) > 0:
            return self.device_type_stack[-1][0]
        else:
            return None
        
    def __enter__(self):
        global _ACTIVE_CONVERSION_CONTEXT
        
        # attach hooks which add converters to methods
        for hook in self.hooks:
            hook.__enter__()
        
        # attach hooks which control the current device type
        for name, module in self.torch2trt_kwargs['module'].named_modules():
            pre_hook_handle = module.register_forward_pre_hook(self._module_pre_hook)
            post_hook_handle = module.register_forward_hook(self._module_post_hook)
            self.module_handles.append(pre_hook_handle)
            self.module_handles.append(post_hook_handle)
            
        _ACTIVE_CONVERSION_CONTEXT = self

        torch.Tensor.size = _size_wrapper
        torch.Tensor.__getattribute__ = _new_getattr

        return self

    def __exit__(self, type, val, tb):
        global _ACTIVE_CONVERSION_CONTEXT
        

        for hook in self.hooks:
            hook.__exit__(type, val, tb)
        for handle in self.module_handles:
            handle.remove()

        _ACTIVE_CONVERSION_CONTEXT = None

        torch.Tensor.size = _original_size
        torch.Tensor.__getattribute__ = _old_getattr


    def add_inputs(self, torch_inputs, names=None, dynamic_axes=None):

        if names is None:
            names = default_input_names(len(torch_inputs))
        self.input_names = names

        for i, torch_input in enumerate(torch_inputs):

            if not hasattr(torch_input, "_trt"):
                
                shape = list(torch_input.shape)
                
                if dynamic_axes is not None:
                    for dim in dynamic_axes[i]:
                        shape[dim] = -1

                shape = tuple(shape)

                trt_tensor = self.network.add_input(
                    name=names[i],
                    shape=shape,
                    dtype=torch_dtype_to_trt(torch_input.dtype),
                )
                trt_tensor.location = torch_device_to_trt(torch_input.device)
                torch_input._trt = trt_tensor

    def mark_outputs(self, torch_outputs, names=None):
        if names is None:
            names = default_output_names(len(torch_outputs))
        self.output_names = names

        for i, torch_output in enumerate(torch_outputs):
            trt_tensor = torch_output._trt
            trt_tensor.name = names[i]
            trt_tensor.location = torch_device_to_trt(torch_output.device)
            trt_tensor.dtype = torch_dtype_to_trt(torch_output.dtype)
            self.network.mark_output(trt_tensor)





def infer_dynamic_axes(min_shapes_flat, max_shapes_flat):
    dynamic_axes = [[] for i in range(len(min_shapes_flat))]
    for i, (mins, maxs) in enumerate(zip(min_shapes_flat, max_shapes_flat)):
        for j, (mins_i, maxs_i) in enumerate(zip(mins, maxs)):
            if mins_i != maxs_i:
                dynamic_axes[i].append(j)
    return dynamic_axes

# protobuf refuses to serialize any message larger than 2 GiB, so a plain
# onnx.save() of a big graph dies with "EncodeError: Failed to serialize proto".
# Whisper's large-family audio encoder (~635M params) crosses that in fp32.
# Writing the weights out as external data keeps the proto itself small; the
# sidecar is resolved relative to the model file, which TensorRT's
# parse_from_file() handles.
_PROTO_SIZE_LIMIT = 2 * 1024 ** 3

# Headroom for the graph structure (nodes, names, value_info) that the
# initializer-only estimate below does not account for.
_PROTO_SIZE_MARGIN = 128 * 1024 ** 2


def _tensor_nbytes(tensor):
    """Approximate a TensorProto's serialized size without serializing it."""
    if tensor.raw_data:
        return len(tensor.raw_data)
    return (
        len(tensor.float_data) * 4
        + len(tensor.int32_data) * 4
        + len(tensor.int64_data) * 8
        + len(tensor.double_data) * 8
        + len(tensor.uint64_data) * 8
        + sum(len(s) for s in tensor.string_data)
    )


def _weights_nbytes(model_proto):
    """Sum the tensor payload of a ModelProto (initializers + attribute tensors).

    ByteSize() would be exact, but it raises on exactly the oversized models
    this estimate exists to detect.
    """
    total = 0
    for tensor in model_proto.graph.initializer:
        total += _tensor_nbytes(tensor)
    for node in model_proto.graph.node:
        for attr in node.attribute:
            if attr.HasField("t"):
                total += _tensor_nbytes(attr.t)
            for tensor in attr.tensors:
                total += _tensor_nbytes(tensor)
    return total


def save_onnx(onnx, model_proto, path):
    """Save ``model_proto`` to ``path``, spilling weights out when it is too big.

    Returns True if the weights were written to an external data file next to
    the model, in which case the model can only be parsed from that path (the
    serialized bytes alone no longer carry the weights).
    """
    if _weights_nbytes(model_proto) < _PROTO_SIZE_LIMIT - _PROTO_SIZE_MARGIN:
        onnx.save(model_proto, path)
        return False

    onnx.save(
        model_proto,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(path) + ".data",
        size_threshold=1024,
        convert_attribute=False,
    )
    return True


def torch2trt(module,
              inputs,
              input_names=None,
              output_names=None,
              log_level=trt.Logger.ERROR,
              fp16_mode=False,
              max_workspace_size=1<<25,
              strict_type_constraints=False,
              keep_network=True,
              int8_mode=False,
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              use_onnx=False,
              default_device_type=trt.DeviceType.GPU,
              dla_core=0,
              gpu_fallback=True,
              device_types={},
              min_shapes=None,
              max_shapes=None,
              opt_shapes=None,
              onnx_opset=None,
              max_batch_size=None,
              avg_timing_iterations=None,
              fp16_op_block_list=None,
              int8_op_block_list=None,
              int8_calibration_eps=None,
              **kwargs):

    # capture arguments to provide to context
    kwargs.update(locals())
    kwargs.pop('kwargs')

    # On strongly typed TensorRT (11.x+) precision lives in the graph, not in
    # builder flags, and only the ONNX path can carry it: FP16 needs AutoCast to
    # rewrite tensor types, INT8 needs Q/DQ nodes from the exported model. Fail
    # here rather than after a long build that silently came out FP32.
    if not WEAK_TYPING_AVAILABLE and (fp16_mode or int8_mode) and not use_onnx:
        raise RuntimeError(
            f"TensorRT {trt.__version__} builds strongly typed networks, so reduced "
            "precision has to come from the parsed graph. Convert with use_onnx=True "
            "to request fp16_mode or int8_mode."
        )

    # handle inputs as dataset of list of tensors
    if issubclass(inputs.__class__, Dataset):
        dataset = inputs
        if len(dataset) == 0:
            raise ValueError('Dataset must have at least one element to use for inference.')
        inputs = dataset[0]
    else:
        dataset = ListDataset()
        dataset.insert(inputs)
        inputs = dataset[0]

    outputs = module(*inputs)
    input_flattener = Flattener.from_value(inputs)
    output_flattener = Flattener.from_value(outputs)

    # infer default parameters from dataset

    if min_shapes is None:
        min_shapes_flat = [tuple(t) for t in dataset.min_shapes(flat=True)]
    else:
        min_shapes_flat = input_flattener.flatten(min_shapes)

    if max_shapes is None:
        max_shapes_flat = [tuple(t) for t in dataset.max_shapes(flat=True)]
    else:
        max_shapes_flat = input_flattener.flatten(max_shapes)
    
    if opt_shapes is None:
        opt_shapes_flat = [tuple(t) for t in dataset.median_numel_shapes(flat=True)]
    else:
        opt_shapes_flat = input_flattener.flatten(opt_shapes)

    # handle legacy max_batch_size
    if max_batch_size is not None:
        min_shapes_flat = [(1,) + s[1:] for s in min_shapes_flat]
        max_shapes_flat = [(max_batch_size,) + s[1:] for s in max_shapes_flat]

    dynamic_axes_flat = infer_dynamic_axes(min_shapes_flat, max_shapes_flat)
    
    if default_device_type == trt.DeviceType.DLA:
        for value in dynamic_axes_flat:
            if len(value) > 0:
                raise ValueError('Dataset cannot have multiple shapes when using DLA')

    logger = trt.Logger(log_level)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    if input_names is None:
        input_names = default_input_names(input_flattener.size)
    if output_names is None:
        output_names = default_output_names(output_flattener.size)

    if use_onnx:
        import onnx_graphsurgeon as gs
        import onnx
        import tempfile

        module_flat = Flatten(module, input_flattener, output_flattener)
        inputs_flat = input_flattener.flatten(inputs)

        # Export, optimize, and parse ONNX via a temp directory (auto-cleaned)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_in_path = os.path.join(tmpdir, "model_in.onnx")
            
            # Check PyTorch version to use legacy ONNX exporter for 2.9+
            torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
            
            export_args = dict(
                model=module_flat,
                args=inputs_flat,
                f=tmp_in_path,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes={
                    name: {int(axis): f"input_{index}_axis_{axis}" for axis in dynamic_axes_flat[index]}
                    for index, name in enumerate(input_names)
                },
            )
            
            # Force legacy ONNX exporter for PyTorch 2.9+ to avoid torch.export issues
            if torch_version >= (2, 9):
                export_args["dynamo"] = False
            if onnx_opset is not None:
                export_args["opset_version"] = onnx_opset
            torch.onnx.export(**export_args)

            # Load and manipulate ONNX graph
            onnx_graph = gs.import_onnx(onnx.load(tmp_in_path))
            onnx_graph.fold_constants().cleanup()

            model_proto = gs.export_onnx(onnx_graph)

            # Save manipulated graph to another temp file
            tmp_out_path = os.path.join(tmpdir, "model_out.onnx")
            external_data = save_onnx(onnx, model_proto, tmp_out_path)

            # Strongly typed TensorRT takes its precision from the graph, so
            # rewrite the graph rather than setting builder flags. Which rewrite
            # depends on what is being asked for: the ONNX quantizer owns INT8
            # and can cast the unquantized remainder to FP16 in the same pass,
            # while AutoCast handles the FP16-only case. They are not
            # interchangeable — AutoCast does not support Q/DQ graphs.
            if not WEAK_TYPING_AVAILABLE and (fp16_mode or int8_mode):
                pre_quantized = graph_has_explicit_quantization(model_proto)
                if int8_mode and not pre_quantized:
                    calib_arrays = calibration_arrays(
                        dataset if int8_calib_dataset is None else int8_calib_dataset,
                        input_flattener,
                        input_names,
                    )
                    quantized_path = os.path.join(tmpdir, "model_int8.onnx")
                    quantize_onnx_int8(
                        tmp_out_path,
                        quantized_path,
                        calib_arrays,
                        fp16=fp16_mode,
                        int8_op_block_list=int8_op_block_list,
                        fp16_op_block_list=fp16_op_block_list,
                        calibration_eps=int8_calibration_eps,
                        external_data=external_data,
                    )
                    tmp_out_path = quantized_path
                elif pre_quantized and fp16_mode:
                    raise RuntimeError(
                        "The exported graph already carries Q/DQ nodes, which AutoCast "
                        "cannot cast to FP16 (it would leave QuantizeLinear scales at a "
                        "type its input no longer has). Either export the module in FP16 "
                        "yourself, or pass an unquantized module plus int8_calib_dataset "
                        "and let the ONNX quantizer do both precisions."
                    )
                elif fp16_mode:
                    model_proto = autocast_onnx_to_fp16(model_proto, fp16_op_block_list)
                    external_data = save_onnx(onnx, model_proto, tmp_out_path)

            # Create network and parser, and parse ONNX inside context
            network = builder.create_network(network_creation_flags())
            parser = trt.OnnxParser(network, logger)
            # Use context manager to read ONNX file if needed
            if hasattr(parser, "parse_from_file"):
                parsed = parser.parse_from_file(tmp_out_path)
            elif external_data:
                # The weights live in a sidecar file, so the serialized model
                # alone is not parseable — only the path-based entry point can
                # follow the external-data reference.
                raise RuntimeError(
                    "ONNX model exceeds protobuf's 2 GB limit and was saved with "
                    "external weights, but this TensorRT build's OnnxParser has no "
                    "parse_from_file(). Upgrade TensorRT to convert this model."
                )
            else:
                with open(tmp_out_path, "rb") as f:
                    parsed = parser.parse(f.read())
            # Log parse errors:
            if not parsed:
                for i in range(parser.num_errors):
                    logger.log(trt.Logger.ERROR, str(parser.get_error(i)))
                raise RuntimeError("Failed to parse ONNX model.")

    else:
        network = builder.create_network(network_creation_flags())
        with ConversionContext(network, torch2trt_kwargs=kwargs, builder_config=config, logger=logger) as ctx:
            
            inputs_flat = input_flattener.flatten(inputs)

            ctx.add_inputs(inputs_flat, input_names, dynamic_axes=dynamic_axes_flat)

            outputs = module(*inputs)

            outputs_flat = output_flattener.flatten(outputs)
            ctx.mark_outputs(outputs_flat, output_names)

    # set max workspace size
    if trt_version() < "10.0":
        config.max_workspace_size = max_workspace_size
    else:
        # TensorRT 10 removed IBuilderConfig.max_workspace_size; the build-time
        # scratch budget is now a memory pool limit. Without setting it the
        # build is unbounded and defaults to the entire device, which OOMs the
        # GPU (and drags host RAM up with it) on larger models.
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    

    # set number of avg timing itrs.
    if avg_timing_iterations is not None:
        config.avg_timing_iterations = avg_timing_iterations

    # Precision flags only exist while TensorRT supports weak typing; on 11.x the
    # graph was already cast above (FP16) or carries Q/DQ (INT8).
    if fp16_mode and WEAK_TYPING_AVAILABLE:
        config.set_flag(trt.BuilderFlag.FP16)

    config.default_device_type = default_device_type
    gpu_fallback_flag = builder_flag('GPU_FALLBACK')
    if gpu_fallback and gpu_fallback_flag is not None:
        config.set_flag(gpu_fallback_flag)
    config.DLA_core = dla_core

    # STRICT_TYPES is gone from 11.x, where a strongly typed network already
    # obeys the graph's types exactly — the request is satisfied by definition.
    strict_types_flag = builder_flag('STRICT_TYPES')
    if strict_type_constraints and strict_types_flag is not None:
        config.set_flag(strict_types_flag)

    calibrator = None

    if int8_mode:
        int8_flag = builder_flag('INT8')
        if int8_flag is not None:
            config.set_flag(int8_flag)

        # A network that already carries Q/DQ (explicit quantization) is fully
        # specified — calibrating it as well would fight the baked-in scales.
        explicit_quantization = network_has_explicit_quantization(network)

        if not LEGACY_INT8_CALIBRATION_AVAILABLE:
            # The ONNX quantizer above should have put them there; a network
            # without them would build silently as a float engine.
            if not explicit_quantization:
                raise RuntimeError(
                    f"int8_mode on TensorRT {trt.__version__} requires explicit "
                    "quantization, but the parsed network has no Q/DQ layers — "
                    "implicit calibration was removed in TensorRT 11.0. Convert with "
                    "use_onnx=True and an int8_calib_dataset so the ONNX quantizer can "
                    "insert them, or hand in an already-quantized module."
                )
        elif not kwargs.get('qat_mode', False) and not explicit_quantization:
            # default to use input tensors for calibration
            if int8_calib_dataset is None:
                int8_calib_dataset = dataset

            calibrator = DatasetCalibrator(
                int8_calib_dataset, algorithm=int8_calib_algorithm
            )
            config.int8_calibrator = calibrator

    # OPTIMIZATION PROFILE
    profile = builder.create_optimization_profile()
    for index, name in enumerate(input_names):
        profile.set_shape(
            name,
            min_shapes_flat[index],
            opt_shapes_flat[index],
            max_shapes_flat[index]
        )
    config.add_optimization_profile(profile)

    # Only an implicit-quantization calibrator needs a profile to calibrate over.
    if calibrator is not None:
        config.set_calibration_profile(profile)

    # BUILD ENGINE

    if trt_version() < "10.0":
        engine = builder.build_engine(network, config)
    else:
        engine = builder.build_serialized_network(network, config)

    module_trt = TRTModule(engine, input_names, output_names, input_flattener=input_flattener, output_flattener=output_flattener)

    if keep_network:
        module_trt.network = network

    return module_trt


# DEFINE ALL CONVERSION FUNCTIONS

def get_module_qualname(name):
    s = name.split('.')

    for i in range(len(s)):
        idx = len(s) - i - 1
        modulename, qualname = ".".join(s[:idx]), ".".join(s[idx:])
        try:
            module = importlib.import_module(modulename)
            return module, modulename, qualname
        except ModuleNotFoundError:
            # keep searching for a valid split point
            continue
        except ImportError as e:
            # Surface unexpected import issues
            raise RuntimeError("Failed to parse ONNX model. " + e.msg)

    raise RuntimeError("Could not import module")


def tensorrt_converter(method, is_real=True, enabled=True, imports=[]):

    if isinstance(method, str):
        module, module_name, qual_name = get_module_qualname(method)
    else:
        module, module_name, qual_name = importlib.import_module(method.__module__), method.__module__, method.__qualname__

    try:
        # No deepcopy needed; store original callable
        method_impl = eval('module.%s' % qual_name)
    except AttributeError:
        enabled = False

    def register_converter(converter):
        CONVERTERS[method] = {
            "converter": converter,
            "is_real": is_real,
            "module": module,
            "module_name": module_name,
            "qual_name": qual_name,
            "method_str": module_name + '.' + qual_name,
            "method_impl": method_impl
        }
        return converter

    def pass_converter(converter):
        return converter

    if enabled:
        return register_converter
    else:
        return pass_converter

    return register_converter


def set_layer_precision(ctx, layer):
    # Supported TRT precisions as given by torch2trt_kwargs.
    INT8_MODE = "int8_mode"
    FP16_MODE = "fp16_mode"

    # Check that args exist as expected in torch2trt_kwargs.
    trt_kwargs = ctx.torch2trt_kwargs
    assert INT8_MODE in trt_kwargs
    assert FP16_MODE in trt_kwargs

    is_int8 = trt_kwargs.get(INT8_MODE, False)
    is_fp16 = trt_kwargs.get(FP16_MODE, False)

    if is_int8:
        layer.precision = trt.int8
        layer.set_output_type(0, trt.int8)
    elif is_fp16:
        layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)


# SHAPE WRAPPING
_int = int
_tuple = tuple
_int_mul = int.__mul__
_int_add = int.__add__
_int_sub = int.__sub__
_int_floordiv = int.__floordiv__

class IntWrapper(int):
    
    @property
    def _trt(self):
        if not hasattr(self, '_raw_trt'):
            ctx = get_conversion_context()
            self._raw_trt = ctx.network._network.add_constant([1], np.array([_int(self)], dtype=trt_int_dtype())).get_output(0)
        return self._raw_trt

    # lhs ops
    def __mul__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_mul(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.PROD).get_output(0)
        return result

    def __add__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_add(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.SUM).get_output(0)
        return result

    def __sub__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_sub(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.SUB).get_output(0)
        return result

    def __floordiv__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_floordiv(self, x))
        result._raw_trt = ctx.network._network.add_elementwise(self._trt, x._trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        return result

    # rhs ops
    def __rmul__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_mul(x, self))
        result._raw_trt = ctx.network._network.add_elementwise(x._trt, self._trt, trt.ElementWiseOperation.PROD).get_output(0)
        return result

    def __radd__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_add(x, self))
        result._raw_trt = ctx.network._network.add_elementwise(x._trt, self._trt, trt.ElementWiseOperation.SUM).get_output(0)
        return result

    def __rsub__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_sub(x, self))
        result._raw_trt = ctx.network._network.add_elementwise(x._trt, self._trt, trt.ElementWiseOperation.SUB).get_output(0)
        return result

    def __rfloordiv__(self, x):
        if not isinstance(x, IntWrapper):
            x = IntWrapper(x)
        ctx = get_conversion_context()
        result = IntWrapper(_int_floordiv(x, self))
        result._raw_trt = ctx.network._network.add_elementwise(x._trt, self._trt, trt.ElementWiseOperation.FLOOR_DIV).get_output(0)
        return result

    def __int__(self):
        return self

def make_int_wrapper(x):
    if isinstance(x, IntWrapper):
        return x
    else:
        return IntWrapper(x)

class SizeWrapper(tuple):

    @property
    def _trt(self):
        if not hasattr(self, '_raw_trt'):
            ctx = get_conversion_context()
            self._raw_trt = ctx.network._network.add_concatenation([d._trt for d in self]).get_output(0)
        return self._raw_trt

    def __tuple__(self):
        return self


def wrap_ints(x):
    for y in x:
        yield make_int_wrapper(y)


def make_size_wrapper(args):
    return SizeWrapper(wrap_ints(args))


_original_size = torch.Tensor.size
_original_getattr = torch.Tensor.__getattribute__


def _size_wrapper(input, dim=None):

    if not hasattr(input, '_trt'):
        if dim is not None:
            return _original_size(input, dim)
        else:
            return _original_size(input)

    ctx = get_conversion_context()

    output = _original_size(input)

    output = make_size_wrapper(output)

    shape_trt = ctx.network._network.add_shape(input._trt).get_output(0)

    for i, d in enumerate(output):
        d._raw_trt = ctx.network._network.add_slice(shape_trt, [i], [1], [1]).get_output(0)

    if dim is not None:
        output = output[dim]

    return output


_old_getattr = torch.Tensor.__getattribute__

def _new_getattr(self, name):
    if name == 'shape' and use_shape_wrapping.stack[0]:
        return _size_wrapper(self)
    else:
        return _old_getattr(self, name)

class use_shape_wrapping:

    stack = [True] # default true

    def __init__(self, value: bool):
        self._value = value
    
    def __enter__(self, *args, **kwargs):
        self.stack.insert(0, self._value)

    def __exit__(self, *args, **kwargs):
        self.stack.pop(0)
