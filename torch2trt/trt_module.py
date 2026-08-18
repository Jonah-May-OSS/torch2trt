import torch
import tensorrt as trt
from .flattener import Flattener
from .misc_utils import (
    torch_dtype_from_trt,
    torch_device_from_trt
)
from .version_utils import (
    trt_version
)


class SharedDeviceMemory:
    """One device scratch buffer shared by several execution contexts.

    By default each :class:`IExecutionContext` owns a private device-memory
    pool sized for its engine's worst-case layer scratch. A model split across
    several engines therefore reserves that scratch once per engine even though
    only one context is ever executing, which on multi-engine models is a
    material share of resident VRAM.

    Passing a pool to :class:`TRTModule` makes every context draw from a single
    buffer sized to the largest engine in the group instead of the sum.

    Contexts sharing a pool must not execute concurrently: the buffer is live
    for the duration of ``enqueue_v3``, so two contexts running at once would
    scribble over each other. Sequential execution on one stream -- the usual
    encoder-then-decoder pipeline -- is exactly the safe case. Do not share a
    pool between contexts driven from different threads or streams.
    """

    def __init__(self, device=None):
        self._device = device
        self._buffer = None
        self._contexts = []

    @property
    def nbytes(self):
        """Bytes currently reserved (0 before the first engine is added)."""
        return 0 if self._buffer is None else self._buffer.numel()

    @staticmethod
    def _engine_nbytes(engine):
        # device_memory_size is deprecated in TRT 10 in favour of the _v2 form,
        # which accounts for engines whose scratch depends on the chosen shapes.
        size = getattr(engine, "device_memory_size_v2", None)
        return int(size if size is not None else engine.device_memory_size)

    def add(self, engine):
        """Create a context for ``engine`` that draws on this shared buffer."""
        context = engine.create_execution_context_without_device_memory()
        if context is None:
            raise RuntimeError(
                "TensorRT refused to create an execution context without device "
                "memory; the engine cannot use a shared device-memory pool."
            )
        self._contexts.append(context)
        self._reserve(self._engine_nbytes(engine))
        return context

    def _reserve(self, nbytes):
        if nbytes <= self.nbytes and self._buffer is not None:
            self._bind(self._contexts[-1:])
            return
        # Growing means a new allocation, so every context already bound to the
        # old buffer is now pointing at memory the allocator may hand out again.
        # Re-bind all of them, and only drop the old buffer afterwards.
        previous = self._buffer
        self._buffer = torch.empty(
            max(nbytes, 1), dtype=torch.uint8, device=self._device or "cuda"
        )
        self._bind(self._contexts)
        del previous

    def _bind(self, contexts):
        address = self._buffer.data_ptr()
        size = self._buffer.numel()
        for context in contexts:
            context.set_device_memory(address, size)


class TRTModule(torch.nn.Module):
    def __init__(self, engine=None, input_names=None, output_names=None, input_flattener=None, output_flattener=None, device_memory=None):
        super(TRTModule, self).__init__()
        self._register_state_dict_hook(TRTModule._on_state_dict)
        # Optional SharedDeviceMemory; when set, this module's context draws its
        # scratch from the pool instead of reserving a private one.
        self.device_memory = device_memory

        if isinstance(engine, str):
            # assume filepath
            with open(engine, 'rb') as f:
                engine = f.read()
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
        elif isinstance(engine, trt.IHostMemory):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                engine = runtime.deserialize_cuda_engine(engine)
            
        self.engine = engine
        if self.engine is not None:
            self.context = self._create_context()
            self._update_name_binindgs_maps()
        self.input_names = input_names
        self.output_names = output_names
        self.input_flattener = input_flattener
        self.output_flattener = output_flattener
    
    def _create_context(self):
        if getattr(self, "device_memory", None) is not None:
            return self.device_memory.add(self.engine)
        return self.engine.create_execution_context()

    def _update_name_binindgs_maps(self):
        if trt_version() >= "10.0":
            self._update_name_binding_maps_trt_10()
        else:
            self._update_name_binding_maps_pre_trt_10()

    def _update_name_binding_maps_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_io_tensors):
            name_i = self.engine.get_tensor_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _update_name_binding_maps_pre_trt_10(self):
        self._name_to_binding = {}
        self._binding_to_name = {}
        for i in range(self.engine.num_bindings):
            name_i = self.engine.get_binding_name(i)
            self._name_to_binding[name_i] = i
            self._binding_to_name[i] = name_i

    def _on_state_dict(self, state_dict, prefix, local_metadata):
        state_dict[prefix + "engine"] = bytearray(self.engine.serialize())
        state_dict[prefix + "input_names"] = self.input_names
        state_dict[prefix + "output_names"] = self.output_names
        state_dict[prefix + "input_flattener"] = self.input_flattener.dict()
        state_dict[prefix + "output_flattener"] = self.output_flattener.dict()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        engine_bytes = state_dict[prefix + "engine"]

        with trt.Logger() as logger, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_bytes)
            if self.engine is not None:
                self.context = self._create_context()

        self.input_names = state_dict[prefix + "input_names"]
        self.output_names = state_dict[prefix + "output_names"]

        if 'input_flattener' in state_dict:
            self.input_flattener = Flattener.from_dict(state_dict['input_flattener'])
        else:
            self.input_flattener = None

        if 'output_flattener' in state_dict:
            self.output_flattener = Flattener.from_dict(state_dict['output_flattener'])
        else:
            self.output_flattener = None

        self._update_name_binindgs_maps()

    def _forward_pre_10(self, *inputs):
        bindings = [None] * (len(self.input_names) + len(self.output_names))
        
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            shape = tuple(inputs[i].shape)
            bindings[idx] = inputs[i].contiguous().data_ptr()
            self.context.set_binding_shape(idx, shape)

        # create output tensors
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            idx = self.engine.get_binding_index(output_name)
            dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch_device_from_trt(self.engine.get_location(idx))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            bindings[idx] = output.data_ptr()

        self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        )

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs

    def _forward_post_10(self, *inputs):
        if self.input_flattener is not None:
            inputs = self.input_flattener.flatten(inputs)

        # set shapes
        for i, input_name in enumerate(self.input_names):
            shape = tuple(inputs[i].shape)
            data_ptr = inputs[i].contiguous().data_ptr()
            self.context.set_tensor_address(input_name, data_ptr)
            self.context.set_input_shape(input_name, shape)

        # execute
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            self.context.set_tensor_address(output_name, output.data_ptr())

        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)

        if self.output_flattener is not None:
            outputs = self.output_flattener.unflatten(outputs)
        else:
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

        return outputs

    def forward(self, *inputs):
        if trt_version() < "10.0":
            return self._forward_pre_10(*inputs)
        else:
            return self._forward_post_10(*inputs)

    def enable_profiling(self):
        if not self.context.profiler:
            self.context.profiler = trt.Profiler()
