import logging

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

logger = logging.getLogger(__name__)


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

    @staticmethod
    def _new_user_managed_context(engine):
        """Create a context whose device memory the application supplies.

        How you ask for one has changed twice, so all three spellings are tried
        newest first: the runtime-config route is what TensorRT 11 documents and
        10.x also accepts, the allocation-strategy argument is the 10.x form, and
        createExecutionContextWithoutDeviceMemory is the pre-10 entry point --
        deprecated in 10.0 and *removed* in 11.0, so it must never be the only
        path. Returns None when none of them are available.
        """
        strategy = getattr(trt, "ExecutionContextAllocationStrategy", None)
        user_managed = getattr(strategy, "USER_MANAGED", None)

        if user_managed is not None and hasattr(engine, "create_runtime_config"):
            runtime_config = engine.create_runtime_config()
            runtime_config.set_execution_context_allocation_strategy(user_managed)
            return engine.create_execution_context(runtime_config)

        if user_managed is not None:
            return engine.create_execution_context(user_managed)

        legacy = getattr(
            engine, "create_execution_context_without_device_memory", None
        )
        return legacy() if legacy is not None else None

    @staticmethod
    def _bind_context(context, address, size):
        """Point one context at ``address``; False when TensorRT has no setter.

        set_device_memory_v2 carries the buffer size and is the TensorRT 11
        spelling; the sizeless-then-sized set_device_memory is the 10.x one.
        """
        setter = getattr(context, "set_device_memory_v2", None) or getattr(
            context, "set_device_memory", None
        )
        if setter is None:
            return False
        setter(address, size)
        return True

    def add(self, engine):
        """Create a context for ``engine`` that draws on this shared buffer.

        Returns None when this TensorRT cannot hand out a context with
        application-supplied memory, leaving it to the caller to fall back to a
        private pool. Sharing is an optimization, so an unsupported TensorRT has
        to cost footprint, never the load itself.
        """
        try:
            context = self._new_user_managed_context(engine)
        except (AttributeError, TypeError) as err:
            # A route that exists but does not accept what we pass it. Treated
            # the same as absent rather than propagating, for the reason above.
            logger.debug("User-managed execution context unavailable: %s", err)
            return None
        if context is None:
            logger.debug(
                "This TensorRT cannot create a user-managed execution context; "
                "the engine keeps a private device-memory pool."
            )
            return None

        self._contexts.append(context)
        if not self._reserve(self._engine_nbytes(engine)):
            # No way to point the context at our buffer, so it has no memory at
            # all: drop it and let the caller make a normally-allocated one.
            self._contexts.pop()
            return None
        return context

    def _reserve(self, nbytes):
        """Ensure the buffer holds ``nbytes`` and every context points at it."""
        if self._buffer is not None and nbytes <= self.nbytes:
            return self._bind(self._contexts[-1:])
        # Growing means a new allocation, so every context already bound to the
        # old buffer is now pointing at memory the allocator may hand out again.
        # Re-bind all of them, and only drop the old buffer afterwards.
        previous = self._buffer
        self._buffer = torch.empty(
            max(nbytes, 1), dtype=torch.uint8, device=self._device or "cuda"
        )
        bound = self._bind(self._contexts)
        del previous
        return bound

    def _bind(self, contexts):
        address = self._buffer.data_ptr()
        size = self._buffer.numel()
        # A list, not a generator: all() short-circuits, and every context must
        # actually be bound rather than abandoned after the first failure.
        return all([self._bind_context(context, address, size) for context in contexts])


def _describe_profile(engine, context, name):
    """Return the optimization-profile bounds for input ``name``, if any."""
    try:
        index = context.active_optimization_profile
        low, opt, high = engine.get_tensor_profile_shape(name, index)
    except Exception:  # pragma: no cover - varies across TRT versions
        return "profile unknown"
    return f"profile min={tuple(low)} opt={tuple(opt)} max={tuple(high)}"


def _describe_io(engine, context):
    """Return a per-tensor summary of the context's current shapes."""
    lines = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        try:
            shape = tuple(context.get_tensor_shape(name))
        except Exception:  # pragma: no cover - varies across TRT versions
            shape = "unknown"
        lines.append(f"  {name}: {shape}")
    return "\n".join(lines)


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
        pool = getattr(self, "device_memory", None)
        if pool is not None:
            context = pool.add(self.engine)
            if context is not None:
                return context
            # Sharing is unsupported here. A private pool per context runs
            # correctly, just with the footprint sharing was meant to avoid.
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

        # TensorRT records only the raw address of each input, so every
        # tensor it points at must stay alive until the enqueue below returns.
        # A non-contiguous input makes .contiguous() a fresh copy; without a
        # reference that copy is freed at the end of the statement and torch's
        # caching allocator can hand the block straight to the output
        # allocations that follow, leaving the engine to read its own output as
        # input. ``held`` keeps them alive; it is load-bearing, not dead code,
        # and must stay in scope until this method returns.
        held = []
        for i, input_name in enumerate(self.input_names):
            idx = self.engine.get_binding_index(input_name)
            contiguous = inputs[i].contiguous()
            held.append(contiguous)
            bindings[idx] = contiguous.data_ptr()
            shape = tuple(contiguous.shape)
            if not self.context.set_binding_shape(idx, shape):
                raise RuntimeError(
                    f"TensorRT rejected shape {shape} for input '{input_name}'."
                )

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

        if not self.context.execute_async_v2(
            bindings, torch.cuda.current_stream().cuda_stream
        ):
            raise RuntimeError(
                "TensorRT failed to enqueue the engine; the output tensors hold "
                "uninitialised memory."
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
        # TensorRT records only the raw address of each input, so every
        # tensor it points at must stay alive until the enqueue below returns.
        # A non-contiguous input makes .contiguous() a fresh copy; without a
        # reference that copy is freed at the end of the statement and torch's
        # caching allocator can hand the block straight to the output
        # allocations that follow, leaving the engine to read its own output as
        # input. ``held`` keeps them alive; it is load-bearing, not dead code,
        # and must stay in scope until this method returns.
        held = []
        for i, input_name in enumerate(self.input_names):
            contiguous = inputs[i].contiguous()
            held.append(contiguous)
            shape = tuple(contiguous.shape)
            # These return a status rather than raising. Ignoring it is how a
            # rejected shape turns into silent corruption: the context keeps
            # its previous shape, the enqueue below does nothing, and the
            # freshly allocated (uninitialised) output tensor is returned as if
            # it held results -- frequently containing the previous call's
            # output, since the allocator hands back the block just freed.
            if not self.context.set_tensor_address(input_name, contiguous.data_ptr()):
                raise RuntimeError(
                    f"TensorRT rejected the address for input '{input_name}'."
                )
            if not self.context.set_input_shape(input_name, shape):
                raise RuntimeError(
                    f"TensorRT rejected shape {shape} for input '{input_name}' "
                    f"({_describe_profile(self.engine, self.context, input_name)}). "
                    "The engine cannot run at this shape."
                )

        # execute
        outputs = [None] * len(self.output_names)
        for i, output_name in enumerate(self.output_names):
            dtype = torch_dtype_from_trt(self.engine.get_tensor_dtype(output_name))
            shape = tuple(self.context.get_tensor_shape(output_name))
            device = torch_device_from_trt(self.engine.get_tensor_location(output_name))
            # A negative extent means TRT could not resolve this output from the
            # inputs set above. Allocating on it would silently size the buffer
            # wrong, so stop here instead.
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    f"TensorRT left output '{output_name}' with unresolved shape "
                    f"{shape}. Current context I/O:\n"
                    f"{_describe_io(self.engine, self.context)}"
                )
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[i] = output
            if not self.context.set_tensor_address(output_name, output.data_ptr()):
                raise RuntimeError(
                    f"TensorRT rejected the address for output '{output_name}'."
                )

        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError(
                "TensorRT failed to enqueue the engine; the output tensors hold "
                "uninitialised memory. Current context I/O:\n"
                f"{_describe_io(self.engine, self.context)}"
            )

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
