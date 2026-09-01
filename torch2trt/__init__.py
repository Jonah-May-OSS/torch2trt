import contextlib
import importlib

import tensorrt as trt

from .converters import *
from .torch2trt import *


def load_plugins():
    # Imported for its registration side effects only.
    importlib.import_module("torch2trt.torch_plugins")

    registry = trt.get_plugin_registry()
    torch2trt_creators = [
        c for c in registry.plugin_creator_list if c.plugin_namespace == "torch2trt"
    ]
    for c in torch2trt_creators:
        registry.register_creator(c, "torch2trt")


with contextlib.suppress(BaseException):
    load_plugins()
