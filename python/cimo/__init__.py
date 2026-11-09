"""cimo: a C compiler written in pure Python."""

from .driver import compile, load, Module, CompileError, KernelError, add_cache_entry
from .cpu import features as cpu_features
from .parallel import set_num_threads, get_num_threads

__version__ = "0.0.5"


def __getattr__(name):
    if name == 'gpu':  # imported on first use: CPU users pay nothing
        import importlib
        return importlib.import_module('.gpu', __name__)
    raise AttributeError("module 'cimo' has no attribute %r" % name)
