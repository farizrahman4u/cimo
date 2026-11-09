"""cimo: a C compiler written in pure Python."""

from .driver import compile, load, Module, CompileError, KernelError, add_cache_entry
from .cpu import features as cpu_features

__version__ = "0.0.4"
