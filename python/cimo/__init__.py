"""cimo: a C compiler written in pure Python."""

from .driver import compile, load, Module, CompileError, KernelError, add_cache_entry

__version__ = "0.0.3"
