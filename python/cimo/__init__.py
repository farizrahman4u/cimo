"""cimo: a C compiler written in pure Python."""

from .driver import compile, load, Module, CompileError, KernelError

__version__ = "0.1.0.dev0"
