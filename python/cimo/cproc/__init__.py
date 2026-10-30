"""cproc (C11 front end, emits QBE IR) ported to Python. See PORTING.md.

    compile(text, filename='<stdin>', target='aarch64') -> str

takes preprocessed C and returns QBE IR; errors raise CprocError whose `msg` is
cproc's exact error message.
"""

from .util import CprocError
from .main import compile, main, DEFAULT_TARGET

__all__ = ['compile', 'main', 'CprocError', 'DEFAULT_TARGET']
