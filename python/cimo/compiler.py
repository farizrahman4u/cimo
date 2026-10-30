"""cimo.compile(): C source in, Python callables out.

Pipeline:

    C --[C cproc]--> QBE IR --[cimo.qbe]--> arm64 asm --[cimo.asm]--> object --[cimo.jit]--> memory

Everything after cproc is Python and runs in this process; until M3 (cproc ported to Python)
lands it still needs `cproc`/`cproc-qbe` (CIMO_CPROC_DIR, default ~/code/cproc). External symbols
(interim: `printf`, libm) are resolved against the running process. The assembly is cached by
content hash; assembling and loading it takes milliseconds.
"""

import ctypes
import hashlib
import io
import os
import platform
import re
import subprocess
import sys
import tempfile

from . import asm as _asm
from . import jit as _jit

__all__ = ['compile', 'load', 'Module', 'CompileError']

_PRELUDE = 'int printf(const char *, ...);\n'
_CTYPE = {'w': ctypes.c_int32, 'l': ctypes.c_int64, 's': ctypes.c_float, 'd': ctypes.c_double}
_SIG = re.compile(r'^export\s*\nfunction\s+(?:([wlsd])\s+)?\$(\w+)\(([^)]*)\)', re.M)


class CompileError(Exception):
    pass


def _cache_dir():
    d = os.environ.get('CIMO_CACHE_DIR') or os.path.join(
        os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'cimo')
    os.makedirs(d, exist_ok=True)
    return d


def _run(cmd, stdin=None):
    p = subprocess.run(cmd, input=stdin, capture_output=True)
    if p.returncode != 0:
        raise CompileError(p.stderr.decode(errors='replace').strip() or 'failed: %s' % cmd[0])
    return p.stdout


_APPLE = sys.platform == 'darwin'
_VERSION = b'cimo-m2-1'  # part of the cache key: bump when the generated assembly changes


def _qbe(ssa):
    """QBE IR text to assembly text, with the Python QBE port, in this process."""
    from .qbe import defs as D
    from .qbe.main import _Driver, _targets
    from .qbe.parse import parse
    D.T = _targets()[1 if _APPLE else 0]  # arm64_apple: Apple's ABI variant
    out = io.StringIO()
    try:
        parse(ssa.encode('latin-1'), '<cimo>', _Driver(out, 0).dbgfile, _Driver(out, 0).data,
              _Driver(out, 0).func)
        D.T.emitfin(out)
    except D.QbeError as e:
        raise CompileError(str(e)) from None
    return out.getvalue()


def _build(src):
    """C source -> (assembly text, signatures), cached by content hash."""
    if platform.machine().lower() not in ('arm64', 'aarch64'):
        raise CompileError('cimo generates arm64 code only; this machine is %s'
                           % platform.machine())
    cproc = os.path.expanduser(os.environ.get('CIMO_CPROC_DIR', '~/code/cproc'))
    key = hashlib.sha256(_VERSION + (b'apple' if _APPLE else b'') + src.encode()).hexdigest()[:24]
    path = os.path.join(_cache_dir(), key + '.s')
    sig = path[:-2] + '.sig'
    if os.path.exists(path) and os.path.exists(sig):
        with open(path) as f, open(sig) as g:
            return f.read(), g.read()
    with tempfile.TemporaryDirectory() as tmp:
        c = os.path.join(tmp, 'k.c')
        with open(c, 'w') as f:
            f.write(_PRELUDE + src)
        pp = _run([os.path.join(cproc, 'cproc'), '-E', c])
        ssa = _run([os.path.join(cproc, 'cproc-qbe')], stdin=pp).decode('latin-1')
    text = _qbe(ssa)
    sigs = '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in _SIG.finditer(ssa))
    for p, data in ((sig, sigs), (path, text)):  # .s last: it marks the entry complete
        with open(p + '.tmp', 'w') as f:
            f.write(data)
        os.replace(p + '.tmp', p)
    return text, sigs


def _address(x):
    """Pointer argument: a buffer (numpy array, bytearray, memoryview, ctypes array) or int."""
    if isinstance(x, int):
        return x
    ai = getattr(x, '__array_interface__', None)
    if ai is not None:
        return ai['data'][0]
    mv = memoryview(x)
    if mv.readonly:
        return ctypes.cast(ctypes.c_char_p(mv.tobytes()), ctypes.c_void_p).value
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class Kernel:
    def __init__(self, image, name, ret, params):
        self._image = image  # keeps the code mapped while the kernel is reachable
        self.address = image.address(name)
        self._f = ctypes.CFUNCTYPE(_CTYPE.get(ret), *[_CTYPE[k] for k in params])(self.address)
        self.__name__ = name
        self._params = params

    def __call__(self, *args):
        if len(args) != len(self._params):
            raise TypeError('%s() takes %d arguments (%d given)'
                            % (self.__name__, len(self._params), len(args)))
        args = [_address(a) if k == 'l' and not isinstance(a, int) else a
                for a, k in zip(args, self._params)]
        try:
            return self._f(*args)
        finally:
            _libc.fflush(None)  # interleave C printf output with Python's
            sys.stdout.flush()

    def __repr__(self):
        return '<cimo kernel %s>' % self.__name__


class Module:
    """Compiled C. Each non-static function is an attribute. Calling the module calls `main`,
    or the only function if there is just one."""

    def __init__(self, text, sigs):
        try:
            obj = _asm.assemble(text, 'macho' if _APPLE else 'elf')
            self._image = _jit.Image(obj)
        except (_asm.AsmError, _jit.LoadError) as e:
            raise CompileError(str(e)) from None
        self._kernels = {}
        for line in sigs.splitlines():
            name, ret, params = line.split(' ')
            self._kernels[name] = Kernel(self._image, name, None if ret == '-' else ret,
                                         params.split(',') if params else [])

    def __getattr__(self, name):
        try:
            return self.__dict__['_kernels'][name]
        except KeyError:
            raise AttributeError(name) from None

    def __dir__(self):
        return list(self._kernels)

    def __call__(self, *args):
        if 'main' in self._kernels:
            return self._kernels['main'](*args)
        if len(self._kernels) == 1:
            return next(iter(self._kernels.values()))(*args)
        raise TypeError('module has no main(); call one of: %s' % ', '.join(self._kernels))


_libc = ctypes.CDLL(None)


def compile(src):
    """Compile C source and return a Module of callables."""
    return Module(*_build(src))


def load(path):
    """Compile a .c file."""
    with open(path) as f:
        return compile(f.read())
