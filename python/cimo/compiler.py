"""cimo.compile(): C source in, Python callables out.

Interim toolchain, until milestones M2 (machine-code encoder + in-process loader) and M3 (cproc
ported to Python) land:

    C --[C cproc]--> QBE IR --[cimo.qbe, Python]--> arm64 asm --[system cc]--> shared library

so for now it needs `cproc`/`cproc-qbe` (CIMO_CPROC_DIR, default ~/code/cproc) and a C compiler
(CC, default cc) on the machine, and links against the host libc, which makes `printf` available.
The Python API is meant to stay the same when those pieces are replaced.
"""

import ctypes
import hashlib
import io
import os
import re
import subprocess
import sys
import tempfile

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


def _qbe(ssa):
    """QBE IR text to assembly text, with the Python QBE port, in this process."""
    from .qbe import defs as D
    from .qbe.main import _Driver, _targets
    from .qbe.parse import parse
    D.T = _targets()[0]
    out = io.StringIO()
    try:
        parse(ssa.encode('latin-1'), '<cimo>', _Driver(out, 0).dbgfile, _Driver(out, 0).data,
              _Driver(out, 0).func)
        D.T.emitfin(out)
    except D.QbeError as e:
        raise CompileError(str(e)) from None
    return out.getvalue()


def _build(src):
    cproc = os.path.expanduser(os.environ.get('CIMO_CPROC_DIR', '~/code/cproc'))
    cc = os.environ.get('CC', 'cc')
    key = hashlib.sha256(src.encode()).hexdigest()[:24]
    so = os.path.join(_cache_dir(), key + '.so')
    sig = so[:-3] + '.sig'
    if os.path.exists(so) and os.path.exists(sig):
        with open(sig) as f:
            return so, f.read()
    with tempfile.TemporaryDirectory() as tmp:
        c = os.path.join(tmp, 'k.c')
        with open(c, 'w') as f:
            f.write(_PRELUDE + src)
        pp = _run([os.path.join(cproc, 'cproc'), '-E', c])
        ssa = _run([os.path.join(cproc, 'cproc-qbe')], stdin=pp).decode('latin-1')
        asm = os.path.join(tmp, 'k.s')
        with open(asm, 'w') as f:
            f.write(_qbe(ssa))
        _run([cc, '-shared', '-o', so + '.tmp', asm])
    sigs = '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                  ','.join(a.split()[0] for a in m.group(3).split(',') if a.strip()))
                     for m in _SIG.finditer(ssa))
    with open(sig, 'w') as f:
        f.write(sigs)
    os.replace(so + '.tmp', so)
    return so, sigs


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
    def __init__(self, cfn, name, ret, params):
        self._f = cfn
        self.__name__ = name
        self._params = params
        cfn.restype = _CTYPE.get(ret)
        cfn.argtypes = [_CTYPE[k] for k in params]

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

    def __init__(self, path, sigs):
        self._lib = ctypes.CDLL(path)
        self._kernels = {}
        for line in sigs.splitlines():
            name, ret, params = line.split(' ')
            self._kernels[name] = Kernel(getattr(self._lib, name), name,
                                         None if ret == '-' else ret,
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
