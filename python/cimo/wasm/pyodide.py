"""Loading cimo wasm modules into Pyodide and calling their kernels.

The module is an Emscripten side module (cimo.wasm.link), loaded with Pyodide's dynamic loader
(`pyodide._module.loadWebAssemblyModule`, a private API: see requirements.md, "Platform
loading findings"). It shares Pyodide's linear memory, so kernels work on numpy arrays,
bytearrays and ctypes buffers in place: a pointer argument is the buffer's address.

The module's imports from Python (cimo/wasm/glue.c): stdout writes, heap chunks for malloc,
and errors (abort, exit, assert, traps), which are recorded and raised as KernelError after the
JavaScript exception thrown through the kernel's frames has unwound them.
"""

import ctypes
import itertools
import sys

import js  # noqa: F401  (Pyodide)
import pyodide_js
from pyodide.ffi import create_proxy, to_js

from ..execution import (KernelError, E_ABORT, E_ASSERT, E_TRAP, E_EXIT)

_ids = itertools.count()
_PY = {'w': int, 'l': int, 's': float, 'd': float}


_wrapper = None


def _wrap():
    global _wrapper
    if _wrapper is None:
        from pyodide.code import run_js
        _wrapper = run_js("(f, t) => (...a) => f(...a.map((x, i) => t[i] === 'l' ? BigInt(x) : x))")
    return _wrapper


def _jsobj(d):
    return to_js(d, dict_converter=js.Object.fromEntries)


class _Fail(Exception):
    pass


def _address(x, keep):
    """A pointer argument as an int: a buffer's address, or an int."""
    if isinstance(x, int):
        return x
    ai = getattr(x, '__array_interface__', None)
    if ai is not None:
        return ai['data'][0]
    if isinstance(x, bytes):
        buf = ctypes.create_string_buffer(x)
        keep.append(buf)
        return ctypes.addressof(buf)
    mv = memoryview(x)
    if mv.readonly:
        buf = ctypes.create_string_buffer(mv.tobytes())
        keep.append(buf)
        return ctypes.addressof(buf)
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


def _cstr(p):
    return ctypes.string_at(p).decode('utf-8', 'replace') if p else None


class WasmModule:
    """A compiled module in Pyodide. Non-static functions are attributes (cimo.Module's
    interface); `exports` is the JavaScript object of the wasm exports."""

    def __init__(self, mod, sigs):
        self._err = None
        self._chunks = {}
        self._proxies = []
        scope = js.Object.new()
        for name, fn in (('__cimo_js_write', self._write), ('__cimo_js_grow', self._grow),
                         ('__cimo_js_release', self._release), ('__cimo_js_fail', self._fail)):
            p = create_proxy(fn)
            self._proxies.append(p)
            js.Reflect.set(scope, name, p)
        flags = _jsobj({'loadAsync': False, 'global': False, 'nodelete': True,
                        'allowUndefined': True})
        self.exports = pyodide_js._module.loadWebAssemblyModule(
            to_js(mod), flags, 'cimo-%d.so' % next(_ids), scope)
        self._getsp = getattr(self.exports, '__cimo_get_sp')  # (getattr: no name mangling)
        self._setsp = getattr(self.exports, '__cimo_set_sp')
        self._flush = getattr(self.exports, '__cimo_flush', None)
        self._kernels = {}
        for line in sigs.splitlines():
            name, ret, params = line.split(' ')
            self._kernels[name] = WasmKernel(self, name, None if ret == '-' else ret,
                                             params.split(',') if params else [])

    # imports

    def _write(self, buf, n):
        sys.stdout.write(ctypes.string_at(buf, n).decode('utf-8', 'replace'))

    def _grow(self, nbytes, kind):
        b = ctypes.create_string_buffer(nbytes + 16)
        p = (ctypes.addressof(b) + 15) & -16
        self._chunks[p] = b
        return p

    def _release(self, p):
        self._chunks.pop(p, None)

    def _fail(self, code, expr, file, line, func):
        self._err = (code, _cstr(expr), _cstr(file), line, _cstr(func))
        raise _Fail('cimo kernel error')

    # interface

    def error(self, kernel, exc):
        """Raise KernelError for the recorded error, or for the wasm trap exc."""
        err, self._err = self._err, None
        if err is None:
            msg = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
            raise KernelError('trap: %s (calling %s)' % (msg, kernel), 'trap', kernel) from None
        code, expr, file, line, func = err
        at = ' in %s()' % func if func else ''
        status = None
        if code == E_ABORT:
            kind, msg = 'abort', 'abort() called%s' % at
        elif code == E_EXIT:
            status = ctypes.c_int32(line).value
            kind, msg = 'exit', 'exit(%d) called%s' % (status, at)
        elif code == E_ASSERT:
            kind, msg = 'assert', 'assertion failed: %s, at %s:%d%s' % (expr, file, line, at)
        elif code == E_TRAP:
            kind, msg = 'trap', 'trap (__builtin_trap or unreachable code)%s' % at
        else:
            kind, msg = 'error', 'kernel error %d%s' % (code, at)
        raise KernelError('%s (calling %s)' % (msg, kernel), kind, kernel, func, expr, file,
                          line if code == E_ASSERT else None, status) from None

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


class WasmKernel:
    def __init__(self, module, name, ret, params):
        self._m = module
        self.__name__ = name
        self._ret = ret
        self._params = params
        self._unsupported = None
        if '...' in params:
            self._unsupported = 'it is variadic'
        elif any(k not in _PY for k in params) or (ret is not None and ret not in _PY):
            self._unsupported = 'it passes or returns a struct or union by value'
        self.function = getattr(module.exports, name)  # the wasm export, callable from JS
        # i64 parameters take a BigInt, which Pyodide never leaves as one (it converts BigInt
        # results back to int): a JavaScript wrapper converts them
        self._f = _wrap()(self.function, ''.join(params))

    def __call__(self, *args):
        if self._unsupported:
            raise TypeError('%s() cannot be called from Python: %s (C callers in the module '
                            'can still use it)' % (self.__name__, self._unsupported))
        if len(args) != len(self._params):
            raise TypeError('%s() takes %d arguments (%d given)'
                            % (self.__name__, len(self._params), len(args)))
        keep = []
        conv = []
        for a, k in zip(args, self._params):
            if k == 'l':
                conv.append(a if isinstance(a, int) else _address(a, keep))
            else:
                conv.append(_PY[k](a))
        m = self._m
        sp = m._getsp()
        try:
            r = self._f(*conv)
        except Exception as e:
            m._setsp(sp)
            m.error(self.__name__, e)
        finally:
            if m._flush is not None:
                m._flush()
        if self._ret == 'l':
            return int(r)
        return r

    def __repr__(self):
        return '<cimo kernel %s (wasm)>' % self.__name__
