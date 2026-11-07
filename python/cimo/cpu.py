"""What this machine's x86-64 CPU can run beyond the baseline (SSE2), read with cpuid.

cimo compiles where the code runs, so it can use what this CPU has, as `-march=native` does: the
features go into the backend (`cimo.backend.ext.x86`) and into every cache key, so a cached module
never runs on a CPU without them. arm64 has no such choice to make: FMA and NEON are baseline.

`CIMO_X86=baseline` (or a comma list, `fma,popcnt`) overrides what is detected, to compare or to
build for older machines.
"""

import ctypes
import os
import sys

# feature: (leaf, subleaf, register index in eax, ebx, ecx, edx, bit)
_BITS = {
    'popcnt': (1, 0, 2, 23),
    'fma': (1, 0, 2, 12),
    'osxsave': (1, 0, 2, 27),
    'avx': (1, 0, 2, 28),
    'f16c': (1, 0, 2, 29),
    'bmi1': (7, 0, 1, 3),
    'avx2': (7, 0, 1, 5),
    'bmi2': (7, 0, 1, 8),
    'lzcnt': (0x80000001, 0, 2, 5),
}
KNOWN = frozenset(('popcnt', 'fma', 'avx', 'avx2', 'bmi1', 'bmi2', 'lzcnt', 'f16c'))

# (leaf, subleaf, out) -> stores eax, ebx, ecx, edx at out; and xgetbv(0) -> XCR0. Assembled by
# cimo's own assembler, for the calling convention of this platform.
_CPUID = {
    False: 'push %rbx\nmovl %edi, %eax\nmovl %esi, %ecx\nmovq %rdx, %r8\ncpuid\n',
    True: 'push %rbx\nmovl %ecx, %eax\nmovl %edx, %ecx\ncpuid\n',
}
_STORE = ('movl %eax, (%r8)\nmovl %ebx, 4(%r8)\nmovl %ecx, 8(%r8)\nmovl %edx, 12(%r8)\n'
          'pop %rbx\nret\n')
_XGETBV = {False: 'movl %edi, %ecx\n', True: ''}
_XGETBV_TAIL = 'xgetbv\nshlq $32, %rdx\norq %rdx, %rax\nret\n'

_cached = None


def _code(text):
    from .assembler_amd64 import assemble
    obj = assemble('.text\n' + text)
    s = obj.sections
    for x in (s.values() if isinstance(s, dict) else s):
        if getattr(x, 'name', None) in (None, '.text'):
            return bytes(getattr(x, 'data', x))
    raise RuntimeError('no .text')


def _detect():
    """The features this CPU and its OS support (AVX needs the OS to save ymm registers)."""
    from . import loader
    win = sys.platform == 'win32'
    cpuid = _code(_CPUID[win] + _STORE)
    xgetbv = _code(_XGETBV[win] + _XGETBV_TAIL)
    page = loader._PAGE_SIZE if hasattr(loader, '_PAGE_SIZE') else 4096
    mem = loader._map(page, loader.PROT_READ | loader.PROT_WRITE)
    try:
        ctypes.memmove(mem, cpuid, len(cpuid))
        ctypes.memmove(mem + 64, xgetbv, len(xgetbv))
        loader._protect(mem, page, loader.PROT_READ | loader.PROT_EXEC)
        f = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)(mem)
        g = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_uint32)(mem + 64)
        out = (ctypes.c_uint32 * 4)()
        regs = {}

        def leaf(n, sub=0):
            if (n, sub) not in regs:
                f(n, sub, ctypes.addressof(out))
                regs[(n, sub)] = tuple(out)
            return regs[(n, sub)]

        top = leaf(0)[0]
        ext = leaf(0x80000000)[0]
        have = set()
        for name, (n, sub, r, bit) in _BITS.items():
            if n >= 0x80000000 and n > ext:
                continue
            if n < 0x80000000 and n > top:
                continue
            if leaf(n, sub)[r] >> bit & 1:
                have.add(name)
        ymm = 'osxsave' in have and (g(0) & 6) == 6  # xmm and ymm state saved by the OS
        if not ymm:
            have -= {'avx', 'avx2', 'fma', 'f16c'}
        if 'avx' not in have:
            have -= {'avx2', 'fma', 'f16c'}
        have.discard('osxsave')
        return frozenset(have)
    finally:
        loader._unmap(mem, page)


def x86_features():
    """This machine's x86-64 features (cpu.KNOWN), or an empty set on any other machine."""
    global _cached
    if _cached is None:
        env = os.environ.get('CIMO_X86')
        from .loader import _X86
        if env is not None:
            names = frozenset(n for n in env.split(',') if n and n != 'baseline')
            bad = names - KNOWN
            if bad:
                raise ValueError('CIMO_X86: unknown feature %s' % ', '.join(sorted(bad)))
            _cached = names if _X86 else frozenset()
        else:
            _cached = _detect() if _X86 else frozenset()
    return _cached


def key():
    """The features as bytes, for cache keys."""
    return ','.join(sorted(x86_features())).encode()
