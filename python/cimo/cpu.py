"""What this machine's CPU can run beyond its baseline: read with cpuid on x86-64, from the OS on arm64.

cimo compiles where the code runs, so it can use what this CPU has, as `-march=native` does: the
features go into the frontend (the intrinsics a kernel may call, and their feature macros), the backend
(`cimo.backend.ext.x86`) and every cache key, so a cached module never runs on a CPU without them.
`cimo.cpu_features()` is the public query.

x86-64 (beyond SSE2): popcnt, fma, avx, avx2, bmi1, bmi2, lzcnt, f16c, and AVX-512's avx512f, avx512bw,
avx512dq, avx512vl, avx512bf16, avx512vnni, the VEX avxvnni, and AMX's amx-tile, amx-bf16, amx-int8 (gcc's
-m names). AVX and AVX-512 need the OS to save their registers (XCR0), AMX its tile state.

arm64 (beyond NEON and FMA, which are baseline): bf16 (FEAT_BF16: bfdot), dotprod (FEAT_DotProd:
sdot, udot) and i8mm (FEAT_I8MM: usdot), from Linux's HWCAP/HWCAP2 (getauxval, else /proc/cpuinfo), macOS's
hw.optional.arm.FEAT_* sysctls, or Windows' IsProcessorFeaturePresent (else the ID registers in the
registry).

`CIMO_X86=baseline` (or a comma list, `fma,popcnt`) overrides what is detected on x86-64, and `CIMO_ARM64`
the same on arm64, to compare or to build for older machines.
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
    'avx512f': (7, 0, 1, 16),
    'avx512dq': (7, 0, 1, 17),
    'avx512bw': (7, 0, 1, 30),
    'avx512vl': (7, 0, 1, 31),
    'avx512vnni': (7, 0, 2, 11),
    'amx-bf16': (7, 0, 3, 22),
    'amx-tile': (7, 0, 3, 24),
    'amx-int8': (7, 0, 3, 25),
    'avxvnni': (7, 1, 0, 4),
    'avx512bf16': (7, 1, 0, 5),
    'lzcnt': (0x80000001, 0, 2, 5),
}
_AVX512 = frozenset(('avx512f', 'avx512dq', 'avx512bw', 'avx512vl', 'avx512vnni', 'avx512bf16'))
_AMX = frozenset(('amx-tile', 'amx-bf16', 'amx-int8'))
KNOWN = frozenset(('popcnt', 'fma', 'avx', 'avx2', 'bmi1', 'bmi2', 'lzcnt', 'f16c', 'avxvnni')) | _AVX512 | _AMX
ARM64_KNOWN = frozenset(('bf16', 'dotprod', 'i8mm'))

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
_cached_arm = None


def _code(text):
    from .assembler_amd64 import assemble
    obj = assemble('.text\n' + text)
    s = obj.sections
    for x in (s.values() if isinstance(s, dict) else s):
        if getattr(x, 'name', None) in (None, '.text'):
            return bytes(getattr(x, 'data', x))
    raise RuntimeError('no .text')


def _consistent(have, xcr0):
    """have, less what the OS doesn't save the registers of (XCR0) or what lacks the features it extends."""
    have = set(have)
    ymm = 'osxsave' in have and (xcr0 & 6) == 6  # xmm and ymm state saved by the OS
    if not ymm:
        have -= {'avx', 'avx2', 'fma', 'f16c', 'avxvnni'}
    if 'avx' not in have:
        have -= {'avx2', 'fma', 'f16c', 'avxvnni'} | _AVX512
    if not ymm or (xcr0 & 0xe0) != 0xe0 or 'avx512f' not in have:  # opmask, upper zmm0-15, zmm16-31
        have -= _AVX512
    if 'osxsave' not in have or (xcr0 & 0x60000) != 0x60000 or 'amx-tile' not in have:  # XTILECFG, XTILEDATA
        have -= _AMX
    have.discard('osxsave')
    return frozenset(have)


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
        sub7 = leaf(7)[0] if top >= 7 else 0  # the highest subleaf of leaf 7
        have = set()
        for name, (n, sub, r, bit) in _BITS.items():
            if n >= 0x80000000 and n > ext:
                continue
            if n < 0x80000000 and n > top:
                continue
            if n == 7 and sub > sub7:
                continue
            if leaf(n, sub)[r] >> bit & 1:
                have.add(name)
        return _consistent(have, g(0) if 'osxsave' in have else 0)
    finally:
        loader._unmap(mem, page)


def _override(env, known, arch):
    names = frozenset(n.strip() for n in env.split(',') if n.strip() and n.strip() != 'baseline')
    bad = names - known
    if bad:
        raise ValueError('%s: unknown feature %s' % (arch, ', '.join(sorted(bad))))
    return names


def x86_features():
    """This machine's x86-64 features (cpu.KNOWN), or an empty set on any other machine."""
    global _cached
    if _cached is None:
        env = os.environ.get('CIMO_X86')
        from .loader import _X86
        if env is not None:
            names = _override(env, KNOWN, 'CIMO_X86')
            _cached = names if _X86 else frozenset()
        else:
            _cached = _detect() if _X86 else frozenset()
    return _cached


# Linux arm64: AT_HWCAP's HWCAP_ASIMDDP, AT_HWCAP2's HWCAP2_I8MM and HWCAP2_BF16 (asm/hwcap.h), and their names in
# /proc/cpuinfo's Features
_AT_HWCAP, _AT_HWCAP2 = 16, 26
_HWCAP = {'dotprod': (_AT_HWCAP, 1 << 20), 'i8mm': (_AT_HWCAP2, 1 << 13), 'bf16': (_AT_HWCAP2, 1 << 14)}
_CPUINFO = {'dotprod': 'asimddp', 'i8mm': 'i8mm', 'bf16': 'bf16'}
# macOS: sysctlbyname
_SYSCTL = {'dotprod': 'hw.optional.arm.FEAT_DotProd', 'i8mm': 'hw.optional.arm.FEAT_I8MM',
           'bf16': 'hw.optional.arm.FEAT_BF16'}
# Windows: IsProcessorFeaturePresent's PF_ARM_V82_DP_INSTRUCTIONS_AVAILABLE, PF_ARM_V82_I8MM_INSTRUCTIONS_AVAILABLE,
# PF_ARM_V86_BF16_INSTRUCTIONS_AVAILABLE; older Windows answers 0 for numbers it doesn't know, so also the ID
# register fields the registry keeps (CP 4030: ID_AA64ISAR0_EL1, DP at bits 47:44; CP 4031: ID_AA64ISAR1_EL1, BF16 at
# 47:44, I8MM at 55:52)
_PF = {'dotprod': 43, 'i8mm': 66, 'bf16': 68}
_IDREG = {'dotprod': ('CP 4030', 44), 'i8mm': ('CP 4031', 52), 'bf16': ('CP 4031', 44)}


def _linux_arm64():
    try:
        libc = ctypes.CDLL(None)
        getauxval = libc.getauxval
        getauxval.restype = ctypes.c_ulong
        getauxval.argtypes = [ctypes.c_ulong]
        caps = {n: getauxval(n) for n in (_AT_HWCAP, _AT_HWCAP2)}
        return frozenset(f for f, (at, bit) in _HWCAP.items() if caps[at] & bit)
    except (OSError, AttributeError):
        pass
    try:
        with open('/proc/cpuinfo') as fh:
            for line in fh:
                if line.startswith('Features'):
                    words = set(line.split(':', 1)[1].split())
                    return frozenset(f for f, w in _CPUINFO.items() if w in words)
    except OSError:
        pass
    return frozenset()


def _macos_arm64():
    try:
        libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
    except OSError:
        return frozenset()
    have = set()
    for f, name in _SYSCTL.items():
        v = ctypes.c_int(0)
        n = ctypes.c_size_t(ctypes.sizeof(v))
        if libc.sysctlbyname(name.encode(), ctypes.byref(v), ctypes.byref(n), None, ctypes.c_size_t(0)) == 0 and v.value:
            have.add(f)
    return frozenset(have)


def _windows_arm64():
    have = set()
    try:
        k32 = ctypes.windll.kernel32
        have = {f for f, pf in _PF.items() if k32.IsProcessorFeaturePresent(pf)}
    except (OSError, AttributeError):
        pass
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as h:
            for f, (name, bit) in _IDREG.items():
                try:
                    v = winreg.QueryValueEx(h, name)[0]
                except OSError:
                    continue
                if isinstance(v, int) and (v >> bit) & 15:
                    have.add(f)
    except (OSError, ImportError):
        pass
    return frozenset(have)


def _detect_arm64():
    if sys.platform == 'darwin':
        return _macos_arm64()
    if sys.platform == 'win32':
        return _windows_arm64()
    if sys.platform.startswith('linux'):
        return _linux_arm64()
    return frozenset()


def arm64_features():
    """This machine's arm64 features (cpu.ARM64_KNOWN), or an empty set on any other machine."""
    global _cached_arm
    if _cached_arm is None:
        from .loader import _ARM64
        env = os.environ.get('CIMO_ARM64')
        if env is not None:
            names = _override(env, ARM64_KNOWN, 'CIMO_ARM64')
            _cached_arm = names if _ARM64 else frozenset()
        else:
            _cached_arm = _detect_arm64() if _ARM64 and sys.platform != 'emscripten' else frozenset()
    return _cached_arm


def features():
    """The features of this machine's CPU that cimo can use (x86-64 or arm64 names, as gcc's -m flags)."""
    return x86_features() | arm64_features()


def key():
    """The features as bytes, for cache keys."""
    return ','.join(sorted(features())).encode()
