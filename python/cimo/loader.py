"""Link a `cimo.assembler` object into executable memory of the running process (arm64).

    img = Image(obj)                      # lay out, resolve, relocate, map, flush
    f = img.function('sum', ctypes.c_double, [ctypes.c_void_p, ctypes.c_int64])

Memory layout: one region for code (the object's executable sections, then 16-byte stubs that
reach external functions), page aligned, followed by one RW region for data (GOT slots, the
other sections, bss, common symbols). The image stays mapped until the `Image` is closed or
garbage collected; keep a reference for as long as its functions may be called.

Executable memory ("Platform loading findings" in requirements.md):
- Linux: anonymous mmap RW, copy code and data in, mprotect the code pages RX (W^X), then
  flush the instruction cache over the code with our own `dc cvau` / `ic ivau` routine.
- Windows: VirtualAlloc RW, copy, VirtualProtect the code pages PAGE_EXECUTE_READ, then
  FlushInstructionCache.
- macOS: code in an `mmap(MAP_JIT)` region written between `pthread_jit_write_protect_np(0)`
  and `(1)`, data in a separate RW mapping placed right after it (needs to stay within adrp
  range, +-4 GB); then `sys_icache_invalidate`. If MAP_JIT fails, or the data mapping lands out
  of range, fall back to the Linux W^X scheme (RW, then mprotect RX). Plain RWX is refused on
  macOS and never used.

External symbols (interim: `printf`, `memcpy`, libm, ...) are looked up with dlsym on the
process (and libm); calls reach them through stubs `ldr x16, 8; br x16; .quad addr`, since
they are beyond branch range. x16 is the intra-procedure-call scratch register, which AAPCS64
lets veneers clobber.
"""

import ctypes
import ctypes.util
import mmap as _mmap
import platform
import struct
import sys
import threading

__all__ = ['Image', 'LoadError', 'flush_icache']

PROT_READ, PROT_WRITE, PROT_EXEC = 1, 2, 4
MAP_PRIVATE = 0x02
_MAC = sys.platform == 'darwin'
_WIN = sys.platform == 'win32'
MAP_ANON = 0x1000 if _MAC else 0x20
MAP_JIT = 0x800
PAGE = _mmap.PAGESIZE
_ARM64 = platform.machine().lower() in ('arm64', 'aarch64')
_FAILED = (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1


class LoadError(Exception):
    pass


class _RangeError(LoadError):
    pass


if not _WIN:
    _libc = ctypes.CDLL(None, use_errno=True)
    _c_mmap = _libc.mmap
    _c_mmap.restype = ctypes.c_void_p
    _c_mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                        ctypes.c_int, ctypes.c_long]
    _c_mprotect = _libc.mprotect
    _c_mprotect.restype = ctypes.c_int
    _c_mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _c_munmap = _libc.munmap
    _c_munmap.restype = ctypes.c_int
    _c_munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
else:
    _libc = None
    from ctypes import wintypes as _wt
    _k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _VirtualAlloc = _k32.VirtualAlloc
    _VirtualAlloc.restype = ctypes.c_void_p
    _VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, _wt.DWORD, _wt.DWORD]
    _VirtualProtect = _k32.VirtualProtect
    _VirtualProtect.restype = _wt.BOOL
    _VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, _wt.DWORD,
                                ctypes.POINTER(_wt.DWORD)]
    _VirtualFree = _k32.VirtualFree
    _VirtualFree.restype = _wt.BOOL
    _VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, _wt.DWORD]
    _FlushInstructionCache = _k32.FlushInstructionCache
    _FlushInstructionCache.restype = _wt.BOOL
    _FlushInstructionCache.argtypes = [_wt.HANDLE, ctypes.c_void_p, ctypes.c_size_t]
    _GetCurrentProcess = _k32.GetCurrentProcess
    _GetCurrentProcess.restype = _wt.HANDLE
    MEM_COMMIT, MEM_RESERVE, MEM_RELEASE = 0x1000, 0x2000, 0x8000
    _PAGE = {PROT_READ | PROT_WRITE: 0x04, PROT_READ | PROT_EXEC: 0x20,  # READWRITE, EXECUTE_READ
             PROT_READ: 0x02}


def _map(size, prot, flags=0, hint=None):
    """Anonymous zeroed memory (VirtualAlloc on Windows, where flags and hint are ignored)."""
    if _WIN:
        a = _VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, _PAGE[prot])
        if not a:
            raise OSError(ctypes.get_last_error(), 'VirtualAlloc failed')
        return a
    a = _c_mmap(hint, size, prot, MAP_PRIVATE | MAP_ANON | flags, -1, 0)
    if a is None or a == _FAILED:
        raise OSError(ctypes.get_errno(), 'mmap failed')
    return a


def _protect(addr, size, prot):
    if _WIN:
        old = _wt.DWORD()
        if not _VirtualProtect(addr, size, _PAGE[prot], ctypes.byref(old)):
            raise OSError(ctypes.get_last_error(), 'VirtualProtect failed')
        return
    if _c_mprotect(addr, size, prot) != 0:
        raise OSError(ctypes.get_errno(), 'mprotect failed')


def _unmap(addr, size):
    """Release a mapping made by _map (the whole of it)."""
    if _WIN:
        _VirtualFree(addr, 0, MEM_RELEASE)
    else:
        _c_munmap(addr, size)


# ---------------------------------------------------------------------------------------------
# instruction cache

# flush(start, end): clean the data cache to the point of unification and invalidate the
# instruction cache over [start, end), line sizes from CTR_EL0 (IDC/DIC bits skip the loops when
# the core doesn't need them). Encodings from GNU as; test_loader.py rechecks them.
_FLUSH_ASM = """\
	mrs	x2, ctr_el0
	tbnz	x2, #28, 1f
	ubfx	x3, x2, #16, #4
	mov	x4, #4
	lsl	x4, x4, x3
	sub	x3, x4, #1
	bic	x3, x0, x3
0:	dc	cvau, x3
	add	x3, x3, x4
	cmp	x3, x1
	b.lo	0b
1:	dsb	ish
	tbnz	x2, #29, 3f
	and	x3, x2, #15
	mov	x4, #4
	lsl	x4, x4, x3
	sub	x3, x4, #1
	bic	x3, x0, x3
2:	ic	ivau, x3
	add	x3, x3, x4
	cmp	x3, x1
	b.lo	2b
	dsb	ish
3:	isb
	ret
"""
_FLUSH_CODE = struct.pack(
    '<25I', 0xd53b0022, 0x37e00142, 0xd3504c43, 0xd2800084, 0x9ac32084, 0xd1000483,
    0x8a230003, 0xd50b7b23, 0x8b040063, 0xeb01007f, 0x54ffffa3, 0xd5033b9f, 0x37e80162,
    0x92400c43, 0xd2800084, 0x9ac32084, 0xd1000483, 0x8a230003, 0xd50b7523, 0x8b040063,
    0xeb01007f, 0x54ffffa3, 0xd5033b9f, 0xd5033fdf, 0xd65f03c0)

_flush_fn = None
_flush_lock = threading.Lock()


def _linux_flush_fn():
    """The flush routine itself is code, so how does it get into the I-cache safely? It lives on
    its own page, written while RW and then made executable for the first time with mprotect.
    Linux arm64 makes a user page I-cache coherent when it first becomes executable
    (set_pte_at -> __sync_icache_dcache), and the page is never written or unmapped again, so
    the routine is coherent before its first call. If libgcc's __clear_cache is already loaded
    in the process we additionally run it over the page, belt and braces."""
    global _flush_fn
    with _flush_lock:
        if _flush_fn is None:
            page = _map(PAGE, PROT_READ | PROT_WRITE)
            ctypes.memmove(page, _FLUSH_CODE, len(_FLUSH_CODE))
            _protect(page, PAGE, PROT_READ | PROT_EXEC)
            try:
                cc = _libc['__clear_cache']
            except AttributeError:
                pass
            else:
                cc.restype = None
                cc.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                cc(page, page + len(_FLUSH_CODE))
            _flush_fn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(page)
        return _flush_fn


def flush_icache(addr, size):
    """Make [addr, addr+size) coherent for instruction fetch after writing code there."""
    if not _ARM64 or size <= 0:
        return
    if _WIN:
        _FlushInstructionCache(_GetCurrentProcess(), addr, size)
    elif _MAC:
        f = _libc.sys_icache_invalidate
        f.restype = None
        f.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        f(addr, size)
    else:
        _linux_flush_fn()(addr, addr + size)


# ---------------------------------------------------------------------------------------------
# external symbols

_libs = None


def _dlsym(name):
    global _libs
    if _libs is None:
        # the process: libc, and libm on macOS (libSystem); the Universal CRT on Windows
        _libs = [_libc] if _libc else [ctypes.CDLL('ucrtbase')]
        if not _MAC and not _WIN:
            for m in ('libm.so.6', 'libm.so', None):
                try:
                    _libs.append(ctypes.CDLL(m or ctypes.util.find_library('m')))
                    break
                except (OSError, TypeError):
                    pass
    for lib in _libs:
        try:
            return ctypes.cast(lib[name], ctypes.c_void_p).value
        except AttributeError:
            pass
    return None


# ---------------------------------------------------------------------------------------------
# relocations

def _rd(buf, off):
    return struct.unpack_from('<I', buf, off)[0]


def _wr(buf, off, w):
    struct.pack_into('<I', buf, off, w)


def _apply(buf, off, P, typ, S, sym):
    """Patch the relocation at buf[off] (runtime address P) for target value S (= symbol + addend,
    or the GOT slot address for GOT relocations)."""
    if typ in ('CALL26', 'JUMP26'):
        d = S - P
        if d & 3 or not -(1 << 27) <= d < 1 << 27:
            raise _RangeError('branch to %s out of range' % sym)
        _wr(buf, off, (_rd(buf, off) & 0xfc000000) | ((d >> 2) & 0x3ffffff))
    elif typ == 'CONDBR19':
        d = S - P
        if d & 3 or not -(1 << 20) <= d < 1 << 20:
            raise _RangeError('branch to %s out of range' % sym)
        _wr(buf, off, (_rd(buf, off) & 0xff00001f) | ((d >> 2) & 0x7ffff) << 5)
    elif typ in ('ADR_PREL_PG_HI21', 'ADR_GOT_PAGE'):
        d = (S & ~0xfff) - (P & ~0xfff)
        if not -(1 << 32) <= d < 1 << 32:
            raise _RangeError('%s is out of adrp range (+-4 GB) of the code (direct reference '
                              'to an external symbol; extern data needs a GOT access)' % sym)
        imm = (d >> 12) & 0x1fffff
        _wr(buf, off, (_rd(buf, off) & 0x9f00001f) | (imm & 3) << 29 | (imm >> 2) << 5)
    elif typ == 'ADD_ABS_LO12_NC':
        _wr(buf, off, (_rd(buf, off) & 0xffc003ff) | (S & 0xfff) << 10)
    elif typ.startswith('LDST') or typ == 'LD64_GOT_LO12_NC':
        sz = {'LDST8_ABS_LO12_NC': 0, 'LDST16_ABS_LO12_NC': 1, 'LDST32_ABS_LO12_NC': 2,
              'LDST64_ABS_LO12_NC': 3, 'LDST128_ABS_LO12_NC': 4, 'LD64_GOT_LO12_NC': 3}[typ]
        lo = S & 0xfff
        if lo & ((1 << sz) - 1):
            raise LoadError('misaligned %s relocation against %s' % (typ, sym))
        _wr(buf, off, (_rd(buf, off) & 0xffc003ff) | (lo >> sz) << 10)
    elif typ == 'ABS64':
        struct.pack_into('<Q', buf, off, S & ((1 << 64) - 1))
    elif typ in ('ABS32', 'ABS16'):
        n = 4 if typ == 'ABS32' else 2
        if not -(1 << (8 * n - 1)) <= S < 1 << (8 * n):
            raise LoadError('address of %s does not fit a %d-byte value' % (sym, n))
        buf[off:off + n] = (S & ((1 << (8 * n)) - 1)).to_bytes(n, 'little')
    else:
        raise LoadError('unsupported relocation %s' % typ)


def _align(x, a):
    return (x + a - 1) & -a


_SKIP = ('.note', '.debug', '.comment')


class Image:
    """`obj` linked into this process. `resolve(name)` overrides external symbol lookup (C name,
    without the Mach-O '_'); it returns an address or None to fall back to dlsym."""

    def __init__(self, obj, resolve=None):
        if not _ARM64:
            raise LoadError('cimo loads arm64 code only; this machine is %s' % platform.machine())
        self._maps = []
        self.prefix = obj.prefix
        self._layout(obj)
        self._link(obj, resolve)

    # layout: offsets within the code and data regions

    def _layout(self, obj):
        secs = [s for s in obj.sections if s.size and not s.name.startswith(_SKIP)]
        place = {}  # section -> ('c' or 'd', offset)
        off = 0
        for s in secs:
            if s.exec:
                off = _align(off, max(s.align, 4))
                place[s] = ('c', off)
                off += s.size
        stubs = {}  # external symbol -> offset of its stub
        got = {}  # symbol -> offset of its GOT slot in the data region
        for s in secs:
            for r in s.relocs:
                y = obj.symbols[r.sym]
                if r.type in ('CALL26', 'JUMP26') and not y.defined and not y.common:
                    stubs.setdefault(r.sym, None)
                elif r.type in ('ADR_GOT_PAGE', 'LD64_GOT_LO12_NC'):
                    got.setdefault(r.sym, None)
        off = _align(off, 8)
        for k in stubs:
            stubs[k] = off
            off += 16
        self.code_size = _align(off, PAGE) if off else 0
        doff = 0
        for k in got:
            got[k] = doff
            doff += 8
        for nobits in (False, True):
            for s in secs:
                if not s.exec and s.nobits == nobits:
                    doff = _align(doff, max(s.align, 1))
                    place[s] = ('d', doff)
                    doff += s.size
        commons = {}
        for y in obj.symbols.values():
            if y.common and not y.defined:
                size, al = y.common
                doff = _align(doff, al)
                commons[y.name] = doff
                doff += size
        self.data_size = _align(doff, PAGE) if doff else 0
        self._secs, self._place, self._stubs, self._got, self._commons = (
            secs, place, stubs, got, commons)

    # mapping, symbol resolution, relocation

    def _link(self, obj, resolve):
        csize, dsize = self.code_size, self.data_size
        if _MAC:
            try:
                code = _map(csize, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_JIT) if csize else 0
                self._maps.append((code, csize))
                data = _map(dsize, PROT_READ | PROT_WRITE, 0, code + csize) if dsize else 0
                self._maps.append((data, dsize))
                cimg, dimg = self._build(obj, resolve, code, data)
                jw = _libc.pthread_jit_write_protect_np
                jw.restype = None
                jw.argtypes = [ctypes.c_int]
                jw(0)
                try:
                    ctypes.memmove(code, bytes(cimg), csize)
                finally:
                    jw(1)
                ctypes.memmove(data, bytes(dimg), len(dimg))
                flush_icache(code, csize)
                return
            except (OSError, _RangeError, AttributeError):
                self.close()  # fall back to W^X
        base = _map(csize + dsize, PROT_READ | PROT_WRITE)
        self._maps.append((base, csize + dsize))
        cimg, dimg = self._build(obj, resolve, base, base + csize)
        ctypes.memmove(base, bytes(cimg), csize)
        ctypes.memmove(base + csize, bytes(dimg), len(dimg))
        if csize:
            try:
                _protect(base, csize, PROT_READ | PROT_EXEC)
            except OSError as e:
                raise LoadError('cannot make memory executable (%s)%s' % (
                    e, '; under the macOS hardened runtime the process needs the '
                       'com.apple.security.cs.allow-jit entitlement' if _MAC else '')) from None
            flush_icache(base, csize)

    def _build(self, obj, resolve, code, data):
        """Images (bytearrays) of the code and data regions for these base addresses; also fills
        self.symbols with the address of every defined symbol."""
        base = {'c': code, 'd': data}
        addr = {}
        for y in obj.symbols.values():
            if y.defined:
                if y.section not in self._place:  # empty or skipped section
                    if y.section.size == 0:
                        continue
                    raise LoadError('symbol %s is in unloaded section %s'
                                    % (y.name, y.section.name))
                k, o = self._place[y.section]
                addr[y.name] = base[k] + o + y.value
            elif y.name in self._commons:
                addr[y.name] = data + self._commons[y.name]
        self.symbols = dict(addr)

        def ext(name):
            if name in addr:
                return addr[name]
            c = name[len(self.prefix):] if name.startswith(self.prefix) else name
            a = resolve(c) if resolve else None
            if a is None:
                a = _dlsym(c)
            if a is None:
                raise LoadError('undefined symbol: %s' % c)
            addr[name] = a
            return a

        cimg = bytearray(self.code_size)
        dimg = bytearray(self.data_size)
        img = {'c': cimg, 'd': dimg}
        for s in self._secs:
            if not s.nobits:
                k, o = self._place[s]
                img[k][o:o + s.size] = s.data
        for name, o in self._stubs.items():
            struct.pack_into('<IIQ', cimg, o, 0x58000050, 0xd61f0200, ext(name))
        for name, o in self._got.items():
            struct.pack_into('<Q', dimg, o, ext(name))
        for s in self._secs:
            k, so = self._place[s]
            for r in s.relocs:
                off = so + r.offset
                P = base[k] + off
                if r.type in ('ADR_GOT_PAGE', 'LD64_GOT_LO12_NC'):
                    S = data + self._got[r.sym]
                elif r.sym in self._stubs and r.type in ('CALL26', 'JUMP26'):
                    S = code + self._stubs[r.sym] + r.addend
                else:
                    S = ext(r.sym) + r.addend
                _apply(img[k], off, P, r.type, S, r.sym)
        return cimg, dimg

    # access

    def address(self, name):
        """Address of a defined symbol, by C name (without the Mach-O '_')."""
        try:
            return self.symbols[self.prefix + name]
        except KeyError:
            raise LoadError('no symbol %s in this image' % name) from None

    def function(self, name, restype, argtypes):
        return ctypes.CFUNCTYPE(restype, *argtypes)(self.address(name))

    def close(self):
        """Unmap the image. Its functions must not be called afterwards."""
        maps, self._maps = self._maps, []
        for a, n in maps:
            if a and n:
                _unmap(a, n)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
