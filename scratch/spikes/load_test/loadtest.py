"""Can pure Python put machine code in executable memory and call it?

Tests every loading strategy available on the current platform with a hand-encoded
`long sum(const long *p, long n)`. Stdlib only. Prints one line per strategy.
"""
import ctypes
import mmap
import platform
import sys

ARCH = platform.machine().lower()
IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# arm64 (AAPCS64, same on Linux/macOS/Windows): from gcc -O1, see sum.c
ARM64 = bytes.fromhex(
    "e40300aa" "3f0000f1" "2d010054" "020080d2" "000080d2" "837862f8"
    "0000038b" "42040091" "5f0001eb" "81ffff54" "c0035fd6" "000080d2" "feffff17"
)
# x86-64 System V: p=rdi, n=rsi
X64_SYSV = bytes.fromhex("31c0" "4885f6" "7e0c" "480307" "4883c708" "48ffce" "75f4" "c3")
# x86-64 Windows: p=rcx, n=rdx
X64_WIN = bytes.fromhex("31c0" "4885d2" "7e0c" "480301" "4883c108" "48ffca" "75f4" "c3")

if ARCH in ("arm64", "aarch64"):
    CODE = ARM64
elif ARCH in ("x86_64", "amd64"):
    CODE = X64_WIN if IS_WIN else X64_SYSV
else:
    sys.exit(f"unsupported arch {ARCH}")

SUM = ctypes.CFUNCTYPE(ctypes.c_long if not IS_WIN else ctypes.c_longlong,
                       ctypes.c_void_p, ctypes.c_longlong)
DATA = (ctypes.c_longlong * 1000)(*range(1000))
EXPECT = sum(range(1000))
SIZE = mmap.PAGESIZE * 4


def call(addr):
    got = SUM(addr)(ctypes.addressof(DATA), 1000)
    return "ok" if got == EXPECT else f"WRONG ({got})"


def flush_icache(addr, n):
    """arm64 needs an explicit instruction-cache flush after writing code."""
    if ARCH not in ("arm64", "aarch64"):
        return "n/a"
    if IS_MAC:
        ctypes.CDLL(None).sys_icache_invalidate(ctypes.c_void_p(addr), ctypes.c_size_t(n))
        return "sys_icache_invalidate"
    if IS_WIN:
        k = ctypes.windll.kernel32
        k.FlushInstructionCache(k.GetCurrentProcess(), ctypes.c_void_p(addr), ctypes.c_size_t(n))
        return "FlushInstructionCache"
    try:
        ctypes.CDLL("libgcc_s.so.1").__clear_cache(ctypes.c_void_p(addr), ctypes.c_void_p(addr + n))
        return "libgcc_s __clear_cache"
    except OSError:
        return "NONE (no libgcc_s)"


def strategy_mmap_rwx():
    """stdlib mmap with PROT_EXEC: simplest, violates W^X."""
    m = mmap.mmap(-1, SIZE, prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    m.write(CODE)
    addr = ctypes.addressof(ctypes.c_char.from_buffer(m))
    note = flush_icache(addr, len(CODE))
    return call(addr) + f" [flush: {note}]"


def strategy_posix_wx():
    """mmap RW via libc, write, mprotect to RX. W^X compliant."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    PROT_R, PROT_W, PROT_X = 1, 2, 4
    MAP_PRIVATE, MAP_ANON = 0x02, (0x1000 if IS_MAC else 0x20)
    addr = libc.mmap(None, SIZE, PROT_R | PROT_W, MAP_PRIVATE | MAP_ANON, -1, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        return f"mmap failed errno={ctypes.get_errno()}"
    ctypes.memmove(addr, CODE, len(CODE))
    if libc.mprotect(addr, SIZE, PROT_R | PROT_X) != 0:
        return f"mprotect RX failed errno={ctypes.get_errno()}"
    note = flush_icache(addr, len(CODE))
    return call(addr) + f" [flush: {note}]"


def strategy_mac_jit():
    """macOS: MAP_JIT + pthread_jit_write_protect_np (required under hardened runtime on arm64)."""
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    MAP_PRIVATE, MAP_ANON, MAP_JIT = 0x02, 0x1000, 0x800
    addr = libc.mmap(None, SIZE, 7, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        return f"mmap MAP_JIT failed errno={ctypes.get_errno()}"
    if ARCH == "arm64":
        libc.pthread_jit_write_protect_np(0)
    ctypes.memmove(addr, CODE, len(CODE))
    if ARCH == "arm64":
        libc.pthread_jit_write_protect_np(1)
    note = flush_icache(addr, len(CODE))
    return call(addr) + f" [flush: {note}]"


def strategy_win_virtualalloc():
    """Windows: VirtualAlloc RW, write, VirtualProtect RX, FlushInstructionCache."""
    k = ctypes.windll.kernel32
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
    k.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong)]
    MEM_COMMIT_RESERVE, PAGE_RW, PAGE_RX = 0x3000, 0x04, 0x20
    addr = k.VirtualAlloc(None, SIZE, MEM_COMMIT_RESERVE, PAGE_RW)
    if not addr:
        return f"VirtualAlloc failed err={k.GetLastError()}"
    ctypes.memmove(addr, CODE, len(CODE))
    old = ctypes.c_ulong()
    if not k.VirtualProtect(addr, SIZE, PAGE_RX, ctypes.byref(old)):
        return f"VirtualProtect RX failed err={k.GetLastError()}"
    note = flush_icache(addr, len(CODE))
    return call(addr) + f" [flush: {note}]"


def main():
    print(f"platform={sys.platform} arch={ARCH} python={platform.python_version()} "
          f"impl={platform.python_implementation()} exe={sys.executable}")
    if IS_WIN:
        tests = [strategy_win_virtualalloc]
    elif IS_MAC:
        tests = [strategy_mac_jit, strategy_posix_wx, strategy_mmap_rwx]
    else:
        tests = [strategy_posix_wx, strategy_mmap_rwx]
    for t in tests:
        try:
            res = t()
        except Exception as e:  # report and keep going; a crash kills the process anyway
            res = f"EXC {type(e).__name__}: {e}"
        print(f"{t.__name__:<28} {res}", flush=True)


if __name__ == "__main__":
    main()
