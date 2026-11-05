"""Hardware faults in a kernel become KernelError: a process-wide handler, native code.

A fault can arrive on a thread running without the GIL, so the handler can't be Python. It is C
(`_SOURCE`), compiled by cimo once per process and cached like any module. It looks the faulting
pc up in a table of the loaded modules' code. If the pc is in a module and the calling thread is
inside one of its entry trampolines, it records the fault in the thread's context and returns to
`__cimo_fault`, which unwinds as `abort()` does (cimo.execution). Anything else goes to the
handler that was there before, so Python's faulthandler and other extensions keep working.

- POSIX: sigaction for SIGSEGV, SIGBUS, SIGFPE and SIGILL, on an alternate stack per thread (so a
  stack overflow can be handled), with SA_NODEFER so a previous handler's raise() comes straight
  back here and the guard in the handler stops it.
- Windows: a vectored exception handler. It runs on the faulting stack, in a guarantee set per
  thread (SetThreadStackGuarantee), and Python re-arms the guard page once it has unwound.
- A fault outside the module's code isn't handled, even under a kernel: a host library or
  Python itself, called from a kernel, may hold locks or half-updated state that unwinding past
  would leave behind. The exception is a call through a bad pointer: the pc is the bad address,
  and the return address says which module called it.
- Handlers are checked at every module load, so one installed since (faulthandler.enable())
  ends up behind ours rather than replacing it. CIMO_FAULTS=0 installs nothing.

The table is read in signal context, without a lock. Python only ever stores whole words, and
writes `hi` last on the way in and first on the way out. Tables replaced by a bigger one stay
mapped: a handler may be walking one.
"""

import ctypes
import os
import sys
import threading
import warnings

from . import loader as _jit

ENABLED = os.environ.get('CIMO_FAULTS', '1') != '0'

F_MEMORY, F_DIVISION, F_ILLEGAL = 1, 2, 3

_WIN = sys.platform == 'win32'
_APPLE = sys.platform == 'darwin'
_SIGNALS = (11, 10 if _APPLE else 7, 8, 4)  # SIGSEGV, SIGBUS, SIGFPE, SIGILL
_ALTSTACK = 128 << 10  # the kernel's signal frame alone is tens of KB with SME
_GUARANTEE = 64 << 10  # Windows, what the kernel's exception dispatch and the handler may use
_SLOT = 32

# The register file inside what the OS passes to a handler: Linux's ucontext, macOS's mcontext
# and Windows' CONTEXT all lay out arm64 alike from x0 - 8; x86-64 differs between Linux's gregs
# and Windows' CONTEXT.
_SOURCE = r'''
typedef unsigned long u64;

/* the head of a thread's context: the layout in cimo.execution */
struct ctx {
    u64 err; void *msg, *file; u64 line; void *func; u64 pc; long depth; u64 limit, tid;
    u64 fault, addr;
};
struct mod { u64 lo, hi; struct ctx *(*ctx)(void); u64 resume; };
struct tab { u64 n, pad; struct mod mods[1]; };

struct tab * volatile __cimo_fault_tab;

extern u64 __cimo_tid(void);

/* where the innermost trampoline frame keeps the sp it was entered with */
#define ENTRY_SP(c) (*(u64 *)((char *)(c) + FRAMES + ((c)->depth - 1) * FRAMESZ + FRAME_SP))

#if defined(CIMO_ARM64)
#define R_CTX 80    /* x9 */
#define R_LR 248
#define R_SP 256
#define R_PC 264
#elif defined(CIMO_WIN_X64)
#define R_CTX 0xd0  /* r11 */
#define R_SP 0x98
#define R_PC 0xf8
#else
#define R_CTX 24    /* r11 */
#define R_SP 120
#define R_PC 128
#endif
#define REG(r, off) (*(u64 *)((r) + (off)))

static struct mod *find(u64 pc)
{
    struct tab *t = __cimo_fault_tab;
    u64 i;
    if (!t)
        return 0;
    for (i = 0; i < t->n; i++)
        if (t->mods[i].lo <= pc && pc < t->mods[i].hi)
            return &t->mods[i];
    return 0;
}

/* 1 if the fault was a kernel's: the registers r now resume in its module's unwind */
static int handle(int fault, u64 addr, char *r)
{
    u64 pc = REG(r, R_PC), sp = REG(r, R_SP);
    struct mod *m = find(pc);
    struct ctx *c;
    if (!m && fault == 1 && addr == pc) {  /* called a bad pointer: blame the caller */
#ifdef R_LR
        pc = REG(r, R_LR) - 1;
#else
        pc = *(u64 *)sp - 1;
#endif
        m = find(pc);
    }
    if (!m)
        return 0;
    c = m->ctx();
    if (!c || c->depth <= 0)
        return 0;
    /* the end of the stack: a fault just below sp, or sp itself out of the stack, below its
       limit or wrapped past the bottom of the address space (macOS and Windows stacks are low) */
    c->err = fault == 4 || (fault == 1 && ((addr < c->limit && addr + 65536 >= sp)
                                           || sp < c->limit || sp > ENTRY_SP(c))) ? 4 : 7;
    c->fault = fault;
    c->addr = addr;
    c->pc = pc;
    REG(r, R_CTX) = (u64)c;
    REG(r, R_PC) = m->resume;
    return 1;
}

#ifdef CIMO_WINDOWS
int __cimo_fault_veh(void **pointers)
{
    char *rec = pointers[0];
    u64 addr = 0;
    int fault;
    switch (*(unsigned *)rec) {
    case 0xc0000005:  /* access violation; information: read or write, address */
    case 0xc0000006:  /* in-page error */
        addr = *(u64 *)(rec + 40);
        fault = 1;
        break;
    case 0x80000002:  /* misaligned */
        fault = 1;
        break;
    case 0xc0000094:  /* divide by zero */
    case 0xc0000095:  /* overflow */
        fault = 2;
        break;
    case 0xc000001d:  /* illegal */
    case 0xc0000096:  /* privileged */
        fault = 3;
        break;
    case 0xc00000fd:  /* stack overflow */
        fault = 4;
        break;
    default:
        return 0;
    }
    return handle(fault, addr, pointers[1]) ? -1 : 0;  /* continue execution, or search on */
}
#else
void (*__cimo_fault_old[32])(int, void *, void *);
u64 __cimo_fault_owner[32];
int (*__cimo_fault_sigaction)(int, const void *, void *);
int (*__cimo_fault_raise)(int);
static char dfl[256];

void __cimo_fault_sig(int sig, char *si, char *uc)
{
    void (*old)(int, void *, void *) = __cimo_fault_old[sig];
    int fault = sig == 8 ? 2 : sig == 4 ? 3 : 1;
    u64 me;
    if (handle(fault, *(u64 *)(si + SI_ADDR), REGS(uc)))
        return;
    me = __cimo_tid();
    if ((u64)old > 1 && __cimo_fault_owner[sig] != me) {
        __cimo_fault_owner[sig] = me;
        old(sig, si, uc);
        __cimo_fault_owner[sig] = 0;
        return;
    }
    /* No handler before us, or that handler raised this on its way out (faulthandler restores
       what it replaced, which can be us). A real fault happens again on return and ends the
       process at the faulting instruction; anything sent has to be raised. */
    __cimo_fault_sigaction(sig, dfl, 0);
    if (SENT(*(int *)(si + 8)))
        __cimo_fault_raise(sig);
}
#endif
'''


def _defines():
    from . import execution as x
    d = {'CIMO_ARM64': None} if _jit._ARM64 else {}
    win64 = _WIN and not _jit._ARM64
    d['FRAMES'], d['FRAMESZ'] = (str(x.WFRAMES), str(x.WFRAMESZ)) if win64 else (str(x.FRAMES), str(x.FRAMESZ))
    d['FRAME_SP'] = '96' if _jit._ARM64 else '72'  # cimo.execution's glue
    if _WIN:
        d['CIMO_WINDOWS'] = None
        if not _jit._ARM64:
            d['CIMO_WIN_X64'] = None
    elif _APPLE:
        d['SI_ADDR'] = '24'
        d['SENT(code)'] = '((code) >= 0x10001)'  # SI_USER, SI_QUEUE, ...
        d['REGS(uc)'] = '(*(char **)((uc) + 48) + 8)'
    else:
        d['SI_ADDR'] = '16'
        d['SENT(code)'] = '((code) <= 0)'
        d['REGS(uc)'] = '((uc) + %d)' % (176 if _jit._ARM64 else 40)
    return d


class _Sigaction(ctypes.Structure):
    # macOS: a 4-byte mask, then the flags. Linux (glibc, musl): a 128-byte mask, the flags and
    # the restorer. Room for either; only the fields used are named.
    _fields_ = [('handler', ctypes.c_void_p), ('rest', ctypes.c_char * 248)]

    FLAGS = 4 if _APPLE else 128

    @property
    def flags(self):
        return ctypes.c_int.from_buffer(self, 8 + self.FLAGS).value

    @flags.setter
    def flags(self, v):
        ctypes.c_int.from_buffer(self, 8 + self.FLAGS).value = v


class _Stack(ctypes.Structure):
    _fields_ = ([('sp', ctypes.c_void_p), ('size', ctypes.c_size_t), ('flags', ctypes.c_int)]
                if _APPLE else
                [('sp', ctypes.c_void_p), ('flags', ctypes.c_int), ('size', ctypes.c_size_t)])


_SA_SIGINFO = 0x40 if _APPLE else 4
_SA_ONSTACK = 1 if _APPLE else 0x08000000
_SA_NODEFER = 0x10 if _APPLE else 0x40000000
_SS_DISABLE = 4 if _APPLE else 2

_build_lock = threading.RLock()
_lock = threading.RLock()  # the table. Never held across a compile: a module's collection takes it
_module = None
_building = False
_veh = None
_tab = 0
_tab_slots = 0
_free = []
_next = 0
_tls = threading.local()


def install():
    """Build the handler if this is the first module, and see that it is still installed."""
    global _module, _building
    if not ENABLED:
        return
    with _build_lock:  # the build's own module comes back here, on this thread
        if _building:
            return
        try:
            if _module is None:
                _building = True
                try:
                    from . import driver
                    _module = driver.compile(_SOURCE, defines=_defines(), fma=False, wrapv=True)
                finally:
                    _building = False
                if not _WIN:
                    _fill_libc(_module)
            if _WIN:
                _install_windows()
            else:
                _install_posix()
        except (OSError, AttributeError, ValueError, _jit.LoadError) as e:
            warnings.warn('cimo: faults in kernels will end the process: %s' % e, stacklevel=2)


def _fill_libc(m):
    img = m._image
    libc = _jit._libc
    for name, f in (('__cimo_fault_sigaction', libc.sigaction),
                    ('__cimo_fault_raise', getattr(libc, 'raise'))):
        ctypes.c_void_p.from_address(img.address(name)).value = \
            ctypes.cast(f, ctypes.c_void_p).value


def _sigaction():
    f = _jit._libc.sigaction
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    return f


def _install_posix():
    img = _module._image
    ours = img.address('__cimo_fault_sig')
    old = img.address('__cimo_fault_old')
    call = _sigaction()
    for sig in _SIGNALS:
        cur = _Sigaction()
        if call(sig, None, ctypes.byref(cur)) != 0:
            raise OSError(ctypes.get_errno(), 'sigaction failed')
        if cur.handler == ours:
            continue
        ctypes.c_void_p.from_address(old + 8 * sig).value = cur.handler
        new = _Sigaction()
        new.handler = ours
        new.flags = _SA_SIGINFO | _SA_ONSTACK | _SA_NODEFER
        if call(sig, ctypes.byref(new), None) != 0:
            raise OSError(ctypes.get_errno(), 'sigaction failed')


def _install_windows():
    # again at every load: faulthandler.enable() puts its own first, and it reports (and carries
    # on) before anyone else has looked
    global _veh
    k32 = ctypes.windll.kernel32
    if _veh is not None:
        k32.RemoveVectoredExceptionHandler.argtypes = [ctypes.c_void_p]
        k32.RemoveVectoredExceptionHandler(_veh)
    k32.AddVectoredExceptionHandler.restype = ctypes.c_void_p
    k32.AddVectoredExceptionHandler.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
    _veh = k32.AddVectoredExceptionHandler(1, _module._image.address('__cimo_fault_veh'))
    if not _veh:
        raise OSError('AddVectoredExceptionHandler failed')


class _AltStack:
    """The calling thread's alternate signal stack, ours; gone with the thread's locals."""

    def __init__(self):
        self.thread = threading.get_ident()
        self.sp = _jit._map(_ALTSTACK, _jit.PROT_READ | _jit.PROT_WRITE)
        s = _Stack()
        s.sp, s.size, s.flags = self.sp, _ALTSTACK, 0
        if _sigaltstack()(ctypes.byref(s), None) != 0:
            _jit._unmap(self.sp, _ALTSTACK)
            self.sp = None
            raise OSError(ctypes.get_errno(), 'sigaltstack failed')

    def __del__(self):
        # another thread can't take this stack away from its owner, so it stays mapped
        try:
            if self.sp is None or threading.get_ident() != self.thread:
                return
            cur, off = _Stack(), _Stack()
            off.flags = _SS_DISABLE
            call = _sigaltstack()
            if call(None, ctypes.byref(cur)) == 0 and cur.sp == self.sp:
                call(ctypes.byref(off), None)
            _jit._unmap(self.sp, _ALTSTACK)
        except Exception:
            pass


def _sigaltstack():
    f = _jit._libc.sigaltstack
    f.restype = ctypes.c_int
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return f


def thread():
    """Called once per module on each thread that runs kernels: a stack to handle faults on,
    unless the thread has one big enough already (faulthandler's is often too small). On Windows
    the handler runs on the faulting stack, so room held back from the end of it."""
    if not ENABLED or getattr(_tls, 'done', False):
        return
    if _WIN:
        size = ctypes.c_ulong(_GUARANTEE)
        ctypes.windll.kernel32.SetThreadStackGuarantee(ctypes.byref(size))
        _tls.done = True
        return
    cur = _Stack()
    if _sigaltstack()(None, ctypes.byref(cur)) == 0 and \
            (cur.flags & _SS_DISABLE or cur.size < _ALTSTACK):
        try:
            _tls.stack = _AltStack()
        except OSError:
            pass  # only a stack overflow suffers
    _tls.done = True


def register(image):
    """Faults in this image's code become errors. Returns the slot for unregister."""
    m = _module
    if m is None:
        return None  # the handler's own module, or faults are off
    with _lock:
        slot = _free.pop() if _free else _new_slot()
        a = _tab + 16 + slot * _SLOT
        u64 = ctypes.c_uint64.from_address
        u64(a + 8).value = 0
        code = image._maps[0][0]
        u64(a + 16).value = image.address('__cimo_ctx')
        u64(a + 24).value = image.address('__cimo_fault')
        u64(a).value = code
        m._state._barrier()
        u64(a + 8).value = code + image.code_size
    return slot


def _new_slot():
    global _tab, _tab_slots, _next
    if _next == _tab_slots:
        n = _tab_slots * 2 or (65536 - 16) // _SLOT
        t = _jit._map(16 + n * _SLOT, _jit.PROT_READ | _jit.PROT_WRITE)
        if _tab:
            ctypes.memmove(t + 16, _tab + 16, _tab_slots * _SLOT)
        ctypes.c_uint64.from_address(t).value = n
        _module._state._barrier()
        ctypes.c_uint64.from_address(_module._image.address('__cimo_fault_tab')).value = t
        _tab, _tab_slots = t, n
    _next += 1
    return _next - 1


def unregister(slot):
    """The image is going away: no kernel of it is running."""
    with _lock:
        ctypes.c_uint64.from_address(_tab + 16 + slot * _SLOT + 8).value = 0
        _free.append(slot)


def stack_recovered():
    """After unwinding from a stack overflow: Windows' guard page has to be put back, or the
    next overflow ends the process."""
    if _WIN:
        for dll in ('ucrtbase', 'msvcrt'):
            try:
                getattr(ctypes.cdll, dll)._resetstkoflw()
                return
            except (OSError, AttributeError):
                pass
