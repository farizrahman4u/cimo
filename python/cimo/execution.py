"""Per-call execution context: error unwinding, the stack guard and the module heap's Python side.

Design (requirements.md, "Execution model"):

- Every module gets a few hand-written assembly routines (`glue()`), assembled and linked with
  it. Python calls a kernel through its *entry trampoline* `__cimo_entry_<name>`, never the
  function itself. The trampoline finds the calling thread's context, saves the callee-saved
  registers, sp and the return address into the next of the context's frames, calls the
  kernel with the arguments untouched (sp is unchanged, so stack arguments stay in place) and
  pops the frame on return.
- `abort()`, a failed `assert`, `__builtin_trap()`, unreachable code and the stack guard store
  an error code (and the assert expression, file, line, function, faulting pc) in the context
  and *unwind*: restore the registers saved by the innermost trampoline and return 0 to its
  caller. Python checks the context's error word after every call and raises `KernelError`.
  The module and the interpreter stay usable: nothing in between needs cleaning up (kernels
  run no destructors, and the heap's bookkeeping is never mid-update when they abort).
- Hardware faults in a module's code take the same unwind (cimo.fault), by way of a process-wide
  handler. Those can be mid-update: memory the kernel was writing may be half-written.
- How kernel code finds its context: a per-module hash table keyed by the OS thread's
  thread-pointer register (`tpidr_el0` on Linux, `tpidrro_el0` on macOS, both readable
  from user mode and unique per live thread; on Windows x18, which always holds the
  thread's TEB), filled by Python the first time a thread calls into the module, or when its `malloc` runs on a
  thread with none (`__cimo_hook_ctx`: code called by address from another module). An error there has no frame
  in this module and goes on to cimo.fault, which moves it to the innermost entry frame on the stack. Rejected
  alternatives: x18 as a free register (reserved on macOS and Windows); TLS (the loader would need TLS relocations and a TLS block per module); a
  hidden argument (changes every function's ABI and the frontend). The lookup costs about 20
  instructions and runs once per kernel call, `malloc` and `free`; never in loops otherwise.
- One context per (module, thread), allocated by Python (two pages). Its layout is shared by
  the assembly below and runtime/heap.c (`CTX_HEAP`):

      0 err   8 msg   16 file   24 line   32 func   40 pc   48 depth   56 stack limit   64 tid
      72 fault kind   80 fault address (these first 88 bytes also in cimo.fault's C)
      88 thread number in a parallel loop   96 its team's size   104 its team (cimo.fault reads it on Ctrl-C)
      112 omp_set_num_threads (runtime/omp.c)
      120 nonzero while a live Python thread has registered the context (State.ctx): a kernel's CPython entry
          (driver._pyentry) takes it then, and a thread reusing a finished one's key goes through Python first
      128 frames: MAXDEPTH x 176 bytes (x19-x28, x29, lr, sp, d8-d15, fpcr; x86-64 Linux: rbx,
          rbp, r12-r15, the return address, rsp, the shadow stack pointer, mxcsr, scratch)
      3072 the thread's heap (struct heap in runtime/heap.c, at most 1024 bytes)
      4096 the thread's stdout buffer (runtime/stdio/stdout.c: length, then 4088 bytes)

- Re-entry (a kernel calls a Python callback that calls a kernel of the same module on the
  same thread) pushes another frame; an error unwinds only the innermost call.
- Stack guard: functions that can recurse (in a call-graph cycle, counting indirect calls as
  edges to every address-taken function) call `__cimo_stkchk` in their prologue, which compares
  sp against the thread's stack bottom + STACK_MARGIN (from pthread_getattr_np /
  pthread_get_stackaddr_np; GetCurrentThreadStackLimits on Windows). `alloca`/VLAs and frames
  larger than the margin run into the guard page instead, which cimo.fault reports the same way.
- Thread safety: each context and heap is used only by its own thread; the table only grows
  (Python inserts under a lock; a full rebuild is published with a `dmb ish` before the
  pointer store, and readers depend on the loaded pointer). Cross-thread `free` uses one
  compare-and-swap (`__cimo_cas`, an ldaxr/stlxr loop: ARMv8.0, so no CPU feature check).
"""

from __future__ import annotations

import bisect
import ctypes
import sys
import threading
import weakref

from . import fault as _fault
from . import loader as _jit

__all__ = ['KernelError', 'glue', 'stack_checked', 'State']

CTX_SIZE = 8192
# Windows x64 frames also hold rdi, rsi and xmm6-xmm15 (256 bytes): they go after the stdout
# buffer, in a third page (_glue_amd64)
_WIN_X64 = sys.platform == 'win32' and _jit._X86
WFRAMES, WFRAMESZ = 8192, 256
if _WIN_X64:
    CTX_SIZE = 12288
ERR, MSG, FILE, LINE, FUNC, PC, DEPTH, LIMIT, TID, FAULT, ADDR = range(0, 88, 8)
FRAMES, FRAMESZ, MAXDEPTH = 128, 176, 16
CTX_HEAP = 3072
CTX_OUT = 4096
CTX_PY = 120
STACK_MARGIN = 128 << 10

E_ABORT, E_ASSERT, E_TRAP, E_STACK, E_DEPTH, E_EXIT, E_FAULT, E_INTERRUPT = 1, 2, 3, 4, 5, 6, 7, 8
_HASH_K = 0x9E3779B97F4A7C15
_M64 = (1 << 64) - 1


class KernelError(Exception):
    """A kernel called abort() or exit(), failed an assert, trapped, overflowed its stack or
    faulted. Attributes: kind ('abort', 'exit', 'assert', 'trap', 'stack overflow', 'nesting',
    'memory fault', 'division fault', 'illegal instruction', 'interrupt': Ctrl-C under a SIGINT handler
    that doesn't raise), kernel (the entry point Python
    called), function (where it happened, if known), for assert: expr, file, line, for exit:
    status, and for a memory fault or a stack overflow caught at the guard page: address."""

    kind: str | None
    kernel: str | None
    function: str | None
    expr: str | None
    file: str | None
    line: int | None
    status: int | None
    address: int | None

    def __init__(self, message: str, kind: str | None = None, kernel: str | None = None,
                 function: str | None = None, expr: str | None = None,
                 file: str | None = None, line: int | None = None,
                 status: int | None = None, address: int | None = None) -> None:
        Exception.__init__(self, message)
        self.kind, self.kernel, self.function = kind, kernel, function
        self.expr, self.file, self.line, self.status = expr, file, line, status
        self.address = address


# ---------------------------------------------------------------------------------------------
# assembly

def _find(apple, s, windows=False):
    """x9 = this thread's context or 0. Clobbers x10-x13 and flags only."""
    return '\n'.join([
        '\tmov\tx10, x18' if windows else  # the TEB
        '\t.word\t0xd53bd06a' if apple else '\t.word\t0xd53bd04a',  # mrs x10, tpidr(ro)_el0
        '\tand\tx10, x10, #0xfffffffffffffff8' if apple else '',  # low bits: old XNU cpu number
        '\tadrp\tx11, %s' % s['tabpage'],
        '\tadd\tx11, x11, %s' % s['taboff'],
        '\tldr\tx11, [x11]',
        '\tmov\tx9, #0',
        '\tcmp\tx11, #0',
        '\tb.eq\t%(L)sdone%(n)s',
        '\tmov\tx9, #%d' % (_HASH_K & 0xffff),
        '\tmovk\tx9, #%d, lsl #16' % (_HASH_K >> 16 & 0xffff),
        '\tmovk\tx9, #%d, lsl #32' % (_HASH_K >> 32 & 0xffff),
        '\tmovk\tx9, #%d, lsl #48' % (_HASH_K >> 48 & 0xffff),
        '\tmul\tx9, x10, x9',
        '\tldr\tx12, [x11, #8]',     # shift = 64 - log2(nslots)
        '\tlsr\tx9, x9, x12',
        '\tlsl\tx9, x9, #4',
        '\tadd\tx13, x11, #16',
        '\tadd\tx13, x13, x9',       # first slot to probe
        '\tldr\tx12, [x11]',         # nslots
        '\tlsl\tx12, x12, #4',
        '\tadd\tx12, x12, x11',
        '\tadd\tx12, x12, #16',      # end of the slots
        '%(L)sprobe%(n)s:',
        '\tldr\tx9, [x13]',
        '\tcmp\tx9, x10',
        '\tb.eq\t%(L)sfound%(n)s',
        '\tcmp\tx9, #0',
        '\tb.eq\t%(L)sdone%(n)s',
        '\tadd\tx13, x13, #16',
        '\tcmp\tx13, x12',
        '\tb.ne\t%(L)sprobe%(n)s',
        '\tadd\tx13, x11, #16',
        '\tb\t%(L)sprobe%(n)s',
        '%(L)sfound%(n)s:',
        '\tldr\tx9, [x13, #8]',
        '%(L)sdone%(n)s:',
    ]) % s + '\n'


_SAVE = [('x19', 'x20', 0), ('x21', 'x22', 16), ('x23', 'x24', 32), ('x25', 'x26', 48),
         ('x27', 'x28', 64)]
# the caller's floating-point control register (rounding mode): each call starts with the exception flags clear (FPSR),
# as a C program does, and a return or an unwind puts the rounding mode back (writing FPCR only when the kernel changed
# it: a write costs several ns), so a kernel's fesetround never reaches Python's own arithmetic. The flags are left as
# the kernel set them (Python doesn't read them; numpy clears them before it does)
FPCR = 168
_DSAVE = [('d8', 'd9', 104), ('d10', 'd11', 120), ('d12', 'd13', 136), ('d14', 'd15', 152)]


def glue(names, apple=False, windows=False, target=None):
    """Assembly text (the syntax `cimo.backend` emits for the target) with the context
    routines and an entry trampoline `__cimo_entry_<name>` for each name."""
    if target and target.startswith('amd64'):
        return _glue_amd64(names, windows)
    p = '_' if apple else ''
    L = 'L' if apple else '.L'
    tab = p + '__cimo_ctxtab'
    s = {'L': L, 'tabpage': tab + ('@PAGE' if apple else ''),
         'taboff': tab + '@PAGEOFF' if apple else '#:lo12:' + tab}
    n = [0]

    def find():
        n[0] += 1
        return _find(apple, dict(s, n='_cimo%d' % n[0]), windows)

    def fn(name):
        return '\t.globl\t%s%s\n\t.p2align\t2\n%s%s:\n' % (p, name, p, name)

    orphan = p + '__cimo_orphan'

    def orphan_x9(label):  # x9 = the orphan context when it is 0
        return ('\tcbnz\tx9, %s%s\n\tadrp\tx9, %s\n\tadd\tx9, x9, %s\n%s%s:\n'
                % (L, label, orphan + ('@PAGE' if apple else ''),
                   orphan + '@PAGEOFF' if apple else '#:lo12:' + orphan, L, label))

    def frame():  # x11 = ctx + FRAMES + depth * FRAMESZ, x10 = depth (x9 = ctx)
        return ('\tldr\tx10, [x9, #%d]\n\tmov\tx11, #%d\n\tmadd\tx11, x10, x11, x9\n'
                '\tadd\tx11, x11, #%d\n' % (DEPTH, FRAMESZ, FRAMES))

    o = ['\t.text\n']
    # __cimo_enter: x14 = kernel, x15 = return address (the trampolines' common tail)
    o.append(fn('__cimo_enter') + find())
    o.append('\tcmp\tx9, #0\n\tb.ne\t%sctx\n\tmov\tx30, x15\n\tbr\tx14\n%sctx:\n' % (L, L))
    o.append(frame())
    o.append('\tcmp\tx10, #%d\n\tb.hs\t%sdeep\n\tadd\tx10, x10, #1\n\tstr\tx10, [x9, #%d]\n'
             % (MAXDEPTH, L, DEPTH))
    o += ['\tstp\t%s, %s, [x11, #%d]\n' % r for r in _SAVE]
    o.append('\tstp\tx29, x15, [x11, #80]\n\tmov\tx12, sp\n\tstr\tx12, [x11, #96]\n')
    o += ['\tstp\t%s, %s, [x11, #%d]\n' % r for r in _DSAVE]
    o.append('\tmrs\tx12, fpcr\n\tstr\tw12, [x11, #%d]\n\tmsr\tfpsr, xzr\n' % FPCR)
    o.append('\tmov\tx19, x9\n\tmov\tx20, x11\n\tblr\tx14\n'
             '\tldr\tw12, [x20, #%d]\n\tmrs\tx13, fpcr\n\tcmp\tw12, w13\n\tb.eq\t%sfpok\n\tmsr\tfpcr, x12\n%sfpok:\n'
             % (FPCR, L, L) +
             '\tldr\tx30, [x20, #88]\n\tldr\tx10, [x19, #%d]\n\tsub\tx10, x10, #1\n'
             '\tstr\tx10, [x19, #%d]\n\tldp\tx19, x20, [x20]\n\tret\n' % (DEPTH, DEPTH))
    o.append('%sdeep:\n\tmov\tx10, #%d\n\tstr\tx10, [x9]\n\tmov\tx30, x15\n\tb\t%sret0\n'
             % (L, E_DEPTH, L))
    # unwind: x9 = ctx with the error stored; back to the innermost trampoline's caller
    o.append('%sunwind:\n' % L + frame())
    o.append('\tcmp\tx10, #0\n\tb.eq\t%snoframe\n\tsub\tx10, x10, #1\n\tstr\tx10, [x9, #%d]\n'
             '\tsub\tx11, x11, #%d\n' % (L, DEPTH, FRAMESZ))
    o += ['\tldp\t%s, %s, [x11, #%d]\n' % r for r in _SAVE]
    o.append('\tldp\tx29, x30, [x11, #80]\n\tldr\tx12, [x11, #96]\n\tmov\tsp, x12\n')
    o += ['\tldp\t%s, %s, [x11, #%d]\n' % r for r in _DSAVE]
    o.append('\tldr\tw12, [x11, #%d]\n\tmsr\tfpcr, x12\n' % FPCR)
    o.append('%sret0:\n\tmov\tx0, #0\n\tmov\tx1, #0\n\tfmov\td0, x0\n\tfmov\td1, x0\n\tret\n'
             '%snoframe:\n' % (L, L) + fn('__cimo_noframe') + '\t.word\t0x000000c1\n')  # udf #0xc1
    # __cimo_fault: where cimo.fault's handler resumes a faulting kernel, with x9 = ctx
    o.append(fn('__cimo_fault') + '\tb\t%sunwind\n' % L)
    # __cimo_fail(code, msg, file, line, func): called by abort()/__assert_fail (C, with a
    # frame, so [x29, #8] is the address in their caller)
    o.append(fn('__cimo_fail') + find() + orphan_x9('failctx'))
    o.append('\tstr\tx0, [x9]\n\tstp\tx1, x2, [x9, #%d]\n'
             '\tstp\tx3, x4, [x9, #%d]\n\tldr\tx10, [x29, #8]\n\tstr\tx10, [x9, #%d]\n'
             '\tb\t%sunwind\n' % (MSG, LINE, PC, L))
    # __builtin_trap / __cimo_trap: pc = the call site
    for name in ('__builtin_trap', '__cimo_trap'):
        o.append(fn(name))
    o.append(find() + orphan_x9('trapctx') + '\tmov\tx10, #%d\n\tstr\tx10, [x9]\n'
             '\tstr\tx30, [x9, #%d]\n\tb\t%sunwind\n' % (E_TRAP, PC, L))
    # __cimo_stkchk: called first thing in the prologue of functions that can recurse;
    # preserves the argument registers (x0-x8, v0-v7)
    o.append(fn('__cimo_stkchk') + find())
    o.append('\tcmp\tx9, #0\n\tb.eq\t%sstkok\n\tldr\tx10, [x9, #%d]\n\tmov\tx11, sp\n'
             '\tcmp\tx11, x10\n\tb.lo\t%sstkovf\n%sstkok:\n\tret\n'
             '%sstkovf:\n\tmov\tx10, #%d\n\tstr\tx10, [x9]\n\tstr\tx30, [x9, #%d]\n'
             '\tb\t%sunwind\n' % (L, LIMIT, L, L, L, E_STACK, PC, L))
    # void *__cimo_ctx(void): this thread's context (0 outside a registered thread)
    o.append(fn('__cimo_ctx') + find() + '\tmov\tx0, x9\n\tret\n')
    # unsigned long __cimo_tid(void): the key the table uses
    o.append(fn('__cimo_tid') + ('\tmov\tx0, x18\n' if windows else
                                 '\t.word\t0xd53bd060\n\tand\tx0, x0, #0xfffffffffffffff8\n'
                                 if apple else '\t.word\t0xd53bd040\n') + '\tret\n')
    # void *__cimo_cas(void **p, void *expect, void *desired): returns the old *p
    o.append(fn('__cimo_cas') +
             '%scas:\n\t.word\t0xc85ffc03\n'  # ldaxr x3, [x0]
             '\tcmp\tx3, x1\n\tb.ne\t%scasne\n'
             '\t.word\t0xc804fc02\n'  # stlxr w4, x2, [x0]
             '\tcmp\tw4, #0\n\tb.ne\t%scas\n\tmov\tx0, x3\n\tret\n'
             '%scasne:\n\t.word\t0xd5033f5f\n\tmov\tx0, x3\n\tret\n' % (L, L, L, L))  # clrex
    o.append(fn('__cimo_barrier') + '\t.word\t0xd5033bbf\n\tret\n')  # dmb ish
    o.append(fn('__cimo_pause') + '\t.word\t0xd503203f\n\tret\n')  # yield: a spinning thread's hint
    # void __cimo_call(fn, a, b, c): fn(a, b, c) through an entry frame of its own, as a trampoline calls a kernel:
    # an error in it unwinds to here and the call returns with it in the context (a parallel loop's part, runtime/omp.c)
    o.append(fn('__cimo_call') + '\tmov\tx14, x0\n\tmov\tx0, x1\n\tmov\tx1, x2\n\tmov\tx2, x3\n'
             '\tmov\tx15, x30\n\tb\t%s__cimo_enter\n' % p)
    # void __cimo_rethrow(ctx): unwind as if the error now in ctx had happened here (one a worker thread recorded)
    o.append(fn('__cimo_rethrow') + '\tmov\tx9, x0\n\tb\t%sunwind\n' % L)
    o.append(_setjmp_arm64(fn))
    for name in names:
        o.append(fn('__cimo_entry_' + name))
        o.append('\tmov\tx15, x30\n\tadrp\tx14, %s%s%s\n\tadd\tx14, x14, %s\n\tb\t%s__cimo_enter\n'
                 % (p, name, '@PAGE' if apple else '',
                    p + name + '@PAGEOFF' if apple else '#:lo12:' + name, p))
    o.append('\t.data\n\t.p2align\t3\n\t.globl\t%s\n%s:\n\t.quad\t0\n' % (tab, tab))
    o.append(_DATA % {'p': p})
    return ''.join(o)


# A context for errors on a thread with none in this module (code called by address from another module, or a thread
# a kernel started): no frames, so the error goes on to cimo.fault, which moves it to the innermost entry frame of any
# module on the stack. One per module, not per thread: only an error in such code, at the same moment on two threads,
# can mix their messages. And the callback that makes a thread's context for malloc (runtime/heap.c).
_DATA = ('\t.globl\t%(p)s__cimo_hook_ctx\n%(p)s__cimo_hook_ctx:\n\t.quad\t0\n'
         '\t.globl\t%(p)s__cimo_orphan\n%(p)s__cimo_orphan:\n\t.zero\t128\n')


_RDSSP_R10 = '\t.byte\t0xf3, 0x49, 0x0f, 0x1e, 0xca\n'  # rdsspq %r10
_RDSSP_RCX = '\t.byte\t0xf3, 0x48, 0x0f, 0x1e, 0xc9\n'  # rdsspq %rcx
_INCSSP_RCX = '\t.byte\t0xf3, 0x48, 0x0f, 0xae, 0xe9\n'  # incsspq %rcx


def _glue_amd64(names, windows):
    """glue() for x86-64 (System V on Linux, Windows x64): the same routines and context layout.

    The thread key is the thread's own control block, which the segment base points at and
    whose first word is its own address: %fs:0 on Linux (glibc and musl), %gs:0x30 (the
    TEB's self pointer) on Windows. __cimo_find does the table lookup, clobbering only r10, r11
    and flags (r11 = the context or 0), so the argument registers survive into the kernel.

    Entry: the trampoline pops the caller's return address into the frame, so the kernel,
    called from there, sees its stack arguments (and Windows' shadow space) where its caller put
    them, then pushes the saved address back and returns to it, so calls and returns stay paired.

    Under a CET shadow stack (Windows' hardware-enforced stack protection, Linux's user shadow
    stacks) the same pairing has to survive an unwind: the frame keeps the shadow stack pointer
    from entry, 0 when shadow stacks are off (rdssp is a nop then, and on CPUs without CET), and
    the unwind pops what the kernel's abandoned calls left on it before its ret."""
    key = '\tmovq\t%gs:0x30, %r10\n' if windows else '\tmovq\t%fs:0, %r10\n'
    if windows:
        frames, framesz = WFRAMES, WFRAMESZ
        regs = [('rbx', 0), ('rbp', 8), ('r12', 16), ('r13', 24), ('r14', 32), ('r15', 40),
                ('rdi', 48), ('rsi', 56)]
        xregs = [('xmm%d' % (6 + k), 80 + 16 * k) for k in range(10)]
    else:
        frames, framesz = FRAMES, FRAMESZ
        regs = [('rbx', 0), ('rbp', 8), ('r12', 16), ('r13', 24), ('r14', 32), ('r15', 40)]
        xregs = []
    RET, RSP = 64, 72
    SSP = 240 if windows else 80
    MXCSR = 248 if windows else 88  # the caller's, its rounding control put back on return and unwind (FPCR on arm64)

    def fn(name):
        return '\t.globl\t%s\n\t.p2align\t4\n%s:\n' % (name, name)

    o = ['\t.text\n']
    o.append(fn('__cimo_find') + '\tpushq\t%rax\n\tpushq\t%rcx\n\tpushq\t%rdx\n' + key +
             '\tmovq\t__cimo_ctxtab(%%rip), %%r11\n'
             '\ttestq\t%%r11, %%r11\n\tjz\t.Lcimo_none\n'
             '\tmovabsq\t$%d, %%rax\n\timulq\t%%r10, %%rax\n'
             '\tmovq\t8(%%r11), %%rcx\n\tshrq\t%%cl, %%rax\n\tshlq\t$4, %%rax\n'
             '\tleaq\t16(%%r11,%%rax), %%rax\n'
             '\tmovq\t(%%r11), %%rdx\n\tshlq\t$4, %%rdx\n\tleaq\t16(%%r11,%%rdx), %%rdx\n'
             '.Lcimo_probe:\n\tmovq\t(%%rax), %%rcx\n\tcmpq\t%%r10, %%rcx\n\tje\t.Lcimo_found\n'
             '\ttestq\t%%rcx, %%rcx\n\tjz\t.Lcimo_none\n\taddq\t$16, %%rax\n'
             '\tcmpq\t%%rdx, %%rax\n\tjne\t.Lcimo_probe\n\tleaq\t16(%%r11), %%rax\n'
             '\tjmp\t.Lcimo_probe\n'
             '.Lcimo_found:\n\tmovq\t8(%%rax), %%r11\n\tjmp\t.Lcimo_done\n'
             '.Lcimo_none:\n\txorl\t%%r11d, %%r11d\n'
             '.Lcimo_done:\n\tpopq\t%%rdx\n\tpopq\t%%rcx\n\tpopq\t%%rax\n\tret\n'
             % (_HASH_K - (1 << 64) if _HASH_K >> 63 else _HASH_K))
    # __cimo_enter: r11 = the kernel, [rsp] = the caller's return address
    o.append(fn('__cimo_enter') +
             '\tpushq\t%r11\n\tcallq\t__cimo_find\n\ttestq\t%r11, %r11\n\tjnz\t.Lcimo_ctx\n'
             '\tpopq\t%r11\n\tjmp\t*%r11\n'  # no context (a raw call): straight to the kernel
             '.Lcimo_ctx:\n')
    o.append('\tmovq\t%d(%%r11), %%r10\n\tcmpq\t$%d, %%r10\n\tjae\t.Lcimo_deep\n'
             '\taddq\t$1, %d(%%r11)\n\timulq\t$%d, %%r10, %%r10\n\tleaq\t%d(%%r11,%%r10), %%r10\n'
             % (DEPTH, MAXDEPTH, DEPTH, framesz, frames))
    o += ['\tmovq\t%%%s, %d(%%r10)\n' % r for r in regs]
    o += ['\tmovdqu\t%%%s, %d(%%r10)\n' % r for r in xregs]
    o.append('\tstmxcsr\t%d(%%r10)\n' % MXCSR)
    o.append('\tmovq\t%%r11, %%r13\n\tmovq\t%%r10, %%r12\n'   # r13 = ctx, r12 = frame
             '\tpopq\t%%r11\n\tpopq\t%%r10\n'                # kernel, return address
             '\tmovq\t%%r10, %d(%%r12)\n\tmovq\t%%rsp, %d(%%r12)\n'
             '\txorl\t%%r10d, %%r10d\n%s\tmovq\t%%r10, %d(%%r12)\n'
             '\tmovl\t%d(%%r12), %%r10d\n\tandl\t$-64, %%r10d\n\tmovl\t%%r10d, %d(%%r12)\n\tldmxcsr\t%d(%%r12)\n'
             '\tcallq\t*%%r11\n\tstmxcsr\t%d(%%r12)\n\tmovl\t%d(%%r12), %%r10d\n\txorl\t%d(%%r12), %%r10d\n'
             '\tandl\t$-64, %%r10d\n\tjz\t.Lcimo_mxok\n\tldmxcsr\t%d(%%r12)\n.Lcimo_mxok:\n'
             '\tsubq\t$1, %d(%%r13)\n\tmovq\t%d(%%r12), %%r10\n'
             '\tmovq\t24(%%r12), %%r13\n\tmovq\t16(%%r12), %%r12\n\tpushq\t%%r10\n\tret\n'
             % (RET, RSP, _RDSSP_R10, SSP, MXCSR, MXCSR + 4, MXCSR + 4, MXCSR + 4, MXCSR + 4, MXCSR, MXCSR, DEPTH,
                RET))
    o.append('.Lcimo_deep:\n\tmovq\t$%d, (%%r11)\n\taddq\t$8, %%rsp\n\tjmp\t.Lcimo_ret0\n'
             % E_DEPTH)
    # unwind: r11 = ctx with the error stored; back to the innermost trampoline's caller
    o.append('.Lcimo_unwind:\n\tmovq\t%d(%%r11), %%r10\n\ttestq\t%%r10, %%r10\n'
             '\tjz\t.Lcimo_noframe\n\tsubq\t$1, %%r10\n\tmovq\t%%r10, %d(%%r11)\n'
             '\timulq\t$%d, %%r10, %%r10\n\tleaq\t%d(%%r11,%%r10), %%r10\n'
             % (DEPTH, DEPTH, framesz, frames))
    o += ['\tmovq\t%d(%%r10), %%%s\n' % (off, r) for r, off in regs]
    o += ['\tmovdqu\t%d(%%r10), %%%s\n' % (off, r) for r, off in xregs]
    o.append('\tldmxcsr\t%d(%%r10)\n' % MXCSR)
    # incssp takes the low 8 bits of its count: at most 255 entries at a time
    o.append('\tmovq\t%d(%%r10), %%rax\n\ttestq\t%%rax, %%rax\n\tjz\t.Lcimo_nossp\n'
             % SSP + _RDSSP_RCX + '\tsubq\t%rcx, %rax\n\tshrq\t$3, %rax\n'
             '\tjz\t.Lcimo_nossp\n.Lcimo_ssp:\n\tmovl\t$255, %ecx\n\tcmpq\t%rcx, %rax\n'
             '\tcmovbq\t%rax, %rcx\n' + _INCSSP_RCX + '\tsubq\t%rcx, %rax\n\tjnz\t.Lcimo_ssp\n'
             '.Lcimo_nossp:\n')
    o.append('\tmovq\t%d(%%r10), %%rsp\n\tmovq\t%d(%%r10), %%r10\n\tpushq\t%%r10\n' % (RSP, RET))
    o.append('.Lcimo_ret0:\n\txorl\t%eax, %eax\n\txorl\t%edx, %edx\n\tpxor\t%xmm0, %xmm0\n'
             '\tpxor\t%xmm1, %xmm1\n\tret\n.Lcimo_noframe:\n' + fn('__cimo_noframe') + '\tud2\n')
    # __cimo_fault: where cimo.fault's handler resumes a faulting kernel, with r11 = ctx
    o.append(fn('__cimo_fault') + '\tjmp\t.Lcimo_unwind\n')
    # __cimo_fail(code, msg, file, line, func): called by abort()/__assert_fail (C, with an rbp
    # frame, so 8(%rbp) is the address in their caller)
    a = ['rcx', 'rdx', 'r8', 'r9'] if windows else ['rdi', 'rsi', 'rdx', 'rcx', 'r8']
    o.append(fn('__cimo_fail') + '\tcallq\t__cimo_find\n\ttestq\t%r11, %r11\n'
             '\tjnz\t.Lcimo_failctx\n\tleaq\t__cimo_orphan(%rip), %r11\n.Lcimo_failctx:\n')
    o.append('\tmovslq\t%%%s, %%r10\n\tmovq\t%%r10, (%%r11)\n' % _w32(a[0]))
    o.append('\tmovq\t%%%s, %d(%%r11)\n\tmovq\t%%%s, %d(%%r11)\n' % (a[1], MSG, a[2], FILE))
    o.append('\tmovslq\t%%%s, %%r10\n\tmovq\t%%r10, %d(%%r11)\n' % (_w32(a[3]), LINE))
    o.append('\tmovq\t40(%%rsp), %%r10\n\tmovq\t%%r10, %d(%%r11)\n' % FUNC if windows else
             '\tmovq\t%%%s, %d(%%r11)\n' % (a[4], FUNC))
    o.append('\tmovq\t8(%%rbp), %%r10\n\tmovq\t%%r10, %d(%%r11)\n\tjmp\t.Lcimo_unwind\n' % PC)
    # __builtin_trap / __cimo_trap: pc = the call site
    o.append(fn('__builtin_trap') + fn('__cimo_trap'))
    o.append('\tcallq\t__cimo_find\n\ttestq\t%%r11, %%r11\n\tjnz\t.Lcimo_trapctx\n'
             '\tleaq\t__cimo_orphan(%%rip), %%r11\n.Lcimo_trapctx:\n'
             '\tmovq\t$%d, (%%r11)\n\tmovq\t(%%rsp), %%r10\n\tmovq\t%%r10, %d(%%r11)\n'
             '\tjmp\t.Lcimo_unwind\n' % (E_TRAP, PC))
    # __cimo_stkchk: called first thing in the prologue of functions that can recurse;
    # preserves every argument register
    o.append(fn('__cimo_stkchk') +
             '\tcallq\t__cimo_find\n\ttestq\t%%r11, %%r11\n\tjz\t.Lcimo_stkok\n'
             '\tcmpq\t%d(%%r11), %%rsp\n\tjb\t.Lcimo_stkovf\n.Lcimo_stkok:\n\tret\n'
             '.Lcimo_stkovf:\n\tmovq\t$%d, (%%r11)\n\tmovq\t(%%rsp), %%r10\n'
             '\tmovq\t%%r10, %d(%%r11)\n\tjmp\t.Lcimo_unwind\n' % (LIMIT, E_STACK, PC))
    o.append(fn('__cimo_ctx') + '\tcallq\t__cimo_find\n\tmovq\t%r11, %rax\n\tret\n')
    o.append(fn('__cimo_tid') + key + '\tmovq\t%r10, %rax\n\tret\n')
    # void *__cimo_cas(void **p, void *expect, void *desired): returns the old *p
    o.append(fn('__cimo_cas') + ('\tmovq\t%rdx, %rax\n\tlock cmpxchgq\t%r8, (%rcx)\n\tret\n'
                                 if windows else
                                 '\tmovq\t%rsi, %rax\n\tlock cmpxchgq\t%rdx, (%rdi)\n\tret\n'))
    o.append(fn('__cimo_barrier') + '\tmfence\n\tret\n')
    o.append(fn('__cimo_pause') + '\tpause\n\tret\n')
    # __cimo_call(fn, a, b, c) and __cimo_rethrow(ctx): as on arm64 (the shadow space the caller made for four
    # arguments serves the callee's three)
    o.append(fn('__cimo_call') + ('\tmovq\t%rcx, %r11\n\tmovq\t%rdx, %rcx\n\tmovq\t%r8, %rdx\n\tmovq\t%r9, %r8\n'
                                  if windows else
                                  '\tmovq\t%rdi, %r11\n\tmovq\t%rsi, %rdi\n\tmovq\t%rdx, %rsi\n\tmovq\t%rcx, %rdx\n')
             + '\tjmp\t__cimo_enter\n')
    o.append(fn('__cimo_rethrow') + '\tmovq\t%%%s, %%r11\n\tjmp\t.Lcimo_unwind\n' % ('rcx' if windows else 'rdi'))
    o.append(_setjmp_amd64(fn, windows))
    for name in names:
        o.append(fn('__cimo_entry_' + name))
        o.append('\tleaq\t%s(%%rip), %%r11\n\tjmp\t__cimo_enter\n' % name)
    o.append('\t.data\n\t.p2align\t3\n\t.globl\t__cimo_ctxtab\n__cimo_ctxtab:\n\t.quad\t0\n')
    o.append(_DATA % {'p': ''})
    return ''.join(o)


# setjmp and longjmp (C11 7.13), in every module's glue: the jmp_buf (include/setjmp.h, 32 longs) holds what a call
# must preserve, the stack pointer and setjmp's return address. arm64: x19-x30 at 0-88, sp at 96, d8-d15 at 104-160.
# x86-64: rbx, rbp, r12-r15 at 0-40, rsp after the return at 48, the return address at 56, the shadow stack pointer
# after the return at 64 (0 without shadow stacks; longjmp pops the entries the abandoned calls left, as the unwind
# does); Windows also rdi, rsi at 72, 80 and xmm6-xmm15 at 96-240. longjmp(env, 0) returns 1. The floating-point
# environment is not restored (C11 7.13.2.1). _setjmp/_longjmp (BSD's) are the same.
_SJ_NAMES = ('setjmp', '_setjmp')
_LJ_NAMES = ('longjmp', '_longjmp')


def _setjmp_arm64(fn):
    o = [''.join(fn(n) for n in _SJ_NAMES)]
    o += ['\tstp\t%s, %s, [x0, #%d]\n' % r for r in _SAVE]
    o.append('\tstp\tx29, x30, [x0, #80]\n\tmov\tx16, sp\n\tstr\tx16, [x0, #96]\n')
    o += ['\tstp\t%s, %s, [x0, #%d]\n' % r for r in _DSAVE]
    o.append('\tmov\tw0, #0\n\tret\n')
    o.append(''.join(fn(n) for n in _LJ_NAMES))
    o += ['\tldp\t%s, %s, [x0, #%d]\n' % r for r in _SAVE]
    o.append('\tldp\tx29, x30, [x0, #80]\n\tldr\tx16, [x0, #96]\n\tmov\tsp, x16\n')
    o += ['\tldp\t%s, %s, [x0, #%d]\n' % r for r in _DSAVE]
    o.append('\tcmp\tw1, #0\n\tcsinc\tw0, w1, wzr, ne\n\tret\n')
    return ''.join(o)


_RDSSP_RDX = '\t.byte\t0xf3, 0x48, 0x0f, 0x1e, 0xca\n'  # rdsspq %rdx


def _setjmp_amd64(fn, windows):
    buf, val = ('rcx', 'edx') if windows else ('rdi', 'esi')
    regs = [('rbx', 0), ('rbp', 8), ('r12', 16), ('r13', 24), ('r14', 32), ('r15', 40)]
    if windows:
        regs += [('rdi', 72), ('rsi', 80)]
    xregs = [('xmm%d' % (6 + k), 96 + 16 * k) for k in range(10)] if windows else []
    o = [''.join(fn(n) for n in _SJ_NAMES)]
    o += ['\tmovq\t%%%s, %d(%%%s)\n' % (r, off, buf) for r, off in regs]
    o += ['\tmovdqu\t%%%s, %d(%%%s)\n' % (r, off, buf) for r, off in xregs]
    o.append('\tleaq\t8(%%rsp), %%rdx\n\tmovq\t%%rdx, 48(%%%s)\n\tmovq\t(%%rsp), %%rdx\n\tmovq\t%%rdx, 56(%%%s)\n'
             % (buf, buf))
    o.append('\txorl\t%%edx, %%edx\n%s\ttestq\t%%rdx, %%rdx\n\tjz\t.Lcimo_sjnossp\n\taddq\t$8, %%rdx\n'
             '.Lcimo_sjnossp:\n\tmovq\t%%rdx, 64(%%%s)\n\txorl\t%%eax, %%eax\n\tret\n' % (_RDSSP_RDX, buf))
    o.append(''.join(fn(n) for n in _LJ_NAMES))
    o.append('\tmovl\t%%%s, %%eax\n\ttestl\t%%eax, %%eax\n\tjnz\t.Lcimo_ljval\n\tmovl\t$1, %%eax\n.Lcimo_ljval:\n'
             % val)
    if windows:  # the buffer's register is rcx, which incssp needs: move it to r11
        o.append('\tmovq\t%rcx, %r11\n')
        buf = 'r11'
    # incssp takes the low 8 bits of its count: at most 255 entries at a time
    o.append('\tmovq\t64(%%%s), %%r10\n\ttestq\t%%r10, %%r10\n\tjz\t.Lcimo_ljnossp\n' % buf + _RDSSP_RCX +
             '\tsubq\t%rcx, %r10\n\tshrq\t$3, %r10\n\tjz\t.Lcimo_ljnossp\n.Lcimo_ljssp:\n\tmovl\t$255, %ecx\n'
             '\tcmpq\t%rcx, %r10\n\tcmovbq\t%r10, %rcx\n' + _INCSSP_RCX + '\tsubq\t%rcx, %r10\n\tjnz\t.Lcimo_ljssp\n'
             '.Lcimo_ljnossp:\n')
    o += ['\tmovq\t%d(%%%s), %%%s\n' % (off, buf, r) for r, off in regs]
    o += ['\tmovdqu\t%d(%%%s), %%%s\n' % (off, buf, r) for r, off in xregs]
    o.append('\tmovq\t48(%%%s), %%rsp\n\tjmpq\t*56(%%%s)\n' % (buf, buf))
    return ''.join(o)


def _w32(r):
    """The 32-bit name of a 64-bit register."""
    return r + 'd' if r[1].isdigit() else 'e' + r[1:]


# ---------------------------------------------------------------------------------------------
# which functions get a stack check

def stack_checked(ssa):
    """Names of the functions in QBE IR text that can recurse: those in a cycle of the call
    graph, where an indirect call counts as a call to every function whose address is taken."""
    import re
    funcs = {}
    taken = set()
    cur = None
    for line in ssa.splitlines():
        m = re.match(r'(?:export\s+)?(?:vector \d+ )?function\s+(?:\S+\s+)?\$([\w.]+)\(', line)
        if m:
            cur = m.group(1)
            funcs[cur] = [set(), False]
            continue
        if line.startswith('}'):
            cur = None
            continue
        calls = re.findall(r'\bcall\s+(?:extern\s+)?([$%][\w.]+)', line)
        refs = re.findall(r'\$([\w.]+)', line)
        if cur is not None:
            for c in calls:
                if c[0] == '$':
                    funcs[cur][0].add(c[1:])
                else:
                    funcs[cur][1] = True
            called = set(c[1:] for c in calls if c[0] == '$')
            taken.update(r for r in refs if r not in called or line.count('$' + r) > 1)
        else:
            taken.update(refs)
    graph = {f: set(c for c in cs if c in funcs) | (set(t for t in taken if t in funcs)
                                                    if ind else set())
             for f, (cs, ind) in funcs.items()}
    # Tarjan's SCC, iteratively
    index, low, on, stack, out = {}, {}, set(), [], set()
    counter = [0]
    for root in graph:
        if root in index:
            continue
        work = [(root, iter(graph[root]))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(graph[w])))
                    break
                if w in on:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    scc = []
                    while True:
                        w = stack.pop()
                        on.discard(w)
                        scc.append(w)
                        if w == v:
                            break
                    if len(scc) > 1 or v in graph[v]:
                        out.update(scc)
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# Python side

def _interrupted():
    """Run Python's SIGINT handler for a Ctrl-C that stopped a kernel: KeyboardInterrupt, unless a handler of the
    program's own says otherwise. Off the main thread, the main thread gets it and this call raises
    KeyboardInterrupt."""
    import signal
    if threading.current_thread() is threading.main_thread():
        signal.raise_signal(signal.SIGINT)
        return
    import _thread
    _thread.interrupt_main()
    raise KeyboardInterrupt


class _Unmark:
    """Clears a context's CTX_PY word when its thread's threading.local goes (the thread ended), unless the module
    went first."""
    __slots__ = ('state', 'words')

    def __init__(self, state, words):
        self.state = weakref.ref(state)
        self.words = words

    def __del__(self):
        if self.state() is not None:
            self.words[CTX_PY // 8] = 0


class _Ctx(ctypes.Structure):
    _fields_ = [('err', ctypes.c_int64), ('msg', ctypes.c_void_p), ('file', ctypes.c_void_p),
                ('line', ctypes.c_int64), ('func', ctypes.c_void_p), ('pc', ctypes.c_uint64),
                ('depth', ctypes.c_int64), ('limit', ctypes.c_uint64), ('tid', ctypes.c_uint64),
                ('fault', ctypes.c_uint64), ('addr', ctypes.c_uint64)]


_GROW = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_NEWCTX = ctypes.CFUNCTYPE(ctypes.c_void_p)
_PAGE = _jit.PAGE


def _stack_bottom():
    """Lowest usable address of the calling thread's stack, or None if unknown."""
    libc = _jit._libc
    try:
        if sys.platform == 'win32':
            lo, hi = ctypes.c_size_t(), ctypes.c_size_t()
            ctypes.windll.kernel32.GetCurrentThreadStackLimits(ctypes.byref(lo), ctypes.byref(hi))
            return lo.value or None
        if sys.platform == 'darwin':
            self_ = libc.pthread_self
            self_.restype = ctypes.c_void_p
            t = self_()
            top, size = libc.pthread_get_stackaddr_np, libc.pthread_get_stacksize_np
            top.restype, top.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
            size.restype, size.argtypes = ctypes.c_size_t, [ctypes.c_void_p]
            return top(t) - size(t)
        self_ = libc.pthread_self
        self_.restype = ctypes.c_ulong
        attr = ctypes.create_string_buffer(256)
        get = libc.pthread_getattr_np
        get.argtypes = [ctypes.c_ulong, ctypes.c_void_p]
        if get(self_(), attr) != 0:
            return None
        addr, size = ctypes.c_void_p(), ctypes.c_size_t()
        libc.pthread_attr_getstack(attr, ctypes.byref(addr), ctypes.byref(size))
        libc.pthread_attr_destroy(attr)
        return addr.value
    except (AttributeError, TypeError, OSError):
        return None


_STATES: "weakref.WeakSet[State]" = weakref.WeakSet()  # every live module's State: errors name functions in any of them


class State:
    """A module's execution state: per-thread contexts, the context table, the heap's
    regions. Kept alive by the module and its kernels; unmaps everything when collected."""

    def __init__(self, image):
        self.image = image
        sym = image.symbols
        pre = image.prefix
        self._tab = image.address('__cimo_ctxtab')
        self._tidf = ctypes.CFUNCTYPE(ctypes.c_uint64)(image.address('__cimo_tid'))
        self._barrier = ctypes.CFUNCTYPE(None)(image.address('__cimo_barrier'))
        self._lock = threading.Lock()
        _STATES.add(self)
        self._tls = threading.local()
        self._ctxs = {}  # tid -> _Ctx
        self._maps = []  # (addr, size): contexts, tables and heap chunks
        self._chunks = []  # (addr, size): heap chunks
        self._large = {}  # addr -> size: large heap blocks, mapped one by one
        self._nslots = 0
        self._used = 0
        self._syms = None
        self._closed = False
        # heap hooks (runtime/heap.c); absent if the module was built without the runtime
        self._grow_cb = _GROW(self._grow)
        self._release_cb = _RELEASE(self._release)
        self._ctx_cb = _NEWCTX(self._newctx)
        for name, cb in (('__cimo_hook_grow', self._grow_cb),
                         ('__cimo_hook_release', self._release_cb),
                         ('__cimo_hook_ctx', self._ctx_cb)):
            if pre + name in sym:
                ctypes.c_void_p.from_address(sym[pre + name]).value = \
                    ctypes.cast(cb, ctypes.c_void_p).value
        if pre + '__cimo_hook_pool' in sym:  # runtime/omp.c: parallel loops, on cimo.parallel's threads
            from . import parallel
            parallel.attach(image, sym[pre + '__cimo_hook_pool'])
        self._reset_c = (ctypes.CFUNCTYPE(None, ctypes.c_void_p)(sym[pre + '__cimo_heap_reset'])
                         if pre + '__cimo_heap_reset' in sym else None)
        self._free_c = (ctypes.CFUNCTYPE(None, ctypes.c_void_p)(sym[pre + 'free'])
                        if pre + 'free' in sym else None)
        _fault.install()
        self._fault_slot = _fault.register(image)
        if self._fault_slot is not None:  # out of the table before the code is unmapped, whichever goes first
            image.on_close.append(self._unregister)

    # contexts

    def ctx(self):
        """The calling thread's context, registering the thread on first use."""
        try:
            return self._tls.ctx
        except AttributeError:
            pass
        _fault.thread()
        tid = self._tidf()
        with self._lock:
            c = self._ctxs.get(tid)
            if c is not None:  # a thread key reused by a new thread (Windows recycles TEBs): its own stack
                bottom = _stack_bottom()
                c.limit = bottom + STACK_MARGIN if bottom else 0
            if c is None:
                addr = self._map(CTX_SIZE)
                c = _Ctx.from_address(addr)
                c.tid = tid
                bottom = _stack_bottom()
                c.limit = bottom + STACK_MARGIN if bottom else 0
                self._ctxs[tid] = c
                self._insert(tid, addr)
        self._tls.ctx = c
        # the context as 64-bit words, for a kernel call's checks with no ctypes object made: err is word 0, the
        # stdout buffer's length word CTX_OUT // 8
        self._tls.words = memoryview((ctypes.c_char * CTX_SIZE).from_address(ctypes.addressof(c))).cast('B').cast('q')
        self._tls.words[CTX_PY // 8] = 1
        self._tls.unmark = _Unmark(self, self._tls.words)
        return c

    def _newctx(self):
        """The calling thread's context, for native code on a thread with none in this module (runtime/heap.c)."""
        try:
            return 0 if self._closed else ctypes.addressof(self.ctx())
        except BaseException:  # noqa: BLE001 (a callback can't raise: malloc returns NULL)
            return 0

    def _map(self, size):
        a = _jit._map(size, _jit.PROT_READ | _jit.PROT_WRITE)
        self._maps.append((a, size))
        return a

    def _insert(self, tid, ctx):
        if 2 * (self._used + 1) > self._nslots:
            n = max(64, 4 * self._nslots)
            shift = 64 - n.bit_length() + 1
            t = self._map(16 + 16 * n)
            for k, c in self._ctxs.items():
                if k != tid:
                    self._put(t, n, shift, k, ctypes.addressof(c))
            self._put(t, n, shift, tid, ctx)
            ctypes.c_uint64.from_address(t).value = n
            ctypes.c_uint64.from_address(t + 8).value = shift
            self._barrier()  # contents before the pointer
            ctypes.c_uint64.from_address(self._tab).value = t
            self._nslots, self._shift = n, shift
        else:
            t = ctypes.c_uint64.from_address(self._tab).value
            self._put(t, self._nslots, self._shift, tid, ctx)
        self._used += 1

    @staticmethod
    def _put(t, n, shift, key, ctx):
        i = ((key * _HASH_K) & _M64) >> shift
        while True:
            slot = t + 16 + 16 * i
            if ctypes.c_uint64.from_address(slot).value == 0:
                ctypes.c_uint64.from_address(slot + 8).value = ctx
                ctypes.c_uint64.from_address(slot).value = key
                return
            i = (i + 1) % n

    # errors

    def error(self, c, kernel):
        """Raise KernelError for the error recorded in context c (and clear it)."""
        code, pc, fault, addr = c.err, c.pc, c.fault, c.addr
        if getattr(self, 'tiles', False):  # an AMX kernel stopped midway: its tiles released (cimo.cpu)
            from . import cpu
            cpu.release_tiles()
        if code == E_INTERRUPT:  # Ctrl-C stopped the kernel (cimo.fault): Python's SIGINT handler, as if it came now
            c.err = c.pc = 0
            _interrupted()  # KeyboardInterrupt, normally; a handler that doesn't raise still has the kernel stopped
            where = self._symbol(pc) if pc else None
            raise KernelError('interrupted%s (calling %s)' % (' in %s()' % where if where else '', kernel),
                              'interrupt', kernel, where, None, None, None, None, None)
        expr = ctypes.string_at(c.msg).decode('utf-8', 'replace') if c.msg else None
        file = ctypes.string_at(c.file).decode('utf-8', 'replace') if c.file else None
        func = ctypes.string_at(c.func).decode('utf-8', 'replace') if c.func else None
        line = c.line
        c.err = c.msg = c.file = c.line = c.func = c.pc = c.fault = c.addr = 0
        # a fault's pc is the instruction itself; the others' are return addresses
        where = func or (self._symbol(pc + 4 if fault else pc) if pc else None)
        at = ' in %s()' % where if where else ''
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
        elif code == E_STACK:
            kind, msg = 'stack overflow', 'stack overflow%s: %s' % (
                at, 'hit the end of the stack' if fault else 'recursion too deep')
            if fault:
                _fault.stack_recovered()
        elif code == E_FAULT and fault == _fault.F_MEMORY:
            kind, msg = 'memory fault', 'memory fault at 0x%x%s' % (addr, at)
        elif code == E_FAULT and fault == _fault.F_DIVISION:
            kind, msg = 'division fault', 'integer division by zero or overflow%s' % at
        elif code == E_FAULT and fault == _fault.F_ILLEGAL:
            kind, msg = 'illegal instruction', 'illegal instruction%s' % at
        elif code == E_DEPTH:
            kind, msg = 'nesting', ('more than %d nested calls into the module on one thread'
                                    % MAXDEPTH)
        else:
            kind, msg = 'error', 'kernel error %d%s' % (code, at)
        raise KernelError('%s (calling %s)' % (msg, kernel), kind, kernel, where, expr, file,
                          line if code == E_ASSERT else None, status,
                          addr if fault == _fault.F_MEMORY else None)

    def _symbol(self, pc):
        """Name of the function containing code address pc: in this module, or in another one whose code this
        module's kernel called by address."""
        for st in list(_STATES):
            if st is not self and st.image._maps and st.image._maps[0][0] <= pc < st.image._maps[0][0] + st.image.code_size:
                return st._symbol(pc)
        if self._syms is None:
            pre = self.image.prefix
            items = sorted((a, n[len(pre):] if n.startswith(pre) else n)
                           for n, a in self.image.symbols.items()
                           if not n.startswith('.L') and not (pre and n.startswith('L')))
            self._syms = ([a for a, _ in items], [n for _, n in items])
        addrs, names = self._syms
        i = bisect.bisect_right(addrs, pc - 4) - 1  # pc is a return address: look before it
        if i < 0:
            return None
        return names[i].split('._omp_fn.')[0]  # a parallel loop's body is its function's (frontend/omp.py)

    # heap

    def _grow(self, heap, size, kind):
        try:
            size = (size + _PAGE - 1) & -_PAGE
            with self._lock:
                if self._closed:
                    return 0
                a = _jit._map(size, _jit.PROT_READ | _jit.PROT_WRITE)
                if kind == 1:
                    self._large[a] = size
                else:
                    self._chunks.append((a, size))
            return a
        except Exception:
            return 0

    def _release(self, ptr):
        try:
            with self._lock:
                size = self._large.pop(ptr, None)
            if size:
                _jit._unmap(ptr, size)
        except Exception:
            pass

    def reset_heap(self):
        """Free every heap block of every thread at once. Chunks stay mapped for reuse; large
        blocks are unmapped. No kernel of the module may be running."""
        if self._reset_c is None:
            raise RuntimeError('module has no heap')
        with self._lock:
            ctxs = list(self._ctxs.values())
            if any(c.depth for c in ctxs):
                raise RuntimeError('reset_heap() while a kernel of this module is running')
            large, self._large = self._large, {}
        for c in ctxs:
            self._reset_c(ctypes.addressof(c) + CTX_HEAP)
        for a, n in large.items():
            _jit._unmap(a, n)

    def free(self, ptr):
        """free() a heap pointer a kernel returned (from any thread)."""
        if self._free_c is None:
            raise RuntimeError('module has no heap')
        if ptr:
            self._free_c(ptr)

    def heap_bytes(self):
        """Bytes mapped for the heap (chunks and large blocks)."""
        with self._lock:
            return sum(n for _, n in self._chunks) + sum(self._large.values())

    def _unregister(self):
        slot, self._fault_slot = getattr(self, '_fault_slot', None), None
        if slot is not None:
            _fault.unregister(slot)

    def close(self):
        self._unregister()
        with self._lock:
            self._closed = True
            maps, self._maps = self._maps + self._chunks, []
            self._chunks = []
            large, self._large = self._large, {}
        for a, n in maps + list(large.items()):
            _jit._unmap(a, n)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
