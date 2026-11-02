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
- How kernel code finds its context: a per-module hash table keyed by the OS thread's
  thread-pointer register (`tpidr_el0` on Linux, `tpidrro_el0` on macOS, both readable
  from user mode and unique per live thread; on Windows x18, which always holds the
  thread's TEB), filled by Python the first time a thread calls into the module. Rejected
  alternatives: x18 as a free register (reserved on macOS and Windows); TLS (the loader would need TLS relocations and a TLS block per module); a
  hidden argument (changes every function's ABI and the frontend). The lookup costs about 20
  instructions and runs once per kernel call, `malloc` and `free`; never in loops otherwise.
- One context per (module, thread), allocated by Python (two pages). Its layout is shared by
  the assembly below and runtime/heap.c (`CTX_HEAP`):

      0 err   8 msg   16 file   24 line   32 func   40 pc   48 depth   56 stack limit   64 tid
      128 frames: MAXDEPTH x 176 bytes (x19-x28, x29, lr, sp, d8-d15; x86-64 Linux: rbx,
          rbp, r12-r15, the return address and rsp)
      3072 the thread's heap (struct heap in runtime/heap.c, at most 1024 bytes)
      4096 the thread's stdout buffer (runtime/stdio/stdout.c: length, then 4088 bytes)

- Re-entry (a kernel calls a Python callback that calls a kernel of the same module on the
  same thread) pushes another frame; an error unwinds only the innermost call.
- Stack guard: functions that can recurse (in a call-graph cycle, counting indirect calls as
  edges to every address-taken function) call `__cimo_stkchk` in their prologue, which compares
  sp against the thread's stack bottom + STACK_MARGIN (from pthread_getattr_np /
  pthread_get_stackaddr_np; GetCurrentThreadStackLimits on Windows). `alloca`/VLAs and frames larger than the margin are not covered.
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

from . import loader as _jit

__all__ = ['KernelError', 'glue', 'stack_checked', 'State']

CTX_SIZE = 8192
# Windows x64 frames also hold rdi, rsi and xmm6-xmm15 (240 bytes): they go after the stdout
# buffer, in a third page (_glue_amd64)
_WIN_X64 = sys.platform == 'win32' and _jit._X86
WFRAMES, WFRAMESZ = 8192, 248
if _WIN_X64:
    CTX_SIZE = 12288
ERR, MSG, FILE, LINE, FUNC, PC, DEPTH, LIMIT, TID = range(0, 72, 8)
FRAMES, FRAMESZ, MAXDEPTH = 128, 176, 16
CTX_HEAP = 3072
CTX_OUT = 4096
STACK_MARGIN = 128 << 10

E_ABORT, E_ASSERT, E_TRAP, E_STACK, E_DEPTH, E_EXIT = 1, 2, 3, 4, 5, 6
_HASH_K = 0x9E3779B97F4A7C15
_M64 = (1 << 64) - 1


class KernelError(Exception):
    """A kernel called abort() or exit(), failed an assert, trapped or overflowed its stack.
    Attributes: kind ('abort', 'exit', 'assert', 'trap', 'stack overflow', 'nesting'), kernel
    (the entry point Python called), function (where it happened, if known), for assert:
    expr, file, line, and for exit: status."""

    kind: str | None
    kernel: str | None
    function: str | None
    expr: str | None
    file: str | None
    line: int | None
    status: int | None

    def __init__(self, message: str, kind: str | None = None, kernel: str | None = None,
                 function: str | None = None, expr: str | None = None,
                 file: str | None = None, line: int | None = None,
                 status: int | None = None) -> None:
        Exception.__init__(self, message)
        self.kind, self.kernel, self.function = kind, kernel, function
        self.expr, self.file, self.line, self.status = expr, file, line, status


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
    o.append('\tmov\tx19, x9\n\tmov\tx20, x11\n\tblr\tx14\n'
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
    o.append('%sret0:\n\tmov\tx0, #0\n\tmov\tx1, #0\n\tfmov\td0, x0\n\tfmov\td1, x0\n\tret\n'
             '%snoframe:\n\tbrk\t#0xc1\n' % (L, L))
    # __cimo_fail(code, msg, file, line, func): called by abort()/__assert_fail (C, with a
    # frame, so [x29, #8] is the address in their caller)
    o.append(fn('__cimo_fail') + find())
    o.append('\tcmp\tx9, #0\n\tb.eq\t%snoframe\n\tstr\tx0, [x9]\n\tstp\tx1, x2, [x9, #%d]\n'
             '\tstp\tx3, x4, [x9, #%d]\n\tldr\tx10, [x29, #8]\n\tstr\tx10, [x9, #%d]\n'
             '\tb\t%sunwind\n' % (L, MSG, LINE, PC, L))
    # __builtin_trap / __cimo_trap: pc = the call site
    for name in ('__builtin_trap', '__cimo_trap'):
        o.append(fn(name))
    o.append(find() + '\tcmp\tx9, #0\n\tb.eq\t%snoframe\n\tmov\tx10, #%d\n\tstr\tx10, [x9]\n'
             '\tstr\tx30, [x9, #%d]\n\tb\t%sunwind\n' % (L, E_TRAP, PC, L))
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
    for name in names:
        o.append(fn('__cimo_entry_' + name))
        o.append('\tmov\tx15, x30\n\tadrp\tx14, %s%s%s\n\tadd\tx14, x14, %s\n\tb\t%s__cimo_enter\n'
                 % (p, name, '@PAGE' if apple else '',
                    p + name + '@PAGEOFF' if apple else '#:lo12:' + name, p))
    o.append('\t.data\n\t.p2align\t3\n\t.globl\t%s\n%s:\n\t.quad\t0\n' % (tab, tab))
    return ''.join(o)


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
    o.append('\tmovq\t%%r11, %%r13\n\tmovq\t%%r10, %%r12\n'   # r13 = ctx, r12 = frame
             '\tpopq\t%%r11\n\tpopq\t%%r10\n'                # kernel, return address
             '\tmovq\t%%r10, %d(%%r12)\n\tmovq\t%%rsp, %d(%%r12)\n'
             '\txorl\t%%r10d, %%r10d\n%s\tmovq\t%%r10, %d(%%r12)\n'
             '\tcallq\t*%%r11\n'
             '\tsubq\t$1, %d(%%r13)\n\tmovq\t%d(%%r12), %%r10\n'
             '\tmovq\t24(%%r12), %%r13\n\tmovq\t16(%%r12), %%r12\n\tpushq\t%%r10\n\tret\n'
             % (RET, RSP, _RDSSP_R10, SSP, DEPTH, RET))
    o.append('.Lcimo_deep:\n\tmovq\t$%d, (%%r11)\n\taddq\t$8, %%rsp\n\tjmp\t.Lcimo_ret0\n'
             % E_DEPTH)
    # unwind: r11 = ctx with the error stored; back to the innermost trampoline's caller
    o.append('.Lcimo_unwind:\n\tmovq\t%d(%%r11), %%r10\n\ttestq\t%%r10, %%r10\n'
             '\tjz\t.Lcimo_noframe\n\tsubq\t$1, %%r10\n\tmovq\t%%r10, %d(%%r11)\n'
             '\timulq\t$%d, %%r10, %%r10\n\tleaq\t%d(%%r11,%%r10), %%r10\n'
             % (DEPTH, DEPTH, framesz, frames))
    o += ['\tmovq\t%d(%%r10), %%%s\n' % (off, r) for r, off in regs]
    o += ['\tmovdqu\t%d(%%r10), %%%s\n' % (off, r) for r, off in xregs]
    # incssp takes the low 8 bits of its count: at most 255 entries at a time
    o.append('\tmovq\t%d(%%r10), %%rax\n\ttestq\t%%rax, %%rax\n\tjz\t.Lcimo_nossp\n'
             % SSP + _RDSSP_RCX + '\tsubq\t%rcx, %rax\n\tshrq\t$3, %rax\n'
             '\tjz\t.Lcimo_nossp\n.Lcimo_ssp:\n\tmovl\t$255, %ecx\n\tcmpq\t%rcx, %rax\n'
             '\tcmovbq\t%rax, %rcx\n' + _INCSSP_RCX + '\tsubq\t%rcx, %rax\n\tjnz\t.Lcimo_ssp\n'
             '.Lcimo_nossp:\n')
    o.append('\tmovq\t%d(%%r10), %%rsp\n\tmovq\t%d(%%r10), %%r10\n\tpushq\t%%r10\n' % (RSP, RET))
    o.append('.Lcimo_ret0:\n\txorl\t%eax, %eax\n\txorl\t%edx, %edx\n\tpxor\t%xmm0, %xmm0\n'
             '\tpxor\t%xmm1, %xmm1\n\tret\n.Lcimo_noframe:\n\tud2\n')
    # __cimo_fail(code, msg, file, line, func): called by abort()/__assert_fail (C, with an rbp
    # frame, so 8(%rbp) is the address in their caller)
    a = ['rcx', 'rdx', 'r8', 'r9'] if windows else ['rdi', 'rsi', 'rdx', 'rcx', 'r8']
    o.append(fn('__cimo_fail') + '\tcallq\t__cimo_find\n\ttestq\t%r11, %r11\n'
             '\tjz\t.Lcimo_noframe\n')
    o.append('\tmovslq\t%%%s, %%r10\n\tmovq\t%%r10, (%%r11)\n' % _w32(a[0]))
    o.append('\tmovq\t%%%s, %d(%%r11)\n\tmovq\t%%%s, %d(%%r11)\n' % (a[1], MSG, a[2], FILE))
    o.append('\tmovslq\t%%%s, %%r10\n\tmovq\t%%r10, %d(%%r11)\n' % (_w32(a[3]), LINE))
    o.append('\tmovq\t40(%%rsp), %%r10\n\tmovq\t%%r10, %d(%%r11)\n' % FUNC if windows else
             '\tmovq\t%%%s, %d(%%r11)\n' % (a[4], FUNC))
    o.append('\tmovq\t8(%%rbp), %%r10\n\tmovq\t%%r10, %d(%%r11)\n\tjmp\t.Lcimo_unwind\n' % PC)
    # __builtin_trap / __cimo_trap: pc = the call site
    o.append(fn('__builtin_trap') + fn('__cimo_trap'))
    o.append('\tcallq\t__cimo_find\n\ttestq\t%%r11, %%r11\n\tjz\t.Lcimo_noframe\n'
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
    for name in names:
        o.append(fn('__cimo_entry_' + name))
        o.append('\tleaq\t%s(%%rip), %%r11\n\tjmp\t__cimo_enter\n' % name)
    o.append('\t.data\n\t.p2align\t3\n\t.globl\t__cimo_ctxtab\n__cimo_ctxtab:\n\t.quad\t0\n')
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
        m = re.match(r'(?:export\s+)?function\s+(?:\S+\s+)?\$([\w.]+)\(', line)
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

class _Ctx(ctypes.Structure):
    _fields_ = [('err', ctypes.c_int64), ('msg', ctypes.c_void_p), ('file', ctypes.c_void_p),
                ('line', ctypes.c_int64), ('func', ctypes.c_void_p), ('pc', ctypes.c_uint64),
                ('depth', ctypes.c_int64), ('limit', ctypes.c_uint64), ('tid', ctypes.c_uint64)]


_GROW = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int)
_RELEASE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
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
        for name, cb in (('__cimo_hook_grow', self._grow_cb),
                         ('__cimo_hook_release', self._release_cb)):
            if pre + name in sym:
                ctypes.c_void_p.from_address(sym[pre + name]).value = \
                    ctypes.cast(cb, ctypes.c_void_p).value
        self._reset_c = (ctypes.CFUNCTYPE(None, ctypes.c_void_p)(sym[pre + '__cimo_heap_reset'])
                         if pre + '__cimo_heap_reset' in sym else None)
        self._free_c = (ctypes.CFUNCTYPE(None, ctypes.c_void_p)(sym[pre + 'free'])
                        if pre + 'free' in sym else None)

    # contexts

    def ctx(self):
        """The calling thread's context, registering the thread on first use."""
        try:
            return self._tls.ctx
        except AttributeError:
            pass
        tid = self._tidf()
        with self._lock:
            c = self._ctxs.get(tid)
            if c is None:
                addr = self._map(CTX_SIZE)
                c = _Ctx.from_address(addr)
                c.tid = tid
                bottom = _stack_bottom()
                c.limit = bottom + STACK_MARGIN if bottom else 0
                self._ctxs[tid] = c
                self._insert(tid, addr)
        self._tls.ctx = c
        return c

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
        code, pc = c.err, c.pc
        expr = ctypes.string_at(c.msg).decode('utf-8', 'replace') if c.msg else None
        file = ctypes.string_at(c.file).decode('utf-8', 'replace') if c.file else None
        func = ctypes.string_at(c.func).decode('utf-8', 'replace') if c.func else None
        line = c.line
        c.err = c.msg = c.file = c.line = c.func = c.pc = 0
        where = func or (self._symbol(pc) if pc else None)
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
            kind, msg = 'stack overflow', 'stack overflow%s: recursion too deep' % at
        elif code == E_DEPTH:
            kind, msg = 'nesting', ('more than %d nested calls into the module on one thread'
                                    % MAXDEPTH)
        else:
            kind, msg = 'error', 'kernel error %d%s' % (code, at)
        raise KernelError('%s (calling %s)' % (msg, kernel), kind, kernel, where, expr, file,
                          line if code == E_ASSERT else None, status)

    def _symbol(self, pc):
        """Name of the function containing code address pc."""
        if self._syms is None:
            pre = self.image.prefix
            items = sorted((a, n[len(pre):] if n.startswith(pre) else n)
                           for n, a in self.image.symbols.items()
                           if not n.startswith('.L') and not (pre and n.startswith('L')))
            self._syms = ([a for a, _ in items], [n for _, n in items])
        addrs, names = self._syms
        i = bisect.bisect_right(addrs, pc - 4) - 1  # pc is a return address: look before it
        return names[i] if i >= 0 else None

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

    def close(self):
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
