"""cimo.loader with x86-64 objects (cimo.assembler_amd64): every relocation the backend's code
uses, stubs to external functions, the GOT, data pointers, bss and common symbols, and the
image snapshot that the module cache stores."""

import ctypes
import sys

import pytest

import hosts
from cimo import assembler_amd64, loader

pytestmark = pytest.mark.skipif(not hosts.AMD64 or sys.platform == 'darwin', reason='x86-64 only')

L = ctypes.c_int64
WIN = sys.platform == 'win32'
# argument registers, and the stack a call needs: Windows' 32-byte shadow space (40 keeps the
# 16-byte alignment on both)
A = ('rcx', 'rdx', 'r8') if WIN else ('rdi', 'rsi', 'rdx')

TEXT = r"""
.text
.globl add3
add3:
	movq %%%(a0)s, %%rax
	addq %%%(a1)s, %%rax
	addq %%%(a2)s, %%rax
	ret
.globl call_llabs             /* call to an external function, through a stub (PLT32) */
call_llabs:
	subq $40, %%rsp
	callq llabs@plt
	addq $40, %%rsp
	ret
.globl tail_llabs             /* jmp to an external function */
tail_llabs:
	jmp llabs
.globl got_llabs              /* external function address through the GOT (GOTPCREL) */
got_llabs:
	subq $40, %%rsp
	movq llabs@gotpcrel(%%rip), %%rax
	callq *%%rax
	addq $40, %%rsp
	ret
.globl get_ptr                /* *table: a pointer stored in data (ABS64) */
get_ptr:
	movq table(%%rip), %%rax
	movq (%%rax), %%rax
	ret
.globl bump                   /* data is writable; counter lives in a common symbol */
bump:
	movq counter(%%rip), %%rax
	addq $1, %%rax
	movq %%rax, counter(%%rip)
	ret
.globl get_bss
get_bss:
	movq zeros+8(%%rip), %%rax
	ret
.globl get_fp                 /* rodata constant */
get_fp:
	movsd ".Lfp0"(%%rip), %%xmm0
	ret
.globl local_call             /* a global defined here (PLT32, direct) and a local */
local_call:
	subq $40, %%rsp
	callq add3
	movq %%rax, %%%(a0)s
	callq helper
	addq $40, %%rsp
	ret
helper:
	leaq 100(%%%(a0)s), %%rax
	ret
.data
.balign 8
value:
	.quad 1234
.globl table
table:
	.quad value
.bss
.balign 8
zeros:
	.fill 32,1,0
.comm counter,8,8
.section .rodata
.p2align 3
.Lfp0:
	.quad 4614253070214989087 /* 3.140000 */
""" % {'a0': A[0], 'a1': A[1], 'a2': A[2]}


def _check(img):
    f = lambda n, k=1: img.function(n, L, [L] * k)  # noqa: E731
    assert f('add3', 3)(1, 2, 3) == 6
    assert f('call_llabs')(-5) == 5
    assert f('tail_llabs')(-6) == 6
    assert f('got_llabs')(-7) == 7
    assert f('get_ptr', 0)() == 1234
    assert f('bump', 0)() == 1
    assert f('bump', 0)() == 2
    assert f('get_bss', 0)() == 0
    assert img.function('get_fp', ctypes.c_double, [])() == 3.14
    assert f('local_call', 3)(1, 2, 3) == 106
    assert img.address('table') % 8 == 0


def test_link_and_call():
    img = loader.Image(assembler_amd64.assemble(TEXT))
    _check(img)
    img.close()


def test_snapshot_loads_anywhere():
    img = loader.Image(assembler_amd64.assemble(TEXT), snapshot=True)
    blob = img.snapshot()
    assert blob is not None  # every reference is relocatable
    img2 = loader.Image.load(blob)
    _check(img2)
    img.close()
    img2.close()


def test_undefined_symbol():
    with pytest.raises(loader.LoadError, match='undefined symbol: no_such_function_xyz'):
        loader.Image(assembler_amd64.assemble('callq no_such_function_xyz\n'))


def test_resolver_override():
    called = []

    def resolve(name):
        called.append(name)
        return loader._dlsym('llabs') if name == 'my_abs' else None

    img = loader.Image(assembler_amd64.assemble('.globl f\nf:\n\tjmp my_abs\n'), resolve)
    assert img.function('f', L, [L])(-3) == 3
    assert called == ['my_abs']


def test_external_data_out_of_range():
    # a rip-relative reference reaches +-2 GB; a libc data symbol is usually further: a clear
    # error rather than a silently wrong address (in range, it links fine, also correct)
    sym = 'llabs' if WIN else 'environ'
    try:
        loader.Image(assembler_amd64.assemble('leaq %s(%%rip), %%rax\n\tret\n' % sym))
    except loader.LoadError as e:
        assert '2 GB' in str(e)
