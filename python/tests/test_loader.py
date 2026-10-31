"""cimo.loader: linking cimo.assembler objects into the running process."""

import ctypes
import gc
import platform
import struct
import sys

import pytest

import assembler_check as A
from cimo import assembler, loader

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='arm64 only')

L = ctypes.c_int64

ELF = r"""
.text
.globl add3
add3:
	add	x0, x0, x1
	add	x0, x0, x2
	ret
.globl call_labs              /* bl to an external function, through a stub */
call_labs:
	stp	x29, x30, [sp, -16]!
	mov	x29, sp
	bl	labs
	ldp	x29, x30, [sp], 16
	ret
.globl tail_labs              /* b (JUMP26) to an external function */
tail_labs:
	b	labs
.globl got_labs               /* external function address through the GOT */
got_labs:
	stp	x29, x30, [sp, -16]!
	mov	x29, sp
	adrp	x1, :got:labs
	ldr	x1, [x1, #:got_lo12:labs]
	blr	x1
	ldp	x29, x30, [sp], 16
	ret
.globl get_ptr                /* *table: a pointer stored in data (ABS64) */
get_ptr:
	adrp	x0, table
	add	x0, x0, #:lo12:table
	ldr	x0, [x0]
	ldr	x0, [x0]
	ret
.globl bump                   /* data is writable; counter lives in a common symbol */
bump:
	adrp	x1, counter
	add	x1, x1, #:lo12:counter
	ldr	x0, [x1]
	add	x0, x0, #1
	str	x0, [x1]
	ret
.globl get_bss
get_bss:
	adrp	x0, zeros+8
	add	x0, x0, #:lo12:zeros+8
	ldr	x0, [x0]
	ret
.globl get_fp                 /* rodata constant */
get_fp:
	adrp	x0, ".Lfp0"
	add	x0, x0, #:lo12:".Lfp0"
	ldr	d0, [x0]
	ret
.globl local_call
local_call:
	stp	x29, x30, [sp, -16]!
	mov	x29, sp
	bl	add3
	bl	helper
	ldp	x29, x30, [sp], 16
	ret
helper:
	add	x0, x0, #100
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
"""


def _macho(text):
    """The same program in Mach-O syntax, as cimo.backend's arm64_apple target writes it."""
    import re
    t = re.sub(r'\b(add3|call_labs|tail_labs|got_labs|get_ptr|bump|get_bss|get_fp|local_call|'
               r'helper|labs|value|table|zeros|counter)\b', r'_\1', text)
    t = re.sub(r'adrp\t(x\d), :got:(\w+)', r'adrp\t\1, \2@gotpage', t)
    t = re.sub(r'\[(x\d), #:got_lo12:(\w+)\]', r'[\1, \2@gotpageoff]', t)
    t = re.sub(r'#:lo12:("?[\w.]+"?)(\+\d+)?', r'\1@pageoff\2', t)
    t = re.sub(r'adrp\t(x\d), ("?[\w.]+"?)(\+\d+)?\n', r'adrp\t\1, \2@page\3\n', t)
    t = t.replace('".Lfp0"', '"Lfp0"').replace('.Lfp0:', 'Lfp0:')
    t = t.replace('.section .rodata', '.section __TEXT,__literal8,8byte_literals')
    return t.replace('.comm _counter,8,8', '.comm _counter,8,3')


@pytest.mark.parametrize('syntax', ['elf', 'macho'])
def test_link_and_call(syntax):
    text = ELF if syntax == 'elf' else _macho(ELF)
    img = loader.Image(assembler.assemble(text, syntax))
    f = lambda n, k=1: img.function(n, L, [L] * k)  # noqa: E731
    assert f('add3', 3)(1, 2, 3) == 6
    assert f('call_labs')(-5) == 5
    assert f('tail_labs')(-6) == 6
    assert f('got_labs')(-7) == 7
    assert f('get_ptr', 0)() == 1234
    assert f('bump', 0)() == 1
    assert f('bump', 0)() == 2
    assert f('get_bss', 0)() == 0
    assert img.function('get_fp', ctypes.c_double, [])() == 3.14
    assert f('local_call', 3)(1, 2, 3) == 106
    assert img.address('table') % 8 == 0
    img.close()


def test_undefined_symbol():
    with pytest.raises(loader.LoadError, match='undefined symbol: no_such_function_xyz'):
        loader.Image(assembler.assemble('bl no_such_function_xyz\n'))


def test_resolver_override():
    called = []

    def resolve(name):
        called.append(name)
        return loader._dlsym('labs') if name == 'my_abs' else None

    img = loader.Image(assembler.assemble('.globl f\nf:\n\tb my_abs\n'), resolve)
    assert img.function('f', L, [L])(-3) == 3
    assert called == ['my_abs']


def test_external_data_out_of_range():
    # adrp can't reach a libc data symbol from the image (it is gigabytes away): clear error
    # rather than a silently wrong address. (If the mapping happens to land within 4 GB of
    # libc, it links fine, which is also correct.)
    try:
        sym = 'labs' if sys.platform == 'win32' else 'environ'  # the UCRT exports little data
        loader.Image(assembler.assemble('adrp x0, %s\n\tadd x0, x0, #:lo12:%s\n\tret\n'
                                        % (sym, sym)))
    except loader.LoadError as e:
        assert 'adrp range' in str(e)


def test_kernel_outlives_module():
    import cimo
    try:
        k = cimo.compile('long twice(long x) { return 2 * x; }').twice
    except cimo.CompileError as e:
        pytest.skip('compile unavailable: %s' % e)
    gc.collect()
    assert k(21) == 42


def test_close_unmaps():
    img = loader.Image(assembler.assemble('.globl f\nf:\n\tret\n'))
    assert img._maps
    img.close()
    assert img._maps == []
    img.close()  # idempotent


def test_flush_routine_encoding():
    if not A.have_gas():
        pytest.skip('GNU as not available')
    sections, relocs, _ = A.parse_elf(A.gas(loader._FLUSH_ASM))
    assert sections['.text'][0] == loader._FLUSH_CODE
    assert not relocs


def test_flush_icache_runs():
    # rewrite code in place: make RW, write, make RX, flush, run the new code
    if loader._MAC:
        pytest.skip('Linux W^X path')
    img = loader.Image(assembler.assemble('.globl f\nf:\n\tmov x0, #1\n\tret\n'))
    f = img.function('f', L, [])
    assert f() == 1
    base, size = img._maps[0]
    loader._protect(base, img.code_size, loader.PROT_READ | loader.PROT_WRITE)
    ctypes.memmove(img.address('f'), struct.pack('<I', 0xd2800040), 4)  # mov x0, #2
    loader._protect(base, img.code_size, loader.PROT_READ | loader.PROT_EXEC)
    loader.flush_icache(img.address('f'), 4)
    assert f() == 2
