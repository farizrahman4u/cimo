"""cimo.asm against GNU as, byte for byte (sections, relocations, symbols).

Skips when GNU as for aarch64 isn't installed. Inputs: the assembly cimo.qbe produces for QBE's
tests and scratch/qbe_eval kernels (both targets; arm64_apple is translated to ELF syntax for
GNU as), a hand-written file with every emitter form (data/arm64_forms.s), and generated sweeps
of immediates and offsets.
"""

import os
import random

import pytest

import asm_check as A
from cimo import asm

pytestmark = pytest.mark.skipif(not A.have_gas(), reason='GNU as (aarch64) not available')

CORPUS = A.corpus()
TLS = {'tls.ssa'}  # thread-local storage: rejected by design


def _id(p):
    return os.path.basename(p)


@pytest.mark.parametrize('path', CORPUS, ids=_id)
@pytest.mark.parametrize('target', ['arm64', 'arm64_apple'])
def test_corpus(path, target):
    text = A.qbe_asm(path, target)
    if os.path.basename(path) in TLS:
        with pytest.raises(asm.AsmError, match='thread-local'):
            asm.assemble(text, 'elf' if target == 'arm64' else 'macho')
        return
    d = A.compare(text) if target == 'arm64' else A.compare_macho(text)
    assert not d, '\n'.join(d)


def test_forms():
    with open(os.path.join(A.HERE, 'data', 'arm64_forms.s')) as f:
        text = f.read()
    d = A.compare(text)
    assert not d, '\n'.join(d)


def _bitmasks(width):
    out = set()
    size = 2
    while size <= width:
        for ones in range(1, size):
            for rot in range(size):
                e = (1 << ones) - 1
                e = ((e >> rot) | (e << (size - rot))) & ((1 << size) - 1)
                v = 0
                for k in range(width // size):
                    v |= e << (k * size)
                out.add(v)
        size *= 2
    return sorted(out)


def test_logical_immediates():
    lines = ['and x0, x1, #%d' % v for v in _bitmasks(64)]
    lines += ['orr w2, w3, #%d' % v for v in _bitmasks(32)]
    lines += ['eor w4, w5, #%d' % (v - (1 << 32)) for v in _bitmasks(32) if v >> 31]
    d = A.compare('\n'.join(lines) + '\n')
    assert not d, '\n'.join(d)


def test_immediate_sweep():
    rnd = random.Random(1)
    lines = []
    # mov: the three cases of loadcon (mov wide, movn, bitmask) for both widths
    vals = [0, 1, 65535, 65536, -1, -65536, 1 << 63, (1 << 64) - 1]
    vals += [rnd.randrange(65536) << (16 * rnd.randrange(4)) for _ in range(200)]
    vals += [~(rnd.randrange(65536) << (16 * rnd.randrange(4))) for _ in range(200)]
    vals += [rnd.choice(_bitmasks(64)) for _ in range(200)]
    for v in vals:
        lines.append('mov x%d, #%d' % (rnd.randrange(31), v if v < 1 << 63 else v - (1 << 64)))
    wvals = [rnd.randrange(65536) << (16 * rnd.randrange(2)) for _ in range(100)]
    wvals += [(~(rnd.randrange(65536) << (16 * rnd.randrange(2)))) & 0xffffffff
              for _ in range(100)]
    wvals += [rnd.choice(_bitmasks(32)) for _ in range(100)]
    for v in wvals:
        lines.append('mov w%d, #%d' % (rnd.randrange(31), v - (1 << 32) if v >> 31 else v))
    for _ in range(100):
        lines.append('movk x%d, #0x%x, lsl #%d'
                     % (rnd.randrange(31), rnd.randrange(65536), 16 * rnd.randrange(4)))
    # add/sub/cmp immediates, including the ones GNU as rewrites (shifted, negative)
    for _ in range(300):
        op = rnd.choice(['add', 'sub'])
        r = rnd.choice('wx')
        v = rnd.choice([rnd.randrange(4096), rnd.randrange(4096) << 12,
                        -rnd.randrange(1, 4096)])
        lines.append('%s %s%d, %s%d, #%d' % (op, r, rnd.randrange(31), r, rnd.randrange(31), v))
        lines.append('cmp %s%d, #%d' % (r, rnd.randrange(31), abs(v)))
    # shifts by immediate
    for r, w in (('w', 32), ('x', 64)):
        for s in range(w):
            for op in ('lsl', 'lsr', 'asr'):
                lines.append('%s %s1, %s2, #%d' % (op, r, r, s))
    # loads and stores: every (mnemonic, class) with scaled and unscaled offsets
    for (mn, cls), (_, sz) in sorted(asm._LDST.items()):
        for _ in range(40):
            if rnd.random() < 0.5:
                off = rnd.randrange(4096) << sz
            else:
                off = rnd.randrange(-256, 256)
            base = rnd.choice(['x%d' % rnd.randrange(31), 'sp'])
            lines.append('%s %s%d, [%s, %d]' % (mn, cls, rnd.randrange(31), base, off))
        for mode in ('[sp, %d]!', '[x3], %d'):
            lines.append(('%s %s%d, ' + mode) % (mn, cls, rnd.randrange(31),
                                                 rnd.randrange(-256, 256)))
    for off in range(-512, 512, 8):
        lines.append('stp x29, x30, [sp, %d]!' % off)
        lines.append('ldp x29, x30, [sp], %d' % off)
        lines.append('stp x1, x2, [x3, %d]' % off)
    d = A.compare('\n'.join(lines) + '\n')
    assert not d, '\n'.join(d)


@pytest.mark.parametrize('src', [
    'mrs x0, tpidr_el0',
    'add x0, x0, #:tprel_lo12_nc:x',
    '.section .tdata,"awT"',
    '.section .tbss,"awT"',
    '.abort "unreachable"',
    'fmadd d0, d1, d2, d3',
    'ldr x0, [x1, x2]',
    'ldr x0, [x1, 32768]',
    'ldr x0, [x1, -257]',
    'ldr w0, [x1, 257]',
    'add x0, x1, #4097',
    'mov x0, #0x12345',
    'and x0, x1, #0',
    'lsl w0, w1, #32',
    'add x0, w1, w2',
    'movk w0, #1, lsl #32',
    '.weird 1',
    '.byte sym',
    'stp x0, x1, [sp, 4]',
])
def test_rejects(src):
    with pytest.raises(asm.AsmError):
        asm.assemble(src + '\n')


def test_macho_rejects_tlv():
    with pytest.raises(asm.AsmError, match='thread-local'):
        asm.assemble('.section __DATA,__thread_vars,thread_local_variables\n', 'macho')
    with pytest.raises(asm.AsmError, match='thread-local'):
        asm.assemble('adrp x0, _x@tlvppage\n', 'macho')


def test_branch_resolution():
    o = asm.assemble('.globl g\ng:\n\tb l\n\tbl g\nl:\n\tret\n')
    t = o.section('.text')
    assert t.data[:4] == bytes.fromhex('02000014')  # b l, resolved
    assert [(r.offset, r.type, r.sym) for r in t.relocs] == [(4, 'CALL26', 'g')]
