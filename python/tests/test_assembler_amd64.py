"""cimo.assembler_amd64 against GNU as (x86-64 ELF), byte for byte (sections, relocations, symbols).

Inputs: the assembly QBE produces for its tests (both amd64 targets; live GNU as only), a
hand-written file with every emitter form (data/amd64_forms.s), and generated sweeps of operand
forms, immediates and jump relaxation. GNU as's output for the forms file and the sweeps is
recorded in data/amd64_expect.json, so those run without GNU as too (refresh with
`python tests/assembler_amd64_check.py --record`). Plus an execution smoke test on x86-64 hosts.
"""

import ctypes
import os
import platform
import random
import sys

import pytest

import assembler_amd64_check as C
from cimo import assembler_amd64

live = pytest.mark.skipif(not C.have_gas(), reason='GNU as (x86-64 ELF) not available')

TLS = {'tls.ssa'}  # thread-local storage: rejected by design


def _id(p):
    return os.path.basename(p)


@live
@pytest.mark.parametrize('path', C.corpus(), ids=_id)
@pytest.mark.parametrize('target', ['amd64_sysv', 'amd64_win'])
def test_corpus(path, target):
    if os.path.basename(path) in TLS:
        if target == 'amd64_win':
            pytest.skip('QBE rejects thread-local storage on amd64_win')
        text = C.qbe_asm(path, target)
        with pytest.raises(assembler_amd64.AssemblerError, match='thread-local'):
            assembler_amd64.assemble(text)
        return
    text = C.qbe_asm(path, target)
    d = C.compare(text)
    assert not d, '\n'.join(d)


# ---------------------------------------------------------------------------------------------
# generated inputs (deterministic: their GNU as output is recorded)

R64 = ['%rax', '%rcx', '%rdx', '%rbx', '%rsp', '%rbp', '%rsi', '%rdi'] + \
      ['%%r%d' % i for i in range(8, 16)]
R32 = ['%eax', '%ecx', '%edx', '%ebx', '%esp', '%ebp', '%esi', '%edi'] + \
      ['%%r%dd' % i for i in range(8, 16)]
R16 = ['%ax', '%cx', '%dx', '%bx', '%sp', '%bp', '%si', '%di'] + \
      ['%%r%dw' % i for i in range(8, 16)]
R8 = ['%al', '%cl', '%dl', '%bl', '%spl', '%bpl', '%sil', '%dil'] + \
     ['%%r%db' % i for i in range(8, 16)]
REGS = {1: R8, 2: R16, 4: R32, 8: R64}
SFX = {1: 'b', 2: 'w', 4: 'l', 8: 'q'}
XMM = ['%%xmm%d' % i for i in range(16)]
DISPS = [0, 1, -1, 8, 127, 128, -128, -129, 4096, -100000, 0x7fffffff, -0x80000000]


def _imms(size):
    v = [0, 1, -1, 2, 127, 128, -128, -129, 255, 256]
    if size >= 2:
        v += [32767, -32768, 65535, 1000, -1000]
    if size >= 4:
        v += [0x7fffffff, -0x80000000, 100000, -100000]
    if size == 4:
        v += [0xffffffff, 0x80000000, 0xffffff80]
    if size == 1:
        v = [x for x in v if -128 <= x <= 255]
    if size == 2:
        v = [x for x in v if -32768 <= x <= 65535]
    return v


def _mem(rnd):
    k = rnd.randrange(8)
    d = rnd.choice(DISPS)
    b = rnd.choice(R64)
    i = rnd.choice([r for r in R64 if r != '%rsp'])
    s = rnd.choice('1248')
    if k == 0:
        return '(%s)' % b
    if k == 1:
        return '%d(%s)' % (d, b)
    if k == 2:
        return '%d(%s,%s,%s)' % (d, b, i, s)
    if k == 3:
        return '(%s,%s)' % (b, i)
    if k == 4:
        return '%d(,%s,%s)' % (d, i, s)
    if k == 5:
        return rnd.choice(['sym', 'sym+8', 'sym-4', '.Lloc', 'ext']) + '(%rip)'
    if k == 6:
        return rnd.choice(['sym', 'ext+4']) + '(%s)' % b
    return str(rnd.choice([0, 16, 0x1000, -8]))


def _rm(rnd, size):
    return rnd.choice(REGS[size]) if rnd.random() < 0.5 else _mem(rnd)


def sweep_integer():
    rnd = random.Random(11)
    out = []
    for size in (1, 2, 4, 8):
        sf = SFX[size]
        for mn in ('add', 'or', 'adc', 'sbb', 'and', 'sub', 'xor', 'cmp', 'test'):
            for _ in range(12):
                r1, r2 = rnd.choice(REGS[size]), rnd.choice(REGS[size])
                out.append('%s%s %s, %s' % (mn, sf, r1, r2))
                out.append('%s%s %s, %s' % (mn, sf, r1, _mem(rnd)))
                out.append('%s%s %s, %s' % (mn, sf, _mem(rnd), r2))
            for v in _imms(size):
                out.append('%s%s $%d, %s' % (mn, sf, v, rnd.choice(REGS[size])))
                out.append('%s%s $%d, %s' % (mn, sf, v, REGS[size][0]))
                out.append('%s%s $%d, %s' % (mn, sf, v, _mem(rnd)))
        for mn in ('mov',):
            for _ in range(30):
                out.append('mov%s %s, %s' % (sf, rnd.choice(REGS[size]), _rm(rnd, size)))
                out.append('mov%s %s, %s' % (sf, _mem(rnd), rnd.choice(REGS[size])))
            for v in _imms(size):
                out.append('mov%s $%d, %s' % (sf, v, rnd.choice(REGS[size])))
                out.append('mov%s $%d, %s' % (sf, v, _mem(rnd)))
        for mn in ('rol', 'ror', 'rcl', 'rcr', 'shl', 'sal', 'shr', 'sar'):
            for _ in range(4):
                out.append('%s%s %%cl, %s' % (mn, sf, _rm(rnd, size)))
                out.append('%s%s $1, %s' % (mn, sf, _rm(rnd, size)))
                out.append('%s%s $%d, %s' % (mn, sf, rnd.randrange(2, 8 * size),
                                             _rm(rnd, size)))
                out.append('%s%s %s' % (mn, sf, _rm(rnd, size)))
        for mn in ('not', 'neg', 'mul', 'div', 'idiv', 'inc', 'dec', 'imul'):
            for _ in range(6):
                out.append('%s%s %s' % (mn, sf, _rm(rnd, size)))
        for _ in range(6):
            out.append('xchg%s %s, %s' % (sf, rnd.choice(REGS[size]), rnd.choice(REGS[size])))
            out.append('xchg%s %s, %s' % (sf, REGS[size][0], rnd.choice(REGS[size])))
            out.append('xchg%s %s, %s' % (sf, rnd.choice(REGS[size]), _mem(rnd)))
            out.append('xchg%s %s, %s' % (sf, _mem(rnd), rnd.choice(REGS[size])))
            out.append('lock cmpxchg%s %s, %s' % (sf, rnd.choice(REGS[size]), _mem(rnd)))
            out.append('lock xadd%s %s, %s' % (sf, rnd.choice(REGS[size]), _mem(rnd)))
            out.append('lock add%s $%d, %s' % (sf, rnd.choice(_imms(size)), _mem(rnd)))
        if size == 1:
            continue
        for _ in range(10):
            out.append('imul%s %s, %s' % (sf, _rm(rnd, size), rnd.choice(REGS[size])))
            v = rnd.choice(_imms(size))
            out.append('imul%s $%d, %s, %s' % (sf, v, _rm(rnd, size), rnd.choice(REGS[size])))
            out.append('imul%s $%d, %s' % (sf, v, rnd.choice(REGS[size])))
            for mn in ('bsf', 'bsr', 'lzcnt', 'tzcnt', 'popcnt'):
                out.append('%s%s %s, %s' % (mn, sf, _rm(rnd, size), rnd.choice(REGS[size])))
            for cc in ('z', 'nz', 'b', 'ae', 'l', 'ge', 'be', 'a', 'p', 'np', 'g', 'le'):
                out.append('cmov%s %s, %s' % (cc, _rm(rnd, size), rnd.choice(REGS[size])))
        if size >= 4:
            for _ in range(10):
                out.append('bswap%s %s' % (sf, rnd.choice(REGS[size])))
                out.append('lea%s %s, %s' % (sf, _mem(rnd), rnd.choice(REGS[size])))
    for r in R64:
        out.append('pushq %s' % r)
        out.append('popq %s' % r)
    for v in _imms(4):
        if -0x80000000 <= v <= 0x7fffffff:
            out.append('pushq $%d' % v)
    for _ in range(10):
        out.append('pushq %s' % _mem(rnd))
        out.append('popq %s' % _mem(rnd))
    for mn, ss, ds in (('movsbw', 1, 2), ('movsbl', 1, 4), ('movsbq', 1, 8), ('movswl', 2, 4),
                       ('movswq', 2, 8), ('movslq', 4, 8), ('movzbw', 1, 2), ('movzbl', 1, 4),
                       ('movzbq', 1, 8), ('movzwl', 2, 4), ('movzwq', 2, 8)):
        for _ in range(12):
            out.append('%s %s, %s' % (mn, _rm(rnd, ss), rnd.choice(REGS[ds])))
    for cc in ('o no b c nae ae nb nc e z ne nz be na a nbe s ns p pe np po l nge ge nl le ng '
               'g nle').split():
        out.append('set%s %s' % (cc, _rm(rnd, 1)))
    for v in [0, 1, -1, 0x7fffffff, 0x80000000, -0x80000000, -0x80000001, 0xffffffff,
              1 << 32, 0x123456789abcdef, -(1 << 63), (1 << 63) - 1]:
        for r in rnd.sample(R64, 4):
            out.append('movq $%d, %s' % (v, r))
            out.append('movabsq $%d, %s' % (v, r))
    for r in R32:
        out.append('movl %s, %s' % (r, r))
    for _ in range(20):
        out.append('callq *%s' % _rm(rnd, 8))
        out.append('jmpq *%s' % _rm(rnd, 8))
    return out


def sweep_sse():
    rnd = random.Random(12)
    out = []

    def xm():
        return rnd.choice(XMM) if rnd.random() < 0.5 else _mem(rnd)

    rr = ['addss', 'addsd', 'subss', 'subsd', 'mulss', 'mulsd', 'divss', 'divsd', 'sqrtss',
          'sqrtsd', 'minss', 'minsd', 'maxss', 'maxsd', 'addps', 'addpd', 'andps', 'andpd',
          'andnps', 'andnpd', 'orps', 'orpd', 'xorps', 'xorpd', 'pxor', 'pand', 'por',
          'ucomiss', 'ucomisd', 'comiss', 'comisd', 'cvtss2sd', 'cvtsd2ss', 'unpcklps',
          'unpcklpd', 'cvtps2pd', 'cvtpd2ps']
    for mn in rr:
        for _ in range(8):
            out.append('%s %s, %s' % (mn, xm(), rnd.choice(XMM)))
    for mn in ('movss', 'movsd', 'movaps', 'movapd', 'movups', 'movupd', 'movdqa', 'movdqu'):
        for _ in range(8):
            out.append('%s %s, %s' % (mn, xm(), rnd.choice(XMM)))
            out.append('%s %s, %s' % (mn, rnd.choice(XMM), _mem(rnd)))
    for _ in range(16):
        out.append('movq %s, %s' % (rnd.choice(XMM), rnd.choice(R64)))
        out.append('movq %s, %s' % (rnd.choice(R64), rnd.choice(XMM)))
        out.append('movq %s, %s' % (xm(), rnd.choice(XMM)))
        out.append('movq %s, %s' % (rnd.choice(XMM), _mem(rnd)))
        out.append('movd %s, %s' % (rnd.choice(XMM), rnd.choice(R32)))
        out.append('movd %s, %s' % (rnd.choice(R32), rnd.choice(XMM)))
        out.append('movd %s, %s' % (_mem(rnd), rnd.choice(XMM)))
        out.append('movd %s, %s' % (rnd.choice(XMM), _mem(rnd)))
    for k in ('ss', 'sd'):
        for _ in range(8):
            for size, regs in ((4, R32), (8, R64)):
                out.append('cvtsi2%s %s, %s' % (k, rnd.choice(regs), rnd.choice(XMM)))
                out.append('cvtsi2%s%s %s, %s' % (k, SFX[size], rnd.choice(regs),
                                                  rnd.choice(XMM)))
                out.append('cvtsi2%s%s %s, %s' % (k, SFX[size], _mem(rnd), rnd.choice(XMM)))
                for t in ('cvtt', 'cvt'):
                    out.append('%s%s2si %s, %s' % (t, k, xm(), rnd.choice(regs)))
                    out.append('%s%s2si%s %s, %s' % (t, k, SFX[size], xm(), rnd.choice(regs)))
            out.append('cvtsi2%s %s, %s' % (k, _mem(rnd), rnd.choice(XMM)))
    for mn in ('roundss', 'roundsd', 'roundps', 'roundpd'):
        for _ in range(6):
            out.append('%s $%d, %s, %s' % (mn, rnd.randrange(16), xm(), rnd.choice(XMM)))
    return out


def sweep_relax(seed):
    """Random code with jumps over random distances, labels, and alignments."""
    rnd = random.Random(seed)
    nlab = 40
    out = ['.text', '.globl g0', 'g0:']
    insns = ['ret', 'nop', 'movq %rax, %rcx', 'addq $1000, %rax', 'movabsq $1234567890123, %r9',
             'callq ext', 'leaq .L5(%rip), %rax', 'leaq g0(%rip), %rax', 'movl $1, %eax']
    defined = set()
    for i in range(600):
        r = rnd.random()
        if r < 0.25:
            lab = rnd.randrange(nlab)
            cc = rnd.choice(['mp', 'z', 'nz', 'l', 'ge', 'b', 'a'])
            out.append('j%s .L%d' % (cc, lab))
        elif r < 0.30:
            lab = rnd.randrange(nlab)
            if lab not in defined:
                defined.add(lab)
                out.append('.L%d:' % lab)
        elif r < 0.33:
            out.append('.p2align %d' % rnd.randrange(1, 6))
        elif r < 0.34:
            out.append('.byte 1')
        elif r < 0.36:
            out.append('.fill %d,1,0x90' % rnd.randrange(1, 140))
        elif r < 0.37:
            out.append('jmp g0')
        else:
            out.append(rnd.choice(insns))
    for lab in range(nlab):
        if lab not in defined:
            out.append('.L%d:' % lab)
            out.append('nop')
    out += ['.section .text.b,"ax",@progbits', 'jmp .L3', 'jz g0', 'callq .L4', 'ret']
    return out


def recorded_inputs():
    with open(C.FORMS) as f:
        forms = f.read()
    tail = ['.Lloc:', 'nop', '.data', 'sym: .quad 0']
    ins = {'forms': forms,
           'sweep_integer': '\n'.join(['.text'] + sweep_integer() + tail) + '\n',
           'sweep_sse': '\n'.join(['.text'] + sweep_sse() + tail) + '\n'}
    for seed in range(4):
        ins['relax%d' % seed] = '\n'.join(sweep_relax(seed)) + '\n'
    return ins


_INPUTS = recorded_inputs()


@pytest.mark.parametrize('name', sorted(_INPUTS))
def test_recorded(name):
    """Against GNU as when available, else against its recorded output."""
    try:
        d = C.compare_recorded(name, _INPUTS[name])
    except LookupError as e:
        pytest.skip(str(e))
    assert not d, '\n'.join(d)


# ---------------------------------------------------------------------------------------------
# behaviour without GNU as


def test_relocation_choices():
    """Which references GNU as resolves, and with what relocation (ELF x86-64, not -shared)."""
    o = assembler_amd64.assemble(
        '.globl g\ng:\n jmp g\n jmp l\n call g\n call l\n jmp ext\n leaq g(%rip), %rax\n'
        ' leaq l(%rip), %rax\n call ext+4\n movl $5, x(%rip)\n movq $g, %rax\n movl $g, %eax\n'
        'l:\n ret\n.data\nx: .quad l\n')
    t = o.section('.text')
    assert t.data[:3] == bytes.fromhex('ebfeeb')
    rel = [(r.offset, r.type, r.sym, r.addend) for r in t.relocs]
    assert rel == [(5, 'PLT32', 'g', -4), (15, 'PLT32', 'ext', -4), (22, 'PC32', 'g', -4),
                   (34, 'PC32', 'ext', 0), (40, 'PC32', 'x', -8), (51, 'ABS32S', 'g', 0),
                   (56, 'ABS32', 'g', 0)]
    assert [(r.type, r.sym) for r in o.section('.data').relocs] == [('ABS64', 'l')]


def test_nop_padding():
    o = assembler_amd64.assemble('ret\n.p2align 4\n.byte 1\n.p2align 3\n')
    assert bytes(o.section('.text').data) == bytes.fromhex(
        'c3' '66662e0f1f840000000000' '0f1f4000' '01' '90' '660f1f440000')


@pytest.mark.parametrize('src', [
    'movq %fs:x@tpoff, %rax',
    'leaq x@tpoff(%rax), %rax',
    'addq x@gottpoff(%rip), %rax',
    '.section .tdata,"awT"',
    '.section .tbss,"awT"',
    '.abort "unreachable"',
    'addq $0x80000000, %rax',
    'movq $1, %eax',
    'movl %eax, %rcx',
    'addl %eax, %cx',
    'movl (%eax), %eax',
    'movq (%rax,%rsp), %rax',
    'movq x(%rip,%rax), %rax',
    'movq x@plt(%rip), %rax',
    'movb %ah, %sil',
    'leaq %rax, %rcx',
    'add $1, (%rax)',
    'shll $sym, %eax',
    'pushl %eax',
    'call *%eax',
    'cvttsd2si %xmm0, %ax',
    'movd %xmm0, %ax',
    'frobnicate %rax',
    '.weird 1',
    '.quad a-b',
    'movq (%rax,%rcx,3), %rdx',
    'lock movq %rax, %rcx',
])
def test_rejects(src):
    with pytest.raises(assembler_amd64.AssemblerError):
        assembler_amd64.assemble(src + '\n')


def test_error_line_number():
    with pytest.raises(assembler_amd64.AssemblerError, match='line 3'):
        assembler_amd64.assemble('ret\nret\nbogus %rax\n')


def test_local_labels():
    """QBE's amd64_win local labels have no leading dot."""
    o = assembler_amd64.assemble('f:\nLbb1:\n jmp Lbb1\n jnz Lbb2\nLbb2:\n ret\n')
    assert bytes(o.section('.text').data) == bytes.fromhex('ebfe7500c3')
    assert not o.section('.text').relocs


# ---------------------------------------------------------------------------------------------
# execution smoke test: run assembled code on this host

SUM = """
.text
.balign 16
.globl sum
sum:
	endbr64
	xorl %eax, %eax
	testq %{n}, %{n}
	jle .Ldone
.Lloop:
	addq (%{p}), %rax
	addq $8, %{p}
	subq $1, %{n}
	jnz .Lloop
.Ldone:
	ret
"""


def _executable(code):
    """Copy code into executable memory; returns (address, keepalive)."""
    if sys.platform == 'win32':
        k32 = ctypes.windll.kernel32
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong,
                                     ctypes.c_ulong]
        k32.VirtualProtect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong,
                                       ctypes.POINTER(ctypes.c_ulong)]
        addr = k32.VirtualAlloc(None, len(code), 0x3000, 0x04)  # MEM_COMMIT|RESERVE, RW
        assert addr
        ctypes.memmove(addr, code, len(code))
        old = ctypes.c_ulong()
        assert k32.VirtualProtect(addr, len(code), 0x20, ctypes.byref(old))  # EXECUTE_READ
        return addr, None
    import mmap
    m = mmap.mmap(-1, len(code), prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC)
    m.write(code)
    return ctypes.addressof(ctypes.c_char.from_buffer(m)), m


@pytest.mark.skipif(platform.machine().lower() not in ('amd64', 'x86_64'),
                    reason='needs an x86-64 host')
def test_execute_sum():
    regs = {'p': 'rcx', 'n': 'rdx'} if sys.platform == 'win32' else {'p': 'rdi', 'n': 'rsi'}
    o = assembler_amd64.assemble(SUM.format(**regs))
    t = o.section('.text')
    assert not t.relocs
    try:
        addr, keep = _executable(bytes(t.data))
    except (OSError, ValueError, PermissionError) as e:
        pytest.skip('cannot map executable memory: %s' % e)
    f = ctypes.CFUNCTYPE(ctypes.c_longlong, ctypes.c_void_p, ctypes.c_longlong)(
        addr + o.symbols['sum'].value)
    vals = [3, -7, 1 << 40, 12345, 0, 99]
    arr = (ctypes.c_longlong * len(vals))(*vals)
    assert f(ctypes.addressof(arr), len(vals)) == sum(vals)
    assert f(ctypes.addressof(arr), 0) == 0
    del keep
