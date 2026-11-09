"""x86-64 assembler for the AT&T-syntax assembly that `cimo.backend` emits for amd64.

    obj = assemble(text)        # targets amd64_sysv and amd64_win (ELF-style syntax)

The result is the same in-memory `Object` that cimo.assembler (arm64) produces: sections with
their bytes, symbols and relocations, which `cimo.loader` links into the running process. It
accepts the instructions, operand forms and directives QBE's amd64 emitter produces (amd64/emit.c,
the shared data emitter python/cimo/backend/emit.py), plus a few more (see _INSNS), and raises
`AssemblerError` for anything else.

Everything follows GNU as (x86-64 ELF, no -O): instruction forms (sign-extended imm8 and
accumulator short forms, shift-by-1, movabs for 64-bit immediates, 32-bit immediates truncated),
operand encodings (disp8/disp32 choice, SIB rules), jump relaxation (jmp/jcc to labels of the same
section use rel8 when it reaches, iterated to a fixpoint), NOP padding of alignment in code
sections, and which references are resolved at assembly time versus left as relocations.
python/tests/test_assembler_amd64.py checks the result byte for byte against GNU as.

Relocations (named after the ELF x86-64 ones; offset = the 4/8-byte field, addend as in RELA):
  PC32      S + A - P, signed 32 (rip-relative operands, calls/jumps to local symbols of another
            section, calls/jumps with an addend); for a rip-relative operand the addend is
            sym_offset - (end of instruction - field offset), e.g. -4, or -8 for `movl $5, x(%rip)`
  PLT32     same math as PC32: calls and jumps to undefined or global symbols, `sym@plt`
  GOTPCREL  G + GOT + A - P: `sym@gotpcrel(%rip)` (GNU as writes REX_GOTPCRELX for it)
  ABS64     S + A, 64 bits (.quad sym, movabsq $sym)
  ABS32     S + A, zero-extended 32 bits (.int sym, $sym in 32-bit operations)
  ABS32S    S + A, sign-extended 32 bits ($sym in 64-bit operations, absolute disp32 `sym(%rax)`)
  ABS16, ABS8   (.short sym, .byte sym)
Thread-local storage is rejected.
"""

import re
import struct

from . import assembler as _arm
from .assembler import AssemblerError, Object, Section, Symbol, Reloc

__all__ = ['assemble', 'AssemblerError', 'Object', 'Section', 'Symbol', 'Reloc']

M64 = (1 << 64) - 1

# ---------------------------------------------------------------------------------------------
# operands
#
# register: ('r', size, number, rexflag)  rexflag 1: needs a REX prefix (spl bpl sil dil),
#                                          2: can't have one (ah ch dh bh)
# xmm:      ('x', 16, number, 0)
# ymm:      ('y', 32, number, 0)         (AVX: VEX-encoded instructions; 16 to 31 and zmm EVEX only)
# zmm:      ('z', 64, number, 0)
# mask:     ('k', 8, number, 0)          (AVX-512's k0 to k7)
# immediate: ('i', sym, value, modifier)
# memory:   ('m', seg, sym, disp, modifier, base, index, scale)   base: None, -1 (%rip) or 0..15
# indirect: ('*', operand)

_REGS = {}
_N64 = ['rax', 'rcx', 'rdx', 'rbx', 'rsp', 'rbp', 'rsi', 'rdi']
_N32 = ['eax', 'ecx', 'edx', 'ebx', 'esp', 'ebp', 'esi', 'edi']
_N16 = ['ax', 'cx', 'dx', 'bx', 'sp', 'bp', 'si', 'di']
_N8 = ['al', 'cl', 'dl', 'bl', 'spl', 'bpl', 'sil', 'dil']
for _i in range(16):
    if _i < 8:
        _REGS['%' + _N64[_i]] = ('r', 8, _i, 0)
        _REGS['%' + _N32[_i]] = ('r', 4, _i, 0)
        _REGS['%' + _N16[_i]] = ('r', 2, _i, 0)
        _REGS['%' + _N8[_i]] = ('r', 1, _i, 1 if _i >= 4 else 0)
    else:
        _REGS['%%r%d' % _i] = ('r', 8, _i, 0)
        _REGS['%%r%dd' % _i] = ('r', 4, _i, 0)
        _REGS['%%r%dw' % _i] = ('r', 2, _i, 0)
        _REGS['%%r%db' % _i] = ('r', 1, _i, 0)
        _REGS['%%r%dl' % _i] = ('r', 1, _i, 0)
    _REGS['%%xmm%d' % _i] = ('x', 16, _i, 0)
    _REGS['%%ymm%d' % _i] = ('y', 32, _i, 0)
for _i in range(32):
    if _i >= 16:
        _REGS['%%xmm%d' % _i] = ('x', 16, _i, 0)
        _REGS['%%ymm%d' % _i] = ('y', 32, _i, 0)
    _REGS['%%zmm%d' % _i] = ('z', 64, _i, 0)
for _i in range(8):
    _REGS['%%k%d' % _i] = ('k', 8, _i, 0)
for _i in range(8):
    _REGS['%%tmm%d' % _i] = ('t', 1024, _i, 0)  # AMX tiles
for _i, _n in enumerate(['ah', 'ch', 'dh', 'bh']):
    _REGS['%' + _n] = ('r', 1, 4 + _i, 2)
_RIP = -1
_SEGS = {'es': 0x26, 'cs': 0x2e, 'ss': 0x36, 'ds': 0x3e, 'fs': 0x64, 'gs': 0x65}
_CL = _REGS['%cl']

_TLS_MODS = ('tpoff', 'gottpoff', 'tlsgd', 'tlsld', 'dtpoff', 'dtpmod', 'tlvp', 'ntpoff',
             'indntpoff', 'gotntpoff', 'tlsdesc', 'tlscall')
_MODS = ('plt', 'gotpcrel')

_TOK = re.compile(r'\s*(?:(0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+)|([A-Za-z_.$][\w.$]*)'
                  r'|("(?:[^"\\]|\\.)*")|(<<|>>|[-+*/%&|^~!()@]))')
_SYMCH = re.compile(r'[A-Za-z_.$][\w.$]*$')


def _num(s):
    if s[:2] in ('0x', '0X'):
        return int(s[2:], 16)
    if s[:2] in ('0b', '0B'):
        return int(s[2:], 2)
    if len(s) > 1 and s[0] == '0':
        try:
            return int(s, 8)
        except ValueError:
            raise AssemblerError('bad number %r' % s) from None
    return int(s)


def _s64(v):
    v &= M64
    return v - (1 << 64) if v >> 63 else v


def _tls(mod):
    if mod in _TLS_MODS:
        raise AssemblerError('thread-local storage is not supported')


def _eval(s):
    """GNU as expression -> (symbol or None, value, modifier or None). The value is a signed
    64-bit integer; at most one symbol, with coefficient +1."""
    s = s.strip()
    if s.isdigit() and (s[0] != '0' or len(s) == 1):
        return None, _s64(int(s)), None
    if s[:1] == '-' and s[1:].isdigit() and (s[1] != '0' or len(s) == 2):
        return None, _s64(-int(s[1:])), None
    if _SYMCH.match(s) and s != '.':
        return s, 0, None
    toks = []
    i = 0
    n = len(s)
    while i < n:
        m = _TOK.match(s, i)
        if not m or m.end() == i:
            if s[i:].strip() == '':
                break
            raise AssemblerError('bad expression %r' % s)
        i = m.end()
        if m.group(1):
            toks.append(('n', _num(m.group(1))))
        elif m.group(2):
            toks.append(('s', m.group(2)))
        elif m.group(3):
            toks.append(('s', _arm._parse_sym(m.group(3), 0)[0]))
        else:
            toks.append(('o', m.group(4)))
    pos = [0]
    mods = [None]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else (None, None)

    def primary():
        t, v = peek()
        pos[0] += 1
        if t == 'n':
            return None, v
        if t == 's':
            if v == '.':
                raise AssemblerError('"." is not supported in expressions')
            t2, v2 = peek()
            if t2 == 'o' and v2 == '@':
                pos[0] += 1
                t3, v3 = peek()
                if t3 != 's':
                    raise AssemblerError('bad relocation modifier in %r' % s)
                pos[0] += 1
                mod = v3.lower()
                _tls(mod)
                if mods[0] is not None or mod not in _MODS:
                    raise AssemblerError('unsupported relocation modifier @%s' % v3)
                mods[0] = mod
            return v, 0
        if t == 'o':
            if v == '(':
                r = binary(1)
                t2, v2 = peek()
                if (t2, v2) != ('o', ')'):
                    raise AssemblerError('missing ) in %r' % s)
                pos[0] += 1
                return r
            if v in '-~!+':
                sym, x = primary()
                if v == '+':
                    return sym, x
                if sym is not None:
                    raise AssemblerError('unsupported expression %r' % s)
                if v == '-':
                    return None, -x & M64
                if v == '~':
                    return None, ~x & M64
                return None, int(x & M64 == 0)
        raise AssemblerError('bad expression %r' % s)

    prec = {'+': 1, '-': 1, '|': 2, '&': 2, '^': 2, '!': 2,
            '*': 3, '/': 3, '%': 3, '<<': 3, '>>': 3}

    def binary(level):
        lhs = primary() if level > 3 else binary(level + 1)
        while True:
            t, op = peek()
            if t != 'o' or prec.get(op) != level:
                return lhs
            pos[0] += 1
            rhs = binary(level + 1)
            (ls, lv), (rs, rv) = lhs, rhs
            if op == '+':
                if ls is not None and rs is not None:
                    raise AssemblerError('unsupported expression %r' % s)
                lhs = (ls or rs, (lv + rv) & M64)
                continue
            if op == '-':
                if rs is not None:
                    if rs != ls:
                        raise AssemblerError('unsupported expression %r' % s)
                    lhs = (None, (lv - rv) & M64)
                else:
                    lhs = (ls, (lv - rv) & M64)
                continue
            if ls is not None or rs is not None:
                raise AssemblerError('unsupported expression %r' % s)
            lv &= M64
            rv &= M64
            if op == '*':
                r = lv * rv
            elif op in ('/', '%'):
                if rv == 0:
                    raise AssemblerError('division by zero')
                a, b = _s64(lv), _s64(rv)
                q = abs(a) // abs(b) * (1 if (a < 0) == (b < 0) else -1)
                r = q if op == '/' else a - q * b
            elif op == '<<':
                r = lv << rv if rv < 64 else 0
            elif op == '>>':
                r = lv >> rv if rv < 64 else 0
            elif op == '|':
                r = lv | rv
            elif op == '&':
                r = lv & rv
            elif op == '^':
                r = lv ^ rv
            else:  # '!': or not
                r = lv | (~rv & M64)
            lhs = (None, r & M64)

    if not toks:
        raise AssemblerError('empty expression')
    sym, v = binary(1)
    if pos[0] != len(toks):
        raise AssemblerError('bad expression %r' % s)
    return sym, _s64(v), mods[0]


def _split(s):
    """Split operands on commas outside parentheses and quotes."""
    if '(' not in s and '"' not in s:
        return [p.strip() for p in s.split(',')] if s else []
    out = []
    depth = 0
    q = False
    start = 0
    for i, c in enumerate(s):
        if q:
            if c == '"':
                q = False
        elif c == '"':
            q = True
        elif c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
    out.append(s[start:].strip())
    return out


def _reg(s):
    r = _REGS.get(s) or _REGS.get(s.lower())
    if r is None:
        raise AssemblerError('bad register name %r' % s)
    return r


def _operand(s):
    if not s:
        raise AssemblerError('missing operand')
    c = s[0]
    if c == '%':
        r = _REGS.get(s)
        if r is not None:
            return r
        if ':' not in s:
            return _reg(s)
    elif c == '$':
        sym, v, mod = _eval(s[1:])
        return ('i', sym, v, mod)
    elif c == '*':
        return ('*', _operand(s[1:].strip()))
    return _mem(s)


def _mem(s):
    seg = None
    if s[0] == '%':
        j = s.index(':')
        seg = s[1:j].strip().lower()
        if seg not in _SEGS:
            raise AssemblerError('bad segment register in %r' % s)
        s = s[j + 1:].strip()
    base = index = None
    scale = 0
    disp = s
    if s.endswith(')'):
        j = s.rfind('(')
        inner = s[j + 1:-1].strip()
        if inner[:1] in ('%', ',') or inner == '':
            disp = s[:j].strip()
            parts = [p.strip() for p in inner.split(',')]
            if len(parts) > 3:
                raise AssemblerError('bad memory operand %r' % s)
            if parts[0]:
                if parts[0].lower() == '%rip':
                    base = _RIP
                else:
                    r = _reg(parts[0])
                    if r[0] != 'r' or r[1] != 8:
                        raise AssemblerError('bad base register in %r' % s)
                    base = r[2]
            if len(parts) > 1:
                r = _reg(parts[1])
                if r[0] != 'r' or r[1] != 8 or r[2] == 4:
                    raise AssemblerError('bad index register in %r' % s)
                if base == _RIP:
                    raise AssemblerError('%%rip can not be used with an index in %r' % s)
                index = r[2]
                if len(parts) > 2:
                    sc = {'1': 0, '2': 1, '4': 2, '8': 3}.get(parts[2])
                    if sc is None:
                        raise AssemblerError('bad scale in %r' % s)
                    scale = sc
            elif len(parts) > 2:
                raise AssemblerError('bad memory operand %r' % s)
    if disp:
        sym, v, mod = _eval(disp)
    else:
        sym, v, mod = None, 0, None
    return ('m', seg, sym, v, mod, base, index, scale)


def _imm_wrap(v, size):
    """GNU as's view of an immediate for an operation of `size` bytes: 8/16/32-bit values
    wrap (with a warning when they don't fit), 64-bit ones must fit a sign-extended imm32."""
    if size == 8:
        if not -0x80000000 <= v <= 0x7fffffff:
            raise AssemblerError('immediate %d does not fit in a sign-extended 32-bit field' % v)
        return v
    bits = size * 8
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


_ABS = {1: 'ABS8', 2: 'ABS16', 4: 'ABS32', 8: 'ABS32S'}


# ---------------------------------------------------------------------------------------------
# encoding


def _enc(a, opc, reg, rm, size=0, mand=b'', imm=None, w=None, dabs='ABS32S', vex=None, evex=None):
    """Emit one instruction: legacy prefixes, REX, opcode `opc` (bytes), ModRM with `reg` (an
    int /digit or a register operand) and `rm` (a register or memory operand), SIB, displacement,
    and immediate `imm` = (nbytes, sym, value, reloc kind) or None. `size` 2 adds the operand-size
    prefix, 8 sets REX.W (unless `w` says otherwise); `mand` is a mandatory SSE prefix; `dabs` is
    the relocation for an absolute symbolic displacement. vex = (pp, map, W, L, vvvv): a VEX prefix in place
    of the mandatory prefix and REX, `opc` then the one opcode byte of that map. evex = (pp, map, W, L'L, vvvv,
    aaa, z, b, N): an EVEX prefix (AVX-512; registers 16 to 31, mask aaa, zeroing, broadcast), a displacement that
    is a multiple of N in a scaled disp8 (disp8*N)."""
    out = bytearray()
    rex = 8 if (size == 8 if w is None else w) else 0
    force = 0
    if reg.__class__ is tuple:
        force = reg[3]
        reg = reg[2]
    regn = reg
    if evex is None and (reg > 15 or rm[0] != 'm' and rm[2] > 15 or vex is not None and vex[4] > 15):
        raise AssemblerError('registers 16 to 31 need an EVEX instruction')
    n8 = evex[8] if evex is not None else 1
    if reg & 8:
        rex |= 4
    reg = (reg & 7) << 3
    fix = None
    dsz = 0
    sib = -1
    if rm[0] == 'm':
        _, seg, sym, disp, mod, base, index, scale = rm
        if seg:
            out.append(_SEGS[seg])
        if mod is not None and mod != 'gotpcrel':
            raise AssemblerError('unsupported relocation modifier @%s' % mod)
        if base == _RIP:
            modrm = 0x05 | reg
            dsz = 4
            if sym is not None:
                fix = ('GOTPCREL' if mod else 'PC32', sym, disp, True)
            elif mod:
                raise AssemblerError('@gotpcrel needs a symbol')
        else:
            if mod:
                raise AssemblerError('@gotpcrel needs a %rip base')
            if base is None:
                modrm = 0x04 | reg
                sib = (scale << 6 | (index & 7) << 3 | 5) if index is not None else 0x25
                if index is not None and index & 8:
                    rex |= 2
                dsz = 4
            else:
                if sym is not None:
                    dsz = 4
                elif disp == 0 and base & 7 != 5:
                    dsz = 0
                elif disp % n8 == 0 and -128 <= disp // n8 <= 127:
                    dsz = 1
                else:
                    dsz = 4
                md = (0, 0x40, 0, 0, 0x80)[dsz]
                if base & 8:
                    rex |= 1
                if index is not None:
                    if index & 8:
                        rex |= 2
                    modrm = md | reg | 4
                    sib = scale << 6 | (index & 7) << 3 | (base & 7)
                elif base & 7 == 4:
                    modrm = md | reg | 4
                    sib = 0x24
                else:
                    modrm = md | reg | (base & 7)
            if sym is not None:
                fix = (dabs, sym, disp, False)
        if dsz == 4 and fix is None and not -0x80000000 <= disp <= 0x7fffffff:
            # a disp32 is sign-extended: 0x80000000 to 0xffffffff would address 0xffffffff80000000 and up
            raise AssemblerError('displacement %d out of range' % disp)
    else:
        force |= rm[3]
        n = rm[2]
        if n & 8:
            rex |= 1
        if n & 16:  # EVEX: X extends a register operand
            rex |= 2
        modrm = 0xc0 | reg | (n & 7)
    if evex is not None:
        if a.lock:
            raise AssemblerError('lock prefix on an EVEX instruction')
        pp, mm, W, LL, vvvv, aaa, z, bc, _ = evex
        R, X, B = (rex & 4) == 0, (rex & 2) == 0, (rex & 1) == 0
        out += bytes((0x62, R << 7 | X << 6 | B << 5 | ((regn & 16) == 0) << 4 | mm,
                      W << 7 | (~vvvv & 15) << 3 | 4 | pp,
                      z << 7 | LL << 5 | bc << 4 | ((vvvv & 16) == 0) << 3 | aaa))
        out += opc
    elif vex is not None:
        if a.lock:
            raise AssemblerError('lock prefix on a VEX instruction')
        pp, mm, W, L, vvvv = vex
        R, X, B = (rex & 4) == 0, (rex & 2) == 0, (rex & 1) == 0
        if mm == 1 and X and B and not W:  # the two-byte form
            out += bytes((0xc5, R << 7 | (~vvvv & 15) << 3 | L << 2 | pp))
        else:
            out += bytes((0xc4, R << 7 | X << 6 | B << 5 | mm, W << 7 | (~vvvv & 15) << 3 | L << 2 | pp))
        out += opc
    else:
        if size == 2:
            out.append(0x66)
        if mand:
            out += mand
        if a.lock:
            out.append(0xf0)
        if rex or force & 1:
            if force & 2:
                raise AssemblerError("can't encode %ah/%bh/%ch/%dh in an instruction requiring REX")
            out.append(0x40 | rex)
        out += opc
    out.append(modrm)
    if sib >= 0:
        out.append(sib)
    doff = len(out)
    if dsz == 1:
        out.append((disp // n8) & 0xff)
    elif dsz == 4:
        out += (0 if fix else disp & 0xffffffff).to_bytes(4, 'little')
    fixups = []
    if imm is not None:
        nb, isym, iv, ikind = imm
        ioff = len(out)
        if isym is not None:
            out += bytes(nb)
            fixups.append((ioff, ikind, isym, iv))
        else:
            out += (iv & ((1 << (8 * nb)) - 1)).to_bytes(nb, 'little')
    if fix is not None:
        kind, sym, add, pcrel = fix
        if pcrel:
            add -= len(out) - doff
        fixups.append((doff, kind, sym, add))
    a.put(out, fixups)


def _enc_o(a, op, r, size, imm=None):
    """Opcode-plus-register forms (push, pop, bswap, xchg with the accumulator, mov imm)."""
    out = bytearray()
    if size == 2:
        out.append(0x66)
    rex = 8 if size == 8 else 0
    if r[2] & 8:
        rex |= 1
    if rex or r[3] & 1:
        if r[3] & 2:
            raise AssemblerError("can't encode %ah/%bh/%ch/%dh in an instruction requiring REX")
        out.append(0x40 | rex)
    out += op[:-1]
    out.append(op[-1] | (r[2] & 7))
    fixups = []
    if imm is not None:
        nb, isym, iv, ikind = imm
        if isym is not None:
            fixups.append((len(out), ikind, isym, iv))
            out += bytes(nb)
        else:
            out += (iv & ((1 << (8 * nb)) - 1)).to_bytes(nb, 'little')
    a.put(out, fixups)


def _size(sz, ops):
    """Operand size from the mnemonic suffix and the general registers among `ops`."""
    for o in ops:
        if o[0] == 'r':
            if sz == 0:
                sz = o[1]
            elif o[1] != sz:
                raise AssemblerError('operand size mismatch')
    if sz == 0:
        raise AssemblerError('no instruction mnemonic suffix given and no register operands')
    return sz


def _nops(n, need):
    if len(n) != need:
        raise AssemblerError('expected %d operand%s' % (need, 's' if need != 1 else ''))


def _rm(o):
    if o[0] not in ('r', 'm'):
        raise AssemblerError('expected a register or memory operand')
    return o


def _gpr(o, sizes=(1, 2, 4, 8)):
    if o[0] != 'r' or o[1] not in sizes:
        raise AssemblerError('expected a general register')
    return o


def _xmm(o, mem=False):
    if o[0] != 'x' and not (mem and o[0] == 'm'):
        raise AssemblerError('expected an %%xmm register%s' % (' or memory' if mem else ''))
    if o[0] == 'x' and o[2] > 15:
        raise AssemblerError('registers 16 to 31 need an EVEX instruction')
    return o


def _immop(o, size, kind=None):
    """Immediate operand -> (sym, value) wrapped to `size`."""
    if o[0] != 'i':
        raise AssemblerError('expected an immediate')
    if o[3] is not None:
        _tls(o[3])
        raise AssemblerError('relocation modifier not allowed in an immediate')
    if o[1] is not None:
        return o[1], o[2]
    return None, _imm_wrap(o[2], size)


# ---------------------------------------------------------------------------------------------
# instruction handlers: h(asm, operands, suffix size or 0, table argument)


def _h_alu(a, ops, sz, n):
    _nops(ops, 2)
    src, dst = ops
    if src[0] == 'i':
        _rm(dst)
        size = _size(sz, (dst,))
        sym, v = _immop(src, size)
        acc = dst[0] == 'r' and dst[2] == 0
        if size == 1:
            if acc:
                return _enc_imm_acc(a, bytes([n << 3 | 4]), size, (1, sym, v, 'ABS8'))
            return _enc(a, b'\x80', n, dst, size, imm=(1, sym, v, 'ABS8'))
        if sym is None and -128 <= v <= 127:
            return _enc(a, b'\x83', n, dst, size, imm=(1, None, v, None))
        isz = 2 if size == 2 else 4
        if acc:
            return _enc_imm_acc(a, bytes([n << 3 | 5]), size, (isz, sym, v, _ABS[size]))
        return _enc(a, b'\x81', n, dst, size, imm=(isz, sym, v, _ABS[size]))
    size = _size(sz, ops)
    w = size != 1
    if src[0] == 'r' and dst[0] in ('r', 'm'):
        return _enc(a, bytes([n << 3 | w]), src, dst, size)
    if src[0] == 'm' and dst[0] == 'r':
        return _enc(a, bytes([n << 3 | 2 | w]), dst, src, size)
    raise AssemblerError('bad operands')


def _enc_imm_acc(a, op, size, imm):
    out = bytearray()
    if size == 2:
        out.append(0x66)
    if a.lock:
        out.append(0xf0)
    if size == 8:
        out.append(0x48)
    out += op
    nb, sym, v, kind = imm
    fixups = []
    if sym is not None:
        fixups.append((len(out), kind, sym, v))
        out += bytes(nb)
    else:
        out += (v & ((1 << (8 * nb)) - 1)).to_bytes(nb, 'little')
    a.put(out, fixups)


def _h_test(a, ops, sz, _):
    _nops(ops, 2)
    src, dst = ops
    if src[0] == 'i':
        _rm(dst)
        size = _size(sz, (dst,))
        sym, v = _immop(src, size)
        isz = min(size, 4)
        if dst[0] == 'r' and dst[2] == 0:
            return _enc_imm_acc(a, b'\xa8' if size == 1 else b'\xa9', size,
                                (isz, sym, v, _ABS[size]))
        return _enc(a, b'\xf6' if size == 1 else b'\xf7', 0, dst, size,
                    imm=(isz, sym, v, _ABS[size]))
    size = _size(sz, ops)
    op = b'\x84' if size == 1 else b'\x85'
    if src[0] == 'r' and dst[0] in ('r', 'm'):
        return _enc(a, op, src, dst, size)
    if src[0] == 'm' and dst[0] == 'r':
        return _enc(a, op, dst, src, size)
    raise AssemblerError('bad operands')


def _h_mov(a, ops, sz, mn):
    _nops(ops, 2)
    src, dst = ops
    if src[0] == 'x' or dst[0] == 'x':
        if mn == 'movq':
            return _h_movq(a, ops, 8, 'movq')
        raise AssemblerError('bad operands')
    if src[0] == 'i':
        _rm(dst)
        size = _size(sz, (dst,))
        if src[3] is not None:
            _tls(src[3])
            raise AssemblerError('relocation modifier not allowed in an immediate')
        sym, v = src[1], src[2]
        if dst[0] == 'r':
            if size == 8:
                if sym is None and not -0x80000000 <= v <= 0x7fffffff:
                    return _enc_o(a, b'\xb8', dst, 8, imm=(8, None, v, None))
                return _enc(a, b'\xc7', 0, dst, 8, imm=(4, sym, v, 'ABS32S'))
            if sym is None:
                v = _imm_wrap(v, size)
            return _enc_o(a, b'\xb0' if size == 1 else b'\xb8', dst, size,
                          imm=(size, sym, v, _ABS[size]))
        sym, v = _immop(src, size)
        return _enc(a, b'\xc6' if size == 1 else b'\xc7', 0, dst, size,
                    imm=(min(size, 4), sym, v, _ABS[size]))
    size = _size(sz, ops)
    w = size != 1
    if src[0] == 'r' and dst[0] in ('r', 'm'):
        return _enc(a, bytes([0x88 | w]), src, dst, size)
    if src[0] == 'm' and dst[0] == 'r':
        return _enc(a, bytes([0x8a | w]), dst, src, size)
    raise AssemblerError('bad operands')


def _h_movabs(a, ops, sz, _):
    _nops(ops, 2)
    src, dst = ops
    if src[0] != 'i' or dst[0] != 'r' or dst[1] != 8 or sz not in (0, 8):
        raise AssemblerError('only movabsq $imm64, %reg64 is supported')
    if src[3] is not None:
        _tls(src[3])
        raise AssemblerError('relocation modifier not allowed in an immediate')
    _enc_o(a, b'\xb8', dst, 8, imm=(8, src[1], src[2], 'ABS64'))


def _h_unary(a, ops, sz, n):
    """neg not mul div idiv (F6/F7 /n), inc dec (FE/FF /n): n = (digit, inc/dec)."""
    _nops(ops, 1)
    o = _rm(ops[0])
    size = _size(sz, ops)
    digit, incdec = n
    if incdec:
        op = b'\xfe' if size == 1 else b'\xff'
    else:
        op = b'\xf6' if size == 1 else b'\xf7'
    _enc(a, op, digit, o, size)


def _h_imul(a, ops, sz, _):
    if len(ops) == 1:
        return _h_unary(a, ops, sz, (5, False))
    if len(ops) == 2:
        src, dst = ops
        if src[0] == 'i':
            ops = [src, dst, dst]
        else:
            _gpr(dst, (2, 4, 8))
            size = _size(sz, ops)
            return _enc(a, b'\x0f\xaf', dst, _rm(src), size)
    if len(ops) != 3:
        raise AssemblerError('imul takes 1 to 3 operands')
    imm, src, dst = ops
    _gpr(dst, (2, 4, 8))
    _rm(src)
    size = _size(sz, ops[1:])
    sym, v = _immop(imm, size)
    if sym is None and -128 <= v <= 127:
        return _enc(a, b'\x6b', dst, src, size, imm=(1, None, v, None))
    _enc(a, b'\x69', dst, src, size, imm=(2 if size == 2 else 4, sym, v, _ABS[size]))


def _h_shift(a, ops, sz, n):
    if len(ops) == 1:
        o = _rm(ops[0])
        size = _size(sz, ops)
        return _enc(a, b'\xd0' if size == 1 else b'\xd1', n, o, size)
    _nops(ops, 2)
    cnt, o = ops
    _rm(o)
    size = _size(sz, (o,))
    if cnt == _CL:
        return _enc(a, b'\xd2' if size == 1 else b'\xd3', n, o, size)
    sym, v = _immop(cnt, 1)
    if sym is not None:
        raise AssemblerError('shift count must be a constant')
    if v == 1:
        return _enc(a, b'\xd0' if size == 1 else b'\xd1', n, o, size)
    _enc(a, b'\xc0' if size == 1 else b'\xc1', n, o, size, imm=(1, None, v, None))


def _h_movx(a, ops, sz, arg):
    """movsbl & co: arg = (opcode, source size, destination size)."""
    _nops(ops, 2)
    op, ssz, dsz = arg
    src, dst = ops
    _gpr(dst, (dsz,))
    if src[0] == 'r':
        _gpr(src, (ssz,))
    elif src[0] != 'm':
        raise AssemblerError('bad operands')
    _enc(a, op, dst, src, dsz)


def _h_movsx(a, ops, sz, arg):
    """Unsuffixed movsx/movzx/movsxd: sizes from the registers."""
    _nops(ops, 2)
    src, dst = ops
    _gpr(dst, (2, 4, 8))
    if src[0] != 'r':
        raise AssemblerError('%s needs a register source (sizes are ambiguous)' % arg)
    ssz = src[1]
    if arg == 'movsxd':
        if ssz != 4:
            raise AssemblerError('bad operands')
        return _enc(a, b'\x63', dst, src, dst[1])
    if ssz not in (1, 2) or ssz >= dst[1]:
        raise AssemblerError('bad operands')
    base = 0xbe if arg == 'movsx' else 0xb6
    _enc(a, bytes([0x0f, base | (ssz == 2)]), dst, src, dst[1])


def _h_lea(a, ops, sz, _):
    _nops(ops, 2)
    src, dst = ops
    if src[0] != 'm':
        raise AssemblerError('lea needs a memory operand')
    _gpr(dst, (2, 4, 8))
    size = _size(sz, (dst,))
    # GNU as (want_disp32): a lea with a 32/16-bit result takes a zero-extended displacement
    _enc(a, b'\x8d', dst, src, size, dabs='ABS32S' if size == 8 else 'ABS32')


def _h_xchg(a, ops, sz, _):
    _nops(ops, 2)
    x, y = ops
    size = _size(sz, ops)
    if x[0] == 'r' and y[0] == 'r':
        if size == 8 and x[2] == y[2] == 0:  # GNU as: a plain nop
            return a.put(b'\x90', ())
        if size != 1 and (x[2] == 0 or y[2] == 0) and not (size == 4 and x[2] == y[2] == 0):
            return _enc_o(a, b'\x90', y if x[2] == 0 else x, size)
        return _enc(a, b'\x86' if size == 1 else b'\x87', x, y, size)
    if x[0] == 'r' and y[0] == 'm':
        return _enc(a, b'\x86' if size == 1 else b'\x87', x, y, size)
    if x[0] == 'm' and y[0] == 'r':
        return _enc(a, b'\x86' if size == 1 else b'\x87', y, x, size)
    raise AssemblerError('bad operands')


def _h_push(a, ops, sz, pop):
    _nops(ops, 1)
    o = ops[0]
    if sz not in (0, 8, 2):
        raise AssemblerError('only 64-bit (or 16-bit) push/pop exist in 64-bit mode')
    if o[0] == 'r':
        if o[1] not in (2, 8) or (sz and sz != o[1]):
            raise AssemblerError('bad register for push/pop')
        return _enc_o(a, b'\x58' if pop else b'\x50', o, 2 if o[1] == 2 else 4)
    if o[0] == 'm':
        return _enc(a, b'\x8f' if pop else b'\xff', 0 if pop else 6, o, 2 if sz == 2 else 0)
    if o[0] == 'i' and not pop:
        size = sz or 8
        sym, v = _immop(o, size)
        out = bytearray(b'\x66' if size == 2 else b'')
        if sym is None and -128 <= v <= 127:
            out += bytes([0x6a, v & 0xff])
            return a.put(out, ())
        nb = 2 if size == 2 else 4
        out.append(0x68)
        if sym is not None:
            fix = [(len(out), 'ABS16' if size == 2 else 'ABS32S', sym, v)]
            out += bytes(nb)
        else:
            fix = ()
            out += (v & ((1 << (8 * nb)) - 1)).to_bytes(nb, 'little')
        return a.put(out, fix)
    raise AssemblerError('bad operand')


def _target(o):
    """A branch target operand (bare expression) -> (sym, addend, modifier)."""
    if o[0] != 'm' or o[1] or o[5] is not None or o[6] is not None or o[2] is None:
        raise AssemblerError('bad branch target')
    return o[2], o[3], o[4]


def _h_call(a, ops, sz, jmp):
    _nops(ops, 1)
    if sz not in (0, 8):
        raise AssemblerError('bad suffix')
    o = ops[0]
    if o[0] == '*' or o[0] == 'r':
        if o[0] == '*':
            o = o[1]
        if o[0] == 'r' and o[1] != 8:
            raise AssemblerError('indirect branch needs a 64-bit register')
        return _enc(a, b'\xff', 4 if jmp else 2, _rm(o), 0)
    sym, add, mod = _target(o)
    if jmp and mod is None:
        return a.branch(-1, sym, add)
    if mod not in (None, 'plt'):
        raise AssemblerError('bad branch target')
    a.put(b'\xe9\0\0\0\0' if jmp else b'\xe8\0\0\0\0',
          [(1, 'PLT32' if mod else 'BRANCH', sym, add - 4)])


def _h_jcc(a, ops, sz, cc):
    _nops(ops, 1)
    sym, add, mod = _target(ops[0])
    if mod is None:
        return a.branch(cc, sym, add)
    if mod != 'plt':
        raise AssemblerError('bad branch target')
    a.put(bytes([0x0f, 0x80 | cc, 0, 0, 0, 0]), [(2, 'PLT32', sym, add - 4)])


def _h_setcc(a, ops, sz, cc):
    _nops(ops, 1)
    o = _rm(ops[0])
    if o[0] == 'r' and o[1] != 1:
        raise AssemblerError('setcc needs a byte register')
    _enc(a, bytes([0x0f, 0x90 | cc]), 0, o, 0)


def _h_cmov(a, ops, sz, cc):
    _nops(ops, 2)
    src, dst = ops
    _gpr(dst, (2, 4, 8))
    size = _size(sz, ops)
    _enc(a, bytes([0x0f, 0x40 | cc]), dst, _rm(src), size)


def _h_fixed(a, ops, sz, code):
    if ops:
        raise AssemblerError('no operands expected')
    a.put(code, ())


def _h_mxcsr(a, ops, sz, digit):
    """stmxcsr m32 (0f ae /3), ldmxcsr m32 (/2)"""
    _nops(ops, 1)
    if ops[0][0] != 'm':
        raise AssemblerError('expected a memory operand')
    _enc(a, b'\x0f\xae', digit, ops[0])


def _h_bswap(a, ops, sz, _):
    _nops(ops, 1)
    r = _gpr(ops[0], (4, 8))
    _enc_o(a, b'\x0f\xc8', r, _size(sz, ops))


def _h_bitscan(a, ops, sz, arg):
    """bsf bsr lzcnt tzcnt popcnt: arg = (mandatory prefix, opcode)."""
    _nops(ops, 2)
    mand, op = arg
    src, dst = ops
    _gpr(dst, (2, 4, 8))
    _enc(a, op, dst, _rm(src), _size(sz, ops), mand=mand)


def _h_xadd(a, ops, sz, op):
    """cmpxchg xadd: reg -> r/m."""
    _nops(ops, 2)
    src, dst = ops
    _gpr(src)
    size = _size(sz, ops)
    _enc(a, bytes([0x0f, op | (size != 1)]), src, _rm(dst), size)


# SSE

def _h_sse(a, ops, sz, arg):
    """xmm/mem -> xmm: arg = (mandatory prefix, opcode)."""
    _nops(ops, 2)
    src, dst = ops
    _enc(a, arg[1], _xmm(dst), _xmm(src, True), 0, mand=arg[0])


def _h_ssemov(a, ops, sz, arg):
    """Loads/stores/moves: arg = (mandatory prefix, load opcode, store opcode)."""
    _nops(ops, 2)
    src, dst = ops
    if dst[0] == 'm':
        return _enc(a, arg[2], _xmm(src), dst, 0, mand=arg[0])
    _enc(a, arg[1], _xmm(dst), _xmm(src, True), 0, mand=arg[0])


def _h_movq(a, ops, sz, mn):
    """movd/movq between general registers, memory and xmm registers."""
    _nops(ops, 2)
    src, dst = ops
    if src[0] == 'x' and dst[0] == 'x':
        if mn != 'movq':
            raise AssemblerError('bad operands')
        return _enc(a, b'\x0f\x7e', dst, src, 0, mand=b'\xf3')
    if dst[0] == 'x':
        if src[0] == 'm':
            if mn == 'movq':
                return _enc(a, b'\x0f\x7e', dst, src, 0, mand=b'\xf3')
            return _enc(a, b'\x0f\x6e', dst, src, 0, mand=b'\x66')
        _gpr(src, (4, 8) if mn == 'movd' else (8,))
        return _enc(a, b'\x0f\x6e', dst, src, 0, mand=b'\x66', w=src[1] == 8)
    if src[0] == 'x':
        if dst[0] == 'm':
            if mn == 'movq':
                return _enc(a, b'\x0f\xd6', src, dst, 0, mand=b'\x66')
            return _enc(a, b'\x0f\x7e', src, dst, 0, mand=b'\x66')
        _gpr(dst, (4, 8) if mn == 'movd' else (8,))
        return _enc(a, b'\x0f\x7e', src, dst, 0, mand=b'\x66', w=dst[1] == 8)
    raise AssemblerError('bad operands')


def _h_cvti2f(a, ops, sz, mand):
    """cvtsi2ss/cvtsi2sd: general register or memory -> xmm; size from suffix or register."""
    _nops(ops, 2)
    src, dst = ops
    _xmm(dst)
    if src[0] == 'r':
        _gpr(src, (4, 8))
        size = _size(sz, (src,))
    elif src[0] == 'm':
        size = sz or 4
    else:
        raise AssemblerError('bad operands')
    _enc(a, b'\x0f\x2a', dst, src, 0, mand=mand, w=size == 8)


def _h_cvtf2i(a, ops, sz, arg):
    """cvt(t)ss2si/cvt(t)sd2si: xmm or memory -> general register."""
    _nops(ops, 2)
    src, dst = ops
    _gpr(dst, (4, 8))
    size = _size(sz, (dst,))
    _enc(a, arg[1], dst, _xmm(src, True), 0, mand=arg[0], w=size == 8)


def _h_round(a, ops, sz, op):
    _nops(ops, 3)
    imm, src, dst = ops
    sym, v = _immop(imm, 1)
    if sym is not None:
        raise AssemblerError('rounding mode must be a constant')
    _enc(a, b'\x0f\x3a' + op, _xmm(dst), _xmm(src, True), 0, mand=b'\x66',
         imm=(1, None, v, None))


def _h_sseimm(a, ops, sz, arg):
    """$imm, xmm/mem -> xmm: arg = (mandatory prefix, opcode)."""
    _nops(ops, 3)
    imm, src, dst = ops
    sym, v = _immop(imm, 1)
    if sym is not None:
        raise AssemblerError('expected a constant')
    _enc(a, arg[1], _xmm(dst), _xmm(src, True), 0, mand=arg[0], imm=(1, None, v, None))


def _h_pshift(a, ops, sz, arg):
    """$imm, xmm: arg = (opcode, /digit)."""
    _nops(ops, 2)
    imm, dst = ops
    sym, v = _immop(imm, 1)
    if sym is not None:
        raise AssemblerError('expected a constant')
    _enc(a, bytes([0x0f, arg[0]]), arg[1], _xmm(dst), 0, mand=b'\x66', imm=(1, None, v, None))


def _h_pextrw(a, ops, sz, _):
    _nops(ops, 3)
    imm, src, dst = ops
    sym, v = _immop(imm, 1)
    if sym is not None or src[0] != 'x':
        raise AssemblerError('bad operands')
    _gpr(dst, (4, 8))
    _enc(a, b'\x0f\xc5', dst, src, 0, mand=b'\x66', imm=(1, None, v, None))


def _h_vfma(a, ops, sz, arg):
    """vfmadd{132,213,231}{ss,sd,ps,pd} %xmm or %ymm (packed), src2 a register or memory: VEX.128 or .256 (FMA3)."""
    _nops(ops, 3)
    w, op = arg
    if a.mask != (0, 0) or a.pref == 'evex' or any(o[0] == 'z' or o[0] in ('x', 'y') and o[2] > 15 for o in ops):
        src2, src1, dst = _vk(ops[0], True), _vk(ops[1]), _vk(ops[2])
        L = _EVEXL[dst[0]]
        aaa, z = a.mask
        n = (16 << L) if not op & 1 else (8 if w else 4)
        return _enc(a, bytes([op]), dst, src2, evex=(1, 2, w, L, src1[2], aaa, z, 0, n))
    if ops[0][0] == 'm':  # a load folded in (vfmadd231ps (%rax), %ymm1, %ymm2): VEX 66 0F38, as the rvm forms
        src1, dst = _vr(ops[1]), _vr(ops[2])
        if src1[0] != dst[0]:
            raise AssemblerError('mixed %xmm and %ymm operands')
        if op & 1 and dst[0] == 'y':
            raise AssemblerError('scalar fused multiply-add on %ymm')
        return _enc(a, bytes([op]), dst, ops[0], vex=(1, 2, w, dst[0] == 'y', src1[2]))
    src2, src1, dst = (_vr(o) for o in ops)
    if len({o[0] for o in (src2, src1, dst)}) != 1:
        raise AssemblerError('mixed %xmm and %ymm operands')
    if op & 1 and dst[0] == 'y':  # the ss/sd forms (odd opcodes) are scalar
        raise AssemblerError('scalar fused multiply-add on %ymm')
    L = 1 if dst[0] == 'y' else 0
    r, v, b = dst[2], src1[2], src2[2]
    out = bytearray((0xc4, (((r >> 3) ^ 1) << 7) | (1 << 6) | (((b >> 3) ^ 1) << 5) | 0x02,
                     (w << 7) | ((v ^ 15) << 3) | (L << 2) | 0x01, op, 0xc0 | ((r & 7) << 3) | (b & 7)))
    a.put(out, ())


def _vr(o, mem=False):
    """An %xmm or %ymm register (or memory, if mem)."""
    if o[0] not in ('x', 'y') and not (mem and o[0] == 'm'):
        raise AssemblerError('expected an %%xmm or %%ymm register%s' % (' or memory' if mem else ''))
    if o[0] != 'm' and o[2] > 15:
        raise AssemblerError('registers 16 to 31 need an EVEX instruction')
    return o


_PP = {b'': 0, b'\x66': 1, b'\xf3': 2, b'\xf2': 3}


_MIXED = {0x18: 'bcast', 0x19: 'bcast', 0x58: 'bcast', 0x59: 'bcast', 0x78: 'bcast', 0x79: 'bcast', 0x13: 'bcast',
          0x5a: 'bcast'}  # 0x13: vcvtph2ps, 4 or 8 halves into %xmm or %ymm; 0x5a: vbroadcasti128, memory into %ymm


def _avxwidths(mn, form, ops, op):
    """xmm and ymm operands that agree, as GNU as requires: one width, but where the instruction mixes them."""
    regs = [o for o in ops if o[0] in ('x', 'y')]
    kinds = {o[0] for o in regs}
    if form in ('rmw', 'rmn'):  # widening and narrowing take mixed widths
        return
    if form == 'mri' and op == 0x1d:  # vcvtps2ph: 4 or 8 floats into an %xmm or memory
        if ops[1][0] == 'y':
            raise AssemblerError('vcvtps2ph writes an %xmm or memory')
        return
    if form in ('mri',):  # vextract*128: ymm source, xmm or memory destination
        if ops[0][0] != 'y' or ops[1][0] == 'y':
            raise AssemblerError('expected %ymm, then %xmm or memory')
        return
    if form == 'rvmi' and op in (0x18, 0x38):  # vinsert*128: xmm or memory, ymm, ymm
        if ops[0][0] == 'y' or ops[1][0] != 'y' or ops[2][0] != 'y':
            raise AssemblerError('expected %xmm or memory, then %ymm, %ymm')
        return
    if form == 'rm' and _MIXED.get(op) == 'bcast' and ops[0][0] in ('x', 'm'):
        if op == 0x19 and ops[1][0] != 'y':
            raise AssemblerError('vbroadcastsd needs a %ymm destination')
        return
    if form == 'rmi' and op == 0x00 and ops[-1][0] != 'y':  # vpermq
        raise AssemblerError('vpermq needs %ymm operands')
    if len(kinds) > 1:
        raise AssemblerError('mixed %xmm and %ymm operands')


def _h_avx(a, ops, sz, arg):
    """AVX and AVX2 (VEX): arg = (prefix, map, opcode, W, form). The length is the destination's (or for a
    store, the source's): ymm 256 bits, xmm 128. Forms, in AT&T operand order:
      rvm   src2(reg/mem), src1, dst          dst = src1 op src2
      rvmi  $imm, src2, src1, dst
      rvmr  mask, src2, src1, dst             vblendv*: the mask's register in imm8[7:4]
      rm    src(reg/mem), dst
      rmi   $imm, src, dst
      mr    src, dst(mem)                     stores
      ld    src, dst / st: the load and store opcodes as (load, store)
      vmi   $imm, src, dst                    shifts by an immediate: /digit in the opcode's place, vvvv = dst
      mri   $imm, src, dst(reg/mem)           vextract*128: rm = dst
      gm    src, dst(general register)        vpmovmskb, vmovmsk*
    With a %zmm, a mask register, a register past 15 or a {%k} decoration, the EVEX form (_h_avx512)."""
    if a.mask != (0, 0) or a.pref == 'evex' or any(o[0] in ('z', 'k') or o[0] in ('x', 'y') and o[2] > 15 for o in ops):
        return _h_avx512(a, ops, sz, arg)
    pfx, mm, op, W, form = arg
    pp = _PP[pfx]
    imm = None
    if form in ('rvmi', 'rmi', 'vmi', 'mri', 'gmi'):
        if ops[0][0] != 'i' or ops[0][1] is not None or not 0 <= ops[0][2] <= 255:
            raise AssemblerError('expected a constant from 0 to 255')
        imm = (1, None, ops[0][2], None)
        ops = ops[1:]
    _avxwidths(a.mnemonic if hasattr(a, 'mnemonic') else None, form, ops, op)
    if form in ('rvm', 'rvmi'):
        _nops(ops, 3)
        src2, src1, dst = _vr(ops[0], True), _vr(ops[1]), _vr(ops[2])
        return _enc(a, bytes([op]), dst, src2, vex=(pp, mm, W, dst[0] == 'y', src1[2]), imm=imm)
    if form == 'rvmr':
        _nops(ops, 4)
        mask, src2, src1, dst = _vr(ops[0]), _vr(ops[1], True), _vr(ops[2]), _vr(ops[3])
        return _enc(a, bytes([op]), dst, src2, vex=(pp, mm, W, dst[0] == 'y', src1[2]), imm=(1, None, mask[2] << 4, None))
    if form == 'rmw':  # widening: the destination's length
        _nops(ops, 2)
        src, dst = _vr(ops[0], True), _vr(ops[1])
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, W, dst[0] == 'y', 0))
    if form == 'rmn':  # narrowing: the source's length (vcvtpd2ps %ymm, %xmm)
        _nops(ops, 2)
        src, dst = _vr(ops[0], True), _vr(ops[1])
        if src[0] == 'm':  # the source's length from the destination's, as GNU as: a %ymm is 512 bits narrowed
            if dst[0] == 'y':
                return _h_avx512(a, ops, sz, arg)
            raise AssemblerError('ambiguous memory operand size')
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, W, src[0] == 'y', 0))
    if form in ('rm', 'rmi'):
        _nops(ops, 2)
        src, dst = _vr(ops[0], True), _vr(ops[1])
        L = dst[0] == 'y'
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, W, L, 0), imm=imm)
    if form == 'ld':
        _nops(ops, 2)
        src, dst = ops
        if dst[0] == 'm':
            _vr(src)
            return _enc(a, bytes([op[1]]), src, dst, vex=(pp, mm, W, src[0] == 'y', 0))
        _vr(dst)
        if src[0] in ('x', 'y') and src[2] & 8 and not dst[2] & 8:
            # register to register: the store opcode puts the high register in VEX.R, so the two-byte form fits
            return _enc(a, bytes([op[1]]), _vr(src), dst, vex=(pp, mm, W, dst[0] == 'y', 0))
        return _enc(a, bytes([op[0]]), dst, _vr(src, True), vex=(pp, mm, W, dst[0] == 'y', 0))
    if form == 'vmi':
        _nops(ops, 2)
        src, dst = _vr(ops[0]), _vr(ops[1])
        return _enc(a, bytes([op[0]]), op[1], src, vex=(pp, mm, W, dst[0] == 'y', dst[2]), imm=imm)
    if form == 'mri':
        _nops(ops, 2)
        src, dst = _vr(ops[0]), _vr(ops[1], True)
        return _enc(a, bytes([op]), src, dst, vex=(pp, mm, W, src[0] == 'y', 0), imm=imm)
    if form == 'gm':
        _nops(ops, 2)
        src, dst = _vr(ops[0]), _gpr(ops[1], (4, 8))
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, W, src[0] == 'y', 0))
    if form == 'gmi':  # vpextrw $imm, xmm, general register
        _nops(ops, 2)
        src, dst = _xmm(ops[0]), _gpr(ops[1], (4, 8))
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, 0, 0, 0), imm=imm)
    if form == 'movs':  # vmovss/vmovsd: load, store, or register merge of three
        if len(ops) == 3:
            src2, src1, dst = _xmm(ops[0]), _xmm(ops[1]), _xmm(ops[2])
            return _enc(a, bytes([op[0]]), dst, src2, vex=(pp, mm, 0, 0, src1[2]))
        _nops(ops, 2)
        src, dst = ops
        if dst[0] == 'm':
            return _enc(a, bytes([op[1]]), _xmm(src), dst, vex=(pp, mm, 0, 0, 0))
        if src[0] != 'm':
            raise AssemblerError('vmovss/vmovsd between registers takes three operands')
        return _enc(a, bytes([op[0]]), _xmm(dst), src, vex=(pp, mm, 0, 0, 0))
    if form == 'cvti':  # vcvtsi2ss/sd general register or memory, xmm, xmm
        _nops(ops, 3)
        src, src1, dst = ops
        _xmm(src1), _xmm(dst)
        if src[0] == 'r':
            _gpr(src, (4, 8))
            size = _size(sz, (src,))
        elif src[0] == 'm':
            size = sz or 4
        else:
            raise AssemblerError('bad operands')
        return _enc(a, bytes([op]), dst, src, vex=(pp, mm, size == 8, 0, src1[2]))
    if form == 'cvtf':  # vcvt(t)ss/sd2si xmm or memory, general register
        _nops(ops, 2)
        src, dst = ops
        _gpr(dst, (4, 8))
        size = _size(sz, (dst,))
        return _enc(a, bytes([op]), dst, _xmm(src, True), vex=(pp, mm, size == 8, 0, 0))
    if form == 'movgx':  # vmovd/vmovq between general registers or memory and xmm; vmovq between xmm
        _nops(ops, 2)
        src, dst = ops
        q = arg[3]
        if src[0] == 'x' and dst[0] == 'x':
            if not q:
                raise AssertionError
            return _enc(a, b'\x7e', dst, src, vex=(2, 1, 0, 0, 0))
        if dst[0] == 'x':
            if src[0] == 'm' and q:
                return _enc(a, b'\x7e', dst, src, vex=(2, 1, 0, 0, 0))
            if src[0] == 'r':
                _gpr(src, (8,) if q else (4,))
            return _enc(a, b'\x6e', dst, src, vex=(1, 1, q, 0, 0))
        if src[0] == 'x':
            if dst[0] == 'm' and q:
                return _enc(a, b'\xd6', src, dst, vex=(1, 1, 0, 0, 0))
            if dst[0] == 'r':
                _gpr(dst, (8,) if q else (4,))
            return _enc(a, b'\x7e', src, dst, vex=(1, 1, q, 0, 0))
        raise AssemblerError('bad operands')
    raise AssertionError(form)


def _vz(o, mem=False):
    """An %xmm, %ymm or %zmm register (or memory, if mem)."""
    if o[0] not in ('x', 'y', 'z') and not (mem and o[0] == 'm'):
        raise AssemblerError('expected an %%xmm, %%ymm or %%zmm register%s' % (' or memory' if mem else ''))
    return o


_EVEXL = {'x': 0, 'y': 1, 'z': 2}


def _h_evex(a, ops, sz, arg):
    """AVX-512 (EVEX): arg = (prefix, map, opcode, W, form, vex). Forms as _h_avx's (rvm so far), a mask
    ({%k1}, {z}) on the destination; a memory operand is a full vector (disp8*N with N its bytes). vex: the
    instruction also has a VEX form ({vex} vpdpbusd: AVX-VNNI), which `{vex}` asks for."""
    pfx, mm, op, W, form, vexok = arg
    pp = _PP[pfx]
    aaa, z = a.mask
    if form == 'rvm':
        _nops(ops, 3)
        src2, src1, dst = _vz(ops[0], True), _vz(ops[1]), _vz(ops[2])
        if len({src1[0], dst[0]} | ({src2[0]} if src2[0] != 'm' else set())) != 1:
            raise AssemblerError('mixed vector register widths')
        if a.pref == 'vex':
            if not vexok or dst[0] == 'z' or aaa or z:
                raise AssemblerError('no VEX encoding for these operands')
            return _enc(a, bytes([op]), dst, src2, vex=(pp, mm, W, dst[0] == 'y', src1[2]))
        L = _EVEXL[dst[0]]
        return _enc(a, bytes([op]), dst, src2, evex=(pp, mm, W, L, src1[2], aaa, z, 0, 16 << L))
    raise AssertionError(form)


_DECO = re.compile(r'\{([^}]*)\}')


def _decorations(a, strs):
    """Takes AVX-512's {%kN} and {z} off the last operand into a.mask."""
    last = strs[-1]
    j = last.find('{')
    if j < 0 or not last.endswith('}'):
        return
    aaa = z = 0
    for d in _DECO.findall(last[j:]):
        d = d.strip().lower()
        if d == 'z':
            z = 1
        elif d[:2] == '%k' and d[2:] in ('1', '2', '3', '4', '5', '6', '7'):
            aaa = int(d[2:])
        else:
            raise AssemblerError('bad operand decoration {%s}' % d)
    if z and not aaa:
        raise AssemblerError('{z} needs a mask')
    strs[-1] = last[:j].strip()
    a.mask = (aaa, z)


# AVX-512 forms of the AVX and AVX2 mnemonics (and the ones only EVEX has): mnemonic -> (W, tuple). The tuple gives
# the N of a disp8*N: fv, the vector's bytes; hv, half of them (and hvm: half, a memory operand of narrower lanes);
# qv, a quarter; t1s<k>, one element of k bytes; t4/t8, 16 or 32 bytes (the 32x4 and 64x4 forms)
_P0, _P66, _PF3, _PF2 = b'', b'\x66', b'\xf3', b'\xf2'
_EVX = {}


def _evx(names, W, tup):
    for n in names.split():
        _EVX[n] = (W, tup)


_evx('vaddps vmulps vsubps vminps vdivps vmaxps vsqrtps vcmpps vmovups vmovaps vcvtdq2ps vcvttps2dq vandps vorps '
     'vxorps vandnps', 0, 'fv')
_evx('vaddpd vmulpd vsubpd vminpd vdivpd vmaxpd vsqrtpd vcmppd vmovupd vmovapd vcvtpd2ps vandpd vorpd vxorpd vandnpd',
     1, 'fv')
_evx('vpaddb vpaddw vpsubb vpsubw vpmullw vpmulhw vpminub vpmaxub vpminsw vpmaxsw vpcmpeqb vpcmpeqw vpcmpgtb '
     'vpcmpgtw vpunpcklbw vpunpcklwd vpunpckhbw vpunpckhwd vpshufb vpminsb vpmaxsb vpminuw vpmaxuw vpackuswb '
     'vpacksswb vpsllw vpsrlw vpsraw', 0, 'fv')
_evx('vpaddd vpsubd vpmulld vpcmpeqd vpcmpgtd vpminsd vpmaxsd vpminud vpmaxud vpunpckldq vpunpckhdq vpsllvd vpsrlvd '
     'vpsravd vpshufd vpslld vpsrld vpsrad vpackusdw vpackssdw', 0, 'fv')
_evx('vpaddq vpsubq vpmuludq vpcmpeqq vpcmpgtq vpunpcklqdq vpunpckhqdq vpsllvq vpsrlvq vpsllq vpsrlq', 1, 'fv')
_evx('vpbroadcastb', 0, 't1s1')
_evx('vpbroadcastw', 0, 't1s2')
_evx('vpbroadcastd vbroadcastss', 0, 't1s4')
_evx('vpbroadcastq vbroadcastsd', 1, 't1s8')
_evx('vpmovsxbw vpmovsxwd vpmovsxdq vpmovzxbw vpmovzxwd vpmovzxdq vcvtph2ps vcvtps2ph', 0, 'hvm')
_evx('vcvtps2pd', 0, 'hv')
# only EVEX has these: (mnemonic, prefix, map, opcode, W, form, tuple). Forms as _h_avx's, and: mk, a mask register
# (rm) into a vector; km, a vector's lane signs into a mask register; mrn, a narrowing store (vpmovwb: the source
# in reg, the destination register or memory in rm)
_EVEX_ONLY = [('vpandd', _P66, 1, 0xdb, 0, 'rvm', 'fv'), ('vpandq', _P66, 1, 0xdb, 1, 'rvm', 'fv'),
              ('vpandnd', _P66, 1, 0xdf, 0, 'rvm', 'fv'), ('vpandnq', _P66, 1, 0xdf, 1, 'rvm', 'fv'),
              ('vpord', _P66, 1, 0xeb, 0, 'rvm', 'fv'), ('vporq', _P66, 1, 0xeb, 1, 'rvm', 'fv'),
              ('vpxord', _P66, 1, 0xef, 0, 'rvm', 'fv'), ('vpxorq', _P66, 1, 0xef, 1, 'rvm', 'fv'),
              ('vpternlogd', _P66, 3, 0x25, 0, 'rvmi', 'fv'), ('vpternlogq', _P66, 3, 0x25, 1, 'rvmi', 'fv'),
              ('vpmovm2b', _PF3, 2, 0x28, 0, 'mk', 'fv'), ('vpmovm2w', _PF3, 2, 0x28, 1, 'mk', 'fv'),
              ('vpmovm2d', _PF3, 2, 0x38, 0, 'mk', 'fv'), ('vpmovm2q', _PF3, 2, 0x38, 1, 'mk', 'fv'),
              ('vpmovb2m', _PF3, 2, 0x29, 0, 'km', 'fv'), ('vpmovw2m', _PF3, 2, 0x29, 1, 'km', 'fv'),
              ('vpmovd2m', _PF3, 2, 0x39, 0, 'km', 'fv'), ('vpmovq2m', _PF3, 2, 0x39, 1, 'km', 'fv'),
              ('vextracti32x4', _P66, 3, 0x39, 0, 'mri', 't4'), ('vextracti64x4', _P66, 3, 0x3b, 1, 'mri', 't8'),
              ('vextractf32x4', _P66, 3, 0x19, 0, 'mri', 't4'), ('vextractf64x4', _P66, 3, 0x1b, 1, 'mri', 't8'),
              ('vinserti32x4', _P66, 3, 0x38, 0, 'rvmi', 't4'), ('vinserti64x4', _P66, 3, 0x3a, 1, 'rvmi', 't8'),
              ('vinsertf32x4', _P66, 3, 0x18, 0, 'rvmi', 't4'), ('vinsertf64x4', _P66, 3, 0x1a, 1, 'rvmi', 't8'),
              ('vbroadcasti32x4', _P66, 2, 0x5a, 0, 'rm', 't4'), ('vbroadcasti64x4', _P66, 2, 0x5b, 1, 'rm', 't8'),
              ('vpmovwb', _PF3, 2, 0x30, 0, 'mrn', 'hvm'), ('vpmovdw', _PF3, 2, 0x33, 0, 'mrn', 'hvm'),
              ('vpmovqd', _PF3, 2, 0x35, 0, 'mrn', 'hvm'),
              ('vrndscaleps', _P66, 3, 0x08, 0, 'rmi', 'fv'), ('vrndscalepd', _P66, 3, 0x09, 1, 'rmi', 'fv'),
              ('vcvtudq2ps', _PF2, 1, 0x7a, 0, 'rm', 'fv'), ('vcvttps2udq', _P0, 1, 0x78, 0, 'rm', 'fv'),
              ('vcvtqq2pd', _PF3, 1, 0xe6, 1, 'rm', 'fv'), ('vcvtuqq2pd', _PF3, 1, 0x7a, 1, 'rm', 'fv'),
              ('vcvttpd2qq', _P66, 1, 0x7a, 1, 'rm', 'fv'), ('vcvttpd2uqq', _P66, 1, 0x78, 1, 'rm', 'fv'),
              ('vpsraq', _P66, 1, (0x72, 4), 1, 'vmi', 'fv'), ('vpsravq', _P66, 2, 0x46, 1, 'rvm', 'fv'),
              ('vpmullq', _P66, 2, 0x40, 1, 'rvm', 'fv'), ('vpminsq', _P66, 2, 0x39, 1, 'rvm', 'fv'),
              ('vpmaxsq', _P66, 2, 0x3d, 1, 'rvm', 'fv'), ('vpminuq', _P66, 2, 0x3b, 1, 'rvm', 'fv'),
              ('vpmaxuq', _P66, 2, 0x3f, 1, 'rvm', 'fv')]


def _tupn(tup, L):
    vl = 16 << L
    return {'fv': vl, 'hv': vl // 2, 'hvm': vl // 2, 'qv': vl // 4, 't1s1': 1, 't1s2': 2, 't1s4': 4, 't1s8': 8,
            't4': 16, 't8': 32}[tup]


def _vk(o, mem=False):
    """An %xmm, %ymm or %zmm register (or memory, if mem), EVEX's: any of the 32."""
    if o[0] not in ('x', 'y', 'z') and not (mem and o[0] == 'm'):
        raise AssemblerError('expected an %%xmm, %%ymm or %%zmm register%s' % (' or memory' if mem else ''))
    return o


def _kreg(o):
    if o[0] != 'k':
        raise AssemblerError('expected a mask register')
    return o


def _h_avx512(a, ops, sz, arg):
    """The EVEX form of an _h_avx instruction (arg as there; the W and tuple from _EVX), or of one only AVX-512 has
    (arg with its own W and tuple). The length is the widest vector operand's."""
    if len(arg) == 6:
        pfx, mm, op, W, form, tup = arg
    else:
        pfx, mm, op, _, form = arg
        e = _EVX.get(a.mnemonic)
        if e is None:
            raise AssemblerError('%s has no AVX-512 form here' % a.mnemonic)
        W, tup = e
    pp = _PP[pfx]
    aaa, z = a.mask
    imm = None
    if form in ('rvmi', 'rmi', 'vmi', 'mri'):
        if ops[0][0] != 'i' or ops[0][1] is not None or not 0 <= ops[0][2] <= 255:
            raise AssemblerError('expected a constant from 0 to 255')
        imm = (1, None, ops[0][2], None)
        ops = ops[1:]
    vec = [o for o in ops if o[0] in ('x', 'y', 'z')]
    if not vec:
        raise AssemblerError('expected a vector register')
    L = max(_EVEXL[o[0]] for o in vec)

    def enc(reg, rm, vvvv=0, LL=L):
        n = _tupn(tup, LL)
        return _enc(a, bytes([op if not isinstance(op, tuple) else op[0]]), reg, rm, imm=imm,
                    evex=(pp, mm, W, LL, vvvv, aaa, z, 0, n))
    if form in ('rvm', 'rvmi'):
        _nops(ops, 3)
        src2, src1, dst = ops
        _vk(src2, True), _vk(src1)
        if dst[0] == 'k':  # a comparison into a mask register: the length is the sources'
            return enc(dst, src2, src1[2], _EVEXL[src1[0]])
        _vk(dst)
        if form == 'rvmi' and tup in ('t4', 't8'):  # vinsert*: the length is the destination's
            return enc(dst, src2, src1[2], _EVEXL[dst[0]])
        return enc(dst, src2, src1[2], _EVEXL[dst[0]])
    if form in ('rm', 'rmi', 'rmw'):
        _nops(ops, 2)
        src, dst = _vk(ops[0], True), _vk(ops[1])
        return enc(dst, src, 0, _EVEXL[dst[0]])
    if form == 'rmn':  # narrowing conversions: the source's length
        _nops(ops, 2)
        src, dst = _vk(ops[0], True), _vk(ops[1])
        return enc(dst, src, 0, _EVEXL[src[0]] if src[0] != 'm' else _EVEXL[dst[0]] + 1)
    if form == 'ld':
        _nops(ops, 2)
        src, dst = ops
        if dst[0] == 'm':
            _vk(src)
            op = op[1]
            return enc(src, dst, 0, _EVEXL[src[0]])
        _vk(dst)
        _vk(src, True)
        op = op[0]
        return enc(dst, src, 0, _EVEXL[dst[0]])
    if form == 'vmi':
        _nops(ops, 2)
        src, dst = _vk(ops[0], True), _vk(ops[1])
        return _enc(a, bytes([op[0]]), op[1], src, imm=imm,
                    evex=(pp, mm, W, _EVEXL[dst[0]], dst[2], aaa, z, 0, _tupn(tup, _EVEXL[dst[0]])))
    if form == 'mri':  # extracts and vcvtps2ph: the source's length, the destination in rm
        _nops(ops, 2)
        src, dst = _vk(ops[0]), _vk(ops[1], True)
        return enc(src, dst, 0, _EVEXL[src[0]])
    if form == 'mrn':  # vpmovwb: the source's length, the destination in rm
        _nops(ops, 2)
        src, dst = _vk(ops[0]), _vk(ops[1], True)
        return enc(src, dst, 0, _EVEXL[src[0]])
    if form == 'mk':
        _nops(ops, 2)
        src, dst = _kreg(ops[0]), _vk(ops[1])
        return enc(dst, src, 0, _EVEXL[dst[0]])
    if form == 'km':
        _nops(ops, 2)
        src, dst = _vk(ops[0]), _kreg(ops[1])
        return enc(dst, src, 0, _EVEXL[src[0]])
    raise AssemblerError('no AVX-512 form of %s' % a.mnemonic)


def _tmm(o):
    if o[0] != 't':
        raise AssemblerError('expected a tile register')
    return o


def _h_amx(a, ops, sz, arg):
    """AMX (VEX.128, map 0F38, W0): arg = (prefix, opcode, form). Forms: cfg, ldtilecfg/sttilecfg (memory); rel,
    tilerelease; zero, tilezero %tmm; ld, tileloadd (base, index, scale), %tmm (the memory operand needs an index:
    a SIB); st, tilestored %tmm, (base, index, scale); dp, the dot products %tmm(src2), %tmm(src1), %tmm(dst), dst
    in reg, src1 in rm, src2 in vvvv."""
    pfx, op, form = arg
    pp = _PP[pfx]
    if form == 'rel':
        _nops(ops, 0)
        return a.put(bytes((0xc4, 0xe2, 0x78, op, 0xc0)), ())
    if form == 'cfg':
        _nops(ops, 1)
        if ops[0][0] != 'm':
            raise AssemblerError('expected memory')
        return _enc(a, bytes([op]), 0, ops[0], vex=(pp, 2, 0, 0, 0))
    if form == 'zero':
        _nops(ops, 1)
        t = _tmm(ops[0])
        return _enc(a, bytes([op]), t, ('t', 1024, 0, 0), vex=(pp, 2, 0, 0, 0))
    if form in ('ld', 'st'):
        _nops(ops, 2)
        m, t = (ops[0], _tmm(ops[1])) if form == 'ld' else (ops[1], _tmm(ops[0]))
        if m[0] != 'm' or m[6] is None or m[5] in (None, _RIP):
            raise AssemblerError('a tile load or store needs a base and an index register')
        return _enc(a, bytes([op]), t, m, vex=(pp, 2, 0, 0, 0))
    if form == 'dp':
        _nops(ops, 3)
        src2, src1, dst = (_tmm(o) for o in ops)
        return _enc(a, bytes([op]), dst, src1, vex=(pp, 2, 0, 0, src2[2]))
    raise AssertionError(form)


# mnemonic table: name -> (handler, suffix size, argument)

_SFX = {'b': 1, 'w': 2, 'l': 4, 'q': 8}
_INSNS = {}


def _add(names, h, arg=None, sfx='bwlq'):
    for n in names.split():
        for c in sfx:
            _INSNS[n + c] = (h, _SFX[c], arg)
        _INSNS[n] = (h, 0, arg)


for _n, _d in (('add', 0), ('or', 1), ('adc', 2), ('sbb', 3), ('and', 4), ('sub', 5),
               ('xor', 6), ('cmp', 7)):
    _add(_n, _h_alu, _d)
for _n, _d in (('rol', 0), ('ror', 1), ('rcl', 2), ('rcr', 3), ('shl', 4), ('sal', 4),
               ('shr', 5), ('sar', 7)):
    _add(_n, _h_shift, _d)
for _n, _d in (('not', (2, False)), ('neg', (3, False)), ('mul', (4, False)),
               ('div', (6, False)), ('idiv', (7, False)), ('inc', (0, True)),
               ('dec', (1, True))):
    _add(_n, _h_unary, _d)
_add('test', _h_test)
_add('imul', _h_imul)
_add('xchg', _h_xchg)
_add('lea', _h_lea, sfx='wlq')
_add('push', _h_push, False, sfx='wq')
_add('pop', _h_push, True, sfx='wq')
_add('bswap', _h_bswap, sfx='lq')
_add('bsf', _h_bitscan, (b'', b'\x0f\xbc'), sfx='wlq')
_add('bsr', _h_bitscan, (b'', b'\x0f\xbd'), sfx='wlq')
_add('tzcnt', _h_bitscan, (b'\xf3', b'\x0f\xbc'), sfx='wlq')
_add('lzcnt', _h_bitscan, (b'\xf3', b'\x0f\xbd'), sfx='wlq')
_add('popcnt', _h_bitscan, (b'\xf3', b'\x0f\xb8'), sfx='wlq')
_add('cmpxchg', _h_xadd, 0xb0)
_add('xadd', _h_xadd, 0xc0)
for _c in 'bwlq':
    _INSNS['mov' + _c] = (_h_mov, _SFX[_c], 'mov' + _c)
_INSNS['mov'] = (_h_mov, 0, 'mov')
_INSNS['movabs'] = _INSNS['movabsq'] = (_h_movabs, 0, None)
_INSNS['movd'] = (_h_movq, 4, 'movd')
for _n, _op, _s, _dd in (('movsbw', b'\x0f\xbe', 1, 2), ('movsbl', b'\x0f\xbe', 1, 4),
                         ('movsbq', b'\x0f\xbe', 1, 8), ('movswl', b'\x0f\xbf', 2, 4),
                         ('movswq', b'\x0f\xbf', 2, 8), ('movslq', b'\x63', 4, 8),
                         ('movzbw', b'\x0f\xb6', 1, 2), ('movzbl', b'\x0f\xb6', 1, 4),
                         ('movzbq', b'\x0f\xb6', 1, 8), ('movzwl', b'\x0f\xb7', 2, 4),
                         ('movzwq', b'\x0f\xb7', 2, 8)):
    _INSNS[_n] = (_h_movx, 0, (_op, _s, _dd))
for _n in ('movsx', 'movzx', 'movsxd'):
    _INSNS[_n] = (_h_movsx, 0, _n)
for _n in ('call', 'jmp'):
    _INSNS[_n] = _INSNS[_n + 'q'] = (_h_call, 0, _n == 'jmp')
for _n, _code in (('ret', b'\xc3'), ('leave', b'\xc9'), ('cqto', b'\x48\x99'), ('cqo', b'\x48\x99'),
                  ('cltd', b'\x99'), ('cdq', b'\x99'), ('cltq', b'\x48\x98'),
                  ('cdqe', b'\x48\x98'), ('cwtl', b'\x98'), ('cwde', b'\x98'),
                  ('cwtd', b'\x66\x99'), ('cwd', b'\x66\x99'), ('cbtw', b'\x66\x98'),
                  ('cbw', b'\x66\x98'), ('ud2', b'\x0f\x0b'), ('endbr64', b'\xf3\x0f\x1e\xfa'),
                  ('nop', b'\x90'), ('int3', b'\xcc'), ('pause', b'\xf3\x90'),
                  ('mfence', b'\x0f\xae\xf0'), ('lfence', b'\x0f\xae\xe8'),
                  ('sfence', b'\x0f\xae\xf8'), ('hlt', b'\xf4'), ('rdtsc', b'\x0f\x31'),
                  ('cpuid', b'\x0f\xa2'), ('syscall', b'\x0f\x05')):
    _INSNS[_n] = (_h_fixed, 0, _code)
_INSNS['retq'] = _INSNS['ret']
_INSNS['stmxcsr'] = (_h_mxcsr, 0, 3)
_INSNS['ldmxcsr'] = (_h_mxcsr, 0, 2)
_INSNS['leaveq'] = _INSNS['leave']

_CC = {'o': 0, 'no': 1, 'b': 2, 'c': 2, 'nae': 2, 'ae': 3, 'nb': 3, 'nc': 3, 'e': 4, 'z': 4,
       'ne': 5, 'nz': 5, 'be': 6, 'na': 6, 'a': 7, 'nbe': 7, 's': 8, 'ns': 9, 'p': 10, 'pe': 10,
       'np': 11, 'po': 11, 'l': 12, 'nge': 12, 'ge': 13, 'nl': 13, 'le': 14, 'ng': 14, 'g': 15,
       'nle': 15}
for _c, _v in _CC.items():
    for _s in 'wlq':
        _INSNS['cmov%s%s' % (_c, _s)] = (_h_cmov, _SFX[_s], _v)
for _c, _v in _CC.items():  # after the suffixed forms: cmovl is "less", not cmov + l
    _INSNS['cmov' + _c] = (_h_cmov, 0, _v)
    _INSNS['set' + _c] = (_h_setcc, 0, _v)
    _INSNS['j' + _c] = (_h_jcc, 0, _v)

for _n, _op in (('add', 0x58), ('mul', 0x59), ('sub', 0x5c), ('min', 0x5d), ('div', 0x5e),
                ('max', 0x5f), ('sqrt', 0x51)):
    for _k, _p in (('ss', b'\xf3'), ('sd', b'\xf2'), ('ps', b''), ('pd', b'\x66')):
        _INSNS[_n + _k] = (_h_sse, 0, (_p, bytes([0x0f, _op])))
for _n, _p, _op in (('andps', b'', 0x54), ('andpd', b'\x66', 0x54), ('andnps', b'', 0x55),
                    ('andnpd', b'\x66', 0x55), ('orps', b'', 0x56), ('orpd', b'\x66', 0x56),
                    ('xorps', b'', 0x57), ('xorpd', b'\x66', 0x57), ('pxor', b'\x66', 0xef),
                    ('pand', b'\x66', 0xdb), ('por', b'\x66', 0xeb),
                    ('ucomiss', b'', 0x2e), ('ucomisd', b'\x66', 0x2e), ('comiss', b'', 0x2f),
                    ('comisd', b'\x66', 0x2f), ('cvtss2sd', b'\xf3', 0x5a),
                    ('cvtsd2ss', b'\xf2', 0x5a), ('unpcklps', b'', 0x14),
                    ('unpcklpd', b'\x66', 0x14), ('cvtps2pd', b'', 0x5a),
                    ('cvtpd2ps', b'\x66', 0x5a), ('cvtdq2ps', b'', 0x5b)):
    _INSNS[_n] = (_h_sse, 0, (_p, bytes([0x0f, _op])))
for _n, _op in (('paddb', 0xfc), ('paddw', 0xfd), ('paddd', 0xfe), ('paddq', 0xd4), ('psubb', 0xf8), ('psubw', 0xf9),
                ('psubd', 0xfa), ('psubq', 0xfb), ('pmullw', 0xd5), ('pmuludq', 0xf4), ('pcmpeqb', 0x74), ('pcmpeqw', 0x75),
                ('pcmpeqd', 0x76), ('pcmpgtb', 0x64), ('pcmpgtw', 0x65), ('pcmpgtd', 0x66), ('pminub', 0xda), ('pmaxub', 0xde),
                ('pminsw', 0xea), ('pmaxsw', 0xee), ('pandn', 0xdf), ('punpcklbw', 0x60), ('punpcklwd', 0x61),
                ('punpckldq', 0x62), ('punpckhdq', 0x6a), ('punpcklqdq', 0x6c), ('punpckhbw', 0x68), ('punpckhwd', 0x69),
                ('punpckhqdq', 0x6d), ('packuswb', 0x67), ('packssdw', 0x6b), ('packsswb', 0x63)):
    _INSNS[_n] = (_h_sse, 0, (b'\x66', bytes([0x0f, _op])))
for _n, _p, _o in (('pshufd', b'\x66', 0x70), ('pshuflw', b'\xf2', 0x70), ('cmpps', b'', 0xc2), ('cmppd', b'\x66', 0xc2)):
    _INSNS[_n] = (_h_sseimm, 0, (_p, bytes([0x0f, _o])))
for _n, _op, _d in (('psllw', 0x71, 6), ('pslld', 0x72, 6), ('psllq', 0x73, 6), ('psrlw', 0x71, 2), ('psrld', 0x72, 2),
                    ('psrlq', 0x73, 2), ('psraw', 0x71, 4), ('psrad', 0x72, 4)):
    _INSNS[_n] = (_h_pshift, 0, (_op, _d))
_INSNS['pextrw'] = (_h_pextrw, 0, None)
for _o, _base in (('132', 0x98), ('213', 0xa8), ('231', 0xb8)):
    for _n, _add in (('vfmadd', 0), ('vfmsub', 2), ('vfnmadd', 4), ('vfnmsub', 6)):
        for _k, _w, _d in (('ps', 0, 0), ('pd', 1, 0), ('ss', 0, 1), ('sd', 1, 1)):
            _INSNS['%s%s%s' % (_n, _o, _k)] = (_h_vfma, 0, (_w, _base + _add + _d))
_INSNS['cpuid'] = (_h_fixed, 0, b'\x0f\xa2')
# AVX-512 (EVEX): (mnemonic, prefix, map, opcode, W, form, has a VEX form)
_EVEX = [('vdpbf16ps', b'\xf3', 2, 0x52, 0, 'rvm', False),  # AVX512-BF16
         ('vpdpbusd', b'\x66', 2, 0x50, 0, 'rvm', True)]  # AVX512-VNNI; {vex}: AVX-VNNI
for _e in _EVEX:
    _INSNS[_e[0]] = (_h_evex, 0, _e[1:])
_INSNS['vzeroupper'] = (_h_fixed, 0, b'\xc5\xf8\x77')
# AVX and AVX2: (mnemonic, prefix, map (1 0F, 2 0F38, 3 0F3A), opcode, W, form)
_AVX = []
for _n, _o in (('add', 0x58), ('mul', 0x59), ('sub', 0x5c), ('min', 0x5d), ('div', 0x5e), ('max', 0x5f),
               ('and', 0x54), ('andn', 0x55), ('or', 0x56), ('xor', 0x57)):
    _AVX += [('v%sps' % _n, _P0, 1, _o, 0, 'rvm'), ('v%spd' % _n, _P66, 1, _o, 0, 'rvm')]
_AVX += [('vsqrtps', _P0, 1, 0x51, 0, 'rm'), ('vsqrtpd', _P66, 1, 0x51, 0, 'rm'),
         ('vcmpps', _P0, 1, 0xc2, 0, 'rvmi'), ('vcmppd', _P66, 1, 0xc2, 0, 'rvmi')]
for _n, _o in (('vpaddb', 0xfc), ('vpaddw', 0xfd), ('vpaddd', 0xfe), ('vpaddq', 0xd4), ('vpsubb', 0xf8),
               ('vpsubw', 0xf9), ('vpsubd', 0xfa), ('vpsubq', 0xfb), ('vpmullw', 0xd5), ('vpmuludq', 0xf4),
               ('vpmulhw', 0xe5), ('vpand', 0xdb), ('vpandn', 0xdf), ('vpor', 0xeb), ('vpxor', 0xef),
               ('vpcmpeqb', 0x74), ('vpcmpeqw', 0x75), ('vpcmpeqd', 0x76), ('vpcmpgtb', 0x64), ('vpcmpgtw', 0x65),
               ('vpcmpgtd', 0x66), ('vpminub', 0xda), ('vpmaxub', 0xde), ('vpminsw', 0xea), ('vpmaxsw', 0xee),
               ('vpunpcklbw', 0x60), ('vpunpcklwd', 0x61), ('vpunpckldq', 0x62), ('vpunpckhdq', 0x6a),
               ('vpunpcklqdq', 0x6c), ('vpunpckhqdq', 0x6d)):
    _AVX.append((_n, _P66, 1, _o, 0, 'rvm'))
for _n, _o in (('vpmulld', 0x40), ('vpcmpeqq', 0x29), ('vpcmpgtq', 0x37), ('vpminsb', 0x38), ('vpminsd', 0x39),
               ('vpminuw', 0x3a), ('vpminud', 0x3b), ('vpmaxsb', 0x3c), ('vpmaxsd', 0x3d), ('vpmaxuw', 0x3e),
               ('vpmaxud', 0x3f)):
    _AVX.append((_n, _P66, 2, _o, 0, 'rvm'))
# widening and narrowing (GNU vectors' __builtin_convertvector): rmw, an %xmm or memory into a wider register (the
# destination's length); rmn, a %ymm into an %xmm (the source's length)
_AVX += [('vpmovsxbw', _P66, 2, 0x20, 0, 'rmw'), ('vpmovsxwd', _P66, 2, 0x23, 0, 'rmw'), ('vpmovsxdq', _P66, 2, 0x25, 0, 'rmw'),
         ('vpmovzxwd', _P66, 2, 0x33, 0, 'rmw'), ('vpmovzxdq', _P66, 2, 0x35, 0, 'rmw'), ('vcvtps2pd', _P0, 1, 0x5a, 0, 'rmw'),
         ('vcvtpd2ps', _P66, 1, 0x5a, 0, 'rmn'), ('vcvtdq2ps', _P0, 1, 0x5b, 0, 'rm'), ('vcvttps2dq', _PF3, 1, 0x5b, 0, 'rm'),
         ('vpackuswb', _P66, 1, 0x67, 0, 'rvm'), ('vpackusdw', _P66, 2, 0x2b, 0, 'rvm')]
_AVX += [('vpsllvd', _P66, 2, 0x47, 0, 'rvm'), ('vpsllvq', _P66, 2, 0x47, 1, 'rvm'), ('vpsrlvd', _P66, 2, 0x45, 0, 'rvm'),
         ('vpsrlvq', _P66, 2, 0x45, 1, 'rvm'), ('vpsravd', _P66, 2, 0x46, 0, 'rvm')]  # AVX2 shifts by a vector
_AVX += [('vpshufb', _P66, 2, 0x00, 0, 'rvm'), ('vpmovzxbw', _P66, 2, 0x30, 0, 'rmw'),  # vectorize's loops over bytes
         ('vbroadcastss', _P66, 2, 0x18, 0, 'rm'), ('vbroadcastsd', _P66, 2, 0x19, 0, 'rm'),
         ('vpbroadcastb', _P66, 2, 0x78, 0, 'rm'), ('vpbroadcastw', _P66, 2, 0x79, 0, 'rm'),
         ('vpbroadcastd', _P66, 2, 0x58, 0, 'rm'), ('vpbroadcastq', _P66, 2, 0x59, 0, 'rm'),
         ('vptest', _P66, 2, 0x17, 0, 'rm'), ('vbroadcasti128', _P66, 2, 0x5a, 0, 'rm'),
         ('vcvtph2ps', _P66, 2, 0x13, 0, 'rm'), ('vcvtps2ph', _P66, 3, 0x1d, 0, 'mri'),  # F16C
         ('vpshufd', _P66, 1, 0x70, 0, 'rmi'), ('vpermq', _P66, 3, 0x00, 1, 'rmi'), ('vroundpd', _P66, 3, 0x09, 0, 'rmi'),
         ('vextracti128', _P66, 3, 0x39, 0, 'mri'), ('vextractf128', _P66, 3, 0x19, 0, 'mri'),
         ('vinserti128', _P66, 3, 0x38, 0, 'rvmi'), ('vinsertf128', _P66, 3, 0x18, 0, 'rvmi'),
         ('vblendvps', _P66, 3, 0x4a, 0, 'rvmr'), ('vblendvpd', _P66, 3, 0x4b, 0, 'rvmr'),
         ('vpblendvb', _P66, 3, 0x4c, 0, 'rvmr'),
         ('vpmovmskb', _P66, 1, 0xd7, 0, 'gm'), ('vmovmskps', _P0, 1, 0x50, 0, 'gm'), ('vmovmskpd', _P66, 1, 0x50, 0, 'gm'),
         ('vmovups', _P0, 1, (0x10, 0x11), 0, 'ld'), ('vmovupd', _P66, 1, (0x10, 0x11), 0, 'ld'),
         ('vmovaps', _P0, 1, (0x28, 0x29), 0, 'ld'), ('vmovapd', _P66, 1, (0x28, 0x29), 0, 'ld'),
         ('vmovdqu', _PF3, 1, (0x6f, 0x7f), 0, 'ld'), ('vmovdqa', _P66, 1, (0x6f, 0x7f), 0, 'ld')]
for _n, _o, _d in (('vpsllw', 0x71, 6), ('vpslld', 0x72, 6), ('vpsllq', 0x73, 6), ('vpsrlw', 0x71, 2),
                   ('vpsrld', 0x72, 2), ('vpsrlq', 0x73, 2), ('vpsraw', 0x71, 4), ('vpsrad', 0x72, 4)):
    _AVX.append((_n, _P66, 1, (_o, _d), 0, 'vmi'))
# the VEX forms of the scalar SSE instructions (the emitter writes them in functions with 256-bit vectors)
for _k, _p in (('ss', _PF3), ('sd', _PF2)):
    for _n, _o in (('add', 0x58), ('mul', 0x59), ('sub', 0x5c), ('min', 0x5d), ('div', 0x5e), ('max', 0x5f),
                   ('sqrt', 0x51)):
        _AVX.append(('v%s%s' % (_n, _k), _p, 1, _o, 0, 'rvm'))
    _AVX.append(('vmov' + _k, _p, 1, (0x10, 0x11), 0, 'movs'))
_AVX += [('vcvtss2sd', _PF3, 1, 0x5a, 0, 'rvm'), ('vcvtsd2ss', _PF2, 1, 0x5a, 0, 'rvm'),
         ('vucomiss', _P0, 1, 0x2e, 0, 'rm'), ('vucomisd', _P66, 1, 0x2e, 0, 'rm'),
         ('vcomiss', _P0, 1, 0x2f, 0, 'rm'), ('vcomisd', _P66, 1, 0x2f, 0, 'rm'),
         ('vpshuflw', _PF2, 1, 0x70, 0, 'rmi'), ('vroundss', _P66, 3, 0x0a, 0, 'rvmi'),
         ('vroundsd', _P66, 3, 0x0b, 0, 'rvmi'), ('vmovd', _P66, 1, 0x6e, 0, 'movgx'), ('vmovq', _P66, 1, 0x6e, 1, 'movgx')]
for _n, _p, _m, _o, _w, _f in _AVX:
    _INSNS[_n] = (_h_avx, 0, (_p, _m, _o, _w, _f))
for _k, _p in (('ss', _PF3), ('sd', _PF2)):
    _INSNS['vcvtsi2' + _k] = (_h_avx, 0, (_p, 1, 0x2a, 0, 'cvti'))
    _INSNS['vcvtsi2%sl' % _k] = (_h_avx, 4, (_p, 1, 0x2a, 0, 'cvti'))
    _INSNS['vcvtsi2%sq' % _k] = (_h_avx, 8, (_p, 1, 0x2a, 0, 'cvti'))
    for _t, _o in (('vcvtt', 0x2c), ('vcvt', 0x2d)):
        for _sfx, _sz in (('', 0), ('l', 4), ('q', 8)):
            _INSNS['%s%s2si%s' % (_t, _k, _sfx)] = (_h_avx, _sz, (_p, 1, _o, 0, 'cvtf'))
_INSNS['vpextrw'] = (_h_avx, 0, (_P66, 1, 0xc5, 0, 'gmi'))
for _e in _EVEX_ONLY:
    _INSNS[_e[0]] = (_h_avx512, 0, _e[1:])
# AMX: (mnemonic, prefix, opcode, form)
for _n, _p, _o, _f in (('ldtilecfg', _P0, 0x49, 'cfg'), ('sttilecfg', _P66, 0x49, 'cfg'), ('tilerelease', _P0, 0x49, 'rel'),
                       ('tilezero', _PF2, 0x49, 'zero'), ('tileloadd', _PF2, 0x4b, 'ld'), ('tileloaddt1', _P66, 0x4b, 'ld'),
                       ('tilestored', _PF3, 0x4b, 'st'), ('tdpbf16ps', _PF3, 0x5c, 'dp'), ('tdpbssd', _PF2, 0x5e, 'dp'),
                       ('tdpbsud', _PF3, 0x5e, 'dp'), ('tdpbusd', _P66, 0x5e, 'dp'), ('tdpbuud', _P0, 0x5e, 'dp')):
    _INSNS[_n] = (_h_amx, 0, (_p, _o, _f))
_INSNS['xgetbv'] = (_h_fixed, 0, b'\x0f\x01\xd0')
for _n, _p, _ld, _st in (('movss', b'\xf3', 0x10, 0x11), ('movsd', b'\xf2', 0x10, 0x11),
                         ('movups', b'', 0x10, 0x11), ('movupd', b'\x66', 0x10, 0x11),
                         ('movaps', b'', 0x28, 0x29), ('movapd', b'\x66', 0x28, 0x29),
                         ('movdqa', b'\x66', 0x6f, 0x7f), ('movdqu', b'\xf3', 0x6f, 0x7f)):
    _INSNS[_n] = (_h_ssemov, 0, (_p, bytes([0x0f, _ld]), bytes([0x0f, _st])))
for _k, _p in (('ss', b'\xf3'), ('sd', b'\xf2')):
    _INSNS['cvtsi2' + _k] = (_h_cvti2f, 0, _p)
    _INSNS['cvtsi2%sl' % _k] = (_h_cvti2f, 4, _p)
    _INSNS['cvtsi2%sq' % _k] = (_h_cvti2f, 8, _p)
    for _t, _op in (('cvtt', 0x2c), ('cvt', 0x2d)):
        _n = '%s%s2si' % (_t, _k)
        _INSNS[_n] = (_h_cvtf2i, 0, (_p, bytes([0x0f, _op])))
        _INSNS[_n + 'l'] = (_h_cvtf2i, 4, (_p, bytes([0x0f, _op])))
        _INSNS[_n + 'q'] = (_h_cvtf2i, 8, (_p, bytes([0x0f, _op])))
for _n, _op in (('roundps', 0x08), ('roundpd', 0x09), ('roundss', 0x0a), ('roundsd', 0x0b)):
    _INSNS[_n] = (_h_round, 0, bytes([_op]))

_NOPS = [b'', bytes.fromhex('90'), bytes.fromhex('6690'), bytes.fromhex('0f1f00'),
         bytes.fromhex('0f1f4000'), bytes.fromhex('0f1f440000'), bytes.fromhex('660f1f440000'),
         bytes.fromhex('0f1f8000000000'), bytes.fromhex('0f1f840000000000'),
         bytes.fromhex('660f1f840000000000'), bytes.fromhex('662e0f1f840000000000'),
         bytes.fromhex('66662e0f1f840000000000')]


def _nop_fill(count, normal):
    """GNU as's code-alignment padding (i386_generate_nops, 64-bit): a 1-byte nop first if the
    previous statement wasn't an instruction, a jump over long runs, then the longest nops first."""
    out = bytearray()
    if not normal:
        out.append(0x90)
        count -= 1
    if count // 11 > 7:
        if count - 2 <= 127:
            out += bytes([0xeb, count - 2])
            count -= 2
        else:
            out.append(0xe9)
            out += (count - 5).to_bytes(4, 'little')
            count -= 5
    q, r = divmod(count, 11)
    out += _NOPS[11] * q + _NOPS[r]
    return out


# ---------------------------------------------------------------------------------------------
# the assembler


class _Var:
    """Variable-size piece after a fragment: a relaxable branch or an alignment."""
    __slots__ = ('cc', 'sym', 'add', 'lineno', 'long', 'align', 'maxskip', 'fill', 'normal',
                 'addr', 'size')

    def __init__(self):
        self.cc = None      # None: alignment; -1: jmp; 0..15: jcc
        self.long = False
        self.size = 0
        self.addr = 0


class _Sec:
    """Assembly state of one section: closed fragments (bytes, _Var) and the open one."""
    __slots__ = ('sec', 'frags', 'cur', 'normal', 'labels', 'addrs')

    def __init__(self, sec):
        self.sec = sec
        self.frags = []
        self.cur = bytearray()
        self.normal = True   # last statement was an instruction (for NOP padding)
        self.labels = {}     # Symbol -> (fragment index, offset)
        self.addrs = None


class _Asm(_arm._Asm):
    def __init__(self, syntax):
        if syntax != 'elf':
            raise ValueError('syntax must be "elf"')
        self.states = {}
        self.st = None
        self.lock = False
        self.mask = (0, 0)  # an EVEX instruction's {%kN} and {z}
        self.pref = None  # {vex} or {evex}: the encoding asked for
        self.mnemonic = None
        self.fixups = []
        _arm._Asm.__init__(self, syntax)

    def switch(self, name, flags=None):
        _arm._Asm.switch(self, name, flags)
        st = self.states.get(self.sec)
        if st is None:
            st = self.states[self.sec] = _Sec(self.sec)
        self.st = st

    # emission

    def put(self, b, fixups):
        """Append an instruction's bytes; fixups: (offset in b, kind, sym, addend)."""
        st = self.st
        if self.sec.nobits:
            raise AssemblerError('instruction in nobits section %s' % self.sec.name)
        base = len(st.cur)
        if fixups:
            fi = len(st.frags)
            lineno = self.lineno
            for off, kind, sym, add in fixups:
                self.fixups.append((st, fi, base + off, kind, sym, add, lineno))
        st.cur += b
        st.normal = True

    def emit(self, b):
        st = self.st
        s = self.sec
        if s.nobits:
            if any(b):
                raise AssemblerError('non-zero data in nobits section %s' % s.name)
            s.size += len(b)
            return
        st.cur += b
        st.normal = False

    def _close(self, var):
        st = self.st
        st.frags.append((st.cur, var))
        st.cur = bytearray()

    def branch(self, cc, sym, add):
        if self.sec.nobits:
            raise AssemblerError('instruction in nobits section %s' % self.sec.name)
        v = _Var()
        v.cc = cc
        v.sym = sym
        v.add = add
        v.lineno = self.lineno
        self._close(v)
        self.st.normal = True

    def align(self, n, maxskip=None, fill=None):
        if n <= 0 or n & (n - 1):
            raise AssemblerError('alignment must be a power of 2')
        s = self.sec
        st = self.st
        s.align = max(s.align, n)
        if s.nobits:
            s.size += -s.size % n
            return
        if not st.frags:  # addresses are still exact: pad now
            pad = -len(st.cur) % n
            if pad and (maxskip is None or pad <= maxskip):
                st.cur += self._pad(pad, st.normal, fill)
            return
        v = _Var()
        v.align = n
        v.maxskip = maxskip
        v.fill = fill
        v.normal = st.normal
        self._close(v)

    def _pad(self, n, normal, fill):
        if fill is not None:
            return bytes([fill & 0xff]) * n
        if self.sec.exec:
            return _nop_fill(n, normal)
        return bytes(n)

    def label(self, name):
        s = self.symbol(name)
        if s.section is not None or s.common is not None:
            raise AssemblerError('symbol %s is already defined' % name)
        s.section = self.sec
        st = self.st
        if self.sec.nobits:
            s.value = self.sec.size
        else:
            st.labels[s] = (len(st.frags), len(st.cur))

    # driver

    def run(self, text):
        text = _arm._strip_comments(text)
        insns = _INSNS
        sym_re = _arm._SYM_RE
        for self.lineno, line in enumerate(text.split('\n'), 1):
            if '#' in line or ';' in line:
                stmts = _statements(line)
            else:
                stmts = (line,)
            for line in stmts:
                line = line.strip()
                if not line:
                    continue
                try:
                    while True:
                        if line[0] == '"':
                            name, j = _arm._parse_sym(line, 0)
                        else:
                            m = sym_re.match(line)
                            if not m:
                                break
                            name, j = m.group(0), m.end()
                        k = j
                        while k < len(line) and line[k] in ' \t':
                            k += 1
                        if k < len(line) and line[k] == ':':
                            self.label(name)
                            line = line[k + 1:].strip()
                            if not line:
                                break
                        else:
                            break
                    if not line:
                        continue
                    parts = line.split(None, 1)
                    mn = parts[0].lower()
                    rest = parts[1] if len(parts) > 1 else ''
                    if mn[0] == '.':
                        self.directive(mn, rest)
                        continue
                    if mn == 'lock':
                        parts = rest.split(None, 1)
                        if not parts:  # a stand-alone prefix (`lock; insn`), as GNU as
                            self.put(b'\xf0', ())
                            self.st.normal = False
                            continue
                        mn = parts[0].lower()
                        rest = parts[1] if len(parts) > 1 else ''
                        self.lock = True
                    if mn in ('{vex}', '{vex3}', '{evex}'):  # GNU as's pseudo-prefixes for the encoding
                        parts = rest.split(None, 1)
                        if not parts:
                            raise AssemblerError('%s without an instruction' % mn)
                        self.pref = mn[1:-1].rstrip('3')
                        mn = parts[0].lower()
                        rest = parts[1] if len(parts) > 1 else ''
                    ent = insns.get(mn)
                    if ent is None:
                        raise AssemblerError('unsupported instruction %r' % parts[0])
                    h, sz, arg = ent
                    self.mnemonic = mn
                    strs = _split(rest) if rest else []
                    if strs and '{' in strs[-1]:
                        _decorations(self, strs)
                    ops = [_operand(o) for o in strs]
                    if self.lock and not any(o[0] == 'm' for o in ops):
                        raise AssemblerError('lock needs a memory destination')
                    if (self.mask != (0, 0) or self.pref == 'evex') and h not in (_h_evex, _h_avx, _h_avx512, _h_vfma):
                        raise AssemblerError('not an AVX-512 instruction')
                    if self.pref == 'vex' and h is not _h_evex and h is not _h_avx:
                        raise AssemblerError('{vex} on a non-VEX instruction')
                    h(self, ops, sz, arg)
                    self.lock = False
                    self.mask = (0, 0)
                    self.pref = None
                except AssemblerError as e:
                    self.lock = False
                    self.mask = (0, 0)
                    self.pref = None
                    raise AssemblerError('line %d: %s: %s' % (self.lineno, e, line)) from None
        self.resolve()
        return self.obj

    def directive(self, d, rest):
        if d in ('.word', '.value', '.short', '.hword', '.2byte', '.long', '.int', '.4byte',
                 '.quad', '.8byte', '.byte'):
            size = {'.byte': 1, '.long': 4, '.int': 4, '.4byte': 4, '.quad': 8,
                    '.8byte': 8}.get(d, 2)
            for e in _split(rest):
                self.data(size, e)
        elif d in ('.balign', '.p2align', '.align'):
            args = _split(rest)
            n = _int_expr(args[0])
            if d == '.p2align':
                n = 1 << n
            fill = None
            if len(args) > 1 and args[1]:
                fill = _int_expr(args[1])
            maxskip = _int_expr(args[2]) if len(args) > 2 and args[2] else None
            self.align(n, maxskip, fill)
        elif d in ('.xword', '.dword'):
            raise AssemblerError('unsupported directive %s' % d)
        else:
            _arm._Asm.directive(self, d, rest)

    def data(self, size, e):
        sym, v, mod = _eval(e)
        if mod is not None:
            raise AssemblerError('relocation modifier in data')
        if self.sec.nobits:
            if sym is not None or v:
                raise AssemblerError('non-zero data in nobits section %s' % self.sec.name)
            self.sec.size += size
            return
        st = self.st
        if sym is None:
            if size < 8 and not -(1 << (8 * size)) < v < 1 << (8 * size):
                raise AssemblerError('value %s does not fit in %d bytes' % (e, size))
            st.cur += (v & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')
        else:
            kind = {8: 'ABS64', 4: 'ABS32', 2: 'ABS16', 1: 'ABS8'}[size]
            self.fixups.append((st, len(st.frags), len(st.cur), kind, sym, v, self.lineno))
            st.cur += bytes(size)
        st.normal = False

    # layout

    def _layout(self, st):
        """Fix the size of every relaxable branch (rel8 when the target is a label of this
        section in reach, as GNU as, iterating since sizes only grow) and every alignment."""
        frags = st.frags
        frags.append((st.cur, None))
        st.cur = None
        labels = st.labels
        syms = self.obj.symbols
        branches = []
        for _, v in frags:
            if v is not None and v.cc is not None:
                s = syms.get(v.sym)
                if s is None or s not in labels:
                    v.long = True
                else:
                    branches.append(v)
        addrs = [0] * len(frags)
        while True:
            addr = 0
            for i, (fixed, v) in enumerate(frags):
                addrs[i] = addr
                addr += len(fixed)
                if v is not None:
                    v.addr = addr
                    if v.cc is None:
                        pad = -addr % v.align
                        if v.maxskip is not None and pad > v.maxskip:
                            pad = 0
                        v.size = pad
                    elif v.long:
                        v.size = 5 if v.cc < 0 else 6
                    else:
                        v.size = 2
                    addr += v.size
            changed = False
            for v in branches:
                if not v.long:
                    fi, off = labels[syms[v.sym]]
                    d = addrs[fi] + off + v.add - v.addr - 2
                    if not -128 <= d <= 127:
                        v.long = True
                        changed = True
            if not changed:
                break
        st.addrs = addrs
        for s, (fi, off) in labels.items():
            s.value = addrs[fi] + off
        out = bytearray()
        sec = st.sec
        for fixed, v in frags:
            out += fixed
            if v is None:
                continue
            if v.cc is None:
                if v.size:
                    out += self._pad_at(sec, v)
                continue
            s = syms.get(v.sym)
            if s is None:
                s = syms[v.sym] = Symbol(v.sym)
            if not v.long:
                out += bytes([0xeb if v.cc < 0 else 0x70 | v.cc,
                              (s.value + v.add - v.addr - 2) & 0xff])
                continue
            op = b'\xe9' if v.cc < 0 else bytes([0x0f, 0x80 | v.cc])
            out += op
            field = v.addr + len(op)
            if s in labels:
                d = s.value + v.add - field - 4
                out += (d & 0xffffffff).to_bytes(4, 'little')
            else:
                out += bytes(4)
                plt = v.add == 0 and (s.section is None or s.globl)
                sec.relocs.append(Reloc(field, 'PLT32' if plt else 'PC32', v.sym, v.add - 4))
        sec.data = out
        sec.size = len(out)

    def _pad_at(self, sec, v):
        if v.fill is not None:
            return bytes([v.fill & 0xff]) * v.size
        if sec.exec:
            return _nop_fill(v.size, v.normal)
        return bytes(v.size)

    def resolve(self):
        for sec in self.obj.sections:
            if not sec.nobits:
                self._layout(self.states[sec])
        syms = self.obj.symbols
        for st, fi, off, kind, name, add, lineno in self.fixups:
            sec = st.sec
            p = st.addrs[fi] + off
            s = syms.get(name)
            if s is None:
                s = syms[name] = Symbol(name)
            if kind == 'BRANCH':  # call: GNU as's need_plt32_p
                kind = 'PLT32' if add == -4 and (s.section is None or s.globl) else 'PC32'
            if kind == 'PC32' and s.section is sec and not s.globl:
                d = s.value + add - p
                if not -0x80000000 <= d <= 0x7fffffff:
                    raise AssemblerError('line %d: pc-relative reference out of range: %s'
                                         % (lineno, name))
                struct.pack_into('<i', sec.data, p, d)
            else:
                sec.relocs.append(Reloc(p, kind, name, add))
        for sec in self.obj.sections:
            sec.relocs.sort(key=lambda r: r.offset)


def _int_expr(s):
    sym, v, mod = _eval(s)
    if sym is not None or mod is not None:
        raise AssemblerError('expected a constant, got %r' % s)
    return v


def _statements(line):
    """Cut a '#' comment and split on ';' (outside string literals)."""
    out = []
    q = False
    start = 0
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if q:
            if c == '\\':
                i += 1
            elif c == '"':
                q = False
        elif c == '"':
            q = True
        elif c == '#':
            n = i
            break
        elif c == ';':
            out.append(line[start:i])
            start = i + 1
        i += 1
    out.append(line[start:n])
    return out


def assemble(text, syntax='elf'):
    """Assemble x86-64 AT&T-syntax assembly text (QBE targets amd64_sysv, amd64_win) into an
    Object."""
    return _Asm(syntax).run(text)
