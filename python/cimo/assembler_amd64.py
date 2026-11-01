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


def _enc(a, opc, reg, rm, size=0, mand=b'', imm=None, w=None, dabs='ABS32S'):
    """Emit one instruction: legacy prefixes, REX, opcode `opc` (bytes), ModRM with `reg` (an
    int /digit or a register operand) and `rm` (a register or memory operand), SIB, displacement,
    and immediate `imm` = (nbytes, sym, value, reloc kind) or None. `size` 2 adds the operand-size
    prefix, 8 sets REX.W (unless `w` says otherwise); `mand` is a mandatory SSE prefix; `dabs` is
    the relocation for an absolute symbolic displacement."""
    out = bytearray()
    rex = 8 if (size == 8 if w is None else w) else 0
    force = 0
    if reg.__class__ is tuple:
        force = reg[3]
        reg = reg[2]
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
                elif -128 <= disp <= 127:
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
            if disp <= 0xffffffff and base is None and index is None:
                disp -= 1 << 32
            else:
                raise AssemblerError('displacement %d out of range' % disp)
    else:
        force |= rm[3]
        n = rm[2]
        if n & 8:
            rex |= 1
        modrm = 0xc0 | reg | (n & 7)
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
        out.append(disp & 0xff)
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
                    ('cvtpd2ps', b'\x66', 0x5a)):
    _INSNS[_n] = (_h_sse, 0, (_p, bytes([0x0f, _op])))
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
                    ent = insns.get(mn)
                    if ent is None:
                        raise AssemblerError('unsupported instruction %r' % parts[0])
                    h, sz, arg = ent
                    ops = [_operand(o) for o in _split(rest)] if rest else []
                    if self.lock and not any(o[0] == 'm' for o in ops):
                        raise AssemblerError('lock needs a memory destination')
                    h(self, ops, sz, arg)
                    self.lock = False
                except AssemblerError as e:
                    self.lock = False
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
