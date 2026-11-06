"""arm64 assembler for the GNU-syntax assembly that `cimo.backend` emits.

    obj = assemble(text, syntax='elf')      # target arm64
    obj = assemble(text, syntax='macho')    # target arm64_apple ('_' prefixes, @PAGE/@PAGEOFF)

The result is an in-memory object (`Object`): sections with their bytes, symbols and
relocations, which `cimo.loader` links into the running process. It accepts exactly the
instructions, operand forms and directives QBE's arm64 emitter produces (python/cimo/backend/arm64/
emit.py, python/cimo/backend/emit.py) and raises `AssemblerError` for anything else.

Encodings, instruction selection for aliases (`mov` immediate: movz, then movn, then orr;
`ldr`/`str` offsets that don't fit the scaled form become ldur/stur; negative add/sub immediates
flip the operation), which branches are resolved at assembly time (to non-global labels in the
same section) and which relocations are emitted all follow GNU as, and
python/tests/test_assembler.py checks the result byte for byte against it.

Relocation types (named as in the ELF arm64 ABI, for both syntaxes):
CALL26, JUMP26, CONDBR19, ADR_PREL_PG_HI21, ADD_ABS_LO12_NC, LDST{8,16,32,64,128}_ABS_LO12_NC,
ADR_GOT_PAGE, LD64_GOT_LO12_NC, ABS64, ABS32, ABS16. Thread-local storage is rejected.
"""

import re
import struct

__all__ = ['assemble', 'AssemblerError', 'Object', 'Section', 'Symbol', 'Reloc']

M64 = (1 << 64) - 1


class AssemblerError(Exception):
    pass


class Reloc:
    __slots__ = ('offset', 'type', 'sym', 'addend')

    def __init__(self, offset, type, sym, addend):
        self.offset = offset
        self.type = type
        self.sym = sym
        self.addend = addend

    def __repr__(self):
        return 'Reloc(%#x, %s, %s%+d)' % (self.offset, self.type, self.sym, self.addend)


class Section:
    """`data` holds the contents; for a nobits section (bss) it stays empty and `size` counts."""

    __slots__ = ('name', 'exec', 'write', 'alloc', 'nobits', 'align', 'data', 'size', 'relocs')

    def __init__(self, name, exec=False, write=False, alloc=True, nobits=False):
        self.name = name
        self.exec = exec
        self.write = write
        self.alloc = alloc
        self.nobits = nobits
        self.align = 1
        self.data = bytearray()
        self.size = 0
        self.relocs = []

    def __repr__(self):
        return '<Section %s size=%d>' % (self.name, self.size)


class Symbol:
    """A defined symbol has a `section`; a common symbol has `common` = (size, align);
    otherwise it is undefined (external)."""

    __slots__ = ('name', 'section', 'value', 'globl', 'common')

    def __init__(self, name):
        self.name = name
        self.section = None
        self.value = 0
        self.globl = False
        self.common = None

    @property
    def defined(self):
        return self.section is not None

    def __repr__(self):
        where = self.section.name if self.section else ('common' if self.common else 'undef')
        return '<Symbol %s %s+%#x%s>' % (self.name, where, self.value,
                                         ' global' if self.globl else '')


class Object:
    """sections: list of Section in creation order; symbols: dict name -> Symbol.
    prefix: the target's symbol prefix ('_' on Mach-O), so C name = symbol name minus prefix."""

    def __init__(self, syntax):
        self.syntax = syntax
        self.prefix = '_' if syntax == 'macho' else ''
        self.sections = []
        self.symbols = {}

    def section(self, name):
        for s in self.sections:
            if s.name == name:
                return s
        return None

    def __repr__(self):
        return '<Object %s: %s>' % (self.syntax, ', '.join(s.name for s in self.sections))


# ---------------------------------------------------------------------------------------------
# operands

_REG = {}
for _n in range(31):
    _REG['w%d' % _n] = ('w', _n, False)
    _REG['x%d' % _n] = ('x', _n, False)
for _n in range(32):
    for _c in 'bhsdq':
        _REG['%s%d' % (_c, _n)] = (_c, _n, False)
_REG.update({'wzr': ('w', 31, False), 'xzr': ('x', 31, False), 'wsp': ('w', 31, True),
             'sp': ('x', 31, True), 'fp': ('x', 29, False), 'lr': ('x', 30, False)})

_COND = {'eq': 0, 'ne': 1, 'cs': 2, 'hs': 2, 'cc': 3, 'lo': 3, 'mi': 4, 'pl': 5, 'vs': 6,
         'vc': 7, 'hi': 8, 'ls': 9, 'ge': 10, 'lt': 11, 'gt': 12, 'le': 13, 'al': 14, 'nv': 15}

_SYM_RE = re.compile(r'[A-Za-z_.$][\w.$]*')
_INT_RE = re.compile(r'0[xX][0-9a-fA-F]+|0[bB][01]+|[0-9]+')
_TLS_MODS = ('tprel', 'tlsdesc', 'tlsgd', 'tlsle', 'tlsie', 'gottprel', 'dtprel', 'tlvp')


def _int(s):
    """GNU as integer literal: 0x hex, 0b binary, leading-0 octal, else decimal."""
    m = _INT_RE.fullmatch(s)
    if not m:
        raise AssemblerError('bad number %r' % s)
    if s[:2] in ('0x', '0X'):
        return int(s[2:], 16)
    if s[:2] in ('0b', '0B'):
        return int(s[2:], 2)
    if len(s) > 1 and s[0] == '0':
        return int(s, 8)
    return int(s)


def _parse_sym(s, i):
    """Symbol name at s[i:] (bare or "quoted"); returns (name, new index)."""
    if s[i] == '"':
        j = i + 1
        out = []
        while j < len(s) and s[j] != '"':
            if s[j] == '\\' and j + 1 < len(s):
                j += 1
            out.append(s[j])
            j += 1
        if j >= len(s):
            raise AssemblerError('unterminated quoted symbol')
        return ''.join(out), j + 1
    m = _SYM_RE.match(s, i)
    if not m:
        raise AssemblerError('expected a symbol in %r' % s)
    return m.group(0), m.end()


def _expr(s):
    """sym, sym+N, sym-N, sym+-N, N, -N, and the relocation modifiers ':mod:expr' (ELF) or
    'sym@mod' / 'sym@mod+N' / 'sym+N@mod' (Mach-O). Returns (symbol or None, addend, modifier)."""
    s = s.strip()
    mod = None
    if s.startswith(':'):
        j = s.find(':', 1)
        if j < 0:
            raise AssemblerError('bad relocation modifier in %r' % s)
        mod = s[1:j].lower()
        s = s[j + 1:].strip()
    sym = None
    add = 0
    i = 0
    n = len(s)
    sign = 1
    first = True
    while i < n:
        c = s[i]
        if c in ' \t':
            i += 1
            continue
        if c in '+-':
            if c == '-':
                sign = -sign
            i += 1
            continue
        if c == '@':
            m = _SYM_RE.match(s, i + 1)
            if not m or mod is not None:
                raise AssemblerError('bad relocation modifier in %r' % s)
            mod = m.group(0).lower()
            i = m.end()
            continue
        if c.isdigit():
            m = _INT_RE.match(s, i)
            add += sign * _int(m.group(0))
            i = m.end()
        elif c == '"' or c == '_' or c == '.' or c == '$' or c.isalpha():
            if sym is not None or sign < 0:
                raise AssemblerError('unsupported expression %r' % s)
            sym, i = _parse_sym(s, i)
        else:
            raise AssemblerError('bad expression %r' % s)
        sign = 1
        first = False
        # the next term needs an operator
        while i < n and s[i] in ' \t':
            i += 1
        if i < n and s[i] not in '+-@':
            raise AssemblerError('bad expression %r' % s)
    if first:
        raise AssemblerError('empty expression')
    if mod is not None and any(t in mod for t in _TLS_MODS):
        raise AssemblerError('thread-local storage is not supported')
    return sym, add, mod


def _split(s):
    """Split operands on top-level commas (outside [] and quotes)."""
    out = []
    depth = 0
    q = False
    start = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if q:
            if c == '\\':
                i += 1
            elif c == '"':
                q = False
        elif c == '"':
            q = True
        elif c in '[{':
            depth += 1
        elif c in ']}':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(s[start:i].strip())
            start = i + 1
        i += 1
    last = s[start:].strip()
    if last or out:
        out.append(last)
    return out


def _reg(s, classes='wx', sp=False, zr=True):
    r = _REG.get(s.strip().lower())
    if r is None or r[0] not in classes:
        raise AssemblerError('expected %s register, got %r'
                       % ('/'.join(classes), s))
    if r[2] and not sp:
        raise AssemblerError('%s not allowed here' % s)
    if not r[2] and r[1] == 31 and r[0] in 'wx' and not zr:
        raise AssemblerError('%s not allowed here' % s)
    return r


def _isreg(s):
    return s.strip().lower() in _REG


def _imm(s, hash_required=False):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:].strip()
    elif hash_required:
        raise AssemblerError('expected immediate, got %r' % s)
    neg = s.startswith('-')
    if neg or s.startswith('+'):
        s = s[1:].strip()
    v = _int(s)
    return -v if neg else v


def _isimm(s):
    s = s.strip()
    if s.startswith('#'):
        s = s[1:].strip()
        if s.startswith(':'):
            return False
    if s[:1] in '+-':
        s = s[1:].strip()
    return bool(_INT_RE.fullmatch(s))


def _lsl12(ops, i):
    """Optional trailing 'lsl #0' / 'lsl #12' shift for add/sub immediates."""
    if len(ops) <= i:
        return 0
    if len(ops) > i + 1:
        raise AssemblerError('too many operands')
    m = re.fullmatch(r'lsl\s*#?\s*(\d+)', ops[i].strip().lower())
    if not m or m.group(1) not in ('0', '12'):
        raise AssemblerError('expected "lsl #12", got %r' % ops[i])
    return 1 if m.group(1) == '12' else 0


def _bitmask(imm, width):
    """Logical-immediate encoding (N, immr, imms) of imm as a `width`-bit value, or None."""
    mask = (1 << width) - 1
    imm &= mask
    if imm == 0 or imm == mask:
        return None
    size = width
    while size > 2:
        half = size // 2
        hm = (1 << half) - 1
        if (imm & hm) != ((imm >> half) & hm):
            break
        size = half
    em = (1 << size) - 1
    elt = imm & em
    ones = bin(elt).count('1')
    run = (1 << ones) - 1
    for r in range(size):
        if ((elt >> r) | (elt << (size - r))) & em == run:
            immr = (size - r) % size
            imms = (~(2 * size - 1) & 0x3f) | (ones - 1)
            return (1 if size == 64 else 0), immr, imms
    return None


def _logimm(value, width):
    """GNU as: the bits above a 32-bit operation's width must be all zeros or all ones."""
    if not -(1 << 64) < value < 1 << 64:
        return None
    value &= M64
    if width == 32:
        hi = value >> 32
        if hi not in (0, 0xffffffff):
            return None
    return _bitmask(value, width)


def _wide(value, is32):
    """(imm16, shift) if value fits a single movz, else None (GNU aarch64_wide_constant_p)."""
    if not -(1 << 64) < value < 1 << 64:
        return None
    value &= M64
    if is32:
        if value >> 32 not in (0, 0xffffffff):
            return None
        value &= 0xffffffff
    for sh in ((0, 16) if is32 else (0, 16, 32, 48)):
        if value & (0xffff << sh) == value:
            return value >> sh, sh
    return None


# ---------------------------------------------------------------------------------------------
# instruction encoders: each takes (asm, mnemonic, operand strings) and emits one word


def _sf(r):
    return (1 << 31) if r[0] == 'x' else 0


def _same(*regs):
    k = regs[0][0]
    for r in regs[1:]:
        if r[0] != k:
            raise AssemblerError('operand size mismatch')


_SHIFTOP = {'lsl': 0, 'lsr': 1, 'asr': 2, 'ror': 3}
_EXTOP = {'uxtw': 2, 'sxtw': 6}


def _shiftop(rest):
    """('lsl', 3) from ['lsl #3'] or ('sxtw', 0) from ['sxtw']."""
    m = re.fullmatch(r'(lsl|lsr|asr|ror|sxtw|uxtw)(?:\s*#?\s*(\d+))?', rest[0].strip().lower())
    if m is None or len(rest) != 1:
        raise AssemblerError('bad shift %r' % rest[0])
    return m.group(1), int(m.group(2) or 0)


_ADDSUB = {'add': (0, 0), 'adds': (0, 1), 'sub': (1, 0), 'subs': (1, 1)}


def _notzr(r):
    """r isn't xzr/wzr: for a field in which 31 means sp, where zr would silently become sp."""
    if not r[2] and r[1] == 31:
        raise AssemblerError('the zero register is not allowed here (31 would be sp)')


def _addsub(a, op, s, rd, rn, ops3, ops_rest):
    """ops3: the third operand string; ops_rest: [optional shift operand]."""
    if _isreg(ops3):
        rm = _reg(ops3)
        if ops_rest:
            how, n = _shiftop(ops_rest)
            if how in _SHIFTOP:
                if how == 'ror' or rd[2] or rn[2]:
                    raise AssemblerError('bad shift %r' % ops_rest[0])
                _same(rd, rn, rm)
                if not 0 <= n < (64 if rd[0] == 'x' else 32):
                    raise AssemblerError('shift amount out of range')
                return (0x0b000000 | _sf(rd) | op << 30 | s << 29 | _SHIFTOP[how] << 22
                        | rm[1] << 16 | n << 10 | rn[1] << 5 | rd[1])
            if how not in _EXTOP or rm[0] != 'w' or rd[0] != 'x' or rn[0] != 'x' or not 0 <= n <= 4:
                raise AssemblerError('bad extended register %r' % ops_rest[0])
            _notzr(rn)  # field 31 is sp here, in Rn always and in Rd unless the flags are set
            if not s:
                _notzr(rd)
            return (0x0b200000 | _sf(rd) | op << 30 | s << 29 | rm[1] << 16 | _EXTOP[how] << 13
                    | n << 10 | rn[1] << 5 | rd[1])
        if rd[2] or rn[2]:
            # extended register form (needed when sp is involved): UXTX / UXTW
            _same(rd, rn)
            _notzr(rn)
            if not s:
                _notzr(rd)
            opt = 3 if rd[0] == 'x' else 2
            if rm[0] != rd[0]:
                raise AssemblerError('operand size mismatch')
            if s and rd[2]:
                raise AssemblerError('sp not allowed as destination of %s' % ('subs' if op else 'adds'))
            return (0x0b200000 | _sf(rd) | op << 30 | s << 29 | rm[1] << 16 | opt << 13
                    | rn[1] << 5 | rd[1])
        _same(rd, rn, rm)
        return 0x0b000000 | _sf(rd) | op << 30 | s << 29 | rm[1] << 16 | rn[1] << 5 | rd[1]
    # immediate forms: register 31 is sp for Rn (and for Rd unless flag-setting)
    _same(rd, rn)
    if not rn[2] and rn[1] == 31:
        raise AssemblerError('zr not allowed as source of an immediate add/sub')
    if s and rd[2]:
        raise AssemblerError('sp not allowed here')
    if not s and not rd[2] and rd[1] == 31:
        raise AssemblerError('zr not allowed as destination of an immediate add/sub')
    t = ops3.strip()
    if _isimm(t):
        v = _imm(t)
        sh = _lsl12(ops_rest, 0)
        if ops_rest:
            if not 0 <= v <= 0xfff:
                raise AssemblerError('immediate out of range')
            imm = v
        else:
            if v < 0:
                op ^= 1
                v = -v
            if v <= 0xfff:
                imm, sh = v, 0
            elif v & 0xfff == 0 and v >> 12 <= 0xfff:
                imm, sh = v >> 12, 1
            else:
                raise AssemblerError('immediate out of range')
        return 0x11000000 | _sf(rd) | op << 30 | s << 29 | sh << 22 | imm << 10 | rn[1] << 5 | rd[1]
    # relocated immediate: add Rd, Rn, #:lo12:sym / sym@PAGEOFF
    if t.startswith('#'):
        t = t[1:]
    sym, add, mod = _expr(t)
    if op or s or ops_rest or mod not in ('lo12', 'pageoff') or sym is None:
        raise AssemblerError('unsupported add/sub operand %r' % ops3)
    a.fixup('ADD_ABS_LO12_NC', sym, add)
    return 0x11000000 | _sf(rd) | rn[1] << 5 | rd[1]


def _i_addsub(a, mn, ops):
    op, s = _ADDSUB[mn]
    if len(ops) < 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rd = _reg(ops[0], sp=not s)
    rn = _reg(ops[1], sp=True)
    return _addsub(a, op, s, rd, rn, ops[2], ops[3:])


def _i_cmp(a, mn, ops):
    if len(ops) < 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rn = _reg(ops[0], sp=True)
    zr = ('w' if rn[0] == 'w' else 'x', 31, False)
    return _addsub(a, 1 if mn == 'cmp' else 0, 1, zr, rn, ops[1], ops[2:])


def _i_neg(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('neg needs 2 operands')
    rd, rm = _reg(ops[0]), _reg(ops[1])
    _same(rd, rm)
    return 0x4b0003e0 | _sf(rd) | rm[1] << 16 | rd[1]


_LOGIC = {'and': (0x0a000000, 0x12000000), 'orr': (0x2a000000, 0x32000000),
          'eor': (0x4a000000, 0x52000000), 'ands': (0x6a000000, 0x72000000),
          'bic': (0x0a200000, None), 'orn': (0x2a200000, None), 'eon': (0x4a200000, None)}


def _i_logic(a, mn, ops):
    if len(ops) not in (3, 4):
        raise AssemblerError('%s needs 3 register/immediate operands' % mn)
    reg_op, imm_op = _LOGIC[mn]
    rn = _reg(ops[1])
    if _isreg(ops[2]):
        rd, rm = _reg(ops[0]), _reg(ops[2])
        _same(rd, rn, rm)
        how, n = _shiftop(ops[3:]) if len(ops) == 4 else ('lsl', 0)
        if how not in _SHIFTOP or not 0 <= n < (64 if rd[0] == 'x' else 32):
            raise AssemblerError('bad shift %r' % ops[3])
        return (reg_op | _sf(rd) | _SHIFTOP[how] << 22 | rm[1] << 16 | n << 10 | rn[1] << 5
                | rd[1])
    if imm_op is None or len(ops) != 3:
        raise AssemblerError('%s takes registers' % mn)
    rd = _reg(ops[0], sp=mn != 'ands', zr=mn == 'ands')
    _same(rd, rn)
    width = 64 if rd[0] == 'x' else 32
    enc = _logimm(_imm(ops[2], True), width)
    if enc is None:
        raise AssemblerError('immediate %s is not a valid logical immediate' % ops[2])
    n, immr, imms = enc
    return imm_op | _sf(rd) | n << 22 | immr << 16 | imms << 10 | rn[1] << 5 | rd[1]


_SHIFT = {'lsl': 0x1ac02000, 'lsr': 0x1ac02400, 'asr': 0x1ac02800}


def _i_shift(a, mn, ops):
    if len(ops) != 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    if _isreg(ops[2]):
        rm = _reg(ops[2])
        _same(rd, rm)
        return _SHIFT[mn] | _sf(rd) | rm[1] << 16 | rn[1] << 5 | rd[1]
    w = 64 if rd[0] == 'x' else 32
    sh = _imm(ops[2], True)
    if not 0 <= sh < w:
        raise AssemblerError('shift amount out of range')
    if mn == 'lsl':
        immr, imms, base = (-sh) % w, w - 1 - sh, 0x53000000
    else:
        immr, imms, base = sh, w - 1, (0x53000000 if mn == 'lsr' else 0x13000000)
    n = 1 if w == 64 else 0
    return base | _sf(rd) | n << 22 | immr << 16 | imms << 10 | rn[1] << 5 | rd[1]


def _i_bitfield(a, mn, ops):
    """ubfx, sbfx rd, rn, #lsb, #width."""
    if len(ops) != 4:
        raise AssemblerError('%s needs 4 operands' % mn)
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    w = 64 if rd[0] == 'x' else 32
    lsb, width = _imm(ops[2], True), _imm(ops[3], True)
    if not (0 <= lsb < w and 1 <= width <= w - lsb):
        raise AssemblerError('bad bit field in %s' % mn)
    base = 0x53000000 if mn == 'ubfx' else 0x13000000
    return (base | _sf(rd) | (w == 64) << 22 | lsb << 16 | (lsb + width - 1) << 10 | rn[1] << 5
            | rd[1])


def _i_bfm(a, mn, ops):
    """bfi, bfxil rd, rn, #lsb, #width."""
    if len(ops) != 4:
        raise AssemblerError('%s needs 4 operands' % mn)
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    w = 64 if rd[0] == 'x' else 32
    lsb, width = _imm(ops[2], True), _imm(ops[3], True)
    if not (0 <= lsb < w and 1 <= width <= w - lsb):
        raise AssemblerError('bad bit field in %s' % mn)
    if mn == 'bfi':
        immr, imms = -lsb % w, width - 1
    else:
        immr, imms = lsb, lsb + width - 1
    return 0x33000000 | _sf(rd) | (w == 64) << 22 | immr << 16 | imms << 10 | rn[1] << 5 | rd[1]


def _i_ubfiz(a, mn, ops):
    """ubfiz rd, rn, #lsb, #width."""
    if len(ops) != 4:
        raise AssemblerError('ubfiz needs 4 operands')
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    w = 64 if rd[0] == 'x' else 32
    lsb, width = _imm(ops[2], True), _imm(ops[3], True)
    if not (0 <= lsb < w and 1 <= width <= w - lsb):
        raise AssemblerError('bad bit field in ubfiz')
    return (0x53000000 | _sf(rd) | (w == 64) << 22 | ((-lsb) % w) << 16 | (width - 1) << 10
            | rn[1] << 5 | rd[1])


def _i_ror(a, mn, ops):
    """ror rd, rn, #n: extr rd, rn, rn, #n."""
    if len(ops) != 3:
        raise AssemblerError('ror needs 3 operands')
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    n = _imm(ops[2], True)
    if not 0 <= n < (64 if rd[0] == 'x' else 32):
        raise AssemblerError('ror amount out of range')
    return (0x13800000 | _sf(rd) | (rd[0] == 'x') << 22 | rn[1] << 16 | n << 10 | rn[1] << 5
            | rd[1])


def _i_dp2(a, mn, ops):
    base = {'sdiv': 0x1ac00c00, 'udiv': 0x1ac00800, 'mul': 0x1b007c00}[mn]
    if len(ops) != 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rd, rn, rm = _reg(ops[0]), _reg(ops[1]), _reg(ops[2])
    _same(rd, rn, rm)
    return base | _sf(rd) | rm[1] << 16 | rn[1] << 5 | rd[1]


_WIDEMUL = {'smaddl': 0x9b200000, 'smsubl': 0x9b208000, 'umaddl': 0x9ba00000, 'umsubl': 0x9ba08000}


def _i_widemul(a, mn, ops):
    """smull, umull rd, wn, wm; smaddl, smsubl, umaddl, umsubl rd, wn, wm, xa."""
    rd, rn, rm = _reg(ops[0], 'x'), _reg(ops[1], 'w'), _reg(ops[2], 'w')
    if mn in ('smull', 'umull'):
        if len(ops) != 3:
            raise AssemblerError('%s needs 3 operands' % mn)
        ra = 31
        base = _WIDEMUL['smaddl' if mn == 'smull' else 'umaddl']
    else:
        if len(ops) != 4:
            raise AssemblerError('%s needs 4 operands' % mn)
        ra = _reg(ops[3], 'x')[1]
        base = _WIDEMUL[mn]
    return base | rm[1] << 16 | ra << 10 | rn[1] << 5 | rd[1]


def _i_madd(a, mn, ops):
    if len(ops) != 4:
        raise AssemblerError('%s needs 4 operands' % mn)
    rd, rn, rm, ra = [_reg(o) for o in ops]
    _same(rd, rn, rm, ra)
    base = 0x1b008000 if mn == 'msub' else 0x1b000000
    return base | _sf(rd) | rm[1] << 16 | ra[1] << 10 | rn[1] << 5 | rd[1]


_EXT = {'sxtb': (0x13000000, 7, 'wx'), 'sxth': (0x13000000, 15, 'wx'),
        'sxtw': (0x13000000, 31, 'x'), 'uxtb': (0x53000000, 7, 'w'),
        'uxth': (0x53000000, 15, 'w')}


def _i_ext(a, mn, ops):
    base, imms, dcls = _EXT[mn]
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rd, rn = _reg(ops[0], dcls), _reg(ops[1], 'w')
    n = 1 if rd[0] == 'x' else 0
    return base | _sf(rd) | n << 22 | imms << 10 | rn[1] << 5 | rd[1]


def _i_mov(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('mov needs 2 operands')
    if _isreg(ops[1]):
        rd, rm = _reg(ops[0], sp=True), _reg(ops[1], sp=True)
        _same(rd, rm)
        if rd[2] or rm[2]:
            _notzr(rd), _notzr(rm)
            return 0x11000000 | _sf(rd) | rm[1] << 5 | rd[1]  # add Rd, Rn, #0
        return 0x2a0003e0 | _sf(rd) | rm[1] << 16 | rd[1]  # orr Rd, zr, Rm
    rd = _reg(ops[0], sp=True)
    if not _isimm(ops[1]):
        raise AssemblerError('unsupported mov operand %r' % ops[1])
    v = _imm(ops[1])
    if not -(1 << 64) < v < 1 << 64:
        raise AssemblerError('immediate %s does not fit 64 bits' % ops[1])  # GNU as: (-2^64, 2^64), wrapped
    is32 = rd[0] == 'w'
    if not rd[2]:
        w = _wide(v, is32)
        if w is not None:
            return 0x52800000 | _sf(rd) | (w[1] // 16) << 21 | w[0] << 5 | rd[1]
        w = _wide(~v & M64, is32)  # the whole value, so _wide sees bits above a w register's 32
        if w is not None:
            return 0x12800000 | _sf(rd) | (w[1] // 16) << 21 | w[0] << 5 | rd[1]
    _notzr(rd)  # the orr below writes sp for register 31
    enc = _logimm(v, 32 if is32 else 64)
    if enc is None:
        raise AssemblerError('immediate %s cannot be moved by a single instruction' % ops[1])
    n, immr, imms = enc
    return 0x320003e0 | _sf(rd) | n << 22 | immr << 16 | imms << 10 | rd[1]


def _i_movk(a, mn, ops):
    base = {'movk': 0x72800000, 'movz': 0x52800000, 'movn': 0x12800000}[mn]
    if len(ops) not in (2, 3):
        raise AssemblerError('%s needs 2 or 3 operands' % mn)
    rd = _reg(ops[0])
    v = _imm(ops[1], True)
    sh = 0
    if len(ops) == 3:
        m = re.fullmatch(r'lsl\s*#?\s*(\d+)', ops[2].strip().lower())
        if not m:
            raise AssemblerError('expected "lsl #n", got %r' % ops[2])
        sh = int(m.group(1))
    if not 0 <= v <= 0xffff or sh not in ((0, 16) if rd[0] == 'w' else (0, 16, 32, 48)):
        raise AssemblerError('%s immediate out of range' % mn)
    return base | _sf(rd) | (sh // 16) << 21 | v << 5 | rd[1]


def _i_cset(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('cset needs 2 operands')
    rd = _reg(ops[0])
    c = _COND.get(ops[1].strip().lower())
    if c is None or c >= 14:
        raise AssemblerError('bad condition %r' % ops[1])
    return 0x1a9f07e0 | _sf(rd) | (c ^ 1) << 12 | rd[1]


# floating point

def _ftype(r):
    return {'s': 0, 'd': 1}[r[0]]


_FP3 = {'fadd': 0x1e202800, 'fsub': 0x1e203800, 'fmul': 0x1e200800, 'fdiv': 0x1e201800,
        'fmaxnm': 0x1e206800, 'fminnm': 0x1e207800, 'fnmul': 0x1e208800}


def _i_fp3(a, mn, ops):
    if len(ops) != 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rd, rn, rm = [_reg(o, 'sd') for o in ops]
    _same(rd, rn, rm)
    return _FP3[mn] | _ftype(rd) << 22 | rm[1] << 16 | rn[1] << 5 | rd[1]


_FP4 = {'fmadd': 0x1f000000, 'fmsub': 0x1f008000, 'fnmadd': 0x1f200000, 'fnmsub': 0x1f208000}


def _i_fp4(a, mn, ops):  # fmadd Fd, Fn, Fm, Fa
    if len(ops) != 4:
        raise AssemblerError('%s needs 4 operands' % mn)
    rd, rn, rm, ra = [_reg(o, 'sd') for o in ops]
    _same(rd, rn, rm, ra)
    return _FP4[mn] | _ftype(rd) << 22 | rm[1] << 16 | ra[1] << 10 | rn[1] << 5 | rd[1]


# floating-point data-processing (1 source); fneg is QBE's, the rest are cimo's math builtins
_FP1 = {'fneg': 0x1e214000, 'fabs': 0x1e20c000, 'fsqrt': 0x1e21c000, 'frintp': 0x1e24c000,
        'frintm': 0x1e254000, 'frintz': 0x1e25c000, 'frinta': 0x1e264000,
        'frintx': 0x1e274000, 'frinti': 0x1e27c000}


def _i_fneg(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rd, rn = _reg(ops[0], 'sd'), _reg(ops[1], 'sd')
    _same(rd, rn)
    return _FP1[mn] | _ftype(rd) << 22 | rn[1] << 5 | rd[1]


# integer data-processing (1 source): cimo's bit builtins; (32-bit, 64-bit) opcodes
_DP1 = {'clz': (0x5ac01000, 0xdac01000), 'rbit': (0x5ac00000, 0xdac00000),
        'rev16': (0x5ac00400, 0xdac00400), 'rev': (0x5ac00800, 0xdac00c00)}


def _i_dp1(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rd, rn = _reg(ops[0]), _reg(ops[1])
    _same(rd, rn)
    return _DP1[mn][rd[0] == 'x'] | rn[1] << 5 | rd[1]


_VREG8B = re.compile(r'^v([0-9]|[12][0-9]|3[01])\.8b$')
_BREG = re.compile(r'^b([0-9]|[12][0-9]|3[01])$')


def _i_simd(a, mn, ops):  # cnt Vd.8B, Vn.8B / addv Bd, Vn.8B (the popcount builtin)
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    d = (_VREG8B if mn == 'cnt' else _BREG).match(ops[0].lower())
    n = _VREG8B.match(ops[1].lower())
    if not d or not n:
        raise AssemblerError('unsupported operands for %s' % mn)
    return (0x0e205800 if mn == 'cnt' else 0x0e31b800) | int(n.group(1)) << 5 | int(d.group(1))


_VREG8H = re.compile(r'^v([0-9]|[12][0-9]|3[01])\.8h$')
_LIST3 = re.compile(r'^\{\s*v([0-9]|[12][0-9]|3[01])\.8b\s*,\s*v([0-9]|[12][0-9]|3[01])\.8b\s*,'
                    r'\s*v([0-9]|[12][0-9]|3[01])\.8b\s*\}$')


def _i_bytes(a, mn, ops):
    """cimo (vectorize's loops over bytes): ushll Vd.8H, Vn.8B, #0 (uxtl); xtn Vd.8B, Vn.8H; ld3 {Vt.8B, Vt+1.8B, Vt+2.8B},
    [Xn], consecutive registers modulo 32."""
    if mn == 'ld3':
        if len(ops) != 2:
            raise AssemblerError('ld3 needs 2 operands')
        m = _LIST3.match(ops[0].strip().lower())
        t = [int(x) for x in m.groups()] if m else None
        if t is None or t[1] != (t[0] + 1) % 32 or t[2] != (t[0] + 2) % 32:
            raise AssemblerError('ld3 takes three consecutive .8b registers: %r' % ops[0])
        b = ops[1].strip().lower()
        if not (b.startswith('[') and b.endswith(']')):
            raise AssemblerError('ld3 takes [Xn]')
        n = _reg(b[1:-1].strip(), 'x', sp=True, zr=False)[1]
        return 0x0c404000 | n << 5 | t[0]
    if mn == 'ushll':
        if len(ops) != 3 or _imm(ops[2], True) != 0:
            raise AssemblerError('ushll takes Vd.8h, Vn.8b, #0')
        d, n = _VREG8H.match(ops[0].strip().lower()), _VREG8B.match(ops[1].strip().lower())
        if not d or not n:
            raise AssemblerError('unsupported operands for ushll')
        return 0x2f08a400 | int(n.group(1)) << 5 | int(d.group(1))
    if len(ops) != 2:  # xtn
        raise AssemblerError('xtn needs 2 operands')
    d, n = _VREG8B.match(ops[0].strip().lower()), _VREG8H.match(ops[1].strip().lower())
    if not d or not n:
        raise AssemblerError('unsupported operands for xtn')
    return 0x0e212800 | int(n.group(1)) << 5 | int(d.group(1))


def _fpimm8(v):
    """abcdefgh such that v is (-1)^a * (16 + efgh) / 16 * 2^n, with n from bcd: 000 is 1, 011 is
    4, 100 is -3, 111 is 0; or None."""
    for imm in range(256):
        sign = -1.0 if imm & 0x80 else 1.0
        bcd = (imm >> 4) & 7
        n = bcd + 1 if bcd < 4 else bcd - 7
        if sign * (16 + (imm & 15)) / 16 * 2.0 ** n == v:
            return imm
    return None


def _i_fmov(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('fmov needs 2 operands')
    if not _isreg(ops[1]):
        rd = _reg(ops[0], 'sd')
        t = ops[1].strip().lstrip('#')
        try:
            imm = _fpimm8(float(t))
        except ValueError:
            imm = None
        if imm is None:
            raise AssemblerError('fmov immediate %r has no encoding' % t)
        return 0x1e201000 | _ftype(rd) << 22 | imm << 13 | rd[1]
    rd, rn = _reg(ops[0], 'wxsd'), _reg(ops[1], 'wxsd')
    fd, fn = rd[0] in 'sd', rn[0] in 'sd'
    if fd and fn:
        _same(rd, rn)
        return 0x1e204000 | _ftype(rd) << 22 | rn[1] << 5 | rd[1]
    pair = rd[0] + rn[0]
    base = {'ws': 0x1e260000, 'sw': 0x1e270000, 'xd': 0x9e660000, 'dx': 0x9e670000}.get(pair)
    if base is None:
        raise AssemblerError('fmov between %s and %s registers' % (rd[0], rn[0]))
    return base | rn[1] << 5 | rd[1]


def _i_fcvt(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('fcvt needs 2 operands')
    rd, rn = _reg(ops[0], 'sd'), _reg(ops[1], 'sd')
    if rd[0] == rn[0]:
        raise AssemblerError('fcvt needs different precisions')
    return (0x1e22c000 if rd[0] == 'd' else 0x1e624000) | rn[1] << 5 | rd[1]


def _i_fcvtz(a, mn, ops):  # fcvtzs/fcvtzu Rd, Fn
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rd, rn = _reg(ops[0]), _reg(ops[1], 'sd')
    base = 0x1e380000 if mn == 'fcvtzs' else 0x1e390000
    return base | _sf(rd) | _ftype(rn) << 22 | rn[1] << 5 | rd[1]


def _i_cvtf(a, mn, ops):  # scvtf/ucvtf Fd, Rn
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rd, rn = _reg(ops[0], 'sd'), _reg(ops[1])
    base = 0x1e220000 if mn == 'scvtf' else 0x1e230000
    return base | _sf(rn) | _ftype(rd) << 22 | rn[1] << 5 | rd[1]


def _i_fcmp(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    e = 0x10 if mn == 'fcmpe' else 0
    rn = _reg(ops[0], 'sd')
    if _isreg(ops[1]):
        rm = _reg(ops[1], 'sd')
        _same(rn, rm)
        return 0x1e202000 | e | _ftype(rn) << 22 | rm[1] << 16 | rn[1] << 5
    if ops[1].strip().replace(' ', '') not in ('#0', '#0.0'):
        raise AssemblerError('%s can only compare with #0.0' % mn)
    return 0x1e202008 | e | _ftype(rn) << 22 | rn[1] << 5


# loads and stores: (mnemonic, register class) -> (unsigned-offset opcode, log2 access size)
_LDST = {
    ('strb', 'w'): (0x39000000, 0), ('ldrb', 'w'): (0x39400000, 0),
    ('ldrsb', 'x'): (0x39800000, 0), ('ldrsb', 'w'): (0x39c00000, 0),
    ('strh', 'w'): (0x79000000, 1), ('ldrh', 'w'): (0x79400000, 1),
    ('ldrsh', 'x'): (0x79800000, 1), ('ldrsh', 'w'): (0x79c00000, 1),
    ('str', 'w'): (0xb9000000, 2), ('ldr', 'w'): (0xb9400000, 2),
    ('ldrsw', 'x'): (0xb9800000, 2),
    ('str', 'x'): (0xf9000000, 3), ('ldr', 'x'): (0xf9400000, 3),
    ('str', 's'): (0xbd000000, 2), ('ldr', 's'): (0xbd400000, 2),
    ('str', 'd'): (0xfd000000, 3), ('ldr', 'd'): (0xfd400000, 3),
    ('str', 'q'): (0x3d800000, 4), ('ldr', 'q'): (0x3dc00000, 4),
}


def _mem(ops, i):
    """Parse a memory operand at ops[i] (and a post-index immediate at ops[i+1]).
    Returns (base reg, mode, offset, (sym, addend, mod) or None) with mode in
    'off', 'pre', 'post'."""
    s = ops[i].strip()
    if not s.startswith('['):
        raise AssemblerError('expected memory operand, got %r' % s)
    pre = s.endswith('!')
    if pre:
        s = s[:-1].rstrip()
    if not s.endswith(']'):
        raise AssemblerError('bad memory operand %r' % ops[i])
    parts = _split(s[1:-1])
    base = _reg(parts[0], 'x', sp=True, zr=False)
    mode = 'pre' if pre else 'off'
    off = 0
    rel = None
    if len(parts) == 2:
        p = parts[1].strip()
        if _isimm(p):
            off = _imm(p)
        elif _isreg(p):
            raise AssemblerError('register offsets are not supported')
        else:
            rel = _expr(p[1:] if p.startswith('#') else p)
            if pre:
                raise AssemblerError('bad memory operand %r' % ops[i])
    elif len(parts) != 1:
        raise AssemblerError('bad memory operand %r' % ops[i])
    n = len(ops) - i
    if n == 2:
        if pre or len(parts) != 1:
            raise AssemblerError('bad post-indexed operand')
        mode = 'post'
        off = _imm(ops[i + 1])
    elif n != 1:
        raise AssemblerError('too many operands')
    elif pre and len(parts) != 2:
        raise AssemblerError('bad pre-indexed operand')
    return base, mode, off, rel


_LO12 = {0: 'LDST8_ABS_LO12_NC', 1: 'LDST16_ABS_LO12_NC', 2: 'LDST32_ABS_LO12_NC',
         3: 'LDST64_ABS_LO12_NC', 4: 'LDST128_ABS_LO12_NC'}


_REGOFF = re.compile(r'\[\s*([^,\s]+)\s*,\s*([wx]\d+)\s*(?:,\s*(lsl|sxtw|uxtw|sxtx)\s*(?:#?\s*(\d+))?\s*)?\]$')
_OPTION = {'lsl': 3, 'sxtx': 7, 'sxtw': 6, 'uxtw': 2}


def _regoffset(ops, sz):
    """Rm, option, S, Rn fields of a register-offset operand, or None."""
    if len(ops) != 2:
        return None
    m = _REGOFF.match(ops[1].strip().lower())
    if m is None:
        return None
    base = _reg(m.group(1), 'x', sp=True, zr=False)
    rm = _reg(m.group(2))
    how = m.group(3) or 'lsl'
    if (rm[0] == 'w') != (how in ('sxtw', 'uxtw')):
        raise AssemblerError('bad memory operand %r' % ops[1])
    s = 0
    if m.group(4) is not None:
        n = int(m.group(4))
        if n not in (0, sz):
            raise AssemblerError('bad shift in %r' % ops[1])
        s = int(sz == 0 or n == sz)  # an explicit #0 sets S only for byte accesses
    return rm[1] << 16 | _OPTION[how] << 13 | s << 12 | base[1] << 5


# cimo (ext.vector): NEON on 128 bits. (opcode, 's' if the lane size goes in bits 23:22, 'f' if bit 22 says double)
_NEON3 = {'add': (0x4e208400, 's'), 'sub': (0x6e208400, 's'), 'mul': (0x4e209c00, 's'), 'cmeq': (0x6e208c00, 's'),
          'cmgt': (0x4e203400, 's'), 'cmge': (0x4e203c00, 's'), 'cmhi': (0x6e203400, 's'), 'cmhs': (0x6e203c00, 's'),
          'smin': (0x4e206c00, 's'), 'smax': (0x4e206400, 's'), 'umin': (0x6e206c00, 's'), 'umax': (0x6e206400, 's'),
          'fadd': (0x4e20d400, 'f'), 'fsub': (0x4ea0d400, 'f'), 'fmul': (0x6e20dc00, 'f'), 'fdiv': (0x6e20fc00, 'f'),
          'fminnm': (0x4ea0c400, 'f'), 'fmaxnm': (0x4e20c400, 'f'), 'fmla': (0x4e20cc00, 'f'), 'fcmeq': (0x4e20e400, 'f'),
          'fcmge': (0x6e20e400, 'f'), 'fcmgt': (0x6ea0e400, 'f'), 'and': (0x4e201c00, ''), 'orr': (0x4ea01c00, ''),
          'eor': (0x6e201c00, ''), 'bsl': (0x6e601c00, '')}
_NEON2 = {'neg': (0x6e20b800, 's'), 'fneg': (0x6ea0f800, 'f'), 'fabs': (0x4ea0f800, 'f'), 'fsqrt': (0x6ea1f800, 'f'),
          'mvn': (0x6e205800, '')}
_NEONSH = {'shl': 0x4f005400, 'ushr': 0x6f000400, 'sshr': 0x4f000400}
_LANES = {'16b': 0, '8h': 1, '4s': 2, '2d': 3}
_NEONW = {'saddw': 0x0ea01000, 'saddw2': 0x4ea01000, 'uaddw': 0x2ea01000, 'uaddw2': 0x6ea01000}  # 2d, 2d, 2s or 4s
_VREG = re.compile(r'v(\d+)\.(16b|8h|4s|2d|2s)$')
_VNARROW = re.compile(r'v(\d+)\.(8b|4h|2s|16b|8h|4s)$')  # saddw's third operand
_VELEM = re.compile(r'v(\d+)\.([bhsd])\[(\d+)\]$')


def _isneon(ops):
    return bool(ops) and ops[0].strip().lower()[:1] == 'v' and _VREG.match(ops[0].strip().lower()) is not None


def _v(s, lanes=None, pattern=_VREG):
    """(register, arrangement) of a vector operand; lanes: the arrangements allowed."""
    m = pattern.match(s.strip().lower())
    if m is None or int(m[1]) > 31:
        raise AssemblerError('expected a vector register, got %r' % s)
    if lanes is not None and m[2] not in lanes:
        raise AssemblerError('arrangement .%s not allowed here: %r' % (m[2], s))
    return int(m[1]), m[2]


_NO2D = frozenset(('mul', 'smin', 'smax', 'umin', 'umax'))


def _elem(s):
    """(register, element size letter, index) of Vn.T[i], the index in range for the element size."""
    e = _VELEM.match(s.strip().lower())
    if e is None or int(e[1]) > 31:
        raise AssemblerError('expected a vector element, got %r' % s)
    k, i = e[2], int(e[3])
    if i >= {'b': 16, 'h': 8, 's': 4, 'd': 2}[k]:
        raise AssemblerError('lane index out of range: %r' % s)
    return int(e[1]), k, i


def _imm5(k, i):
    return {'b': 1, 'h': 2, 's': 4, 'd': 8}[k] | i << ({'b': 1, 'h': 2, 's': 3, 'd': 4}[k])


def _neon(mn, ops):
    """The word of a NEON instruction whose first operand is a vector register. Arrangements are checked as GNU as
    does: one for all the operands, and one the instruction has (.2s only in saddw and uaddw)."""
    if mn in _NEONW:  # saddw: .8h/.4s/.2d twice, then .8b/.4h/.2s; saddw2: then .16b/.8h/.4s
        d, lane = _v(ops[0], ('8h', '4s', '2d'))
        n, _ = _v(ops[1], (lane,))
        size = {'8h': 0, '4s': 1, '2d': 2}[lane]
        narrow = ('16b', '8h', '4s') if mn.endswith('2') else ('8b', '4h', '2s')
        m, _ = _v(ops[2], (narrow[size],), _VNARROW)
        return (_NEONW[mn] & ~0x00c00000) | size << 22 | m << 16 | n << 5 | d
    if mn == 'dup':
        d, lane = _v(ops[0], _LANES)
        size = _LANES[lane]
        s = ops[1].strip().lower()
        if _VELEM.match(s):
            n, k, i = _elem(s)
            if k != 'bhsd'[size]:
                raise AssemblerError('element size does not match .%s' % lane)
            return 0x4e000400 | _imm5(k, i) << 16 | n << 5 | d
        r = _reg(s, 'x' if size == 3 else 'w')
        return 0x4e000c00 | (1 << size) << 16 | r[1] << 5 | d
    d, lane = _v(ops[0], _LANES)
    size = _LANES[lane]
    same = (lane,)
    if mn == 'mov':
        n, _ = _v(ops[1], ('16b',) if lane == '16b' else same)
        if lane != '16b':
            raise AssemblerError('mov of vectors takes .16b')
        return 0x4ea01c00 | n << 16 | n << 5 | d
    if mn in _NEONSH:
        n, _ = _v(ops[1], same)
        amount = _imm(ops[2], True)
        bits = 8 << size
        if not (0 <= amount < bits if mn == 'shl' else 1 <= amount <= bits):
            raise AssemblerError('shift amount %d out of range for .%s' % (amount, lane))
        immhb = bits + amount if mn == 'shl' else 2 * bits - amount
        return _NEONSH[mn] | immhb << 16 | n << 5 | d
    if mn in _NEON2:
        op, kind = _NEON2[mn]
        _lanes_ok(mn, kind, lane)
        n, _ = _v(ops[1], same)
        w = op | n << 5 | d
        return w | (size << 22 if kind == 's' else (lane == '2d') << 22 if kind == 'f' else 0)
    if mn in _NEON3:
        op, kind = _NEON3[mn]
        _lanes_ok(mn, kind, lane)
        n, _ = _v(ops[1], same)
        m, _ = _v(ops[2], same)
        w = op | m << 16 | n << 5 | d
        return w | (size << 22 if kind == 's' else (lane == '2d') << 22 if kind == 'f' else 0)
    raise AssemblerError('unsupported vector instruction %r' % mn)


def _lanes_ok(mn, kind, lane):
    if kind == 'f' and lane not in ('4s', '2d') or kind == '' and lane != '16b' or \
            kind == 's' and lane == '2d' and mn in _NO2D:
        raise AssemblerError('%s has no .%s form' % (mn, lane))


def _i_vscalar(a, mn, ops):
    """umov Wd, Vn.T[i]; dup Sd, Vn.s[i] (and Dd)."""
    n, k, i = _elem(ops[1])
    imm5 = _imm5(k, i)
    if mn == 'umov':  # an x register exactly for a .d element
        r = _reg(ops[0], 'x' if k == 'd' else 'w')
        return (0x4e003c00 if r[0] == 'x' else 0x0e003c00) | imm5 << 16 | n << 5 | r[1]
    r = _reg(ops[0], {'s': 's', 'd': 'd'}.get(k, 'sd'))
    if k not in 'sd':
        raise AssemblerError('dup to a scalar register takes .s or .d')
    return 0x5e000400 | imm5 << 16 | n << 5 | r[1]


def _i_ldst(a, mn, ops):
    if len(ops) < 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rt = _reg(ops[0], 'wxsdq')
    ent = _LDST.get((mn, rt[0]))
    if ent is None:
        raise AssemblerError('%s does not take %s registers' % (mn, rt[0]))
    op, sz = ent
    x = _regoffset(ops, sz)
    if x is not None:
        return (op & ~0x01000000) | 1 << 21 | x | 2 << 10 | rt[1]
    base, mode, off, rel = _mem(ops, 1)
    rr = base[1] << 5 | rt[1]
    if rel is not None:
        sym, add, mod = rel
        if sym is None:
            raise AssemblerError('bad memory operand %r' % ops[1])
        if mod in ('got_lo12', 'gotpageoff') and mn == 'ldr' and rt[0] == 'x':
            if add:
                raise AssemblerError('GOT references take no addend')
            a.fixup('LD64_GOT_LO12_NC', sym, 0)
        elif mod in ('lo12', 'pageoff'):
            a.fixup(_LO12[sz], sym, add)
        else:
            raise AssemblerError('unsupported relocation in %r' % ops[1])
        return op | rr
    if mode == 'off':
        if off >= 0 and off & ((1 << sz) - 1) == 0 and off >> sz <= 0xfff:
            return op | (off >> sz) << 10 | rr
        if -256 <= off <= 255:  # ldur/stur
            return (op & ~0x01000000) | (off & 0x1ff) << 12 | rr
        raise AssemblerError('immediate offset out of range')
    if not -256 <= off <= 255:
        raise AssemblerError('immediate offset out of range')
    return (op & ~0x01000000) | (off & 0x1ff) << 12 | (3 if mode == 'pre' else 1) << 10 | rr


# ldp/stp: register class -> (opc, V, log2 size)
_PAIR = {'w': (0, 0, 2), 'x': (2, 0, 3), 's': (0, 1, 2), 'd': (1, 1, 3), 'q': (2, 1, 4)}


def _i_pair(a, mn, ops):
    if len(ops) < 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    r1, r2 = _reg(ops[0], 'wxsdq'), _reg(ops[1], 'wxsdq')
    _same(r1, r2)
    opc, v, sz = _PAIR[r1[0]]
    base, mode, off, rel = _mem(ops, 2)
    if rel is not None:
        raise AssemblerError('relocations are not supported in %s' % mn)
    if off & ((1 << sz) - 1) or not -64 <= off >> sz <= 63:
        raise AssemblerError('immediate offset out of range')
    idx = {'post': 1, 'off': 2, 'pre': 3}[mode]
    load = 1 if mn == 'ldp' else 0
    return (opc << 30 | 0x28000000 | v << 26 | idx << 23 | load << 22
            | ((off >> sz) & 0x7f) << 15 | r2[1] << 10 | base[1] << 5 | r1[1])


# branches and system

def _i_b(a, mn, ops):
    if len(ops) != 1:
        raise AssemblerError('%s needs 1 operand' % mn)
    sym, add, mod = _expr(ops[0])
    if sym is None or mod is not None:
        raise AssemblerError('bad branch target %r' % ops[0])
    if mn == 'b':
        a.fixup('JUMP26', sym, add)
        return 0x14000000
    if mn == 'bl':
        a.fixup('CALL26', sym, add)
        return 0x94000000
    c = _COND[mn[2:] if mn[1] == '.' else mn[1:]]
    a.fixup('CONDBR19', sym, add)
    return 0x54000000 | c


def _i_csel(a, mn, ops):
    if len(ops) != 4:
        raise AssemblerError('%s needs 4 operands' % mn)
    rd, rn, rm = [_reg(o, 'sd' if mn == 'fcsel' else 'wx') for o in ops[:3]]
    _same(rd, rn, rm)
    c = _COND.get(ops[3].strip().lower())
    if c is None:
        raise AssemblerError('bad condition %r' % ops[3])
    if mn == 'fcsel':
        return 0x1e200c00 | _ftype(rd) << 22 | rm[1] << 16 | c << 12 | rn[1] << 5 | rd[1]
    base = {'csel': 0x1a800000, 'csinc': 0x1a800400, 'csinv': 0x5a800000, 'csneg': 0x5a800400}[mn]
    return base | _sf(rd) | rm[1] << 16 | c << 12 | rn[1] << 5 | rd[1]


def _i_mulh(a, mn, ops):
    if len(ops) != 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rd, rn, rm = [_reg(o, 'x') for o in ops]
    return (0x9bc07c00 if mn == 'umulh' else 0x9b407c00) | rm[1] << 16 | rn[1] << 5 | rd[1]


def _i_ccmp(a, mn, ops):
    """ccmp rn, rm or #imm5, #nzcv, cond."""
    if len(ops) != 4:
        raise AssemblerError('ccmp needs 4 operands')
    rn = _reg(ops[0])
    nzcv = _imm(ops[2], True)
    c = _COND.get(ops[3].strip().lower())
    if c is None or not 0 <= nzcv <= 15:
        raise AssemblerError('bad ccmp operands')
    if _isreg(ops[1]):
        rm = _reg(ops[1])
        _same(rn, rm)
        return 0x7a400000 | _sf(rn) | rm[1] << 16 | c << 12 | rn[1] << 5 | nzcv
    n = _imm(ops[1], True)
    if not 0 <= n <= 31:
        raise AssemblerError('ccmp immediate out of range')
    return 0x7a400800 | _sf(rn) | n << 16 | c << 12 | rn[1] << 5 | nzcv


def _i_tbz(a, mn, ops):
    if len(ops) != 3:
        raise AssemblerError('%s needs 3 operands' % mn)
    rt = _reg(ops[0])
    n = _imm(ops[1], True)
    if not 0 <= n < (64 if rt[0] == 'x' else 32):
        raise AssemblerError('bit number out of range')
    sym, add, mod = _expr(ops[2])
    if sym is None or mod is not None:
        raise AssemblerError('bad branch target %r' % ops[2])
    a.fixup('TSTBR14', sym, add)
    return ((0x37000000 if mn == 'tbnz' else 0x36000000) | (n >> 5) << 31 | (n & 31) << 19 | rt[1])


def _i_tst(a, mn, ops):
    """tst rn, rm or #imm: ands zr, rn, ..."""
    rn = _reg(ops[0])
    return _i_logic(a, 'ands', ['wzr' if rn[0] == 'w' else 'xzr'] + list(ops))


def _i_cbz(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('%s needs 2 operands' % mn)
    rt = _reg(ops[0])
    sym, add, mod = _expr(ops[1])
    if sym is None or mod is not None:
        raise AssemblerError('bad branch target %r' % ops[1])
    a.fixup('CONDBR19', sym, add)
    return (0x35000000 if mn == 'cbnz' else 0x34000000) | _sf(rt) | rt[1]


def _i_blr(a, mn, ops):
    if len(ops) != 1:
        raise AssemblerError('%s needs 1 operand' % mn)
    r = _reg(ops[0], 'x')
    return (0xd63f0000 if mn == 'blr' else 0xd61f0000) | r[1] << 5


def _i_ret(a, mn, ops):
    if len(ops) > 1:
        raise AssemblerError('ret takes at most 1 operand')
    r = _reg(ops[0], 'x') if ops else ('x', 30, False)
    return 0xd65f0000 | r[1] << 5


def _i_brk(a, mn, ops):
    if len(ops) != 1:
        raise AssemblerError('%s needs 1 operand' % mn)
    v = _imm(ops[0], True)
    if mn == 'brk':
        if not 0 <= v <= 0xffff:
            raise AssemblerError('brk immediate out of range')
        return 0xd4200000 | v << 5
    if not 0 <= v <= 127:
        raise AssemblerError('hint immediate out of range')
    return 0xd503201f | v << 5


def _i_nop(a, mn, ops):
    if ops:
        raise AssemblerError('nop takes no operands')
    return 0xd503201f


def _i_adrp(a, mn, ops):
    if len(ops) != 2:
        raise AssemblerError('adrp needs 2 operands')
    rd = _reg(ops[0], 'x')
    sym, add, mod = _expr(ops[1])
    if sym is None:
        raise AssemblerError('bad adrp operand %r' % ops[1])
    if mod in (None, 'page'):
        a.fixup('ADR_PREL_PG_HI21', sym, add)
    elif mod in ('got', 'gotpage') and not add:
        a.fixup('ADR_GOT_PAGE', sym, 0)
    else:
        raise AssemblerError('unsupported adrp operand %r' % ops[1])
    return 0x90000000 | rd[1]


def _i_tls(a, mn, ops):
    raise AssemblerError('thread-local storage is not supported')


_INSNS = {
    'add': _i_addsub, 'adds': _i_addsub, 'sub': _i_addsub, 'subs': _i_addsub,
    'cmp': _i_cmp, 'cmn': _i_cmp, 'neg': _i_neg,
    'and': _i_logic, 'orr': _i_logic, 'eor': _i_logic, 'ands': _i_logic,
    'bic': _i_logic, 'orn': _i_logic, 'eon': _i_logic, 'ubfx': _i_bitfield, 'sbfx': _i_bitfield,
    'ror': _i_ror, 'ubfiz': _i_ubfiz, 'bfi': _i_bfm, 'bfxil': _i_bfm,
    'lsl': _i_shift, 'lsr': _i_shift, 'asr': _i_shift,
    'mul': _i_dp2, 'sdiv': _i_dp2, 'udiv': _i_dp2, 'madd': _i_madd, 'msub': _i_madd,
    'smull': _i_widemul, 'umull': _i_widemul, 'smaddl': _i_widemul, 'smsubl': _i_widemul,
    'umaddl': _i_widemul, 'umsubl': _i_widemul,
    'cbz': _i_cbz, 'cbnz': _i_cbz, 'csel': _i_csel, 'fcsel': _i_csel,
    'csinc': _i_csel, 'csinv': _i_csel, 'csneg': _i_csel, 'tbz': _i_tbz, 'tbnz': _i_tbz, 'tst': _i_tst, 'ccmp': _i_ccmp,
    'smulh': _i_mulh, 'umulh': _i_mulh,
    'sxtb': _i_ext, 'sxth': _i_ext, 'sxtw': _i_ext, 'uxtb': _i_ext, 'uxth': _i_ext,
    'mov': _i_mov, 'movk': _i_movk, 'movz': _i_movk, 'movn': _i_movk, 'cset': _i_cset,
    'fadd': _i_fp3, 'fsub': _i_fp3, 'fmul': _i_fp3, 'fdiv': _i_fp3, 'fneg': _i_fneg,
    'fminnm': _i_fp3, 'fmaxnm': _i_fp3, 'fnmul': _i_fp3, 'fabs': _i_fneg, 'fsqrt': _i_fneg, 'frintp': _i_fneg,
    'frintm': _i_fneg, 'frintz': _i_fneg, 'frinta': _i_fneg, 'frintx': _i_fneg,
    'frinti': _i_fneg,
    'clz': _i_dp1, 'rbit': _i_dp1, 'rev16': _i_dp1, 'rev': _i_dp1, 'cnt': _i_simd,
    'addv': _i_simd, 'ushll': _i_bytes, 'xtn': _i_bytes, 'ld3': _i_bytes,
    'fmadd': _i_fp4, 'fmsub': _i_fp4, 'fnmadd': _i_fp4, 'fnmsub': _i_fp4,
    'fmov': _i_fmov, 'fcvt': _i_fcvt, 'fcvtzs': _i_fcvtz, 'fcvtzu': _i_fcvtz,
    'scvtf': _i_cvtf, 'ucvtf': _i_cvtf, 'fcmp': _i_fcmp, 'fcmpe': _i_fcmp,
    'ldp': _i_pair, 'stp': _i_pair,
    'b': _i_b, 'bl': _i_b, 'blr': _i_blr, 'br': _i_blr, 'ret': _i_ret,
    'brk': _i_brk, 'hint': _i_brk, 'nop': _i_nop, 'adrp': _i_adrp, 'mrs': _i_tls,
}
for _k in _LDST:
    _INSNS[_k[0]] = _i_ldst
for _c in _COND:
    _INSNS['b' + _c] = _i_b
    _INSNS['b.' + _c] = _i_b


# ---------------------------------------------------------------------------------------------
# strings and comments

_ESC = {'b': 8, 'f': 12, 'n': 10, 'r': 13, 't': 9, 'v': 11, '"': 34, '\\': 92, "'": 39}


def _string(s, i):
    """GNU as string literal starting at s[i] == '"'; returns (bytes, new index)."""
    out = bytearray()
    i += 1
    n = len(s)
    while True:
        if i >= n:
            raise AssemblerError('unterminated string')
        c = s[i]
        if c == '"':
            return bytes(out), i + 1
        if c != '\\':
            out.extend(c.encode('latin-1'))
            i += 1
            continue
        i += 1
        if i >= n:
            raise AssemblerError('unterminated string')
        c = s[i]
        if c in '01234567':
            j = i
            while j < n and j < i + 3 and s[j] in '01234567':
                j += 1
            out.append(int(s[i:j], 8) & 0xff)
            i = j
        elif c in 'xX':
            j = i + 1
            while j < n and s[j] in '0123456789abcdefABCDEF':
                j += 1
            out.append(int(s[i + 1:j] or '0', 16) & 0xff)
            i = j
        elif c in _ESC:
            out.append(_ESC[c])
            i += 1
        else:
            out.extend(c.encode('latin-1'))
            i += 1


def _strip_comments(text):
    """Remove /* */ and // comments outside string literals."""
    if '/' not in text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = i
        # copy up to the next interesting character quickly
        while j < n and text[j] not in '"/':
            j += 1
        out.append(text[i:j])
        if j >= n:
            break
        c = text[j]
        if c == '"':
            k = j + 1
            while k < n and text[k] != '"' and text[k] != '\n':
                k += 2 if text[k] == '\\' else 1
            out.append(text[j:k + 1])
            i = k + 1
        elif text.startswith('/*', j):
            k = text.find('*/', j + 2)
            if k < 0:
                raise AssemblerError('unterminated comment')
            out.append('\n' * text.count('\n', j, k))  # keep line numbers
            i = k + 2
        elif text.startswith('//', j):
            k = text.find('\n', j)
            i = n if k < 0 else k
        else:
            out.append('/')
            i = j + 1
    return ''.join(out)


# ---------------------------------------------------------------------------------------------
# the assembler

_NOP = struct.pack('<I', 0xd503201f)


class _Asm:
    def __init__(self, syntax):
        if syntax not in ('elf', 'macho'):
            raise ValueError('syntax must be "elf" or "macho"')
        self.syntax = syntax
        self.obj = Object(syntax)
        self.fixups = []
        self.sec = None
        self.switch('.text')

    # sections

    def switch(self, name, flags=None):
        """Switch to section `name`, creating it from its name (and flags) on first use."""
        obj = self.obj
        s = obj.section(name)
        if s is None:
            s = self._newsec(name, flags)
            obj.sections.append(s)
        self.sec = s

    def _newsec(self, name, flags):
        if self.syntax == 'macho' and ',' in name:
            parts = [p.strip() for p in name.split(',')]
            seg, sect = parts[0], parts[1]
            typ = parts[2] if len(parts) > 2 else ''
            attrs = parts[3] if len(parts) > 3 else ''
            if typ.startswith('thread_') or sect.startswith('__thread'):
                raise AssemblerError('thread-local storage is not supported')
            ex = sect == '__text' or 'pure_instructions' in attrs
            nob = typ == 'zerofill' or sect in ('__bss', '__common')
            return Section(name, exec=ex, write=seg != '__TEXT', nobits=nob)
        if name.startswith(('.tdata', '.tbss')) or (flags and 'T' in flags[0]):
            raise AssemblerError('thread-local storage is not supported')
        if flags:
            f = flags[0].strip().strip('"')
            nob = any(t.strip() in ('@nobits', '%nobits') for t in flags[1:])
            return Section(name, exec='x' in f, write='w' in f, alloc='a' in f, nobits=nob)
        if name.startswith('.text'):
            return Section(name, exec=True)
        if name.startswith('.data'):
            return Section(name, write=True)
        if name.startswith('.bss'):
            return Section(name, write=True, nobits=True)
        if name.startswith('.rodata'):
            return Section(name)
        return Section(name, alloc=False)

    # emission

    def emit(self, b):
        s = self.sec
        if s.nobits:
            if any(b):
                raise AssemblerError('non-zero data in nobits section %s' % s.name)
            s.size += len(b)
            return
        s.data += b
        s.size += len(b)

    def align(self, n):
        if n <= 0 or n & (n - 1):
            raise AssemblerError('alignment must be a power of 2')
        s = self.sec
        s.align = max(s.align, n)
        pad = -s.size % n
        if not pad:
            return
        if s.exec and not s.nobits and s.size % 4 == 0:
            self.emit(_NOP * (pad // 4))
        else:
            self.emit(bytes(pad))

    def fixup(self, kind, sym, addend):
        self.fixups.append((self.sec, self.sec.size, kind, sym, addend, self.lineno))

    def symbol(self, name):
        s = self.obj.symbols.get(name)
        if s is None:
            s = self.obj.symbols[name] = Symbol(name)
        return s

    def label(self, name):
        s = self.symbol(name)
        if s.section is not None or s.common is not None:
            raise AssemblerError('symbol %s is already defined' % name)
        s.section = self.sec
        s.value = self.sec.size

    # driver

    def run(self, text):
        text = _strip_comments(text)
        insns = _INSNS
        for self.lineno, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                # labels (possibly several, possibly followed by a statement)
                while True:
                    if line[0] == '"':
                        name, j = _parse_sym(line, 0)
                    else:
                        m = _SYM_RE.match(line)
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
                ops = _split(rest)
                f = insns.get(mn)
                if f is _i_bytes:
                    pass
                elif _isneon(ops):
                    f = lambda a, mn, ops: _neon(mn, ops)
                elif mn in ('umov', 'dup') and ops:
                    f = _i_vscalar
                if f is None:
                    raise AssemblerError('unsupported instruction %r' % parts[0])
                if self.sec.nobits:
                    raise AssemblerError('instruction in nobits section %s' % self.sec.name)
                word = f(self, mn, ops)
                if self.sec.size % 4:
                    raise AssemblerError('misaligned instruction')
                self.sec.align = max(self.sec.align, 4)
                self.emit(struct.pack('<I', word))
            except AssemblerError as e:
                raise AssemblerError('line %d: %s: %s' % (self.lineno, e, line)) from None
        self.resolve()
        return self.obj

    def directive(self, d, rest):
        if d in ('.text', '.data', '.bss'):
            if rest.strip():
                raise AssemblerError('subsections are not supported')
            self.switch(d)
        elif d == '.section':
            args = _split(rest)
            if not args or not args[0]:
                raise AssemblerError('.section needs a name')
            if self.syntax == 'macho':
                self.switch(','.join(args))
            else:
                name = args[0]
                if name.startswith('"'):
                    name = _parse_sym(name, 0)[0]
                self.switch(name, args[1:] or None)
        elif d in ('.globl', '.global'):
            for n in _split(rest):
                self.symbol(_parse_sym(n, 0)[0]).globl = True
        elif d == '.balign':
            args = _split(rest)
            self._fill_check(args[1:])
            self.align(_int(args[0]))
        elif d == '.p2align':
            args = _split(rest)
            self._fill_check(args[1:])
            self.align(1 << _int(args[0]))
        elif d in ('.byte', '.short', '.hword', '.2byte', '.int', '.word', '.long', '.4byte',
                   '.quad', '.8byte', '.xword', '.dword'):
            size = {'.byte': 1, '.short': 2, '.hword': 2, '.2byte': 2, '.quad': 8, '.8byte': 8,
                    '.xword': 8, '.dword': 8}.get(d, 4)
            for e in _split(rest):
                self.data(size, e)
        elif d in ('.ascii', '.asciz', '.string'):
            i = 0
            rest = rest.strip()
            while True:
                if i >= len(rest) or rest[i] != '"':
                    raise AssemblerError('expected a string')
                b, i = _string(rest, i)
                self.emit(b + (b'\0' if d != '.ascii' else b''))
                while i < len(rest) and rest[i] in ' \t':
                    i += 1
                if i >= len(rest):
                    break
                if rest[i] != ',':
                    raise AssemblerError('junk after string')
                i += 1
                while i < len(rest) and rest[i] in ' \t':
                    i += 1
        elif d in ('.fill', '.zero', '.skip', '.space'):
            args = _split(rest)
            if d == '.fill':
                n = _int(args[0])
                size = _int(args[1]) if len(args) > 1 else 1
                val = _imm(args[2]) if len(args) > 2 else 0
                if size > 8 or size < 0:
                    raise AssemblerError('.fill size must be at most 8')
                unit = struct.pack('<Q', val & M64)[:min(size, 4)].ljust(size, b'\0')
                self.emit(unit * n)
            else:
                n = _int(args[0])
                val = _imm(args[1]) if len(args) > 1 else 0
                self.emit(bytes([val & 0xff]) * n)
        elif d == '.comm':
            args = _split(rest)
            if len(args) not in (2, 3):
                raise AssemblerError('.comm needs a name, a size and optionally an alignment')
            s = self.symbol(_parse_sym(args[0], 0)[0])
            if s.section is not None:
                raise AssemblerError('symbol %s is already defined' % s.name)
            size = _int(args[1])
            if len(args) > 2:
                al = _int(args[2])
                if self.syntax == 'macho':
                    al = 1 << al  # Mach-O: log2 of the alignment
            else:  # GNU as: the next power of 2 >= size, at most 16
                al = 1
                while al < size and al < 16:
                    al *= 2
            if al & (al - 1):
                raise AssemblerError('alignment must be a power of 2')
            s.common = (size, al)
        elif d in ('.type', '.size', '.file', '.loc', '.ident', '.local', '.align_fn',
                   '.cfi_startproc', '.cfi_endproc', '.subsections_via_symbols'):
            pass  # symbol metadata and debug info: nothing to load
        elif d == '.abort':
            raise AssemblerError('assembly aborted: %s' % rest.strip())
        elif d in ('.tbss', '.tdata', '.tlv'):
            raise AssemblerError('thread-local storage is not supported')
        else:
            raise AssemblerError('unsupported directive %s' % d)

    @staticmethod
    def _fill_check(args):
        for a in args:
            if a.strip() and _int(a.strip()) != 0:
                raise AssemblerError('alignment fill values are not supported')

    def data(self, size, e):
        sym, add, mod = _expr(e)
        if mod is not None:
            raise AssemblerError('relocation modifier in data')
        if sym is None:
            lo = -(1 << (8 * size - 1))
            if not lo <= add < 1 << (8 * size):
                raise AssemblerError('value %s does not fit in %d bytes' % (e, size))
            self.emit((add & ((1 << (8 * size)) - 1)).to_bytes(size, 'little'))
            return
        kind = {8: 'ABS64', 4: 'ABS32', 2: 'ABS16'}.get(size)
        if kind is None:
            raise AssemblerError('cannot relocate a %d-byte value' % size)
        if self.sec.nobits:
            raise AssemblerError('relocation in nobits section')
        self.fixup(kind, sym, add)
        self.emit(bytes(size))

    def resolve(self):
        """Resolve branches to non-global labels of the same section (as GNU as does) and turn
        everything else into relocations."""
        syms = self.obj.symbols
        for sec, off, kind, name, add, lineno in self.fixups:
            s = syms.get(name)
            if s is None:
                s = syms[name] = Symbol(name)
            if kind in ('JUMP26', 'CALL26', 'CONDBR19', 'TSTBR14') and s.section is sec and not s.globl:
                d = s.value + add - off
                if d & 3:
                    raise AssemblerError('line %d: misaligned branch target %s' % (lineno, name))
                bits = {'CONDBR19': 19, 'TSTBR14': 14}.get(kind, 26)  # tbz: 32 KB either way
                if not -(1 << (bits + 1)) <= d < 1 << (bits + 1):
                    raise AssemblerError('line %d: branch out of range: %s' % (lineno, name))
                w = struct.unpack_from('<I', sec.data, off)[0]
                if bits == 26:
                    w |= (d >> 2) & 0x3ffffff
                else:
                    w |= ((d >> 2) & ((1 << bits) - 1)) << 5
                struct.pack_into('<I', sec.data, off, w)
            else:
                sec.relocs.append(Reloc(off, kind, name, add))


def assemble(text, syntax='elf'):
    """Assemble arm64 GNU-syntax assembly text into an Object. syntax: 'elf' (QBE target
    arm64) or 'macho' (arm64_apple)."""
    return _Asm(syntax).run(text)
