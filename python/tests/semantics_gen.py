"""Random C expressions with results computed by a Python model of C11 (test_semantics.py).

Types: _Bool, char (signedness per target), signed/unsigned char, short, int, long, long long
(and unsigned), float, double, and bit-fields of 1-31 bits. Operators: arithmetic, shifts,
bitwise, comparisons, logical, unary, casts, ?:, comma, compound assignment, ++/--. Operands are
constants (exercising constant folding) or parameters (exercising code generation). Undefined
behavior is never generated: signed overflow, out-of-range shifts, division by zero,
out-of-range float-to-integer conversions, and float results that are NaN, infinite or
subnormal are rejected and regenerated. Implementation-defined choices are cimo's (and gcc's):
two's complement, modulo conversion to narrower signed types, LP64.
"""

import math
import random
import struct


class UB(Exception):
    pass


class T:
    def __init__(self, name, kind, bits, signed, rank):
        self.name, self.kind, self.bits, self.signed, self.rank = name, kind, bits, signed, rank

    def __repr__(self):
        return self.name

    @property
    def isint(self):
        return self.kind == 'int'


def types(char_signed):
    ts = [T('_Bool', 'int', 1, False, 0),
          T('char', 'int', 8, char_signed, 1), T('signed char', 'int', 8, True, 1),
          T('unsigned char', 'int', 8, False, 1),
          T('short', 'int', 16, True, 2), T('unsigned short', 'int', 16, False, 2),
          T('int', 'int', 32, True, 3), T('unsigned', 'int', 32, False, 3),
          T('long', 'int', 64, True, 4), T('unsigned long', 'int', 64, False, 4),
          T('long long', 'int', 64, True, 5), T('unsigned long long', 'int', 64, False, 5),
          T('float', 'float', 32, True, 0), T('double', 'float', 64, True, 0)]
    return {t.name: t for t in ts}


def rng(t):
    if t.name == '_Bool':
        return 0, 1
    if t.signed:
        return -(1 << (t.bits - 1)), (1 << (t.bits - 1)) - 1
    return 0, (1 << t.bits) - 1


def f32(x):
    """x rounded to binary32."""
    return struct.unpack('f', struct.pack('f', x))[0]


def okfloat(x):
    if x != x or math.isinf(x) or (x != 0 and abs(x) < 1e-30) or abs(x) > 1e30:
        raise UB('float range')
    return x


def int_to_float(v, t):
    if t.bits == 64:
        return float(v)  # correctly rounded
    # binary32: round to 24 significant bits, ties to even
    if v == 0:
        return 0.0
    a = abs(v)
    sh = a.bit_length() - 24
    if sh > 0:
        q, r = divmod(a, 1 << sh)
        half = 1 << (sh - 1)
        if r > half or (r == half and q & 1):
            q += 1
        a = q << sh
    return float(a) if v > 0 else -float(a)


class Model:
    def __init__(self, char_signed):
        self.ty = types(char_signed)

    def conv(self, v, src, dst):
        """C conversion of value v from type src to dst (raises UB)."""
        if dst.name == '_Bool':
            return int(v != 0)
        if dst.isint:
            if not src.isint:
                if v != v or math.isinf(v):
                    raise UB('float to int')
                v = int(v)  # truncation toward zero
                lo, hi = rng(dst)
                if not lo <= v <= hi:
                    raise UB('float to int range')
                return v
            v &= (1 << dst.bits) - 1
            if dst.signed and v >> (dst.bits - 1):
                v -= 1 << dst.bits
            return v
        if src.isint:
            return okfloat(int_to_float(v, dst))
        return okfloat(f32(v) if dst.bits == 32 else v)

    def promote(self, t):
        if t.isint and t.rank < 3:
            return self.ty['int']
        return t

    def uac(self, a, b):
        if not a.isint or not b.isint:
            if 'double' in (a.name, b.name):
                return self.ty['double']
            return self.ty['float']
        a, b = self.promote(a), self.promote(b)
        if a is b:
            return a
        if a.signed == b.signed:
            return a if a.rank > b.rank else b
        u, s = (a, b) if not a.signed else (b, a)
        if u.rank >= s.rank:
            return u
        if s.bits > u.bits:
            return s
        return self.ty['unsigned ' + s.name if s.name != 'int' else 'unsigned']

    def checked(self, v, t):
        """An arithmetic result in type t: wraps for unsigned, UB on signed overflow."""
        if not t.isint:
            return okfloat(f32(v) if t.bits == 32 else v)
        lo, hi = rng(t)
        if t.signed:
            if not lo <= v <= hi:
                raise UB('signed overflow')
            return v
        return v & hi

    def binary(self, op, a, ta, b, tb):
        """(value, type) of a op b."""
        if op in ('<<', '>>'):
            if not ta.isint or not tb.isint:
                raise UB('shift of float')
            t = self.promote(ta)
            a = self.conv(a, ta, t)
            n = self.conv(b, tb, self.promote(tb))
            if not 0 <= n < t.bits:
                raise UB('shift count')
            if op == '>>':
                return a >> n, t  # arithmetic for signed: cimo's (implementation-defined)
            if t.signed and (a < 0 or (a << n) > rng(t)[1]):
                raise UB('left shift')
            return self.checked(a << n, t), t
        if op in ('&&', '||'):
            ra, rb = a != 0, b != 0
            return int(ra and rb if op == '&&' else ra or rb), self.ty['int']
        t = self.uac(ta, tb)
        x, y = self.conv(a, ta, t), self.conv(b, tb, t)
        if op in ('<', '<=', '>', '>=', '==', '!='):
            return int({'<': x < y, '<=': x <= y, '>': x > y, '>=': x >= y,
                        '==': x == y, '!=': x != y}[op]), self.ty['int']
        if op in ('&', '|', '^', '%') and not t.isint:
            raise UB('float operand')
        if op == '+':
            r = x + y
        elif op == '-':
            r = x - y
        elif op == '*':
            r = x * y
        elif op in ('/', '%'):
            if y == 0:
                raise UB('division by zero')
            if not t.isint:
                r = x / y
            else:
                q = abs(x) // abs(y)
                q = q if (x < 0) == (y < 0) else -q  # truncation toward zero
                r = q if op == '/' else x - q * y
                if t.signed and not rng(t)[0] <= q <= rng(t)[1]:
                    raise UB('division overflow')
        elif op == '&':
            r = x & y
        elif op == '|':
            r = x | y
        else:
            r = x ^ y
        if t.isint and not t.signed:
            r &= rng(t)[1]
        return self.checked(r, t), t

    def unary(self, op, a, ta):
        if op == '!':
            return int(a == 0), self.ty['int']
        if op == '~':
            if not ta.isint:
                raise UB('float operand')
            t = self.promote(ta)
            return self.conv(~self.conv(a, ta, t), self.ty['long long'], t), t
        t = self.promote(ta)
        a = self.conv(a, ta, t)
        if op == '+':
            return a, t
        if not t.isint:
            return -a, t
        return self.checked(-a if t.signed else (-a) & rng(t)[1], t), t


def clit(v, t):
    """A C constant expression of type t and value v."""
    if not t.isint:
        if v == 0 and math.copysign(1, v) < 0:
            return '(-0.0%s)' % ('f' if t.bits == 32 else '')
        s = repr(v) if t.bits == 64 else repr(v) + 'f'
        if 'e' not in s and '.' not in s:
            s = s.replace('f', '.0f') if t.bits == 32 else s + '.0'
        return '(%s)%s' % (t.name, s)
    if t.name == '_Bool':
        return '(_Bool)%d' % v
    if v == -(1 << 63):
        return '(%s)(-9223372036854775807LL - 1)' % t.name
    if v < 0:
        return '(%s)(%dLL)' % (t.name, v)
    return '(%s)%dULL' % (t.name, v)


class Gen:
    OPS = ['+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^', '<', '<=', '>', '>=', '==',
           '!=', '&&', '||']

    def __init__(self, seed, char_signed):
        self.r = random.Random(seed)
        self.m = Model(char_signed)
        self.ty = self.m.ty

    def value(self, t):
        r = self.r
        if not t.isint:
            v = r.choice([0.0, -0.0, 1.0, -1.0, 0.5, 2.5, -3.75, 1e10, -7e-3, 16777217.0,
                          9007199254740993.0, 123456.789, r.randint(-10 ** 6, 10 ** 6) / 64])
            return self.m.conv(v, self.ty['double'], t)
        lo, hi = rng(t)
        pick = r.choice([lo, hi, 0, 1, lo + 1, hi - 1, -1, 2, r.randint(lo, hi),
                         r.randint(-300, 300), 1 << r.randint(0, t.bits - 1)])
        return self.m.conv(pick, self.ty['long long'] if pick < 2 ** 63 else
                           self.ty['unsigned long long'], t)

    def leaf(self, params):
        r = self.r
        if r.random() < 0.15:  # a bit-field
            w = r.randint(1, 31)
            signed = r.random() < 0.5
            t = self.ty['int' if signed else 'unsigned']
            lo, hi = (-(1 << (w - 1)), (1 << (w - 1)) - 1) if signed else (0, (1 << w) - 1)
            v = r.randint(lo, hi)
            params.append(('bf', w, signed, v))
            # value read from the field: promoted to int when it fits (C11 6.3.1.1p2)
            pt = self.ty['int'] if (signed or w < 32) else t
            return 'b%d.f' % (len(params) - 1), pt, v
        t = self.ty[r.choice(list(self.ty))]
        v = self.value(t)
        if r.random() < 0.5:
            return clit(v, t), t, v
        params.append(('p', t, v))
        return 'p%d' % (len(params) - 1), t, v

    def expr(self, depth, params):
        """(C text, type, value)."""
        r = self.r
        for _ in range(50):
            n = len(params)
            try:
                if depth <= 0 or r.random() < 0.2:
                    return self.leaf(params)
                k = r.random()
                if k < 0.55:
                    a, ta, va = self.expr(depth - 1, params)
                    b, tb, vb = self.expr(depth - 1, params)
                    op = r.choice(self.OPS)
                    v, t = self.m.binary(op, va, ta, vb, tb)
                    return '(%s %s %s)' % (a, op, b), t, v
                if k < 0.7:
                    a, ta, va = self.expr(depth - 1, params)
                    op = r.choice(['-', '~', '!', '+'])
                    v, t = self.m.unary(op, va, ta)
                    return '(%s%s)' % (op, a), t, v
                if k < 0.85:
                    a, ta, va = self.expr(depth - 1, params)
                    t = self.ty[r.choice(list(self.ty))]
                    return '((%s)%s)' % (t.name, a), t, self.m.conv(va, ta, t)
                if k < 0.95:
                    c, tc, vc = self.expr(depth - 1, params)
                    a, ta, va = self.expr(depth - 1, params)
                    b, tb, vb = self.expr(depth - 1, params)
                    t = self.m.uac(ta, tb)
                    v = self.m.conv(va if vc != 0 else vb, ta if vc != 0 else tb, t)
                    # the other operand is converted too: UB there counts
                    self.m.conv(vb if vc != 0 else va, tb if vc != 0 else ta, t)
                    return '(%s ? %s : %s)' % (c, a, b), t, v
                a, ta, va = self.expr(depth - 1, params)
                b, tb, vb = self.expr(depth - 1, params)
                return '(%s, %s)' % (a, b), tb, vb
            except UB:
                del params[n:]
        raise UB('no expression')

    def statement(self, params):
        """Compound assignment or ++/-- on a local: (C statements, type, value)."""
        r = self.r
        t = self.ty[r.choice(list(self.ty))]
        x = self.value(t)
        lines = ['%s x = %s;' % (t.name, clit(x, t))]
        for _ in range(r.randint(1, 4)):
            if r.random() < 0.3:
                op = r.choice(['++', '--'])
                one = self.ty['int']
                v, rt = self.m.binary(op[0], x, t, 1, one)
                x = self.m.conv(v, rt, t)
                lines.append('%sx;' % op if r.random() < 0.5 else 'x%s;' % op)
                continue
            b, tb, vb = self.expr(2, params)
            op = r.choice(['+', '-', '*', '/', '%', '<<', '>>', '&', '|', '^'])
            v, rt = self.m.binary(op, x, t, vb, tb)
            x = self.m.conv(v, rt, t)
            lines.append('x %s= %s;' % (op, b))
        return lines, t, x

    def function(self, k):
        """C source of function tK and the expected u64 result, or None (all UB)."""
        r = self.r
        params = []
        try:
            if r.random() < 0.25:
                body, t, v = self.statement(params)
                result = 'x'
            else:
                e, t, v = self.expr(r.randint(1, 4), params)
                body = ['%s v = %s;' % (t.name, e)]
                result = 'v'
        except UB:
            return None
        args, setup = [], []
        for i, p in enumerate(params):
            if p[0] == 'p':
                _, pt, pv = p
                if pt.isint:
                    args.append('long long a%d' % i)
                    setup.append('%s p%d = (%s)a%d;' % (pt.name, i, pt.name, i))
                else:
                    args.append('double a%d' % i)
                    setup.append('%s p%d = (%s)a%d;' % (pt.name, i, pt.name, i))
            else:
                _, w, signed, pv = p
                args.append('long long a%d' % i)
                setup.append('struct { %s f : %d; } b%d; b%d.f = a%d;'
                             % ('int' if signed else 'unsigned', w, i, i, i))
        if t.isint:
            ret = '(unsigned long long)%s' % result
            want = v & (2 ** 64 - 1)
        elif t.bits == 32:
            ret = 'fbits(%s)' % result
            want = struct.unpack('<I', struct.pack('<f', v))[0]
        else:
            ret = 'dbits(%s)' % result
            want = struct.unpack('<Q', struct.pack('<d', v))[0]
        src = 'unsigned long long t%d(%s) {\n  %s\n  return %s;\n}\n' % (
            k, ', '.join(args) or 'void', '\n  '.join(setup + body), ret)
        callargs = [p[2] if p[0] == 'p' else p[3] for p in params]
        return src, callargs, want


HEADER = '''static unsigned long long fbits(float x) { union { float f; unsigned u; } c; c.f = x; return c.u; }
static unsigned long long dbits(double x) { union { double f; unsigned long long u; } c; c.f = x; return c.u; }
'''


def cases(seed, count, char_signed):
    """(C source, [(function name, args, expected u64)])."""
    g = Gen(seed, char_signed)
    srcs, out = [HEADER], []
    k = 0
    while len(out) < count:
        f = g.function(k)
        if f is not None:
            srcs.append(f[0])
            out.append(('t%d' % k, f[1], f[2]))
        k += 1
    return '\n'.join(srcs), out


if __name__ == '__main__':
    import sys
    print(cases(int(sys.argv[1]) if len(sys.argv) > 1 else 1,
                int(sys.argv[2]) if len(sys.argv) > 2 else 10, False)[0])
