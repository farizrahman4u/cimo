"""Random loop programs for differential testing of the middle end (test_middle_random.py).

Each function has the signature of tests/middle/loops.c:
    double f(int n, int k, float *fa, float *fb, int *ia, int *ib, double *da,
             unsigned char *ua)
with n in [0, 100] and k in [-5, 5]; buffers are valid from -256 to 767. Loop nests of depth
1-3 with random bounds, steps, directions and IV types; bodies of array updates with affine
indexes (kept in bounds by interval arithmetic on the IVs), reductions, conditionals and
early exits. All integer arithmetic is wrapping-safe for gcc -fwrapv (the reference is
built with -fwrapv, and cimo never assumes signed overflow is undefined).
"""

import random

LO, HI = -250, 760


class Gen:
    def __init__(self, seed, restrict=True):
        self.r = random.Random(seed)
        self.restrict = restrict
        self.ivn = 0

    def pick(self, *xs):
        return self.r.choice(xs)

    # IVs: (name, lo, hi, is_int)
    def loop(self, ivs, depth):
        r = self.r
        self.ivn += 1
        v = 'i%d' % self.ivn
        ty = self.pick('int', 'int', 'int', 'long', 'unsigned')
        kind = r.randrange(7)
        nb = 'n' if not ivs else self.pick('n', 'n', ivs[-1][0])
        nb_hi = 100 if nb == 'n' else ivs[-1][2]
        nb_lo = 0 if nb == 'n' else max(ivs[-1][1], 0)
        if ty == 'unsigned':
            kind = self.pick(0, 1, 5)
            nb, nb_hi, nb_lo = 'n', 100, 0
        if kind == 3:
            nb, nb_hi, nb_lo = 'n', 100, 0
        if kind == 0:     # for (v = c; v < bound + d; v += s)
            c = r.randrange(0, 4)
            d = r.randrange(-3, 4) if ty != 'unsigned' else r.randrange(0, 4)
            s = self.pick(1, 1, 1, 2, 3)
            hdr = 'for (%s %s = %d; %s < %s + %d; %s += %d)' % (ty, v, c, v, nb, d, v, s)
            lo, hi = c, nb_hi + d + s
        elif kind == 1:   # <=
            c = r.randrange(0, 3)
            s = self.pick(1, 1, 2, 4)
            hdr = 'for (%s %s = %d; %s <= %s; %s += %d)' % (ty, v, c, v, nb, v, s)
            lo, hi = c, nb_hi + s
        elif kind == 2:   # down
            s = self.pick(1, 1, 2, 3)
            c = r.randrange(-3, 3)
            hdr = 'for (%s %s = %s - 1; %s >= %d; %s -= %d)' % (ty, v, nb, v, c, v, s)
            lo, hi = c, nb_hi
        elif kind == 3:   # != with step 1
            hdr = 'for (%s %s = 0; %s != %s; %s++)' % (ty, v, v, nb, v)
            lo, hi = 0, nb_hi
        elif kind == 4:   # k-dependent start
            hdr = 'for (%s %s = k; %s < %s; %s++)' % (ty, v, v, nb, v)
            lo, hi = -5, nb_hi
        elif kind == 5:   # small constant trip count
            t = r.randrange(0, 6)
            hdr = 'for (%s %s = 0; %s < %d; %s++)' % (ty, v, v, t, v)
            lo, hi = 0, t
        else:             # while loop
            hdr = None
            lo, hi = 0, nb_hi
        body = self.block(ivs + [('(int)' + v if ty == 'unsigned' else v, lo, hi)], depth)
        if hdr is None:
            return ('{ %s %s = 0; while (%s < %s) { %s %s++; } }'
                    % (ty, v, v, nb, body, v))
        return '%s { %s }' % (hdr, body)

    def index(self, ivs, span=0):
        """An affine index expression in bounds, as a string."""
        r = self.r
        for _ in range(20):
            terms = []
            lo = hi = 0
            for name, a, b in ivs:
                if r.random() < 0.5:
                    continue
                c = self.pick(1, 1, 1, 2, 3, -1)
                terms.append((c, name))
                x, y = c * a, c * b
                lo += min(x, y)
                hi += max(x, y)
            off = r.randrange(LO - lo, max(LO - lo + 1, HI - hi - span))
            if lo + off < LO or hi + off + span > HI:
                continue
            s = ' + '.join('%d * %s' % (c, n) if c != 1 else n for c, n in terms) or '0'
            if self.r.random() < 0.2 and terms:
                s = '(int)(%s)' % s
            return '%s + %d' % (s, off)
        return str(r.randrange(0, 50))

    def fexpr(self, ivs, d=0):
        r = self.r
        x = r.random()
        if d > 2 or x < 0.35:
            a = self.pick('fa', 'fb')
            return '%s[%s]' % (a, self.index(ivs))
        if x < 0.45:
            return '%d.%df' % (r.randrange(-3, 4), r.randrange(10))
        if x < 0.5 and ivs:
            return '(float)%s' % self.pick(*[n for n, _, _ in ivs])
        op = self.pick('+', '-', '*', '+', '*')
        return '(%s %s %s)' % (self.fexpr(ivs, d + 1), op, self.fexpr(ivs, d + 1))

    def iexpr(self, ivs, d=0):
        r = self.r
        x = r.random()
        if d > 2 or x < 0.3:
            a = self.pick('ia', 'ib', 'ua')
            return '%s[%s]' % (a, self.index(ivs))
        if x < 0.45:
            return str(r.randrange(-9, 10))
        if x < 0.6 and ivs:
            return '(int)' + self.pick(*[n for n, _, _ in ivs])
        op = self.pick('+', '-', '*', '^', '&', '>>')
        if op == '>>':
            return '(%s >> %d)' % (self.iexpr(ivs, d + 1), r.randrange(0, 5))
        return '(%s %s %s)' % (self.iexpr(ivs, d + 1), op, self.iexpr(ivs, d + 1))

    def stmt(self, ivs, depth):
        r = self.r
        x = r.random()
        if depth > 0 and x < 0.2:
            return self.loop(ivs, depth - 1)
        if x < 0.4:
            a = self.pick('fa', 'fb')
            return '%s[%s] = %s;' % (a, self.index(ivs), self.fexpr(ivs))
        if x < 0.55:
            a = self.pick('ia', 'ib')
            return '%s[%s] = %s;' % (a, self.index(ivs), self.iexpr(ivs))
        if x < 0.6:
            return 'ua[%s] = (unsigned char)(%s);' % (self.index(ivs), self.iexpr(ivs))
        if x < 0.72:
            return 'fs = fs * 0.5f + %s;' % self.fexpr(ivs)
        if x < 0.82:
            return 's = s * 3 + %s;' % self.iexpr(ivs)
        if x < 0.87:
            return 'ds += da[%s] * %s;' % (self.index(ivs), self.fexpr(ivs))
        if x < 0.93:
            return 'if (%s > %d) %s else %s' % (self.iexpr(ivs), r.randrange(-50, 50),
                                               self.stmt(ivs, 0), self.stmt(ivs, 0))
        if x < 0.96:
            return 'if (%s == %d) break;' % (self.iexpr(ivs), r.randrange(-3, 3))
        return 'da[%s] = %s;' % (self.index(ivs), self.fexpr(ivs))

    def block(self, ivs, depth):
        return ' '.join(self.stmt(ivs, depth) for _ in range(self.r.randrange(1, 4)))

    def function(self, name):
        q = 'restrict ' if self.restrict else ''
        body = ' '.join(self.loop([], self.r.randrange(0, 3)) for _ in range(
            self.r.randrange(1, 3)))
        return ('double %s(int n, int k, float *%sfa, float *%sfb, int *%sia, int *%sib, '
                'double *%sda, unsigned char *%sua) {\n  long s = 0; float fs = 0; '
                'double ds = 0;\n  %s\n  return (double)s + fs + ds;\n}\n'
                % (name, q, q, q, q, q, q, body))


def program(seed, nfuncs=40):
    """C source with functions r0..r{nfuncs-1} (odd ones without restrict)."""
    out = []
    names = []
    for i in range(nfuncs):
        g = Gen(seed * 1000 + i, restrict=(i % 2 == 0))
        name = 'r%d%s' % (i, '' if i % 2 == 0 else '_alias')
        names.append(name)
        out.append(g.function(name))
    return '\n'.join(out), names
