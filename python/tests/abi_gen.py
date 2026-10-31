"""Random calling-convention tests (test_abi.py): C source with, for each random signature N,

    R fN(args)                    hashes every argument (member by member) into its result
    unsigned long long callN(R (*)(args))   calls with fixed arguments, hashes the result

compiled once by a reference compiler (gcc, clang) into a shared library and once by cimo.
Calling the reference's fN from cimo's callN, and cimo's fN from the reference's callN, must give
what the reference gives calling itself. Types: all integer widths, float, double, pointers,
structs of 1 to 80 bytes (float/double aggregates, odd sizes, arrays, nesting), enough
arguments to use the stack, and variadic functions (scalar variadic arguments). No `long`
(32-bit in Windows DLLs) and no C library (Windows DLLs are built without one).
"""

import random

SCALARS = ['signed char', 'unsigned char', 'short', 'unsigned short', 'int', 'unsigned',
           'long long', 'unsigned long long', 'float', 'double', 'void *', '_Bool']
INTS = {'signed char': (-128, 127), 'unsigned char': (0, 255), 'short': (-32768, 32767),
        'unsigned short': (0, 65535), 'int': (-2 ** 31, 2 ** 31 - 1), 'unsigned': (0, 2 ** 32 - 1),
        'long long': (-2 ** 63, 2 ** 63 - 1), 'unsigned long long': (0, 2 ** 64 - 1),
        '_Bool': (0, 1)}

HEADER = r'''
typedef unsigned long long u64;
#ifdef _MSC_VER /* the reference DLL (clang); cimo does not need it */
#define API __declspec(dllexport)
#else
#define API
#endif
static u64 hu(u64 h, u64 v) { return (h ^ v) * 0x100000001b3ULL; }
static u64 hf(u64 h, float v) { union { float f; unsigned u; } x; x.f = v; return hu(h, x.u); }
static u64 hd(u64 h, double v) { union { double f; u64 u; } x; x.f = v; return hu(h, x.u); }
static u64 hp(u64 h, void *v) { return hu(h, (u64)(__SIZE_TYPE__)v); }
#include <stdarg.h>
'''


class Gen:
    def __init__(self, seed):
        self.r = random.Random(seed)
        self.structs = []  # (name, [(type, member name, array length or 0)])

    # --- types ---------------------------------------------------------------------------

    def struct(self):
        r = self.r
        kind = r.random()
        if kind < 0.3:  # homogeneous float aggregate (1-4 members, possibly an array)
            t = r.choice(['float', 'double'])
            n = r.randint(1, 5)
            members = [(t, 'm0', n)] if r.random() < 0.3 else [(t, 'm%d' % i, 0) for i in range(n)]
        elif kind < 0.45:  # bytes of an odd size
            members = [('unsigned char', 'm0', r.choice([1, 2, 3, 5, 6, 7, 9, 12, 15, 17, 33]))]
        else:
            members = []
            for i in range(r.randint(1, 6)):
                if self.structs and r.random() < 0.1:
                    members.append(('struct ' + r.choice(self.structs)[0], 'm%d' % i, 0))
                else:
                    t = r.choice([s for s in SCALARS if s != '_Bool'])
                    members.append((t, 'm%d' % i, r.choice([0, 0, 0, 2, 3])))
        name = 'S%d' % len(self.structs)
        self.structs.append((name, members))
        return 'struct ' + name

    def type(self, allow_struct=True):
        if allow_struct and self.r.random() < 0.35:
            if self.structs and self.r.random() < 0.4:
                return 'struct ' + self.r.choice(self.structs)[0]
            return self.struct()
        return self.r.choice(SCALARS)

    def members(self, t):
        return dict(self.structs)[t.split()[1]]

    # --- values and hashing ---------------------------------------------------------------

    def value(self, t):
        """A C expression of type t (an initializer for structs)."""
        r = self.r
        if t.startswith('struct '):
            parts = []
            for mt, _, n in self.members(t):
                if n:
                    parts.append('{%s}' % ', '.join(self.value(mt) for _ in range(n)))
                else:
                    parts.append(self.value(mt))
            return '{%s}' % ', '.join(parts)
        if t == 'float':
            return '%.1ff' % (r.randint(-4000, 4000) / 8)
        if t == 'double':
            return '%r' % (r.randint(-10 ** 6, 10 ** 6) / 64)
        if t == 'void *':
            return '(void *)%#xULL' % r.randint(1, 2 ** 48)
        lo, hi = INTS[t]
        v = r.randint(lo, hi)
        if t == 'long long' and v == lo:
            return '(-0x7fffffffffffffffLL - 1)'
        return '(%s)%d%s' % (t, v, 'ULL' if v > 2 ** 62 else 'LL')

    def hash(self, t, e, out):
        """Statements adding expression e (of type t) to u64 h."""
        if t.startswith('struct '):
            for mt, m, n in self.members(t):
                if n:
                    for i in range(n):
                        self.hash(mt, '%s.%s[%d]' % (e, m, i), out)
                else:
                    self.hash(mt, '%s.%s' % (e, m), out)
        elif t == 'float':
            out.append('h = hf(h, %s);' % e)
        elif t == 'double':
            out.append('h = hd(h, %s);' % e)
        elif t == 'void *':
            out.append('h = hp(h, %s);' % e)
        else:
            out.append('h = hu(h, (u64)%s);' % e)

    def make(self, t, src, out):
        """Statements setting variable r (of type t) from u64 src."""
        if t.startswith('struct '):
            k = [0]

            def mix():
                k[0] += 1
                return '(%s * %d + %d)' % (src, 2 * k[0] + 1, k[0])
            for mt, m, n in self.members(t):
                for i in range(n or 1):
                    lv = 'r.%s%s' % (m, '[%d]' % i if n else '')
                    self.set(mt, lv, mix(), out)
        else:
            self.set(t, 'r', src, out)

    def set(self, t, lv, src, out):
        if t.startswith('struct '):
            for mt, m, n in self.members(t):
                for i in range(n or 1):
                    self.set(mt, '%s.%s%s' % (lv, m, '[%d]' % i if n else ''), '(%s ^ %d)' % (src, i), out)
        elif t in ('float', 'double'):
            out.append('%s = (%s)(int)(%s %% 100000) / 8;' % (lv, t, src))
        elif t == 'void *':
            out.append('%s = (void *)(__SIZE_TYPE__)(%s & 0xffffffffffffULL);' % (lv, src))
        else:
            out.append('%s = (%s)%s;' % (lv, t, src))

    # --- one test ---------------------------------------------------------------------------

    def signature(self, n):
        r = self.r
        ret = self.type()
        variadic = r.random() < 0.25
        nparams = r.randint(1, 4) if variadic else r.randint(0, 14)
        params = [self.type() for _ in range(nparams)]
        if variadic and not params:
            params = ['int']
        vargs = []
        if variadic:  # promoted scalar types only (cimo: no struct va_arg)
            vargs = [r.choice(['int', 'unsigned', 'long long', 'unsigned long long', 'double',
                               'void *']) for _ in range(r.randint(0, 12))]
        return ret, params, variadic, vargs

    def test(self, n):
        ret, params, variadic, vargs = self.signature(n)
        plist = ', '.join('%s p%d' % (t, i) for i, t in enumerate(params)) or 'void'
        if variadic:
            plist += ', ...'
        ftype = '%s (*)(%s)' % (ret, plist)
        body = ['u64 h = %d;' % (n + 1)]
        for i, t in enumerate(params):
            self.hash(t, 'p%d' % i, body)
        if variadic:
            body.append('va_list ap; va_start(ap, p%d);' % (len(params) - 1))
            for i, t in enumerate(vargs):
                self.hash(t, 'va_arg(ap, %s)' % t, body)
            body.append('va_end(ap);')
        body.append('%s r;' % ret)
        self.make(ret, 'h', body)
        body.append('return r;')
        callee = 'API %s @f%d(%s) {\n  %s\n}\n' % (ret, n, plist, '\n  '.join(body))
        decls = []
        args = []
        for i, t in enumerate(params + vargs):
            if t.startswith('struct '):
                decls.append('%s a%d = %s;' % (t, i, self.value(t)))
                args.append('a%d' % i)
            else:
                args.append(self.value(t))
        cbody = decls + ['%s v = f(%s);' % (ret, ', '.join(args)), 'u64 h = 7;']
        self.hash(ret, 'v', cbody)
        cbody.append('return h;')
        caller = 'API u64 @call%d(%s) {\n  %s\n}\n' % (n, ftype.replace('(*)', '(*f)'),
                                                     '\n  '.join(cbody))
        glue = ('%s ref_f%d(%s);\nu64 ref_call%d(%s);\n'
                'API u64 x%d(void) { return call%d(ref_f%d); }\n'
                'API u64 y%d(void) { return ref_call%d(f%d); }\n'
                'API u64 z%d(void) { return call%d(f%d); }\n'
                % (ret, n, plist, n, ftype.replace('(*)', '(*)'), n, n, n, n, n, n, n, n, n))
        return callee + caller, glue


def sources(seed, count):
    """(reference source, cimo source) for `count` random signatures. The reference's functions
    are ref_fN and ref_callN; cimo's module has fN, callN and, calling across:
    xN = callN(ref_fN), yN = ref_callN(fN), zN = callN(fN)."""
    g = Gen(seed)
    tests = [g.test(n) for n in range(count)]
    structs = []
    for name, members in g.structs:
        ms = ' '.join('%s %s%s;' % (t, m, '[%d]' % n if n else '') for t, m, n in members)
        structs.append('struct %s { %s };' % (name, ms))
    common = HEADER + '\n'.join(structs) + '\n\n'
    code = '\n'.join(c for c, _ in tests)
    return (common + code.replace('@', 'ref_'),
            common + code.replace('@', '') + '\n' + '\n'.join(gl for _, gl in tests))


if __name__ == '__main__':
    import sys
    print(sources(int(sys.argv[1]) if len(sys.argv) > 1 else 1,
                  int(sys.argv[2]) if len(sys.argv) > 2 else 20)[1])
