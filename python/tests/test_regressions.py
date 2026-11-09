"""Small programs for bugs found by the third-party corpora (test_corpora.py), so that they
stay fixed without the corpora. Several are inherited from cproc or QBE."""

import sys

import pytest

import cimo
from cimo.driver import KernelError


def test_block_scope_extern_hides_a_local():
    # cproc took the local's (absent) linkage and crashed (gcc scope-1.c)
    m = cimo.compile('int v = 3;\n'
                     'int f(void) { int v = 4; { extern int v; return v; } }\n')
    assert m.f() == 3


def test_global_address_plus_large_offset():
    # QBE's isel put the offset in an add immediate (gcc 960321-1.c, 20030209-1.c)
    m = cimo.compile('char a[10] = "deadbeef";\n'
                     'char acc(long i) { return a[i - 2000000000L]; }\n'
                     'char acc2(long i) { return a[i + 5000]; }\n')
    assert m.acc(2000000000) == ord('d')
    assert m.acc2(-4999) == ord('e')


def test_reading_an_uninitialized_local_compiles():
    # QBE: "slot is read but never stored to" (gcc pr34099.c)
    m = cimo.compile('int f(int b, int c) { int x; if (b) return x & c; return 1; }\n'
                     'double g(int b) { double x; if (b) return x * 0; return 1; }\n')
    assert m.f(1, 0) == 0 and m.f(0, 0) == 1 and m.g(0) == 1


def test_uninitialized_struct_passed_by_value():
    # QBE killed the never-stored slot and passed address 0 (gcc va-arg-14.c)
    m = cimo.compile('struct S { long a[5]; };\n'
                     'long f(struct S s) { return 3; }\n'
                     'long g(void) { struct S s; return f(s); }\n')
    assert m.g() == 3


def test_variably_modified_typedef_size_at_declaration():
    # cproc computed the size at the first use, in one branch (gcc 20040411-1.c)
    m = cimo.compile('''
int f(int i, int j) { typedef int c[i + 2]; if (j == 2) return sizeof(c); return sizeof(c) * 3; }
int g(int n) { typedef char c[n]; n += 5; return sizeof(c); }
int h(int n) { int s = 0; for (int i = 1; i <= n; i++) { typedef char c[i]; s += sizeof(c); }
               return s; }
''')
    assert m.f(20, 3) == 66 * 4 and m.f(20, 2) == 22 * 4
    assert m.g(3) == 3 and m.h(4) == 10


def test_parameter_array_length_is_evaluated():
    # C11 6.9.1p10; cproc dropped it with the array type (gcc 970217-1.c)
    m = cimo.compile('int sub(int i, int array[i++]) { return i; }\n'
                     'int g;\n'
                     'int h(void) { return ++g; }\n'
                     'int f(int n, double a[h()][n]) { return g * 10 + (int)(sizeof(*a) / 8); }\n')
    assert m.sub(10, 0) == 11
    assert m.f(4, 0) == 14


def test_zero_length_array_local():
    # a size-0 struct was taken for a VLA (gcc va-arg-22.c)
    m = cimo.compile('typedef struct { char x[0]; } A0;\n'
                     'int f(void) { A0 a; int b = 5; return sizeof a + b; }\n')
    assert m.f() == 5


def test_gnu_attributes_after_pointer():
    m = cimo.compile('struct S { int a; };\n'
                     'static struct S s = {7};\n'
                     'struct S * __attribute__((noinline, noclone)) get(void) { return &s; }\n'
                     'int * __attribute__((unused)) const p = 0;\n'
                     'int f(void) { return get()->a + (p == 0); }\n')
    assert m.f() == 8


def test_builtin_library_functions(capsys):
    m = cimo.compile(r'''
int f(char *d, const char *s, double x) {
  __builtin_memcpy(d, s, 4); d[4] = 0;
  if (__builtin_memcmp(d, s, 4) || __builtin_strcmp(d, "abcd")) __builtin_abort();
  __builtin_prefetch(d, 0);
  return __builtin_strlen(s) + (int)__builtin_sqrt(x) + __builtin_printf("%s\n", d);
}
int g(void) { __builtin_abort(); }
''')
    import ctypes
    b = ctypes.create_string_buffer(8)
    assert m.f(b, b'abcdef', 16.0) == 6 + 4 + 5
    assert capsys.readouterr().out == 'abcd\n'
    with pytest.raises(KernelError):
        m.g()


def test_gcc_float_macros():
    m = cimo.compile('double f(void) { return __DBL_MAX__; }\n'
                     'float g(void) { return __FLT_MIN__; }\n'
                     'double e(void) { return __DBL_EPSILON__; }\n'
                     'int h(void) { return __DBL_MANT_DIG__ * 1000 - __DBL_MIN_EXP__; }\n')
    assert m.f() == sys.float_info.max and m.e() == sys.float_info.epsilon
    assert m.g() == 2.0 ** -126 and m.h() == 53 * 1000 + 1021


def test_load_of_a_non_utf8_source(tmp_path):
    p = tmp_path / 'latin1.c'
    p.write_bytes(b'/* caf\xe9 */ int f(void) { return 7; }\n')
    assert cimo.load(str(p)).f() == 7


# found by Csmith (fuzz_csmith.py)

def test_unsigned_long_with_long_long_is_unsigned():
    # cproc's common type of unsigned long and long long was long long (C11 6.3.1.8: unsigned
    # long long), so uint64_t compared with a LL constant was signed
    m = cimo.compile('#include <stdint.h>\n'
                     'static uint64_t g = 0xB67E062F4CF3161CULL;\n'
                     'int a(void) { return 0x49E8EF73C2DC727CLL <= g; }\n'
                     'int b(void) { return (g + 1LL) > 0; }\n'
                     'unsigned long long c(void) { return g / 3LL; }\n')
    assert m.a() == 1 and m.b() == 1 and m.c() == 0xB67E062F4CF3161C // 3


def test_logical_operators_fold_to_int_0_or_1():
    # cproc folded `5L || x` to the long 5 and `1 && 7` to 7; in a comparison inside && this
    # gave IR the backend rejected
    m = cimo.compile('unsigned char x; int y = 2;\n'
                     'int f(long l) { return ((y != ((5L || l) < (x = 255))) && y); }\n'
                     'int k(void) { return (1 && 7) + (0 || 0.5) * 10 + (-0.0 || 0) * 100; }\n'
                     'static int arr[(2 || 0) + (3 && 4)];\n'
                     'int n(void) { return sizeof arr / sizeof arr[0]; }\n')
    assert m.f(0) == 1 and m.k() == 11 and m.n() == 2


def test_function_with_more_constants_than_temporaries():
    # the port indexed fn.tmp with a constant's number (IndexError); and constant lookup was
    # quadratic (minutes for a large local initializer)
    import time
    elems = ','.join('{%d,{%d,%dL,%d},%dU,{1,2,3}}' % (k, k, k * 7, k % 100, k)
                     for k in range(512))
    t = time.time()
    m = cimo.compile('struct S1 { int a; long b; short c; };\n'
                     'struct S2 { char x; struct S1 s; unsigned y; struct S1 t; };\n'
                     'int f(int i) { const struct S2 l[512] = {%s}; return l[i].s.a + l[i].t.c; }\n'
                     % elems)
    assert m.f(300) == 303
    assert time.time() - t < 30


def test_pointers_differing_in_signedness_or_inner_qualifiers():
    # accepted as gcc does (cproc: incompatible); Csmith generates both
    m = cimo.compile('unsigned g = 5; int h = -1;\n'
                     'int f(void) { int *p = &h; unsigned *q = p; const int *const *pp = 0;\n'
                     '  int **r = 0; unsigned char *s = (char *)0;\n'
                     '  return (*q == 0xffffffffu) + (pp == r) + (s == 0) + (q != &g); }\n')
    assert m.f() == 4
    with pytest.raises(cimo.CompileError):
        cimo.compile('int *f(long *p) { return p; }')


def test_bit_fields_promote_through_comma_and_conditional():
    # cproc: (0, s.f2) and (x ? s.f2 : s.f2) were unsigned int, not promoted to int as a
    # 15-bit field is (gcc and clang agree); and c ? ch : ch kept char (C11 6.5.15p5)
    m = cimo.compile('struct S1 { unsigned f0 : 10; signed f1 : 15; unsigned f2 : 15; };\n'
                     'int g5 = -8; char c1 = 1;\n'
                     'int b(void) { struct S1 s = {20, -17, 4}; return (0, s.f2) < g5; }\n'
                     'int d(int x) { struct S1 s = {20, -17, 4}; return (x ? s.f2 : s.f2) < g5; }\n'
                     'int e(int x) { return sizeof(x ? c1 : c1); }\n')
    assert m.b() == 0 and m.d(1) == 0 and m.e(1) == 4


def test_bit_fields_promote_through_assignment_and_increment():
    # cproc: (s.f7 = 0), (s.f7 += 1) and s.f7++ were unsigned int, not promoted to int as a
    # 2-bit field is, so comparing them with a negative int went unsigned (Csmith: a branch
    # taken the other way, into a loop that never ends); a 32-bit field stays unsigned int
    m = cimo.compile('struct S { unsigned f7 : 2; unsigned w : 32; signed s : 3; } g = {1, 1, 1};\n'
                     'int neg = -5;\n'
                     'int a(void) { return (g.f7 = 0) >= neg; }\n'
                     'int b(void) { return (g.f7 += 1) >= neg; }\n'
                     'int c(void) { return g.f7++ >= neg; }\n'
                     'int d(void) { return (g.w = 0) >= neg; }\n'
                     'int e(void) { return (g.f7 = 7) - 4; }\n'
                     'int f(void) { return g.s = 7; }\n')
    assert (m.a(), m.b(), m.c(), m.d(), m.e(), m.f()) == (1, 1, 1, 0, -1, -1)


# found by the semantics model (test_semantics.py)

def test_bool_increment_and_decrement():
    # cproc stored x + 1 and x - 1 unconverted: 2 and 255
    m = cimo.compile('int c(void) { _Bool x = 1; x++; return x; }\n'
                     'int d(void) { _Bool x = 0; x--; return x; }\n'
                     'int e(void) { _Bool x = 1; return x++ * 10 + x; }\n'
                     'struct S { _Bool b : 1; };\n'
                     'int g(void) { struct S s = {1}; s.b++; s.b++; return s.b; }\n')
    assert (m.c(), m.d(), m.e(), m.g()) == (1, 1, 11, 1)


def test_constant_conversions_to_bool_and_unsigned():
    # cproc folded (_Bool)2 to 0 (low bit), (_Bool)0.5 to 0, and rejected (_Bool)-0.5 and
    # (unsigned)-0.5 (the integer part, 0, is representable)
    m = cimo.compile('static _Bool g = 2; static _Bool h = (_Bool)-0.5; static _Bool z = (_Bool)0.0;\n'
                     'int a(void) { return g * 100 + h * 10 + z; }\n'
                     'int b(void) { return (_Bool)2 + (_Bool)(float)0.5f * 10; }\n'
                     'unsigned u(void) { return (unsigned)-0.75; }\n')
    assert (m.a(), m.b(), m.u()) == (110, 11, 0)


def test_constant_conditional_with_a_bit_field_operand():
    # folding `1 ? b.f : 0u` left the bit-field, which then promoted to int again
    m = cimo.compile('struct { unsigned f : 14; } b1;\n'
                     'long long u(void) { b1.f = 9182; return 1 - (1 ? b1.f : 0u); }\n'
                     'long long y(void) { b1.f = 9182; return 1 - (1 ? b1.f : b1.f); }\n')
    assert m.u() == 4294958115 and m.y() == -9181


def test_jnz_on_a_narrowed_constant():
    # QBE took `jnz 0x8000000000000` (a long constant converted to int, so 0) as true
    m = cimo.compile('int c(int x) { return x && (int)2251799813685248LL; }\n'
                     'int h(int x) { if ((int)2251799813685248LL) return 1; return 0; }\n'
                     'int n(int x) { return x && (int)4294967297LL; }\n')
    assert (m.c(1), m.h(1), m.n(1)) == (0, 0, 1)


VOLATILE = r"""
volatile int flag;
int spin(volatile int *p, int n) { int c = 0; for (int i = 0; i < n; i++) c += *p; return c; }
void three(volatile int *p) { *p = 1; *p = 2; *p = 3; }
int local(void) { volatile int x = 5; x = x + 1; return x; }
struct S { int a : 3; volatile int b : 5; } s;
int bits(void) { s.b = 7; s.b++; return s.b; }
struct T { int x, y; };
volatile struct T vt;
int copy(void) { struct T t = {1, 2}; vt = t; struct T u = vt; return u.x + u.y; }
int discard(volatile int *p) { *p; return 0; }
"""


def test_volatile_accesses():
    # cproc rejected volatile stores (gcc torture: 31 programs) and treated volatile loads as
    # plain ones; every access is now performed, in order
    m = cimo.compile(VOLATILE)
    import ctypes
    a = (ctypes.c_int * 1)(4)
    assert m.spin(a, 3) == 12 and m.local() == 6 and m.bits() == 8 and m.copy() == 3
    m.three(a)
    assert a[0] == 3 and m.discard(a) == 0
    from cimo import driver
    asm = driver._build(VOLATILE)[0]
    three = asm.split('three:')[1].split('ret')[0]
    discard = asm.split('discard:')[1].split('ret')[0]
    import re
    if driver.TARGET.startswith("amd64"):
        assert len(re.findall(r'\tmovl \$\d, \(%', three)) == 3  # no store merged away
        assert len(re.findall(r'\tmovl \(%\w+\), %', discard)) == 1  # the read happens
    else:
        assert three.count('str\tw') == 3  # no store merged away
        assert discard.count('ldr\tw17') == 1  # the unused volatile read happens


def test_vlas_are_freed_when_their_scope_ends():
    # cproc kept every VLA until the function returned, so one in a loop, or before a
    # backward goto, overflowed the stack (gcc 20040811-1.c, pr43220.c, vla-dealloc-1.c)
    m = cimo.compile(r"""
void *volatile p;
long loop(long n) { long s = 0; for (long i = 0; i < n; i++) { int x[i % 1000 + 1]; x[0] = i; p = x; s += x[0]; } return s; }
long back(void) { int n = 0; lab:; int x[n % 1000 + 1]; x[0] = 1; x[n % 1000] = 2; p = x; n++; if (n < 1000000) goto lab; return n; }
long brk(long n) { long c = 0; while (1) { int a[n]; a[0] = 1; for (int j = 0; j < 3; j++) { int b[n + j]; b[0] = j; if (j == 1) continue; c += b[0]; } if (++c > 300000) break; } return c; }
long fwd(long n) { long c = 0; for (long i = 0; i < n; i++) { { int a[i % 100 + 1]; a[0] = 1; p = a; if (i & 1) goto out; } out: c++; } return c; }
long nested(long n) { long t = 0; for (long i = 0; i < n; i++) { int a[i % 50 + 1]; a[0] = 3; { int b[i % 70 + 1]; b[0] = 4; t += a[0] + b[0]; } int c[5]; c[0] = 1; t += c[0] + a[0]; } return t; }
long sw(long n) { long t = 0; for (long i = 0; i < n; i++) { switch (i % 3) { case 0: { int a[i % 64 + 1]; a[0] = 1; t += a[0]; break; } default: t += 2; } } return t; }
""")
    assert m.loop(2000000) == 1999999000000
    assert m.back() == 1000000
    assert m.brk(10) == 300003
    assert m.fwd(1000000) == 1000000
    assert m.nested(400000) == 4400000
    assert m.sw(1000000) == 1666666


VA_STRUCTS = r"""
#include <stdarg.h>
struct s1 { char c; }; struct s3 { char c[3]; }; struct s12 { int a, b, c; };
struct s16 { long long a, b; }; struct s24 { long long a, b, c; };
struct f2 { float x, y; }; struct d4 { double a, b, c, d; }; struct f3 { float x[3]; };
struct mix { double d; long long l; }; struct a16 { _Alignas(16) long long a; };
struct z { char x[0]; };
double vsum(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt); double t = 0; int k = 1;
    for (const char *p = fmt; *p; p++, k++) {
        switch (*p) {
        case 'i': t += k * va_arg(ap, int); break;
        case 'd': t += k * va_arg(ap, double); break;
        case '1': { struct s1 v = va_arg(ap, struct s1); t += k * v.c; break; }
        case '3': { struct s3 v = va_arg(ap, struct s3); t += k * (v.c[0] + 2 * v.c[1] + 3 * v.c[2]); break; }
        case 'c': { struct s12 v = va_arg(ap, struct s12); t += k * (v.a + 2 * v.b + 3 * v.c); break; }
        case 'g': { struct s16 v = va_arg(ap, struct s16); t += k * (v.a + 2 * v.b); break; }
        case 'h': { struct s24 v = va_arg(ap, struct s24); t += k * (v.a + 2 * v.b + 3 * v.c); break; }
        case 'f': { struct f2 v = va_arg(ap, struct f2); t += k * (v.x + 2 * v.y); break; }
        case 'D': { struct d4 v = va_arg(ap, struct d4); t += k * (v.a + 2 * v.b + 3 * v.c + 4 * v.d); break; }
        case 'F': { struct f3 v = va_arg(ap, struct f3); t += k * (v.x[0] + 2 * v.x[1] + 3 * v.x[2]); break; }
        case 'm': { struct mix v = va_arg(ap, struct mix); t += k * (v.d + 2 * v.l); break; }
        case 'a': { struct a16 v = va_arg(ap, struct a16); t += k * v.a; break; }
        case 'z': { struct z v = va_arg(ap, struct z); (void)v; break; }
        }
    }
    va_end(ap);
    return t;
}
static struct s1 S1 = {5}; static struct s3 S3 = {{1, 2, 3}}; static struct s12 S12 = {1, 2, 3};
static struct s16 S16 = {100, 200}; static struct s24 S24 = {1000, 2000, 3000};
static struct f2 F2 = {1.5f, 2.25f}; static struct d4 D4 = {0.5, 1.5, 2.5, 3.5};
static struct f3 F3 = {{1, 2, 4}}; static struct mix MX = {0.125, 77}; static struct a16 A16 = {4242};
double run(int which) {
    struct z Z;
    switch (which) {
    case 0: return vsum("1cghfDFma", S1, S12, S16, S24, F2, D4, F3, MX, A16);
    case 1: return vsum("iiiiiiicg", 1, 2, 3, 4, 5, 6, 7, S12, S16);
    case 2: return vsum("ddddddfD", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, F2, D4);
    case 3: return vsum("ddddddddDf3", 1., 2., 3., 4., 5., 6., 7., 8., D4, F2, S3);
    case 4: return vsum("iiiiiiahaga", 1, 2, 3, 4, 5, 6, A16, S24, A16, S16, A16);
    case 5: return vsum("zizFz", Z, 9, Z, F3, Z);
    }
    return -1;
}
"""


def test_va_arg_of_structs():
    # cproc rejected va_arg with a struct type (gcc torture: 14 programs); every kind of struct,
    # with register save areas running out mid-way (values checked against gcc)
    m = cimo.compile(VA_STRUCTS)
    want = [97243.0, 4752.0, 333.0, 643.0, 231625.0, 86.0]  # gcc's
    assert [m.run(i) for i in range(6)] == want


def test_struct_layout_and_alignment_in_calls():
    # cproc emitted QBE types without _Alignas or padding (too small: calls copied half the
    # struct) and with a byte for a zero-length array; QBE refused 16-byte alignment, put such
    # a struct in an odd register, gave a pointer to one 16-byte stack alignment, and passed a
    # zero-size struct by pointer (gcc: nothing)
    m = cimo.compile(r"""
struct a16 { _Alignas(16) long long a; };
struct b16 { _Alignas(16) long long a, b; };
struct p { char c; _Alignas(8) int x; char d; };
struct fl { int n; double d[]; };
struct z { char x[0]; };
static long long g(int x, struct a16 a, int y, struct b16 b, int z, struct a16 c, struct b16 d,
                   struct a16 e, struct b16 f, int w) {
    return x + 2 * a.a + 3 * y + 5 * b.a + 7 * b.b + 11 * z + 13 * c.a + 17 * d.a + 19 * d.b
           + 23 * e.a + 29 * f.a + 31 * f.b + 37 * w;
}
long long call(void) {
    struct a16 A = {1}, C = {2}, E = {3}; struct b16 B = {4, 5}, D = {6, 7}, F = {8, 9};
    return g(10, A, 20, B, 30, C, D, E, F, 40);
}
static int pf(struct p s, int k) { return s.c + s.x * 10 + s.d * 100 + k * 1000; }
int callp(void) { struct p s = {1, 2, 3}; return pf(s, 4); }
static long long flf(struct fl s, struct z q, long long k) { return s.n + k; }
long long callfl(void) { struct fl s = {4}; struct z q; return flf(s, q, 6); }
struct a16 zeroed(void) { struct a16 v = {0}; return v; }
""")
    assert m.call() == 2778 and m.callp() == 4321 and m.callfl() == 10


OVERFLOW = r"""
typedef unsigned long long u64;
u64 run(int k) {
    unsigned char uc; signed char sc; int i; unsigned u; long long ll; unsigned long long ull;
    float nf = __builtin_nanf(""); double nd = __builtin_nan("");
    switch (k) {
    case 0: return __builtin_add_overflow(200, 100, &uc) * 1000 + uc;
    case 1: return __builtin_sub_overflow(-100, 100, &sc) * 1000 + (unsigned char)sc;
    case 2: return __builtin_mul_overflow(0x7fffffffffffffffLL, 2, &ll) * 1000 + (u64)ll % 997;
    case 3: return __builtin_mul_overflow(-1LL, (unsigned long long)-1, &ll) * 1000 + (u64)ll % 997;
    case 4: return __builtin_sub_overflow(0u, 1, &i) * 1000 + (u64)(i + 5);
    case 5: return __builtin_add_overflow(-1, 0u, &u) * 1000 + u % 997;
    case 6: return __builtin_mul_overflow(0x100000000ULL, 0x100000000ULL, &ull) * 1000 + ull;
    case 7: return __builtin_mul_overflow_p(65536, 65536, (int)0) * 10 + __builtin_add_overflow_p(1, 2, (char)0);
    case 8: return __builtin_smul_overflow(46341, 46341, &i) * 10 + __builtin_uadd_overflow(1u, ~0u, &u);
    case 9: return __builtin_isunordered(nd, 1.0) * 100 + __builtin_isgreater(2.0, nd) * 10 + __builtin_isless(1.0f, 2.0);
    case 10: return __builtin_islessgreater(1.0, 2.0) * 100 + __builtin_isgreaterequal(nf, nf) * 10 + __builtin_islessequal(2.0, 2.0);
    case 11: return __builtin_issignaling(__builtin_nansf("")) * 100 + __builtin_issignaling(__builtin_nans("")) * 10 + __builtin_issignaling(nd);
    case 12: return __builtin_isinff(__builtin_inff()) * 10 + __builtin_isinff(1.0f);
    }
    return 0;
}
"""


def test_gnu_overflow_and_float_comparison_builtins():
    # __builtin_{add,sub,mul}_overflow(_p) for every integer type (exact result, not just the
    # common type's), the typed variants, the quiet comparisons and issignaling; values from
    # gcc (3000 random cases matched it when this was written)
    m = cimo.compile(OVERFLOW)
    assert [m.run(k) for k in range(13)] == [1044, 1056, 1959, 1001, 4, 1965, 1000, 10, 11,
                                             101, 101, 110, 10]


def test_bit_builtins_are_instructions():
    # __builtin_clz/ctz/popcount (+l, ll) and bswap16/32/64: one instruction (or rbit+clz,
    # cnt through v31) instead of a call to the runtime's portable C; values against Python
    from cimo import driver
    src = r"""
int pc(unsigned x) { return __builtin_popcount(x); }
int pcll(unsigned long long x) { return __builtin_popcountll(x); }
int cz(unsigned x) { return __builtin_clz(x); }
int czll(unsigned long long x) { return __builtin_clzll(x); }
int tz(unsigned x) { return __builtin_ctz(x); }
int tzl(unsigned long x) { return __builtin_ctzl(x); }
unsigned short b16(unsigned short x) { return __builtin_bswap16(x); }
unsigned b32(unsigned x) { return __builtin_bswap32(x); }
unsigned long b64(unsigned long x) { return __builtin_bswap64(x); }
"""
    m = cimo.compile(src)
    import random
    r = random.Random(5)
    for _ in range(300):
        x = r.getrandbits(64) >> r.randrange(64) or 1
        w = x & 0xffffffff or 1
        assert m.pc(w) == bin(w).count('1') and m.pcll(x) == bin(x).count('1')
        assert m.cz(w) == 32 - w.bit_length() and m.czll(x) == 64 - x.bit_length()
        assert m.tz(w) == (w & -w).bit_length() - 1 and m.tzl(x) == (x & -x).bit_length() - 1
        assert m.b16(x & 0xffff) == int.from_bytes((x & 0xffff).to_bytes(2, 'little'), 'big')
        # (32- and 64-bit results come back signed)
        assert m.b32(w) & 0xffffffff == int.from_bytes(w.to_bytes(4, 'little'), 'big')
        assert m.b64(x) & (2 ** 64 - 1) == int.from_bytes(x.to_bytes(8, 'little'), 'big')
    assert 'bl\t' not in driver._build(src)[0]


SPILL_FAR = r"""
#define V(X) X(0) X(1) X(2) X(3) X(4) X(5) X(6) X(7) X(8) X(9) X(10) X(11) X(12) X(13) \
    X(14) X(15) X(16) X(17) X(18) X(19)
float in[80], out[20];
int incs[20];
struct huge { int i[0x4000]; };    /* stack slots beyond a store's immediate offset */
int sink(struct huge h) { return h.i[0]; }
int big(int n) {
    struct huge h = {0};
#define D(i) float *p##i = in + i, r##i = 0; int s##i = incs[i];
    V(D)
#define L(i) r##i += *p##i; p##i += s##i;
    while (n--) { V(L) }
#define O(i) out[i] = r##i;
    V(O)
    return sink(h);
}
int check(void) {
    for (int i = 0; i < 80; i++) in[i] = i;
    for (int i = 0; i < 20; i++) incs[i] = 20;
    big(4);
    for (int i = 0; i < 20; i++)
        if (out[i] != 4 * i + 20 * 6) return 1 + i;
    return 0;
}
"""


def test_spill_to_a_far_slot_keeps_64_bits():
    # without x18 (Windows, Apple) a spill of x17 to a slot past the immediate range went
    # through s31: 4 of its 8 bytes (QBE's Apple path has the same bug; gcc pr28982b.c)
    assert cimo.compile(SPILL_FAR).check() == 0
    import platform
    if sys.platform.startswith('linux') and platform.machine() == 'aarch64':
        from cimo import driver  # Windows code generation runs here too (no variadics)
        ssa, params = driver._ir(driver._preprocess(SPILL_FAR), '<s>', False)
        m = driver.Module(driver._qbe(ssa, params, target='arm64_win'), driver._sigs(ssa))
        assert m.check() == 0


def test_long_programs_do_not_hit_the_recursion_limit():
    # the ported passes recurse as deep as the program is long: a 3000-statement dependency
    # chain overflowed Python's recursion limit in gcm, and a long chain of blocks in
    # cfg.reachrec (YARPGen seed 2075800198)
    n = 3000
    m = cimo.compile('long f(long x) {\n' + '    x = x * 3 + 1;\n' * n + '    return x;\n}\n')
    v = 5
    for _ in range(n):
        v = (v * 3 + 1) & (2 ** 64 - 1)
    assert m.f(5) & (2 ** 64 - 1) == v
    from cimo.backend import cfg

    class B:
        def __init__(self):
            self.s1 = self.s2 = None
            self.visit = 0
    blocks = [B() for _ in range(5000)]
    for a, b in zip(blocks, blocks[1:]):
        a.s1 = b
    assert cfg.reachrec(blocks[0], blocks[-1]) == 1


def test_empty_braces_for_an_array_element():
    # cproc's `{}` left the cursor on the array itself: `{{}, {}, 5, 5}` put the 5s in element 0
    # (gcc.dg/torture pr123310.c)
    m = cimo.compile('struct a { short b; };\n'
                     'struct a f[3][2] = {{}, {}, 5, 5};\n'
                     'int g[3] = {{}, 7, {}};\n'
                     'int h[] = {{}, {}, 3};\n'
                     'int get(int i, int j) { return f[i][j].b; }\n'
                     'int sum(void) { return g[0] * 100 + g[1] * 10 + g[2] + sizeof h / sizeof *h * 1000 + h[2]; }\n')
    assert [m.get(i, j) for i in range(3) for j in range(2)] == [0, 0, 0, 0, 5, 5]
    assert m.sum() == 3073


def test_bitint_wraps_at_its_width():
    # a negative value to a wider unsigned _BitInt, and unary minus of an unsigned _BitInt,
    # wrap at the _BitInt's width (gcc.dg/torture bitint-1.c, bitint-2.c)
    m = cimo.compile('long conv(_BitInt(6) a) { unsigned _BitInt(9) b = a; return b; }\n'
                     'long conv2(int a) { unsigned _BitInt(40) b = a; return b; }\n'
                     'long neg(void) { return -1uwb; }\n'
                     'long neg2(void) { return -13uwb; }\n')
    assert m.conv(-25) == 487 and m.conv2(-1) == 2 ** 40 - 1
    assert m.neg() == 1 and m.neg2() == 3


def test_overflow_builtins_into_a_bitint():
    # the result's width, not its size: 2^30 * 2 overflows an unsigned _BitInt(31)
    # (gcc.dg/torture pr116135.c)
    m = cimo.compile('int f(unsigned long long a, unsigned long long b, long *v) {\n'
                     '    unsigned _BitInt(31) r; int o = __builtin_mul_overflow(a, b, &r); *v = r; return o; }\n'
                     'int g(int a, int b, long *v) {\n'
                     '    _BitInt(5) r; int o = __builtin_add_overflow(a, b, &r); *v = r; return o; }\n')
    import ctypes
    v = ctypes.c_long()
    assert m.f(1 << 30, 2, v) == 1 and v.value == 0
    assert m.f(1 << 15, (1 << 15) - 1, v) == 0 and v.value == (1 << 30) - (1 << 15)
    assert m.g(10, 5, v) == 0 and v.value == 15
    assert m.g(10, 6, v) == 1 and v.value == -16


def test_struct_with_a_pointer_to_const_member_is_assignable():
    # a pointer type's qualifiers are its target's: `const struct t *p` is not a const member
    # (gcc.dg/torture pr55305.c)
    m = cimo.compile('struct m { const int *p; int *const q[0 + 1 - 1 + 1]; };\n'
                     'struct w { const char *s; int n; };\n'
                     'int f(void) { struct w a = {"x", 3}, b; b = a; return b.n; }\n')
    assert m.f() == 3
    with pytest.raises(cimo.CompileError, match="'const' member"):
        cimo.compile('struct w { int *const p; };\n'
                     'void f(struct w *a, struct w *b) { *a = *b; }\n')


def test_float_conversion_of_a_forwarded_address():
    # QBE's fold stopped at `swtof $P` (a load forwarded the stored address): left to run
    # (gcc.dg/torture pr65077.c)
    m = cimo.compile('double P[2];\n'
                     'double f(void) { union { void *p; int i[2]; } u; u.p = P; P[0] = u.i[0]; return P[0]; }\n'
                     'long g(void) { return (unsigned)(long)P; }\n')
    import ctypes
    assert m.f() == float(ctypes.c_int32(m.g()).value)


def test_empty_asm_statements():
    # an empty template does nothing: inputs evaluated, a tied output assigned its input,
    # other outputs evaluated once (gcc's optimization barrier; gcc 990130-1.c)
    m = cimo.compile('int g, dummy;\n'
                     'int *bar(void) { ++g; return &dummy; }\n'
                     'int f(void) {\n'
                     '    int x = 5, y = 7, z = 0, w = 1;\n'
                     '    __asm__ volatile ("" ::: "memory");\n'
                     '    asm ("");\n'
                     '    __asm__ ("" : "+r" (x));\n'
                     '    __asm__ ("" : "=r" (z) : "0" (y + 1));\n'
                     '    __asm__ ("" : [o] "=g" (w) : "r" (g++), "[o]" (40 + x) : "memory", "cc");\n'
                     '    __asm__ ("" : "+r" (*bar()));\n'
                     '    __asm__ (" \\n\\t" : : "r" (x), "g" (&y));\n'
                     '    return x * 1000000 + y * 100000 + z * 10000 + w * 10 + g; }\n')
    assert m.f() == 5000000 + 700000 + 80000 + 450 + 2
    with pytest.raises(cimo.CompileError, match='inline assembly'):
        cimo.compile('int f(void) { __asm__ ("nop"); return 0; }\n')


def test_builtin_expect_evaluates_its_second_argument():
    # cproc dropped it: `__builtin_expect (c, z++)` lost the increment (gcc pr85156.c)
    m = cimo.compile('int f(int x, int z) { if (__builtin_expect(x != 0, z++)) return 7; return z; }\n')
    assert m.f(0, 10) == 11 and m.f(1, 10) == 7


def test_constant_integer_to_float_rounds_once():
    # folded through double, 2^62 + 2^38 + 1 was a tie and rounded to even (2^62) instead of up
    # (gcc.dg/torture fp-int-convert-float.c, once `mode(TI)` let it compile)
    m = cimo.compile('static float c = (1L << 62) + (1L << 38) + 1;\n'
                     'static float d = (1UL << 63) + (1UL << 39) + 1;\n'
                     'static float e = -((1L << 62) + (1L << 38) + 1);\n'
                     'double get(int i) { return i == 0 ? c : i == 1 ? d : e; }\n')
    assert m.get(0) == 2.0 ** 62 + 2.0 ** 39
    assert m.get(1) == 2.0 ** 63 + 2.0 ** 40
    assert m.get(2) == -(2.0 ** 62 + 2.0 ** 39)


def test_dereferenced_void_pointer_as_a_statement():
    # cproc emitted a load of no type for `*p` (gcc.c-torture/compile 20000701-1.c)
    m = cimo.compile('int f(void *p, volatile void *q, int i) { *p; i ? *p : *p; *q, *q; (void)*p; return i; }\n')
    assert m.f(0, 0, 7) == 7


def test_sum_with_an_address_subtracted():
    # cse's reassociation took `x - $t` for a subtraction of an integer constant and crashed
    # (gcc.c-torture/compile pr34029-2.c)
    m = cimo.compile('static const char s[] = "ab.cd", t[] = "abcde";\n'
                     'long f(void) { const char *a = __builtin_strchr(s, \'.\'); return (long)a + (1 - (long)t); }\n'
                     'long g(void) { return (long)__builtin_strchr(s, \'.\') - (long)t; }\n')
    assert m.f() == m.g() + 1


def test_parameter_array_bound_calls_a_function():
    # cproc read __func__'s declaration before making it (gcc.c-torture/compile pr77754-5.c)
    m = cimo.compile('int n, calls;\n'
                     'int fn3(void) { calls++; return n; }\n'
                     'long f(int x[][fn3()]) { return sizeof x[0]; }\n'
                     'long g(void) { n = 5; long r = f(0); n = 7; return r * 100 + f(0) + calls * 10000; }\n')
    assert m.g() == 22028


def test_pointer_to_an_array_of_unknown_size_of_vlas():
    # cproc asserted on `int (*)[][n]` (gcc.c-torture/compile pr94621.c)
    m = cimo.compile('struct S { int c, e[]; };\n'
                     'static int foo(struct S *m, int r, int c) {\n'
                     '    int (*a)[][m->c] = (int (*)[][m->c])&m->e; return (*a)[r][c]; }\n'
                     'int g(void) { static struct { int c; int e[12]; } s = {4, {0,1,2,3,4,5,6,7,8,9,10,11}};\n'
                     '    return foo((struct S *)&s, 2, 1); }\n')
    assert m.g() == 9


def test_huge_static_array():
    # the assembler wrote the .bss array's zeros out (MemoryError); it is now a size, and a module whose static
    # data can't be mapped is a CompileError (gcc.c-torture/compile pr65680.c)
    from cimo import assembler
    o = assembler.assemble('.bss\n.globl a\na:\n.zero 1200000000000000012\n')
    assert [s.size for s in o.sections if s.name == '.bss'] == [1200000000000000012]
    with pytest.raises(cimo.CompileError):
        cimo.compile('char a[1L << 60]; long f(void) { return (long)a; }')


def test_wide_strings_that_share_a_first_character_stay_apart():
    # cproc's string table keyed a literal by its first ssize bytes, ssize counting elements: L"ab" was L"ac"
    # (gcc.dg utf32-2.c)
    m = cimo.compile('typedef __WCHAR_TYPE__ wchar_t; typedef __CHAR16_TYPE__ c16;\n'
                     'const wchar_t *a = L"ab", *b = L"ac"; const c16 *c = u"ab", *d = u"ad";\n'
                     'long f(int k) { return k == 0 ? a[1] : k == 1 ? b[1] : k == 2 ? c[1] : d[1]; }\n')
    assert [chr(m.f(k)) for k in range(4)] == ['b', 'c', 'b', 'd']


def test_designators_into_a_string_initialized_array():
    # a designator past the literal's end was dropped, and one ending where the array ends replaced the string
    # (gcc.dg c99-init-1.c, pr67410.c)
    m = cimo.compile('typedef __WCHAR_TYPE__ wchar_t;\n'
                     'struct H { char I[6]; int J; } k[] = { { { "foo" }, 1 }, [0].I[2] = \'x\', [0].I[4] = \'y\' };\n'
                     'struct K { wchar_t L[6]; int M; } l[] = { { { L"foo" }, 1 }, [0].L[2] = L\'x\', [0].L[5] = L\'y\' };\n'
                     'int g(int w, int i) { return w ? l[0].L[i] : k[0].I[i]; }\n'
                     'void h(int *o) { char c[6] = { "f", [5] = \'y\', [1] = \'q\' }; for (int i = 0; i < 6; i++) o[i] = c[i]; }\n')
    assert [m.g(0, i) for i in range(6)] == [102, 111, 120, 0, 121, 0]
    assert [m.g(1, i) for i in range(6)] == [102, 111, 120, 0, 0, 121]
    import ctypes
    o = (ctypes.c_int * 6)()
    m.h(o)
    assert list(o) == [102, 113, 0, 0, 0, 121]


def test_braces_for_a_designated_subobject_initialize_all_of_it():
    # C11 6.7.9p19, as gcc: `[0] = {.a = 9}` after `[0].b = 5` leaves b 0 (gcc.dg c99-init-1.c)
    m = cimo.compile('struct P { int a, b; } p[2] = { [0] = {1, 2}, [0].b = 5, [1] = {3, 4}, [0] = {.a = 9} };\n'
                     'struct P q[2] = { [0].b = 5, [1] = {3, 4}, [0] = {} };\n'
                     'int a3[2][2] = { {1, 2}, [0][1] = 7, {3} };\n'
                     'struct Q { int x[2]; int y; } r = { .x[1] = 5, .y = 6, .x = {1} };\n'
                     'int g(int k, int i) { int *t[] = {(int *)p, (int *)q, (int *)a3, (int *)&r}; return t[k][i]; }\n'
                     'void h(int *o) { struct P l[2] = { [0] = {1, 2}, [0].b = 5, [1] = {3, 4}, [0] = {.a = 9} };\n'
                     '    for (int i = 0; i < 4; i++) o[i] = ((int *)l)[i]; }\n')
    assert [m.g(0, i) for i in range(4)] == [9, 0, 3, 4]
    assert [m.g(1, i) for i in range(4)] == [0, 0, 3, 4]
    assert [m.g(2, i) for i in range(4)] == [1, 7, 3, 0]
    assert [m.g(3, i) for i in range(3)] == [1, 0, 6]
    import ctypes
    o = (ctypes.c_int * 4)()
    m.h(o)
    assert list(o) == [9, 0, 3, 4]


def test_mode_attribute_in_a_type_name():
    # ignored in casts and sizeof (gcc.dg 20011214-1.c)
    m = cimo.compile('long f(void) { return sizeof(__attribute__((mode(QI))) int); }\n'
                     'long g(int y) { return (__attribute__((mode(QI))) int)y; }\n'
                     'long h(void) { return sizeof(int __attribute__((mode(HI)))); }\n')
    assert (m.f(), m.g(0x401), m.h()) == (1, 1, 2)


def test_pragma_pack():
    # dropped by the preprocessor: struct layouts silently differed from gcc's (gcc.dg pragma-pack-3.c)
    m = cimo.compile('#pragma pack(1)\nstruct A { char b; int c; };\n#pragma pack()\n'
                     '#pragma pack(push, 2)\nstruct B { char b; int c; double d; };\n#pragma pack(pop)\n'
                     'struct C { char b; int c; };\n'
                     '#define N 4\n#pragma pack(push, r1, N)\n#pragma pack(push, 1)\n#pragma pack(pop, r1)\n'
                     'struct D { char b; double c; };\n'
                     'struct S { char a[3];\n#pragma pack(1)\n struct T { char b; int c; } d;\n#pragma pack /*/ */ () // x\n int e; };\n'
                     '#pragma pack(2)\nunion U { char b[3]; int c; };\n#pragma pack()\n'
                     'long f(int k) { long s[] = { sizeof(struct A), sizeof(struct B), sizeof(struct C), sizeof(struct D),\n'
                     '    sizeof(struct S), sizeof(union U), _Alignof(union U), __builtin_offsetof(struct B, d) }; return s[k]; }\n')
    assert [m.f(k) for k in range(8)] == [5, 14, 8, 16, 12, 4, 2, 6]
    with pytest.raises(cimo.CompileError, match='1, 2, 4, 8 or 16'):
        cimo.compile('#pragma pack(3)\nstruct T { int a; };\n')
    with pytest.raises(cimo.CompileError, match='bit-field'):
        cimo.compile('#pragma pack(1)\nstruct T { int a : 3; };\n')


def test_universal_character_names_in_literals():
    # rejected as invalid escapes (gcc.dg utf16-1.c, utf32-1.c)
    m = cimo.compile('typedef __CHAR16_TYPE__ char16_t; typedef __CHAR32_TYPE__ char32_t;\n'
                     'const char *s = "h\\u00e1\\U0000010Dk"; char16_t c = u\'\\u2029\'; char32_t d = U\'\\U0001F600\';\n'
                     'const char16_t w[] = u"\\U0001F600x";\n'
                     'long f(int k, int i) { return k == 0 ? (unsigned char)s[i] : k == 1 ? c : k == 2 ? d : w[i]; }\n')
    assert [m.f(0, i) for i in range(6)] == [104, 195, 161, 196, 141, 107]
    assert (m.f(1, 0), m.f(2, 0)) == (0x2029, 0x1f600)
    assert [m.f(3, i) for i in range(4)] == [0xd83d, 0xde00, 0x78, 0]
    for bad in ('"\\u0041"', '"\\ud800"', '"\\u12"'):
        with pytest.raises(cimo.CompileError):
            cimo.compile('const char *s = %s;\n' % bad)


def test_vla_bounds_in_type_names_follow_the_expression():
    # cproc computed a type name's array sizes as soon as an expression of that type started, before the side
    # effects before the type name: sizeof (*(++a, (char (*)[a])0)) used a's old value; typeof didn't evaluate a
    # variably modified operand, and evaluated it once per declarator (gcc.dg typename-vla-1.c, vla-13.c, vla-15.c,
    # pr124576.c)
    m = cimo.compile('char g[1]; void *p; int i;\n'
                     'int f1(void) { int a = 1; int s = sizeof (*(++a, (char (*)[a])0)); return a * 100 + s; }\n'
                     'int f2(void) { int i = 0; int j = sizeof (typeof (*(++i, (char (*)[i])g))); return i * 100 + j; }\n'
                     'int f3(void) { int i = 0, j = 0; __typeof__((__typeof__(int[i++]) *)(j++, (void *)0)) a, b, c;\n'
                     '    return i * 10 + j; }\n'
                     'int f4(int j) { i = 10; (++i, (void)j, (int (*)[++i]){p}); return i; }\n'
                     'int f5(int n) { int a[20]; int (*q)[n] = 0;\n'
                     '    return (int)((char *)((n ? (int (*)[n])a : q) + 1) - (char *)a); }\n'
                     'int f6(int n) { int s = 0; for (int k = 1; k <= n; k++) s += sizeof(*(int (*)[k])0); return s; }\n'
                     'static char tmp[2];\n'
                     'static int h(int n, char (*x)[sizeof (*(++n, (char (*)[n])&tmp))]) { return sizeof *x; }\n'
                     'int f7(void) { return h(1, &tmp); }\n')
    assert (m.f1(), m.f2(), m.f3(), m.f4(0), m.f5(3), m.f6(3), m.f7()) == (202, 101, 11, 12, 12, 24, 2)


def test_float_constants_round_once():
    # a float constant was rounded to double, then to float: the wrong neighbor next to a halfway value
    # (gcc.dg float-exact-1.c); and _Float16's (f16)
    m = cimo.compile('float f(int k) { float t[] = {\n'
                     '    7.0064923216240853546186479164495806564013097093825788587853414194489554134293030074331909418106079101'
                     '5624999e-46f,\n'
                     '    170141193601674033557522515689509748735.0f, 170141193601674033557522515689509748736.0f,\n'
                     '    170141193601674033557522515689509748737.0f,\n'
                     '    0x1.000001p0f, 0x1.0000010000000000001p0f, 1e39f }; return t[k]; }\n')
    import struct
    bits = [struct.unpack('<I', struct.pack('<f', m.f(k)))[0] for k in range(7)]
    # halfway between 0x7f000000 and 0x7f000001: below, on (to even), above; then the same in hex
    assert bits == [0, 0x7f000000, 0x7f000000, 0x7f000001, 0x3f800000, 0x3f800001, 0x7f800000]


@pytest.mark.parametrize('target', ['amd64_sysv', 'amd64_windows'])
def test_amd64_call_to_an_address_plus_offset_and_shift_by_an_address(target):
    # `callq x+16@plt` is no operand (gcc.c-torture/compile 20010107-1.c); a shift count that is an address skipped
    # %cl and failed the 2-address assertion (pr65014.c)
    from cimo import assembler_amd64
    from test_amd64_fixes import _asm86
    text = _asm86('unsigned long x[4];\nvoid foo(void) { ((void (*)())(x + 2))(); }\n'
                  'extern int y;\nunsigned g(unsigned v) { return (v << ((__INTPTR_TYPE__)&y)) | (v >> (32 - (__INTPTR_TYPE__)&y)); }\n',
                  target)
    assert 'x+16@plt' not in text and '%cl' in text
    assembler_amd64.assemble(text)


def test_branch_after_unsigned_conversion():
    # x86-64: a branch on a comparison made before a uint64-to-double conversion (or a double-to-uint64 one) tested
    # the flags of the conversion's lowering (and, shifts, add), so `y == 0.0 ? ... : ...` took the wrong way
    # (scramjit, tests/test_uint64.py)
    src = '''#include <math.h>
#include <stdint.h>
static inline double fmod_py(double a, double b) {
    double m = fmod(a, b);
    if (m != 0.0) { if ((b < 0) != (m < 0)) m += b; }
    else m = copysign(0.0, b);
    return m;
}
double k1(uint64_t *a, double y) { uint64_t t = a[0]; return y == 0.0 ? fmod((double)t, y) : fmod_py((double)t, y); }
double k2(uint64_t *a, double y) { return fmod_py((double)a[0], y); }
double k4(int64_t *a, double y) { int64_t t = a[0]; return y == 0.0 ? fmod((double)t, y) : fmod_py((double)t, y); }
uint64_t k5(double *a, int y) { double t = a[0]; return y == 0 ? (uint64_t)t + 1 : (uint64_t)t * 2; }
float k6(uint64_t *a, float y) { uint64_t t = a[0]; return y == 0.0f ? (float)t + 1 : (float)t * 2; }
'''
    import array
    m = cimo.compile(src, cache=False)
    u, s = array.array('Q', [1]), array.array('q', [1])
    assert m.k1(u, -1.5) == -0.5 and m.k2(u, -1.5) == -0.5 and m.k4(s, -1.5) == -0.5
    big = array.array('d', [2.0 ** 63 + 2048])
    assert m.k5(big, 0) == (1 << 63) + 2049 and m.k5(big, 1) == ((1 << 63) + 2048) * 2 % (1 << 64)
    big_u = array.array('Q', [(1 << 63) + (1 << 40)])
    assert m.k6(big_u, 0.0) == float((1 << 63) + (1 << 40)) and m.k6(big_u, 1.0) == 2.0 * ((1 << 63) + (1 << 40))
