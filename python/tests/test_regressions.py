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
    assert three.count('str\tw') == 3  # no store merged away
    discard = asm.split('discard:')[1].split('ret')[0]
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
