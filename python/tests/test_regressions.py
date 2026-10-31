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
