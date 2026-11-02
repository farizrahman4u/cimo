"""What a module says about itself: its source, options, C types and the data it can change."""

import cimo

SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
struct p { int x; };
typedef enum { A, B } kind;
const double T[2] = {1, 2};
const char *const NAMES[1] = {"x"};
double G[2] = {1, 2};
static int hidden = 3;
const char *moved = "lit";
double dsum(const double *a, int64_t n) { double s = 0; for (int64_t i = 0; i < n; i++) s += a[i] * T[0]; return s; }
void blur(int h, int w, const float src[h][w], float dst[h][w]) { dst[0][0] = src[0][0]; }
uint8_t bytes(uint8_t *p, char c, short s, unsigned u, _Bool b, kind k, void *v) { return 0; }
float by_value(struct p a) { return a.x; }
int counter(int x) { static int n; n += x; return n + hidden; }
int show(const char *fmt, ...) { return 0; }
const int *const_result(void) { return 0; }
static int not_exported(int x) { return x + 1; }
int uses(int x) { return not_exported(x) + (int)G[0] + moved[0]; }
"""


def test_source_and_options():
    m = cimo.compile(SOURCE, defines={'N': '3'})
    assert m.source == SOURCE and m.options == {'defines': (('N', '3'),)}
    assert cimo.compile('int f(void) { return 1; }').options == {}
    assert m.dsum.module is m
    assert cimo.compile('int f(void) { return 1; }', fma=False, wrapv=True).options == {'fma': False, 'wrapv': True}


def test_signatures():
    m = cimo.compile(SOURCE)
    assert m.signature('dsum') == ('double', ('const double *', 'int64_t'))
    assert m.signature('blur') == ('void', ('int32_t', 'int32_t', 'const float *', 'float *'))
    ret, params = m.signature('bytes')
    assert ret == 'uint8_t' and params[0] == 'uint8_t *' and params[1] in ('int8_t', 'uint8_t') and params[2:] == ('int16_t', 'uint32_t', '_Bool', 'uint32_t', 'void *')
    assert m.signature('by_value') == ('float', ('struct',))
    assert m.signature('show')[1][1] == '...' and m.signature('show')[1][0].startswith('const ')
    assert m.signature('const_result') == ('const int32_t *', ())


def test_mutable_data():
    data = cimo.compile(SOURCE).mutable_data
    assert sorted(data) == ['G', 'hidden', 'moved', 'n']      # n is the static local; string literals aren't there
    assert cimo.compile('const int K = 3; int f(void) { return K; }').mutable_data == []


def test_the_module_still_works():
    m = cimo.compile(SOURCE)
    m.signature('dsum')
    assert m.counter(2) == 5 and m.counter(2) == 7
