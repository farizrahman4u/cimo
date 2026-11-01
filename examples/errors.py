"""assert, abort() and exit() in a kernel raise cimo.KernelError; the module and the interpreter
keep running."""

import array

import cimo

k = cimo.compile(r"""
#include <assert.h>
#include <stdlib.h>

double mean(const double *x, int n) {
    assert(n > 0);
    double s = 0;
    for (int i = 0; i < n; i++)
        s += x[i];
    return s / n;
}

int parse_digit(int c) {
    if (c < '0' || c > '9')
        abort();
    return c - '0';
}
""")

print('mean:', k.mean(array.array('d', [1, 2, 3, 4]), 4))
for call in (lambda: k.mean(array.array('d', [1.0]), 0), lambda: k.parse_digit(ord('x'))):
    try:
        call()
    except cimo.KernelError as e:
        print('KernelError:', e)
print('still usable:', k.parse_digit(ord('7')))
