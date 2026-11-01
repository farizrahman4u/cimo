"""Runtime specialization: turn a formula typed at run time into C, compile it, and run it over a
numpy array; cimo is the JIT, no code generator per target needed."""

import math

import numpy as np

import cimo


def compile_formula(expr):
    """expr: a C expression in x (and the <math.h> functions). Returns f(array) -> array."""
    k = cimo.compile(r"""
#include <math.h>
void apply(long n, const double *restrict xs, double *restrict ys) {
    for (long i = 0; i < n; i++) {
        double x = xs[i];
        ys[i] = %s;
    }
}
""" % expr)

    def f(xs):
        ys = np.empty_like(xs)
        k.apply(len(xs), xs, ys)
        return ys
    return f


xs = np.linspace(-4, 4, 1_000_001)
for expr, ref in [('x * x - 2 * x + 1', lambda x: x * x - 2 * x + 1),
                  ('sin(x) * exp(-x * x / 2)', lambda x: math.sin(x) * math.exp(-x * x / 2)),
                  ('x > 0 ? sqrt(x) : -cbrt(-x)', lambda x: math.sqrt(x) if x > 0 else -(-x) ** (1 / 3))]:
    ys = compile_formula(expr)(xs)
    err = max(abs(ys[i] - ref(xs[i])) for i in range(0, len(xs), 9973))
    print('%-28s f(1.5) = %.6f   max error vs Python %.1e' % (expr, ys[687500], err))
