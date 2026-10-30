import numpy as np

import cimo

k = cimo.compile(r"""
double sum(const double *x, long n) {
    double s = 0;
    for (long i = 0; i < n; i++)
        s += x[i];
    return s;
}
""")

x = np.arange(1_000_000, dtype=np.float64)
print(k.sum(x, len(x)), x.sum())
