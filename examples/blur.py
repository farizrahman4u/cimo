"""The README's stencil: a 5-point blur over a numpy image, in place, then split across threads
(kernels release the GIL), checked against numpy."""

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import cimo

k = cimo.compile(r"""
void blur(int block, int nblocks, int h, int w, const float src[h][w], float dst[h][w]) {
    int rows = (h - 2 + nblocks - 1) / nblocks;
    int lo = 1 + block * rows;
    int hi = lo + rows < h - 1 ? lo + rows : h - 1;
    for (int i = lo; i < hi; i++)
        for (int j = 1; j < w - 1; j++)
            dst[i][j] = 0.2f * (src[i][j] + src[i-1][j] + src[i+1][j]
                                          + src[i][j-1] + src[i][j+1]);
}
""")

h, w = 4096, 4096
src = np.random.rand(h, w).astype(np.float32)
want = src.copy()
want[1:-1, 1:-1] = 0.2 * (src[1:-1, 1:-1] + src[:-2, 1:-1] + src[2:, 1:-1]
                          + src[1:-1, :-2] + src[1:-1, 2:])

dst = src.copy()
t = time.perf_counter()
k.blur(0, 1, h, w, src, dst)
one = time.perf_counter() - t
assert np.allclose(dst, want, atol=1e-6)

n = 16
dst = src.copy()
with ThreadPoolExecutor() as pool:
    t = time.perf_counter()
    list(pool.map(lambda b: k.blur(b, n, h, w, src, dst), range(n)))
    many = time.perf_counter() - t
assert np.allclose(dst, want, atol=1e-6)
print('blur %dx%d: %.1f ms on one thread, %.1f ms on %d threads' % (h, w, one * 1e3, many * 1e3, n))
