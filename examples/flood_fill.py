"""A loop numpy can't express: flood fill with an explicit stack on the module's heap
(malloc/free), counting the connected regions of a random maze, against pure Python."""

import time

import numpy as np

import cimo

k = cimo.compile(r"""
#include <stdlib.h>

/* labels each 4-connected region of open cells (grid == 0) with 1, 2, ...; returns the count */
int regions(int h, int w, const unsigned char grid[h][w], int label[h][w]) {
    int *stack = malloc(sizeof(int) * h * w), n = 0;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            label[i][j] = 0;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            if (grid[i][j] || label[i][j])
                continue;
            int top = 0;
            label[i][j] = ++n;
            stack[top++] = i * w + j;
            while (top) {
                int c = stack[--top], y = c / w, x = c % w;
                int ny[4] = {y - 1, y + 1, y, y}, nx[4] = {x, x, x - 1, x + 1};
                for (int d = 0; d < 4; d++)
                    if (ny[d] >= 0 && ny[d] < h && nx[d] >= 0 && nx[d] < w
                            && !grid[ny[d]][nx[d]] && !label[ny[d]][nx[d]]) {
                        label[ny[d]][nx[d]] = n;
                        stack[top++] = ny[d] * w + nx[d];
                    }
            }
        }
    free(stack);
    return n;
}
""")


def regions_py(grid):
    h, w = grid.shape
    label = [[0] * w for _ in range(h)]
    n = 0
    for i in range(h):
        for j in range(w):
            if grid[i, j] or label[i][j]:
                continue
            n += 1
            label[i][j] = n
            stack = [(i, j)]
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not grid[ny, nx] and not label[ny][nx]:
                        label[ny][nx] = n
                        stack.append((ny, nx))
    return n


grid = (np.random.rand(1000, 1000) < 0.45).astype(np.uint8)
label = np.empty(grid.shape, dtype=np.int32)
t = time.perf_counter()
n = k.regions(*grid.shape, grid, label)
tc = time.perf_counter() - t
t = time.perf_counter()
assert regions_py(grid) == n
tp = time.perf_counter() - t
print('%d regions: cimo %.1f ms, pure Python %.0f ms (%.0fx)' % (n, tc * 1e3, tp * 1e3, tp / tc))
