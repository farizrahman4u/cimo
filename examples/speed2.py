"""The same loop in pure Python and in C compiled by cimo, timed."""

import time

import cimo

k = cimo.compile(r"""
long longest_collatz(long n) {    /* the start below n with the longest Collatz chain */
    long best = 1, best_len = 1;
    for (long i = 1; i < n; i++) {
        long x = i, len = 1;
        while (x != 1) { x = x % 2 ? 3 * x + 1 : x / 2; len++; }
        if (len > best_len) { best_len = len; best = i; }
    }
    return best;
}
""")


def longest_collatz(n):
    best, best_len = 1, 1
    for i in range(1, n):
        x, length = i, 1
        while x != 1:
            x = 3 * x + 1 if x % 2 else x // 2
            length += 1
        if length > best_len:
            best, best_len = i, length
    return best


times = {}
for name, f in (('pure Python', longest_collatz), ('cimo', k.longest_collatz)):
    t = time.perf_counter()
    result = f(100_000)
    times[name] = time.perf_counter() - t
    print('%-12s answer %d in %7.1f ms' % (name, result, times[name] * 1e3))
print('cimo is %.0fx faster' % (times['pure Python'] / times['cimo']))
