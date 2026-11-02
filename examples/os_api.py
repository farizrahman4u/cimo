"""Kernels that use the operating system: link the host C library and use its headers (Linux and
macOS). Files, clocks and threads, as from any C program."""

import os
import tempfile

import cimo

k = cimo.compile(r"""
#include <pthread.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

long file_size(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fputs(text, f);
    fclose(f);
    struct stat st;
    return stat(path, &st) ? -1 : st.st_size;
}

double monotonic(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *square(void *p) { long *x = p; *x *= *x; return 0; }

long in_thread(long x) {           /* a POSIX thread created from C */
    pthread_t t;
    if (pthread_create(&t, 0, square, &x)) return -1;
    pthread_join(t, 0);
    return x;
}

long pid(void) { return getpid(); }
""", link=["libc"], host_headers=True)

path = os.path.join(tempfile.mkdtemp(), 'hello.txt')
print('wrote', k.file_size(path.encode(), b'hello from C\n'), 'bytes:', open(path).read().strip())
print('pid matches:', k.pid() == os.getpid())
print('12 squared in a pthread:', k.in_thread(12))
t = k.monotonic()
print('monotonic clock ticks:', k.monotonic() > t)
