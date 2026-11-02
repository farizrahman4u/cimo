"""macOS on Apple silicon (target arm64_apple): Apple's arm64 calling convention where it differs
from AAPCS64 (variadic arguments all on the stack, stack arguments packed at their own size, no
even register pair for 16-byte aligned composites), the macOS SDK's headers, and the MAP_JIT
loader. The code generation checks run on any host; the rest needs macOS on arm64. Calls in
both directions with clang-built code are in test_abi.py."""

import os
import platform
import re
import sys
import threading

import pytest

import cimo
import extcompile as X
from cimo import frontend
from cimo.preprocessor import preprocess

MAC_ARM64 = sys.platform == 'darwin' and platform.machine().lower() in ('arm64', 'aarch64')
needs_mac = pytest.mark.skipif(not MAC_ARM64, reason='needs macOS on arm64')


def ir(src, target='arm64_apple'):
    front = {'arm64_apple': 'aarch64-apple', 'arm64': 'aarch64'}[target]
    return frontend.compile(preprocess(src, 't.c', target=target), 't.c', target=front)


def asm(src, target='arm64_apple'):
    return X.backend(ir(src, target), target=target)


def test_types():
    out = ir(r'''
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <inttypes.h>
#define CHECK(c) _Static_assert(c, #c);
CHECK(sizeof(long) == 8 && sizeof(void *) == 8 && __LP64__)
CHECK(sizeof(wchar_t) == 4 && (wchar_t)-1 < 0 && (char)-1 < 0 && CHAR_MIN < 0)
CHECK(sizeof(long double) == 8)
CHECK(__APPLE__ && __MACH__ && __aarch64__ && __arm64__ && __LITTLE_ENDIAN__)
int f(void) { return sizeof(PRId64); }  /* "lld": int64_t is long long, as in Apple's headers */
''')
    assert 'ret 4\n' in out


def test_variadic_struct_arguments_go_on_the_stack():
    # Apple passes every variadic argument on the stack; structs were put in x1/x2 (the callee,
    # reading the stack, got garbage)
    src = 'struct s16 { long a, b; };\ndouble v(int, ...);\ndouble f(struct s16 *p) { return v(1, *p); }\n'
    s = asm(src)
    assert 'str\tx2, [x1]' in s and 'str\tx0, [x1, 8]' in s and 'add\tx1, sp, #0' in s
    assert 'ldr\tx1, [x0]' in asm(src, 'arm64')  # AAPCS64: in registers


def test_16_byte_aligned_struct_takes_the_next_registers():
    # AAPCS64 starts a 16-byte aligned composite at an even register (x2); Apple does not (x1)
    src = 'struct a16 { _Alignas(16) long x; };\nvoid g(void *, struct a16);\nvoid c(struct a16 *p) { g(0, *p); }\n'
    s = asm(src)
    assert 'ldr\tx1, [x0]' in s and 'ldr\tx2, [x0, 8]' in s
    assert 'ldr\tx2, [x0]' in asm(src, 'arm64')


SUBWORD = r'''
long g(int, int, int, int, int, int, int, int, unsigned char, short);
long c(void) { return g(1, 2, 3, 4, 5, 6, 7, 8, 200, -3); }
long f(int a, int b, int c, int d, int e, int f, int g, int h, unsigned char p, short q) {
    return a + h * 10 + p * 1000 + q * 1000000L;
}
'''


def test_char_and_short_stack_arguments_are_packed():
    # Apple packs stack arguments at their own size: a char at [sp], the short at [sp, 2]; they
    # took 8-byte slots as on Linux. The frontend says so with QBE's sub-word classes.
    out = ir(SUBWORD)
    assert re.search(r'ub 200, sh %\S+\)', out) and re.search(r'ub %\S+, sh %\S+\) \{', out)
    assert re.search(r'w 200, w %\S+\)', ir(SUBWORD, 'arm64'))  # cproc's IR elsewhere
    s = asm(SUBWORD)
    assert re.search(r'strb\tw\d+, \[sp\]', s) and re.search(r'strh\tw\d+, \[sp, 2\]', s)
    assert re.search(r'ldrb\tw\d+, \[x29, 16\]', s) and re.search(r'ldrsh\tw\d+, \[x29, 18\]', s)


def test_sdk_style_declarations():
    # from the macOS SDK's headers: an alignment attribute after a struct body, and a stray `;`
    # at file scope (a macro that expands to nothing); both as gcc and clang accept them
    out = ir(r'''
struct z { char z[16][256]; } __attribute__((aligned(4)));
struct w { char c; } __attribute__((__aligned__(16)));
struct ip { unsigned char v; unsigned short len; unsigned int src; } __attribute__((__packed__));
union u { char c; int i; } __attribute__ ((packed));
struct pa { char c; short h; } __attribute__((packed, aligned(4)));
struct __attribute__((packed)) pb { char c; short h; } __attribute__((aligned(4)));
;
_Static_assert(_Alignof(struct z) == 4 && _Alignof(struct w) == 16 && sizeof(struct w) == 16, "");
_Static_assert(sizeof(struct ip) == 7 && _Alignof(struct ip) == 1 && sizeof(union u) == 4, "");
_Static_assert(sizeof(struct pa) == 4 && _Alignof(struct pa) == 4 && sizeof(struct pb) == 4, "");
_Static_assert(__builtin_offsetof(struct ip, src) == 3 && __builtin_offsetof(struct pa, h) == 1, "");
int f(void) { return 1; };
''')
    assert 'function w $f()' in out


def test_packed_after_the_body_keeps_a_members_specified_alignment():
    # gcc and clang: packed removes padding, but not what _Alignas or aligned(N) on a member ask for
    out = ir(r'''
struct a { _Alignas(8) char a; char b; } __attribute__((packed));
struct b { long a; _Alignas(8) char b; char c; } __attribute__((packed));
struct c { char a __attribute__((aligned(8))); char b; } __attribute__((packed));
struct d { long a; char b __attribute__((aligned(4))); } __attribute__((packed));
struct e { char a; _Alignas(8) char b; } __attribute__((packed));
union f { _Alignas(8) char a; char b; } __attribute__((packed));
_Static_assert(sizeof(struct a) == 8 && _Alignof(struct a) == 8, "");
_Static_assert(sizeof(struct b) == 16 && _Alignof(struct b) == 8, "");
_Static_assert(sizeof(struct c) == 8 && _Alignof(struct c) == 8, "");
_Static_assert(sizeof(struct d) == 12 && _Alignof(struct d) == 4, "");
_Static_assert(sizeof(struct e) == 16 && __builtin_offsetof(struct e, b) == 8, "");
_Static_assert(sizeof(union f) == 8 && _Alignof(union f) == 8, "");
int f(void) { return 1; }
''')
    assert 'function w $f()' in out


def test_cli_names_the_sub_word_classes():
    from cimo.cli.commands import _pytype
    assert [_pytype(k) for k in ('sb', 'ub', 'sh', 'uh')] == ['int8', 'uint8', 'int16', 'uint16']


@needs_mac
def test_python_calls_with_packed_stack_arguments():
    # ctypes (libffi) packs the char and the short like Apple's C compilers
    m = cimo.compile(SUBWORD + 'long g(int a, int b, int c, int d, int e, int f_, int g, int h, '
                     'unsigned char p, short q) { return f(a, b, c, d, e, f_, g, h, p, q); }\n')
    assert m.f(1, 2, 3, 4, 5, 6, 7, 8, 200, -3) == 1 + 80 + 200000 - 3000000
    assert m.c() == m.f(1, 2, 3, 4, 5, 6, 7, 8, 200, -3)
    assert m.f(0, 0, 0, 0, 0, 0, 0, 0, 300, 70000) == 44 * 1000 + 4464 * 1000000  # wrap


@needs_mac
def test_variadic_structs():
    from test_regressions import VA_STRUCTS
    m = cimo.compile(VA_STRUCTS)
    assert [m.run(i) for i in range(6)] == [97243.0, 4752.0, 333.0, 643.0, 231625.0, 86.0]


@needs_mac
def test_jit_image_is_cached(tmp_path, monkeypatch):
    # the data region sits right after the MAP_JIT code (mapped over the tail of a JIT mapping
    # of both sizes), so the linked image can be snapshotted and cached like on Linux
    from cimo import driver
    monkeypatch.setenv('CIMO_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(driver, '_memory_cache', {})
    src = 'long tab[4] = {1, 2, 3, 4};\nlong f(long i) { tab[i & 3] += i; return tab[i & 3]; }\n'
    assert cimo.compile(src).f(5) == 7
    assert [os.path.splitext(n)[1] for n in os.listdir(str(tmp_path))] == ['.img']
    monkeypatch.setattr(driver, '_memory_cache', {})
    m = cimo.compile(src)  # from the image
    assert m.f(1) == 3
    # MAP_JIT pages are RX for a thread while kernels run; data must stay writable from every
    # thread at once
    out = []
    ts = [threading.Thread(target=lambda: out.append(sum(m.f(4) for _ in range(1000))))
          for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(out) == 4


@needs_mac
def test_host_headers_and_libc(tmp_path):
    # the SDK's headers (they need __arm64__, and _OSSwapInt32 for htonl without __GNUC__) and
    # the OS through libSystem: errno, environ, files, threads, sockets
    m = cimo.compile(r'''
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <crt_externs.h>
extern char **environ;
long f(void) { errno = 0; errno = 7; return errno * 10 + (environ == *_NSGetEnviron()); }
static void *work(void *a) { *(long *)a *= 3; return a; }
long th(long x) { pthread_t t; pthread_create(&t, 0, work, &x); pthread_join(t, 0); return x; }
unsigned swap(unsigned x) { return htonl(x) + ntohs(0x1234); }
int wr(const char *path) {
    FILE *fp = fopen(path, "w"); if (!fp) return -1;
    fprintf(fp, "hello %d\n", 42); fclose(fp);
    struct stat st; return stat(path, &st) ? -errno : (int)st.st_size;
}
int sock(void) {
    int s = socket(AF_INET, SOCK_STREAM, 0); if (s < 0) return -1;
    struct sockaddr_in a; memset(&a, 0, sizeof a);
    a.sin_family = AF_INET; a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    socklen_t n = sizeof a;
    int ok = !bind(s, (struct sockaddr *)&a, n) && !getsockname(s, (struct sockaddr *)&a, &n);
    close(s);
    return ok && ntohs(a.sin_port) > 0;
}
''', link=['libc'], host_headers=True)
    assert m.f() == 71 and m.th(14) == 42
    assert m.swap(0x11223344) == 0x44332211 + 0x3412
    path = str(tmp_path / 'w.txt')
    assert m.wr(path.encode()) == 9 and open(path).read() == 'hello 42\n'
    assert m.sock() == 1
