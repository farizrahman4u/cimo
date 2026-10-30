"""M3 "clear errors": unsupported C gives a CompileError with a location and a message naming the
feature, never a silent miscompile, crash, or backend/assembler error (cimo.frontend.diag,
switched on by cimo.compile). Also the GNU spellings and C99 inline definitions it accepts."""

import os
import platform
import re

import pytest

import cimo
from cimo.frontend import compile as frontend_compile

pytestmark = pytest.mark.skipif(platform.machine().lower() not in ('arm64', 'aarch64'),
                                reason='cimo.compile generates arm64 code')

LOC = re.compile(r'^<string>:\d+:\d+: error: ')

REJECTED = [
    # attributes that would silently change the meaning of the program
    ('typedef int v4 __attribute__((vector_size(16)));', 'vector types (vector_size) are not supported yet'),
    ('typedef int v [[gnu::vector_size(16)]];', 'vector types (vector_size) are not supported yet'),
    ('typedef int q __attribute__((mode(QI)));', "attribute 'mode' is not supported"),
    ('void g(int *); int f(void){ int x __attribute__((cleanup(g))) = 1; return x; }',
     "attribute 'cleanup' is not supported"),
    ('typedef union { int a; } __attribute__((transparent_union)) u;',
     "attribute 'transparent_union' is not supported"),
    ('int f(void) __attribute__((alias("g")));', "attribute 'alias' is not supported"),
    ('[[clang::always_inline]] int f(void);', "attribute 'clang::always_inline' is not supported"),
    # types
    ('long double x;', 'long double is not supported'),
    ('long double f(long double a){ return a; }', 'long double is not supported'),
    ('int f(double a){ return (long double)a > 1; }', 'long double is not supported'),
    ('struct s { long double x; }; struct s v;', 'long double is not supported'),
    ('long double g(void); double f(void){ return g(); }', 'long double is not supported'),
    ('__int128 x;', "type '__int128' is not supported"),
    ('unsigned __int128 x;', "type 'unsigned __int128' is not supported"),
    ('__uint128_t f(void){ return 0; }', "type 'unsigned __int128' is not supported"),
    ('int f(void){ __int128_t x = 0; return 0; }', "type '__int128' is not supported"),
    ('_Float16 x;', "type '_Float16' is not supported"),
    ('__fp16 x;', "type '__fp16' is not supported"),
    ('__bf16 x;', "type '__bf16' is not supported"),
    ('foo_t x;', "unknown type name 'foo_t'"),
    ('int f(foo_t a);', "unknown type name 'foo_t'"),
    ('int f(void){ foo_t x; return 0; }', "unknown type name 'foo_t'"),
    ('struct s { foo_t *p; };', "unknown type name 'foo_t'"),
    # thread-local storage
    ('_Thread_local int t;', 'thread-local storage is not supported'),
    ('int f(void){ static _Thread_local int t; return t; }', 'thread-local storage is not supported'),
    ('extern __thread int t; int f(void){ return t; }', 'thread-local storage is not supported'),
    # inline assembly
    ('int f(void){ asm("nop"); return 0; }', 'inline assembly is not supported'),
    ('int f(void){ __asm__ volatile("nop"); return 0; }', 'inline assembly is not supported'),
    ('asm(".globl x");', 'inline assembly is not supported'),
    # GNU extensions
    ('int f(void){ return ({ 1; }); }', 'statement expressions'),
    ('int f(void){ void *p = &&l; l: return 0; }', 'labels as values'),
    ('int f(void *p){ goto *p; }', 'computed goto'),
    ('int f(void){ int g(void){ return 1; } return g(); }', 'nested functions'),
    ('int f(int x){ switch (x) { case 1 ... 3: return 1; } return 0; }', 'case ranges'),
    ('int a[4] = { [0 ... 3] = 1 };', 'range designators'),
    ('int f(void){ __label__ l; l: return 0; }', "local labels ('__label__') are not supported"),
    ('int f(void){ __auto_type x = 1; return x; }', "'__auto_type' is not supported"),
    ('int f(void){ return __builtin_choose_expr(1, 2, 3); }', "'__builtin_choose_expr' is not supported"),
    ('int f(int x){ return __builtin_ctzg(x); }', "built-in function '__builtin_ctzg' is not supported"),
    ('int f(int *p){ return __atomic_load_n(p, 0); }', "built-in function '__atomic_load_n' is not supported"),
    ('double f(void){ return 1.0i; }', 'imaginary constants'),
    ('int f(void){ return 2i; }', 'imaginary constants'),
    ('struct s {};', 'empty structs and unions'),
    # hosted libc without the host's headers: a hint
    ('int f(void){ return fopen("x", "r") != 0; }',
     "undeclared identifier: fopen; fopen is not in cimo's runtime: pass system_headers=True"),
    ('int f(void){ return (int)time(0); }', "time is not in cimo's runtime"),
    ('int f(void){ return pthread_self() != 0; }', "pthread_self is not in cimo's runtime"),
    ('int f(void){ FILE *fp = 0; return fp != 0; }', "FILE is not in cimo's runtime"),
]


@pytest.mark.parametrize('src,msg', REJECTED, ids=[s for s, m in REJECTED])
def test_rejected(src, msg):
    with pytest.raises(cimo.CompileError) as ei:
        cimo.compile(src)
    text = str(ei.value)
    assert LOC.match(text), text
    assert msg in text, text


HEADERS = [
    ('#include <setjmp.h>\n', 'setjmp.h: No such file or directory; setjmp/longjmp are not supported'),
    ('#include <time.h>\n', 'time.h is not bundled with cimo: pass system_headers=True'),
    ('#include <complex.h>\n', 'complex.h is not bundled with cimo'),
]


@pytest.mark.parametrize('src,msg', HEADERS, ids=[m.split(':')[0] for s, m in HEADERS])
def test_missing_header(src, msg):
    with pytest.raises(cimo.CompileError) as ei:
        cimo.compile(src)
    text = str(ei.value)
    assert re.match(r'^<string>:1: error: ', text), text
    assert msg in text, text


ACCEPTED = [
    # GNU spellings of keywords, and __func__'s GNU names
    ('int f(int *__restrict p, int *__restrict__ q){ return *p + *q; }', None),
    ('__const int c = 3; int f(void){ return c + __alignof(int) + __alignof__(int); }', 11),
    ('int f(void){ return __PRETTY_FUNCTION__[0] + __FUNCTION__[0] - 2 * __func__[0]; }', 0),
    # C99 inline definitions without an external definition: local functions
    ('inline int g(void){ return 1; } int f(void){ return g(); }', 1),
    ('__inline int g(void){ return 2; } int f(void){ int (*p)(void) = g; return p(); }', 2),
    ('inline int fact(int n){ return n < 2 ? 1 : n * fact(n - 1); } int f(void){ return fact(5); }', 120),
    # C23 nullptr (a null pointer constant)
    ('int f(void){ int *p = nullptr; return p == nullptr && !nullptr; }', 1),
    # attributes that don't change the meaning, or that cimo implements
    ('__attribute__((noinline, unused)) static int g(int x){ return x; } '
     '[[nodiscard, gnu::pure]] int f(void){ return g(4); }', 4),
    ('__attribute__((const, __const__)) int f(void){ return 5; }', 5),
    ('static float buf[16] __attribute__((aligned(64))); long f(void){ return (long)buf % 64; }', 0),
    ('struct s { char c; char m[3] __attribute__((aligned(8))); }; '
     'int f(void){ return sizeof(struct s) * 100 + _Alignof(struct s); }', 1608),
    ('typedef struct { long a[3]; } T __attribute__((aligned(16))); '
     'int f(void){ return _Alignof(T) * 100 + sizeof(T); }', 1624),
    # declaring what cimo cannot generate code for is fine
    ('long double g(long double); extern _Thread_local int t; '
     'struct s { __uint128_t v[2]; }; int f(void){ return sizeof(struct s) + sizeof(long double); }', 48),
    # the GNU conditional with omitted operand evaluates its condition once
    ('int f(void){ int x = 1; return (++x ?: 0) * 10 + x; }', 22),
]


@pytest.mark.parametrize('src,result', ACCEPTED, ids=[s for s, r in ACCEPTED])
def test_accepted(src, result):
    m = cimo.compile(src)
    if result is not None:
        assert m.f() == result


@pytest.mark.skipif(not os.path.exists('/usr/include/stdio.h') or platform.system() != 'Linux',
                    reason='needs glibc headers')
def test_system_headers():
    """glibc's headers compile (they reach the frontend with their attributes: cimo keeps
    sys/cdefs.h from defining __attribute__ away), and attributes after them are still checked."""
    hs = ('stdio.h stdlib.h string.h unistd.h time.h errno.h fcntl.h sys/stat.h sys/types.h '
          'pthread.h signal.h math.h dirent.h sys/mman.h stdint.h inttypes.h ctype.h wchar.h '
          'locale.h fenv.h sys/time.h sys/wait.h sys/socket.h netinet/in.h poll.h sched.h '
          'semaphore.h ucontext.h dlfcn.h regex.h wctype.h strings.h').split()
    src = ''.join('#include <%s>\n' % h for h in hs) + '''
int f(void) {
    char buf[32];
    snprintf(buf, sizeof buf, "%d", 42);
    FILE *fp = fopen("/dev/null", "w");
    fputs(buf, fp);
    fclose(fp);
    errno = 0;
    return (int)strlen(buf) + (int)sqrt(16.0) + abs(-1) + (errno != 0);
}
'''
    assert cimo.compile(src, system_headers=True, link=['libc']).f() == 7
    with pytest.raises(cimo.CompileError, match=r'<string>:2:\d+: error: vector types'):
        cimo.compile('#include <stdio.h>\ntypedef int v4 __attribute__((vector_size(16)));\n',
                     system_headers=True)


def test_cli_behavior_unchanged():
    """Without diagnostics=True (the cproc-qbe command line) the frontend is cproc's."""
    src = 'typedef int v4 __attribute__((vector_size(16))); long double x;'
    assert 'data $x' in frontend_compile(src)
    with pytest.raises(Exception, match='vector types'):
        frontend_compile(src, diagnostics=True)
