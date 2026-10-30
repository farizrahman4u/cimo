"""Tests for cimo.preprocessor (the preprocessor) and the bundled headers in cimo/include.

The differential parts need GNU cpp (token comparison), C cproc at CIMO_CPROC_DIR (default
~/code/cproc; IR comparison) and C QBE + gcc (running the header test program); they are skipped
when those are missing. `python3 python/tests/preprocessor_compare.py` prints the same comparisons as a
report.
"""

import os
import shutil
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import preprocessor_compare  # noqa: E402
from cimo import preprocessor  # noqa: E402
from cimo.preprocessor import PreprocessError, preprocess  # noqa: E402

QBE = os.environ.get('CIMO_QBE', os.path.expanduser('~/code/qbe/qbe'))
HAVE_CPP = shutil.which('cpp') is not None
HAVE_CPROC = preprocessor_compare.cproc_available()
HAVE_GCC = shutil.which('gcc') is not None
HAVE_QBE = os.path.exists(QBE)
IS_ARM64_LINUX = sys.platform.startswith('linux') and os.uname().machine == 'aarch64'


def toks(src, **kw):
    kw.setdefault('warn', lambda m: None)
    return preprocessor_compare.tokens(preprocess(src, **kw))


def err(src, **kw):
    kw.setdefault('warn', lambda m: None)
    with pytest.raises(PreprocessError) as e:
        preprocess(src, **kw)
    return str(e.value)


# --- API and semantics ---------------------------------------------------------------------

def test_basic_expansion():
    assert toks('#define N 4\nint a[N];\n') == ['int', 'a', '[', '4', ']', ';']
    assert toks('#define f(x, y) x ## y\nf(a, b) f(, 1)\n') == ['ab', '1']
    assert toks('#define s(x) #x\ns( a  "b\\n"  c )\n') == ['"a \\"b\\\\n\\" c"']


def test_line_markers_and_lines():
    out = preprocess('a\n\n\nb\n' + '\n' * 20 + 'c\n', 'f.c')
    lines = out.split('\n')
    assert lines[0] == '# 1 "f.c"'
    assert lines[1] == 'a' and lines[4] == 'b'
    assert '# 25 "f.c"' in lines and lines[lines.index('# 25 "f.c"') + 1] == 'c'


def test_line_markers_for_includes(tmp_path):
    (tmp_path / 'h.h').write_text('in_header\n')
    main = tmp_path / 'm.c'
    main.write_text('before\n#include "h.h"\nafter\n')
    out = preprocess(main.read_text(), str(main))
    assert '# 1 "%s" 1' % (tmp_path / 'h.h') in out
    assert '# 3 "%s" 2' % main in out


def test_include_search_order(tmp_path):
    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    (tmp_path / 'a' / 'x.h').write_text('from_a\n')
    (tmp_path / 'b' / 'x.h').write_text('from_b\n#include_next <x.h>\n')
    (tmp_path / 'b' / 'stddef.h').write_text('user_stddef\n')
    assert toks('#include <x.h>\n', include_dirs=[str(tmp_path / 'a'), str(tmp_path / 'b')]) \
        == ['from_a']
    assert toks('#include <x.h>\n', include_dirs=[str(tmp_path / 'b'), str(tmp_path / 'a')]) \
        == ['from_b', 'from_a']
    # User directories come before the bundled headers.
    assert toks('#include <stddef.h>\n', include_dirs=[str(tmp_path / 'b')]) == ['user_stddef']
    # Quotes search the including file's directory first.
    (tmp_path / 'a' / 'q.h').write_text('#include "x.h"\n')
    assert toks('#include "q.h"\n', include_dirs=[str(tmp_path / 'b'), str(tmp_path / 'a')]) \
        == ['from_a']
    assert toks('#define H <x.h>\n#include H\n', include_dirs=[str(tmp_path / 'a')]) == ['from_a']


def test_defines_argument():
    assert toks('A B SQ(3)\n', defines={'A': None, 'B': '2', 'SQ(x)': '((x)*(x))'}) == \
        ['1', '2', '(', '(', '3', ')', '*', '(', '3', ')', ')']
    assert toks('#ifdef __APPLE__\napple\n#elif defined __linux__\nlinux\n#endif\n') == ['linux']
    assert toks('#ifdef __APPLE__\napple __MACH__\n#elif defined __linux__\nlinux\n#endif\n',
                target='arm64_apple') == ['apple', '1']
    with pytest.raises(ValueError):
        preprocess('', target='x86')


def test_predefined():
    assert toks('__STDC__ __STDC_VERSION__ __STDC_HOSTED__ __aarch64__ __LP64__\n') == \
        ['1', '201112L', '0', '1', '1']
    assert toks('__LINE__ __FILE__ __COUNTER__ __COUNTER__\n__LINE__\n', filename='x.c') == \
        ['1', '"x.c"', '0', '1', '2']
    t = toks('__DATE__ __TIME__\n')
    assert len(t) == 2 and len(t[0]) == 13 and len(t[1]) == 10


def test_if_expressions():
    def ev(e, **kw):
        return toks('#if %s\nyes\n#else\nno\n#endif\n' % e, **kw) == ['yes']
    assert ev('-1 > 0u')
    assert ev('(0 && 1/0) || 1')
    assert ev('1 ? 2 : 1/0')
    assert not ev("'\\377' < 0")                    # plain char is unsigned on Linux aarch64
    assert ev("'\\377' < 0", target='arm64_apple')  # and signed on Apple arm64
    assert ev('0xffffffffffffffff == -1')
    assert ev('(-9223372036854775807 - 1) / -1 < 0')  # wraps, like gcc
    assert ev('defined(__linux__) && !defined __APPLE__')
    assert ev('__has_include(<stdint.h>) && !__has_include("nope.h")')
    assert ev('__has_builtin(__builtin_va_arg) && !__has_builtin(__builtin_popcount)')
    assert 'division by zero in #if' in err('#if 1/0\n#endif\n')
    assert 'floating constant' in err('#if 1.0\n#endif\n')
    assert 'missing binary operator' in err('#if 1 2\n#endif\n')


def test_errors_have_locations(tmp_path):
    assert err('\n#error stop here\n', filename='e.c') == 'e.c:2: error: #error stop here'
    assert err('#if 1\n', filename='u.c').startswith('u.c:1: error: unterminated #if')
    assert err('#endif\n', filename='u.c') == 'u.c:1: error: #endif without #if'
    assert 'does not give a valid preprocessing token' in err('#define c(a,b) a##b\nc(+,/)\n')
    assert 'unterminated argument list invoking macro "f"' in err('#define f(x) x\nf(1\n')
    assert 'requires 2 arguments, but only 1 given' in err('#define f(x,y) x\nf(1)\n')
    assert err('#include "missing.h"\n', filename='i.c') == \
        'i.c:1: error: missing.h: No such file or directory'
    assert 'unterminated comment' in err('/* x\n')
    (tmp_path / 'bad.h').write_text('ok\n#error in header\n')
    assert err('#include "bad.h"\n', filename=str(tmp_path / 'm.c')) == \
        '%s:2: error: #error in header' % (tmp_path / 'bad.h')


def test_warnings():
    seen = []
    preprocess('#warning careful\n#define A 1\n#define A 2\n', 'w.c', warn=seen.append)
    assert seen == ['w.c:1: warning: #warning careful', 'w.c:3: warning: "A" redefined']


def test_pragmas():
    out = preprocess('#pragma once\n#pragma GCC optimize("O3")\n#define N 8\n'
                     '#pragma omp parallel for num_threads(N)\nx; _Pragma("omp simd") y;\n'
                     '_Pragma("STDC FP_CONTRACT ON") z;\n')
    assert '#pragma omp parallel for num_threads(8)\n' in out
    assert '#pragma omp simd\n' in out
    assert 'GCC' not in out and 'STDC' not in out and 'once' not in out
    assert preprocessor_compare.tokens(out) == ['#', 'pragma', 'omp', 'parallel', 'for', 'num_threads',
                                       '(', '8', ')', 'x', ';', '#', 'pragma', 'omp', 'simd',
                                       'y', ';', 'z', ';']
    assert 'stop' in err('#pragma GCC error "stop"\n')


def test_push_pop_macro():
    assert toks('#define X 1\n#pragma push_macro("X")\n#undef X\n#define X 2\nX\n'
                '#pragma pop_macro("X")\nX\n') == ['2', '1']


def test_digraphs_are_printed_canonically():
    out = preprocess('<: :> <% %> %:define X 1\n')
    assert '[ ] { }' in out


def test_cli(tmp_path):
    f = tmp_path / 'c.c'
    f.write_text('#include "h.h"\nV W\n')
    (tmp_path / 'inc').mkdir()
    (tmp_path / 'inc' / 'h.h').write_text('#define V v\n')
    env = dict(os.environ, PYTHONPATH=os.path.dirname(HERE))
    p = subprocess.run([sys.executable, '-m', 'cimo.preprocessor', '-I', str(tmp_path / 'inc'),
                        '-DW=w', str(f)], capture_output=True, text=True, env=env)
    assert p.returncode == 0, p.stderr
    assert preprocessor_compare.tokens(p.stdout) == ['v', 'w']
    f.write_text('#error boom\n')
    p = subprocess.run([sys.executable, '-m', 'cimo.preprocessor', str(f)], capture_output=True,
                       text=True, env=env)
    assert p.returncode == 1 and p.stderr.strip() == '%s:1: error: #error boom' % f


def test_bundled_headers_preprocess():
    names = sorted(os.listdir(preprocessor.BUNDLED_INCLUDE))
    expected = ['assert.h', 'ctype.h', 'float.h', 'inttypes.h', 'iso646.h', 'limits.h', 'math.h', 'stdalign.h',
                'stdarg.h', 'stdbool.h', 'stddef.h', 'stdint.h', 'stdio.h', 'stdlib.h',
                'stdnoreturn.h', 'string.h']
    assert names == expected
    src = ''.join('#include <%s>\n' % n for n in names) * 2   # twice: include guards
    for target in preprocessor.TARGETS:
        preprocess(src, target=target, warn=lambda m: pytest.fail(m))


# --- Differential: GNU cpp, C cproc ----------------------------------------------------------

CORPUS = preprocessor_compare.corpus()
TOKEN_CORPUS = [f for f in CORPUS if not preprocessor_compare.missing_headers(f)
                and not preprocessor_compare.uses_system_headers(f)]


def _id(path):
    return os.path.relpath(path, os.path.dirname(os.path.dirname(preprocessor_compare.CPROC_DIR)))


@pytest.mark.skipif(not HAVE_CPP, reason='needs GNU cpp')
@pytest.mark.parametrize('path', TOKEN_CORPUS, ids=_id)
def test_tokens_match_gnu_cpp(path):
    assert preprocessor_compare.compare_tokens(path) is None


@pytest.mark.skipif(not HAVE_CPROC, reason='needs C cproc and GNU cpp')
@pytest.mark.parametrize('path', CORPUS, ids=_id)
def test_cproc_ir_matches(path):
    if preprocessor_compare.missing_headers(path):
        # Headers cimo doesn't bundle (stdio.h): check against cproc -E on glibc's headers.
        dirs = preprocessor_compare.system_include_dirs()
        assert preprocessor_compare.compare_tokens(path, dirs) is None
        assert preprocessor_compare.compare_ir(path, dirs, strip=False) is None
    else:
        assert preprocessor_compare.compare_ir(path) is None


@pytest.mark.skipif(not HAVE_CPP, reason='needs GNU cpp')
def test_random_macros_match_gnu_cpp(tmp_path):
    """Random mutually recursive macros (rescanning, painting, #, ##, __VA_ARGS__)."""
    failures, valid = preprocessor_compare.fuzz(1, 400, str(tmp_path))
    assert valid > 100
    assert failures == [], failures[0]


SYSTEM_PROGRAM = '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>
#include <inttypes.h>
#include <limits.h>
#include <float.h>
#include <stdarg.h>
#include <stddef.h>
#include <errno.h>
#include <time.h>
#include <signal.h>
#include <assert.h>
#include <wchar.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>
#include <fcntl.h>
int main(void) { printf("%d %s\\n", INT_MAX, strerror(errno)); assert(1);
  return isnan(NAN) + isinf(HUGE_VAL) + fpclassify(1.0); }
'''


@pytest.mark.skipif(not HAVE_CPP, reason='needs GNU cpp')
@pytest.mark.parametrize('which', ['libc', 'python'])
def test_system_headers_match_gnu_cpp(tmp_path, which):
    """Real-world headers (glibc, Python.h) with gcc's full predefined macro set."""
    if which == 'libc':
        src = SYSTEM_PROGRAM
        extra = []
    else:
        import sysconfig
        inc = sysconfig.get_paths().get('include')
        if not inc or not os.path.exists(os.path.join(inc, 'Python.h')):
            pytest.skip('no Python.h')
        src = '#include <Python.h>\n'
        extra = [inc]
    f = tmp_path / 'sys.c'
    f.write_text(src)
    dm = subprocess.run(['cpp', '-dM', '/dev/null'], capture_output=True, text=True).stdout
    ours = dict(preprocessor.TARGETS['arm64'])
    defs = []
    for line in dm.splitlines():
        parts = line.split(' ', 2)
        if parts[1] in ours or parts[1] in ('__STDC__', '__STDC_VERSION__', '__STDC_HOSTED__'):
            continue
        defs.append((parts[1], parts[2] if len(parts) > 2 else ''))
    dirs = extra + preprocessor_compare.system_include_dirs()
    out = preprocess(src, str(f), dirs, defs, warn=lambda m: None)
    cmd = ['cpp', '-nostdinc', '-U__STDC_VERSION__', '-D__STDC_VERSION__=201112L',
           '-U__STDC_HOSTED__', '-D__STDC_HOSTED__=0']
    cmd += ['-I' + d for d in dirs]
    cmd += ['-D%s=%s' % (n, '1' if v is None else v) for n, v in preprocessor.TARGETS['arm64']]
    gnu = subprocess.run(cmd + [str(f)], capture_output=True, text=True)
    assert gnu.returncode == 0, gnu.stderr
    assert preprocessor_compare.tokens(out) == preprocessor_compare.tokens(gnu.stdout)


# --- Bundled headers under cproc ------------------------------------------------------------

ALL_HEADERS = os.path.join(HERE, 'preprocessor_headers', 'all_headers.c')


@pytest.mark.skipif(not HAVE_CPROC, reason='needs C cproc')
def test_all_headers_compile_for_apple():
    out = preprocess(open(ALL_HEADERS).read(), ALL_HEADERS, target='arm64_apple')
    rc, ir, errs = preprocessor_compare.qbe_ir(out)
    assert rc == 0, errs


@pytest.mark.skipif(not (HAVE_CPROC and HAVE_QBE and HAVE_GCC and IS_ARM64_LINUX),
                    reason='needs C cproc, C QBE and gcc on Linux aarch64')
def test_all_headers_values_match_gcc(tmp_path):
    """Sizes, limits and library results: cimo.preprocessor + bundled headers + cproc + QBE vs gcc with
    the system headers."""
    pp = preprocess(open(ALL_HEADERS).read(), ALL_HEADERS)
    rc, ir, errs = preprocessor_compare.qbe_ir(pp)
    assert rc == 0, errs
    (tmp_path / 'a.qbe').write_text(ir)
    asm = subprocess.run([QBE, '-t', 'arm64', str(tmp_path / 'a.qbe')], capture_output=True,
                         text=True, check=True).stdout
    (tmp_path / 'a.s').write_text(asm)
    subprocess.run(['gcc', '-o', str(tmp_path / 'cimo'), str(tmp_path / 'a.s'), '-lm'],
                   check=True)
    subprocess.run(['gcc', '-std=gnu11', '-fno-builtin', '-w', '-o', str(tmp_path / 'gcc'),
                    ALL_HEADERS, '-lm'], check=True)
    ours = subprocess.run([str(tmp_path / 'cimo')], capture_output=True, text=True,
                          check=True).stdout.splitlines()
    gnu = subprocess.run([str(tmp_path / 'gcc')], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert len(ours) == len(gnu) > 200
    diffs = [(a, b) for a, b in zip(gnu, ours) if a != b]
    assert diffs == []
