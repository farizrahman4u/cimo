"""Shrink a C program on which cimo and gcc disagree (from fuzz_csmith.py), keeping it free of
undefined behavior as far as gcc's sanitizers can tell.

    python tests/reduce.py prog.c [-I DIR ...]      # writes prog.reduced.c

Deletes lines and brace-balanced blocks (ddmin over lines, then single lines) while the
program still: compiles with gcc; runs under -fsanitize=address,undefined without a report;
prints the same with gcc -O0 and -O2; and prints something else under cimo (or cimo fails to
compile or crashes in a way gcc does not). Simple and slow (minutes), but needs only gcc.
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 10


def gcc_run(src, flags, inc, d):
    exe = os.path.join(d, 'a.out')
    p = subprocess.run(['gcc', '-w', '-o', exe, src, '-lm'] + flags + ['-I' + i for i in inc],
                       capture_output=True)
    if p.returncode:
        return None
    try:
        r = subprocess.run([exe], capture_output=True, timeout=TIMEOUT,
                           env=dict(os.environ, UBSAN_OPTIONS='halt_on_error=1',
                                    ASAN_OPTIONS='detect_leaks=0'))
    except subprocess.TimeoutExpired:
        return None
    if r.returncode or r.stderr:
        return None
    return r.stdout


def cimo_run(src, inc):
    env = dict(os.environ, CIMO_INCLUDE=os.pathsep.join(inc))
    try:
        p = subprocess.run([sys.executable, os.path.join(HERE, 'corpora_worker.py')],
                           input=src + '\n', capture_output=True, text=True, env=env,
                           timeout=4 * TIMEOUT)
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
    return p.stdout.strip() or 'DIED %d' % p.returncode


def interesting(lines, inc, d, keep_compile_errors):
    src = os.path.join(d, 't.c')
    with open(src, 'w') as f:
        f.writelines(lines)
    out = gcc_run(src, ['-O0', '-fsanitize=address,undefined', '-fno-sanitize-recover=all'],
                  inc, d)
    if out is None or gcc_run(src, ['-O2'], inc, d) != out:
        return False
    with open(src + '.expected', 'wb') as f:
        f.write(out)
    res = cimo_run(src, inc)
    if res.startswith('COMPILE '):
        return keep_compile_errors
    return res != 'PASS'


def balanced(chunk):
    s = ''.join(chunk)
    return s.count('{') == s.count('}') and s.count('(') == s.count(')')


def reduce(lines, inc, d, keep_compile_errors):
    test = lambda ls: interesting(ls, inc, d, keep_compile_errors)
    n = 2
    while len(lines) >= 2:
        size = max(1, len(lines) // n)
        removed = False
        i = 0
        while i < len(lines):
            chunk = lines[i:i + size]
            cand = lines[:i] + lines[i + size:]
            if balanced(chunk) and test(cand):
                lines = cand
                removed = True
                sys.stderr.write('%d lines\n' % len(lines))
            else:
                i += size
        if not removed:
            if size == 1:
                break
            n = min(len(lines), n * 2)
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('file')
    ap.add_argument('-I', dest='inc', action='append', default=[])
    ap.add_argument('--compile-errors', action='store_true',
                    help='a cimo compile error counts as a failure to keep')
    a = ap.parse_args()
    with open(a.file) as f:
        lines = f.readlines()
    with tempfile.TemporaryDirectory() as d:
        if not interesting(lines, a.inc, d, a.compile_errors):
            sys.exit('the input is not interesting (gcc fails, sanitizers report, or cimo agrees)')
        lines = reduce(lines, a.inc, d, a.compile_errors)
    out = a.file[:-2] + '.reduced.c'
    with open(out, 'w') as f:
        f.writelines(lines)
    print('%s: %d lines' % (out, len(lines)))


if __name__ == '__main__':
    main()
