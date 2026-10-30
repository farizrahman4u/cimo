"""Benchmark cimo against gcc on the 15 kernels of scratch/qbe_eval/kernels.c.

    python3 python/bench/run.py [kernels...] [-n TRIALS] [--cimo NAME=FLAGS ...] [--no-default]

Configurations (all called in-process through ctypes and the same C timing loops, bench.c):
  cimo        cimo.compile(kernels.c), i.e. whatever cimo.driver currently does (fresh cache)
  gcc-nofma   gcc -O3 -mcpu=native -fno-tree-vectorize -ffp-contract=off  (like-for-like scalar)
  gcc-novec   gcc -O3 -mcpu=native -fno-tree-vectorize                    (default FMA contraction)
  gcc-O3      gcc -O3 -mcpu=native                                         (vectorized)

`--cimo NAME=FLAGS` adds a cimo build with explicit extension flags instead of the driver's
choice: FLAGS is a comma list of fma, ldp, layout (or `none`), e.g.
`--cimo base=none --cimo all=fma,ldp,layout`. `--no-default` drops the plain `cimo` config.
`--gen-opt` gives the --cimo builds the hand-optimized IR of scratch/qbe_eval/gen_opt.py for
saxpy, gemm, jacobi2d and conv3x3 (a stand-in for the middle stage).

Each kernel runs TRIALS times per configuration, configurations interleaved (rotating order),
best trial reported. Single runs drift ~10% on WSL; compare ratios, not a single table.
Checksums: `=` matches gcc-nofma exactly, `=f` matches gcc-novec (FMA) exactly, otherwise
the relative difference from gcc-nofma.
"""

import argparse
import ctypes
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYROOT = os.path.dirname(HERE)
REPO = os.path.dirname(PYROOT)
KERNELS_C = os.path.join(REPO, 'scratch', 'qbe_eval', 'kernels.c')
BUILD = os.path.join(HERE, 'build')
sys.path.insert(0, PYROOT)

KERNELS = ['saxpy', 'gemm', 'jacobi2d', 'conv3x3', 'rgb2gray', 'dot', 'isum', 'prefix_sum',
           'iir', 'edit_distance', 'histogram', 'mandelbrot', 'nbody_acc', 'flood_fill',
           'sieve']

GCC = os.environ.get('CC', 'gcc')
GCC_CONFIGS = [
    ('gcc-nofma', ['-O3', '-mcpu=native', '-fno-tree-vectorize', '-ffp-contract=off']),
    ('gcc-novec', ['-O3', '-mcpu=native', '-fno-tree-vectorize']),
    ('gcc-O3', ['-O3', '-mcpu=native']),
]
FLAGS = ('fma', 'ldp', 'layout')


def _cc(args):
    subprocess.run([GCC] + args, check=True)


def _stale(out, *deps):
    return not os.path.exists(out) or any(os.path.getmtime(d) > os.path.getmtime(out)
                                          for d in deps)


def build_loops():
    so = os.path.join(BUILD, 'bench.so')
    src = os.path.join(HERE, 'bench.c')
    if _stale(so, src):
        _cc(['-O2', '-shared', '-fPIC', src, '-o', so])
    return ctypes.CDLL(so)


def build_gcc(name, flags):
    so = os.path.join(BUILD, 'kernels-%s.so' % name)
    if _stale(so, KERNELS_C, __file__):
        _cc(flags + ['-shared', '-fPIC', KERNELS_C, '-o', so, '-lm'])
    lib = ctypes.CDLL(so)
    return {k: ctypes.cast(getattr(lib, k), ctypes.c_void_p).value for k in KERNELS}, lib


class _FreshCache:
    """Point cimo's compile cache at an empty directory, so the current compiler runs."""

    def __enter__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get('CIMO_CACHE_DIR')
        os.environ['CIMO_CACHE_DIR'] = self.tmp.name

    def __exit__(self, *a):
        if self.old is None:
            del os.environ['CIMO_CACHE_DIR']
        else:
            os.environ['CIMO_CACHE_DIR'] = self.old
        self.tmp.cleanup()


def build_cimo_default(src):
    import cimo
    with _FreshCache():
        t0 = time.perf_counter()
        mod = cimo.compile(src)
        dt = time.perf_counter() - t0
    return mod, dt


def gen_opt_ir(ssa):
    """Replace saxpy, gemm, jacobi2d, conv3x3 in cproc's IR with the hand-optimized IR of
    scratch/qbe_eval/gen_opt.py (what the middle stage is meant to produce)."""
    import re
    sys.path.insert(0, os.path.join(REPO, 'scratch', 'qbe_eval'))
    import gen_opt
    for name in gen_opt.KERNELS:
        ssa = re.sub(r'export\nfunction (\w+ )?\$' + name + r'\(.*?\n}\n', '', ssa, flags=re.S)
    return ssa + '\n' + '\n\n'.join('\n'.join(f()) for f in gen_opt.KERNELS.values()) + '\n'


def compile_flags(src, flags, filename='kernels.c', gen_opt=False):
    """C source -> (assembly text, signatures) with explicit extension flags, no cache.
    Same pipeline as cimo.driver._build."""
    from cimo import driver
    from cimo.preprocessor import preprocess
    from cimo import frontend
    from cimo.backend import ext
    apple = sys.platform == 'darwin'
    pp = preprocess(driver._PRELUDE + '#line 1\n' + src, filename,
                    target='arm64_apple' if apple else 'arm64')
    kw = {'fma': True} if 'fma' in flags else {}
    ssa, params = driver._frontend(pp, filename, **kw)   # params: restrict info, middle end
    with ext.options(**{f: f in flags for f in FLAGS}):
        text = driver._qbe(gen_opt_ir(ssa) if gen_opt else ssa, params)
    sigs = '\n'.join('%s %s %s' % (m.group(2), m.group(1) or '-',
                                   ','.join(a.split()[0] for a in m.group(3).split(',')
                                            if a.strip()))
                     for m in driver._SIG.finditer(ssa))
    return text, sigs


def build_cimo_flags(src, flags, gen_opt=False):
    from cimo import driver
    t0 = time.perf_counter()
    text, sigs = compile_flags(src, flags, gen_opt=gen_opt)
    dt = time.perf_counter() - t0
    return driver.Module(text, sigs), dt


def parse_variant(s):
    name, _, fl = s.partition('=')
    flags = set() if fl in ('', 'none') else set(fl.split(','))
    bad = flags - set(FLAGS)
    if bad:
        raise SystemExit('unknown flag(s) %s; known: %s' % (', '.join(sorted(bad)),
                                                            ', '.join(FLAGS)))
    return name, flags


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('kernels', nargs='*', help='subset of kernels (default: all 15)')
    ap.add_argument('-n', '--trials', type=int, default=5)
    ap.add_argument('--cimo', action='append', default=[], metavar='NAME=FLAGS',
                    help='extra cimo build with explicit flags (fma,ldp,layout or none)')
    ap.add_argument('--no-default', action='store_true', help='skip plain cimo.compile')
    ap.add_argument('--gen-opt', action='store_true',
                    help='--cimo builds use the hand-optimized IR of scratch/qbe_eval/gen_opt.py '
                    'for saxpy, gemm, jacobi2d, conv3x3 (gen_opt never emits fma)')
    ap.add_argument('--gcc', default='nofma,novec,O3',
                    help='gcc configs to run (comma list of nofma, novec, O3)')
    args = ap.parse_args(argv)

    if platform.machine().lower() not in ('arm64', 'aarch64'):
        raise SystemExit('arm64 only')
    kernels = args.kernels or KERNELS
    for k in kernels:
        if k not in KERNELS:
            raise SystemExit('unknown kernel %s' % k)
    os.makedirs(BUILD, exist_ok=True)
    loops = build_loops()
    with open(KERNELS_C) as f:
        src = f.read()

    configs = []  # (name, {kernel: address}, keepalive)
    cimo_names = []
    if not args.no_default:
        mod, dt = build_cimo_default(src)
        print('cimo: compiled in %.2f s' % dt, file=sys.stderr)
        configs.append(('cimo', {k: getattr(mod, k).address for k in KERNELS}, mod))
        cimo_names.append('cimo')
    for v in args.cimo:
        name, flags = parse_variant(v)
        mod, dt = build_cimo_flags(src, flags, args.gen_opt)
        print('%s (%s): compiled in %.2f s' % (name, ','.join(sorted(flags)) or 'none', dt),
              file=sys.stderr)
        configs.append((name, {k: getattr(mod, k).address for k in KERNELS}, mod))
        cimo_names.append(name)
    want = set('gcc-' + g for g in args.gcc.split(',') if g)
    gcc_names = []
    for name, flags in GCC_CONFIGS:
        if name in want:
            addrs, lib = build_gcc(name, flags)
            configs.append((name, addrs, lib))
            gcc_names.append(name)

    names = [c[0] for c in configs]
    best = {n: {} for n in names}
    sums = {n: {} for n in names}
    for k in kernels:
        getattr(loops, 'init_' + k)()
        run = getattr(loops, 'run_' + k)
        run.restype = ctypes.c_double
        run.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
        cs = ctypes.c_double()
        for t in range(args.trials):
            r = t % len(configs)
            for name, addrs, _ in configs[r:] + configs[:r]:
                ms = run(addrs[k], ctypes.byref(cs))
                if ms < best[name].get(k, 1e300):
                    best[name][k] = ms
                prev = sums[name].setdefault(k, cs.value)
                if prev != cs.value and not (math.isnan(prev) and math.isnan(cs.value)):
                    sums[name][k] = float('nan')  # result changed between trials
        print('.', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    report(kernels, names, cimo_names, gcc_names, best, sums)


def _check(k, n, sums):
    v = sums[n][k]
    ref = sums.get('gcc-nofma', {}).get(k)
    fref = sums.get('gcc-novec', {}).get(k)
    if ref is not None and v == ref:
        return '='
    if fref is not None and v == fref:
        return '=f'
    if ref is None or math.isnan(v):
        return '?'
    return '%.0e' % (abs(v - ref) / max(abs(ref), 1e-300))


def report(kernels, names, cimo_names, gcc_names, best, sums):
    """ms per config; ratios cimo/gcc (lower is better). With one cimo config, against every
    gcc config; with several, against the first gcc config only, plus a geomean summary."""
    short = {n: n.replace('gcc-', '') for n in names}
    ref = gcc_names[:1] if len(cimo_names) > 1 else gcc_names
    ratios = [(c, g) for c in cimo_names for g in ref]
    rshort = {(c, g): ('%s/%s' % (c, short[g]) if len(cimo_names) > 1 else '/' + short[g])
              for c, g in ratios}
    w = max([9] + [len(short[n]) + 2 for n in names])
    rw = max([8] + [len(rshort[r]) + 2 for r in ratios])
    print('%-14s' % 'kernel' + ''.join('%*s' % (w, short[n]) for n in names)
          + ''.join('%*s' % (rw, rshort[r]) for r in ratios) + '  check')
    for k in kernels:
        row = '%-14s' % k + ''.join('%*.1f' % (w, best[n][k]) for n in names)
        row += ''.join('%*.2fx' % (rw - 1, best[c][k] / best[g][k]) for c, g in ratios)
        row += '  ' + ' '.join('%s:%s' % (short[n], _check(k, n, sums)) for n in cimo_names)
        print(row)
    print('(ms, best of trials; ratio = cimo time / gcc time)')
    if len(kernels) > 1 and gcc_names:
        print('\ngeomean ratio' + ''.join('%*s' % (w + 1, '/' + short[g]) for g in gcc_names))
        for c in cimo_names:
            print('%-13s' % c + ''.join(
                '%*.2fx' % (w, math.exp(sum(math.log(best[c][k] / best[g][k])
                                            for k in kernels) / len(kernels)))
                for g in gcc_names))


if __name__ == '__main__':
    main()
