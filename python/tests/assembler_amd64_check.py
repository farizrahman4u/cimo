"""Compare cimo.assembler_amd64 against GNU as (x86-64 ELF): assemble the same text with both and
diff the sections, relocations and symbols. Used by test_assembler_amd64.py; also a script:

    python tests/assembler_amd64_check.py [files.s/.ssa ...]   # default: QBE's tests
    python tests/assembler_amd64_check.py --record             # refresh data/amd64_expect.json

GNU as: $AS_AMD64 if set, else `x86_64-linux-gnu-as`/`as` if it writes x86-64 ELF, else `as` in a
WSL distribution (Windows). Relocation targets are normalized to (section, offset) for local
symbols, since GNU as rewrites references to local symbols as section symbol + addend.
data/amd64_expect.json holds GNU as's output for the deterministic inputs (the forms file and the
generated sweeps) so the comparison also runs where GNU as is missing.
"""

import functools
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import assembler_check as A
from cimo import assembler_amd64

HERE = A.HERE
PYROOT = A.PYROOT
EXPECT = os.path.join(HERE, 'data', 'amd64_expect.json')
FORMS = os.path.join(HERE, 'data', 'amd64_forms.s')

# ELF x86-64 relocation types -> cimo names. GNU as writes REX_GOTPCRELX (GOTPCRELX without REX)
# for sym@gotpcrel(%rip) so that linkers may relax the load; cimo calls all three GOTPCREL.
RELOC_NAMES = {1: 'ABS64', 2: 'PC32', 4: 'PLT32', 9: 'GOTPCREL', 10: 'ABS32', 11: 'ABS32S',
               12: 'ABS16', 14: 'ABS8', 41: 'GOTPCREL', 42: 'GOTPCREL'}


def _is_x86_elf(path):
    with open(path, 'rb') as f:
        h = f.read(20)
    return h[:4] == b'\x7fELF' and h[18:20] == b'\x3e\x00'


def _wsl_distros():
    try:
        p = subprocess.run(['wsl', '-l', '-q'], capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return []
    out = p.stdout.decode('utf-16-le', errors='ignore').replace('\0', '')
    return [d.strip() for d in out.splitlines() if d.strip() and 'docker' not in d.lower()]


def _wslpath(p):
    p = os.path.abspath(p).replace('\\', '/')
    return '/mnt/%s%s' % (p[0].lower(), p[2:])


def _try(cmd, wsl):
    """Does `cmd` assemble x86-64 into ELF?"""
    try:
        with tempfile.TemporaryDirectory() as d:
            s, o = os.path.join(d, 'a.s'), os.path.join(d, 'a.o')
            with open(s, 'w') as f:
                f.write('ret\n')
            conv = _wslpath if wsl else (lambda x: x)
            p = subprocess.run(cmd + ['-o', conv(o), conv(s)], capture_output=True, timeout=60)
            return p.returncode == 0 and _is_x86_elf(o)
    except (OSError, subprocess.TimeoutExpired):
        return False


@functools.lru_cache(maxsize=None)
def gas_command():
    """(argv prefix, runs in WSL) for an x86-64 ELF GNU as, or None."""
    env = os.environ.get('AS_AMD64')
    if env:
        cmd, wsl = env.split(), env.startswith('wsl')
        return (tuple(cmd), wsl) if _try(cmd, wsl) else None
    for name in ('x86_64-linux-gnu-as', 'as'):
        if shutil.which(name) and _try([name], False):
            return ((name,), False)
    if os.name == 'nt' and shutil.which('wsl'):
        for d in _wsl_distros():
            cmd = ['wsl', '-d', d, '--', 'as']
            if _try(cmd, True):
                return (tuple(cmd), True)
    return None


def have_gas():
    return gas_command() is not None


def gas(text):
    cmd, wsl = gas_command()
    conv = _wslpath if wsl else (lambda x: x)
    with tempfile.TemporaryDirectory() as d:
        s = os.path.join(d, 'a.s')
        o = os.path.join(d, 'a.o')
        with open(s, 'w', newline='\n') as f:
            f.write(text)
        p = subprocess.run(list(cmd) + ['-o', conv(o), conv(s)], capture_output=True)
        if p.returncode != 0:
            raise RuntimeError('as failed: ' + p.stderr.decode(errors='replace'))
        with open(o, 'rb') as f:
            return f.read()


def gas_normalized(text):
    secs, relocs, globs = A.parse_elf(gas(text), RELOC_NAMES)
    globs.pop('_GLOBAL_OFFSET_TABLE_', None)  # GNU as adds it for @gotpcrel
    return secs, relocs, globs


def cimo_normalized(text):
    return A.normalize(assembler_amd64.assemble(text))


def compare(text):
    """Assemble `text` with GNU as and cimo.assembler_amd64; return a list of differences."""
    return A.diff(gas_normalized(text), cimo_normalized(text))


# recorded expectations (JSON-friendly form of the normalized objects)

def _to_json(norm):
    secs, relocs, globs = norm
    return {
        'sections': {n: [d.hex() if isinstance(d, (bytes, bytearray)) else list(d), al]
                     for n, (d, al) in secs.items()},
        'relocs': {n: [[o, t, list(tg)] for o, t, tg in rs] for n, rs in relocs.items()},
        'globals': {n: list(v) for n, v in globs.items()},
    }


def _from_json(j):
    secs = {n: (bytes.fromhex(d) if isinstance(d, str) else tuple(d), al)
            for n, (d, al) in j['sections'].items()}
    relocs = {n: [(o, t, tuple(tg)) for o, t, tg in rs] for n, rs in j['relocs'].items()}
    globs = {n: tuple(v) for n, v in j['globals'].items()}
    return secs, relocs, globs


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_expect():
    if not os.path.exists(EXPECT):
        return {}
    with open(EXPECT) as f:
        return json.load(f)


def compare_recorded(name, text):
    """Differences against GNU as: live if available, else the recorded output for `name`."""
    if have_gas():
        return compare(text)
    e = load_expect().get(name)
    if e is None:
        raise LookupError('no recorded GNU as output for %s' % name)
    if e['sha256'] != _digest(text):
        raise LookupError('%s changed since GNU as output was recorded (run '
                          'tests/assembler_amd64_check.py --record where GNU as is available)'
                          % name)
    return A.diff(_from_json(e['object']), cimo_normalized(text))


def record(inputs):
    """inputs: name -> text. Writes GNU as's output for each to data/amd64_expect.json."""
    out = {}
    for name, text in sorted(inputs.items()):
        out[name] = {'sha256': _digest(text), 'object': _to_json(gas_normalized(text))}
    with open(EXPECT, 'w', newline='\n') as f:  # one input per line
        f.write('{\n%s\n}\n' % ',\n'.join(
            '%s: %s' % (json.dumps(k), json.dumps(v, sort_keys=True, separators=(',', ':')))
            for k, v in sorted(out.items())))


# QBE corpus

def corpus():
    return sorted(glob.glob(os.path.expanduser('~/code/qbe/test/*.ssa')))


def c_qbe():
    """A C QBE binary, for when cimo.backend can't produce amd64 yet."""
    env = os.environ.get('QBE')
    if env:
        return env
    for p in ('~/code/qbe-build/qbe.exe', '~/code/qbe-build/qbe', '~/code/qbe/qbe'):
        p = os.path.expanduser(p)
        if os.path.exists(p):
            return p
    return shutil.which('qbe')


def qbe_asm(path, target='amd64_sysv'):
    """QBE IR file -> assembly text: the Python QBE port, else C QBE."""
    env = dict(os.environ, PYTHONPATH=PYROOT)
    p = subprocess.run([sys.executable, '-m', 'cimo.backend', '-t', target, path],
                       capture_output=True, env=env)
    if p.returncode == 0 and p.stdout:
        return p.stdout.decode('latin-1')
    q = c_qbe()
    if q is None:
        raise RuntimeError('qbe failed on %s: %s' % (path, p.stderr.decode(errors='replace')))
    p = subprocess.run([q, '-t', target, path], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError('qbe failed on %s: %s' % (path, p.stderr.decode(errors='replace')))
    return p.stdout.decode('latin-1')


def main():
    args = sys.argv[1:]
    if args == ['--record']:
        import test_assembler_amd64 as T
        record(T.recorded_inputs())
        print('wrote', EXPECT)
        return 0
    files = args or corpus()
    bad = 0
    n = 0
    for f in files:
        for target in (('amd64_sysv', 'amd64_win') if f.endswith('.ssa') else (None,)):
            try:
                text = open(f).read() if target is None else qbe_asm(f, target)
            except RuntimeError as e:
                print(e)
                continue
            n += 1
            try:
                d = compare(text)
            except assembler_amd64.AssemblerError as e:
                d = ['cimo.assembler_amd64: %s' % e]
            if d:
                bad += 1
                print('%s (%s):\n  %s' % (os.path.basename(f), target, '\n  '.join(d[:8])))
    print('%d of %d differ' % (bad, n))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
