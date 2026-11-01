"""Third-party C programs that check their own results (corpora.py fetches them): gcc's
torture tests and c-testsuite. Each program is compiled by cimo and its main() run, in worker
processes so that a crash or hang fails one test only.

corpora_expect.txt lists the programs that are not expected to pass:
  unsupported  uses something cimo rejects (GNU extensions, long double, ...): a compile
               error is expected; passing is fine too (other targets accept some).
  skip         not run: relies on gcc-specific behavior, or too slow (the note says which).
  bug          a known cimo bug: must still fail (an unexpected pass asks to update the list).
Every other program must pass.

    python tests/corpora.py                      # fetch the corpora (once)
    pytest tests/test_corpora.py
    python tests/test_corpora.py --update        # rewrite the 'unsupported' entries (on Linux:
                                                 # macOS accepts long double, so its list differs)
"""

import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import corpora

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECT = os.path.join(HERE, 'corpora_expect.txt')
TIMEOUT = 120  # seconds per program, compile and run


def read_expect():
    """{'corpus/file': (category, note)}"""
    out = {}
    with open(EXPECT, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if line and not line.startswith('#'):
                key, cat, note = (line.split(None, 2) + [''])[:3]
                out[key] = (cat, note)
    return out


class Worker:
    """A corpora_worker.py process, restarted after a crash or a timeout."""

    def __init__(self, env):
        self.env = env
        self.p = None

    def run(self, path):
        if self.p is None:
            self.p = subprocess.Popen(
                [sys.executable, os.path.join(HERE, 'corpora_worker.py')],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                env=self.env, universal_newlines=True, encoding='utf-8', bufsize=1)
            self.q = queue.Queue()
            threading.Thread(target=self._read, args=(self.p, self.q), daemon=True).start()
        self.p.stdin.write(path + '\n')
        self.p.stdin.flush()
        try:
            line = self.q.get(timeout=TIMEOUT)
        except queue.Empty:
            self.close()
            return 'TIMEOUT after %d s' % TIMEOUT
        if line is None:
            rc = self.p.wait()
            self.p = None
            return 'DIED exit status %d' % rc
        return line

    @staticmethod
    def _read(p, q):
        for line in p.stdout:
            q.put(line.rstrip('\n'))
        q.put(None)

    def close(self):
        if self.p is not None:
            self.p.kill()
            self.p.wait()
            self.p = None


def run_all(keys, jobs=None):
    """{'corpus/file': result line} for the given keys."""
    env = dict(os.environ)
    env.setdefault('CIMO_CACHE_DIR', os.path.join(corpora.DIR, '.cache'))
    os.makedirs(env['CIMO_CACHE_DIR'], exist_ok=True)
    jobs = jobs or os.cpu_count() or 4
    local = threading.local()
    workers = []

    def one(key):
        w = getattr(local, 'w', None)
        if w is None:
            w = local.w = Worker(env)
            workers.append(w)
        name, rel = key.split('/', 1)
        line = os.path.join(corpora.root(name), rel)
        h = corpora.harness(name, rel)
        if h:
            line += '\t%s\t%s' % (h[0], os.pathsep.join(h[1]))
        return key, w.run(line)
    try:
        with ThreadPoolExecutor(jobs) as ex:
            return dict(ex.map(one, keys))
    finally:
        for w in workers:
            w.close()


def all_keys():
    return [n + '/' + f for n in corpora.CORPORA for f in corpora.files(n)]


KEYS = all_keys()
_results = {}


@pytest.fixture(scope='module')
def expect():
    return read_expect()


@pytest.fixture
def result(request, expect):
    """This program's result; the first call runs every selected program of the session at
    once, in parallel."""
    if not _results:
        keys = [i.callspec.params['key'] for i in request.session.items
                if getattr(i, 'callspec', None) and 'key' in i.callspec.params
                and i.module is request.module]
        keys = [k for k in keys if expect.get(k, ('',))[0] != 'skip']
        _results.update(run_all(keys))
    return _results.get(request.node.callspec.params['key'])


@pytest.mark.skipif(not KEYS, reason='corpora not fetched: python tests/corpora.py')
@pytest.mark.parametrize('key', KEYS or ['none'])
def test_program(key, expect, request):
    cat, note = expect.get(key, ('', ''))
    if cat == 'skip':
        pytest.skip(note)
    res = request.getfixturevalue('result')
    if cat == 'bug':
        assert res != 'PASS', 'known bug now passes: remove %s from corpora_expect.txt' % key
        pytest.xfail('%s (%s)' % (note, res))
    if cat == 'unsupported':
        assert res == 'PASS' or res.startswith('COMPILE '), res
        return
    assert res == 'PASS', res


def _note(res):
    """A short, path-free reason from a COMPILE result line."""
    msg = res[len('COMPILE '):]
    msg = msg.split(': error: ', 1)[-1]
    return msg.split(';', 1)[0][:120]


def update():
    old = read_expect() if os.path.exists(EXPECT) else {}
    keys = [k for k in KEYS if old.get(k, ('',))[0] != 'skip']
    res = run_all(keys)
    lines = []
    triage = []
    for k in KEYS:
        cat, note = old.get(k, ('', ''))
        r = res.get(k)
        if cat in ('skip', 'bug'):
            lines.append((k, cat, note))
        elif r.startswith('COMPILE '):
            lines.append((k, 'unsupported', _note(r)))
        elif r != 'PASS':
            triage.append((k, r))
    head = []
    with open(EXPECT, encoding='utf-8') as f:
        for line in f:
            if not line.startswith('#'):
                break
            head.append(line)
    w = max(len(k) for k, _, _ in lines)
    with open(EXPECT, 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(head)
        for k, cat, note in lines:
            f.write(('%-*s  %-11s  %s' % (w, k, cat, note)).rstrip() + '\n')
    n = len(res)
    print('%d run: %d pass, %d unsupported, %d need triage (add as skip or bug):'
          % (n, sum(r == 'PASS' for r in res.values()),
             sum(c == 'unsupported' for _, c, _ in lines), len(triage)))
    for k, r in triage:
        print('  %s  %s' % (k, r))


if __name__ == '__main__':
    if sys.argv[1:] == ['--update']:
        update()
    else:
        sys.exit(__doc__)
