"""Hardware faults in a kernel raise KernelError (cimo.fault), and faults elsewhere end the process
as they did. Each case runs in its own process: a failure here is a crash."""

import os
import re
import subprocess
import sys
import textwrap

import pytest

import hosts

pytestmark = pytest.mark.skipif(not hosts.RUNS, reason='cimo does not run here')

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN = sys.platform == 'win32'

PRELUDE = r'''
import cimo, ctypes, faulthandler, sys, threading
SRC = r"""
int rd(long a) { return *(int *)a; }
void wr(long a) { *(int *)a = 1; }
int dv(int a, int b) { return a / b; }
int call(long a) { int (*f)(void) = (int (*)(void))a; return f() + 1; }
int rec(int n) { char b[1 << 16]; b[n & 1023] = 1; return n == -1 ? b[3] : rec(n + 1) + b[n & 1023]; }
int big(long n) { char b[n]; b[0] = 1; b[n - 1] = 2; return b[0]; }
int ok(int a) { return a + 1; }
"""
k = cimo.compile(SRC)
def fails(f, *a):
    try:
        f(*a)
    except cimo.KernelError as e:
        assert k.ok(1) == 2, 'the module still works'
        return e
    raise AssertionError('no KernelError')
'''


def run(code, env=None, args=()):
    e = dict(os.environ, PYTHONPATH=PY, **(env or {}))
    return subprocess.run([sys.executable, *args, '-c', PRELUDE + textwrap.dedent(code)], env=e, cwd=PY,
                          capture_output=True, text=True, timeout=120)


def check(code, env=None):
    r = run(code, env)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


def crashed(r):
    """The process ended over the fault. On Windows ctypes' own handler can catch what's left, as
    OSError."""
    if WIN:
        return r.returncode == 0xC0000005 or (r.returncode == 1 and 'access violation' in r.stderr)
    return r.returncode == -11


def test_null_read():
    check('''
        e = fails(k.rd, 8)
        assert e.kind == 'memory fault' and e.function == 'rd' and e.kernel == 'rd', e
        assert e.address == 8
        assert str(e) == 'memory fault at 0x8 in rd() (calling rd)', e
    ''')


def test_null_write():
    check('''
        e = fails(k.wr, 16)
        assert e.kind == 'memory fault' and e.function == 'wr' and e.address == 16, e
    ''')


def test_past_the_end_of_a_mapping():
    check('''
        import mmap
        m = mmap.mmap(-1, 2 * mmap.PAGESIZE)
        a = ctypes.addressof(ctypes.c_char.from_buffer(m))
        k2 = cimo.compile("int sum(int *p, long n) { int s = 0; for (long i = 0; i < n; i++) s += p[i]; return s; }")
        assert k2.sum(a, 16) == 0
        del m  # unmapped
        try:
            k2.sum(a, 16)
        except cimo.KernelError as e:
            assert e.kind == 'memory fault' and e.function == 'sum', e
            assert a <= e.address < a + 8192
        else:
            raise AssertionError
    ''')


def test_call_through_a_bad_pointer():
    check('''
        for a in (0, 0x1234):
            e = fails(k.call, a)
            assert e.kind == 'memory fault' and e.function == 'call' and e.address == a, e
    ''')


def test_integer_division():
    out = check('''
        import cimo.loader
        if cimo.loader._X86:
            for a, b in ((1, 0), (-2**31, -1)):
                e = fails(k.dv, a, b)
                assert e.kind == 'division fault' and e.function == 'dv', e
        else:
            assert k.dv(1, 0) == 0
        print('done')
    ''')
    assert 'done' in out


def test_illegal_instruction():
    check('''
        import cimo.loader
        data = b'\\x0f\\x0b' if cimo.loader._X86 else b'\\0\\0\\0\\0'  # ud2, udf
        # a kernel whose own code has the instruction: a constant in its text can't be written from
        # C, so overwrite the start of a function of its own
        k2 = cimo.compile("int f(void) { return 1; } int g(void) { return f() + 1; }")
        a = k2.f.address
        p = cimo.loader
        code = k2._image._maps[0][0]
        size = (k2._image.code_size + p.PAGE - 1) & -p.PAGE
        if sys.platform == 'darwin':
            sys.exit(0)  # MAP_JIT pages
        p._protect(code, size, p.PROT_READ | p.PROT_WRITE)
        ctypes.memmove(a, data, len(data))
        p._protect(code, size, p.PROT_READ | p.PROT_EXEC)
        p.flush_icache(code, size)
        try:
            k2.f()
        except cimo.KernelError as e:
            assert e.kind == 'illegal instruction' and e.function == 'f', e
        else:
            raise AssertionError
    ''')


def test_stack_overflow_past_the_check():
    # the guard check in recursive prologues doesn't see a VLA
    check('''
        e = fails(k.big, 1 << 40)
        assert e.kind == 'stack overflow' and e.function == 'big', e
        e = fails(k.big, 1 << 40)  # and again, with the guard page back
        assert e.kind == 'stack overflow', e
    ''')


def test_stack_overflow_through_touched_pages():
    # down from the top of a huge VLA, one page at a time, into the real end of the stack, on the
    # main thread and on another
    check('''
        k2 = cimo.compile("""
        int down(long n) {
            char b[n];
            for (long i = n - 1; i >= 0; i -= 4096) b[i] = 1;
            return b[0];
        }
        """)
        def go():
            for _ in range(2):
                try:
                    k2.down(1 << 34)
                except cimo.KernelError as e:
                    assert e.kind == 'stack overflow' and e.function == 'down', e
                else:
                    raise AssertionError
        go()
        res = []
        t = threading.Thread(target=lambda: (go(), res.append(1)))
        t.start()
        t.join()
        assert res == [1]
    ''')


def test_threads():
    check('''
        res = []
        def go(n):
            for i in range(200):
                e = fails(k.rd, 8 + 8 * i)
                assert e.address == 8 + 8 * i and k.ok(n) == n + 1
            res.append(n)
        ts = [threading.Thread(target=go, args=(n,)) for n in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert sorted(res) == list(range(8))
    ''')


def test_threads_keep_faulting_while_others_run():
    check('''
        stop, res = [], []
        def run():
            n = 0
            while not stop:
                assert k.ok(n) == n + 1
                n += 1
            res.append(n)
        t = threading.Thread(target=run)
        t.start()
        for _ in range(500):
            fails(k.rd, 8)
        stop.append(1)
        t.join()
        assert res[0] > 0
    ''')


def test_fault_inside_a_callback_unwinds_only_the_inner_kernel():
    check('''
        k2 = cimo.compile("""
        int outer(int (*cb)(void)) { return cb() + 1; }
        int bad(long a) { return *(int *)a; }
        """)
        seen = []
        @ctypes.CFUNCTYPE(ctypes.c_int)
        def cb():
            try:
                k2.bad(8)
            except cimo.KernelError as e:
                seen.append(e.kind)
            return 41
        assert k2.outer(ctypes.cast(cb, ctypes.c_void_p).value) == 42
        assert seen == ['memory fault']
    ''')


def test_many_modules_and_their_slots():
    # more than one table's worth over time, and more live at once than the first table holds
    check('''
        import cimo.fault as f
        for i in range(3):
            m = cimo.compile("int rd%d(long a) { return *(int *)a; }" % i)
            e = None
            try:
                getattr(m, 'rd%d' % i)(8)
            except cimo.KernelError as x:
                e = x
            assert e.kind == 'memory fault'
            del m
        slots = f._tab_slots
        keep = [f.register(k._image) for _ in range(slots + 10)]
        assert f._tab_slots > slots
        fails(k.rd, 8)
        for s in keep: f.unregister(s)
        fails(k.rd, 8)
    ''')


def test_a_cached_image_is_covered():
    # the second process loads the image from the cache
    code = '''
        k3 = cimo.compile("int rd(long a) { return *(int *)a; } int same_in_both;")
        e = None
        try:
            k3.rd(8)
        except cimo.KernelError as x:
            e = x
        assert e.kind == 'memory fault' and e.function == 'rd', e
    '''
    check(code)
    check(code)


def test_fault_outside_any_kernel_still_ends_the_process():
    r = run('ctypes.string_at(8)')
    assert crashed(r), (r.returncode, r.stderr[-500:])


def test_raw_call_of_a_kernel_address_still_ends_the_process():
    # no entry trampoline, so nothing to unwind to
    r = run('''
        f = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int64)(k.rd.address)
        f(8)
    ''')
    assert crashed(r), (r.returncode, r.stderr[-500:])


@pytest.mark.parametrize('order', ['before cimo', 'after cimo'])
def test_faulthandler_keeps_reporting(order):
    # -X faulthandler is on before cimo's handler goes in; enable() after it replaces it until
    # the next module load puts cimo's back, in front
    r = run('''
        if not faulthandler.is_enabled():
            faulthandler.enable()
        k2 = cimo.compile("int rd2(long a) { return *(int *)a; }")
        e = None
        try:
            k2.rd2(8)
        except cimo.KernelError as x:
            e = x
        assert e.kind == 'memory fault', e
        sys.stderr.write('caught\\n')
        sys.stderr.flush()
        def deep_python_frame():
            ctypes.string_at(8)
        deep_python_frame()
    ''', args=['-X', 'faulthandler'] if order == 'before cimo' else [])
    assert 'caught' in r.stderr, r.stderr[-1000:]
    assert len(re.findall(r'line \d+ in deep_python_frame', r.stderr)) == 1, r.stderr[-1000:]
    assert r.stderr.index('caught') < r.stderr.index('deep_python_frame'), 'nothing reported early'
    assert crashed(r), (r.returncode, r.stderr[-500:])


def test_a_previous_handler_is_called():
    if WIN:
        pytest.skip('POSIX signals')
    r = run('''
        import signal
        fired = []
        libc = ctypes.CDLL(None)
        @ctypes.CFUNCTYPE(None, ctypes.c_int)
        def h(sig):
            import os
            os.write(1, b'previous handler\\n')
            os._exit(7)
        libc.signal.restype = ctypes.c_void_p
        libc.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.signal(signal.SIGSEGV, ctypes.cast(h, ctypes.c_void_p).value)
        k2 = cimo.compile("int rd2(long a) { return *(int *)a; }")
        try:
            k2.rd2(8)
        except cimo.KernelError as e:
            assert e.kind == 'memory fault'
        else:
            raise AssertionError
        ctypes.string_at(8)
    ''')
    assert r.returncode == 7 and 'previous handler' in r.stdout, (r.returncode, r.stderr[-500:])


def test_fault_sent_by_kill_ends_the_process():
    if WIN:
        pytest.skip('POSIX signals')
    r = run('''
        import os, signal
        os.kill(os.getpid(), signal.SIGSEGV)
        import time
        time.sleep(5)
        print('survived')
    ''')
    assert 'survived' not in r.stdout and crashed(r), (r.returncode, r.stdout, r.stderr[-300:])


def test_cache_false_writes_nothing(tmp_path):
    # the handler is the process's first compile: under the caller's cache=, like any other
    r = subprocess.run([sys.executable, '-c', 'import cimo\nk = cimo.compile("int f(int *p) { return *p; }", cache=False)\n'
                        'try:\n    k.f(8)\nexcept cimo.KernelError as e:\n    print(e.kind)'],
                       env=dict(os.environ, PYTHONPATH=PY, CIMO_CACHE_DIR=str(tmp_path / 'c'),
                                CIMO_RUNTIME_CACHE=str(tmp_path / 'r')),
                       cwd=PY, capture_output=True, text=True, timeout=120)
    assert r.stdout.strip() == 'memory fault', r.stderr[-1000:]
    assert [str(p) for p in tmp_path.rglob('*')] == []


def test_off():
    r = run('fails(k.rd, 8)', env={'CIMO_FAULTS': '0'})
    assert crashed(r), (r.returncode, r.stderr[-500:])


def test_context_layout_matches_the_handler():
    check('''
        from cimo import execution as x, fault as f
        for off, field in ((x.ERR, 'u64 err'), (x.FAULT, 'u64 fault'), (x.ADDR, 'addr')):
            assert field in f._SOURCE
        assert (x.ERR, x.PC, x.DEPTH, x.LIMIT, x.FAULT, x.ADDR) == (0, 40, 48, 56, 72, 80)
        glue = x.glue(['k'], target='amd64_sysv') + x.glue(['k'], target='amd64_windows') + x.glue(['k'])
        assert 'movq\t%rsp, 72(%r12)' in glue and 'str\tx12, [x11, #96]' in glue, 'the entry sp f._defines names'
        assert (x.E_STACK, x.E_FAULT) == (4, 7) and (f.F_MEMORY, f.F_DIVISION, f.F_ILLEGAL) == (1, 2, 3)
    ''')
