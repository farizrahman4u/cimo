"""The thread pool that runs parallel loops (`#pragma omp parallel for`): native code, one per process.

A module's runtime/omp.c starts a region with the pool's `begin(team, n)`: it claims a pool (one per
master thread running a region at the time, as libgomp keeps one per thread, and per team size), starts workers until it has
n - 1, and hands each the team through a word of its own that only it watches (so the workers a region has no part
for stay asleep, off the cores and hyperthread siblings the others use). Each worker runs its part through
the team's `share` (in the module, so in the worker's context of that module) and marks its own `done` word with
the region's generation; the master runs part 0 itself, then `end(team)` waits for every worker's mark. Workers spin for a while
after a region (`OMP_WAIT_POLICY`; 200 us by default), then sleep on their word: a futex on Linux,
`__ulock_wait` on macOS, `WaitOnAddress` on Windows. Threads are the host's (`pthread_create`,
`CreateThread`), reached through function pointers filled here from ctypes, as cimo.fault reaches
`sigaction`, so the code itself imports nothing.

Ctrl-C (cimo.fault's SIGINT handler, POSIX only) calls `stop(team)`: the team's stop word is set and every
thread of the team still running its part is sent SIGINT, which unwinds it if it is in the loop's own code;
the master sends it again every millisecond to those that were elsewhere (the runtime, a host library) until
all are back. The region then raises the interrupt.

Python's side: the default team size (`set_num_threads`, else `OMP_NUM_THREADS`, else the CPUs this process
may run on), the workers' stack size (`OMP_STACKSIZE`, else 8 MB) and the spin. After a fork the child forgets
the parent's pools (their threads did not come along).
"""

import ctypes
import os
import sys
import threading

from . import loader as _jit

_WIN = sys.platform == 'win32'
_APPLE = sys.platform == 'darwin'

_SOURCE = r'''
typedef unsigned long u64;
typedef unsigned int u32;

#define MAXW 1023
#define ALTSTACK (128 << 10)
#define SLACK (384 << 10)  /* below a worker's first frame: the stack guard's margin and what the thread library
                              keeps at the top */

struct pool;
struct team {
    volatile u64 stop;
    struct pool *pool;
    u64 master;
    long n;
    void *fn, *data;
    volatile long next;
    void (*share)(struct team *, long, u64);
};
struct worker {
    struct pool *pool;
    long index;
    u64 handle;
    struct team *volatile task;  /* its part of the region to start, from the master; taken once */
    volatile long busy;          /* running it */
    volatile long kick;          /* sent SIGINT to stop its part, not yet delivered (see ack) */
    volatile long done;          /* the generation of the last region whose part it finished */
    long pad[8];                 /* done on a cache line of its own, as far as the next worker's */
    u64 limit;
    volatile long go;            /* bumped by the master when it hands over a part; the worker waits on it */
    volatile long sleeping;      /* the worker is asleep on go */
};
struct pool {
    volatile long gen;       /* bumped to start a region; workers wait on its low 32 bits */
    volatile long wakes;     /* bumped by a worker that finds the master asleep; it waits on the low 32 bits */
    volatile long claimed;   /* a master's */
    volatile long mwait;     /* the master is asleep on wakes */
    volatile long mkick;     /* the master was sent SIGINT, not yet delivered */
    u64 master;              /* the master's handle, while claimed */
    long nw;
    struct pool *next;
    struct worker *w[MAXW];
};

struct pool *volatile __cimo_pool_list;
long __cimo_pool_default = 1, __cimo_pool_procs = 1, __cimo_pool_stack = 8 << 20, __cimo_pool_spin = 200;  /* us */
double __cimo_pool_tick;  /* Windows: seconds per QueryPerformanceCounter count */

void *(*__cimo_pool_calloc)(u64, u64);
#ifdef CIMO_WINDOWS
void *(*__cimo_pool_create)(void *, u64, unsigned (*)(void *), void *, unsigned, unsigned *);
int (*__cimo_pool_wait)(volatile void *, void *, u64, unsigned);
void (*__cimo_pool_wake)(volatile void *);
int (*__cimo_pool_guarantee)(unsigned *);
int (*__cimo_pool_yield)(void);
int (*__cimo_pool_counter)(long long *);
#else
int (*__cimo_pool_create)(u64 *, void *, void *(*)(void *), void *);
int (*__cimo_pool_attr_init)(void *);
int (*__cimo_pool_attr_stack)(void *, u64);
int (*__cimo_pool_attr_destroy)(void *);
u64 (*__cimo_pool_self)(void);
int (*__cimo_pool_kill)(u64, int);
int (*__cimo_pool_altstack)(void *, void *);
int (*__cimo_pool_yield)(void);
int (*__cimo_pool_clock)(int, long *);
#ifdef CIMO_APPLE
int (*__cimo_pool_ulock_wait)(u32, volatile void *, u64, u32);
int (*__cimo_pool_ulock_wake)(u32, volatile void *, u64);
#else
long (*__cimo_pool_syscall)(long, ...);
#endif
#endif

void *__cimo_cas(void **p, void *expect, void *desired);
void __cimo_barrier(void);
void __cimo_pause(void);

static long add(volatile long *a, long d)
{
    long o;
    do
        o = *a;
    while ((long)__cimo_cas((void **)a, (void *)o, (void *)(o + d)) != o);
    return o + d;
}

/* sleep while the 32 bits at a are v, for at most ms milliseconds (forever if negative) */
static void wait32(volatile long *a, long v, long ms)
{
#ifdef CIMO_WINDOWS
    u32 cmp = (u32)v;
    __cimo_pool_wait(a, &cmp, 4, ms < 0 ? 0xffffffffu : (unsigned)ms);
#elif defined(CIMO_APPLE)
    __cimo_pool_ulock_wait(1 | 0x01000000, a, (u32)v, ms < 0 ? 0 : (u32)ms * 1000);  /* UL_COMPARE_AND_WAIT, NO_ERRNO */
#else
    long ts[2];
    ts[0] = ms / 1000;
    ts[1] = ms % 1000 * 1000000;
    __cimo_pool_syscall(SYS_FUTEX, a, 128, (long)(u32)v, ms < 0 ? (long *)0 : ts);  /* FUTEX_WAIT_PRIVATE */
#endif
}

static void wake32(volatile long *a)
{
#ifdef CIMO_WINDOWS
    __cimo_pool_wake(a);
#elif defined(CIMO_APPLE)
    __cimo_pool_ulock_wake(1 | 0x100 | 0x01000000, a, 0);  /* ULF_WAKE_ALL */
#else
    __cimo_pool_syscall(SYS_FUTEX, a, 129, 0x7fffffffL);  /* FUTEX_WAKE_PRIVATE */
#endif
}

static double now(void)
{
#ifdef CIMO_WINDOWS
    long long c = 0;
    __cimo_pool_counter(&c);
    return c * __cimo_pool_tick;
#else
    long ts[2];
    ts[0] = ts[1] = 0;
    __cimo_pool_clock(CLOCK_MONOTONIC, ts);
    return ts[0] + ts[1] * 1e-9;
#endif
}

/* the master and the workers of the team still in their parts, except the calling thread: SIGINT */
static void kick(struct team *t)
{
#ifndef CIMO_WINDOWS
    struct pool *p = t->pool;
    u64 me = __cimo_pool_self();
    long i;
    for (i = 0; i < t->n - 1; i++)
        if (p->w[i]->busy && p->w[i]->handle != me) {
            p->w[i]->kick = 1;
            __cimo_barrier();
            __cimo_pool_kill(p->w[i]->handle, 2);
        }
    if (t->master != me) {
        p->mkick = 1;
        __cimo_barrier();
        __cimo_pool_kill(t->master, 2);
    }
#endif
}

/* SIGINT on this thread (cimo.fault's handler): 1 if the pool sent it to stop a part (the region has seen to it,
   or will: so it is not Python's), and it has now arrived. Signals of one number merge while pending, so a flag. */
int __cimo_pool_ack(void)
{
#ifndef CIMO_WINDOWS
    struct pool *p;
    u64 me = __cimo_pool_self();
    long i;
    for (p = __cimo_pool_list; p; p = p->next) {
        if (p->claimed && p->master == me && p->mkick) {
            p->mkick = 0;
            return 1;
        }
        for (i = 0; i < p->nw; i++)
            if (p->w[i]->handle == me) {
                if (!p->w[i]->kick)
                    return 0;
                p->w[i]->kick = 0;
                return 1;
            }
    }
#endif
    return 0;
}

/* Ctrl-C in a thread of team t (cimo.fault's handler, so async-signal-safe only): 1 if the region is still
   running (it will raise the interrupt), 0 if it has ended */
int __cimo_pool_stop(struct team *t)
{
    if (!t->pool)
        return 0;
    if (!t->stop) {
        t->stop = 1;
        kick(t);
    }
    return 1;
}

static void loop(struct worker *w)
{
    struct pool *p = w->pool;
    struct team *t;
    long seen = 0, n;
    double t0;
    for (;;) {
        n = 0;
        t0 = 0;
        while (w->go == seen) {  /* its own word: a region with no part for it doesn't wake it */
            if (++n & 63) {
                __cimo_pause();
                continue;
            }
            if (!t0)
                t0 = now();
            if ((now() - t0) * 1e6 < __cimo_pool_spin) {
                if (!(n & 1023))
                    __cimo_pool_yield();  /* to a thread of the team that may be waiting for a core */
                continue;
            }
            w->sleeping = 1;
            __cimo_barrier();
            if (w->go == seen)
                wait32(&w->go, seen, -1);
            w->sleeping = 0;
        }
        seen = w->go;
        __cimo_barrier();
        t = w->task;
        if (t) {
            w->task = 0;
            w->busy = 1;
            t->share(t, w->index + 1, w->limit);
            w->busy = 0;
            __cimo_barrier();
            w->done = p->gen;  /* the region's (unchanged until it ends); a word each: 15 threads counting down one would
                                 queue on its cache line */
            __cimo_barrier();
            if (p->mwait) {
                add(&p->wakes, 1);
                wake32(&p->wakes);
            }
        }
    }
}

static void setup(struct worker *w)
{
    char here;
    w->limit = (u64)&here - (u64)__cimo_pool_stack + SLACK;
#ifdef CIMO_WINDOWS
    {
        unsigned g = 64 << 10;
        __cimo_pool_guarantee(&g);
    }
#else
    {
        /* an alternate signal stack, to handle a stack overflow on (cimo.fault) */
        char *sp = __cimo_pool_calloc(1, ALTSTACK);
        u64 ss[3];
        if (sp) {
#ifdef CIMO_APPLE
            ss[0] = (u64)sp, ss[1] = ALTSTACK, ss[2] = 0;
#else
            ss[0] = (u64)sp, ss[1] = 0, ss[2] = ALTSTACK;
#endif
            __cimo_pool_altstack(ss, 0);
        }
    }
#endif
}

#ifdef CIMO_WINDOWS
static unsigned run(void *arg)
{
    setup(arg);
    loop(arg);
    return 0;
}
#else
static void *run(void *arg)
{
    setup(arg);
    loop(arg);
    return 0;
}
#endif

static int start(struct pool *p)
{
    struct worker *w = __cimo_pool_calloc(1, sizeof *w);
    if (!w)
        return 0;
    w->pool = p;
    w->index = p->nw;
#ifdef CIMO_WINDOWS
    {
        unsigned tid;
        void *h = __cimo_pool_create(0, (u64)__cimo_pool_stack, run, w, 0x10000, &tid);  /* a reservation */
        if (!h)
            return 0;
        w->handle = (u64)h;
    }
#else
    {
        u64 attr[16];
        int bad;
        __cimo_pool_attr_init(attr);
        __cimo_pool_attr_stack(attr, (u64)__cimo_pool_stack);
        bad = __cimo_pool_create(&w->handle, attr, run, w);
        __cimo_pool_attr_destroy(attr);
        if (bad)
            return 0;
    }
#endif
    p->w[p->nw++] = w;
    return 1;
}

static long begin(struct team *t, long n)
{
    struct pool *p;
    long i;
    if (n > MAXW + 1)
        n = MAXW + 1;
    /* a pool of exactly n - 1 workers, else one to grow to that: the threads of a team of 16 left on the hyperthread
       siblings the scheduler gave them, then woken for a team of 8, ran a memory-bound loop at 0.6x of a pool of
       their own (Sapphire Rapids) */
    for (p = __cimo_pool_list; p; p = p->next)
        if (p->nw == n - 1 && !p->claimed && !__cimo_cas((void **)&p->claimed, 0, (void *)1))
            break;
    if (!p)
        for (p = __cimo_pool_list; p; p = p->next)
            if (p->nw < n - 1 && !p->claimed && !__cimo_cas((void **)&p->claimed, 0, (void *)1))
                break;
    if (!p) {
        p = __cimo_pool_calloc(1, sizeof *p);
        if (!p) {
            t->n = 1;
            return 1;
        }
        p->claimed = 1;
        do
            p->next = __cimo_pool_list;
        while (__cimo_cas((void **)&__cimo_pool_list, p->next, p) != p->next);
    }
    while (p->nw < n - 1 && start(p))
        ;
    if (n > p->nw + 1)
        n = p->nw + 1;
    if (n <= 1) {
        __cimo_barrier();
        p->claimed = 0;
        t->n = 1;
        return 1;
    }
    t->pool = p;
    t->n = n;
#ifndef CIMO_WINDOWS
    t->master = p->master = __cimo_pool_self();
#endif
    p->gen++;  /* the generation the workers mark done with */
    for (i = 0; i < n - 1; i++)
        p->w[i]->task = t;
    __cimo_barrier();
    for (i = 0; i < n - 1; i++)
        p->w[i]->go++;
    __cimo_barrier();
    for (i = 0; i < n - 1; i++)
        if (p->w[i]->sleeping)
            wake32(&p->w[i]->go);
    return n;
}

/* whether the parts of the region all finished */
static int finished(struct pool *p, struct team *t)
{
    long i, g = p->gen;
    for (i = 0; i < t->n - 1; i++)
        if (p->w[i]->done != g)
            return 0;
    return 1;
}

static void end(struct team *t)
{
    struct pool *p = t->pool;
    long n = 0, v;
    double t0 = 0;
    while (!finished(p, t)) {
        if (t->stop) {  /* Ctrl-C: interrupt the parts again, as they come back to their loop's code */
            kick(t);
            wait32(&p->wakes, p->wakes, 1);
            continue;
        }
        if (++n & 63) {
            __cimo_pause();
            continue;
        }
        if (!t0)
            t0 = now();
        if ((now() - t0) * 1e6 < __cimo_pool_spin) {
            if (!(n & 1023))
                __cimo_pool_yield();
            continue;
        }
        v = p->wakes;
        p->mwait = 1;
        __cimo_barrier();
        if (!finished(p, t))
            wait32(&p->wakes, v, 100);  /* woken by Ctrl-C too (EINTR); the timeout is a backstop */
        p->mwait = 0;
    }
    if (t->stop) {  /* the SIGINTs sent arrive before the region ends, so none is left for the next one */
        for (;;) {
            long k = p->mkick;
            for (n = 0; n < t->n - 1; n++)
                k |= p->w[n]->kick;
            if (!k)
                break;
            wait32(&p->wakes, p->wakes, 1);
        }
    }
    __cimo_barrier();
    t->pool = 0;  /* a late Ctrl-C (cimo.fault) finds no team to stop */
    __cimo_barrier();
    p->claimed = 0;
}

static long nthreads(void) { return __cimo_pool_default; }
static long nprocs(void) { return __cimo_pool_procs; }
static double wtime(void) { return now(); }
static void yield(void) { __cimo_pool_yield(); }

struct api {
    long (*begin)(struct team *, long);
    void (*end)(struct team *);
    long (*nthreads)(void);
    long (*nprocs)(void);
    double (*wtime)(void);
    void (*yield)(void);
} __cimo_pool_api = {begin, end, nthreads, nprocs, wtime, yield};

/* after a fork: the parent's workers are not in this process */
void __cimo_pool_forget(void)
{
    __cimo_pool_list = 0;
}
'''

_lock = threading.Lock()
_module = None
_default = None  # set_num_threads(); None: the environment's or the CPUs'


def _defines():
    d = {}
    if _WIN:
        d['CIMO_WINDOWS'] = None
    elif _APPLE:
        d['CIMO_APPLE'] = None
        d['CLOCK_MONOTONIC'] = '6'
    else:
        d['SYS_FUTEX'] = '98' if _jit._ARM64 else '202'
        d['CLOCK_MONOTONIC'] = '1'
    return d


def _cpus():
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _env_threads():
    v = os.environ.get('OMP_NUM_THREADS', '').split(',')[0].strip()
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def _env_stack():
    v = os.environ.get('OMP_STACKSIZE', '').strip().upper()
    if not v:
        return 8 << 20
    unit = {'B': 1, 'K': 1 << 10, 'M': 1 << 20, 'G': 1 << 30}.get(v[-1])
    try:
        n = int(v[:-1] if unit else v) * (unit or 1 << 10)  # a bare number is kilobytes
    except ValueError:
        return 8 << 20
    return max(n, 1 << 20)


def _env_spin():
    v = os.environ.get('OMP_WAIT_POLICY', '').strip().lower()
    return 0 if v == 'passive' else 1e9 if v == 'active' else 200


def _fill(img):
    def put(name, f):
        ctypes.c_void_p.from_address(img.address(name)).value = ctypes.cast(f, ctypes.c_void_p).value

    if _WIN:
        k32 = ctypes.WinDLL('kernel32')
        sync = ctypes.WinDLL('api-ms-win-core-synch-l1-2-0')
        crt = ctypes.CDLL('ucrtbase')
        put('__cimo_pool_calloc', crt.calloc)
        put('__cimo_pool_create', k32.CreateThread)
        put('__cimo_pool_wait', sync.WaitOnAddress)
        put('__cimo_pool_wake', sync.WakeByAddressAll)
        put('__cimo_pool_guarantee', k32.SetThreadStackGuarantee)
        put('__cimo_pool_yield', k32.SwitchToThread)
        put('__cimo_pool_counter', k32.QueryPerformanceCounter)
        freq = ctypes.c_longlong()
        k32.QueryPerformanceFrequency(ctypes.byref(freq))
        ctypes.c_double.from_address(img.address('__cimo_pool_tick')).value = 1.0 / (freq.value or 1)
        return
    libc = _jit._libc
    for name, sym in (('calloc', 'calloc'), ('create', 'pthread_create'), ('attr_init', 'pthread_attr_init'),
                      ('attr_stack', 'pthread_attr_setstacksize'), ('attr_destroy', 'pthread_attr_destroy'),
                      ('self', 'pthread_self'), ('kill', 'pthread_kill'), ('altstack', 'sigaltstack'),
                      ('yield', 'sched_yield'), ('clock', 'clock_gettime')):
        put('__cimo_pool_' + name, getattr(libc, sym))
    if _APPLE:
        put('__cimo_pool_ulock_wait', libc.__ulock_wait)
        put('__cimo_pool_ulock_wake', libc.__ulock_wake)
    else:
        put('__cimo_pool_syscall', libc.syscall)


def _module_():
    """The pool's module, built on first use (under the caller's cache=, as cimo.fault's handler)."""
    global _module
    if _module is not None:
        return _module
    with _lock:
        if _module is None:
            from . import driver, linker
            m = driver.compile(_SOURCE, defines=_defines(), fma=False, wrapv=True, cache=linker.cache_writes.get())
            _fill(m._image)
            _word_set(m, '__cimo_pool_procs', _cpus())
            _word_set(m, '__cimo_pool_stack', _env_stack())
            _word_set(m, '__cimo_pool_spin', int(_env_spin()))
            _module = m
            _apply()
            from . import fault
            fault.set_stop(m._image.address('__cimo_pool_stop'), m._image.address('__cimo_pool_ack'))
            if hasattr(os, 'register_at_fork'):
                os.register_at_fork(after_in_child=_forget)
    return _module


def _word_set(m, name, v):
    ctypes.c_int64.from_address(m._image.address(name)).value = v


def _apply():
    n = _default or _env_threads() or _cpus()
    _word_set(_module, '__cimo_pool_default', n)


def _forget():
    if _module is not None:
        ctypes.c_uint64.from_address(_module._image.address('__cimo_pool_list')).value = 0


def attach(image, address):
    """A module that runs parallel loops: its runtime/omp.c gets the pool's functions."""
    ctypes.c_void_p.from_address(address).value = _module_()._image.address('__cimo_pool_api')


def set_num_threads(n=None):
    """How many threads a parallel loop (`#pragma omp parallel for`) runs on, the calling thread included,
    unless its `num_threads` clause or the thread's `omp_set_num_threads` says otherwise. None: the
    `OMP_NUM_THREADS` environment variable's, else as many as there are CPUs this process may run on."""
    global _default
    if n is not None:
        n = int(n)
        if n < 1:
            raise ValueError('set_num_threads() needs a count of 1 or more, not %d' % n)
    with _lock:
        _default = n
        if _module is not None:
            _apply()


def get_num_threads():
    """How many threads a parallel loop runs on by default (set_num_threads)."""
    return _default or _env_threads() or _cpus()
