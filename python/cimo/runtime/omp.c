/* cimo runtime: parallel loops (`#pragma omp parallel for`) and <omp.h>.

   The frontend (frontend/omp.py) outlines a parallel loop's body into a function
   fn(data, thread, team) that runs its share of the iterations, and calls
   __cimo_par(fn, data, nthreads) where the loop was. This runs fn on a team of threads: the caller
   and nthreads - 1 workers of the process-wide pool (cimo/parallel.py, threads of the host's own:
   pthread or Win32), and returns when all are done.

   Exports: __cimo_par, __cimo_par_next (dynamic and guided schedules), __cimo_lock and
            __cimo_unlock (critical, atomic, reductions), their lock words __cimo_crit,
            __cimo_atomic and __cimo_reduce, __cimo_flush_mem (flush), omp_* (omp.h),
            __cimo_hook_pool   the pool's functions, set by Python at load (struct pool below)
   Imports (cimo/execution.py's glue): __cimo_ctx, __cimo_cas, __cimo_barrier, __cimo_pause,
            __cimo_call (a call through an entry frame of its own), __cimo_rethrow (unwind with the
            error now in a context), __cimo_hook_ctx; __cimo_flush (stdio/stdout.c)

   Each thread of the team runs its share in its own context of this module (a worker's comes from
   Python the first time it runs code of the module), so it has its own heap and stdout buffer, and
   through an entry frame, so an error or a fault in it unwinds that thread only. The caller waits for
   the others, then raises the error of the lowest-numbered thread that had one, as if it had happened
   on the calling thread. Ctrl-C (cimo.fault) sets the team's stop word and interrupts every thread
   of it; the region then ends with an interrupt. A region inside a region runs on a team of one. */

typedef unsigned long u64;

/* the context's words (cimo/execution.py): error record, then words 88-112, which are this file's */
#define CTX_DEPTH 48
#define CTX_LIMIT 56
#define CTX_TNUM 88  /* omp_get_thread_num */
#define CTX_NTHR 96  /* omp_get_num_threads: 0 or 1 outside a region */
#define CTX_TEAM 104 /* the team this thread runs a part of, for cimo.fault's Ctrl-C (0 outside) */
#define CTX_NSET 112 /* omp_set_num_threads (0: the process's default) */
#define CTX_OUT 4096
#define W(c, off) (*(long *)((char *)(c) + (off)))
#define E_INTERRUPT 8
#define NONE 0x7fffffffffffffffL

/* shared with cimo/parallel.py's C and cimo/fault.py's (stop at 0) */
struct team {
	volatile u64 stop;  /* 0: Ctrl-C: set by cimo.fault's handler (the pool's stop) */
	void *pool;         /* 8 */
	u64 master;         /* 16: the calling thread's handle */
	long n;             /* 24: threads (the outlined code reads it) */
	void (*fn)(void *, long, struct team *); /* 32 */
	void *data;         /* 40 */
	volatile long next; /* 48: the next iteration a dynamic schedule hands out */
	void (*share)(struct team *, long, u64); /* 56: runs part i on a worker, with its stack limit */
	volatile long lock; /* 64: guards err */
	long erridx;        /* 72: the thread whose error err holds (NONE: none) */
	long err[11];       /* 80: its context's first 88 bytes */
	long nset;          /* 168: the master's omp_set_num_threads, which the team's threads see */
};

struct pool {
	long (*begin)(struct team *, long n); /* starts parts 1..n-1 on workers; the threads it got */
	void (*end)(struct team *);           /* waits for them */
	long (*nthreads)(void);               /* the process's default team size */
	long (*nprocs)(void);
	double (*wtime)(void);
	void (*yield)(void);
};

struct pool *__cimo_hook_pool;
long __cimo_crit, __cimo_atomic, __cimo_reduce;

void *__cimo_ctx(void);
void *__cimo_cas(void **p, void *expect, void *desired);
void __cimo_barrier(void);
void __cimo_pause(void);
void __cimo_call(void (*fn)(void *, long, struct team *), void *a, long b, struct team *c);
_Noreturn void __cimo_rethrow(void *ctx);
void __cimo_flush(void);
extern void *(*__cimo_hook_ctx)(void);

static char *
ctx(void)
{
	char *c = __cimo_ctx();
	if (!c && __cimo_hook_ctx)
		c = __cimo_hook_ctx();
	return c;
}

void
__cimo_lock(volatile long *w)
{
	int n = 0;
	while (__cimo_cas((void **)w, 0, (void *)1) != 0) {
		do {
			if (++n < 256)
				__cimo_pause();
			else if (__cimo_hook_pool)
				__cimo_hook_pool->yield();
		} while (*w);
	}
}

void
__cimo_unlock(volatile long *w)
{
	__cimo_barrier();
	*w = 0;
}

void
__cimo_flush_mem(void)
{
	__cimo_barrier();
}

/* the error in context c, the part of thread i: the team's, if no lower-numbered thread had one */
static void
record(struct team *t, char *c, long i)
{
	long k;
	__cimo_lock(&t->lock);
	if (i < t->erridx) {
		t->erridx = i;
		for (k = 0; k < 11; k++)
			t->err[k] = W(c, 8 * k);
	}
	__cimo_unlock(&t->lock);
	for (k = 0; k < 6; k++)
		W(c, 8 * k) = 0;
	W(c, 72) = W(c, 80) = 0;
}

/* part i on a worker thread of the pool, whose stack ends at limit (0: unknown) */
static void
share(struct team *t, long i, u64 limit)
{
	char *c = ctx();
	if (!c) {  /* no context (the module is going away): no frame to unwind to, so as a call by address */
		t->fn(t->data, i, t);
		return;
	}
	if (limit)
		W(c, CTX_LIMIT) = limit;
	W(c, CTX_DEPTH) = 0;  /* a worker runs nothing else of the module (stale after a fork) */
	W(c, CTX_TNUM) = i;
	W(c, CTX_NTHR) = t->n;
	W(c, CTX_TEAM) = (long)t;
	W(c, CTX_NSET) = t->nset;
	__cimo_call(t->fn, t->data, i, t);
	W(c, CTX_TEAM) = 0;
	W(c, CTX_NTHR) = 0;
	W(c, CTX_NSET) = 0;
	if (W(c, 0))
		record(t, c, i);
	if (W(c, CTX_OUT))
		__cimo_flush();
}

void
__cimo_par(void (*fn)(void *, long, struct team *), void *data, long nthreads)
{
	struct team t;
	char *c = ctx();
	long n, tnum, nthr, k;

	t.stop = 0;
	t.pool = 0;
	t.master = 0;
	t.n = 1;
	t.fn = fn;
	t.data = data;
	t.next = 0;
	t.share = share;
	t.lock = 0;
	t.erridx = NONE;
	t.nset = 0;
	if (!c) {
		fn(data, 0, &t);
		return;
	}
	tnum = W(c, CTX_TNUM);
	nthr = W(c, CTX_NTHR);
	t.nset = W(c, CTX_NSET);
	n = nthreads > 0 ? nthreads : W(c, CTX_NSET) > 0 ? W(c, CTX_NSET)
	    : __cimo_hook_pool ? __cimo_hook_pool->nthreads() : 1;
	if (nthr > 1 || W(c, CTX_TEAM) || !__cimo_hook_pool)
		n = 1;  /* inside a region (or no threads): a team of one */
	W(c, CTX_TNUM) = 0;
	if (n <= 1) {
		W(c, CTX_NTHR) = 1;
		fn(data, 0, &t);  /* an error unwinds the caller, as any code of it would */
		W(c, CTX_TNUM) = tnum;
		W(c, CTX_NTHR) = nthr;
		return;
	}
	n = __cimo_hook_pool->begin(&t, n);  /* sets t.n */
	W(c, CTX_NTHR) = n;
	if (n > 1)
		W(c, CTX_TEAM) = (long)&t;
	__cimo_call(fn, data, 0, &t);
	if (W(c, 0))
		record(&t, c, 0);
	if (n > 1)
		__cimo_hook_pool->end(&t);
	W(c, CTX_TEAM) = 0;
	W(c, CTX_TNUM) = tnum;
	W(c, CTX_NTHR) = nthr;
	if (t.erridx == NONE && !t.stop)
		return;
	if (t.erridx == NONE) {  /* Ctrl-C, with no thread in the loop's own code at the time */
		for (k = 0; k < 11; k++)
			t.err[k] = 0;
		t.err[0] = E_INTERRUPT;
	}
	for (k = 0; k < 6; k++)
		W(c, 8 * k) = t.err[k];
	W(c, 72) = t.err[9];
	W(c, 80) = t.err[10];
	__cimo_rethrow(c);
}

/* A dynamic or guided schedule's next chunk of n iterations: returns its first (n when there are none
   left) and sets *hi past its last. */
u64
__cimo_par_next(struct team *t, u64 n, u64 chunk, long guided, u64 *hi)
{
	u64 lo, c, q, end;
	if (!chunk)
		chunk = 1;
	for (;;) {
		lo = (u64)t->next;
		if (lo >= n)
			return n;
		c = chunk;
		if (guided) {
			q = (n - lo + 2 * t->n - 1) / (2 * t->n);
			if (q > c)
				c = q;
		}
		end = n - lo <= c ? n : lo + c;
		if ((u64)__cimo_cas((void **)&t->next, (void *)lo, (void *)end) == lo) {
			*hi = end;
			return lo;
		}
	}
}

/* omp.h */

int
omp_get_thread_num(void)
{
	char *c = __cimo_ctx();
	return c ? (int)W(c, CTX_TNUM) : 0;
}

int
omp_get_num_threads(void)
{
	char *c = __cimo_ctx();
	return c && W(c, CTX_NTHR) > 1 ? (int)W(c, CTX_NTHR) : 1;
}

int
omp_in_parallel(void)
{
	return omp_get_num_threads() > 1;
}

int
omp_get_max_threads(void)
{
	char *c = __cimo_ctx();
	if (c && W(c, CTX_NSET) > 0)  /* as libgomp: the setting, also inside a region (where a region gets one thread) */
		return (int)W(c, CTX_NSET);
	return __cimo_hook_pool ? (int)__cimo_hook_pool->nthreads() : 1;
}

void
omp_set_num_threads(int n)
{
	char *c = ctx();
	if (c)
		W(c, CTX_NSET) = n > 0 ? n : 0;
}

int
omp_get_num_procs(void)
{
	return __cimo_hook_pool ? (int)__cimo_hook_pool->nprocs() : 1;
}

int
omp_get_thread_limit(void)
{
	return 1024;
}

int
omp_get_level(void)
{
	return omp_in_parallel();
}

int
omp_get_active_level(void)
{
	return omp_in_parallel();
}

int
omp_get_dynamic(void)
{
	return 0;
}

void
omp_set_dynamic(int on)
{
	(void)on;
}

int
omp_get_nested(void)
{
	return 0;
}

void
omp_set_nested(int on)
{
	(void)on;
}

int
omp_get_max_active_levels(void)
{
	return 1;
}

void
omp_set_max_active_levels(int n)
{
	(void)n;
}

double
omp_get_wtime(void)
{
	return __cimo_hook_pool ? __cimo_hook_pool->wtime() : 0.0;
}

double
omp_get_wtick(void)
{
	return 1e-9;
}

/* 4 bytes, as gcc's omp.h has it (a program compiled with host_headers=True may use that one) */
typedef struct { volatile int word; } omp_lock_t;

/* compare-and-swap of 4 bytes, through the 8 that hold them (little-endian) */
static int
cas32(volatile int *p, int expect, int desired)
{
	volatile long *w = (volatile long *)((u64)p & ~7UL);
	int shift = (u64)p & 4 ? 32 : 0;
	long old, nw;
	for (;;) {
		old = *w;
		if ((int)(old >> shift) != expect)
			return 0;
		nw = (old & ~(0xffffffffL << shift)) | (long)((u64)(unsigned)desired << shift);
		if ((long)__cimo_cas((void **)w, (void *)old, (void *)nw) == old)
			return 1;
	}
}

void
omp_init_lock(omp_lock_t *l)
{
	l->word = 0;
}

void
omp_destroy_lock(omp_lock_t *l)
{
	(void)l;
}

void
omp_set_lock(omp_lock_t *l)
{
	int n = 0;
	while (!cas32(&l->word, 0, 1)) {
		do {
			if (++n < 256)
				__cimo_pause();
			else if (__cimo_hook_pool)
				__cimo_hook_pool->yield();
		} while (l->word);
	}
}

void
omp_unset_lock(omp_lock_t *l)
{
	__cimo_barrier();
	l->word = 0;
}

int
omp_test_lock(omp_lock_t *l)
{
	return cas32(&l->word, 0, 1);
}
