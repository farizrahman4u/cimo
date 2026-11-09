/* cimo wasm runtime: runtime/omp.c for Pyodide, which has one thread. A parallel loop runs on a
   team of one, the calling thread, with the same privatization and reductions (the frontend's
   outlining is the same on every target); locks are never contended. */

typedef unsigned long u64;

struct team {
	volatile u64 stop;
	void *pool;
	u64 master;
	long n;
	void (*fn)(void *, long, struct team *);
	void *data;
	volatile long next;
};

long __cimo_crit, __cimo_atomic, __cimo_reduce;
static int nset;
double __cimo_js_time(void);  /* time.perf_counter, from cimo/wasm/pyodide.py */

void
__cimo_lock(volatile long *w)
{
	*w = 1;
}

void
__cimo_unlock(volatile long *w)
{
	*w = 0;
}

void
__cimo_flush_mem(void)
{
}

void
__cimo_par(void (*fn)(void *, long, struct team *), void *data, long nthreads)
{
	struct team t;
	(void)nthreads;
	t.stop = 0;
	t.pool = 0;
	t.master = 0;
	t.n = 1;
	t.fn = fn;
	t.data = data;
	t.next = 0;
	fn(data, 0, &t);
}

u64
__cimo_par_next(struct team *t, u64 n, u64 chunk, long guided, u64 *hi)
{
	u64 lo = (u64)t->next;
	(void)guided;
	if (lo >= n)
		return n;
	*hi = n;  /* a team of one takes the rest at once */
	t->next = (long)n;
	(void)chunk;
	return lo;
}

int omp_get_thread_num(void) { return 0; }
int omp_get_num_threads(void) { return 1; }
int omp_in_parallel(void) { return 0; }
int omp_get_max_threads(void) { return 1; }
void omp_set_num_threads(int n) { nset = n; }  /* kept, though a team is always one thread */
int omp_get_num_procs(void) { return 1; }
int omp_get_thread_limit(void) { return 1; }
int omp_get_level(void) { return 0; }
int omp_get_active_level(void) { return 0; }
int omp_get_dynamic(void) { return 0; }
void omp_set_dynamic(int on) { (void)on; }
int omp_get_nested(void) { return 0; }
void omp_set_nested(int on) { (void)on; }
int omp_get_max_active_levels(void) { return 1; }
void omp_set_max_active_levels(int n) { (void)n; }
double omp_get_wtime(void) { return __cimo_js_time(); }
double omp_get_wtick(void) { return 1e-9; }

typedef struct { volatile int word; } omp_lock_t;

void omp_init_lock(omp_lock_t *l) { l->word = 0; }
void omp_destroy_lock(omp_lock_t *l) { (void)l; }
void omp_set_lock(omp_lock_t *l) { l->word = 1; }
void omp_unset_lock(omp_lock_t *l) { l->word = 0; }
int omp_test_lock(omp_lock_t *l) { if (l->word) return 0; l->word = 1; return 1; }
