/* cimo: what a native kernel's CPython entry (`__cimo_pys_<signature>`, driver._pyentry) calls: argument conversions,
   the GIL, results, and the way back to the kernel's Python call. Python fills __cimo_pyapi when the module loads
   (driver._pyapi), so nothing here depends on Python's headers or on where its library is. */

typedef long long ll;

/* Python's Py_buffer on a 64-bit target */
struct pybuf {
	char *buf;
	void *obj;
	ll len, itemsize;
	int readonly, ndim;
	void *format, *shape, *strides, *suboffsets, *internal;
};

enum { OBTYPE, FLOAT, INT, NONE, ASDOUBLE, FROMDOUBLE, ASLL, FROMLL, FROMULL, ERROCC, ERRCLR, GETBUF, RELBUF, SAVE,
       RESTORE, VCALL, DECREF, INCREF, ASPTR, NAPI };
void *__cimo_pyapi[NAPI];  /* in driver._PYAPI's order */

#define CTX_PY 120   /* execution.CTX_PY: a live Python thread registered the context */
#define CTX_OUT 4096 /* execution.CTX_OUT: the stdout buffer's length */
#define PyBUF_WRITABLE 1

void *__cimo_ctx(void);

static void *type_of(void *o)
{
	return *(void **)((char *)o + (ll)__cimo_pyapi[OBTYPE]);
}

static int failed(void)
{
	if (!((void *(*)(void))__cimo_pyapi[ERROCC])())
		return 0;
	((void (*)(void))__cimo_pyapi[ERRCLR])();
	return 1;
}

static double todouble(void *o, int *st)
{
	double v;
	if (type_of(o) == __cimo_pyapi[FLOAT])
		return *(double *)((char *)o + (ll)__cimo_pyapi[OBTYPE] + 8);
	v = ((double (*)(void *))__cimo_pyapi[ASDOUBLE])(o);
	if (v == -1.0 && failed())
		*st = 1;
	return v;
}

/* an exact int that fits 64 bits; anything else (bool, numpy's integers) takes the Python call */
static ll tolong(void *o, int *st)
{
	int ovf = 0;
	ll v;
	if (type_of(o) != __cimo_pyapi[INT]) {
		*st = 1;
		return 0;
	}
	v = ((ll (*)(void *, int *))__cimo_pyapi[ASLL])(o, &ovf);
	if (ovf || (v == -1 && failed()))
		*st = 1;
	return v;
}

/* a pointer: an int, or a writable buffer of at least a byte, held until __cimo_pyfin* */
static ll topointer(void *o, int *st, struct pybuf *b)
{
	b->obj = 0;
	if (type_of(o) == __cimo_pyapi[INT])
		return tolong(o, st);
	if (((int (*)(void *, struct pybuf *, int))__cimo_pyapi[GETBUF])(o, b, PyBUF_WRITABLE)) {
		failed();
		b->obj = 0;
		*st = 1;
		return 0;
	}
	if (b->len < 1)
		*st = 1;
	return (ll)b->buf;
}

static void release(struct pybuf *b, ll n)
{
	for (; n > 0; n--, b++)
		if (b->obj) {
			((void (*)(struct pybuf *))__cimo_pyapi[RELBUF])(b);
			b->obj = 0;
		}
}

/* the kernel's words (driver.Kernel._fastcall), from the entry's self: its Python call, its post, the number of
   arguments, of pointers, its trampoline, and a kind for each argument: d (double), s (float), p (pointer), i (an
   integer class) */
static ll *words(void *self)
{
	return ((ll *(*)(void *))__cimo_pyapi[ASPTR])(self);
}

/* The arguments into slots, a float as a float; the thread's context, with the GIL released (*ts) and the words in
   *kp, or 0 for the Python call */
char *__cimo_pyargs(void *self, void *const *args, ll nargs, ll *slots, struct pybuf *bufs, void **ts, ll **kp)
{
	ll *k = words(self);
	const char *kind = (const char *)(k + 5);
	struct pybuf *b = bufs;
	char *c;
	int st = 0;
	ll i;
	if (nargs != k[2] || !(c = __cimo_ctx()) || !*(ll *)(c + CTX_PY))
		return 0;
	for (i = 0; i < nargs; i++) {
		switch (kind[i]) {
		case 'd': *(double *)&slots[i] = todouble(args[i], &st); break;
		case 's': *(float *)&slots[i] = (float)todouble(args[i], &st); break;
		case 'p': slots[i] = topointer(args[i], &st, b++); break;
		default: slots[i] = tolong(args[i], &st);
		}
	}
	if (st) {
		release(bufs, b - bufs);
		return 0;
	}
	*ts = ((void *(*)(void))__cimo_pyapi[SAVE])();
	*kp = k;
	return c;
}

typedef void *(*vcall)(void *, void *const *, ll, void *);

/* the GIL back, the buffers released, then the result, or after a call that failed or wrote output the kernel's
   post(result): it flushes and raises */
static void *finish(void *o, char *ctx, ll *k, void *ts, struct pybuf *bufs)
{
	void *r;
	if (!*(ll *)ctx && !*(ll *)(ctx + CTX_OUT))
		return o;
	r = ((vcall)__cimo_pyapi[VCALL])((void *)k[1], &o, 1, 0);
	((void (*)(void *))__cimo_pyapi[DECREF])(o);
	return r;
}

static void back(ll *k, void *ts, struct pybuf *bufs)
{
	((void (*)(void *))__cimo_pyapi[RESTORE])(ts);
	release(bufs, k[3]);
}

void *__cimo_pyfind(double v, char *ctx, ll *k, void *ts, struct pybuf *bufs)
{
	back(k, ts, bufs);
	return finish(((void *(*)(double))__cimo_pyapi[FROMDOUBLE])(v), ctx, k, ts, bufs);
}

void *__cimo_pyfinl(ll v, char *ctx, ll *k, void *ts, struct pybuf *bufs)
{
	back(k, ts, bufs);
	return finish(((void *(*)(ll))__cimo_pyapi[FROMLL])(v), ctx, k, ts, bufs);
}

void *__cimo_pyfinu(ll v, char *ctx, ll *k, void *ts, struct pybuf *bufs)
{
	back(k, ts, bufs);
	return finish(((void *(*)(ll))__cimo_pyapi[FROMULL])(v), ctx, k, ts, bufs);
}

void *__cimo_pyfinv(char *ctx, ll *k, void *ts, struct pybuf *bufs)
{
	back(k, ts, bufs);
	((void (*)(void *))__cimo_pyapi[INCREF])(__cimo_pyapi[NONE]);
	return finish(__cimo_pyapi[NONE], ctx, k, ts, bufs);
}

/* the kernel's Python call (word 0 of k) */
void *__cimo_pyslow(void *self, void *const *args, ll nargs)
{
	return ((vcall)__cimo_pyapi[VCALL])((void *)words(self)[0], args, nargs, 0);
}
