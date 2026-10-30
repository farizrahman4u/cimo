/* cimo runtime: the module heap (malloc, calloc, realloc, free, aligned_alloc).

   Exports: malloc, calloc, realloc, free, aligned_alloc,
            __cimo_heap_reset(heap)   called by Python (Module.reset_heap), no kernel running
            __cimo_hook_grow          function pointer, set by Python after loading:
                                      void *grow(void *heap, unsigned long bytes, int kind)
                                      maps bytes (page multiple) RW and returns it, 0 on failure;
                                      kind 0 = a chunk for the heap, 1 = one large block
            __cimo_hook_release       function pointer, set by Python: void release(void *p)
                                      unmaps a large block returned by grow(.., 1)
   Imports (assembly glue in cimo/execution.py, linked into every module):
            void *__cimo_ctx(void)    the calling thread's context (0 if none)
            void *__cimo_cas(void **p, void *expect, void *desired)   returns the old *p

   Freestanding C; no other dependencies.

   Design. Each (module, thread) context holds one heap at offset CTX_HEAP, so malloc/free on
   the owning thread take no lock. Every block has a 16-byte header (owner heap, capacity |
   kind), so the user pointer stays 16-byte aligned and free() needs no lookup. Sizes up to
   MAXSMALL are rounded to one of NCLASS size classes (16..128 by 16, then 4 per power of two)
   and carved from the heap's current chunk with a bump pointer; freed blocks go on per-class
   free lists and are reused for the same class only. A block freed by another thread is pushed
   on its owner's `deferred` list with one compare-and-swap; the owner takes the whole list
   (CAS to 0) when a class list is empty and sorts the blocks into its class lists. Push plus
   take-all has no ABA problem. Chunks come from Python (grow hook, 1 MB doubling to 64 MB) and
   are kept across reset_heap() for reuse; blocks larger than MAXSMALL are mapped one by one
   and unmapped on free. */

#define CTX_HEAP 3072
#define HEAP_BYTES 1024
#define NCLASS 52
#define MAXSMALL 262144UL
#define KIND_SMALL 1UL
#define KIND_LARGE 2UL
#define KIND_ALIGNED 3UL
#define CHUNK0 (1UL << 20)
#define CHUNKMAX (64UL << 20)

typedef unsigned long size_t;

struct heap;
struct hdr {
	struct heap *owner;
	unsigned long info; /* capacity (multiple of 16) | kind; aligned: offset to the block | kind */
};
struct chunk {
	struct chunk *next;
	unsigned long size;
};
struct heap {
	void *free[NCLASS];
	void *deferred; /* blocks freed by other threads (CAS push) */
	char *bump, *end;
	struct chunk *chunks; /* every chunk, oldest first */
	struct chunk *cur;
	unsigned long next; /* size of the next chunk to map */
};

_Static_assert(sizeof(struct heap) <= HEAP_BYTES, "struct heap must fit its context slot");

void *__cimo_ctx(void);
void *__cimo_cas(void **p, void *expect, void *desired);

void *(*__cimo_hook_grow)(void *heap, unsigned long bytes, int kind);
void (*__cimo_hook_release)(void *p);

static struct heap *
myheap(void)
{
	char *c = __cimo_ctx();
	return c ? (struct heap *)(c + CTX_HEAP) : 0;
}

static unsigned long
cls_of(unsigned long n)
{
	unsigned long b, m;

	if (n <= 128)
		return n ? (n - 1) / 16 : 0;
	/* 2^b < n <= 2^(b+1), b >= 7 */
	m = n - 1;
	b = 0;
	while (m >> (b + 1))
		b++;
	return 8 + (b - 7) * 4 + ((n - 1 - (1UL << b)) >> (b - 2));
}

static unsigned long
cls_size(unsigned long c)
{
	unsigned long b;

	if (c < 8)
		return (c + 1) * 16;
	b = 7 + (c - 8) / 4;
	return (1UL << b) + ((c - 8) % 4 + 1) * (1UL << (b - 2));
}

static int
refill(struct heap *h, unsigned long need)
{
	struct chunk *c, *t;
	unsigned long sz;

	c = h->cur ? h->cur->next : h->chunks;
	while (c && c->size - sizeof(struct chunk) < need)
		c = c->next;
	if (!c) {
		sz = h->next ? h->next : CHUNK0;
		while (sz - sizeof(struct chunk) < need)
			sz *= 2;
		if (!__cimo_hook_grow)
			return 0;
		c = __cimo_hook_grow(h, sz, 0);
		if (!c)
			return 0;
		c->next = 0;
		c->size = sz;
		if (!h->chunks) {
			h->chunks = c;
		} else {
			for (t = h->chunks; t->next; t = t->next)
				;
			t->next = c;
		}
		h->next = sz < CHUNKMAX ? sz * 2 : sz;
	}
	h->cur = c;
	h->bump = (char *)c + sizeof(struct chunk);
	h->end = (char *)c + c->size;
	return 1;
}

static void
drain(struct heap *h)
{
	void *l, *o, *n;
	unsigned long c;

	l = h->deferred;
	while ((o = __cimo_cas(&h->deferred, l, 0)) != l)
		l = o;
	while (l) {
		n = *(void **)l;
		c = cls_of(((struct hdr *)l - 1)->info & ~15UL);
		*(void **)l = h->free[c];
		h->free[c] = l;
		l = n;
	}
}

static void *
large(struct heap *h, unsigned long n)
{
	struct hdr *d;
	unsigned long total;

	if (n > (1UL << 62) || !__cimo_hook_grow)
		return 0;
	total = (n + sizeof(struct hdr) + 4095) & ~4095UL;
	d = __cimo_hook_grow(h, total, 1);
	if (!d)
		return 0;
	d->owner = h;
	d->info = (total - sizeof(struct hdr)) | KIND_LARGE;
	return d + 1;
}

static void *
alloc(struct heap *h, unsigned long n)
{
	struct hdr *d;
	unsigned long c, sz;
	void *p;

	if (n > MAXSMALL)
		return large(h, n);
	c = cls_of(n);
	p = h->free[c];
	if (!p && h->deferred) {
		drain(h);
		p = h->free[c];
	}
	if (p) {
		h->free[c] = *(void **)p;
		return p;
	}
	sz = cls_size(c) + sizeof(struct hdr);
	if ((unsigned long)(h->end - h->bump) < sz && !refill(h, sz))
		return 0;
	d = (struct hdr *)h->bump;
	h->bump += sz;
	d->owner = h;
	d->info = cls_size(c) | KIND_SMALL;
	return d + 1;
}

void *
malloc(size_t n)
{
	struct heap *h = myheap();

	return h ? alloc(h, n) : 0;
}

/* the block's own header, and the usable bytes from p on */
static struct hdr *
block(void *p, unsigned long *usable)
{
	struct hdr *d = (struct hdr *)p - 1;
	unsigned long off = 0;

	if ((d->info & 15) == KIND_ALIGNED) {
		off = d->info & ~15UL;
		d = (struct hdr *)((char *)p - off) - 1;
	}
	if (usable)
		*usable = (d->info & ~15UL) - off;
	return d;
}

void
free(void *p)
{
	struct hdr *d;
	struct heap *h;
	void *o;

	if (!p)
		return;
	d = block(p, 0);
	p = d + 1;
	if ((d->info & 15) == KIND_LARGE) {
		__cimo_hook_release(d);
		return;
	}
	h = d->owner;
	if (h == myheap()) {
		*(void **)p = h->free[cls_of(d->info & ~15UL)];
		h->free[cls_of(d->info & ~15UL)] = p;
		return;
	}
	o = h->deferred;
	for (;;) {
		*(void **)p = o;
		if (__cimo_cas(&h->deferred, o, p) == o)
			break;
		o = h->deferred;
	}
}

static void
zero(void *p, unsigned long n)
{
	unsigned long *w = p, i;

	for (i = 0; i < (n + 7) / 8; i++)
		w[i] = 0;
}

void *
calloc(size_t n, size_t m)
{
	void *p;

	if (m && n > ~0UL / m)
		return 0;
	p = malloc(n * m);
	if (p && (block(p, 0)->info & 15) != KIND_LARGE) /* fresh mappings are zero */
		zero(p, n * m);
	return p;
}

void *
realloc(void *p, size_t n)
{
	unsigned long have, i;
	unsigned char *q;

	if (!p)
		return malloc(n);
	block(p, &have);
	if (n <= have)
		return p;
	q = malloc(n);
	if (!q)
		return 0;
	if ((have | (unsigned long)p) % 8 == 0) {
		for (i = 0; i < have / 8; i++)
			((unsigned long *)q)[i] = ((unsigned long *)p)[i];
	} else {
		for (i = 0; i < have; i++)
			q[i] = ((unsigned char *)p)[i];
	}
	free(p);
	return q;
}

void *
aligned_alloc(size_t al, size_t n)
{
	char *raw, *p;
	struct hdr *d;

	if (!al || al & (al - 1))
		return 0;
	if (al <= 16)
		return malloc(n);
	if (n > ~0UL - al)
		return 0;
	raw = malloc(n + al);
	if (!raw || !((unsigned long)raw & (al - 1)))
		return raw;
	p = (char *)(((unsigned long)raw + al - 1) & ~(al - 1));
	d = (struct hdr *)p - 1;
	d->owner = ((struct hdr *)raw - 1)->owner;
	d->info = (unsigned long)(p - raw) | KIND_ALIGNED;
	return p;
}

void
__cimo_heap_reset(void *heap)
{
	struct heap *h = heap;
	int i;

	for (i = 0; i < NCLASS; i++)
		h->free[i] = 0;
	h->deferred = 0;
	h->cur = 0;
	h->bump = 0;
	h->end = 0;
}
