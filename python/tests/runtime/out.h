/* Output helpers for the runtime tests: each test file exports
       long run(char *out, long cap)
   which writes its results as text lines into out and returns the length. The same file is
   compiled by cimo (cimo's runtime), by gcc against the runtime sources (musl as gcc builds
   it) and by gcc against glibc; tests/test_runtime.py compares the outputs. The helpers use
   no library function, so they cannot hide differences. */
static char *o_buf;
static long o_cap, o_len;

static void o_init(char *out, long cap)
{
	o_buf = out;
	o_cap = cap;
	o_len = 0;
}

static void o_c(int c)
{
	if (o_len < o_cap - 1) o_buf[o_len++] = c;
}

static void o_s(const char *s)
{
	if (!s) s = "<NULL>";
	while (*s) o_c(*s++);
}

/* bytes with non-printable ones escaped */
static void o_mem(const char *s, long n)
{
	for (long i = 0; i < n; i++) {
		unsigned char c = s[i];
		if (c >= 32 && c < 127 && c != '\\') o_c(c);
		else {
			o_c('\\');
			o_c("0123456789abcdef"[c >> 4]);
			o_c("0123456789abcdef"[c & 15]);
		}
	}
}

static void o_x(unsigned long long v)
{
	char b[17];
	int i = 16;
	do b[--i] = "0123456789abcdef"[v & 15]; while (v >>= 4);
	while (i < 16) o_c(b[i++]);
}

static void o_d(long long v)
{
	char b[21];
	int i = 21;
	unsigned long long u = v < 0 ? -(unsigned long long)v : v;
	do b[--i] = '0' + u % 10; while (u /= 10);
	if (v < 0) o_c('-');
	while (i < 21) o_c(b[i++]);
}

static void o_f(double d)
{
	union { double d; unsigned long long u; } x = { d };
	o_x(x.u);
}

static void o_ff(float f)
{
	union { float f; unsigned u; } x = { f };
	o_x(x.u);
}

static int o_sign(int v)
{
	return (v > 0) - (v < 0);
}

static long o_end(void)
{
	o_buf[o_len] = 0;
	return o_len;
}

/* deterministic random numbers (splitmix64) */
static unsigned long long o_rng = 0x9e3779b97f4a7c15ULL;

static unsigned long long o_rand(void)
{
	unsigned long long z = (o_rng += 0x9e3779b97f4a7c15ULL);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
	z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
	return z ^ (z >> 31);
}
