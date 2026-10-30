/* string.h: edges of the word-at-a-time paths (lengths and alignments), overlap, signedness. */
#include <string.h>
#include "out.h"

static char a[512], b[512];

static void fill(char *p, long n, int seed)
{
	for (long i = 0; i < n; i++) p[i] = (char)(seed + i * 7 + (i >> 3));
}

static long sum(const char *p, long n)
{
	long h = 0;
	for (long i = 0; i < n; i++) h = h * 31 + (unsigned char)p[i];
	return h;
}

long run(char *out, long cap)
{
	o_init(out, cap);
	/* strlen, strnlen */
	o_s("strlen");
	for (int off = 0; off < 16; off++)
		for (int len = 0; len < 72; len++) {
			memset(a, 'x', sizeof a);
			a[off + len] = 0;
			o_c(' ');
			o_d(strlen(a + off));
			o_c('/');
			o_d(strnlen(a + off, len / 2 + off % 3));
		}
	o_c('\n');
	/* memset */
	o_s("memset");
	for (int off = 0; off < 16; off++)
		for (int n = 0; n < 140; n += (n < 70 ? 1 : 13)) {
			fill(a, sizeof a, off);
			memset(a + off, 0xa5 + n, n);
			o_c(' ');
			o_d(sum(a, 200));
		}
	o_c('\n');
	/* memcpy */
	o_s("memcpy");
	for (int so = 0; so < 8; so++)
		for (int d = 0; d < 8; d++)
			for (int n = 0; n < 140; n += (n < 40 ? 1 : 11)) {
				fill(a, sizeof a, n);
				fill(b, sizeof b, so + 3);
				o_c(' ');
				o_d(memcpy(b + d, a + so, n) == b + d);
				o_d(sum(b, 200));
			}
	o_c('\n');
	/* memmove, both directions */
	o_s("memmove");
	for (int sh = -9; sh <= 9; sh++)
		for (int n = 0; n < 80; n += (n < 40 ? 1 : 7)) {
			fill(a, sizeof a, n + sh);
			o_c(' ');
			o_d(memmove(a + 100 + sh, a + 100, n) == a + 100 + sh);
			o_d(sum(a, 300));
		}
	o_c('\n');
	/* comparisons (signs only: magnitudes are unspecified) */
	o_s("cmp");
	for (int i = 0; i < 300; i++) {
		int n = o_rand() % 20;
		for (int k = 0; k < n + 1; k++) {
			a[k] = "ab\x80\xff\x01z"[o_rand() % 6];
			b[k] = (o_rand() % 4) ? a[k] : "ab\x80\xff\x01z"[o_rand() % 6];
		}
		a[o_rand() % (n + 1)] = o_rand() % 3 ? a[0] : 0;
		b[n] = 0;
		a[n] = 0;
		o_c(' ');
		o_d(o_sign(memcmp(a, b, n)));
		o_d(o_sign(strcmp(a, b)));
		o_d(o_sign(strncmp(a, b, i % 7)));
		o_d(o_sign(strcoll(a, b)));
	}
	o_c('\n');
	/* searching */
	o_s("chr");
	for (int off = 0; off < 9; off++)
		for (int c = -1; c < 260; c += 1 + (c > 10) * 17) {
			for (int k = 0; k < 60; k++)
				a[off + k] = (char)(k * 37 + 1);
			a[off + 60] = 0;
			char *p = memchr(a + off, c, 61);
			char *q = strchr(a + off, c);
			char *r = strrchr(a + off, c);
			o_c(' ');
			o_d(p ? p - a : -1);
			o_c(',');
			o_d(q ? q - a : -1);
			o_c(',');
			o_d(r ? r - a : -1);
		}
	o_c('\n');
	o_s("span");
	{
		static const char *s[] = { "", "abcabc", "hello, world", "  \t x", "xyz", "aaaa" };
		static const char *t[] = { "", "a", "abc", " \t", "lo", "zyx", "\x80" };
		for (int i = 0; i < 6; i++)
			for (int j = 0; j < 7; j++) {
				const char *p = strpbrk(s[i], t[j]);
				o_c(' ');
				o_d(strspn(s[i], t[j]));
				o_c(',');
				o_d(strcspn(s[i], t[j]));
				o_c(',');
				o_d(p ? p - s[i] : -1);
			}
	}
	o_c('\n');
	o_s("strstr");
	for (int i = 0; i < 400; i++) {
		int n = o_rand() % 40, m = o_rand() % (i < 200 ? 4 : 12);
		int al = 2 + i % 3;
		for (int k = 0; k < n; k++) a[k] = 'a' + o_rand() % al;
		for (int k = 0; k < m; k++) b[k] = 'a' + o_rand() % al;
		a[n] = b[m] = 0;
		if (i % 5 == 0 && m <= n) memcpy(a + (n - m) / 2, b, m);
		char *p = strstr(a, b);
		o_c(' ');
		o_d(p ? p - a : -1);
	}
	o_c('\n');
	/* copying */
	o_s("copy");
	for (int n = 0; n < 12; n++) {
		memset(a, 'Q', 40);
		memset(b, 0, 40);
		strcpy(b, "hello");
		o_c(' ');
		o_d(strncpy(a, b, n) == a);
		o_mem(a, 14);
		memset(a, 0, 40);
		strcpy(a, "ab");
		o_c(' ');
		o_d(strncat(a, "cdefgh", n) == a);
		o_s(a);
		o_c(' ');
		o_d(strcat(a, "!") == a);
		o_s(a);
		o_c(' ');
		o_d(strxfrm(b, "xfrm", n));
		if (n > 4) o_s(b);
	}
	o_c('\n');
	o_s("strtok");
	{
		char s[] = "  a,b;;c d,,; e ";
		for (char *t = strtok(s, " ,;"); t; t = strtok(0, " ,;")) {
			o_c(' ');
			o_s(t);
		}
		char s2[] = ";;;";
		o_c(' ');
		o_s(strtok(s2, ";"));
	}
	o_c('\n');
	return o_end();
}
