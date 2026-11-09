/* What the compiler calls for big copies and zeroing in GPU code: byte by byte (they are rare in kernels). */
typedef unsigned long size_t;

void *memcpy(void *restrict d, const void *restrict s, size_t n)
{
	unsigned char *p = d;
	const unsigned char *q = s;
	for (size_t i = 0; i < n; i++)
		p[i] = q[i];
	return d;
}

void *memmove(void *d, const void *s, size_t n)
{
	unsigned char *p = d;
	const unsigned char *q = s;
	if (p < q)
		for (size_t i = 0; i < n; i++)
			p[i] = q[i];
	else
		for (size_t i = n; i-- > 0;)
			p[i - 0] = q[i];
	return d;
}

void *memset(void *d, int c, size_t n)
{
	unsigned char *p = d;
	for (size_t i = 0; i < n; i++)
		p[i] = (unsigned char)c;
	return d;
}
