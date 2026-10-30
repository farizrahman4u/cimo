/* hits an assertion in C cproc (qbe.c funcexpr: nullptr constant); both must abort */
int f(void) { void *p = nullptr; return p == nullptr; }
