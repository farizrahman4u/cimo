/* cimo runtime: there is no errno. musl code that sets it writes to a per-file dummy. */
#ifndef __CIMO_RT_ERRNO_H
#define __CIMO_RT_ERRNO_H
static int __cimo_errno_unused;
#define errno __cimo_errno_unused
#define EDOM 33
#define ERANGE 34
#define EINVAL 22
#define EOVERFLOW 75
#define EILSEQ 84
#endif
