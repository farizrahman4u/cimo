/* cimo: <string.h> (C11 7.24) plus strnlen, strdup, strndup (POSIX), provided by cimo's
   runtime. strerror knows only EDOM, ERANGE and EILSEQ (there is no errno). */
#ifndef __CIMO_STRING_H
#define __CIMO_STRING_H

#ifndef __CIMO_SIZE_T
#define __CIMO_SIZE_T
typedef unsigned long size_t;
#endif
#undef NULL
#define NULL ((void *)0)

void *memcpy(void *restrict s1, const void *restrict s2, size_t n);
void *memmove(void *s1, const void *s2, size_t n);
char *strcpy(char *restrict s1, const char *restrict s2);
char *strncpy(char *restrict s1, const char *restrict s2, size_t n);
char *strcat(char *restrict s1, const char *restrict s2);
char *strncat(char *restrict s1, const char *restrict s2, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);
int strcmp(const char *s1, const char *s2);
int strcoll(const char *s1, const char *s2);
int strncmp(const char *s1, const char *s2, size_t n);
size_t strxfrm(char *restrict s1, const char *restrict s2, size_t n);
void *memchr(const void *s, int c, size_t n);
char *strchr(const char *s, int c);
size_t strcspn(const char *s1, const char *s2);
char *strpbrk(const char *s1, const char *s2);
char *strrchr(const char *s, int c);
size_t strspn(const char *s1, const char *s2);
char *strstr(const char *s1, const char *s2);
char *strtok(char *restrict s1, const char *restrict s2);
void *memset(void *s, int c, size_t n);
char *strerror(int errnum);
size_t strlen(const char *s);
size_t strnlen(const char *s, size_t maxlen);
char *strdup(const char *s);
char *strndup(const char *s, size_t n);

#endif
