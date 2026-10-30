#ifndef FLOATSCAN_H
#define FLOATSCAN_H

#include "shgetc.h"

/* cimo: musl's __floatscan returns long double; here long double is double (what musl
   does on targets where they are the same type) */
hidden double __floatscan(FILE *, int, int);

#endif
