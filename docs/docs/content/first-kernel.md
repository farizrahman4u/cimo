# Your first kernel

A kernel is a C function you call from Python. Compile some C, and every function in it that isn't `static` becomes an attribute of the result.

```python
import cimo

k = cimo.compile(r"""
int add(int a, int b) {
    return a + b;
}
""")

k.add(2, 3)   # 5
```

`cimo.compile` returns a module object. Use a raw string (`r"""..."""`) so backslashes in the C, like `\n`, stay as written.

## Arguments and results follow the C signature

cimo reads each function's C signature and converts arguments to match: Python ints for C integers, floats for `float` and `double`. The result comes back as a Python value.

```python
k = cimo.compile(r"""
double mean(double a, double b) { return (a + b) / 2; }
long square(long x) { return x * x; }
""")

k.mean(1, 2)      # 1.5
k.square(10**9)   # 1000000000000000000
```

Calling with the wrong number of arguments is a `TypeError`, as for a Python function.

## printf goes to Python's stdout

```python
k = cimo.compile(r"""
#include <stdio.h>

int hello(void) {
    printf("hello from C\n");
    return 7;
}
""")

k.hello()   # prints "hello from C", returns 7
```

Output goes through `sys.stdout`, so it appears in notebooks and can be captured with `contextlib.redirect_stdout`, in order with Python's own prints.

## A whole program

A module with a `main` can be called directly, and `cimo run` runs a file from the shell:

```python
m = cimo.compile("int main(void) { return 42; }")
m()   # 42
```

```console
$ cimo run program.c arg1 arg2
```

## Several functions, one module

Functions in a module can call each other; only those that aren't `static` are visible from Python:

```python
k = cimo.compile(r"""
static int sq(int x) { return x * x; }
int sum_of_squares(int a, int b) { return sq(a) + sq(b); }
""")

dir(k)   # ['sum_of_squares']
```

Next, pass arrays to kernels, which is where they become useful.
