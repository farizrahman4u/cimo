import cimo

k = cimo.compile(r"""
#include <stdio.h>

int main(void) {
    printf("Hello World!\n");
    return 0;
}
""")
k()
