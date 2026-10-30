import cimo

k = cimo.compile(r"""
int main(void) {
    printf("Hello World!\n");
    return 0;
}
""")
k()
