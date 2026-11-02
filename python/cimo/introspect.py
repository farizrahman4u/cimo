"""What a compiled module was made from, and what the C says, for tools built on cimo: a module's
source, its options, each function's C types and the data it can change. Worked out from the source
when asked, so compiling and the cache don't pay for it."""


class Introspection:
    """Part of a compiled module (Module, WasmModule)."""

    source = None
    options = None
    _found = None

    def _record(self, source, filename, options):
        self.source = source    # the C text it was compiled from
        self.options = options  # the arguments of compile() that aren't the defaults
        self._filename = filename
        for k in self._kernels.values():
            k.module = self

    def _find(self):
        if self._found is None:
            from .driver import describe
            self._found = describe(self.source, self._filename, self.options)
        return self._found

    def signature(self, name):
        """A function's C types, from its name: (return type, (parameter types)). Scalars by size, as
        'int32_t', 'uint8_t', 'double', 'float', '_Bool', 'void'; pointers as 'double *' or 'const double *',
        whatever arrays are between (a VLA is a pointer to its elements). 'struct' and 'union' by value, '...' last if variadic."""
        return self._find()[0][name]

    @property
    def mutable_data(self):
        """Names of the data that isn't const, static locals included: state that compiling the same
        source again wouldn't share with this module."""
        return list(self._find()[1])
