"""Run cimo wasm modules under Node (tests/wasm/host.mjs): a stand-in for Pyodide's loader."""

import base64
import json
import os
import shutil
import struct
import subprocess

HOST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'host.mjs')


def available():
    return shutil.which('node') is not None


class Host:
    def __init__(self):
        self.p = subprocess.Popen(['node'] + os.environ.get('NODE_FLAGS', '').split() + [HOST], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  universal_newlines=True)

    def req(self, **m):
        self.p.stdin.write(json.dumps(m) + '\n')
        self.p.stdin.flush()
        r = json.loads(self.p.stdout.readline())
        return r

    def load(self, mod):
        r = self.req(load=base64.b64encode(mod).decode())
        if 'err' in r:
            raise RuntimeError(r['err'])
        return r['exports']

    def call(self, name, sig, *args):
        """sig: the parameter classes ('w,l,d'); returns (value, stdout) or raises."""
        types = [k for k in sig.split(',') if k]
        def enc(a):
            if isinstance(a, float):
                return {'inf': 'Infinity', '-inf': '-Infinity', 'nan': 'NaN'}.get(repr(a), repr(a))
            return a
        r = self.req(call=name, args=[enc(a) for a in args], types=types)
        if 'err' in r:
            raise KernelFailure(r['err'], r.get('out', ''))
        v = r['ret']
        if isinstance(v, str):
            v = float(v) if v in ('NaN', 'Infinity', '-Infinity') else int(v)
        return v, r['out']

    def alloc(self, n):
        return self.req(alloc=n)['ptr']

    def write(self, ptr, data):
        self.req(write=ptr, b64=base64.b64encode(bytes(data)).decode())

    def read(self, ptr, n):
        return base64.b64decode(self.req(read=ptr, n=n)['b64'])

    def array(self, fmt, values):
        data = struct.pack('<%d%s' % (len(values), fmt), *values)
        p = self.alloc(len(data))
        self.write(p, data)
        return p

    def close(self):
        self.p.stdin.close()
        self.p.wait()


class KernelFailure(Exception):
    def __init__(self, err, out):
        Exception.__init__(self, err)
        self.err, self.out = err, out
