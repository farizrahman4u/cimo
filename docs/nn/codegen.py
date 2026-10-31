"""Neural network graph (JSON) -> C: forward pass, backward pass and Adam, specialized to the
graph. Used by the website's training demo (docs/nn.js), which compiles the C with cimo.

Graph: {"nodes": [{"id": "a", "op": "input"}, {"id": "b", "op": "dense", "units": 8,
"inputs": ["a"]}, ..., {"id": "z", "op": "output", "inputs": ["y"]}]}

Ops: input (x, y: 2 values), dense (units), relu, tanh, sigmoid, sin, square, add, concat,
output (binary classifier: one logit in, sigmoid + cross-entropy).

The generated C exports
    void  init(int seed, float *P)
    float train(int steps, int n, const float *X, const float *Y, float *P, float *M,
                float *V, float *WS, float lr, int t)
    void  predict(int n, const float *X, const float *P, float *WS, float *out)
    void  acts(int n, const float *X, const float *P, float *WS, float *out)
P/M/V: parameters and Adam moments; WS: workspace (activations, their gradients, parameter
gradients) laid out by the generator; n <= NMAX.
"""

import json

NMAX = 4096
ELEMWISE = ('relu', 'tanh', 'sigmoid', 'sin', 'square')
MAX_UNITS = 32


class GraphError(Exception):
    pass


def _check(graph):
    """Validate; return (order of node ids needed by the output, nodes by id, widths)."""
    nodes = {}
    for n in graph.get('nodes', []):
        if n['id'] in nodes:
            raise GraphError('duplicate node id %s' % n['id'])
        nodes[n['id']] = n
    outs = [n for n in nodes.values() if n['op'] == 'output']
    if len(outs) != 1:
        raise GraphError('the graph needs exactly one Output node')
    want = {'input': 0, 'dense': 1, 'output': 1, 'add': 2, 'concat': 2}
    order, state = [], {}

    def visit(i):
        if state.get(i) == 1:
            raise GraphError('the graph has a cycle')
        if state.get(i) == 2:
            return
        n = nodes.get(i)
        if n is None:
            raise GraphError('a connection points to a missing node')
        k = want.get(n['op'], 1 if n['op'] in ELEMWISE else None)
        if k is None:
            raise GraphError('unknown op %s' % n['op'])
        ins = n.get('inputs') or []
        if len(ins) != k or any(x is None for x in ins):
            raise GraphError('%s needs %d input%s connected' % (label(n), k, 's' if k > 1 else ''))
        state[i] = 1
        for x in ins:
            visit(x)
        state[i] = 2
        order.append(i)

    visit(outs[0]['id'])
    width = {}
    for i in order:
        n = nodes[i]
        ins = [width[x] for x in n.get('inputs') or []]
        op = n['op']
        if op == 'input':
            width[i] = 2
        elif op == 'dense':
            u = int(n.get('units', 8))
            if not 1 <= u <= MAX_UNITS:
                raise GraphError('Dense units must be between 1 and %d' % MAX_UNITS)
            width[i] = u
        elif op in ELEMWISE:
            width[i] = ins[0]
        elif op == 'add':
            if ins[0] != ins[1]:
                raise GraphError('Add needs inputs of the same width (got %d and %d)' % tuple(ins))
            width[i] = ins[0]
        elif op == 'concat':
            width[i] = ins[0] + ins[1]
        elif op == 'output':
            if ins[0] != 1:
                raise GraphError('Output takes one value per point (a class score): end with '
                                 'Dense(1), not a width of %d' % ins[0])
            width[i] = 1
    return order, nodes, width


def label(n):
    return {'dense': 'Dense(%s)' % n.get('units', 8)}.get(n['op'], n['op'].capitalize())


def generate(graph):
    """Graph (dict or JSON text) -> {'c', 'nparams', 'ws', 'acts': [(id, width, offset)]}."""
    if isinstance(graph, str):
        graph = json.loads(graph)
    order, nodes, width = _check(graph)
    idx = {i: k for k, i in enumerate(order)}
    name = {}
    off = 0
    acts_layout = []
    for i in order:  # one activation array per node (the input is X itself)
        n = nodes[i]
        if n['op'] == 'input':
            name[i] = 'X'
        elif n['op'] == 'output':
            name[i] = name[n['inputs'][0]]
        else:
            name[i] = 'a%d' % idx[i]
    nparams = 0
    poff = {}
    for i in order:
        n = nodes[i]
        if n['op'] == 'dense':
            fan_in = width[n['inputs'][0]]
            poff[i] = (nparams, nparams + fan_in * width[i])  # W, b
            nparams += fan_in * width[i] + width[i]
    hidden = [i for i in order if nodes[i]['op'] not in ('input', 'output')]
    wsoff = {}
    for i in hidden:
        wsoff[i] = off
        off += NMAX * width[i]
    gradoff = {}
    for i in hidden:
        gradoff[i] = off
        off += NMAX * width[i]
    gp = off
    off += max(nparams, 1)
    aoff = 0
    for i in hidden:
        acts_layout.append((i, width[i], aoff))
        aoff += width[i]

    out_node = next(i for i in order if nodes[i]['op'] == 'output')
    logit = nodes[out_node]['inputs'][0]
    L = []
    w = L.append
    chain = ' -> '.join(label(nodes[i]) for i in order)
    w('/* Generated from the graph by docs/nn/codegen.py: %s.' % chain)
    w('   %d parameters. Activations, their gradients and the parameter gradients live in WS. */'
      % nparams)
    w('#include <math.h>')
    w('')
    w('#define NP %d' % nparams)
    w('')
    w('static void views(float *WS, float **a, float **g)')
    w('{')
    for i in hidden:
        w('    a[%d] = WS + %d; g[%d] = WS + %d;' % (idx[i], wsoff[i], idx[i], gradoff[i]))
    w('}')
    w('')

    def arr(i):
        return 'X' if nodes[i]['op'] == 'input' else 'a[%d]' % idx[i]

    def grad(i):
        return None if nodes[i]['op'] == 'input' else 'g[%d]' % idx[i]

    # forward
    w('static void forward(int n, const float *X, const float *P, float **a)')
    w('{')
    for i in order:
        n = nodes[i]
        op = n['op']
        if op in ('input', 'output'):
            continue
        ins = n['inputs']
        y, wd = arr(i), width[i]
        w('    /* %s */' % label(n))
        if op == 'dense':
            x, fi = arr(ins[0]), width[ins[0]]
            W, B = poff[i]
            w('    for (int s = 0; s < n; s++)')
            w('        for (int j = 0; j < %d; j++) {' % wd)
            w('            float acc = P[%d + j];' % B)
            w('            for (int i = 0; i < %d; i++)' % fi)
            w('                acc += %s[s * %d + i] * P[%d + i * %d + j];' % (x, fi, W, wd))
            w('            %s[s * %d + j] = acc;' % (y, wd))
            w('        }')
        elif op in ELEMWISE:
            x = arr(ins[0])
            f = {'relu': '%s[k] > 0 ? %s[k] : 0', 'tanh': 'tanhf(%s[k])',
                 'sigmoid': '1 / (1 + expf(-%s[k]))', 'sin': 'sinf(%s[k])',
                 'square': '%s[k] * %s[k]'}[op]
            e = f % ((x,) * f.count('%s'))
            w('    for (int k = 0; k < n * %d; k++)' % wd)
            w('        %s[k] = %s;' % (y, e))
        elif op == 'add':
            x1, x2 = arr(ins[0]), arr(ins[1])
            w('    for (int k = 0; k < n * %d; k++)' % wd)
            w('        %s[k] = %s[k] + %s[k];' % (y, x1, x2))
        elif op == 'concat':
            x1, x2 = arr(ins[0]), arr(ins[1])
            w1, w2 = width[ins[0]], width[ins[1]]
            w('    for (int s = 0; s < n; s++) {')
            w('        for (int i = 0; i < %d; i++) %s[s * %d + i] = %s[s * %d + i];'
              % (w1, y, wd, x1, w1))
            w('        for (int i = 0; i < %d; i++) %s[s * %d + %d + i] = %s[s * %d + i];'
              % (w2, y, wd, w1, x2, w2))
            w('    }')
    w('}')
    w('')

    # backward: returns the mean loss; fills g[] and GP
    lg = arr(logit)
    w('static float backward(int n, const float *X, const float *Y, const float *P, float **a, '
      'float **g, float *GP)')
    w('{')
    w('    float loss = 0;')
    for i in hidden:
        w('    for (int k = 0; k < n * %d; k++) g[%d][k] = 0;' % (width[i], idx[i]))
    w('    for (int k = 0; k < NP; k++) GP[k] = 0;')
    w('    /* sigmoid + binary cross-entropy on the logit */')
    w('    for (int s = 0; s < n; s++) {')
    w('        float z = %s[s], y = Y[s];' % lg)
    w('        loss += (z > 0 ? z : 0) - z * y + logf(1 + expf(-fabsf(z)));')
    if grad(logit):
        w('        %s[s] += (1 / (1 + expf(-z)) - y) / n;' % grad(logit))
    w('    }')
    for i in reversed(order):
        n = nodes[i]
        op = n['op']
        if op in ('input', 'output'):
            continue
        ins = n['inputs']
        gy, y, wd = grad(i), arr(i), width[i]
        w('    /* d %s */' % label(n))
        if op == 'dense':
            x, fi, gx = arr(ins[0]), width[ins[0]], grad(ins[0])
            W, B = poff[i]
            w('    for (int s = 0; s < n; s++)')
            w('        for (int j = 0; j < %d; j++) {' % wd)
            w('            float d = %s[s * %d + j];' % (gy, wd))
            w('            GP[%d + j] += d;' % B)
            w('            for (int i = 0; i < %d; i++) {' % fi)
            w('                GP[%d + i * %d + j] += %s[s * %d + i] * d;' % (W, wd, x, fi))
            if gx:
                w('                %s[s * %d + i] += P[%d + i * %d + j] * d;' % (gx, fi, W, wd))
            w('            }')
            w('        }')
        elif op in ELEMWISE:
            x, gx = arr(ins[0]), grad(ins[0])
            if not gx:
                continue
            d = {'relu': '(%(x)s[k] > 0 ? %(g)s[k] : 0)',
                 'tanh': '%(g)s[k] * (1 - %(y)s[k] * %(y)s[k])',
                 'sigmoid': '%(g)s[k] * %(y)s[k] * (1 - %(y)s[k])',
                 'sin': '%(g)s[k] * cosf(%(x)s[k])',
                 'square': '%(g)s[k] * 2 * %(x)s[k]'}[op] % {'x': x, 'y': y, 'g': gy}
            w('    for (int k = 0; k < n * %d; k++)' % wd)
            w('        %s[k] += %s;' % (gx, d))
        elif op == 'add':
            for x in ins:
                if grad(x):
                    w('    for (int k = 0; k < n * %d; k++) %s[k] += %s[k];' % (wd, grad(x), gy))
        elif op == 'concat':
            w1, w2 = width[ins[0]], width[ins[1]]
            for x, base, wx in ((ins[0], 0, w1), (ins[1], w1, w2)):
                if grad(x):
                    w('    for (int s = 0; s < n; s++)')
                    w('        for (int i = 0; i < %d; i++) %s[s * %d + i] += %s[s * %d + %d + i];'
                      % (wx, grad(x), wx, gy, wd, base))
    w('    return loss / n;')
    w('}')
    w('')

    # init: Xavier-scaled weights from a small LCG, zero biases
    w('void init(int seed, float *P)')
    w('{')
    w('    unsigned r = 2463534242u ^ (unsigned)seed * 2654435761u;')
    w('    for (int k = 0; k < NP; k++) P[k] = 0;')
    for i in order:
        if nodes[i]['op'] == 'dense':
            W, B = poff[i]
            fi, fo = width[nodes[i]['inputs'][0]], width[i]
            w('    for (int k = 0; k < %d; k++) {  /* %s */' % (fi * fo, label(nodes[i])))
            w('        float u = 0;')
            w('        for (int q = 0; q < 4; q++) { r ^= r << 13; r ^= r >> 17; r ^= r << 5; '
              'u += (r >> 8) * (1.0f / 16777216.0f) - 0.5f; }')
            w('        P[%d + k] = u * %.6ff;' % (W, (2.0 / (fi + fo)) ** 0.5 * 1.7320508))
            w('    }')
    w('}')
    w('')
    w('float train(int steps, int n, const float *X, const float *Y, float *P, float *M, '
      'float *V, float *WS, float lr, int t)')
    w('{')
    w('    float *a[%d], *g[%d], loss = 0;' % (len(order), len(order)))
    w('    views(WS, a, g);')
    w('    for (int it = 0; it < steps; it++) {')
    w('        forward(n, X, P, a);')
    w('        loss = backward(n, X, Y, P, a, g, WS + %d);' % gp)
    w('        float c1 = 1 - powf(0.9f, ++t), c2 = 1 - powf(0.999f, t);')
    w('        for (int k = 0; k < NP; k++) {  /* Adam */')
    w('            float d = WS[%d + k];' % gp)
    w('            M[k] = 0.9f * M[k] + 0.1f * d;')
    w('            V[k] = 0.999f * V[k] + 0.001f * d * d;')
    w('            P[k] -= lr * (M[k] / c1) / (sqrtf(V[k] / c2) + 1e-7f);')
    w('        }')
    w('    }')
    w('    return loss;')
    w('}')
    w('')
    w('void predict(int n, const float *X, const float *P, float *WS, float *out)')
    w('{')
    w('    float *a[%d], *g[%d];' % (len(order), len(order)))
    w('    views(WS, a, g);')
    w('    forward(n, X, P, a);')
    w('    for (int s = 0; s < n; s++) out[s] = 1 / (1 + expf(-%s[s]));' % lg)
    w('}')
    w('')
    w('/* every hidden activation, node after node: out[node offset * n + s * width + j] */')
    w('void acts(int n, const float *X, const float *P, float *WS, float *out)')
    w('{')
    w('    float *a[%d], *g[%d];' % (len(order), len(order)))
    w('    views(WS, a, g);')
    w('    forward(n, X, P, a);')
    for i, wd, ao in acts_layout:
        w('    for (int k = 0; k < n * %d; k++) out[%d * n + k] = a[%d][k];' % (wd, ao, idx[i]))
    w('}')
    return {'c': '\n'.join(L) + '\n', 'nparams': nparams, 'ws': off,
            'acts': [{'id': i, 'width': wd, 'offset': ao} for i, wd, ao in acts_layout],
            'widths': width, 'used': order}
