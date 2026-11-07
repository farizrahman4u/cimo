// The website's live demos (docs/demo.js, docs/nn.js), their Python run in Pyodide with this checkout's cimo, as the
// page runs it: every preset compiled and its kernels called the way the page's JavaScript calls them.
//   node site_demos.mjs        (PYODIDE_JS, CIMO_ONEFILE as for pyodide_run.mjs)
// Prints SITE DEMOS OK when all pass.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const docs = join(here, '..', '..', '..', 'docs');

// the demos' scripts without a page: they export what the tests need and stop before the DOM
const ctx = {};
vm.runInNewContext(readFileSync(join(docs, 'demo.js'), 'utf8') + '\n' + readFileSync(join(docs, 'nn.js'), 'utf8'), ctx);
const demo = ctx.cimoDemo, nn = ctx.cimoNN;
if (!demo || !nn) throw new Error('docs/demo.js and docs/nn.js must set globalThis.cimoDemo and cimoNN');

const { loadPyodide } = await import(process.env.PYODIDE_JS ? pathToFileURL(process.env.PYODIDE_JS).href : 'pyodide');
const py = await loadPyodide();
const pkg = join(here, '..', '..', 'cimo');
const copy = (dir) => {
  for (const name of readdirSync(dir)) {
    if (name === '__pycache__') continue;
    const p = join(dir, name);
    const dest = '/lib/cimo/' + relative(pkg, p).split('\\').join('/');
    if (statSync(p).isDirectory()) { py.FS.mkdirTree(dest); copy(p); }
    else py.FS.writeFile(dest, readFileSync(p));
  }
};
if (process.env.CIMO_ONEFILE) py.FS.writeFile('/lib/cimo.py', readFileSync(process.env.CIMO_ONEFILE));
else { py.FS.mkdirTree('/lib/cimo'); copy(pkg); }
py.FS.mkdirTree('/home/pyodide');
for (const f of ['codegen.py'])  // as nn.js fetches it (wat is cimo.wat)
  py.FS.writeFile('/home/pyodide/' + f, readFileSync(join(docs, 'nn', f)));
py.runPython("import sys; sys.path.insert(0, '/lib'); sys.path.insert(0, '/home/pyodide')");

const sources = {};
for (const p of demo.presets) sources[p] = demo.source(p);
py.globals.set('ATTRACTOR_PY', demo.PY);
py.globals.set('ATTRACTOR_SOURCES', py.toPy(sources));
py.globals.set('NN_PY', nn.PY);
py.globals.set('NN_GRAPHS', py.toPy(nn.presets));
try {
  await py.runPythonAsync(readFileSync(join(here, 'site_demos.py'), 'utf8'));
} catch (e) {
  console.log(String(e && e.message || e));  // the Python traceback, without Pyodide's source
  process.exit(1);
}
