// Run a Python script inside Pyodide (Node) with the cimo package from this checkout:
//   node pyodide_run.mjs script.py
// Needs the `pyodide` npm package: PYODIDE_JS=<path to node_modules/pyodide/pyodide.mjs>
// (scratch/spikes/pyodide has one), else it must resolve as 'pyodide'.
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const { loadPyodide } = await import(process.env.PYODIDE_JS ? pathToFileURL(process.env.PYODIDE_JS).href : 'pyodide');

const here = dirname(fileURLToPath(import.meta.url));
const pkg = join(here, '..', '..', 'cimo');
const py = await loadPyodide();
const copy = (dir) => {
  for (const name of readdirSync(dir)) {
    if (name === '__pycache__') continue;
    const p = join(dir, name);
    const dest = '/lib/cimo/' + relative(pkg, p).split('\\').join('/');
    if (statSync(p).isDirectory()) { py.FS.mkdirTree(dest); copy(p); }
    else py.FS.writeFile(dest, readFileSync(p));
  }
};
py.FS.mkdirTree('/lib/cimo');
copy(pkg);
py.runPython("import sys; sys.path.insert(0, '/lib')");
if (process.env.PYODIDE_PACKAGES) await py.loadPackage(process.env.PYODIDE_PACKAGES.split(','));
await py.runPythonAsync(readFileSync(process.argv[2], 'utf8'));
