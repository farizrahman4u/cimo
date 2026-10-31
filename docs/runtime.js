// Pyodide with cimo imported, loaded once and shared by the demos on the page.
//   cimoRuntime.load(status) -> Promise<pyodide>; status(message) is called while loading.
(function () {
  'use strict';
  const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';
  let promise = null;
  const listeners = [];
  const say = m => listeners.forEach(f => f(m));

  function script(src) {
    return new Promise((ok, fail) => {
      const s = document.createElement('script');
      s.src = src; s.onload = ok; s.onerror = () => fail(new Error('could not load ' + src));
      document.head.appendChild(s);
    });
  }

  async function boot() {
    say('Loading Pyodide (CPython in WebAssembly)…');
    if (!window.loadPyodide) await script(PYODIDE + 'pyodide.js');
    const py = await loadPyodide({ indexURL: PYODIDE });
    say('Loading cimo (one 1.1 MB Python file)…');
    const text = await (await fetch('cimo.py')).text();
    py.FS.writeFile('/home/pyodide/cimo.py', text);
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide')");
    say('Importing cimo…');
    await py.runPythonAsync('import cimo');
    return py;
  }

  window.cimoRuntime = {
    script,
    load(status) {
      if (status) listeners.push(status);
      if (!promise) {
        promise = boot().catch(e => { promise = null; throw e; });
        dispatchEvent(new Event('cimo-runtime'));  // other demos may start with it
      }
      return promise;
    },
    started: () => promise !== null,
  };
})();
