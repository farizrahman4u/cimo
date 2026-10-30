import { loadPyodide } from "pyodide";
import { readFileSync } from "node:fs";

const py = await loadPyodide();
console.log("pyodide", py.version);
await py.runPythonAsync(readFileSync(new URL("./wasmtest.py", import.meta.url), "utf8"));
