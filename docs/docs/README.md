# cimo docs

Served at https://www.cimo.dev/docs/. No build step: `docs.js` renders the Markdown in the browser.

- `content/<page>.md`: one page each, plain Markdown (GitHub flavored: tables, fenced code).
- `nav.json`: the order and grouping. Parts with `"walk": true` are the walkthrough (numbered,
  with progress and Next links); the others are reference.

Writing pages:

- Link to another page as `[text](arrays.md)` or `[text](arrays.md#a-heading)`; headings get
  ids from their text (`## Match the element type` -> `#match-the-element-type`).
- Code fences: `c`, `python`, `console` (lines starting with `$ ` are commands), `text`.
- Callouts: a blockquote starting with `[!NOTE]`, `[!TIP]`, `[!WARNING]` or `[!PLANNED]`
  (for designed but unbuilt features).
- Check examples by running them; the docs describe the code as it is, and mark what isn't
  built yet as Planned.

Preview locally: `python3 -m http.server -d docs` and open http://localhost:8000/docs/.
