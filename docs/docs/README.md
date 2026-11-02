# cimo docs

Served at https://www.cimo.dev/docs/. No build step: `docs.js` fetches the Markdown and renders it
in the browser, so a pushed edit is live once GitHub Pages redeploys (about a minute). Pages are
fetched with `cache: 'no-cache'`, so readers get the new version on their next load. After
changing `docs.js` or `docs.css`, update the `?v=` on their tags in `index.html`.

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

The Prompt for Agents corner here peels back over the page to show the prompt (`../agent-prompt.txt`), as the
landing page's banner does. Both use `../agent.js` and `../agent.css`; after changing those, update their `?v=`
in both `index.html` files. `../llms.txt` indexes these raw pages for agents; add new pages there too
(`python/tests/test_site.py` checks it).
