# Instructions for coding agents

- **Commits:** commit as the repository's configured git user only. Never add yourself (or any
  AI agent or tool) as an author or co-author: no `Co-Authored-By:` trailers, no
  "Generated with ..." lines, in commit messages or PR descriptions.
- **Spec:** `requirements.md` is the source of truth for what to build and in what order. Follow
  its "Engineering rules" and "Milestones and acceptance criteria".
- **Findings:** when you learn something that changes the plan or that the next person needs
  (a measurement, a platform quirk, a dead end), add it to `requirements.md` concisely. It is
  a spec, not a log; keep raw output and transcripts out of it.
- **Secrets:** never commit tokens, keys or credentials. The git remote URL contains a token;
  never print it or copy it into files.
- **Runtime dependencies:** the `cimo` package must stay pure Python and standard library only.
