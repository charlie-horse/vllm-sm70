# vllm-sm70: upstream vLLM for Volta (sm_70)

A Nix flake that builds a pinned upstream vLLM release for Volta, with an
out-of-tree plugin (`sm70/`) and a series of upstream-file patches
(`patches/`), plus NixOS modules for serving. `AGENTS.md` is a symlink to
this file. How everything works: [docs/README.md](docs/README.md).

## Access

- Host access (SSH, sudo) follows the flake
  repo's [CLAUDE.md](https://github.com/charlie-horse/flake/blob/main/CLAUDE.md#access).
- SSH to the hosts and GPU runs only when the user has allowed it for the
  task.
- Never put IPs, tailnet names, passwords or tokens in code, logs, docs or a
  PR.

## Changes

- All work goes on a branch and a PR to `main`; nobody merges their own PR.
- Change format, kill switches and the patch workflow:
  [docs/patch-header.md](docs/patch-header.md).
- CI (`.github/workflows/ci.yml`) is the gate for merging: patch series,
  CPU unit tests, Nix eval. Before pushing, run what it runs for the parts
  you changed. Add no heuristic or pattern-matching checks.

## Documentation policy

1. **The code is the source of truth; docs don't restate it.** Don't describe
   what the code declares (pins, hashes, options, defaults, kill switches,
   model entries, versions): read it from the source. Docs hold only what the
   code can't say: procedures, and rules whose breach the code can't prevent.
   Docs cover only the code in this repository, plus the minimum hardware
   detail needed to understand or use it. Name hosts or environments only
   where a procedure strictly needs it.
2. **Docs say how things work now.** No history, dates of past events,
   rationale ("why"), measurements of past runs, plans, decisions, open
   questions, owners or status notes. Present tense. A constraint you must
   follow is written as a rule.
3. **Tasks live in the project, not in git.** No TODO/FIXME/XXX, follow-ups
   or "later" notes in code or docs.
4. **Where docs live.** Prose goes in `docs/`; [docs/README.md](docs/README.md)
   lists every doc and the topic it owns. Access lives only in this file. A
   directory may keep a `README.md` of at most 15 lines that points into
   `docs/`. A `PROVENANCE.md` next to vendored code records only its source,
   license and changes from that source.
5. **One home per topic.** Edit a topic in its doc and link to it elsewhere.
   A new topic gets a row in docs/README.md.
6. **Text in code follows these rules too.** Comments, docstrings, patch
   headers and `PROVENANCE.md` files obey rules 1-3 and 8. A comment block
   or docstring is at most 15 lines; longer prose goes to a doc, with a
   one-line pointer in the code. No docs embedded in code or patches.
7. **Docs change in the same PR as the code.** Moving or renaming a doc or
   heading fixes every link to it.
8. **Plain language, low complexity.** Write so a layperson new to GPUs and
   vLLM can follow: short sentences, plain words, each term of art defined
   on first use. Keep each doc as short as it can be while still usable.
