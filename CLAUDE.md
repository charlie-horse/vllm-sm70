# vllm-sm70: upstream vLLM for V100 (sm_70)

A Nix flake that builds a pinned upstream vLLM release for Volta, with an
out-of-tree plugin (`sm70/`) and a series of upstream-file patches
(`patches/`), plus NixOS modules for serving. `AGENTS.md` is a symlink to
this file. How everything works: [docs/README.md](docs/README.md).

## Access

- Host access (Tailscale in a cloud session, SSH, sudo) follows the flake
  repo's [CLAUDE.md](https://github.com/charlie-horse/flake/blob/main/CLAUDE.md#access).
- SSH to the hosts and GPU runs on c4130 only when the user has allowed it
  for the task.
- Never put IPs, tailnet names, passwords or tokens in code, logs, docs or a
  PR. Host names (c4130, r440) are fine.

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
2. **Docs say how things work now.** No history, dates of past events,
   rationale ("why"), measurements of past runs, plans, decisions, open
   questions, owners or status notes. Present tense, terse. A constraint you
   must follow is written as a rule.
3. **Tasks live in the project, not in git.** No TODO/FIXME/XXX, follow-ups
   or "later" notes in code or docs.
4. **Where docs live.** Prose goes in `docs/`; [docs/README.md](docs/README.md)
   lists every doc and the topic it owns. Access lives only in this file. A
   directory may keep a `README.md` of at most 15 lines that points into
   `docs/`. A `PROVENANCE.md` next to vendored code records only its source,
   license and changes from that source.
5. **One home per topic.** Edit a topic in its doc and link to it elsewhere.
   A new topic gets a row in docs/README.md.
6. **Code comments stay short and local.** At most 15 contiguous comment
   lines; longer prose goes to the doc, with a one-line pointer in the code.
7. **Docs change in the same PR as the code.** Moving or renaming a doc or
   heading fixes every link to it.
