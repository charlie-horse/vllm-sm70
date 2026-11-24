# Provenance: scripts/bench/flint_swebench/

Source: charlie-horse/vLLM, branch `claude/flint-swebench`, commit
bd0123f04f516094cbf9f1a500404702b93b96c6, path `benchmarks/flint_swebench/`
(every file the branch added there, same relative paths and file modes).

Changes from the source:
- `r440/env.sh`: the `FLINT_API_BASE` host is the placeholder `<server-ip>`;
  set it before use.
- Comments and docstrings shortened; no code changed.

Absolute paths in `r440/conf/storage.conf` and the `$HOME/flint-swebench`
paths in the scripts are kept as in the source; adjust them before use.
