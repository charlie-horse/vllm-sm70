# Provenance: scripts/bench/flint_swebench/

Copied from charlie-horse/vLLM branch `claude/flint-swebench` at bd0123f04f516094cbf9f1a500404702b93b96c6 (2026-11-04), fork path `benchmarks/flint_swebench/` (every file the branch added there, same relative paths and file modes).

Changes on import:
- `r440/env.sh` line 3: the c4130 LAN and tailnet IP addresses are redacted to `<c4130-lan-ip>` / `<c4130-tailnet-ip>`. Set `FLINT_API_BASE` to the real endpoint before use.

Everything else is byte for byte. Paths inside the scripts (`$HOME/flint-swebench`, `/home/charlie/...` in `r440/conf/storage.conf`) are not adjusted.
