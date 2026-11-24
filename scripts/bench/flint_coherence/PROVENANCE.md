# Provenance: scripts/bench/flint_coherence/

Source: charlie-horse/vLLM, branch `claude/flint-swebench`, commit
bd0123f04f516094cbf9f1a500404702b93b96c6, path `benchmarks/flint_coherence/`.

Changes from the source:
- Comments and docstrings shortened; no code changed.
- Not copied (generated, over 1 MB): `fixtures/teacher_forced.jsonl`
  (1,576,671 bytes, sha256
  `bd4ff617a098a011c0a0e42f8caaea7f4235336c31acdc982be256b967bfd3e8`).
  `gate_a.py` and `test_gate_a.py` read it; `build_fixtures.py` writes it
  into `--out`. Compare a regenerated file against the hash above.
