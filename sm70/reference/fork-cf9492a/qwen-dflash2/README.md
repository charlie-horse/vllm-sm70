# Reference: Qwen DFlash2 verifier bundle and Flash-V100 Qwen XQA variants

Source: charlie-horse/vLLM, commit `cf9492a01`, at the source's relative
paths. Files are byte-identical copies, except the two `*.excerpt` files:
verbatim line ranges, each block headed by its source path and line numbers.
Nothing here is built, imported, tested or linted (`nix/sm70-ext.nix` and
`sm70/pyproject.toml` exclude `sm70/reference`).

## Contents

- `csrc/sm70_turbomind/ops/`: GDN verify, exact gated RMSNorm, Qwen3.8 GDN
  input batch, exact row all-reduce and QPN8 channel-FP8 (with `ba` split)
  kernels; op registration in `csrc/bindings.excerpt`.
- `vllm/`: the DFlash2 verifier config, speculator and sparse rejection
  sampler, GDN linear attention, Gemma RMSNorm, and their Python wrappers;
  switch definitions in `vllm/envs.py.excerpt`.
- `flash-attention-v100/kernel/flash_decode_paged.cu`: split-reduce XQA
  kernels at lines 2893-3361, GQA-group-6 XQA routing at lines 6619-7148.

## License

- Files with `SPDX-License-Identifier: Apache-2.0` (every `.py` and the
  `csrc/` `.cu` files): Apache-2.0, Copyright contributors to the vLLM
  project. The source's top-level `LICENSE` is copied unchanged as `LICENSE`.
- `flash-attention-v100/`: BSD 3-Clause, Copyright (c) 2025 D.Skryabin;
  `flash-attention-v100/LICENSE` is copied unchanged.
- `csrc/sm70_turbomind/ops/fp8_qpn8_sm70.cu`: the QPN8 layout derives from
  dnv2003/v100-skinny (MIT) and haohervchb/sglang-V100; the MIT notice is
  `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`.
- Code taken from here carries these notices and records its changes in a
  PROVENANCE.md.
