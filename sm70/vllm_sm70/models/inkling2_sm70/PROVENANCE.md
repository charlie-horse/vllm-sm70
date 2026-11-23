# Provenance: inkling2_sm70

`configs.py` is a byte-identical copy of an upstream vLLM file. Edit nothing
in it; SM70 differences live in `sm70_configs.py`.

| | |
|---|---|
| Upstream repo | https://github.com/vllm-project/vllm |
| Upstream path | `vllm/models/inkling2/configs.py` |
| Upstream commit | `c32251b5d192f0f72f9ca8c5595d1d43f1dd37b8` |
| sha256 | `ed9a063b12c6cb0aeb35a74e74ea8656c6ca9220d18fccec2369b15014abb231` |
| License | Apache-2.0 (upstream `LICENSE`; SPDX header kept in the file) |

`sm70/tests/models/test_m12_inkling2_registration_cpu.py` checks the file
against this sha256.

The file imports only `transformers.configuration_utils.PreTrainedConfig`.

`processing.py` is an adapted port of `vllm/models/inkling2/processing.py` at
the same upstream commit; its module docstring lists the differences.

`inkling2_encoding.py`, `renderer.py` and `parsers.py` are ported from the
same upstream commit; they are not byte-identical copies.

| File | Upstream path | Ported |
|---|---|---|
| `inkling2_encoding.py` | `vllm/renderers/inkling_encoding.py` | Lines 23-108 (constants, tokenizer protocol), `_collect_tools`, `_append_message`, `_expect_role`, `_as_mapping`, `_tool_call_name_and_args` and lines 436-780 (the Inkling 2 renderer), unchanged apart from the dropped audio part types. The Inkling 1 renderer is left out. |
| `renderer.py` | `vllm/renderers/inkling.py` | The reasoning-effort names and resolution (lines 44-74), `_HfBackedTmlTokenizer` (lines 77-117) and the Inkling 2 branch of `InklingRenderer` (lines 152-163, 172-186), adapted to vLLM v0.33.0. |
| `parsers.py` | `vllm/parser/inkling.py` | The Inkling 2 parts of `InklingParser`, as a subclass of vLLM v0.33.0's class: `INKLING_CONTROL_TOKENS` (lines 90-101), the vocabulary check (lines 267-270), `adjust_request` (lines 272-334) and `_inkling2_is_reasoning_end` (lines 374-397). The tool adapter subclass follows `vllm/tool_parsers/inkling_tool_parser.py`. |
| `sm70/tests/models/test_m12_inkling2_renderer_cpu.py` | `tests/renderers/test_inkling.py` | The Inkling 2 vocabulary and tool fixtures (lines 599-646) and `TestInkling2Rendering` (lines 666-1004). |
| `sm70/tests/models/test_m12_inkling2_parsers_cpu.py` | `tests/parser/engine/test_inkling.py` | The tokenize and streaming helpers (lines 86-101, 165-173, 191-222) and `TestInkling2` (lines 1044-1209); the mock tokenizer follows `tests/parser/engine/conftest.py`. |

`ops/norm.py`, `ops/attn_res.py` and `ops/loop.py` are vendored Triton kernel
files of the same upstream commit. Each is its upstream file plus the patch
`ops/patches/<file>.patch`; the patch header holds the upstream path, the
commit and the hash of the unpatched file. Change a vendored file only
together with its patch: `sm70/tests/ops/test_m12_inkling2_triton_norms_cpu.py`
takes the patch back out and checks the hash. Kernels whose work is a GEMM
(`tl.dot`) are not vendored. `ops/fused.py` is not vendored: the message-gate
projections and the residual add of the decode route, written for SM70.

| File | Upstream path | Ported |
|---|---|---|
| `ops/norm.py` | `vllm/models/inkling2/nvidia/ops/norm.py` | `rsqrt`, `_rms_norm_kernel` and `rms_norm` (lines 1-71), with a `tl.rsqrt` form of `rsqrt` for Triton's interpreter. `_l2_clip` and `qr_clip_rel_tokens` (lines 74-185, a `tl.dot` kernel and its helper) are left out. |
| `ops/attn_res.py` | `vllm/models/inkling2/nvidia/ops/attn_res.py` | The whole file. Changed: `gdc_wait` behind the `USE_GDC` constexpr and `launch_pdl` from `_compat.use_gdc()`; the `stride_t` argument (stored blocks with any token stride); the `STACK_NORMED` constexpr (stored blocks already RMS-normalized); the `q_scale` argument (queries not pre-scaled); `out_dtype`; `exp` from `_compat`; no JIT specialization on `T`. |
| `ops/loop.py` | `vllm/models/inkling2/nvidia/ops/loop.py` | The whole file. Changed: `out_dtype` in place of the bfloat16 outputs, `eps` of `pair_rms_norm` as an argument, `tanh` from `_compat`, no JIT specialization on `T`. |

`sm70/tests/models/data/inkling2_mini/` holds `config.json`,
`hf_quant_config.json` and `model.safetensors.index.json` of
`thinkingmachines/Inkling-2-Mini` at revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache-2.0 per the model card),
unchanged.
