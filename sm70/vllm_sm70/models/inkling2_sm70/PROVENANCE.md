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

`inkling2_encoding.py` and `renderer.py` are ported from the same upstream
commit; they are not byte-identical copies.

| File | Upstream path | Ported |
|---|---|---|
| `inkling2_encoding.py` | `vllm/renderers/inkling_encoding.py` | Lines 23-108 (constants, tokenizer protocol), `_collect_tools`, `_append_message`, `_expect_role`, `_as_mapping`, `_tool_call_name_and_args` and lines 436-780 (the Inkling 2 renderer), unchanged apart from the dropped audio part types. The Inkling 1 renderer is left out. |
| `renderer.py` | `vllm/renderers/inkling.py` | The reasoning-effort names and resolution (lines 44-74), `_HfBackedTmlTokenizer` (lines 77-117) and the Inkling 2 branch of `InklingRenderer` (lines 152-163, 172-186), adapted to vLLM v0.33.0. |
| `sm70/tests/models/test_m12_inkling2_renderer_cpu.py` | `tests/renderers/test_inkling.py` | The Inkling 2 vocabulary and tool fixtures (lines 599-646) and `TestInkling2Rendering` (lines 666-1004). |

`sm70/tests/models/data/inkling2_mini/` holds `config.json`,
`hf_quant_config.json` and `model.safetensors.index.json` of
`thinkingmachines/Inkling-2-Mini` at revision
`ed40e7da625f594fc9c4990f5f66be24873c22d7` (Apache-2.0 per the model card),
unchanged.
