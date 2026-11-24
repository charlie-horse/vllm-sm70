# Provenance: sm70/tools/flint/

Source: charlie-horse/vLLM, commit cf9492a01eeca0cac731c0fa62015e2c85aaf272.
License: Apache-2.0 (SPDX headers in the files). Text unchanged.

| file here | source path | source last touch |
|---|---|---|
| `convert_gguf.py` | `vllm/models/flint/convert_gguf.py` | 90c783594 |
| `gguf_nvfp4.py` | `vllm/models/flint/gguf_nvfp4.py` (imported by the converter) | 90c783594 |
| `agent_e2e.py` | `tools/flint/agent_e2e.py` | 55aa87e21 |
| `needle.py` | `tools/flint/needle.py` | 55aa87e21 |
| `fp16_range_audit.py` | `tools/flint/fp16_range_audit.py` | 0f6264c6c |
| `sm70_validate.sh` | `tools/flint/sm70_validate.sh` | 8a7706357 |

`fp16_range_audit.py` and `sm70_validate.sh` use source-tree paths and do
not run from here unedited.

## Converter inputs

Dependencies: stdlib, `torch`, `numpy`, `regex`, `huggingface_hub`,
`gguf >= 0.19.0`; from vLLM only `download_weights_from_hf` and
`any_pattern_in_repo_files`, used when `SRC` is an HF repo id.

- `SRC`: HF `RecursiveSI/flint-nvfp4-gguf` rev `6ec43e19b565fc8d560e2d04985d7724f20673bf`, file `flint-nvfp4.gguf`, sha256 `97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
- `--hf-config`: a directory of these files from HF `RecursiveSI/flint` rev `2a247be0c0aec895ccb0ac7ca92229ec6df37288` (sha256): `config.json` a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7, `generation_config.json` a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3, `tokenizer.json` a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68, `tokenizer_config.json` 159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf, `chat_template.jinja` 7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466, `LICENSE` 43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1.
