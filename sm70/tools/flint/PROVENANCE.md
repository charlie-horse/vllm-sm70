# Provenance: sm70/tools/flint/

Copied from charlie-horse/vLLM at cf9492a01eeca0cac731c0fa62015e2c85aaf272 (`main`/`latest` of the fork when it was retired). Text unchanged.

| file here | fork path | fork last touch |
|---|---|---|
| `convert_gguf.py` | `vllm/models/flint/convert_gguf.py` | 90c783594 |
| `gguf_nvfp4.py` | `vllm/models/flint/gguf_nvfp4.py` (imported by the converter) | 90c783594 |
| `agent_e2e.py` | `tools/flint/agent_e2e.py` | 55aa87e21 |
| `needle.py` | `tools/flint/needle.py` | 55aa87e21 |
| `fp16_range_audit.py` | `tools/flint/fp16_range_audit.py` | 0f6264c6c |
| `sm70_validate.sh` | `tools/flint/sm70_validate.sh` | 8a7706357 |

## Converter (`convert_gguf.py`, `gguf_nvfp4.py`)

The only code that produces the Flint checkpoint (`/nix/store/0qxvf6z208m6bff62acgwfg6ajhbinpp-flint-nvfp4-modelopt`, `nix/checkpoint.nix`). The fork's recipe (`nix/flint.nix`, `mkCheckpoint`) ran it from converter pin f66728d9b95e44dd6526b0f2033832477e6a79de; both files are identical at f66728d and cf9492a (`git diff` empty).

Imports:
- stdlib, `torch`, `numpy`, `regex`, `huggingface_hub`, `gguf` (lazily; needs gguf >= 0.19.0 for `GGMLQuantizationType.NVFP4`; the fork's env pinned 0.19.0);
- from vLLM, in `gguf_nvfp4.py` only: `vllm.model_executor.model_loader.weight_utils.download_weights_from_hf` and `vllm.transformers_utils.repo_utils.any_pattern_in_repo_files`. Both exist with the same signatures in upstream v0.33.0 (this repo's BASE). They are used only when `SRC` is a Hugging Face repo id, not for a local `.gguf`;
- nothing from the fork's `vllm.models.flint` other than these two files (the relative import `.gguf_nvfp4`).

Runnability against this repo's env: **not as is.** The `.#vllm` Python env has torch, numpy, regex, huggingface_hub, safetensors and vLLM 0.33, but not `gguf` (absent from `nix/pyenv/uv.lock`). With `gguf==0.19.0` added it should run; the module path changes from `vllm.models.flint.convert_gguf` to:

```sh
PYTHONPATH=sm70/tools python -m flint.convert_gguf SRC OUT --hf-config <dir> --device cpu
```

Checked 2026-11-16 on CPU only (no real GGUF): the fork's `tests/models/flint/test_convert_gguf.py` and `test_gguf_nvfp4.py`, with imports rewritten to `flint.*`, in a scratch venv (Python 3.13, torch 2.15 CPU, gguf 0.19.0, the two vLLM helpers stubbed): 17 passed, 1 skipped, 1 deselected (`test_quant_config_parses_through_vllm_loader`, needs the full vLLM). Not checked: a full conversion reproducing `checkpointHash` (needs the ~23 GB GGUF, ~25 GB RAM).

Inputs the fork's recipe pinned (from fork `nix/flint.nix` at cf9492a):
- GGUF: HF `RecursiveSI/flint-nvfp4-gguf` rev `6ec43e19b565fc8d560e2d04985d7724f20673bf`, file `flint-nvfp4.gguf`, sha256 `97cc8009ff9b132d5cf3c488e6767962bb5a5dc158a90ed661bbbddf876b8bde`.
- `--hf-config`: a directory of these files from HF `RecursiveSI/flint` rev `2a247be0c0aec895ccb0ac7ca92229ec6df37288` (sha256): `config.json` a57c33d11f725099105b657043c113149763066d204fd472d0b2ee399abc2ba7, `generation_config.json` a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3, `tokenizer.json` a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68, `tokenizer_config.json` 159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf, `chat_template.jinja` 7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466, `LICENSE` 43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1.
- Environment: `HOME=$TMPDIR HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1`; after converting, `rm -f $out/conversion_report.json` (run metadata, not reproducible).

## Other tools

`agent_e2e.py`, `needle.py` (HTTP clients against a running server; `openai` for agent_e2e), `fp16_range_audit.py` (drives vLLM's `LLM` with a worker extension it loads by the fork-relative name `tools.flint.fp16_range_audit`, so it does not run from this path without editing `EXTENSION_QUALNAME`/`REPO_ROOT`) and `sm70_validate.sh` (builds and serves the fork tree and calls the others by fork paths). Kept as reference; not adapted.
