# Provenance: qwen

`dflash_sm70.py` is ported from charlie-horse/vLLM (a vLLM fork) at commit
`cf9492a`, under `vllm/model_executor/models/`; it is not a byte-identical
copy. License: Apache-2.0, as vLLM (SPDX header kept).

| Source | Here |
|---|---|
| `dflash_sm70.py` (Triton kernels, wrappers, `DFlashSM70RMSNorm`, `DFlashSM70MLP`) | the kernels, wrappers and modules, plus torch references of each kernel |
| `dflash_sm70.py` layered RMSNorm | not ported |
| `qwen3_dflash2.py` `_use_sm70_bf16_emulation` | `emulation_enabled`; an FP16-runtime mismatch turns the emulation off instead of raising |
| `qwen3_dflash2.py` layer and model emulation | `apply_layer_emulation`, `apply_model_emulation`, called from subclasses of upstream's DFlash2 classes |
| `qwen3_dflash.py` o_proj input scale | `sm70_attention_forward` |
| `qwen3_dflash.py` `_normalize_context_states` | `_project_context_kv` |
| `DFLASH2_BF16_EMULATION` switch | `VLLM_SM70_DFLASH2_EMULATION` |
