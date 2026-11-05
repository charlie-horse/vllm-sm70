# SPDX-License-Identifier: Apache-2.0
"""Flash-V100 attention backend (K06) over the K05 `_sm70_C::fv100_*` ops.

  backend.py   FlashAttnV100Backend (class flags, KV layout, block sizes)
  metadata.py  FlashAttnV100Metadata(+Builder): decode/prefill split, groups
  impl.py      FlashAttnV100Impl: forward, Triton fallback
  dispatch.py  decode/prefill routing onto vllm_sm70.ops.fv100

Importing this package does not import vLLM; the modules do.
"""

BACKEND_PATH = "vllm_sm70.attention.flash_v100.backend.FlashAttnV100Backend"
