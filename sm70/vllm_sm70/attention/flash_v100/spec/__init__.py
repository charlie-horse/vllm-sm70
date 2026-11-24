# SPDX-License-Identifier: Apache-2.0
"""Flash-V100 speculative-decoding verify: smallq -> XQA.

Verify rows (query length 1 + num_speculative_tokens, e.g. 8 for DFlash2
with 7 drafts) are expanded token-wise into decode rows, so the D256 XQA
decode (GQA 6/8) or the scalar paged decode verifies them. Kill switch
VLLM_SM70_FLASH_V100_SPEC=0: verify rows take the causal paged prefill,
and CUDA graphs cover single-token decode only.
"""

from __future__ import annotations

import os

KILL_SWITCH = "VLLM_SM70_FLASH_V100_SPEC"


def spec_verify_enabled() -> bool:
    """Only "0" acts (turns the smallq -> XQA verify route off)."""
    return os.environ.get(KILL_SWITCH) != "0"
