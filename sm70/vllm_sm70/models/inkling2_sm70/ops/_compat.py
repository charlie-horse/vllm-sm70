# SPDX-License-Identifier: Apache-2.0
"""What the vendored Triton kernels (norm.py, attn_res.py, loop.py) take from
their surroundings, as SM70 needs it.

    triton, tl    vllm.triton_utils' modules (placeholders without a GPU
                  driver); Triton's own under TRITON_INTERPRET=1, where the
                  kernels run in its CPU interpreter
    INTERPRET     that interpreter is on
    AVAILABLE     the kernels can run
    MODEL_DTYPE   storage dtype of the kernel outputs (bfloat16 upstream)
    use_gdc()     the kernels' USE_GDC constexpr and their launch_pdl: the
                  gdc_wait / gdc_launch_dependents calls (PDL) assemble only
                  for compute capability 9.0 and up
    exp, tanh     libdevice on the GPU; tl.exp forms in the interpreter, which
                  runs neither libdevice nor inline assembly
"""

from __future__ import annotations

import functools
import os

import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"

if INTERPRET:
    import triton
    import triton.language as tl

    AVAILABLE = True
else:
    from vllm.triton_utils import HAS_TRITON as AVAILABLE
    from vllm.triton_utils import tl, tldevice, triton

MODEL_DTYPE = torch.float16

__all__ = [
    "AVAILABLE",
    "INTERPRET",
    "MODEL_DTYPE",
    "exp",
    "tanh",
    "tl",
    "triton",
    "use_gdc",
]


@functools.cache
def use_gdc() -> bool:
    if INTERPRET:
        return False
    try:
        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        return False
    return major >= 9


if INTERPRET:

    @triton.jit
    def exp(x):
        return tl.exp(x)

    @triton.jit
    def tanh(x):
        return 1.0 - 2.0 / (tl.exp(2.0 * x) + 1.0)

else:

    @triton.jit
    def exp(x):
        return tldevice.exp(x)

    @triton.jit
    def tanh(x):
        return tldevice.tanh(x)
