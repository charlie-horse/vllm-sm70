# SPDX-License-Identifier: Apache-2.0
"""Construct upstream's `Attention` layer with Flash-V100 selected (CPU only).

Regression for the c4130 Flint FV100 serve crash: upstream v0.33.0's
`Attention.__init__` does `AttentionBackendEnum[attn_backend.get_name()]`
(UP model_executor/layers/attention/attention.py:433) after building the
impl, so a backend whose get_name() is not an enum member passes every
backend-level test and still kills every worker at model init.

This builds real `Attention` layers inside `set_current_vllm_config` with
`attn_backend=None`, so the backend comes from upstream's real selector
(`get_attn_backend` -> `current_platform.get_attn_backend_cls`) running
Sm70CudaPlatform's routing. Mocked, and only these: the platform's device
queries (capability 7.0, device name; no NVML/CUDA on CPU), and
`vllm_sm70.attention._ops_available` (no compiled _sm70_C). The enum
lookup, the impl constructor (FlashAttnV100Impl -> TritonAttentionImpl with
the SM70 KV-dtype checks) and the static_forward_context registration are
upstream's and ours, unpatched.

Shapes: Flint per rank (16 q / 2 kv heads, head 128, fp16, fp32 learned
sinks, sliding window 513 and full attention) and a Qwen-like layer (head
256, no sinks).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM source not importable")


@pytest.fixture(autouse=True)
def _stub_stable_libtorch(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )


@pytest.fixture
def sm70_platform(monkeypatch):
    """Sm70CudaPlatform as `current_platform`, reporting one V100 (cc 7.0)."""
    import vllm.platforms
    from vllm.model_executor.layers.attention import attention as attn_mod
    from vllm.platforms.interface import DeviceCapability
    from vllm.v1.attention import selector
    from vllm.v1.attention.backends import triton_attn
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70 import attention
    from vllm_sm70.platform import Sm70CudaPlatform

    monkeypatch.setattr(
        Sm70CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(7, 0)),
    )
    monkeypatch.setattr(
        Sm70CudaPlatform,
        "get_device_name",
        classmethod(lambda cls, device_id=0: "Tesla V100-PCIE-32GB"),
    )
    platform = Sm70CudaPlatform()
    # selector reads vllm.platforms.current_platform lazily; attention.py and
    # triton_attn.py bound it at import.
    monkeypatch.setattr(vllm.platforms, "_current_platform", platform)
    monkeypatch.setattr(attn_mod, "current_platform", platform)
    monkeypatch.setattr(triton_attn, "current_platform", platform)
    monkeypatch.setattr(attention, "_ops_available", lambda: True)
    monkeypatch.delenv("VLLM_SM70_FLASH_V100", raising=False)

    monkeypatch.setattr(attention, "_registered", False)
    attention.register()
    selector._cached_get_attn_backend.cache_clear()
    yield platform
    selector._cached_get_attn_backend.cache_clear()
    AttentionBackendEnum.CUSTOM.clear_override()


def _build_layer(*, num_heads, num_kv_heads, head_size, window, sinks, prefix):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention

    vllm_config = VllmConfig()
    old_dtype = torch.get_default_dtype()
    # Models are built under the activation dtype (UP attention.py: dtype =
    # torch.get_default_dtype()); the c4130 stack runs --dtype half.
    torch.set_default_dtype(torch.float16)
    try:
        with set_current_vllm_config(vllm_config):
            layer = Attention(
                num_heads,
                head_size,
                head_size**-0.5,
                num_kv_heads=num_kv_heads,
                per_layer_sliding_window=window,
                prefix=prefix,
                sinks=sinks,
            )
    finally:
        torch.set_default_dtype(old_dtype)
    return layer, vllm_config


FLINT = {"num_heads": 16, "num_kv_heads": 2, "head_size": 128}


@pytest.mark.parametrize(
    "shape,window,with_sinks",
    [
        pytest.param(FLINT, 513, True, id="flint-sliding-sinks"),
        pytest.param(FLINT, None, True, id="flint-full-sinks"),
        pytest.param(
            {"num_heads": 16, "num_kv_heads": 2, "head_size": 256},
            None,
            False,
            id="qwen-d256",
        ),
    ],
)
def test_attention_layer_resolves_to_flash_v100(
    sm70_platform, shape, window, with_sinks
):
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70.attention.flash_v100.backend import FlashAttnV100Backend
    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl

    sinks = (
        torch.linspace(-2.0, 6.0, shape["num_heads"], dtype=torch.float32)
        if with_sinks
        else None
    )
    prefix = "model.layers.0.self_attn.attn"
    layer, vllm_config = _build_layer(
        **shape, window=window, sinks=sinks, prefix=prefix
    )

    # Chosen by the real selector + Sm70CudaPlatform routing, not passed in.
    assert layer.attn_backend is FlashAttnV100Backend
    # The line that crashed every serve (UP attention.py:433).
    assert layer.backend is AttentionBackendEnum.CUSTOM
    assert type(layer.impl) is FlashAttnV100Impl
    assert layer.impl._fv100_reason is None  # FV100 kernels, not Triton fallback
    assert layer.dtype == torch.float16
    assert layer.has_sink is with_sinks
    assert layer.impl.sinks is sinks
    expected_window = (-1, -1) if window is None else (window - 1, 0)
    assert tuple(layer.impl.sliding_window) == expected_window
    assert vllm_config.compilation_config.static_forward_context[prefix] is layer


def test_explicit_custom_backend_also_constructs(sm70_platform):
    """A layer handed the CUSTOM class directly (attn_backend=) takes the
    same enum lookup; covers models that pin their backend."""
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    with set_current_vllm_config(VllmConfig()):
        layer = Attention(
            16,
            128,
            128**-0.5,
            num_kv_heads=2,
            prefix="model.layers.1.self_attn.attn",
            attn_backend=AttentionBackendEnum.CUSTOM.get_class(),
            sinks=torch.zeros(16, dtype=torch.float32),
        )
    assert layer.backend is AttentionBackendEnum.CUSTOM
