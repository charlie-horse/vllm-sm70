# SPDX-License-Identifier: Apache-2.0
"""Flint on SM70 -> Flash-V100 with sinks: engine-wide backend choice (CPU).

Port of the backend-choice cases of fork cf9492a
tests/v1/attention/test_sm70_flint_backend_route.py (:46-120 capability
flag, :166-283 FlintModelConfig choice) onto
vllm_sm70.attention.flint_route, plus:

* a Flint-like vllm_config (real AttentionConfig/CacheConfig, Flint's
  32q/4kv D128 text config, sliding window 513, fp32 sinks);
* explicit CUSTOM resolves through Sm70CudaPlatform.get_attn_backend_cls
  (upstream's selected-backend branch) to our FlashAttnV100Backend, and the
  Flint TP2 layer (16q/2kv per rank) is admitted by FlashAttnV100Impl with
  the K14 fused decode engaged and decode routed to the fused op.

Dropped from the fork file: sm70_deployment / platform capability cases
(M5 checks cc 7.0 itself), the fork impl's executor/admission internals
(our K06 impl and its CPU tests replace them).
"""

from __future__ import annotations

import importlib
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM source not importable")

ENVS = (
    "VLLM_SM70_FLASH_V100",
    "VLLM_SM70_FLINT_FLASH_V100",
    "VLLM_SM70_FLINT_ATTN_FUSED_DECODE",
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def route(monkeypatch):
    """flint_route with CUSTOM registered to us and the sink ops 'built'."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    from vllm_sm70 import attention
    from vllm_sm70.attention import flint_route
    from vllm_sm70.ops import fv100

    monkeypatch.setattr(attention, "_registered", False)
    attention.register()
    built = set(fv100.OPS) | {fv100.FUSED_DECODE_OP}
    monkeypatch.setattr(fv100, "has_op", lambda name: name in built)
    logged: list[str] = []
    monkeypatch.setattr(
        flint_route.logger, "info", lambda msg, *a: logged.append(msg % a)
    )
    flint_route.logged = logged
    flint_route.built = built
    yield flint_route
    AttentionBackendEnum.CUSTOM.clear_override()


def _flint_config(
    *,
    backend=None,
    dtype=torch.float16,
    head_dim=128,
    cache_dtype="auto",
    block_size=None,
    speculative_config=None,
):
    """A Flint-like vllm_config: 32q/4kv, D128, sliding window 513."""
    from vllm.config import AttentionConfig, CacheConfig

    cache = CacheConfig(cache_dtype=cache_dtype)
    if block_size is not None:
        cache.block_size = block_size
    text = SimpleNamespace(
        architectures=["FlintSm70ForCausalLM"],
        head_dim=head_dim,
        num_attention_heads=32,
        num_key_value_heads=4,
        sliding_window=513,
    )
    return SimpleNamespace(
        attention_config=AttentionConfig(backend=backend),
        model_config=SimpleNamespace(
            dtype=dtype,
            hf_text_config=text,
            get_head_size=lambda: head_dim,
        ),
        cache_config=cache,
        speculative_config=speculative_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )


def _enum():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    return AttentionBackendEnum


# -------------------------------------------------------- capability (fork)


@pytest.mark.parametrize(
    ("ops", "expected"),
    [
        (("fv100_decode_paged_sinks", "fv100_prefill_paged_sinks"), True),
        (("fv100_prefill_paged_sinks",), False),  # prefill sinks only
        (("fv100_decode_paged_sinks",), False),  # decode sinks only
        ((), False),  # no extension
    ],
)
def test_supports_sinks_needs_both_sink_ops(monkeypatch, ops, expected):
    """fork :46-120 (SUPPORTS_SINKS computed from the two sink entries)."""
    from vllm_sm70.attention import flint_route
    from vllm_sm70.ops import fv100

    monkeypatch.setattr(fv100, "has_op", lambda name: name in ops)
    assert flint_route.flash_v100_supports_sinks() is expected


def test_head_sizes_and_switch_registered():
    from vllm_sm70 import envs
    from vllm_sm70.attention import flint_route

    assert flint_route.FLASH_V100_SINK_HEAD_SIZES == (128,)
    assert "VLLM_SM70_FLINT_FLASH_V100" in envs._SECTIONS["attention"]


# ------------------------------------------------------ backend choice (fork)


def test_flint_selects_flash_v100_by_default(route):
    cfg = _flint_config()
    assert route.flint_sink_route_rejection(cfg) is None
    route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().CUSTOM
    (line,) = route.logged
    assert "Flint on SM70: using Flash-V100 (sinks; K14 fused decode on)" in line


@pytest.mark.parametrize(
    ("fused_env", "drop_op", "state"),
    [("0", False, "VLLM_SM70_FLINT_ATTN_FUSED_DECODE=0"), (None, True, "not built")],
)
def test_info_line_reports_k14_state(route, monkeypatch, fused_env, drop_op, state):
    if fused_env is not None:
        monkeypatch.setenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE", fused_env)
    if drop_op:
        route.built.discard("fv100_decode_paged_sinks_fused")
    cfg = _flint_config()
    route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().CUSTOM
    assert f"K14 fused decode off: {state}" in route.logged[-1]


@pytest.mark.parametrize(
    ("env", "overrides", "reason"),
    [
        ("VLLM_SM70_FLASH_V100", {}, "VLLM_SM70_FLASH_V100=0"),
        ("VLLM_SM70_FLINT_FLASH_V100", {}, "VLLM_SM70_FLINT_FLASH_V100=0"),
        (None, {"dtype": torch.float32}, "float16"),
        (None, {"dtype": torch.bfloat16}, "float16"),
        (None, {"head_dim": 256}, "head_dim=256"),
        (None, {"head_dim": 64}, "head_dim=64"),
        (None, {"cache_dtype": "fp8_e5m2"}, "FP16 KV cache"),
        (None, {"block_size": 8}, "multiple of 16"),
        (None, {"speculative_config": object()}, "speculative"),
    ],
    ids=[
        "k06-kill-switch",
        "flint-kill-switch",
        "fp32",
        "bf16",
        "d256",
        "d64",
        "fp8-e5m2-kv",
        "block8",
        "spec",
    ],
)
def test_each_rejection_selects_triton(route, monkeypatch, env, overrides, reason):
    """fork :244-260 plus our kill switches and block size."""
    if env is not None:
        monkeypatch.setenv(env, "0")
    cfg = _flint_config(**overrides)
    assert reason in route.flint_sink_route_rejection(cfg)
    if "cache_dtype" in overrides:
        # M5's rule: TRITON_ATTN on SM70 needs an FP16 KV cache too.
        with pytest.raises(ValueError, match="FP16 KV cache"):
            route.select_flint_attention_backend(cfg)
    else:
        route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().TRITON_ATTN
    (line,) = route.logged
    assert "TRITON_ATTN" in line and reason in line


@pytest.mark.parametrize("value", ["", "1", "off"])
def test_only_zero_acts_on_the_flint_switch(route, monkeypatch, value):
    monkeypatch.setenv("VLLM_SM70_FLINT_FLASH_V100", value)
    cfg = _flint_config()
    route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().CUSTOM


@pytest.mark.parametrize(
    "missing", ["fv100_decode_paged_sinks", "fv100_prefill_paged_sinks"]
)
def test_missing_sink_op_selects_triton(route, missing):
    """fork :236-241 (missing extension / SUPPORTS_SINKS false)."""
    route.built.discard(missing)
    cfg = _flint_config()
    route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().TRITON_ATTN
    assert "sink ops" in route.logged[-1]


def test_custom_overridden_elsewhere_selects_triton(route):
    from vllm.v1.attention.backends.registry import register_backend

    register_backend(
        _enum().CUSTOM,
        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend",
    )
    cfg = _flint_config()
    assert "CUSTOM" in route.flint_sink_route_rejection(cfg)


@pytest.mark.parametrize("name", ["CUSTOM", "TRITON_ATTN", "FLEX_ATTENTION"])
def test_explicit_backend_is_kept(route, monkeypatch, name):
    """fork :271-277: an explicit choice is never changed, whatever the
    rejection says."""
    backend = getattr(_enum(), name)
    for env in (None, "0"):
        if env is not None:
            monkeypatch.setenv("VLLM_SM70_FLINT_FLASH_V100", env)
        cfg = _flint_config(backend=backend)
        route.select_flint_attention_backend(cfg)
        assert cfg.attention_config.backend is backend
    assert route.logged == []


@pytest.mark.parametrize("cache_dtype", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("explicit", [False, True])
def test_triton_with_fp8_kv_cache_raises(route, cache_dtype, explicit):
    """M5's rule (the fork only warned, :406-415)."""
    backend = _enum().TRITON_ATTN if explicit else None
    cfg = _flint_config(cache_dtype=cache_dtype, backend=backend)
    with pytest.raises(ValueError, match="FP16 KV cache"):
        route.select_flint_attention_backend(cfg)
    assert cfg.attention_config.backend == _enum().TRITON_ATTN


# ------------------------------------- CUSTOM resolves; Flint layers admitted


def test_explicit_custom_resolves_to_flash_v100(route, monkeypatch):
    """Sm70CudaPlatform.get_attn_backend_cls(CUSTOM, Flint layer) takes
    upstream's selected-backend branch: CUSTOM.get_class() is our backend
    and its validate_configuration admits fp16 D128 sinks + window."""
    from vllm.v1.attention.selector import AttentionSelectorConfig

    from vllm_sm70.attention.flash_v100 import BACKEND_PATH
    from vllm_sm70.platform import Sm70CudaPlatform

    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(
        Sm70CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, i=0: DeviceCapability(7, 0)),
    )
    for sliding in (True, False):  # sliding_attention and full layers
        layer = AttentionSelectorConfig(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="auto",
            block_size=16,
            has_sink=True,
            has_sliding_window=sliding,
        )
        path = Sm70CudaPlatform.get_attn_backend_cls(_enum().CUSTOM, layer, 16)
        assert path == BACKEND_PATH


@pytest.mark.parametrize("heads", [(16, 2), (32, 4)], ids=["tp2-16q2kv", "tp1-32q4kv"])
@pytest.mark.parametrize("window", [513, None], ids=["sliding", "full"])
def test_flint_layer_is_admitted_with_fused_decode(route, monkeypatch, heads, window):
    """The per-layer admission an engine routed here relies on: no Triton
    fallback reason, K14 fused decode marked, decode routed to the fused op."""
    from vllm_sm70.attention.flash_v100 import dispatch
    from vllm_sm70.attention.flash_v100.impl import FlashAttnV100Impl
    from vllm_sm70.ops import fv100

    num_heads, num_kv_heads = heads
    sinks = torch.linspace(-2, 6, num_heads, dtype=torch.float32)
    impl = FlashAttnV100Impl(
        num_heads, 128, 128**-0.5, num_kv_heads, None, window, "auto", sinks=sinks
    )
    assert impl._fv100_reason is None
    assert impl._fv100_fused_sinks is True
    assert tuple(impl.sliding_window) == ((512, 0) if window else (-1, -1))

    calls: list[str] = []
    monkeypatch.setattr(fv100, "_op", lambda name: lambda *a: calls.append(name))
    monkeypatch.setattr(fv100, "_cuda_graph_capture_active", lambda: False)
    fv100.clear_workspaces()
    q = torch.zeros((2, num_heads, 128), dtype=torch.float16)
    k = torch.zeros((4, 16, num_kv_heads, 128), dtype=torch.float16)
    md = SimpleNamespace(
        decode_rows=2,
        decode_seq_lens=torch.tensor([20, 7], dtype=torch.int32),
        decode_block_table=torch.zeros((2, 2), dtype=torch.int32),
        cudagraph_capture=False,
        max_model_len=4096,
        max_seq_len=20,
    )
    dispatch.run_decode(
        q,
        k,
        k,
        torch.empty_like(q),
        md,
        scale=impl.scale,
        window=tuple(impl.sliding_window),
        sinks=impl.sinks,
    )
    assert calls == ["fv100_decode_paged_sinks_fused"]
    fv100.clear_workspaces()
