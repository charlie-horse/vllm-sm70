# SPDX-License-Identifier: Apache-2.0
"""K15 CPU checks: fused q/k norm + RoPE + KV-write admission, kill switch
and the splitting-op hook.

Port of FORK tests/models/flint/test_qk_norm_rope_kv_admission.py (117).
Changes: the kill switch resolves through Sm70Config (the fork used
KernelConfig.sm70_flint_qk_rope_kv), the splitting op is installed by the
platform hook instead of CompilationConfig itself, an FP32 RoPE table is
admitted (upstream Flint builds it in FP32), and kv_cache_dtype "float16" is
admitted next to "auto".
"""

from __future__ import annotations

import pytest
import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

from vllm_sm70.flint import QK_ROPE_KV_SPLITTING_OP, install_splitting_op
from vllm_sm70.flint.qk_norm_rope_kv import SPLITTING_OP, fused_route_rejection


def _rope(**overrides):
    kwargs = dict(
        head_size=128,
        rotary_dim=128,
        max_position_embeddings=64,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float16,
    )
    kwargs.update(overrides)
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        return RotaryEmbedding(**kwargs)


def _args(**overrides):
    args = dict(
        enabled=True,
        is_sm70=True,
        dtype=torch.float16,
        head_dim=128,
        q_eps=1e-6,
        k_eps=1e-6,
        rotary_emb=None,
        kv_cache_dtype="auto",
        attn_type="decoder",
        calculate_kv_scales=False,
        kv_sharing=False,
        backend_writes_kv_in_forward=False,
    )
    args.update(overrides)
    return args


def test_admits_flint_shapes_with_and_without_rope():
    assert fused_route_rejection(**_args()) is None
    assert fused_route_rejection(**_args(rotary_emb=_rope())) is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_admits_fp16_and_fp32_rope_tables(dtype):
    rope = _rope(dtype=dtype)
    assert rope.cos_sin_cache.dtype == dtype
    assert fused_route_rejection(**_args(rotary_emb=rope)) is None


def test_admits_explicit_float16_kv_cache():
    assert fused_route_rejection(**_args(kv_cache_dtype="float16")) is None


@pytest.mark.parametrize(
    "overrides",
    [
        dict(enabled=False),
        dict(is_sm70=False),
        dict(dtype=torch.bfloat16),
        dict(head_dim=64),
        dict(head_dim=256),
        dict(k_eps=1e-5),
        dict(kv_cache_dtype="fp8_e4m3"),
        dict(kv_cache_dtype="bfloat16"),
        dict(attn_type="encoder_only"),
        dict(calculate_kv_scales=True),
        dict(kv_sharing=True),
        dict(backend_writes_kv_in_forward=True),
    ],
)
def test_rejects_other_configs(overrides):
    assert fused_route_rejection(**_args(**overrides)) is not None


@pytest.mark.parametrize(
    "rope_overrides",
    [
        dict(is_neox_style=False),
        dict(rotary_dim=64),
        dict(dtype=torch.bfloat16),
    ],
)
def test_rejects_other_rotary_embeddings(rope_overrides):
    rope = _rope(**rope_overrides)
    assert fused_route_rejection(**_args(rotary_emb=rope)) is not None


def test_rejects_rotary_subclasses():
    class Scaled(RotaryEmbedding):
        pass

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
        rope = Scaled(128, 128, 64, 10000.0, True, torch.float16)
    assert fused_route_rejection(**_args(rotary_emb=rope)) is not None


@pytest.mark.parametrize("value,expected", [(None, True), ("1", True), ("0", False)])
def test_kill_switch_resolves_into_sm70_config(monkeypatch, value, expected):
    from vllm_sm70.config import Sm70Config, apply_kill_switches, sm70_config

    if value is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_QK_ROPE_KV", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", value)
    assert sm70_config(None).flint_qk_rope_kv is expected
    # The environment can only disable: an explicit False stays False.
    off = apply_kill_switches(Sm70Config(flint_qk_rope_kv=False))
    assert off.flint_qk_rope_kv is False


def test_splitting_op_name():
    assert SPLITTING_OP == QK_ROPE_KV_SPLITTING_OP
    assert SPLITTING_OP.startswith("vllm::")


def _compilation_config(**kwargs):
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode

    kwargs.setdefault("mode", CompilationMode.VLLM_COMPILE)
    kwargs.setdefault("use_inductor_graph_partition", False)
    return CompilationConfig(**kwargs)


def test_op_is_a_default_splitting_op():
    config = _compilation_config()
    assert config.splitting_ops is None
    assert install_splitting_op(config, enabled=True)
    assert SPLITTING_OP in config.splitting_ops
    assert "vllm::unified_kv_cache_update" in config.splitting_ops
    # Upstream's own call after the platform hook keeps the list.
    config.set_splitting_ops_for_v1(all2all_backend="", data_parallel_size=1)
    assert SPLITTING_OP in config.splitting_ops
    assert config.splitting_ops.count(SPLITTING_OP) == 1
    assert config.splitting_ops_contain_attention()
    # Idempotent.
    assert install_splitting_op(config, enabled=True)
    assert config.splitting_ops.count(SPLITTING_OP) == 1


def test_default_list_matches_upstream_plus_ours():
    ours = _compilation_config()
    install_splitting_op(ours, enabled=True)
    upstream = _compilation_config()
    upstream.set_splitting_ops_for_v1(all2all_backend="", data_parallel_size=1)
    assert ours.splitting_ops == upstream.splitting_ops + [SPLITTING_OP]


def test_kill_switch_leaves_splitting_ops_alone():
    config = _compilation_config()
    assert not install_splitting_op(config, enabled=False)
    assert config.splitting_ops is None


def test_not_added_where_upstream_does_not_split_the_kv_update():
    from vllm.config.compilation import CompilationMode

    # Inductor graph partition: no Dynamo-level split of the KV update.
    config = _compilation_config(use_inductor_graph_partition=True)
    assert not install_splitting_op(config, enabled=True)
    assert config.splitting_ops is None
    # Not VLLM_COMPILE.
    config = _compilation_config(mode=CompilationMode.NONE)
    assert not install_splitting_op(config, enabled=True)
    # A user list without the KV update op is left as given.
    config = _compilation_config(splitting_ops=["vllm::unified_attention"])
    assert not install_splitting_op(config, enabled=True)
    assert config.splitting_ops == ["vllm::unified_attention"]


def test_user_list_with_kv_update_gets_ours():
    config = _compilation_config(
        splitting_ops=["vllm::unified_attention", "vllm::unified_kv_cache_update"]
    )
    assert install_splitting_op(config, enabled=True)
    assert config.splitting_ops[-1] == SPLITTING_OP
