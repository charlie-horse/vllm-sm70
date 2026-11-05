# SPDX-License-Identifier: Apache-2.0
"""Sm70CudaPlatform: resolves lazily, subclasses CudaPlatform, U1 defaults."""

from types import SimpleNamespace

import pytest
from conftest import requires_vllm

pytestmark = requires_vllm


@pytest.fixture
def platform_cls(stub_stable_libtorch):
    from vllm_sm70 import platform

    return platform.Sm70CudaPlatform


def test_qualname_resolves(platform_cls):
    from vllm.utils.import_utils import resolve_obj_by_qualname

    from vllm_sm70.platform import PLATFORM_QUALNAME

    assert resolve_obj_by_qualname(PLATFORM_QUALNAME) is platform_cls


def test_is_cuda_platform(platform_cls):
    from vllm.platforms.cuda import CudaPlatform

    assert issubclass(platform_cls, CudaPlatform)
    assert platform_cls().is_cuda()


def test_u1_hooks_default_to_upstream(platform_cls, monkeypatch):
    from vllm_sm70.ops import dflash2

    # Without the K18 ops (CPU-only build) the GEMM hook keeps upstream.
    monkeypatch.setattr(dflash2, "gemm_available", lambda: False)
    assert platform_cls.get_unquantized_gemm("auto") is None
    assert platform_cls.kernel_warmup(object()) is None


def test_u1_gemm_hook_k18(platform_cls, monkeypatch):
    from vllm_sm70.ops import dflash2

    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    monkeypatch.delenv("VLLM_SM70_DFLASH2_FP16", raising=False)
    assert platform_cls.get_unquantized_gemm("auto") is dflash2.dflash2_unquantized_gemm
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FP16", "0")
    assert platform_cls.get_unquantized_gemm("auto") is None


def test_check_and_update_config_installs_pass(platform_cls, monkeypatch):
    from vllm.config import CompilationConfig
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70.passes import PASS_KEY, Sm70CompositePass

    monkeypatch.setattr(
        CudaPlatform, "check_and_update_config", classmethod(lambda cls, cfg: None)
    )
    cfg = SimpleNamespace(compilation_config=CompilationConfig())
    platform_cls.check_and_update_config(cfg)
    platform_cls.check_and_update_config(cfg)
    assert isinstance(
        cfg.compilation_config.inductor_compile_config[PASS_KEY], Sm70CompositePass
    )


@pytest.mark.parametrize("kill", [None, "0"])
def test_check_and_update_config_installs_flint_splitting_op(
    platform_cls, monkeypatch, kill
):
    """K15: the fused Flint q/k-norm+RoPE+KV op is split out next to
    unified_kv_cache_update unless VLLM_SM70_FLINT_QK_ROPE_KV=0."""
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70.config import clear_cache
    from vllm_sm70.flint import QK_ROPE_KV_SPLITTING_OP

    if kill is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_QK_ROPE_KV", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_QK_ROPE_KV", kill)
    clear_cache()
    monkeypatch.setattr(
        CudaPlatform, "check_and_update_config", classmethod(lambda cls, cfg: None)
    )
    cc = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE, use_inductor_graph_partition=False
    )
    cfg = SimpleNamespace(compilation_config=cc)
    platform_cls.check_and_update_config(cfg)
    clear_cache()
    if kill is None:
        assert QK_ROPE_KV_SPLITTING_OP in cc.splitting_ops
    else:
        assert cc.splitting_ops is None


def test_get_attn_backend_cls_falls_through(platform_cls, monkeypatch):
    from vllm.platforms.cuda import CudaPlatform

    seen = []
    monkeypatch.setattr(
        CudaPlatform,
        "get_attn_backend_cls",
        classmethod(lambda cls, *a: seen.append(a) or "upstream.Backend"),
    )
    assert platform_cls.get_attn_backend_cls(None, "cfg", 8) == "upstream.Backend"
    assert seen == [(None, "cfg", 8)]
