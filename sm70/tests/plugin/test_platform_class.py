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


def test_u1_hooks_default_to_upstream(platform_cls):
    assert platform_cls.get_unquantized_gemm("auto") is None
    assert platform_cls.kernel_warmup(object()) is None


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


@pytest.mark.parametrize("inductor", [True, False])
def test_fused_add_rms_norm_priority_is_vllm_c(platform_cls, inductor):
    """K17: on SM70 fused_add_rms_norm lowers to vllm_c (FP16 residual) under
    Inductor too, where upstream's CUDA default is ["native"] (FP32 residual).
    Other ops keep upstream's defaults."""
    from vllm.config.compilation import CompilationMode
    from vllm.platforms.cuda import CudaPlatform

    cc = SimpleNamespace(
        backend="inductor" if inductor else "eager",
        mode=CompilationMode.VLLM_COMPILE if inductor else CompilationMode.NONE,
    )
    cfg = SimpleNamespace(compilation_config=cc)
    ours = platform_cls.get_default_ir_op_priority(cfg)
    theirs = CudaPlatform.get_default_ir_op_priority(cfg)
    assert ours.fused_add_rms_norm == ["vllm_c", "native"]
    assert ours.rms_norm == theirs.rms_norm
    assert ours.gelu_and_mul_sparse == theirs.gelu_and_mul_sparse
    if inductor:
        assert theirs.fused_add_rms_norm == ["native"]


def test_fused_add_rms_norm_priority_merges_into_kernel_config(
    platform_cls, monkeypatch
):
    """Through upstream's KernelConfig.set_platform_defaults (config/kernel.py:
    451-478): vllm_c first unless the user set a priority."""
    import vllm.platforms
    from vllm.config.compilation import CompilationMode
    from vllm.config.kernel import IrOpPriorityConfig, KernelConfig

    monkeypatch.setattr(vllm.platforms, "_current_platform", platform_cls())
    cc = SimpleNamespace(backend="inductor", mode=CompilationMode.VLLM_COMPILE)
    cfg = SimpleNamespace(
        compilation_config=cc,
        model_config=None,
        parallel_config=SimpleNamespace(),
    )
    kc = KernelConfig()
    monkeypatch.setattr(
        "vllm.config.kernel.bind_passthrough_all2all_backend", lambda *a: None
    )
    kc.set_platform_defaults(cfg)
    assert kc.ir_op_priority.fused_add_rms_norm[0] == "vllm_c"
    user = KernelConfig(ir_op_priority=IrOpPriorityConfig(fused_add_rms_norm=["native"]))
    user.set_platform_defaults(cfg)
    assert user.ir_op_priority.fused_add_rms_norm == ["native", "vllm_c"]
