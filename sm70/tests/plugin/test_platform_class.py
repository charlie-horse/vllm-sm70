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
    dflash2.reset_selection_cache()
    assert platform_cls.get_unquantized_gemm("auto") is None
    assert platform_cls.kernel_warmup(object()) is None


def test_u1_gemm_hook_k18(platform_cls, monkeypatch):
    from vllm_sm70.ops import dflash2

    monkeypatch.setattr(dflash2, "gemm_available", lambda: True)
    for value, expected in ((None, None), ("0", None), ("1", True)):
        if value is None:
            monkeypatch.delenv("VLLM_SM70_DFLASH2_FP16", raising=False)
        else:
            monkeypatch.setenv("VLLM_SM70_DFLASH2_FP16", value)
        dflash2.reset_selection_cache()
        got = platform_cls.get_unquantized_gemm("auto")
        if expected is None:
            assert got is None, value  # opt-in: default is upstream
        else:
            assert got is dflash2.dflash2_unquantized_gemm
    dflash2.reset_selection_cache()


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


@pytest.mark.parametrize("admitted", [True, False])
@pytest.mark.parametrize("inductor", [True, False])
def test_fused_add_rms_norm_priority_is_vllm_c(
    platform_cls, monkeypatch, inductor, admitted
):
    """K17: when K17 admits the model (Flint), fused_add_rms_norm lowers to
    vllm_c (FP16 residual) under Inductor too, where upstream's CUDA default
    is ["native"] (FP32 residual). Other models (e.g. Qwen) keep upstream's
    priority (fork parity). Other ops always keep upstream's defaults."""
    from vllm.config.compilation import CompilationMode
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70 import ar_rms_fusion

    monkeypatch.setattr(
        ar_rms_fusion,
        "rejection",
        lambda config, tp: None if admitted else "it is Flint-only",
    )

    cc = SimpleNamespace(
        backend="inductor" if inductor else "eager",
        mode=CompilationMode.VLLM_COMPILE if inductor else CompilationMode.NONE,
    )
    cfg = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    ours = platform_cls.get_default_ir_op_priority(cfg)
    theirs = CudaPlatform.get_default_ir_op_priority(cfg)
    if admitted:
        # Both norms take the C kernel, as eager does (T4).
        assert ours.fused_add_rms_norm == ["vllm_c", "native"]
        assert ours.rms_norm == ["vllm_c", "native"]
    else:
        assert ours.fused_add_rms_norm == theirs.fused_add_rms_norm
        assert ours.rms_norm == theirs.rms_norm
    assert ours.gelu_and_mul_sparse == theirs.gelu_and_mul_sparse
    if inductor:
        assert theirs.fused_add_rms_norm == ["native"]
        assert theirs.rms_norm == ["native"]


def test_fused_add_rms_norm_priority_merges_into_kernel_config(
    platform_cls, monkeypatch
):
    """Through upstream's KernelConfig.set_platform_defaults (config/kernel.py:
    451-478): vllm_c first unless the user set a priority."""
    import vllm.platforms
    from vllm.config.compilation import CompilationMode
    from vllm.config.kernel import IrOpPriorityConfig, KernelConfig

    from vllm_sm70 import ar_rms_fusion

    monkeypatch.setattr(vllm.platforms, "_current_platform", platform_cls())
    monkeypatch.setattr(ar_rms_fusion, "rejection", lambda config, tp: None)
    cc = SimpleNamespace(backend="inductor", mode=CompilationMode.VLLM_COMPILE)
    cfg = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(architectures=["FlintSm70ForCausalLM"]),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    kc = KernelConfig()
    monkeypatch.setattr(
        "vllm.config.kernel.bind_passthrough_all2all_backend", lambda *a: None
    )
    kc.set_platform_defaults(cfg)
    assert kc.ir_op_priority.fused_add_rms_norm[0] == "vllm_c"
    assert kc.ir_op_priority.rms_norm[0] == "vllm_c"
    user = KernelConfig(ir_op_priority=IrOpPriorityConfig(fused_add_rms_norm=["native"]))
    user.set_platform_defaults(cfg)
    assert user.ir_op_priority.fused_add_rms_norm == ["native", "vllm_c"]


@pytest.mark.parametrize(
    "reason, expect",
    [
        (
            None,
            "pinned to rms_norm=['vllm_c', 'native'] "
            "fused_add_rms_norm=['vllm_c', 'native']",
        ),
        (
            "_sm70_C lacks tp2_all_reduce_fused_add_rms_norm",
            "not pinned (_sm70_C lacks tp2_all_reduce_fused_add_rms_norm)",
        ),
    ],
)
def test_fused_add_rms_norm_pin_is_logged(platform_cls, monkeypatch, reason, expect):
    """The pin's state is logged at INFO, with K17's reason when off, so a
    missing kernel can never silently leave it off."""
    from vllm.config.compilation import CompilationMode

    from vllm_sm70 import _cuda_platform, ar_rms_fusion

    monkeypatch.setattr(ar_rms_fusion, "rejection", lambda config, tp: reason)
    monkeypatch.setattr(_cuda_platform, "_PIN_LOGGED", set())
    seen = []
    monkeypatch.setattr(
        _cuda_platform.logger, "info", lambda msg, *a: seen.append(msg % a if a else msg)
    )
    cc = SimpleNamespace(backend="inductor", mode=CompilationMode.VLLM_COMPILE)
    cfg = SimpleNamespace(
        compilation_config=cc,
        model_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )
    platform_cls.get_default_ir_op_priority(cfg)
    platform_cls.get_default_ir_op_priority(cfg)  # logged once, not twice
    pin_lines = [m for m in seen if "fused_add_rms_norm IR priority" in m]
    assert len(pin_lines) == 1
    assert expect in pin_lines[0]
