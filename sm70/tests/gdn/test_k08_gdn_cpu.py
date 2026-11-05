# SPDX-License-Identifier: Apache-2.0
"""K08 CPU checks for the FlashQLA GDN ops (no GPU, no built _sm70_C).

What: the GDN schemas in sm70/csrc/gdn are prefixed `gdn_`, mutate only
  `Tensor(a!)`/`Tensor(b!)` outputs and have CUDA impls; every tensor op has
  a fake in vllm_sm70/gdn/kernels.py with the schema's arguments; the
  Python wrappers allocate the right shapes (run on meta tensors through
  the fakes); both routes refuse what they cannot run and honour their
  kill switches; the OOT ChunkGatedDeltaRule is installed and picks
  FlashQLA only on SM70 (needs vLLM's Python source).
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest
from _gdn_helpers import gdn_schemas, mock_ctor_env

torch = pytest.importorskip("torch")

SM70 = pathlib.Path(__file__).resolve().parents[2]
KERNELS_PY = SM70 / "vllm_sm70" / "gdn" / "kernels.py"
GDN_CU = SM70 / "csrc" / "gdn" / "gdn_forward.cu"

H, HV, D = 8, 16, 128  # Qwen3.6-35B-A3B at TP2


def _tensor_ops() -> dict[str, str]:
    return {n: s for n, s in gdn_schemas().items() if "Tensor" in s.split("->", 1)[0]}


# ------------------------------------------------------------------ schemas


def test_schemas():
    schemas = gdn_schemas()
    assert set(schemas) == {
        "gdn_forward_vlk_varlen_out",
        "gdn_decode_mixed_qkv_global_state",
        "gdn_resolve_column_groups_per_block",
    }
    for name, schema in schemas.items():
        assert name.startswith("gdn_"), name
        assert "native_policy" not in schema
    s = schemas["gdn_forward_vlk_varlen_out"]
    assert "Tensor(a!) output" in s and "Tensor(b!) final_state" in s
    assert s.endswith("-> ()")
    s = schemas["gdn_decode_mixed_qkv_global_state"]
    assert "Tensor(a!) state" in s and "Tensor(b!) output" in s
    assert s.endswith("-> ()")
    for name, schema in _tensor_ops().items():
        torch._C.parse_schema(schema)  # well-formed


def test_impl_keys_and_no_pybind():
    text = GDN_CU.read_text()
    assert "PYBIND11_MODULE" not in text and "torch/extension.h" not in text
    assert "ddtree_state" not in text and "GdnPolicy" not in text
    assert "getenv" not in text
    cuda = re.search(r"TORCH_LIBRARY_IMPL\(_sm70_C, CUDA, m\) \{(.*?)\n\}", text, re.S)
    assert cuda is not None
    assert set(re.findall(r'm\.impl\("(\w+)"', cuda.group(1))) == set(_tensor_ops())
    cea = re.search(
        r"TORCH_LIBRARY_IMPL\(_sm70_C, CompositeExplicitAutograd, m\) \{(.*?)\n\}",
        text,
        re.S,
    )
    assert cea is not None
    assert re.findall(r'm\.impl\("(\w+)"', cea.group(1)) == [
        "gdn_resolve_column_groups_per_block"
    ]


def test_sources_cmake_lists_the_tu():
    cmake = (SM70 / "csrc" / "gdn" / "sources.cmake").read_text()
    assert "gdn_forward.cu" in cmake
    top = (SM70 / "CMakeLists.txt").read_text()
    assert "csrc/gdn/sources.cmake" in top


def test_every_tensor_op_has_a_fake():
    text = KERNELS_PY.read_text()
    fakes = set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', text))
    assert fakes == set(_tensor_ops())


def test_fake_signatures_match_schemas(gdn_stub):
    problems = []
    for name, schema in _tensor_ops().items():
        fake = getattr(gdn_stub, f"_{name}_fake", None)
        if fake is None:
            problems.append(f"{name}: no _{name}_fake")
            continue
        args = torch._C.parse_schema(schema).arguments
        params = list(inspect.signature(fake).parameters.values())
        if [p.name for p in params] != [a.name for a in args]:
            problems.append(f"{name}: {[p.name for p in params]}")
    assert not problems, problems
    assert gdn_stub.NO_FAKE_OPS == {"gdn_resolve_column_groups_per_block"}


# ------------------------------------------------------------- env switches


def test_kill_switches_registered():
    from vllm_sm70 import envs

    gdn = envs._SECTIONS["gdn"]
    for name in (
        "VLLM_SM70_GDN_FLASHQLA_PREFILL",
        "VLLM_SM70_GDN_DECODE_FLASHQLA",
        "VLLM_SM70_FLA_RECURRENT_SCHED",
    ):
        assert name in gdn
        assert gdn[name].__name__ == f"env_{name}"  # a kill_switch getter


def test_kill_switch_values(monkeypatch, gdn_stub):
    for value, on in ((None, True), ("1", True), ("", True), ("0", False)):
        for var, fn in (
            (gdn_stub.PREFILL_KILL_SWITCH, gdn_stub.prefill_enabled),
            (gdn_stub.DECODE_KILL_SWITCH, gdn_stub.decode_enabled),
        ):
            if value is None:
                monkeypatch.delenv(var, raising=False)
            else:
                monkeypatch.setenv(var, value)
            assert fn() is on, (var, value)


# --------------------------------------------------- wrappers on meta tensors


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_prefill_wrapper_shapes(gdn_stub):
    T, N = 40, 2
    q = _meta(1, T, H, D)
    k = _meta(1, T, H, D)
    v = _meta(1, T, HV, D)
    g = _meta(1, T, HV, dtype=torch.float32)
    beta = _meta(1, T, HV, dtype=torch.float32)
    h0 = _meta(N, HV, D, D, dtype=torch.float32)
    cu = _meta(N + 1, dtype=torch.int32)
    o, ht = gdn_stub.flashqla_chunk_gated_delta_rule(
        q, k, v, g, beta, h0, True, cu_seqlens=cu
    )
    assert (o.shape, o.dtype) == ((1, T, HV, D), torch.float16)
    assert (ht.shape, ht.dtype) == ((N, HV, D, D), torch.float32)
    # Direct output into a matching core_attn_out buffer.
    core = _meta(T + 8, HV, D)
    o2, ht2 = gdn_stub.flashqla_chunk_gated_delta_rule(
        q, k, v, g, beta, h0, False, cu_seqlens=cu, core_attn_out=core
    )
    assert o2.shape == (1, T, HV, D) and ht2 is None


def _decode_args(T=2, slots=5, dtype=torch.float16, device="meta"):
    def mk(*s, dt=dtype):
        return torch.empty(*s, dtype=dt, device=device)

    return dict(
        mixed_qkv=mk(T, 2 * H * D + HV * D),
        a=mk(T, HV),
        b=mk(T, HV),
        A_log=mk(HV, dt=torch.float32),
        dt_bias=mk(HV),
        ssm_state=mk(slots, HV, D, D, dt=torch.float32),
        state_indices=mk(T, dt=torch.int32),
        out=mk(T, HV, D),
    )


def test_decode_runs_through_fake_when_admitted(monkeypatch, gdn_stub):
    monkeypatch.setattr(gdn_stub, "_is_sm70", lambda device: True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    args = _decode_args()
    assert gdn_stub.decode_rejection(**args) is None
    assert gdn_stub.flashqla_decode(**args) is True


def test_decode_kill_switch_touches_nothing(monkeypatch, gdn_stub):
    monkeypatch.setenv("VLLM_SM70_GDN_DECODE_FLASHQLA", "0")
    called = []
    monkeypatch.setattr(
        gdn_stub, "gdn_decode_mixed_qkv_global_state", lambda *a: called.append(a)
    )
    assert gdn_stub.flashqla_decode(**_decode_args()) is False
    assert not called


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda d: d.update(mixed_qkv=d["mixed_qkv"].float()), "dtype"),
        (lambda d: d.update(A_log=d["A_log"].half()), "dtype"),
        (lambda d: d.update(state_indices=d["state_indices"].long()), "state_indices"),
        (
            lambda d: d.update(ssm_state=torch.empty(5, HV, 64, 64, device="meta")),
            "state_shape",
        ),
        (
            lambda d: d.update(
                ssm_state=torch.empty(5, HV, D, D, device="meta").transpose(2, 3)
            ),
            "state_layout",
        ),
        (
            lambda d: d.update(a=torch.empty(2, HV + 1, device="meta").half()),
            "gate_shape",
        ),
        (
            lambda d: d.update(out=torch.empty(2, HV, 64, device="meta").half()),
            "out_shape",
        ),
        (
            lambda d: d.update(
                mixed_qkv=torch.empty(2, 3 * D + HV * D, device="meta").half()
            ),
            "heads",
        ),
    ],
)
def test_decode_rejections(monkeypatch, gdn_stub, change, reason):
    monkeypatch.setattr(gdn_stub, "_is_sm70", lambda device: True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    args = _decode_args()
    change(args)
    assert gdn_stub.decode_rejection(**args) == reason
    assert gdn_stub.flashqla_decode(**args) is False


def test_cpu_tensors_are_refused(gdn_stub):
    args = _decode_args(device="cpu")
    assert gdn_stub.decode_rejection(**args) == "device"
    q = torch.empty(1, 4, H, D, dtype=torch.float16)
    v = torch.empty(1, 4, HV, D, dtype=torch.float16)
    g = torch.empty(1, 4, HV)
    assert gdn_stub.prefill_rejection(q, q, v, g, g, None) == "device"


@pytest.mark.parametrize(
    "dtype, head_dim, reason",
    [
        (torch.float16, 128, None),
        (torch.bfloat16, 128, "dtype"),  # no bf16 on Volta
        (torch.float16, 64, "head_dim"),
    ],
)
def test_prefill_admission(monkeypatch, gdn_stub, dtype, head_dim, reason):
    monkeypatch.setattr(gdn_stub, "_is_sm70", lambda device: True)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = _meta(1, 4, H, head_dim, dtype=dtype)
    v = _meta(1, 4, HV, head_dim, dtype=dtype)
    g = _meta(1, 4, HV, dtype=torch.float32)
    assert gdn_stub.prefill_rejection(q, q, v, g, g, None) == reason


# ------------------------------------------------------ area registration


def test_register_without_extension_is_a_noop(monkeypatch):
    import vllm_sm70.gdn as gdn_area
    from vllm_sm70 import _ext

    monkeypatch.setattr(gdn_area, "_registered", False)
    monkeypatch.setattr(_ext, "is_loaded", lambda: False)
    gdn_area.register()
    assert gdn_area._registered


# ------------------------------------- OOT ChunkGatedDeltaRule (needs vLLM)


@pytest.fixture
def oot(monkeypatch, gdn_stub):
    pytest.importorskip("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")
    from vllm.model_executor.custom_op import op_registry_oot

    m = mock_ctor_env(monkeypatch)
    monkeypatch.setattr(gdn_stub, "_oot_class", None)
    monkeypatch.delitem(op_registry_oot, "ChunkGatedDeltaRule", raising=False)
    cls = gdn_stub.register_chunk_gated_delta_rule()
    yield m, cls
    op_registry_oot.pop("ChunkGatedDeltaRule", None)


def _pretend_sm70(monkeypatch, gdn_stub):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (7, 0))
    monkeypatch.setattr(gdn_stub, "has_op", lambda name: True)


def test_oot_installed_and_idempotent(oot, gdn_stub):
    from vllm.model_executor.custom_op import op_registry_oot

    m, cls = oot
    assert op_registry_oot["ChunkGatedDeltaRule"] is cls
    assert gdn_stub.register_chunk_gated_delta_rule() is cls
    assert issubclass(cls, m.ChunkGatedDeltaRule)


def test_oot_without_sm70_keeps_upstream(oot):
    m, cls = oot
    op = m.ChunkGatedDeltaRule()
    assert type(op) is cls
    assert op.sm70_flashqla is False
    assert op._forward_method.__name__ == "forward_native"
    assert op.gdn_prefill_backend == "triton"


def test_oot_on_sm70_routes_prefill_to_flashqla(monkeypatch, oot, gdn_stub):
    m, _ = oot
    _pretend_sm70(monkeypatch, gdn_stub)
    op = m.ChunkGatedDeltaRule()
    assert op.sm70_flashqla is True
    assert op._forward_method.__name__ == "forward_flashqla_sm70"
    assert op.gdn_prefill_backend == "flashqla_sm70"


def test_oot_prefill_kill_switch(monkeypatch, oot, gdn_stub):
    m, _ = oot
    _pretend_sm70(monkeypatch, gdn_stub)
    monkeypatch.setenv("VLLM_SM70_GDN_FLASHQLA_PREFILL", "0")
    op = m.ChunkGatedDeltaRule()
    assert op.sm70_flashqla is False
    assert op._forward_method.__name__ == "forward_native"


def test_oot_rejected_tensors_fall_back_to_native(monkeypatch, oot, gdn_stub):
    m, _ = oot
    _pretend_sm70(monkeypatch, gdn_stub)
    op = m.ChunkGatedDeltaRule()
    seen = {}
    monkeypatch.setattr(op, "forward_native", lambda **kw: seen.setdefault("kw", kw))
    q = torch.empty(1, 4, H, D, dtype=torch.float16)  # CPU: rejected ("device")
    v = torch.empty(1, 4, HV, D, dtype=torch.float16)
    g = torch.empty(1, 4, HV)
    op.forward_flashqla_sm70(q, q, v, g, g, None, True)
    assert seen["kw"]["q"] is q
