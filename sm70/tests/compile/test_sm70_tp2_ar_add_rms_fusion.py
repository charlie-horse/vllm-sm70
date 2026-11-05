# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K17 TP2 all-reduce + fused_add_rms_norm: pattern, admission, runtime gate.

Ported from fork cf9492a tests/compile/passes/test_sm70_tp2_ar_add_rms_fusion.py.
CPU only, and none of it needs the custom_ar kernel (U5) in _sm70_C. The
patterns are matched on fake-tensor graphs built like the post-grad graph of
a decoder: the all-reduce, an optional elementwise scale or two summed
partial outputs, then the fused_add_rms_norm IR op.

Differences from the fork: the fork's model-config hook and communication
policy are gone; admission is ar_rms_fusion.rejection() plus register()'s
kill-switch / kernel gate, and the runtime gate is can_fuse().
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

EPS = 1e-6
H = 2048

# Upstream's fusion package resolves these torch.ops._C ops at import
# (matcher_utils.py, rms_quant_fusion.py). Without vLLM's compiled _C (CPU
# tests), define placeholders so BasePattern can be imported; no test
# calls them.
_C_IMPORT_OPS = (
    "rotary_embedding",
    "static_scaled_fp8_quant",
    "dynamic_scaled_fp8_quant",
    "dynamic_per_token_scaled_fp8_quant",
    "silu_and_mul",
    "fused_add_rms_norm",
    "rms_norm_static_fp8_quant",
    "fused_add_rms_norm_static_fp8_quant",
    "rms_norm_dynamic_per_token_quant",
)
_STUB_LIB = None


def _stub_missing_c_ops() -> None:
    global _STUB_LIB
    missing = [n for n in _C_IMPORT_OPS if not hasattr(torch.ops._C, n)]
    if not missing:
        return
    if _STUB_LIB is None:
        _STUB_LIB = torch.library.Library("_C", "FRAGMENT")  # noqa: TOR901
    for name in missing:
        _STUB_LIB.define(f"{name}(Tensor x) -> Tensor")


@pytest.fixture(scope="module")
def fusion():
    """(vllm_sm70.ar_rms_fusion, upstream allreduce_rms_fusion), op registered."""
    _stub_missing_c_ops()
    from vllm.compilation.passes.fusion import allreduce_rms_fusion as upstream
    from vllm.distributed import parallel_state  # noqa: F401  (vllm::all_reduce)
    from vllm.model_executor.layers import layernorm  # noqa: F401  (IR norm op)
    from vllm_sm70 import ar_rms_fusion

    ar_rms_fusion.register_op()
    return ar_rms_fusion, upstream


@pytest.fixture
def tp_group(fusion, monkeypatch):
    _, upstream = fusion
    monkeypatch.setattr(
        upstream, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    monkeypatch.setattr(upstream, "get_tensor_model_parallel_world_size", lambda: 2)
    return fusion[0]


def _fused_op():
    return torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm.default


def _all_reduce_op():
    return torch.ops.vllm.all_reduce.default


def _norm(reduced, residual, weight, eps=EPS):
    import vllm.ir.ops

    return vllm.ir.ops.fused_add_rms_norm(reduced, residual, weight, eps)


def _apply(
    fusion_mod,
    model,
    *,
    hidden=H,
    dtype=torch.float16,
    sum2=False,
    eps=EPS,
    weight_dtype=None,
):
    from torch._inductor.pattern_matcher import PatternMatcherPass
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    with FakeTensorMode():
        inputs = [torch.empty(2, hidden, dtype=dtype)]
        if sum2:
            inputs.append(torch.empty(2, hidden, dtype=dtype))
        inputs += [
            torch.empty(2, hidden, dtype=dtype),
            torch.empty(hidden, dtype=weight_dtype or dtype),
        ]
        graph = make_fx(model)(*inputs)
        patterns = PatternMatcherPass()
        fusion_mod.Tp2AllReduceFusedAddRMSNormPattern(eps, dtype, "cpu").register(
            patterns
        )
        matches = patterns.apply(graph.graph)
    fused = [n for n in graph.graph.nodes if n.target == _fused_op()]
    return matches, fused, graph


# ---------------------------------------------------------------------------
# Pattern matching (fork tests, unchanged in substance)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keep_residual", [False, True])
def test_plain_boundary_is_fused(tp_group, keep_residual):
    def model(x, residual, weight):
        out = _norm(
            torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0"), residual, weight
        )
        return out if keep_residual else out[0]

    matches, fused, graph = _apply(tp_group, model)
    assert matches == 1 and len(fused) == 1
    node = fused[0]
    # (tensor, tensor_b, residual, weight, epsilon, scale, group_name)
    assert node.args[1] is None
    assert node.args[4] == EPS and node.args[5] == 1.0
    assert "tp:0" in (*node.args[6:], *node.kwargs.values())
    assert not any(n.target == _all_reduce_op() for n in graph.graph.nodes)


@pytest.mark.parametrize("keep_residual", [False, True])
def test_partial_sum_boundary_is_fused(tp_group, keep_residual):
    def model(a, b, residual, weight):
        reduced = torch.ops.vllm.all_reduce(a.relu() + b.relu(), group_name="tp:0")
        out = _norm(reduced, residual, weight)
        return out if keep_residual else out[0]

    matches, fused, graph = _apply(tp_group, model, sum2=True)
    assert matches == 1 and len(fused) == 1
    assert fused[0].args[1] is not None
    assert not any(n.target == torch.ops.aten.add.Tensor for n in graph.graph.nodes)


@pytest.mark.parametrize("scale, expected", [(0.125, 1), (0.5, 1), (0.3, 0)])
def test_scaled_boundary_needs_a_power_of_two(tp_group, scale, expected):
    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced * scale, residual, weight)

    matches, fused, _ = _apply(tp_group, model)
    assert matches == expected and len(fused) == expected
    if fused:
        assert fused[0].args[5] == scale


def test_added_sum_with_another_user_is_not_folded(tp_group):
    def model(a, b, residual, weight):
        total = a + b
        reduced = torch.ops.vllm.all_reduce(total, group_name="tp:0")
        out, res = _norm(reduced, residual, weight)
        return out, res, total

    matches, fused, _ = _apply(tp_group, model, sum2=True)
    # The add stays (it has another user); the collective + norm still fuse.
    assert matches == 1 and fused[0].args[1] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 4096},  # another hidden size
        {"dtype": torch.bfloat16},
        {"eps": 1e-5},  # the model's epsilon differs from the pattern's
    ],
)
def test_other_shapes_are_left_alone(tp_group, kwargs):
    hidden = kwargs.get("hidden", H)

    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced, residual, weight, 1e-6)

    matches, fused, _ = _apply(
        tp_group,
        model,
        hidden=hidden,
        dtype=kwargs.get("dtype", torch.float16),
        eps=kwargs.get("eps", EPS),
    )
    assert matches == 0 and not fused


@pytest.mark.parametrize("gemma_style", [False, True])
def test_fp32_weight_is_not_matched(tp_group, gemma_style):
    """A GemmaRMSNorm-style fp32 weight (+ 1) never matches."""

    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        w = weight.float() + 1 if gemma_style else weight
        return _norm(reduced, residual, w)

    matches, fused, graph = _apply(
        tp_group, model, weight_dtype=torch.float16 if gemma_style else torch.float32
    )
    assert matches == 0 and not fused
    assert any(n.target == _all_reduce_op() for n in graph.graph.nodes)


def test_unfused_boundary_counter(tp_group):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced * 0.3, residual, weight)

    with FakeTensorMode():
        graph = make_fx(model)(
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(H, dtype=torch.float16),
        )
    assert tp_group.unfused_boundaries(graph.graph) == 1


def test_op_schema_and_fake(fusion):
    """The op the pattern targets; Penny/Mara rely on this exact interface."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    schema = str(_fused_op()._schema)
    assert schema.startswith("vllm_sm70::tp2_all_reduce_fused_add_rms_norm(")
    for arg in (
        "Tensor tensor",
        "Tensor? tensor_b",
        "Tensor residual",
        "Tensor weight",
        "float epsilon",
        "float scale",
        "str group_name",
    ):
        assert arg in schema, schema
    assert schema.endswith("-> (Tensor, Tensor)")
    with FakeTensorMode():
        x = torch.empty(3, H, dtype=torch.float16)
        out, res = torch.ops.vllm_sm70.tp2_all_reduce_fused_add_rms_norm(
            x, None, x, torch.empty(H, dtype=torch.float16), EPS, 1.0, "tp:0"
        )
    assert out.shape == res.shape == (3, H)


# ---------------------------------------------------------------------------
# Admission: engine config (fork tp2_add_rms_rejection)
# ---------------------------------------------------------------------------


FLINT = "FlintForCausalLM"
QWEN36_LIKE = "Qwen3_5MoeForConditionalGeneration"  # hidden 2048, GemmaRMSNorm


def _config(**overrides):
    values = {
        "pp": 1,
        "dtype": torch.float16,
        "hidden": H,
        "disable_ca": False,
        "arch": FLINT,
    }
    values.update(overrides)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=values["pp"],
            disable_custom_all_reduce=values["disable_ca"],
        ),
        model_config=SimpleNamespace(
            architecture=values["arch"],
            dtype=values["dtype"],
            get_hidden_size=lambda: values["hidden"],
            hf_text_config=SimpleNamespace(rms_norm_eps=1e-6),
        ),
    )


@pytest.mark.parametrize(
    "overrides, tp, has_kernel, reason",
    [
        ({}, 2, True, None),
        ({}, 1, True, "TP2"),
        ({}, 4, True, "TP2"),
        ({"pp": 2}, 2, True, "TP2"),
        ({"disable_ca": True}, 2, True, "custom all-reduce"),
        ({"dtype": torch.bfloat16}, 2, True, "FP16"),
        ({"hidden": 4096}, 2, True, "4096"),
        ({}, 2, False, "VLLM_SM70_VLLM_CSRC_DIR"),
        ({"arch": QWEN36_LIKE}, 2, True, "Flint-only"),
        ({"arch": "LlamaForCausalLM"}, 2, True, "Flint-only"),
    ],
)
def test_pass_admission(fusion, monkeypatch, overrides, tp, has_kernel, reason):
    mod, _ = fusion
    monkeypatch.setattr(mod, "kernel_available", lambda: has_kernel)
    result = mod.rejection(_config(**overrides), tp)
    if reason is None:
        assert result is None
    else:
        assert reason in result


# ---------------------------------------------------------------------------
# Registration: kill switch and U5 (kernel) presence
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_registry(fusion, monkeypatch):
    from vllm_sm70 import passes

    monkeypatch.setattr(passes, "_PATTERNS", [])
    fusion[0]._log_once.cache_clear()
    yield passes
    fusion[0]._log_once.cache_clear()


@pytest.mark.parametrize("raw, enabled", [(None, True), ("1", True), ("0", False)])
def test_kill_switch(monkeypatch, raw, enabled):
    from vllm_sm70 import envs

    if raw is None:
        monkeypatch.delenv("VLLM_SM70_FLINT_AR_RMSNORM", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FLINT_AR_RMSNORM", raw)
    assert envs.VLLM_SM70_FLINT_AR_RMSNORM is enabled
    assert "VLLM_SM70_FLINT_AR_RMSNORM" in envs._SECTIONS["passes"]
    assert "VLLM_SM70_VLLM_CSRC_DIR" in envs._SECTIONS["build"]


def test_no_kernel_no_pattern_logged_once(fusion, clean_registry, monkeypatch, caplog):
    mod, _ = fusion
    monkeypatch.delenv("VLLM_SM70_FLINT_AR_RMSNORM", raising=False)
    monkeypatch.setattr(mod, "kernel_available", lambda: False)
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        assert mod.register() is False
        assert mod.register() is False
    assert clean_registry.patterns() == []
    hits = [r for r in caplog.records if mod.KERNEL_NAME in r.getMessage()]
    assert len(hits) == 1


def test_kill_switch_off_no_pattern(fusion, clean_registry, monkeypatch, caplog):
    mod, _ = fusion
    monkeypatch.setenv("VLLM_SM70_FLINT_AR_RMSNORM", "0")
    monkeypatch.setattr(mod, "kernel_available", lambda: True)
    with caplog.at_level(logging.INFO, logger="vllm.sm70"):
        assert mod.register() is False
        assert mod.register() is False
    assert clean_registry.patterns() == []
    assert sum("=0" in r.getMessage() for r in caplog.records) == 1


def test_registers_with_kernel_and_changes_uuid(fusion, clean_registry, monkeypatch):
    mod, _ = fusion
    monkeypatch.delenv("VLLM_SM70_FLINT_AR_RMSNORM", raising=False)
    monkeypatch.setattr(mod, "kernel_available", lambda: True)
    before = clean_registry.Sm70CompositePass().uuid()
    assert mod.register() is True
    assert clean_registry.patterns() == [(mod.PATTERN_NAME, mod.PATTERN_VERSION)]
    assert clean_registry.Sm70CompositePass().uuid() != before


def test_passes_area_without_kernel_registers_nothing(clean_registry, monkeypatch):
    # The real build state of the CPU suite: _sm70_C absent or without U5.
    from vllm_sm70 import ar_rms_fusion

    if ar_rms_fusion.kernel_available():
        pytest.skip("_sm70_C has the custom_ar kernel")
    clean_registry.register()
    assert clean_registry.patterns() == []


# ---------------------------------------------------------------------------
# Runtime gate and fallback (no kernel needed)
# ---------------------------------------------------------------------------


def _ops(rows=2, hidden=H, dtype=torch.float16, sum2=False):
    x = torch.zeros(rows, hidden, dtype=dtype)
    return (
        x,
        (torch.zeros_like(x) if sum2 else None),
        torch.zeros_like(x),
        (torch.zeros(hidden, dtype=dtype)),
    )


@pytest.mark.parametrize("rows", [1, 2, 5, 16])
@pytest.mark.parametrize("sum2", [False, True])
def test_operands_admitted(fusion, rows, sum2):
    assert fusion[0].operands_admitted(*_ops(rows, sum2=sum2))


@pytest.mark.parametrize(
    "kwargs", [{"rows": 17}, {"rows": 64}, {"hidden": 4096}, {"dtype": torch.bfloat16}]
)
def test_operands_rejected(fusion, kwargs):
    assert not fusion[0].operands_admitted(*_ops(**kwargs))


def test_operands_rejected_layout(fusion):
    mod = fusion[0]
    x, _, res, w = _ops(4)
    assert not mod.operands_admitted(x.t().contiguous().t()[:, :H], None, res, w)
    big = torch.zeros(4 * H + 1, dtype=torch.float16)
    assert not mod.operands_admitted(big[1:].view(4, H), None, res, w)  # 2-byte off
    assert not mod.operands_admitted(x, None, res[:2], w)
    assert not mod.operands_admitted(x, None, res, w.float())


def _ca(**overrides):
    values = {
        "_ptr": 1234,
        "disabled": False,
        "rank": 0,
        "world_size": 2,
        "buffer_ptrs": [0x1000, 0x2000],
        "max_size": 8 * 1024 * 1024,
        "should_custom_ar": lambda inp: True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "ca, ok",
    [
        (_ca(), True),
        (None, False),
        (_ca(disabled=True), False),
        (_ca(world_size=4, buffer_ptrs=[1, 2, 3, 4]), False),
        (_ca(_ptr=0), False),
        (_ca(buffer_ptrs=[0x1000, 0x1000]), False),  # multi-node fallback
        (SimpleNamespace(disabled=False), False),  # not upstream's interface
    ],
)
def test_communicator_admitted(fusion, ca, ok):
    assert fusion[0].communicator_admitted(ca) is ok


def test_cpu_operands_never_fuse(fusion, monkeypatch):
    mod = fusion[0]
    monkeypatch.setattr(mod, "kernel_available", lambda: True)
    assert not mod.can_fuse(_ca(), *_ops(2))


@pytest.mark.parametrize("sum2", [False, True])
@pytest.mark.parametrize("scale", [1.0, 0.125])
def test_op_falls_back_to_the_unfused_ops(fusion, monkeypatch, sum2, scale):
    """No usable custom all-reduce: the op runs all_reduce (+ scale) + norm."""
    import vllm.ir.ops
    from vllm.distributed import parallel_state

    mod = fusion[0]
    calls = []

    def all_reduce(t):
        calls.append(t.clone())
        return t * 2

    group = SimpleNamespace(
        device_communicator=SimpleNamespace(ca_comm=None),
        _all_reduce_out_place=all_reduce,
    )
    monkeypatch.setitem(parallel_state._groups, "k17:test", lambda: group)

    def fake_norm(x, residual, weight, epsilon):
        assert epsilon == EPS
        return x + residual + weight, x + residual

    monkeypatch.setattr(vllm.ir.ops, "fused_add_rms_norm", fake_norm)
    x = torch.arange(2 * H, dtype=torch.float16).view(2, H) % 7
    b = torch.ones_like(x) if sum2 else None
    res = torch.full_like(x, 3)
    w = torch.full((H,), 5, dtype=torch.float16)
    out, res_out = mod.tp2_all_reduce_fused_add_rms_norm(
        x, b, res, w, EPS, scale, "k17:test"
    )
    total = x + b if sum2 else x
    assert torch.equal(calls[0], total)
    reduced = total * 2 * scale if scale != 1.0 else total * 2
    assert torch.equal(res_out, reduced + res)
    assert torch.equal(out, reduced + res + w)


def test_fork_keys_on_the_flint_architecture(fusion):
    assert fusion[0].FLINT_ARCHITECTURE == FLINT
    assert "Flint-only" in fusion[0].rejection(SimpleNamespace(model_config=None), 2)


@pytest.mark.parametrize("arch, expect_patterns", [(QWEN36_LIKE, False), (FLINT, True)])
def test_apply_builds_patterns_only_for_flint(
    tp_group, fusion, monkeypatch, arch, expect_patterns
):
    """Qwen3.6-like (hidden 2048, FP16, TP2): rejected, no pattern built, graph
    untouched. Flint: patterns built and the boundary fused."""
    import vllm.config
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from vllm.distributed import parallel_state

    mod = tp_group
    import vllm.distributed

    monkeypatch.setattr(mod, "kernel_available", lambda: True)
    monkeypatch.setattr(
        vllm.config, "get_current_vllm_config", lambda: _config(arch=arch)
    )
    monkeypatch.setattr(
        parallel_state, "get_tensor_model_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        vllm.distributed, "get_tp_group", lambda: SimpleNamespace(unique_name="tp:0")
    )
    mod._pattern_pass.cache_clear()
    mod._log_once.cache_clear()

    def model(x, residual, weight):
        reduced = torch.ops.vllm.all_reduce(x.relu(), group_name="tp:0")
        return _norm(reduced, residual, weight)

    with FakeTensorMode():
        graph = make_fx(model)(
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(2, H, dtype=torch.float16),
            torch.empty(H, dtype=torch.float16),
        )
        matched = mod.apply(graph.graph)
    built = mod._pattern_pass.cache_info().currsize
    mod._pattern_pass.cache_clear()
    has_fused = any(n.target == _fused_op() for n in graph.graph.nodes)
    if expect_patterns:
        assert matched == 1 and built == 1 and has_fused
    else:
        assert matched == 0 and built == 0 and not has_fused
        assert any(n.target == _all_reduce_op() for n in graph.graph.nodes)
