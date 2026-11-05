# SPDX-License-Identifier: Apache-2.0
"""K12 CPU checks: Flint NVFP4 MoE M<=2 QPN decode (no GPU, no built _sm70_C).

What:
  * Bindings: exactly the two out ops, CUDA impls only, fakes whose
    parameters match the schemas and that run on meta tensors; no fork
    namespace, policy or single-launch (persistent) code left; the C++
    contract constants equal the Python ones; sources.cmake builds the one TU
    without fast math; PROVENANCE.md checks out.
  * Admission, ported from the CPU decode cases of fork cf9492a
    tests/kernels/moe/test_sm70_nvfp4_contract_flint.py:
      test_decode_qpn_contract_is_flint_tp2_only (:554-566),
      test_decode_qpn_scale_shift_keeps_fp16_headroom (:569-585),
      test_decode_qpn_scale_shift_rejects_non_finite_scales (:588-592),
      test_decode_qpn_load_admits_flint_tp2_with_workspace (:615-624),
      test_decode_qpn_load_pre_shifts_large_scales (:627-632),
      test_decode_qpn_load_falls_back (:635-651),
      test_decode_qpn_kill_switch_env (:661-674).
    Dropped: test_decode_qpn_explicit_request_fails_closed_without_ops
    (:654; the environment can only turn the decode off here, so there is no
    explicit request) and every persistent case (:677-792).
  * New: the per-call (metadata-only) guard, the composite wrapper's
    argument order and int64 ID handling, workspace allocation refused under
    CUDA graph capture (R4), and M4's decode constants agree with K12's.
"""

from __future__ import annotations

import ast
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SM70 = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "ops"))
from _schemas import sm70_schemas  # noqa: E402

AREA = "flint_moe"
CSRC = SM70 / "csrc" / AREA
CU = CSRC / "nvfp4_moe_decode_qpn_sm70.cu"
PY = SM70 / "vllm_sm70" / "ops" / "flint_moe.py"
W13 = "nvfp4_moe_decode_w13_swiglu_sm70_out"
W2 = "nvfp4_moe_decode_w2_reduce_sm70_out"
EXPECTED_SCHEMAS = {
    W13: (
        f"{W13}(Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
        "Tensor expert_ids, Tensor(b!) partials, Tensor(c!) counters, "
        "int scale_shift) -> ()"
    ),
    W2: (
        f"{W2}(Tensor(a!) out, Tensor input, Tensor weights, Tensor scales, "
        "Tensor expert_ids, Tensor topk_weights, int scale_shift) -> ()"
    ),
}


def _code_only(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


# ------------------------------------------------------------------ bindings


def test_exactly_the_two_kernel_ops_are_defined():
    assert sm70_schemas(area=AREA) == EXPECTED_SCHEMAS


def test_every_op_has_a_cuda_impl_only():
    impls: dict[str, set[str]] = {}
    for block in re.finditer(
        r"TORCH_LIBRARY_IMPL\(\s*_sm70_C\s*,\s*(\w+)\s*,\s*m\s*\)\s*\{(.*?)\n\}",
        CU.read_text(),
        re.S,
    ):
        for op in re.findall(r'm\.impl\(\s*"(\w+)"', block.group(2)):
            impls.setdefault(op, set()).add(block.group(1))
    assert impls == {W13: {"CUDA"}, W2: {"CUDA"}}


def test_every_op_has_a_matching_fake():
    text = PY.read_text()
    assert set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', text)) == {W13, W2}
    defs = {
        n.name: [a.arg for a in n.args.args]
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.FunctionDef)
    }
    for name, schema in sm70_schemas(area=AREA).items():
        args = schema[schema.index("(") + 1 : schema.rindex(") ->")]
        names = [a.strip().split(" ")[-1] for a in args.split(",")]
        assert defs[f"_{name}_fake"] == names, name
        assert defs[name] == names, name  # the wrapper keeps the schema order


def test_no_fork_namespace_policy_or_single_launch_left():
    code = _code_only(CU.read_text())
    bad = [
        pat
        for pat in (
            r"PYBIND11_MODULE",
            r"TORCH_LIBRARY\w*\(\s*_C\b",
            r"\bwith_policy\b",
            r"\bnative_policy\b",
            r"\bgetenv\b",
            r"\bfused_decode_kernel\b",
            r"\bnvfp4_moe_decode_fused_sm70_out\b",
            r"\bFLINT_FUSED_DECODE_KERNEL\b",
            r"\bfused_resident_blocks\b",
            r"\bkFlintSyncCounters\b",
            r"\bload_acquire_gpu\b",
            r"\bmma_tile_shared\b",
            r"\bstd::atomic\b",
        )
        if re.search(pat, code)
    ]
    py = PY.read_text()
    bad += [
        pat
        for pat in (r"torch\.ops\._C\b", r"native_policy\s*[:=,)]", r"persistent\b")
        if re.search(pat, _code_only(py.split('"""', 2)[2]))
    ]
    assert not bad, bad


def _cpp_constant(name: str) -> str:
    m = re.search(rf"constexpr\s+\w+\s+{name}\s*=\s*([^;]+);", CU.read_text())
    assert m, name
    return m.group(1).strip()


def test_cpp_contract_matches_python_constants():
    text = CU.read_text()
    consts = {}
    for name in (
        "kFlintHidden",
        "kFlintIntermediate",
        "kFlintTopK",
        "kFlintExperts",
        "kFlintMaxTokens",
        "kFlintW13SplitCta",
    ):
        consts[name] = int(_cpp_constant(name))
    assert "static_assert(kFlintCounters == 96 && kFlintPartials == 98304" in text
    src = PY.read_text()
    tree = ast.parse(src)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                pass
    assert values["FLINT_TP2"] == (
        consts["kFlintExperts"],
        consts["kFlintHidden"],
        consts["kFlintIntermediate"],
        consts["kFlintTopK"],
    )
    # Flint TP2: hidden 2048, moe_intermediate 512 / 2 ranks, 256 experts, top-6.
    assert values["FLINT_TP2"] == (256, 2048, 256, 6)
    assert values["MAX_TOKENS"] == consts["kFlintMaxTokens"] == 2
    assert values["DECODE_COUNTERS"] == 96
    assert values["DECODE_PARTIALS"] == 96 * consts["kFlintW13SplitCta"] * 2 * 32
    assert values["MAX_SCALE_SHIFT"] == 14
    assert "scale_shift <= 14" in text


def test_sources_cmake_builds_the_one_tu_without_fast_math():
    text = (CSRC / "sources.cmake").read_text()
    assert re.findall(r'CMAKE_CURRENT_LIST_DIR\}/([\w.]+)"', text) == [CU.name]
    flags = _code_only(text.replace("#", "//"))
    assert "use_fast_math" not in flags
    assert "-U__CUDA_NO_HALF_OPERATORS__" in flags
    top = (SM70 / "CMakeLists.txt").read_text()
    assert 'include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/flint_moe/sources.cmake")' in top


def test_provenance_check_passes():
    script = SM70 / "tools" / "check_flint_moe_provenance.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_kill_switch_is_registered_and_default_on(monkeypatch):
    from vllm_sm70 import envs

    assert "VLLM_SM70_FLINT_MOE_QPN_DECODE" in envs.environment_variables()
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", raising=False)
    assert envs.VLLM_SM70_FLINT_MOE_QPN_DECODE is True
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "1")
    assert envs.VLLM_SM70_FLINT_MOE_QPN_DECODE is True
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    assert envs.VLLM_SM70_FLINT_MOE_QPN_DECODE is False


def test_ops_package_registers_the_module():
    text = (SM70 / "vllm_sm70" / "ops" / "__init__.py").read_text()
    assert "from vllm_sm70.ops import flint_moe" in text


# -------------------------------------------------------- Python (needs torch)

torch = pytest.importorskip("torch")

FLINT_GLOBAL = (2048, 512, 256, 6)  # (hidden, global intermediate, experts, top-k)
QWEN36 = (2048, 512, 256, 8)
QWEN38 = (2560, 640, 512, 10)
GLM53 = (4096, 2048, 288, 8)


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_has_op_and_available(flint_stub):
    assert flint_stub.available()
    assert not flint_stub.has_op("definitely_not_an_op")
    with pytest.raises(RuntimeError, match="not available"):
        flint_stub._op("definitely_not_an_op")


def test_fakes_return_none_on_meta(flint_stub):
    ops = torch.ops._sm70_C
    bank = (_meta(256, 2048, 64, dtype=torch.int32), _meta(256, 128, 512))
    ids = _meta(12, dtype=torch.int32)
    assert (
        getattr(ops, W13)(
            _meta(12, 256),
            _meta(2, 2048),
            *bank,
            ids,
            _meta(98304, dtype=torch.float32),
            _meta(96, dtype=torch.int32),
            0,
        )
        is None
    )
    assert (
        getattr(ops, W2)(
            _meta(2, 2048),
            _meta(12, 256),
            _meta(256, 256, 256, dtype=torch.int32),
            _meta(256, 16, 2048),
            ids,
            _meta(2, 6, dtype=torch.float32),
            3,
        )
        is None
    )


@pytest.mark.parametrize(
    ("contract", "tp_size", "admitted"),
    [
        (FLINT_GLOBAL, 2, True),
        (FLINT_GLOBAL, 1, False),
        (FLINT_GLOBAL, 4, False),
        (QWEN36, 2, False),
        (QWEN38, 4, False),
        (GLM53, 8, False),
    ],
)
def test_decode_contract_is_flint_tp2_only(flint_stub, contract, tp_size, admitted):
    hidden, inter, experts, top_k = contract
    local = (experts, hidden, inter // tp_size, top_k)
    assert (flint_stub.contract_rejection(local) is None) is admitted


@pytest.mark.parametrize(
    ("peak", "shift"),
    [
        (1.0, 0),
        (1.1826171875, 0),  # largest prepared W2 scale in the real checkpoint
        (3.998046875, 0),  # 3.998 * 2^14 = 65504, the FP16 max
        (4.0, 1),
        (14.0, 2),
        (448.0 * 16, 11),
        (65504.0, 14),
    ],
)
def test_scale_shift_keeps_fp16_headroom(flint_stub, peak, shift):
    scales = torch.full((2, 3), peak / 4, dtype=torch.float16)
    scales[1, 2] = peak
    assert flint_stub.scale_shift(scales) == shift
    assert peak * 2.0 ** (14 - shift) <= 65504.0


@pytest.mark.parametrize("bad", (float("inf"), float("nan")))
def test_scale_shift_rejects_non_finite_scales(flint_stub, bad):
    scales = torch.ones(4, dtype=torch.float16)
    scales[2] = bad
    assert flint_stub.scale_shift(scales) is None


def _meta_banks(inter=256, hidden=2048, experts=256):
    return (
        _meta(experts, hidden, 2 * inter // 8, dtype=torch.int32),
        _meta(experts, hidden // 16, 2 * inter),
        _meta(experts, inter, hidden // 8, dtype=torch.int32),
        _meta(experts, inter // 16, hidden),
    )


def test_bank_rejection_admits_flint_tp2_banks(flint_stub):
    assert flint_stub.bank_rejection(*_meta_banks()) is None


@pytest.mark.parametrize(
    "mutate",
    ["tp1", "tp4", "w13_dtype", "w2_scales_dtype", "non_contiguous", "device"],
)
def test_bank_rejection_refuses(flint_stub, mutate):
    banks = list(_meta_banks())
    if mutate == "tp1":
        banks = list(_meta_banks(inter=512))
    elif mutate == "tp4":
        banks = list(_meta_banks(inter=128))
    elif mutate == "w13_dtype":
        banks[0] = _meta(256, 2048, 64, dtype=torch.int64)
    elif mutate == "w2_scales_dtype":
        banks[3] = _meta(256, 16, 2048, dtype=torch.float32)
    elif mutate == "non_contiguous":
        banks[2] = _meta(256, 256, 256, dtype=torch.int32).transpose(1, 2)
    elif mutate == "device":
        banks[1] = torch.empty(256, 128, 512, dtype=torch.float16)  # CPU
    assert flint_stub.bank_rejection(*banks) is not None


def _admit(mod, monkeypatch, *, peak=1.0, contract=None, available=True, **kw):
    """admit_layer with real (small) scales; the bank shapes are checked by
    bank_rejection, tested above on meta banks."""
    monkeypatch.setattr(mod, "available", lambda: available)
    monkeypatch.setattr(mod, "bank_rejection", lambda *a: None)
    w13_scales = torch.full((4, 8), 0.01, dtype=torch.float16)
    w2_scales = torch.full((4, 8), 0.01, dtype=torch.float16)
    w2_scales[3, 1] = peak
    return mod.admit_layer(
        contract or mod.FLINT_TP2, None, w13_scales, None, w2_scales, **kw
    )


def test_admit_layer_admits_flint_tp2(flint_stub, monkeypatch):
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", raising=False)
    plan = _admit(flint_stub, monkeypatch)
    assert plan == flint_stub.DecodePlan(0, 0)


def test_admit_layer_pre_shifts_large_scales(flint_stub, monkeypatch):
    plan = _admit(flint_stub, monkeypatch, peak=14.0, policy_enabled=True)
    assert plan == flint_stub.DecodePlan(0, 2)


@pytest.mark.parametrize(
    ("contract", "enabled", "available", "peak"),
    [
        ((256, 2048, 512, 6), True, True, 1.0),  # Flint TP1
        ((256, 2048, 128, 6), True, True, 1.0),  # Flint TP4
        ((256, 2048, 256, 8), True, True, 1.0),  # Qwen3.6 TP2
        (None, False, True, 1.0),  # kill switch (Sm70Config)
        (None, True, False, 1.0),  # ops absent
        (None, True, True, float("inf")),  # no FP16 range
    ],
)
def test_admit_layer_falls_back(
    flint_stub, monkeypatch, contract, enabled, available, peak
):
    plan = _admit(
        flint_stub,
        monkeypatch,
        peak=peak,
        contract=contract,
        available=available,
        policy_enabled=enabled,
    )
    assert plan is None


def test_admit_layer_reads_the_kill_switch_env(flint_stub, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    assert not flint_stub.enabled()
    assert _admit(flint_stub, monkeypatch) is None
    monkeypatch.delenv("VLLM_SM70_FLINT_MOE_QPN_DECODE")
    assert flint_stub.enabled()
    assert _admit(flint_stub, monkeypatch) is not None


def test_admit_layer_kill_switch_beats_policy_enabled(flint_stub, monkeypatch):
    """VLLM_SM70_FLINT_MOE_QPN_DECODE=0 wins even when the caller passes
    policy_enabled=True: no caller can bypass the kill."""
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "0")
    assert _admit(flint_stub, monkeypatch, policy_enabled=True) is None
    monkeypatch.setenv("VLLM_SM70_FLINT_MOE_QPN_DECODE", "1")  # only "0" acts
    assert _admit(flint_stub, monkeypatch, policy_enabled=True) is not None
    assert _admit(flint_stub, monkeypatch, policy_enabled=False) is None


def test_allocate_workspace_sizes_and_zero_counters(flint_stub):
    ws = flint_stub.allocate_workspace("cpu")
    assert ws.partials.shape == (98304,) and ws.partials.dtype == torch.float32
    assert ws.counters.shape == (96,) and ws.counters.dtype == torch.int32
    assert ws.ids.shape == (12,) and ws.ids.dtype == torch.int32
    assert not ws.counters.any()
    # 384 KiB of partials + 384 B of counters per workspace.
    assert ws.partials.numel() * 4 == 384 * 1024


def test_allocate_workspace_refuses_under_capture(flint_stub, monkeypatch):
    """R4: no workspace may be born inside a CUDA graph capture."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="graph capture"):
        flint_stub.allocate_workspace("cuda")


def test_no_lazy_state_in_the_module():
    """R4: no module-level caches that a first call could fill under capture."""
    tree = ast.parse(PY.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            assert not isinstance(value, (ast.Dict, ast.List, ast.Set)), ast.dump(node)
    assert "functools.cache" not in PY.read_text()
    assert "lru_cache" not in PY.read_text()


def _call(tokens=1, hidden=2048, top_k=6, x_dtype=torch.float16, **over):
    x = over.pop("x", _meta(tokens, hidden, dtype=x_dtype))
    weights = over.pop("weights", _meta(tokens, top_k, dtype=torch.float32))
    ids = over.pop("ids", _meta(tokens, top_k, dtype=torch.int32))
    return x, weights, ids


@pytest.mark.parametrize("tokens", (1, 2))
@pytest.mark.parametrize("ids_dtype", (torch.int32, torch.int64))
def test_call_rejection_admits_m1_m2(flint_stub, tokens, ids_dtype):
    x, w, _ = _call(tokens)
    ids = _meta(tokens, 6, dtype=ids_dtype)
    assert flint_stub.call_rejection(x, w, ids) is None
    assert flint_stub.can_decode(x, w, ids)


_REFUSED = {
    "m0": _call(0),
    "m3": _call(3),
    "m8": _call(8),
    "hidden_2560": _call(hidden=2560),
    "x_fp32": _call(x_dtype=torch.float32),
    "x_non_contiguous": _call(x=_meta(2048, 2).t()[:1]),
    "x_3d": _call(x=_meta(1, 1, 2048)),
    "weights_fp16": _call(weights=_meta(1, 6, dtype=torch.float16)),
    "weights_non_contiguous": _call(weights=_meta(6, 2, dtype=torch.float32).t()[:1]),
    "ids_int16": _call(ids=_meta(1, 6, dtype=torch.int16)),
    "top8": _call(top_k=8),  # Qwen3.6 routing
    "ids_shape": _call(ids=_meta(1, 8, dtype=torch.int32)),
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_call_rejection_refuses(flint_stub, case):
    assert flint_stub.call_rejection(*_REFUSED[case]) is not None


def test_call_rejection_checks_alignment_on_real_tensors(flint_stub):
    storage = torch.zeros(2049, dtype=torch.float16)
    x = storage[1:].view(1, 2048)
    w = torch.zeros(1, 6)
    ids = torch.zeros(1, 6, dtype=torch.int32)
    assert flint_stub.call_rejection(x, w, ids) == "x is not 16-byte aligned"


def _record(mod, monkeypatch):
    calls = []
    monkeypatch.setattr(mod, W13, lambda *a: calls.append(("w13", a)), raising=True)
    monkeypatch.setattr(mod, W2, lambda *a: calls.append(("w2", a)), raising=True)
    return calls


@pytest.mark.parametrize("tokens", (1, 2))
def test_composite_threads_arguments_in_schema_order(flint_stub, monkeypatch, tokens):
    mod = flint_stub
    calls = _record(mod, monkeypatch)
    ws = mod.allocate_workspace("cpu")
    plan = mod.DecodePlan(1, 3)
    out = torch.empty(tokens, 2048, dtype=torch.float16)
    act = torch.empty(tokens * 6, 256, dtype=torch.float16)
    x = torch.zeros(tokens, 2048, dtype=torch.float16)
    banks = [object(), object(), object(), object()]
    weights = torch.zeros(tokens, 6)
    ids = torch.arange(tokens * 6, dtype=torch.int32).view(tokens, 6)
    result = mod.flint_moe_decode_out(out, act, x, *banks, weights, ids, ws, plan)
    assert result is out
    (n13, a13), (n2, a2) = calls
    assert (n13, n2) == ("w13", "w2")
    assert a13[0] is act and a13[1] is x and a13[2:4] == tuple(banks[:2])
    assert a13[4].data_ptr() == ids.data_ptr() and a13[4].shape == (tokens * 6,)
    assert a13[5] is ws.partials and a13[6] is ws.counters and a13[7] == 1
    assert a2[0] is out and a2[1] is act and a2[2:4] == tuple(banks[2:])
    assert a2[4] is a13[4] and a2[5] is weights and a2[6] == 3


def test_composite_copies_int64_ids_into_the_workspace(flint_stub, monkeypatch):
    mod = flint_stub
    calls = _record(mod, monkeypatch)
    ws = mod.allocate_workspace("cpu")
    ids = torch.tensor([[5, 4, 3, 2, 1, 255]], dtype=torch.int64)
    mod.flint_moe_decode_out(
        torch.empty(1, 2048, dtype=torch.float16),
        torch.empty(6, 256, dtype=torch.float16),
        torch.zeros(1, 2048, dtype=torch.float16),
        None,
        None,
        None,
        None,
        torch.zeros(1, 6),
        ids,
        ws,
        mod.DecodePlan(0, 0),
    )
    passed = calls[0][1][4]
    assert passed.dtype == torch.int32
    assert passed.data_ptr() == ws.ids.data_ptr()
    assert passed.tolist() == [5, 4, 3, 2, 1, 255]


def test_m4_uses_k12_admission_not_a_private_copy(flint_stub):
    """M4 (vllm_sm70/quant/nvfp4_moe.py) wires the decode through this module
    (admit_layer, allocate_workspace, can_decode, flint_moe_decode_out); the
    fork constants it pre-wired with are gone, so they cannot drift.

    Taken over from Mara's PR #52 (models/m4-k12-wiring @ 3d2cc70). Skipped
    while M4 (main before #52) still carries its private DECODE_* copy."""
    pytest.importorskip("vllm")
    try:
        from vllm_sm70.quant import nvfp4_moe as m4
    except ImportError as exc:  # pragma: no cover - vLLM without modelopt
        pytest.skip(f"vllm_sm70.quant.nvfp4_moe not importable: {exc}")
    if hasattr(m4, "DECODE_QPN_CONTRACTS"):
        pytest.skip("M4 still carries private DECODE_* constants (pre-#52)")
    assert m4._flint_moe() is flint_stub
    for name in ("DECODE_QPN_CONTRACTS", "DECODE_QPN_COUNTERS", "DECODE_W13_OP"):
        assert not hasattr(m4, name), name
    src = pathlib.Path(m4.__file__).read_text()
    for call in ("admit_layer(", "can_decode(", "flint_moe_decode_out("):
        assert f"flint_moe.{call}" in src, call
