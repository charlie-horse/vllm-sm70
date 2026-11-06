# SPDX-License-Identifier: Apache-2.0
"""K09 CPU checks for the QPN8 ops (no GPU, no built _sm70_C).

What: the QPN8 schemas in sm70/csrc/qpn8 are the three carried ops, with
  `Tensor(a!)` outputs, CUDA impls and no `native_policy` or raw-pointer
  argument; every op has a fake in vllm_sm70/ops/qpn8.py whose arguments
  match the schema and whose shapes match the C++ op; the gemm guards admit
  M in [1, 8] and no longer name a TP4 shape; every TP2 Qwen3.6/3.8 dense
  FP8 shape is admitted and gets a split-K the kernel accepts; the
  qpn8_native_linear custom op picks QPN8 / TurboMind / dequantize+mm by M
  (ports the kernel-level parts of fork tests/quantization/
  test_block_fp8_qpn8_capabilities.py).
"""

from __future__ import annotations

import inspect
import pathlib
import re
import subprocess
import sys

import pytest
from _qpn8_helpers import TP2_SHAPES, qpn8_schemas

torch = pytest.importorskip("torch")

SM70 = pathlib.Path(__file__).resolve().parents[2]
QPN8_PY = SM70 / "vllm_sm70" / "ops" / "qpn8.py"
QPN8_CU = SM70 / "csrc" / "qpn8" / "fp8_qpn8_sm70.cu"


# ------------------------------------------------------------------ schemas


def test_schemas():
    schemas = qpn8_schemas()
    assert set(schemas) == {
        "fp8_qpn8_prepare_sm70",
        "fp8_qpn8_dequantize_sm70_out",
        "fp8_qpn8_gemm_sm70_out",
    }
    for name, schema in schemas.items():
        assert "native_policy" not in schema and "_ptr" not in schema, name
        torch._C.parse_schema(schema)
    assert schemas["fp8_qpn8_gemm_sm70_out"].startswith(
        "fp8_qpn8_gemm_sm70_out(Tensor(a!) out,"
    )
    assert schemas["fp8_qpn8_dequantize_sm70_out"].startswith(
        "fp8_qpn8_dequantize_sm70_out(Tensor(a!) out,"
    )


def test_cuda_impls_and_trim():
    text = QPN8_CU.read_text()
    block = re.search(r"TORCH_LIBRARY_IMPL\(_sm70_C, CUDA, m\) \{(.*?)\n\}", text, re.S)
    assert block is not None
    assert set(re.findall(r'm\.impl\("(\w+)"', block.group(1))) == set(qpn8_schemas())
    for gone in (
        "sm70_policy.h",
        "dense_weight_ptr",
        "from_blob",
        "gated_pair_sm70_out",
        "hc_dispatch",
        "TORCH_LIBRARY_FRAGMENT(_C",
    ):
        assert gone not in text, gone


def _gemm_checks() -> str:
    body = QPN8_CU.read_text().split("void fp8_qpn8_gemm_sm70_out(", 1)[1]
    body = body.split("\n}\n", 1)[0]
    return " ".join(re.findall(r"TORCH_CHECK\((.*?)\);", body, re.S))


def test_gemm_guards_are_shape_generic():
    """The TP4-only lesson (#34/#40): no guard may name a model shape, and
    block scales are not refused above M=8 any more."""
    joined = _gemm_checks()
    for literal in re.findall(r"\b\d{3,}\b", joined):
        assert literal in ("128",), f"shape literal {literal} in a guard"
    assert "requires channel scales" not in joined


@pytest.fixture
def m32(monkeypatch):
    """Set VLLM_SM70_FP8_QPN8_M32: m32(True) -> "1", m32(False) -> unset."""

    def apply(on: bool):
        if on:
            monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32", "1")
        else:
            monkeypatch.delenv("VLLM_SM70_FP8_QPN8_M32", raising=False)

    apply(False)
    return apply


def test_m_cap_has_one_source(qpn8_stub, m32):
    """QPN8_MAX_M is the compiled C++ bound; qpn8_max_m() derives from it and
    is what the dispatch and the TurboMind-copy rule use."""
    cap = qpn8_stub.QPN8_MAX_M
    assert isinstance(cap, int) and cap == 32
    bounds = re.findall(r"m >= 1 && m <= (\d+)", _gemm_checks())
    assert bounds == [str(cap)], bounds
    src = QPN8_PY.read_text()
    assert "if m <= qpn8_max_m(block_scaled):" in src
    assert (
        "> qpn8_max_m(block_scaled)" in src.split("def needs_turbomind_fallback", 1)[1]
    )
    for on, block_cap in ((False, 8), (True, cap)):
        m32(on)
        assert qpn8_stub.qpn8_max_m() == block_cap
        assert qpn8_stub.qpn8_max_m(block_scaled=True) == block_cap
        assert qpn8_stub.qpn8_max_m(block_scaled=False) == cap
        assert qpn8_stub.needs_turbomind_fallback(block_cap, 0) is False
        assert qpn8_stub.needs_turbomind_fallback(block_cap + 1, 0) is True
        assert qpn8_stub.needs_turbomind_fallback(cap, 0, block_scaled=False) is False


@pytest.mark.parametrize("value", [None, "", "0", "true", "on", " 1"])
def test_m32_switch_only_1_enables(qpn8_stub, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("VLLM_SM70_FP8_QPN8_M32", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_FP8_QPN8_M32", value)
    assert qpn8_stub.qpn8_max_m() == 8


def test_m32_switch_registered():
    from vllm_sm70 import envs

    assert "VLLM_SM70_FP8_QPN8_M32" in envs.environment_variables()


def test_sources_cmake_lists_the_tu():
    assert "fp8_qpn8_sm70.cu" in (SM70 / "csrc/qpn8/sources.cmake").read_text()
    assert "csrc/qpn8/sources.cmake" in (SM70 / "CMakeLists.txt").read_text()


def test_every_op_has_a_fake():
    fakes = set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', QPN8_PY.read_text()))
    assert fakes == set(qpn8_schemas())


def test_fake_signatures_match_schemas(qpn8_stub):
    problems = []
    for name, schema in qpn8_schemas().items():
        fake = getattr(qpn8_stub, f"_{name}_fake", None)
        if fake is None:
            problems.append(f"{name}: no fake")
            continue
        args = [a.name for a in torch._C.parse_schema(schema).arguments]
        params = list(inspect.signature(fake).parameters)
        if params != args:
            problems.append(f"{name}: {params} vs {args}")
    assert not problems, problems
    assert qpn8_stub.NO_FAKE_OPS == frozenset()


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


@pytest.mark.parametrize("n, k", [(128, 128), (4608, 2048), (17408, 5120)])
def test_prepare_fake_shapes(qpn8_stub, n, k):
    codes, scales = torch.ops._sm70_C.fp8_qpn8_prepare_sm70(
        _meta(n, k, dtype=torch.float8_e4m3fn),
        _meta(n // 128, k // 128, dtype=torch.float32),
    )
    assert (codes.shape, codes.dtype) == ((k, n), torch.uint8)
    assert (scales.shape, scales.dtype) == ((k // 128, n // 32), torch.float16)
    codes, scales = torch.ops._sm70_C.fp8_qpn8_prepare_sm70(
        _meta(n, k, dtype=torch.float8_e4m3fn), _meta(n, 1, dtype=torch.float32)
    )
    assert scales.shape == (1, n)


def test_out_op_fakes_return_none(qpn8_stub):
    out = _meta(4, 256)
    codes = _meta(512, 256, dtype=torch.uint8)
    scales = _meta(4, 8)
    assert (
        torch.ops._sm70_C.fp8_qpn8_gemm_sm70_out(
            out, _meta(4, 512), codes, scales, 16, 2, True, False
        )
        is None
    )
    assert (
        torch.ops._sm70_C.fp8_qpn8_dequantize_sm70_out(_meta(512, 256), codes, scales)
        is None
    )


def test_provenance_check_passes():
    script = SM70 / "tools" / "check_qpn8_provenance.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------- config/admission


@pytest.mark.parametrize("name", sorted(TP2_SHAPES))
def test_tp2_dense_shapes_are_admitted(qpn8_stub, name):
    n, k = TP2_SHAPES[name]
    assert qpn8_stub.qpn8_rejection(n, k) is None
    split, chains, prefetch = qpn8_stub.qpn8_config(k, n)
    # The C++ guards of fp8_qpn8_gemm_sm70_out for M up to 32 (split <= 16).
    assert split in (4, 8, 12, 16) and (k // 16) % split == 0
    assert chains in (1, 2)
    assert split != 12 or (chains == 2 and not prefetch)
    assert (k, n, False) not in qpn8_stub.QPN8_CONFIGS  # generic at TP2


def test_tp4_table_entries_kept(qpn8_stub):
    assert qpn8_stub.qpn8_config(4352, 5120) == (16, 1, False)
    assert qpn8_stub.qpn8_config(5120, 8704) == (16, 1, True)
    assert qpn8_stub.qpn8_config(2048 + 128, 512) == (8, 2, False)  # K % 256


@pytest.mark.parametrize(
    "n, k", [(0, 128), (128, 0), (127, 128), (128, 129), (32, 2048)]
)
def test_unaligned_and_empty_weights_report_reason(qpn8_stub, n, k):
    reason = qpn8_stub.qpn8_rejection(n, k)
    assert reason is not None and "divisible by 128" in reason


@pytest.mark.parametrize(
    "max_num_seqs, spec, copy_default, copy_m32",
    [
        (1, 5, False, False),
        (8, 0, False, False),
        (9, 0, True, False),
        (2, 5, True, False),
        (1, 7, False, False),  # Qwen3.8 + DFlash2 k=7, 1 request: 8 rows
        (4, 7, True, False),  # 4 requests: 32 rows
        (5, 7, True, True),
        (33, 0, True, True),
    ],
)
def test_turbomind_fallback_follows_decode_rows(
    qpn8_stub, m32, max_num_seqs, spec, copy_default, copy_m32
):
    assert qpn8_stub.needs_turbomind_fallback(max_num_seqs, spec) is copy_default
    m32(True)
    assert qpn8_stub.needs_turbomind_fallback(max_num_seqs, spec) is copy_m32


# ------------------------------------------------- the opaque linear op


def _operands(n=256, k=512):
    return (
        torch.zeros(k, n, dtype=torch.uint8),
        torch.zeros(k // 128, n // 32, dtype=torch.float16),
    )


@pytest.fixture
def recorder(monkeypatch, qpn8_stub):
    calls = []

    def gemm(out, x, codes, scales, split, chains, fast, prefetch):
        calls.append(("qpn8", x.shape[0], split, chains))
        out.fill_(1)

    def dequant(out, codes, scales):
        calls.append(("dequant", tuple(out.shape)))
        out.fill_(0.5)

    class FakeTm:
        @staticmethod
        def fp8_gemm_sm70_out(out, x, w, s, g, k_ld, q_ld, gated):
            calls.append(("turbomind", x.shape[0], k_ld, q_ld))
            out.fill_(2)

    import vllm_sm70.ops as ops_pkg

    monkeypatch.setattr(qpn8_stub, "fp8_qpn8_gemm_sm70_out", gemm)
    monkeypatch.setattr(qpn8_stub, "fp8_qpn8_dequantize_sm70_out", dequant)
    monkeypatch.setattr(ops_pkg, "tm", FakeTm, raising=False)
    monkeypatch.setitem(__import__("sys").modules, "vllm_sm70.ops.tm", FakeTm)
    return calls


def test_empty_activation_does_not_launch(recorder, qpn8_stub):
    codes, scales = _operands()
    y = qpn8_stub._qpn8_native_linear_impl(
        torch.empty(0, 512, dtype=torch.float16), codes, scales, 16, 2, False
    )
    assert y.shape == (0, 256) and y.dtype == torch.float16
    assert recorder == []


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_block_m_le_8_uses_qpn8_by_default(recorder, qpn8_stub, m32, m):
    codes, scales = _operands()
    y = qpn8_stub._qpn8_native_linear_impl(
        torch.ones(m, 512, dtype=torch.float16), codes, scales, 16, 2, False
    )
    assert recorder == [("qpn8", m, 16, 2)] and y.shape == (m, 256)


@pytest.mark.parametrize("m", [9, 16, 17, 24, 32])
def test_block_m9_32_default_takes_the_fallback(recorder, qpn8_stub, m32, m):
    codes, scales = _operands()
    x = torch.ones(m, 512, dtype=torch.float16)
    qpn8_stub._qpn8_native_linear_impl(x, codes, scales, 16, 2, False)
    assert recorder == [("dequant", (512, 256))]
    recorder.clear()
    w, s = torch.zeros(1), torch.zeros(1)
    qpn8_stub._qpn8_native_linear_impl(x, codes, scales, 16, 2, False, w, s, 7, 9)
    assert recorder == [("turbomind", m, 7, 9)]


@pytest.mark.parametrize("m", [9, 16, 17, 24, 32])
def test_block_m9_32_with_opt_in_uses_qpn8(recorder, qpn8_stub, m32, m):
    m32(True)
    codes, scales = _operands()
    y = qpn8_stub._qpn8_native_linear_impl(
        torch.ones(m, 512, dtype=torch.float16), codes, scales, 16, 2, False
    )
    assert recorder == [("qpn8", m, 16, 2)] and y.shape == (m, 256)


@pytest.mark.parametrize("m", [9, 16, 32])
def test_channel_m9_32_uses_qpn8_by_default(recorder, qpn8_stub, m32, m):
    codes = torch.zeros(512, 256, dtype=torch.uint8)
    scales = torch.zeros(1, 256, dtype=torch.float16)  # channel scales [1, N]
    qpn8_stub._qpn8_native_linear_impl(
        torch.ones(m, 512, dtype=torch.float16), codes, scales, 16, 2, False
    )
    assert recorder == [("qpn8", m, 16, 2)]


def test_m_gt_32_uses_turbomind_when_held(recorder, qpn8_stub):
    codes, scales = _operands()
    w, s = torch.zeros(1), torch.zeros(1)
    y = qpn8_stub._qpn8_native_linear_impl(
        torch.ones(33, 512, dtype=torch.float16),
        codes,
        scales,
        16,
        2,
        False,
        w,
        s,
        7,
        9,
    )
    assert recorder == [("turbomind", 33, 7, 9)] and bool((y == 2).all())


def test_m_gt_32_dequantizes_into_caller_scratch(recorder, qpn8_stub):
    codes, scales = _operands()
    scratch = torch.zeros(512 * 256 + 10, dtype=torch.float16)
    x = torch.ones(48, 512, dtype=torch.float16)
    y = qpn8_stub._qpn8_native_linear_impl(
        x, codes, scales, 16, 2, False, dense_scratch=scratch
    )
    assert recorder == [("dequant", (512, 256))]
    assert bool((scratch[: 512 * 256] == 0.5).all())  # written in place
    assert torch.allclose(y.float(), torch.full((48, 256), 256.0))
    recorder.clear()
    y2 = qpn8_stub._qpn8_native_linear_impl(x, codes, scales, 16, 2, False)
    assert recorder == [("dequant", (512, 256))]
    assert torch.equal(y, y2)


def test_custom_op_fake_and_wrapper_shapes(qpn8_stub):
    prepared = qpn8_stub.Qpn8Prepared(
        _meta(512, 384, dtype=torch.uint8), _meta(4, 12), 384, 512, 16, 2, False
    )
    y = qpn8_stub.qpn8_linear(_meta(2, 3, 512), prepared)
    assert y.shape == (2, 3, 384)
    with pytest.raises(TypeError, match="FP16"):
        qpn8_stub.qpn8_linear(_meta(2, 512, dtype=torch.float32), prepared)


def test_ops_area_imports_qpn8():
    init = (SM70 / "vllm_sm70" / "ops" / "__init__.py").read_text()
    assert "from vllm_sm70.ops import qpn8" in init
