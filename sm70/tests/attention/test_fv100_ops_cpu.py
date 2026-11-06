# SPDX-License-Identifier: Apache-2.0
"""K05 CPU checks for the Flash-V100 op bindings (no GPU, no built _sm70_C).

What: every `_sm70_C::fv100_*` schema is an out op with a CUDA impl and a
  register_fake in vllm_sm70/ops/fv100.py whose parameters match the
  schema; the fakes run on meta tensors; nothing references the fork's
  pybind module, `_C` namespace or env policy; the fused one-launch decode
  (K14) and its op stay behind their #ifdef, which the build turns on; the KV-view helpers accept upstream's LBNHC
  interleave; the decode plan and workspace logic; the wrappers thread their
  arguments in schema order; the SDPA reference used by the GPU tests
  agrees with a direct softmax reference.
How: as tests/ops/test_tm_ops_cpu.py, schemas are parsed from the C++
  sources and, when `_sm70_C` is not built, defined in a Python
  torch.library so fv100.py registers its fakes against them.
"""

from __future__ import annotations

import importlib
import inspect
import pathlib
import re
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
SM70 = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "ops"))
from _schemas import sm70_schemas  # noqa: E402

CSRC = SM70 / "csrc" / "flash_v100"
FV100_PY = SM70 / "vllm_sm70" / "ops" / "fv100.py"
AREA = "flash_v100"
EXPECTED_OPS = {
    "fv100_decode_paged",
    "fv100_decode_paged_sinks",
    "fv100_decode_paged_xqa",
    "fv100_prefill_paged",
    "fv100_prefill_paged_sinks",
    "fv100_decode_paged_sinks_fused",  # K14, under FUSED_GUARD
}
FUSED_OP = "fv100_decode_paged_sinks_fused"
FUSED_GUARD = "VLLM_SM70_FV100_FUSED_DECODE"


def _code_only(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)


def _py_code_only(path: pathlib.Path) -> str:
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(path.read_text()).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return " ".join(out)


# ------------------------------------------------------------------ schemas


def test_exactly_the_lean_ops_are_defined():
    assert set(sm70_schemas(area=AREA)) == EXPECTED_OPS


@pytest.mark.parametrize("name", sorted(EXPECTED_OPS))
def test_schema_is_an_out_op(name):
    schema = sm70_schemas(area=AREA)[name]
    assert schema.startswith(
        f"{name}(Tensor(a!) out, Tensor q, Tensor k_cache, "
        "Tensor v_cache, Tensor block_table, Tensor seq_lens"
    )
    assert schema.endswith("-> ()")
    banned = ("policy", "anchor")
    if name != FUSED_OP:
        banned += ("counters", "fused")
    for word in banned:
        assert word not in schema, (name, word)


def test_fused_schema_takes_sinks_and_mutable_counters():
    schema = sm70_schemas(area=AREA)[FUSED_OP]
    sinks_schema = sm70_schemas(area=AREA)["fv100_decode_paged_sinks"]
    start = len("fv100_decode_paged_sinks")
    prefix = sinks_schema[start : sinks_schema.index(", Tensor? sinks")]
    assert schema == f"{FUSED_OP}{prefix}, Tensor sinks, Tensor(e!) counters) -> ()"


def test_every_op_has_a_cuda_impl_only():
    text = (CSRC / "fv100_ops.cpp").read_text()
    impls = {}
    for block in re.finditer(
        r"TORCH_LIBRARY_IMPL\(\s*_sm70_C\s*,\s*(\w+)\s*,\s*m\s*\)\s*\{(.*?)\n\}",
        text,
        re.S,
    ):
        for op in re.findall(r'm\.impl\("(\w+)"', block.group(2)):
            impls.setdefault(op, set()).add(block.group(1))
    assert impls == {op: {"CUDA"} for op in EXPECTED_OPS}


def test_every_op_has_a_matching_fake():
    """Fake parameter names equal the schema argument names, in order."""
    text = FV100_PY.read_text()
    fakes = set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', text))
    assert fakes == EXPECTED_OPS
    import ast

    tree = ast.parse(text)
    defs = {
        n.name: [a.arg for a in n.args.args]
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
    }
    for name, schema in sm70_schemas(area=AREA).items():
        args = schema[schema.index("(") + 1 : schema.rindex(") ->")]
        names = [a.strip().split(" ")[-1] for a in args.split(",")]
        assert defs[f"_{name}_fake"] == names, name


def test_no_fork_namespace_pybind_or_policy_left():
    bad = []
    for path in sorted(CSRC.iterdir()):
        if path.suffix not in (".cu", ".cuh", ".h", ".cpp", ".cmake"):
            continue
        code = _code_only(path.read_text())
        for pat in (
            r"PYBIND11_MODULE",
            r"\bpy::",
            r"_C::",
            r"TORCH_LIBRARY\w*\(\s*_C\b",
            r"\bwith_policy\b",
            r"\bPreparedPolicy\b",
            r"\bgetenv\b",
            r"\bflash_v100_policy\.h\b",
        ):
            if re.search(pat, code):
                bad.append(f"{path.name}: {pat}")
    py = _py_code_only(FV100_PY)
    for pat in (
        r"torch \. ops \. _C\b",
        r"flash_attn_v100_cuda",
        r"_runtime",
        r"legacy_policy",
        r"_legacy",
    ):
        if re.search(pat, py):
            bad.append(f"fv100.py: {pat}")
    assert not bad, bad


def _guarded_lines(text: str) -> list[str]:
    out, depth = [], 0
    for line in text.splitlines():
        if line.startswith(f"#ifdef {FUSED_GUARD}"):
            depth += 1
        elif line.startswith("#endif") and FUSED_GUARD in line:
            depth -= 1
        elif depth:
            out.append(line)
    assert depth == 0
    return out


def test_fused_decode_is_guarded_and_built_by_default():
    """The C4/K14 one-launch path and its op exist only under the #ifdef,
    and sources.cmake defines it unless configured OFF."""
    cmake = (CSRC / "sources.cmake").read_text()
    assert re.search(
        r"option\(VLLM_SM70_FV100_FUSED_DECODE\s+\"[^\"]*\"\s+ON\)", cmake
    )
    assert "COMPILE_DEFINITIONS VLLM_SM70_FV100_FUSED_DECODE=1" in cmake
    ops_cpp = (CSRC / "fv100_ops.cpp").read_text()
    guarded_ops = "\n".join(_guarded_lines(ops_cpp))
    assert FUSED_OP in guarded_ops and "decode_paged_sinks_fused" in guarded_ops
    unguarded = [
        line
        for line in ops_cpp.splitlines()
        if "sinks_fused" in line
        and not line.lstrip().startswith("//")
        and line not in _guarded_lines(ops_cpp)
    ]
    assert not unguarded, unguarded
    text = (CSRC / "flash_decode_paged.cu").read_text()
    guarded = set()
    depth = 0
    for line in text.splitlines():
        if line.startswith(f"#ifdef {FUSED_GUARD}"):
            depth += 1
            continue
        if line.startswith("#endif") and FUSED_GUARD in line:
            depth -= 1
            continue
        if depth:
            guarded.update(re.findall(r"\w+", line))
    assert depth == 0
    for symbol in (
        "flash_attention_decode_gqa8_d128_fused_kernel",
        "gqa8_d128_fused_ctas_per_head",
        "flash_attention_decode_paged_sinks_fused",
        "kGqaFusedMaxChunk",
    ):
        assert symbol in guarded, symbol
    unguarded = []
    depth = 0
    for line in text.splitlines():
        if line.startswith(f"#ifdef {FUSED_GUARD}"):
            depth += 1
        elif line.startswith("#endif") and FUSED_GUARD in line:
            depth -= 1
        elif not depth and "gqa8_d128_fused_kernel" in line:
            unguarded.append(line)
    assert not unguarded, unguarded


def test_sources_cmake_builds_the_lean_files_only():
    text = (CSRC / "sources.cmake").read_text()
    listed = set(re.findall(r'_ROOT\}/([\w.]+)"', text))
    assert listed == {
        "flash_decode_paged.cu",
        "fused_mha_forward_paged.cu",
        "fv100_ops.cpp",
    }
    top = (SM70 / "CMakeLists.txt").read_text()
    assert 'include("${CMAKE_CURRENT_SOURCE_DIR}/csrc/flash_v100/sources.cmake")' in top
    # Per-file nvcc options set by the area must survive K01's gencode step.
    assert "APPEND PROPERTY COMPILE_OPTIONS" in top


def test_provenance_check_passes():
    script = SM70 / "tools" / "check_flash_v100_provenance.py"
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


# -------------------------------------------------------- Python (needs torch)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def fv100():
    """fv100.py with its fakes registered (against stub schemas if needed)."""
    from vllm_sm70.ops import fv100 as _probe  # noqa: F401  (imports cleanly)

    lib = None
    if not hasattr(torch.ops._sm70_C, "fv100_decode_paged"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in sm70_schemas(area=AREA).items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.fv100 as mod

    mod = importlib.reload(mod)
    mod.clear_workspaces()
    yield mod
    mod.clear_workspaces()
    if lib is not None:
        lib._destroy()


def _meta(*shape, dtype=torch.float16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_has_op_and_available(fv100):
    assert fv100.available()
    assert not fv100.has_op("definitely_not_an_op")
    with pytest.raises(RuntimeError, match="not available"):
        fv100._op("definitely_not_an_op")


def test_fakes_return_none_on_meta(fv100):
    ops = torch.ops._sm70_C
    q = _meta(4, 16, 128)
    k = _meta(10, 16, 2, 128)
    common = (
        _meta(4, 16, 128),
        q,
        k,
        k,
        _meta(4, 3, dtype=torch.int32),
        _meta(4, dtype=torch.int32),
        _meta(4, 16, 2, 128),
        _meta(4, 16, 2, dtype=torch.float32),
        _meta(4, 16, 2, dtype=torch.float32),
        _meta(1, dtype=torch.int32),
        0.088,
        256,
        2,
        "auto",
        1.0,
        1.0,
        512,
        0,
    )
    assert ops.fv100_decode_paged(*common) is None
    assert ops.fv100_decode_paged_sinks(*common, _meta(16, dtype=torch.float32)) is None
    assert ops.fv100_decode_paged_sinks(*common, None) is None
    assert (
        ops.fv100_decode_paged_sinks_fused(
            *common, _meta(16, dtype=torch.float32), _meta(64, dtype=torch.int32)
        )
        is None
    )
    assert ops.fv100_decode_paged_xqa(*common, 0) is None
    qp = _meta(1, 16, 40, 128)
    prefill = (
        _meta(1, 16, 40, 128),
        qp,
        k,
        k,
        _meta(1, 3, dtype=torch.int32),
        _meta(1, dtype=torch.int32),
        0.088,
        "auto",
        1.0,
        1.0,
        True,
        -1,
        -1,
    )
    assert ops.fv100_prefill_paged(*prefill) is None
    assert (
        ops.fv100_prefill_paged_sinks(*prefill, _meta(16, dtype=torch.float32)) is None
    )


def test_no_fake_exemptions(fv100):
    assert fv100.NO_FAKE_OPS == frozenset()


# ------------------------------------------------------------- KV cache views


def _lbnhc(num_blocks=5, block_size=16, h_kv=2, d=128):
    storage = torch.zeros(num_blocks, block_size, h_kv, 2 * d, dtype=torch.float16)
    return storage.permute(0, 2, 1, 3)  # upstream's logical (B, H, N, 2D)


def test_split_lbnhc_views_interleave_k_and_v(fv100):
    logical = _lbnhc()
    k, v = fv100.split_lbnhc_kv_cache(logical, 128)
    assert k.shape == v.shape == (5, 16, 2, 128)
    assert k.stride() == v.stride() == (16 * 2 * 256, 2 * 256, 256, 1)
    assert v.data_ptr() - k.data_ptr() == 128 * 2
    k.fill_(1.0)
    assert v.abs().sum() == 0  # disjoint halves of each head's row
    fv100.check_kv_cache_views(k, v, head_size=128, num_kv_heads=2)


def test_check_kv_views_accepts_the_fork_layout(fv100):
    k = torch.zeros(5, 16, 2, 256, dtype=torch.float16)
    fv100.check_kv_cache_views(k, k.clone(), head_size=256)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda k: k.float(), "float16"),
        (lambda k: k[..., :64], "head size 64"),
        (lambda k: k.transpose(2, 3), "contiguous"),
        (lambda k: k[:, :, :1], "differ"),
        (lambda k: k[0], "4-D"),
    ],
)
def test_check_kv_views_rejects(fv100, mutate, match):
    k = torch.zeros(5, 16, 2, 128, dtype=torch.float16)
    with pytest.raises(ValueError, match=match):
        fv100.check_kv_cache_views(mutate(k), k)


def test_check_kv_views_rejects_unaligned_strides(fv100):
    storage = torch.zeros(5 * 16 * 2 * 132, dtype=torch.float16)
    k = storage.as_strided((5, 16, 2, 128), (16 * 2 * 132, 2 * 132, 132, 1))
    with pytest.raises(ValueError, match="multiples of 8"):
        fv100.check_kv_cache_views(k, k)


# ------------------------------------------------------- decode plan/workspace


def test_default_partition_size_follows_the_fork(fv100):
    q = torch.zeros(2, 16, 128, dtype=torch.float16)
    k = torch.zeros(4096, 16, 2, 128, dtype=torch.float16)
    table = torch.zeros(2, 4096, dtype=torch.int32)
    plan = fv100._get_decode_plan(q, k, table, max_seq_len_hint=8192)
    assert (plan.partition_size, plan.launch_num_partitions) == (256, 32)
    plan = fv100._get_decode_plan(q, k, table, max_seq_len_hint=32768)
    assert (plan.partition_size, plan.launch_num_partitions) == (1024, 32)
    plan = fv100._get_decode_plan(
        q, k, table, max_seq_len_hint=600, workspace_seq_capacity_hint=16384
    )
    assert plan.actual_num_partitions == 3 and plan.launch_num_partitions == 64
    with pytest.raises(ValueError, match="partition_size_hint"):
        fv100._get_decode_plan(q, k, table, partition_size_hint=128)


def test_workspace_grows_and_keeps_captured_buffers(fv100):
    fv100.clear_workspaces()
    q = torch.zeros(2, 16, 128, dtype=torch.float16)
    plan = fv100._DecodePlan(256, 3, 3, 3)
    tmp, mx, es, active = fv100._get_decode_workspace_for_plan(
        q, batch_capacity=2, num_heads=16, head_dim=128, plan=plan
    )
    assert tmp.shape == (2, 16, 4, 128) and mx.shape == es.shape == (2, 16, 4)
    assert int(active) == 3
    first = next(iter(fv100._decode_workspace_cache.values()))
    first.captured = True
    q8 = torch.zeros(8, 16, 128, dtype=torch.float16)
    tmp8, *_ = fv100._get_decode_workspace_for_plan(
        q8, batch_capacity=8, num_heads=16, head_dim=128, plan=plan
    )
    grown = next(iter(fv100._decode_workspace_cache.values()))
    assert tmp8.shape[0] == 8 and grown.previous is first


# ------------------------------------------------------- wrapper argument flow


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call


def test_decode_wrapper_threads_schema_order(fv100, monkeypatch):
    # The two-launch sink op; the fused one (K14) is covered in
    # test_fv100_fused_decode_cpu.py.
    monkeypatch.setenv("VLLM_SM70_FLINT_ATTN_FUSED_DECODE", "0")
    rec = _Recorder()
    monkeypatch.setattr(fv100, "_op", rec)
    fv100.clear_workspaces()
    q = torch.zeros(2, 16, 128, dtype=torch.float16)
    k = torch.zeros(8, 16, 2, 128, dtype=torch.float16)
    table = torch.zeros(2, 4, dtype=torch.int32)
    lens = torch.tensor([20, 7], dtype=torch.int32)
    sinks = torch.zeros(16, dtype=torch.float32)
    out = fv100.flash_attn_decode_paged(
        q,
        k,
        k,
        table,
        lens,
        softmax_scale=0.5,
        window_size=(512, 0),
        max_seq_len_hint=20,
        sinks=sinks,
    )
    ((name, args),) = rec.calls
    assert name == "fv100_decode_paged_sinks"
    schema = sm70_schemas(area=AREA)[name]
    names = [
        a.strip().split(" ")[-1]
        for a in schema[schema.index("(") + 1 : schema.rindex(") ->")].split(",")
    ]
    bound = dict(zip(names, args))
    assert bound["out"] is out and bound["q"] is q and bound["sinks"] is sinks
    assert (bound["softmax_scale"], bound["partition_size"]) == (0.5, 256)
    assert (bound["window_size_left"], bound["window_size_right"]) == (512, 0)
    assert bound["kv_cache_dtype"] == "auto" and bound["launch_num_partitions"] == 1
    fv100.flash_attn_decode_paged(q, k, k, table, lens)
    assert rec.calls[-1][0] == "fv100_decode_paged"
    assert len(rec.calls[-1][1]) == len(names) - 1


def test_prefill_wrapper_permutes_to_bhmd_and_back(fv100, monkeypatch):
    seen = {}

    def fake(name):
        def call(out, q, *rest):
            seen["name"], seen["q_shape"], seen["rest"] = name, q.shape, rest
            out.copy_(q * 2)

        return call

    monkeypatch.setattr(fv100, "_op", fake)
    q = torch.randn(2, 5, 4, 128).half()  # [B, M, H, D]
    k = torch.zeros(8, 16, 1, 128, dtype=torch.float16)
    out = fv100.flash_attn_prefill_paged(
        q,
        k,
        k,
        torch.zeros(2, 4, dtype=torch.int32),
        torch.tensor([20, 9], dtype=torch.int32),
        causal=False,
    )
    assert seen["name"] == "fv100_prefill_paged"
    assert seen["q_shape"] == (2, 4, 5, 128)
    assert seen["rest"][-3:] == (False, -1, -1)
    torch.testing.assert_close(out, q * 2)
    with pytest.raises(ValueError, match="num_q_heads"):
        fv100.flash_attn_prefill_paged(
            q,
            k,
            k,
            torch.zeros(2, 4, dtype=torch.int32),
            torch.tensor([20, 9], dtype=torch.int32),
            sinks=torch.zeros(5, dtype=torch.float32),
        )


def test_xqa_supported_shapes(fv100):
    k = torch.zeros(1, 16, 2, 256, dtype=torch.float16)
    for heads, ok in ((8, True), (12, True), (16, True), (10, False), (2, False)):
        q = torch.zeros(1, heads, 256, dtype=torch.float16)
        assert fv100.xqa_supported(q, k) is ok, heads
    assert not fv100.xqa_supported(torch.zeros(1, 16, 128), k[..., :128])


# --------------------------------------------- the GPU tests' SDPA reference


def _naive(q, k, v, scale, causal, window, sinks):
    m, h, _ = q.shape
    n, h_kv, _ = k.shape
    g = h // h_kv
    kk = k.float().repeat_interleave(g, dim=1)
    vv = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("mhd,nhd->hmn", q.float(), kk) * scale
    qp = torch.arange(m)[:, None] + max(n - m, 0)
    kp = torch.arange(n)[None, :]
    keep = torch.ones(m, n, dtype=torch.bool)
    if causal:
        keep &= kp <= qp
    if window[0] >= 0:
        keep &= kp >= qp - window[0]
    s = s.masked_fill(~keep, float("-inf"))
    if sinks is not None:
        s = torch.cat([s, sinks.view(h, 1, 1).expand(h, m, 1)], dim=-1)
        p = torch.softmax(s, -1)[..., :n]
    else:
        p = torch.softmax(s, -1)
    return torch.einsum("hmn,nhd->mhd", p, vv)


@pytest.mark.parametrize("window", [(-1, -1), (7, 0)])
@pytest.mark.parametrize("with_sinks", [False, True])
@pytest.mark.parametrize("m,n", [(1, 40), (6, 40), (40, 40)])
def test_sdpa_reference_matches_naive(window, with_sinks, m, n):
    sys.path.insert(0, str(HERE))
    from _fv100_ref import sdpa_reference

    gen = torch.Generator().manual_seed(m + n)
    q = torch.randn(m, 8, 16, generator=gen)
    k = torch.randn(n, 2, 16, generator=gen)
    v = torch.randn(n, 2, 16, generator=gen)
    sinks = torch.randn(8, generator=gen) * 3 if with_sinks else None
    got = sdpa_reference(q, k, v, scale=0.25, causal=True, window=window, sinks=sinks)
    want = _naive(q, k, v, 0.25, True, window, sinks)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)


def test_fake_signatures_have_no_defaults(fv100):
    for name in EXPECTED_OPS:
        fake = getattr(fv100, f"_{name}_fake")
        assert all(
            p.default is inspect.Parameter.empty
            for p in inspect.signature(fake).parameters.values()
        )
