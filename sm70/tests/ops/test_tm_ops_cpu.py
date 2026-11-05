# SPDX-License-Identifier: Apache-2.0
"""K03 CPU checks for the TurboMind op bindings (no GPU, no built _sm70_C).

What: every `_sm70_C` schema registered by the TurboMind sources has a
  register_fake in vllm_sm70/ops/tm.py; the prepare fakes return the shapes,
  dtypes and devices the C++ ops return (meta on CPU); nothing references
  the fork's `_C` namespace or `native_policy` any more.
How: when `_sm70_C` is not built, the schemas parsed from the C++ sources
  are defined in a Python torch.library so tm.py registers its fakes against
  them; the ops are then called on meta tensors.
"""

from __future__ import annotations

import importlib
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from _schemas import registration_sources, sm70_schemas  # noqa: E402

PKG = pathlib.Path(__file__).resolve().parents[2] / "vllm_sm70"
TM_PY = PKG / "ops" / "tm.py"


def _has_tensor_arg(schema: str) -> bool:
    args = schema[schema.index("(") + 1 : schema.rindex("->")]
    return "Tensor" in args


def _needs_fake() -> dict[str, str]:
    """Ops that take a Tensor. Ops without one run eagerly (never traced),
    and torch refuses a fake on them when the C++ impl is the default
    CompositeImplicitAutograd from m.def(schema, fn) (seen on the real .so)."""
    return {
        n: s for n, s in sm70_schemas(area="turbomind").items() if _has_tensor_arg(s)
    }


def test_no_fake_for_ops_without_tensor_args():
    fakes = set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', TM_PY.read_text()))
    no_tensor = set(sm70_schemas(area="turbomind")) - set(_needs_fake())
    assert no_tensor, "expected some ops without Tensor args"
    assert not (fakes & no_tensor), sorted(fakes & no_tensor)


def test_every_schema_has_a_fake():
    fakes = set(re.findall(r'register_fake\("_sm70_C::(\w+)"\)', TM_PY.read_text()))
    schemas = _needs_fake()
    assert schemas, "no _sm70_C schemas found"
    missing = sorted(set(schemas) - fakes)
    assert not missing, f"ops without a fake: {missing}"
    extra = sorted(fakes - set(schemas))
    assert not extra, f"fakes for ops that no source defines: {extra}"


def _code_only(path: pathlib.Path) -> str:
    """Source without comments and docstrings (they may name the fork API)."""
    text = path.read_text(errors="replace")
    if path.suffix == ".py":
        import ast
        import io
        import tokenize

        doc_lines = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                body = getattr(node, "body", [])
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None), ast.Constant)
                ):
                    doc_lines.update(range(body[0].lineno, body[0].end_lineno + 1))
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT or tok.start[0] in doc_lines:
                continue
            out.append(tok.string)
        return " ".join(out)
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def test_no_fork_namespace_or_policy_left():
    bad = []
    for path in [TM_PY, *registration_sources()]:
        text = _code_only(path)
        for pat in (
            r'["\']_C::',
            r"torch\.ops\._C\b",
            r"TORCH_LIBRARY\w*\(\s*_C\b",
            r"\bnative_policy\b",
            r"\bwith_policy\b",
            r"\bcall_native\b",
        ):
            if re.search(pat, text):
                bad.append(f"{path.name}: {pat}")
    assert not bad, bad


def test_schemas_have_no_policy_arg_and_cuda_impls():
    for name, schema in sm70_schemas().items():
        assert "native_policy" not in schema, name
    # Every def that takes a tensor has a CUDA impl in the same source tree.
    impls = set()
    for path in registration_sources():
        text = path.read_text(errors="replace")
        for block in re.finditer(
            r"TORCH_LIBRARY_IMPL\(\s*_sm70_C\s*,\s*(\w+)\s*,\s*m\s*\)\s*\{(.*?)\n\}",
            text,
            re.S,
        ):
            for op in re.findall(r'm\.impl\("(\w+)"', block.group(2)):
                impls.add((op, block.group(1)))
    # K02's two workspace ops are registered with m.def(schema, &fn), a
    # catch-all kernel, which is fine for ops without tensor arguments.
    catch_all = {"sm70_gemm_configure_workspace", "sm70_gemm_workspace_storage"}
    for name, schema in sm70_schemas().items():
        if "Tensor" in schema.split("->", 1)[0]:
            assert (name, "CUDA") in impls, f"{name}: no CUDA impl"
        else:
            assert (name, "CompositeExplicitAutograd") in impls or name in catch_all, (
                f"{name}: an op without tensor args needs a "
                "CompositeExplicitAutograd impl"
            )


# ---------------------------------------------------------------- fake shapes

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def tm():
    """tm.py with its fakes registered (against stub schemas if needed)."""
    from vllm_sm70.ops import tm as _tm_probe  # noqa: F401  (imports cleanly)

    lib = None
    if not hasattr(torch.ops._sm70_C, "nvfp4_sm70_prepare"):
        try:
            lib = torch.library.Library("_sm70_C", "DEF")
        except RuntimeError:
            lib = torch.library.Library("_sm70_C", "FRAGMENT")
        for name, schema in sm70_schemas().items():
            if not hasattr(torch.ops._sm70_C, name):
                lib.define(schema)
    import vllm_sm70.ops.tm as tm_mod

    tm_mod = importlib.reload(tm_mod)
    yield tm_mod
    if lib is not None:
        lib._destroy()


def _meta(*shape, dtype):
    return torch.empty(*shape, dtype=dtype, device="meta")


def test_has_op(tm):
    assert tm.has_op("nvfp4_sm70_prepare")
    assert not tm.has_op("definitely_not_an_op")
    with pytest.raises(RuntimeError, match="not available"):
        tm._op("definitely_not_an_op")


def test_nvfp4_prepare_fake_contract(tm):
    k, n = 512, 256
    w, s, meta = torch.ops._sm70_C.nvfp4_sm70_prepare(
        _meta(k, n, dtype=torch.uint8), _meta(k // 16, n, dtype=torch.float16)
    )
    assert (w.shape, w.dtype) == ((k, n // 8), torch.int32)
    assert (s.shape, s.dtype) == ((k // 16, n), torch.float16)
    assert (meta.shape, meta.dtype, meta.device.type) == ((2,), torch.int64, "cpu")


def test_fp8_prepare_fake(tm):
    n, k = 256, 512
    w, s, meta = torch.ops._sm70_C.fp8_sm70_prepare(
        _meta(n, k, dtype=torch.float8_e4m3fn),
        _meta(n // 128, k // 128, dtype=torch.float32),
        128,
        False,
    )
    assert (w.shape, w.dtype) == ((k, n), torch.uint8)
    assert s.dtype == torch.float16 and s.shape[1] == n
    assert (meta.shape, meta.dtype, meta.device.type) == ((2,), torch.int64, "cpu")


def test_f16_prepare_fake(tm):
    w, meta = torch.ops._sm70_C.sm70_f16_prepare(_meta(256, 512, dtype=torch.float16))
    assert (w.shape, w.dtype) == ((256, 512), torch.float16)
    assert (meta.shape, meta.dtype, meta.device.type) == ((1,), torch.int64, "cpu")


def test_strided_ptrs_fake(tm):
    ptrs = torch.ops._sm70_C.awq_moe_build_strided_ptrs(
        _meta(3, 512, 32, dtype=torch.int32),
        _meta(3, 32, 256, dtype=torch.float16),
        1,
        1,
        3,
    )
    assert [p.shape for p in ptrs] == [(48,), (48,)]


def test_out_op_fake_returns_none(tm):
    out = _meta(4, 256, dtype=torch.float16)
    x = _meta(4, 512, dtype=torch.float16)
    assert (
        torch.ops._sm70_C.nvfp4_gemm_sm70_out(
            out,
            x,
            _meta(512, 32, dtype=torch.int32),
            _meta(32, 256, dtype=torch.float16),
            16,
            1,
            1,
        )
        is None
    )


def test_register_without_extension_is_a_noop(monkeypatch):
    import vllm_sm70.ops as ops_area
    from vllm_sm70 import _ext

    monkeypatch.setattr(ops_area, "_registered", False)
    monkeypatch.setattr(_ext, "is_loaded", lambda: False)
    ops_area.register()  # must not raise or import tm against a missing .so
    assert ops_area._registered


def test_fake_signatures_match_schemas(tm):
    """Each fake takes the schema's arguments, with the schema's defaults."""
    import inspect

    problems = []
    for name, schema in _needs_fake().items():
        if not tm.has_op(name):
            # Not in this build (e.g. AWQ ops when the real .so is built
            # without VLLM_SM70_TM_AWQ): tm.py defines no fake for it.
            continue
        fake = getattr(tm, f"_{name}_fake", None)
        if fake is None:
            problems.append(f"{name}: no _{name}_fake in tm.py")
            continue
        args = torch._C.parse_schema(schema).arguments
        params = list(inspect.signature(fake).parameters.values())
        if len(params) != len(args):
            problems.append(f"{name}: {len(params)} params vs {len(args)} schema args")
            continue
        for p, a in zip(params, args):
            if a.has_default_value() and p.default is inspect.Parameter.empty:
                problems.append(f"{name}.{p.name}: schema has a default, fake has not")
    assert not problems, problems
