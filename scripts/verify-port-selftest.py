#!/usr/bin/env python3
"""Self-test for scripts/verify-port.py against a checkout of the fork.

    scripts/verify-port-selftest.py <fork-checkout> [workdir]

1. Identity: several fork files compared with themselves must be all IDENTICAL
   (run with an empty rule set, so normalization cannot hide anything).
2. Mutation: a port copy is made with the expected adaptations done by an
   independent AST rewrite (not the regex rules), plus planted defects. The
   report must flag every planted defect and nothing else.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "verify-port.py"
IDENTITY = [
    "csrc/attention/sm70_79t/prefill.cu",
    "csrc/attention/sm70_grouped_long/kernel/grouped-attention.cu",
    "csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh",
    "csrc/torch_bindings.cpp",
    "vllm/_sm70/runtime.py",
    "vllm/_sm70/policy.py",
    "vllm/_sm70/linear.py",
]


def run(args, cwd=None):
    r = subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, cwd=cwd)
    if r.returncode not in (0, 1):
        sys.exit(f"verify-port failed:\n{r.stderr}")
    return r.stdout


def results(work: Path, fork, port, config=None):
    j = work / "out.json"
    args = ["--fork", *map(str, fork), "--port", *map(str, port), "--json", str(j)]
    if config:
        args += ["--config", str(config)]
    report = run(args)
    return report, json.loads(j.read_text())


class PortRewrite(ast.NodeTransformer):
    """The expected Python adaptations, written as an AST rewrite."""

    def visit_arguments(self, a):
        self.generic_visit(a)
        nd = len(a.args) - len(a.defaults)
        keep = [(i, x) for i, x in enumerate(a.args) if x.arg != "native_policy"]
        a.defaults = [a.defaults[i - nd] for i, _ in keep if i >= nd]
        a.args = [x for _, x in keep]
        kw = [(x, d) for x, d in zip(a.kwonlyargs, a.kw_defaults) if x.arg != "native_policy"]
        a.kwonlyargs, a.kw_defaults = [x for x, _ in kw], [d for _, d in kw]
        return a

    def visit_Call(self, c):
        self.generic_visit(c)
        if (isinstance(c.func, ast.Name) and c.func.id == "call_native" and len(c.args) >= 2
                and isinstance(c.args[1], ast.Name) and c.args[1].id == "native_policy"):
            return ast.Call(c.args[0], c.args[2:], c.keywords)
        c.args = [x for x in c.args if not (isinstance(x, ast.Name) and x.id == "native_policy")]
        c.keywords = [k for k in c.keywords if k.arg != "native_policy"]
        return c

    def visit_Attribute(self, n):
        self.generic_visit(n)
        if n.attr == "_C" and isinstance(n.value, ast.Attribute) and n.value.attr == "ops":
            n.attr = "_sm70_C"
        return n

    def visit_Constant(self, n):
        if isinstance(n.value, str) and n.value.startswith("_C::"):
            n.value = "_sm70_C::" + n.value[4:]
        return n


def cut_function(src: str, header_re: str) -> str:
    m = re.search(header_re, src)
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        depth += src[j] == "{"
        depth -= src[j] == "}"
        if depth == 0:
            return src[:m.start()] + src[j + 1:]
    raise ValueError(header_re)


def main():
    fork_root = Path(sys.argv[1]).resolve()
    work = Path(sys.argv[2] if len(sys.argv) > 2 else tempfile.mkdtemp(prefix="vp-selftest-"))
    work.mkdir(parents=True, exist_ok=True)
    empty = work / "empty.toml"
    empty.write_text("")
    ok = True

    print("== identity (no rules)")
    for rel in IDENTITY:
        f = fork_root / rel
        _, res = results(work, [f], [f], empty)
        bad = [r for r in res if r["status"] != "IDENTICAL"]
        print(f"  {'ok  ' if not bad else 'FAIL'} {rel}: {len(res)} entities, {len(bad)} not IDENTICAL")
        ok &= not bad

    print("== mutation")
    fd, pd = work / "fork", work / "port"
    fd.mkdir(exist_ok=True)
    pd.mkdir(exist_ok=True)
    cu = (fork_root / "csrc/attention/sm70_79t/prefill.cu").read_text()
    py = (fork_root / "vllm/_sm70/linear.py").read_text()
    (fd / "prefill.cu").write_text(cu)
    (fd / "linear.py").write_text(py)

    # Planted C++ defects.
    p = cu.replace("__launch_bounds__(256, 2) void warp_specialized_pv_kernel(",
                   "__launch_bounds__(256, 1) void warp_specialized_pv_kernel(", 1)   # LAUNCH
    p = p.replace("#define QK_STAGES 2", "#define QK_STAGES 3", 1)                      # NUMERIC macro
    p = cut_function(p, r"__device__ __forceinline__ float prefix_poly_exp2\(")         # DROPPED
    p = p.replace("__device__ __forceinline__ int pv_task_index() {",
                  "__device__   __forceinline__\nint pv_task_index()   // cosmetic\n{ /* c */", 1)  # must stay IDENTICAL
    p = p.replace("int64_t row_sum_output_index(int row)", "int64_t row_sum_out_idx(int row)", 1)  # MOVED
    p += "\n__device__ int brand_new_helper(int x) { return x * 2; }\n"                 # NEW
    (pd / "prefill.cu").write_text(p)

    # Expected Python adaptations (AST rewrite) + one planted numeric change.
    t = PortRewrite().visit(ast.parse(py))
    ast.fix_missing_locations(t)
    q = ast.unparse(t).replace("group_size: int=32", "group_size: int=64", 1)
    (pd / "linear.py").write_text(q)

    # Expected C++ binding adaptations, done with plain string edits, plus one
    # planted schema change.
    tb = (fork_root / "csrc/torch_bindings.cpp").read_text()
    (fd / "torch_bindings.cpp").write_text(tb)
    b = re.sub(r'"\s*\n\s*"', "", tb)                      # join wrapped schema strings
    b = b.replace(", str? native_policy=None", "").replace(", str? native_policy", "")
    b = re.sub(r"vllm::sm70::with_policy\(\s*(&\w+)\s*\)", r"\1", b)
    b = b.replace("TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops)", "TORCH_LIBRARY_EXPAND(_sm70_C, ops)")
    b = b.replace("vllm::", "vllm_sm70::")
    b = b.replace('"int topk, Tensor! hidden_states', '"int topk, Tensor hidden_states', 1)  # no-op if absent
    b = b.replace("int bits, int group_size=32) -> Tensor[]", "int bits, int group_size=64) -> Tensor[]", 1)
    (pd / "torch_bindings.cpp").write_text(b)
    expect_extra = {"def:gguf_affine_sm70_prepare": ("CHANGED", "NUMERIC")}

    report, res = results(work, [fd], [pd])
    (work / "mutation-report.md").write_text(report)
    st = {r["name"]: r for r in res}
    first64 = next(r["name"] for r in res if r["status"] == "CHANGED" and r["port"].startswith("linear.py"))

    expect = {
        "warp_specialized_pv_kernel": ("CHANGED", "LAUNCH"),
        "#QK_STAGES": ("CHANGED", "NUMERIC"),
        "prefix_poly_exp2": ("DROPPED", None),
        "pv_task_index": ("IDENTICAL", None),
        "row_sum_out_idx": ("MOVED", None),
        "brand_new_helper": ("NEW", None),
    }
    expect.update(expect_extra)
    for name, (status, flag) in expect.items():
        r = st.get(name)
        good = r is not None and r["status"] == status and (flag is None or flag in r.get("flags", []))
        print(f"  {'ok  ' if good else 'FAIL'} {name}: want {status}{'+' + flag if flag else ''}, "
              f"got {r and r['status']} {r and r.get('flags', '')}")
        ok &= good

    # Nothing else may be flagged, except callers of the renamed function and
    # the one planted Python change.
    allowed_changed = {"warp_specialized_pv_kernel", "#QK_STAGES", first64}
    callers = {r["name"] for r in res if r["status"] == "CHANGED" and "row_sum_output_index" in "\n".join(r["diff"])}
    extra = [r for r in res if r["status"] not in ("IDENTICAL",)
             and r["name"] not in expect and r["name"] not in allowed_changed | callers]
    py_changed = [r for r in res if r["status"] == "CHANGED" and r["port"].startswith("linear.py")]
    print(f"  {'ok  ' if len(py_changed) == 1 else 'FAIL'} python: {len(py_changed)} CHANGED "
          f"(want 1: planted group_size 32->64 in {first64})")
    print(f"  {'ok  ' if not extra else 'FAIL'} no unexpected findings ({len(extra)}); "
          f"{len(callers)} callers of the renamed function reported CHANGED")
    for r in extra[:20]:
        print(f"      unexpected: {r['status']} {r['name']}")
    ok &= len(py_changed) == 1 and not extra
    print(f"report: {work / 'mutation-report.md'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
