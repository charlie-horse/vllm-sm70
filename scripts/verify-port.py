#!/usr/bin/env python3
"""Compare ported code against the fork, function by function.

    scripts/verify-port.py --port sm70/csrc/foo.cu --fork ../vLLM/csrc/foo.cu
    scripts/verify-port.py --port sm70/ --fork ../vLLM/csrc ../vLLM/vllm/sm70
    scripts/verify-port.py --diff pr.diff --port-root . --fork ../vLLM

Python files are split with `ast` (top-level functions, classes, methods,
module constants). C++/CUDA files are split with a comment-aware brace
matcher (functions, kernels, methods, structs/classes, #define macros,
top-level constexpr/const values).

Before comparison the fork text is rewritten with the rules in
scripts/verify-port.toml (the expected renames), then both sides have
comments and whitespace normalized. Each entity is reported as:

  IDENTICAL  same body after normalization
  CHANGED    body differs; a unified diff follows, with heuristic flags
             (NUMERIC, LAUNCH, FP16, CONTROL) to steer review
  MOVED      same body under a different name or file
  NEW        in the port only (NEW(expected) if it matches expected_new)
  DROPPED    in the fork only (DROPPED(expected) if it matches expected_dropped)

Exit status: 0 normally; with --strict, 1 if anything is CHANGED, NEW or
DROPPED without an "expected" tag.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PY_EXT = {".py", ".pyi"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx", ".inl"}
CPP_KEYWORDS_NOT_FUNCS = {
    "if", "for", "while", "switch", "catch", "return", "sizeof", "decltype",
    "alignas", "alignof", "static_assert", "__launch_bounds__", "defined",
    "noexcept", "requires", "do", "else", "try", "operator",
}


def lang_of(path: Path) -> str | None:
    if path.suffix in PY_EXT:
        return "py"
    if path.suffix in CPP_EXT:
        return "cpp"
    return None


# ---------------------------------------------------------------- config

@dataclass
class Rule:
    name: str
    lang: str
    side: str
    regex: re.Pattern
    repl: str
    exclude_paths: tuple = ()     # fnmatch globs on the file path; rule skipped there


@dataclass
class Config:
    rules: list[Rule]
    expected_dropped: list[re.Pattern]
    expected_new: list[re.Pattern]
    ignore_docstrings: bool = True

    @classmethod
    def load(cls, path: Path) -> "Config":
        data = tomllib.loads(path.read_text())
        rules = []
        for r in data.get("rule", []):
            flags = re.MULTILINE | (re.DOTALL if r.get("dotall") else 0)
            rules.append(Rule(r["name"], r.get("lang", "all"), r.get("side", "fork"),
                              re.compile(r["pattern"], flags), r.get("repl", ""),
                              tuple(r.get("exclude_paths", []))))
        pats = lambda k: [re.compile(p) for p in data.get(k, {}).get("patterns", [])]
        return cls(rules, pats("expected_dropped"), pats("expected_new"),
                   data.get("options", {}).get("ignore_docstrings", True))

    def apply(self, text: str, lang: str, side: str, path: str = "") -> tuple[str, list[str]]:
        hits = []
        if lang == "cpp":
            # "a, str? " "native_policy=None" -> "a, str? native_policy=None", so
            # schema rules and comparisons do not depend on how strings wrap.
            n = 1
            while n:
                text, n = _ADJ_STR.subn(lambda m: m.group(1) + m.group(2), text)
        for r in self.rules:
            if r.lang not in ("all", lang) or r.side not in ("both", side):
                continue
            if path and any(fnmatch.fnmatch(path, g) for g in r.exclude_paths):
                hits.append(f"{r.name} SKIPPED (exclude_paths)")
                continue
            text, n = r.regex.subn(r.repl, text)
            if n:
                hits.append(f"{r.name} x{n}")
        return text, hits


_ADJ_STR = re.compile(r'((?<!\\)"(?:\\.|[^"\\\n])*)"\s*\n?\s*"((?:\\.|[^"\\\n])*")')


# ---------------------------------------------------------------- entities

@dataclass
class Entity:
    key: str          # qualified name, unique within a tree
    kind: str         # func, kernel, class, method, macro, const
    file: str
    line: int         # 1-based line in the (rewritten) source
    body: str         # normalized text used for comparison
    raw: str          # original text, for diffs
    sig: str = ""     # short signature for display


def dedupe(ents: list[Entity]) -> list[Entity]:
    seen: dict[str, int] = {}
    for e in ents:
        if e.key in seen:
            seen[e.key] += 1
            e.key = f"{e.key}#{seen[e.key]}"
        else:
            seen[e.key] = 1
    return ents


# ---------------------------------------------------------------- python

class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
        return node

    visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = visit_Module = _strip


def py_entities(src: str, fname: str, strip_doc: bool) -> list[Entity]:
    tree = ast.parse(src, filename=fname)
    out: list[Entity] = []

    def norm(node) -> str:
        import copy
        n = copy.deepcopy(node)
        if strip_doc:
            n = _StripDocstrings().visit(n)
        if isinstance(n, ast.ClassDef):
            # A class body is compared without its methods; methods are
            # their own entities.
            n.body = [b for b in n.body if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))] or [ast.Pass()]
        return ast.unparse(n)

    def raw(node) -> str:
        seg = ast.get_source_segment(src, node, padded=True) or ""
        deco = "".join((ast.get_source_segment(src, d) or "") and "@" + ast.get_source_segment(src, d) + "\n"
                       for d in getattr(node, "decorator_list", []))
        return deco + seg

    def walk(body, prefix: str, in_class: bool):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = prefix + node.name
                out.append(Entity(key, "method" if in_class else "func", fname, node.lineno,
                                  norm(node), raw(node), f"def {node.name}({ast.unparse(node.args)})"))
            elif isinstance(node, ast.ClassDef):
                key = prefix + node.name
                out.append(Entity(key, "class", fname, node.lineno, norm(node), raw(node),
                                  f"class {node.name}"))
                walk(node.body, key + ".", True)
            elif isinstance(node, (ast.If, ast.Try, ast.With, ast.AsyncWith)) or type(node).__name__ == "TryStar":
                # e.g. `if hasattr(torch.ops._C, "x"): @register_fake(...) def ...`
                walk(node.body, prefix, in_class)
                walk(getattr(node, "orelse", []), prefix, in_class)
                for h in getattr(node, "handlers", []):
                    walk(h.body, prefix, in_class)
                walk(getattr(node, "finalbody", []), prefix, in_class)
            elif not in_class and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [t.id for t in targets if isinstance(t, ast.Name)]
                if names and all(n.isupper() or n.startswith("_") and n[1:].isupper() for n in names):
                    for n in names:
                        out.append(Entity(prefix + n, "const", fname, node.lineno,
                                          ast.unparse(node), raw(node), n))
    walk(tree.body, "", False)
    # Module imports/re-exports, so a dropped export shows up.
    names = sorted({f"{getattr(n, 'module', '') or ''}:{a.name}" + (f" as {a.asname}" if a.asname else "")
                    for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names})
    if names:
        out.append(Entity(f"imports({Path(fname).stem})", "imports", fname, 1, "\n".join(names),
                          "\n".join(names), f"{len(names)} names"))
    return dedupe(out)


# ---------------------------------------------------------------- c++ / cuda

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<lc>//[^\n]*)
  | (?P<bc>/\*.*?\*/)
  | (?P<str>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
  | (?P<rstr>R"(?P<delim>[^(\s]*)\(.*?\)(?P=delim)")
  | (?P<num>\.?\d(?:[\w.]|[eEpP][+-])*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op><<<|>>>|::|->|\+\+|--|<<=|>>=|<=|>=|==|!=|&&|\|\||[-+*/%&|^!~<>=?:;,.(){}\[\]#\\])
""", re.VERBOSE | re.DOTALL)


def strip_cpp_comments(src: str) -> str:
    """Replace comments with spaces (keeping newlines) so offsets and lines stay put."""
    out = []
    for m in _TOKEN.finditer(src):
        if m.lastgroup in ("lc", "bc"):
            out.append(re.sub(r"[^\n]", " ", m.group()))
        else:
            out.append(m.group())
    return "".join(out)


def cpp_norm(text: str) -> str:
    toks: list[str] = []
    for m in _TOKEN.finditer(text):
        if m.lastgroup in ("ws", "lc", "bc"):
            continue
        t = m.group()
        if m.lastgroup == "str" and toks and toks[-1].startswith('"') and t.startswith('"'):
            toks[-1] = toks[-1][:-1] + t[1:]      # "a" "b" -> "ab"
        else:
            toks.append(t)
    return " ".join(toks)


def _blank(text: str, a: int, b: int) -> str:
    return text[:a] + re.sub(r"[^\n]", " ", text[a:b]) + text[b:]


def _match_brace(text: str, i: int) -> int:
    """text[i] == '{'; return index just past the matching '}'. Skips strings/chars."""
    depth = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            q = c
            i += 1
            while i < n and text[i] != q:
                i += 2 if text[i] == "\\" else 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


_OPREG = re.compile(r'\b\w+\s*\.\s*(?:(def)|(impl))\s*\(\s*"(\w+(?:\.\w+)?)(?:\\.|[^"\\])*"(?:[^;"]|"(?:\\.|[^"\\])*")*;')
_MACRO = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)((?:\([^)]*\))?)(?:[^\n\\]|\\.|\\\n)*", re.MULTILINE)
_PP = re.compile(r"^[ \t]*#(?:[^\n\\]|\\.|\\\n)*", re.MULTILINE)
_SCOPE_HDR = re.compile(r"\b(namespace|struct|class|union|enum(?:\s+class|\s+struct)?)\b\s*((?:[\w:]+\s*)?)(?:final\b)?[^;{()]*$", re.DOTALL)
_CONST = re.compile(r"(?:^|[;}{])\s*((?:template\s*<[^;{}]*>\s*)?(?:(?:static|inline|constexpr|const|__constant__|__device__|__host__|extern)\s+)+[\w:<>, *&]+?\b(\w+)\s*(?:\[[^\]]*\]\s*)*=\s*[^;{}]*(?:\{[^;{}]*\}[^;{}]*)?;)")


def _one_branch(text: str, last: bool = False) -> str:
    """Keep one branch of every #if chain and blank the rest, keeping offsets.

    Alternative branches often repeat a function header with its own '{'
    (one signature per #if), which would unbalance brace matching. Structure
    is found twice, on the first and on the last branch of every chain, so a
    definition that lives only in an #else is still seen. Bodies are always
    compared on the full text, so no branch's code is ignored.
    """
    lines = text.splitlines(keepends=True)
    path: list[tuple] = []          # per line: ((chain, branch), ...) of enclosing chains
    is_pp: list[bool] = []
    count: dict[int, int] = {}
    stack: list[list[int]] = []     # [chain id, branch index]
    nxt = 0
    cont = False
    for line in lines:
        pp = cont or line.lstrip().startswith("#")
        if pp and not cont:
            m = re.match(r"\s*#\s*(\w+)", line)
            d = m.group(1) if m else ""
            if d in ("if", "ifdef", "ifndef"):
                stack.append([nxt, 0])
                count[nxt] = 1
                nxt += 1
            elif d in ("elif", "else") and stack:
                stack[-1][1] += 1
                count[stack[-1][0]] += 1
            elif d == "endif" and stack:
                stack.pop()
        cont = pp and line.rstrip("\r\n").endswith("\\")
        is_pp.append(pp)
        path.append(tuple((c, b) for c, b in stack))
    out = []
    for line, pp, pth in zip(lines, is_pp, path):
        keep = not pp and all(b == (count[c] - 1 if last else 0) for c, b in pth)
        out.append(line if keep else re.sub(r"[^\n]", " ", line))
    return "".join(out)


def cpp_entities(src: str, fname: str) -> list[Entity]:
    out: list[Entity] = []
    lines_at = lambda off: src.count("\n", 0, off) + 1

    full = strip_cpp_comments(src)          # bodies are compared on this
    for m in _MACRO.finditer(full):
        out.append(Entity("#" + m.group(1), "macro", fname, lines_at(m.start()),
                          cpp_norm(m.group().replace("\\\n", " ")), src[m.start():m.end()],
                          f"#define {m.group(1)}{m.group(2)}"))
    spans: dict[int, tuple[int, int]] = {}  # id(entity) -> (start, end)

    def scan(lo: int, hi: int, prefix: str, in_record: bool):
        i = lo
        stmt_start = lo
        while i < hi:
            c = text[i]
            if c in "\"'":
                q = c
                i += 1
                while i < hi and text[i] != q:
                    i += 2 if text[i] == "\\" else 1
                i += 1
                continue
            if c == ";":
                hdr = text[stmt_start:i + 1]
                cm = _CONST.search(" " + hdr)
                if cm:
                    cm = _CONST.search(" " + full[stmt_start:i + 1]) or cm
                if cm and "(" not in cm.group(1).split("=")[0]:
                    name = cm.group(2)
                    out.append(Entity(prefix + name, "const", fname, lines_at(stmt_start + len(hdr) - len(hdr.lstrip())),
                                      cpp_norm(cm.group(1)), hdr.strip(), name))
                stmt_start = i + 1
            elif c == "}":
                stmt_start = i + 1
            elif c == "{":
                end = _match_brace(text, i)
                hdr = text[stmt_start:i]
                hdr_s = hdr.strip()
                body_start = stmt_start + (len(hdr) - len(hdr.lstrip()))
                sm = _SCOPE_HDR.search(hdr_s)
                reg = re.match(r"^(TORCH_LIBRARY\w*|PYBIND11_MODULE)\s*\(", hdr_s)
                fm = None if reg else _func_name(hdr_s)
                if fm:
                    name = fm
                    kind = "kernel" if "__global__" in hdr_s else ("method" if in_record or "::" in name else "func")
                    e = Entity(prefix + name, kind, fname, lines_at(body_start),
                               cpp_norm(full[body_start:end]), src[body_start:end],
                               cpp_norm(hdr_s)[:160])
                    spans[id(e)] = (body_start, end)
                    out.append(e)
                elif sm:
                    what, name = sm.group(1), sm.group(2).strip()
                    if what == "namespace":
                        scan(i + 1, end - 1, prefix + (name + "::" if name else ""), False)
                    else:
                        if name:
                            key = prefix + name
                            # Compare the record without its member function bodies.
                            inner_ents_before = len(out)
                            scan(i + 1, end - 1, key + "::", True)
                            body = full[:end]
                            for e in sorted(out[inner_ents_before:], key=lambda e: -spans.get(id(e), (0, 0))[0]):
                                if id(e) in spans and e.kind in ("method", "func", "kernel"):
                                    a, b = spans[id(e)]
                                    body = body[:a] + " " + body[b:]
                            body = body[body_start:]
                            out.append(Entity(key, "class", fname, lines_at(body_start),
                                              cpp_norm(body), src[body_start:end], cpp_norm(hdr_s)[:160]))
                elif re.search(r'extern\s*"C(\+\+)?"\s*$', hdr_s):
                    scan(i + 1, end - 1, prefix, False)
                elif reg:
                    key = re.sub(r"\s+", "", hdr_s)
                    # Each def/impl is its own entity; the block keeps the rest.
                    out.append(Entity(prefix + key, "registration", fname, lines_at(body_start),
                                      cpp_norm(_OPREG.sub(" ", full[body_start:end])), src[body_start:end], key))
                    for om in _OPREG.finditer(full, i + 1, end - 1):
                        out.append(Entity(f"{'def' if om.group(1) else 'impl'}:{om.group(3)}", "op" + ("def" if om.group(1) else "impl"), fname,
                                          lines_at(om.start()), cpp_norm(om.group()), om.group(),
                                          om.group(3)))
                i = end
                stmt_start = end
                continue
            i += 1

    seen: set[tuple[str, int]] = set()
    merged: list[Entity] = [e for e in out]   # macros
    for last in (False, True):
        text = _one_branch(full, last)
        out = []
        scan(0, len(text), "", False)
        for e in out:
            k = (e.key, e.line, e.kind)
            if k not in seen:
                seen.add(k)
                merged.append(e)
    return dedupe(merged)


def _func_name(hdr: str) -> str | None:
    """Return the function name if `hdr` (text before '{') is a function definition."""
    if not hdr or hdr.endswith(("=", ",", "(", "return")) or re.search(r"\b(namespace|enum)\b", hdr):
        return None
    # strip trailing qualifiers / ctor init list / trailing return type
    h = hdr
    depth = 0
    close = -1
    # find the ')' that closes the parameter list: last top-level ')' before
    # any ':' initializer list or '->' trailing return
    for idx, ch in enumerate(h):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0 and ch == ")":
                close = idx
                tail = h[idx + 1:]
                if re.match(r"^\s*(const|override|final|noexcept(\([^)]*\))?|volatile|&|&&|mutable|\s)*"
                            r"(->[^{]*|:\s*\w[^{]*|__launch_bounds__\([^)]*\)|)\s*$", tail, re.DOTALL):
                    break
                close = -1
    if close < 0:
        return None
    depth = 0
    for j in range(close, -1, -1):
        if h[j] == ")":
            depth += 1
        elif h[j] == "(":
            depth -= 1
            if depth == 0:
                open_ = j
                break
    else:
        return None
    m = re.search(r"((?:[\w~]+(?:<[^<>()]*>)?::)*(?:operator\s*\S+?|~?\w+))\s*(?:<[^()]*>)?\s*$", h[:open_])
    if not m:
        return None
    name = re.sub(r"\s+", "", m.group(1))
    if name.split("::")[-1] in CPP_KEYWORDS_NOT_FUNCS:
        return None
    if re.search(r"[=]\s*$", h[:m.start()]) or re.search(r"\b(struct|class|union)\b[^()]*$", h[:m.start()]):
        return None
    return name


# ---------------------------------------------------------------- trees

@dataclass
class Tree:
    side: str
    entities: dict[str, Entity] = field(default_factory=dict)
    files: list[str] = field(default_factory=list)
    rule_hits: dict[str, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    base: str = "."


def collect(paths: list[Path], side: str, cfg: Config, keyed_by_file: bool) -> Tree:
    t = Tree(side)
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file() and lang_of(f))
        elif p.is_file():
            files.append(p)
        else:
            t.errors.append(f"missing: {p}")
    import os
    roots = [str(p.resolve()) for p in paths if p.exists()]
    base = os.path.commonpath(roots) if roots else "."
    if Path(base).is_file():
        base = str(Path(base).parent)
    for f in files:
        lang = lang_of(f)
        if not lang:
            continue
        disp = os.path.relpath(f.resolve(), base)
        try:
            src = f.read_text(errors="replace")
        except OSError as e:
            t.errors.append(f"{f}: {e}")
            continue
        src, hits = cfg.apply(src, lang, side, Path(f).resolve().as_posix())
        if hits:
            t.rule_hits[str(f)] = hits
        try:
            ents = py_entities(src, disp, cfg.ignore_docstrings) if lang == "py" else cpp_entities(src, disp)
        except SyntaxError as e:
            t.errors.append(f"{f}: parse error: {e}")
            continue
        t.files.append(str(f))
        t.base = base
        for e in ents:
            key = f"{Path(e.file).name}:{e.key}" if keyed_by_file else e.key
            if key in t.entities:  # same name in two files
                n = 2
                while f"{key}@{n}" in t.entities:
                    n += 1
                key = f"{key}@{n}"
            e.key = key
            t.entities[key] = e
    return t


# ---------------------------------------------------------------- compare

_NUM = re.compile(r"(?<![\w.])(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)[fFuUlLhH]*")
FP16_WORDS = re.compile(r"\b(__half2?|half2?|__float2half\w*|__half2float|__hmul2?|__hfma2?|__hadd2?|"
                        r"h2exp|hexp|scale\w*|inv_sqrt\w*|rsqrt\w*|softmax_scale|fp16|float16|bfloat16|"
                        r"HALF_MAX|65504|__shfl\w*)\b")
LAUNCH = re.compile(r"<<<.*?>>>|__launch_bounds__\s*\([^)]*\)|\bdim3\s*\w*\s*\([^)]*\)|\bdim3\b|\bgridDim\b|\bblockDim\b|__launch_bounds__|num_warps\s*=\s*\w+|num_stages\s*=\s*\w+|num_warps|num_stages|BLOCK_\w+(?:\s*=\s*\w+)?|"
                    r"\bgrid\s*=|\bblock\s*=|threads?_per_block|WARPS?_\w*|kNThreads|kBlock\w*|__syncthreads")
CONTROL = re.compile(r"\b(if|else|for|while|return|break|continue|switch|case|assert|raise|TORCH_CHECK)\b")


def flags_for(a: str, b: str) -> list[str]:
    fl = []
    if sorted(_NUM.findall(a)) != sorted(_NUM.findall(b)):
        fl.append("NUMERIC")
    if sorted(LAUNCH.findall(a)) != sorted(LAUNCH.findall(b)):
        fl.append("LAUNCH")
    if sorted(FP16_WORDS.findall(a)) != sorted(FP16_WORDS.findall(b)):
        fl.append("FP16")
    if sorted(CONTROL.findall(a)) != sorted(CONTROL.findall(b)):
        fl.append("CONTROL")
    return fl


def body_diff(fe: Entity, pe: Entity, ctx: int) -> list[str]:
    # Diff the normalized form one token-statement per line so whitespace
    # and comment noise never shows up.
    def lines(e: Entity):
        if e.file.endswith(tuple(PY_EXT)):
            return e.body.splitlines()
        return [s.strip() for s in re.split(r"(?<=[;{}])\s", e.body) if s.strip()]
    return list(difflib.unified_diff(lines(fe), lines(pe), f"fork/{fe.file}:{fe.line}",
                                     f"port/{pe.file}:{pe.line}", n=ctx, lineterm=""))


def compare(fork: Tree, port: Tree, cfg: Config, ctx: int) -> list[dict]:
    res = []
    matched_f, matched_p = set(), set()

    # Group by (name without file prefix / overload suffix, kind). Inside a
    # group, pair identical bodies first, then the most similar bodies, so
    # overloads and same-named helpers in different files pair correctly.
    groups: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for k, e in fork.entities.items():
        groups.setdefault((_name(k), e.kind), ([], []))[0].append(k)
    for k, e in port.entities.items():
        groups.setdefault((_name(k), e.kind), ([], []))[1].append(k)

    pairs: list[tuple[str, str]] = []
    for (_, _), (fks, pks) in groups.items():
        if not fks or not pks:
            continue
        if len(fks) == 1 and len(pks) == 1:
            pairs.append((fks[0], pks[0]))
            continue
        fl, pl = list(fks), list(pks)
        for fk in list(fl):
            for pk in pl:
                if fork.entities[fk].body == port.entities[pk].body:
                    pairs.append((fk, pk))
                    fl.remove(fk)
                    pl.remove(pk)
                    break
        scored = []
        for fk in fl:
            for pk in pl:
                sm = difflib.SequenceMatcher(None, fork.entities[fk].body, port.entities[pk].body, autojunk=False)
                same_file = Path(fork.entities[fk].file).name == Path(port.entities[pk].file).name
                scored.append((sm.quick_ratio() if len(fl) * len(pl) > 400 else sm.ratio(), same_file, fk, pk))
        scored.sort(reverse=True)
        used_f, used_p = set(), set()
        for _, _, fk, pk in scored:
            if fk not in used_f and pk not in used_p:
                used_f.add(fk)
                used_p.add(pk)
                pairs.append((fk, pk))

    for fk, pk in pairs:
        fe, pe = fork.entities[fk], port.entities[pk]
        matched_f.add(fk)
        matched_p.add(pk)
        r = dict(name=_bare(pk), kind=pe.kind, fork=f"{fe.file}:{fe.line}", port=f"{pe.file}:{pe.line}")
        if fe.body == pe.body:
            r["status"] = "IDENTICAL"
        else:
            r["status"] = "CHANGED"
            r["flags"] = flags_for(fe.body, pe.body)
            r["similarity"] = round(difflib.SequenceMatcher(None, fe.body, pe.body, autojunk=False).ratio(), 3)
            r["diff"] = body_diff(fe, pe, ctx)
        res.append(r)

    # Body-identical but renamed/moved.
    drop = {k: e for k, e in fork.entities.items() if k not in matched_f}
    new = {k: e for k, e in port.entities.items() if k not in matched_p}
    def anon(e: Entity) -> str:
        last = re.split(r"::|\.", _bare(e.key).split("#")[0])[-1]
        return re.sub(r"(?<![\w])" + re.escape(last) + r"(?![\w])", "<NAME>", e.body)
    by_body: dict[str, list[str]] = {}
    movable = ("func", "kernel", "method", "class", "macro")   # consts/imports are too generic
    for k, e in drop.items():
        if len(e.body) > 40 and e.kind in movable and not _trivial_record(e):
            by_body.setdefault(anon(e), []).append(k)
    for pk, pe in list(new.items()):
        ks = by_body.get(anon(pe)) if pe.kind in movable and not _trivial_record(pe) else None
        if ks:
            fk = ks.pop(0)
            fe = drop.pop(fk)
            new.pop(pk)
            res.append(dict(name=_bare(pk), kind=pe.kind, status="MOVED", was=_bare(fk),
                            fork=f"{fe.file}:{fe.line}", port=f"{pe.file}:{pe.line}"))

    for k, e in drop.items():
        exp = any(p.search(_bare(k)) for p in cfg.expected_dropped)
        res.append(dict(name=_bare(k), kind=e.kind, status="DROPPED(expected)" if exp else "DROPPED",
                        fork=f"{e.file}:{e.line}"))
    for k, e in new.items():
        exp = any(p.search(_bare(k)) for p in cfg.expected_new)
        res.append(dict(name=_bare(k), kind=e.kind, status="NEW(expected)" if exp else "NEW",
                        port=f"{e.file}:{e.line}"))
    order = {"CHANGED": 0, "DROPPED": 1, "NEW": 2, "MOVED": 3, "DROPPED(expected)": 4,
             "NEW(expected)": 5, "IDENTICAL": 6}
    res.sort(key=lambda r: (order[r["status"]], r.get("port") or r.get("fork") or "", r["name"]))
    return res


def _trivial_record(e: Entity) -> bool:
    """A class whose body, without its methods, is only a header (and pass).

    Such bodies match each other by shape alone, so pairing them as MOVED
    would be noise (seen on PR M5: FlintSm70SparseMoeBlock "was" FlintAttention).
    """
    if e.kind != "class":
        return False
    lines = [l.strip() for l in e.body.splitlines() if l.strip()]
    if e.file.endswith(tuple(PY_EXT)):
        return all(l.startswith("class ") or l == "pass" for l in lines)
    return e.body.count(";") <= 1


def _name(k: str) -> str:
    return re.sub(r"#\d+$", "", _bare(k))


def _bare(k: str) -> str:
    # strip "file.cu:" prefix and "@n" collision suffix
    k = re.sub(r"@\d+$", "", k)
    m = re.match(r"^[^:]+\.(?:py|pyi|c|cc|cpp|cxx|cu|cuh|h|hh|hpp|hxx|inl):(?!:)", k)
    return k[m.end():] if m else k


# ---------------------------------------------------------------- diff input

def files_from_diff(diff: str, root: Path) -> list[Path]:
    out = []
    for m in re.finditer(r"^\+\+\+ b/(.+)$", diff, re.MULTILINE):
        p = root / m.group(1).strip()
        if lang_of(p) and p.is_file():
            out.append(p)
    return out


# ---------------------------------------------------------------- report

def report(res: list[dict], fork: Tree, port: Tree, show_identical: bool, out) -> None:
    from collections import Counter
    c = Counter(r["status"] for r in res)
    w = out.write
    w("# verify-port report\n\n")
    w(f"fork: {len(fork.files)} files under {fork.base}\nport: {len(port.files)} files under {port.base}\n")
    w("summary: " + "  ".join(f"{k}={c[k]}" for k in
      ["IDENTICAL", "CHANGED", "MOVED", "NEW", "NEW(expected)", "DROPPED", "DROPPED(expected)"] if c[k]) + "\n")
    for t in (fork, port):
        for e in t.errors:
            w(f"ERROR ({t.side}): {e}\n")
    if fork.rule_hits:
        w("\nnormalization rules applied to fork:\n")
        for f, hits in fork.rule_hits.items():
            w(f"  {Path(f).name}: {'; '.join(hits)}\n")
    w("\n| status | kind | name | fork | port | flags |\n|---|---|---|---|---|---|\n")
    for r in res:
        if r["status"] == "IDENTICAL" and not show_identical:
            continue
        name = r["name"] + (f" (was {r['was']})" if r.get("was") else "")
        fl = ",".join(r.get("flags", []))
        if "similarity" in r:
            fl = (fl + " " if fl else "") + f"sim={r['similarity']}"
        w(f"| {r['status']} | {r['kind']} | `{name}` | {r.get('fork', '')} | {r.get('port', '')} | {fl} |\n")
    if not show_identical and c["IDENTICAL"]:
        w(f"\n({c['IDENTICAL']} IDENTICAL rows hidden; pass --show-identical)\n")
    for r in res:
        if r["status"] == "CHANGED":
            w(f"\n## CHANGED {r['kind']} `{r['name']}`  [{','.join(r['flags']) or 'no flags'}]\n")
            w("```diff\n" + "\n".join(r["diff"]) + "\n```\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fork", nargs="+", type=Path, required=True, help="fork file(s) or dir(s)")
    ap.add_argument("--port", nargs="*", type=Path, default=[], help="ported file(s) or dir(s)")
    ap.add_argument("--diff", type=Path, help="PR diff; ported files = files it adds/changes under --port-root")
    ap.add_argument("--port-root", type=Path, default=Path("."))
    ap.add_argument("--config", type=Path, default=Path(__file__).with_suffix(".toml"))
    ap.add_argument("--by-file", action="store_true", help="pair entities only within same-named files")
    ap.add_argument("--show-identical", action="store_true")
    ap.add_argument("--context", type=int, default=3)
    ap.add_argument("--json", type=Path, help="also write machine-readable results here")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--list", action="store_true", help="just list the entities found in --fork and --port")
    a = ap.parse_args(argv)

    cfg = Config.load(a.config)
    port_paths = list(a.port)
    if a.diff:
        port_paths += files_from_diff(a.diff.read_text(errors="replace"), a.port_root)
    if not port_paths:
        ap.error("give --port and/or --diff")
    fork = collect(a.fork, "fork", cfg, a.by_file)
    port = collect(port_paths, "port", cfg, a.by_file)
    if a.list:
        for t in (fork, port):
            for e in t.entities.values():
                print(f"{t.side}\t{e.kind}\t{e.key}\t{e.file}:{e.line}\t{e.sig}")
            for err in t.errors:
                print(f"{t.side}\tERROR\t{err}")
        return 0
    res = compare(fork, port, cfg, a.context)
    report(res, fork, port, a.show_identical, sys.stdout)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))
    if a.strict and any(r["status"] in ("CHANGED", "NEW", "DROPPED") for r in res):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
