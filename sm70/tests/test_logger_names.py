"""Guard: vllm_sm70 loggers must live under vLLM's configured "vllm" logger.

vLLM's default logging config attaches its handler only to "vllm"
(propagate=False). A logger named after the module ("vllm_sm70.…") is not
under it, so its INFO lines (route decisions) are silently dropped in every
engine process. Use init_logger("vllm.sm70.<area>") or getLogger("vllm.sm70").
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1] / "vllm_sm70"
BAD = re.compile(r"(?:init_logger|getLogger)\(\s*__name__\s*\)")
# Files exempt from the "vllm.sm70" logger rule.
ALLOW = {
    "ops/flint_moe.py",
}


def test_no_module_named_loggers():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel in ALLOW:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if BAD.search(line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, "loggers outside 'vllm' are dropped:\n" + "\n".join(offenders)
