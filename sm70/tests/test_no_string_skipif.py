# SPDX-License-Identifier: Apache-2.0
"""Guard: no string conditions in pytest.mark.skipif / xfail.

A string condition is evaluated in the decorated function's globals. Under
@torch.inference_mode() those are torch's, so a helper defined in the test
module raises NameError at setup (#34, and again in the M3 K11 gate).
"""

import ast
import pathlib


def test_no_string_skipif_conditions():
    root = pathlib.Path(__file__).parent
    bad = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") in ("skipif", "xfail")
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                bad.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not bad, "use a boolean skipif/xfail condition, not a string: " + ", ".join(bad)
