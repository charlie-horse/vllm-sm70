# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the pure helpers in sm70/scripts/qwen_phase2/probe.py."""

import importlib.util
import pathlib

_PATH = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "qwen_phase2" / "probe.py"
_spec = importlib.util.spec_from_file_location("qp2_probe", _PATH)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)


def test_problem_set_sizes():
    assert len(probe.MATH) == 10 and len(probe.CODE) == 5
    assert len(probe.GIBBERISH_PROMPTS) == 20


def test_parse_numeric_answer():
    p = probe.parse_numeric_answer
    assert p("so 17*23 = 391\nANSWER: 391") == 391
    assert p("ANSWER: **5,050**") == 5050
    assert p("ANSWER: $36$") == 36
    assert p("answer: \\(12\\)") == 12
    assert p("first ANSWER: 3, corrected ANSWER: 4") == 4
    assert p("the total is 270 km") == 270  # no tag: last number
    assert p("ANSWER: -2.5") == -2.5
    assert p("no digits here") is None


def test_extract_code_and_run():
    text = "Here:\n```python\nimport os\n```\n```python\ndef f(x):\n    return x + 1\n```\n"
    code = probe.extract_code(text, "f")
    assert "def f(" in code
    assert probe.run_code_check(code, "assert f(1) == 2\n") == (True, "")
    ok, err = probe.run_code_check(code, "assert f(1) == 3\n")
    assert not ok and "AssertionError" in err
    ok, err = probe.run_code_check("while True: pass", "", timeout=1)
    assert not ok and "timeout" in err
    assert probe.extract_code("def g(): pass", "g") == "def g(): pass"


def test_reference_solutions_pass_the_code_asserts():
    sols = {
        "is_palindrome": "def is_palindrome(s):\n t=[c.lower() for c in s if c.isalnum()]\n return t==t[::-1]",
        "fizzbuzz": "def fizzbuzz(n):\n return ['FizzBuzz' if i%15==0 else 'Fizz' if i%3==0 "
                    "else 'Buzz' if i%5==0 else str(i) for i in range(1,n+1)]",
        "second_largest": "def second_largest(nums):\n u=sorted(set(nums))\n return u[-2] if len(u)>1 else None",
        "word_count": "def word_count(text):\n d={}\n for w in text.lower().split(): d[w]=d.get(w,0)+1\n return d",
        "flatten": "def flatten(l):\n o=[]\n for x in l:\n  o.extend(flatten(x)) if isinstance(x,list) else o.append(x)\n return o",
    }
    for name, _, asserts in probe.CODE:
        assert probe.run_code_check(sols[name], asserts)[0], name


def test_gibberish_detector():
    g = probe.gibberish_reasons
    prose = ("A rainbow forms when sunlight enters raindrops, bends, reflects off the back "
             "of the drop and bends again as it leaves, splitting white light into colours. "
             "Each colour leaves at a slightly different angle, so you see red on the outside.")
    assert g(prose) == []
    assert g("It\u2019s \u201cfine\u201d \u2014 really\u2026") == []
    assert g("") == ["empty"]
    assert any("non-ascii" in r for r in g("\u4e00\u4e8c\u4e09\u56db\u4e94 abc " * 10))
    assert any("repeated-4grams" in r for r in g("the cat sat down " * 20))
    assert any("word-run" in r for r in g("ok " + "the " * 10 + "end"))
    assert any("char-run" in r for r in g("x" * 50))
    assert "replacement-char" in g("broken \ufffd text")


def test_needle_prompt_deterministic_and_placed():
    p1, c1 = probe.needle_prompt(100, 0.9, 1)
    p2, c2 = probe.needle_prompt(100, 0.9, 1)
    assert (p1, c1) == (p2, c2) and p1.count(c1) == 1
    assert p1.index(c1) > 0.8 * len(p1.split("\n\n")[0])


def test_spec_metrics_delta():
    def prom(d, t, a, pos):
        lines = [
            "# HELP vllm:spec_decode_num_drafts Number of spec decoding drafts.",
            f'vllm:spec_decode_num_drafts_total{{engine="0",model_name="qwen"}} {d}',
            f'vllm:spec_decode_num_drafts_created{{engine="0",model_name="qwen"}} 1.7e9',
            f'vllm:spec_decode_num_draft_tokens_total{{engine="0",model_name="qwen"}} {t}',
            f'vllm:spec_decode_num_accepted_tokens_total{{engine="0",model_name="qwen"}} {a}',
        ]
        lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",'
                  f'model_name="qwen",position="{i}"}} {v}' for i, v in enumerate(pos)]
        return probe.parse_prom("\n".join(lines))

    before, after = prom(10, 70, 20, [8, 6]), prom(110, 770, 320, [88, 66])
    d = probe.spec_delta(before, after)
    assert d["drafts"] == 100 and d["accepted_tokens"] == 300
    assert d["mean_acceptance_length"] == 4.0
    assert abs(d["acceptance_rate"] - 300 / 700) < 1e-12
    assert d["per_position_rate"] == [0.8, 0.6]
    assert probe.spec_delta({}, {})["mean_acceptance_length"] is None


def test_verdict_gates():
    phases = {"math": {"correct": 9, "total": 10}, "code": {"correct": 5, "total": 5},
              "needle": {"correct": 3, "total": 3}, "gibberish": {"flagged": 0}}
    assert probe.verdict({"phases": phases, "gated": True})[0] == "PASS"
    spec = {"phases": phases, "gated": True, "spec": True, "spec_decode": {}}
    assert probe.verdict(spec)[0] == "FAIL"  # spec row with no drafts
    zero_acc = dict(spec, spec_decode={"overall": {"drafts": 50, "accepted_tokens": 0}})
    verdict, why = probe.verdict(zero_acc)
    assert verdict == "FAIL" and any("0 accepted" in w for w in why)
    ok_acc = dict(spec, spec_decode={"overall": {"drafts": 50, "accepted_tokens": 120}})
    assert probe.verdict(ok_acc)[0] == "PASS"
    bad = dict(phases, math={"correct": 8, "total": 10})
    assert probe.verdict({"phases": bad, "gated": True})[0] == "FAIL"
    assert probe.verdict({"phases": bad, "gated": False})[0] == "INFO"
