# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/checks/greedy: the parameterised greedy runner and
the teacher-forced comparator (served vs reference logprobs, per-visit
relative L2). Skipped when the repo's scripts/ tree is not next to sm70/."""

import importlib.util
import io
import json
import pathlib

import pytest

_DIR = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "checks" / "greedy"
if not (_DIR / "teacher_forced.py").is_file():
    pytest.skip("scripts/checks/greedy not in this tree", allow_module_level=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"greedy_{name}", _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


greedy_run = _load("greedy_run")
tf = _load("teacher_forced")

PROMPTS = json.loads((_DIR / "greedy-prompts.json").read_text())


class _FakeServer:
    """Stands in for urllib.request.urlopen; records every request."""

    def __init__(self, respond):
        self.respond = respond
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append((request.full_url, request.data))
        return io.BytesIO(json.dumps(self.respond(json.loads(request.data))).encode())


def _chat_response(body):
    return {
        "choices": [{"message": {"content": "ok", "reasoning": "r"}}],
        "usage": {"prompt_tokens": 3},
    }


# ------------------------------------------------------------- greedy_run


def test_default_arguments_send_the_flint_bodies(monkeypatch, tmp_path, capsys):
    server = _FakeServer(_chat_response)
    monkeypatch.setattr(greedy_run.urllib.request, "urlopen", server)
    out = tmp_path / "out.json"
    greedy_run.main(["http://h:1/", str(_DIR / "greedy-prompts.json"), str(out), "1"])
    expected = [
        json.dumps(
            {
                "model": "flint",
                "messages": [{"role": "user", "content": p["content"]}],
                **PROMPTS["params"],
            }
        ).encode()
        for p in PROMPTS["prompts"]
    ]
    assert [data for _, data in server.requests] == expected
    assert {url for url, _ in server.requests} == {"http://h:1/v1/chat/completions"}
    result = json.loads(out.read_text())
    assert list(result) == [p["id"] for p in PROMPTS["prompts"]]
    assert all(r["identical_within_run"] for r in result.values())


def test_repeats_and_reference_arguments(monkeypatch, tmp_path, capsys):
    server = _FakeServer(_chat_response)
    monkeypatch.setattr(greedy_run.urllib.request, "urlopen", server)
    prompts = tmp_path / "p.json"
    prompts.write_text(
        json.dumps(
            {"params": {"temperature": 0}, "prompts": [{"id": "a", "content": "x"}]}
        )
    )
    ref = tmp_path / "ref.json"
    greedy_run.main(["http://h:1", str(prompts), str(ref)])
    assert len(server.requests) == 2
    greedy_run.main(
        ["http://h:1", str(prompts), str(tmp_path / "o.json"), "3", str(ref)]
    )
    assert len(server.requests) == 5
    assert "vs ref a match" in capsys.readouterr().out


def test_model_effort_special_tokens_and_images():
    prompt = {"id": "a", "content": "describe", "images": ["data:image/png;base64,AA"]}
    params = {"temperature": 0, "chat_template_kwargs": {"x": 1}}
    body = greedy_run.build_body(
        prompt,
        params,
        "inkling-2-mini",
        reasoning_effort="high",
        keep_special_tokens=True,
    )
    assert body["model"] == "inkling-2-mini"
    assert body["chat_template_kwargs"] == {"x": 1, "reasoning_effort": "high"}
    assert body["skip_special_tokens"] is False
    assert body["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        {"type": "text", "text": "describe"},
    ]
    assert params == {"temperature": 0, "chat_template_kwargs": {"x": 1}}


# --------------------------------------------------------- teacher forcing


def _case(n=200, cid="a"):
    """A reference case: top-1 is the output token, top-2 trails by 1 nat."""
    output = list(range(100, 100 + n))
    positions = [{"top": [[t, -0.3], [t + 1000, -1.3]]} for t in output]
    return {
        "id": cid,
        "prompt_ids": [1, 2, 3],
        "output_ids": output,
        "positions": positions,
    }


def _run(*cases):
    return {"cases": [json.loads(json.dumps(c)) for c in cases]}


def test_request_body_and_response_parsing():
    case = {"id": "a", "prompt_ids": [1, 2], "output_ids": [7, 8]}
    assert tf.build_body("m", case) == {
        "model": "m",
        "prompt": [1, 2, 7, 8],
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 2,
        "add_special_tokens": False,
    }
    response = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    {"2": {"logprob": -0.1, "rank": 1}},
                    {
                        "9": {"logprob": -0.2, "rank": 1},
                        "7": {"logprob": -2.0, "rank": 2},
                    },
                    # The prompt token ranks third: it is returned as well.
                    {
                        "8": {"logprob": -3.0, "rank": 3},
                        "5": {"logprob": -1.0, "rank": 2},
                        "4": {"logprob": -0.5, "rank": 1},
                    },
                ]
            }
        ]
    }
    assert tf.positions_from_response(response, case) == [
        {"top": [[9, -0.2], [7, -2.0]]},
        {"top": [[4, -0.5], [5, -1.0], [8, -3.0]]},
    ]
    short = {"choices": [{"prompt_logprobs": [None, {}]}]}
    with pytest.raises(ValueError, match="prompt_logprobs"):
        tf.positions_from_response(short, case)


def test_identical_runs_pass():
    report = tf.compare_runs(_run(_case()), _run(_case()))
    assert report["pass"] and report["agreement"] == 1.0 and report["flips"] == 0
    assert report["cases"][0]["max_abs_delta"] == 0.0


def test_logprob_delta_of_the_reference_top1():
    cand = _case()
    cand["positions"][5]["top"] = [[105, -0.35], [1105, -1.2]]
    cand["positions"][6]["top"] = [[106, -0.2], [1106, -1.5]]
    case = tf.compare_runs(_run(_case()), _run(cand))["cases"][0]
    assert case["agree"] == 200
    assert case["max_abs_delta"] == pytest.approx(0.1)
    assert case["mean_abs_delta"] == pytest.approx(0.15 / 200)


def test_near_tie_flip_passes():
    ref, cand = _case(), _case()
    ref["positions"][50]["top"] = [[150, -0.70], [1150, -0.80]]
    cand["positions"][50]["top"] = [[1150, -0.72], [150, -0.78]]
    report = tf.compare_runs(_run(ref), _run(cand))
    assert report["pass"]
    assert report["agreement"] == pytest.approx(199 / 200)
    assert (report["near_tie_flips"], report["real_flips"]) == (1, 0)
    (flip,) = report["cases"][0]["flips"]
    assert flip["pos"] == 50 and flip["gap"] == pytest.approx(0.1)
    assert flip["near_tie"] and flip["cand_is_ref_top2"]
    # The same flip is a real one under a tighter rule.
    assert not tf.compare_runs(_run(ref), _run(cand), gap_max=0.05)["pass"]


def test_real_divergence_fails():
    cand = _case()
    cand["positions"][50]["top"] = [[9, -0.1], [150, -4.0]]
    report = tf.compare_runs(_run(_case()), _run(cand))
    assert report["agreement"] >= 0.99 and not report["pass"]
    assert report["real_flips"] == 1
    flip = report["cases"][0]["flips"][0]
    assert flip["gap"] == pytest.approx(1.0) and not flip["near_tie"]
    assert not flip["cand_is_ref_top2"]


def test_too_many_near_tie_flips_fail_the_agreement_floor():
    ref, cand = _case(), _case()
    for pos in (10, 20, 30):
        tok = 100 + pos
        ref["positions"][pos]["top"] = [[tok, -0.70], [tok + 1000, -0.75]]
        cand["positions"][pos]["top"] = [[tok + 1000, -0.70], [tok, -0.75]]
    report = tf.compare_runs(_run(ref), _run(cand))
    assert report["real_flips"] == 0 and report["agreement"] < 0.99
    assert not report["pass"]
    assert tf.compare_runs(_run(ref), _run(cand), min_agreement=0.98)["pass"]


def test_flip_without_a_reference_top2_is_real():
    ref, cand = _case(), _case()
    ref["positions"][0]["top"] = [[100, -0.7]]
    cand["positions"][0]["top"] = [[1100, -0.7], [100, -0.7]]
    report = tf.compare_runs(_run(ref), _run(cand))
    assert report["cases"][0]["flips"][0]["gap"] is None and not report["pass"]


def test_mismatched_runs_are_rejected():
    other = _case()
    other["output_ids"][0] = 5
    with pytest.raises(ValueError, match="output_ids"):
        tf.compare_runs(_run(_case()), _run(other))
    with pytest.raises(ValueError, match="missing"):
        tf.compare_runs(_run(_case()), _run(_case(cid="b")))
    assert not tf.compare_runs({"cases": []}, {"cases": []})["pass"]


def _serve(shift):
    """A server whose top-1 is the prompt token, except where shift says."""

    def respond(body):
        entries = [None]
        for pos, tok in enumerate(body["prompt"][1:], start=1):
            entry = {str(tok): {"logprob": -0.4}, str(tok + 1000): {"logprob": -0.5}}
            if pos in shift:
                entry[str(tok)]["logprob"] = shift[pos]
            entries.append(entry)
        return {"choices": [{"prompt_logprobs": entries}]}

    return respond


def test_two_served_configurations(monkeypatch, tmp_path, capsys):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps(_run(_case(120), _case(80, "b"))))
    paths = []
    # Position 13 of the prompt is output position 10 (3 prompt tokens).
    for name, shift in (("a", {}), ("b", {13: -0.55}), ("c", {13: -3.0})):
        server = _FakeServer(_serve(shift))
        monkeypatch.setattr(tf.urllib.request, "urlopen", server)
        paths.append(str(tmp_path / f"{name}.json"))
        argv = ["capture", "--base-url", "http://h:1/", "--model", "m"]
        assert tf.main([*argv, "--cases", str(cases), "--out", paths[-1]]) == 0
        assert [url for url, _ in server.requests] == ["http://h:1/v1/completions"] * 2
        assert json.loads(server.requests[0][1])["prompt"] == [
            1,
            2,
            3,
            *range(100, 220),
        ]
    a, b, c = paths
    assert tf.main(["compare", a, a]) == 0
    # b flips one position that is a near tie in a (gap 0.1).
    report = tmp_path / "report.json"
    assert tf.main(["compare", a, b, "--out", str(report)]) == 0
    assert json.loads(report.read_text())["near_tie_flips"] == 2
    # Seen from c the same position has a 2.5 nat gap: a real flip.
    assert tf.main(["compare", c, a]) == 1
    assert "REAL" in capsys.readouterr().out


# ------------------------------------------------------- per-visit states


def _states(np, visits=48):
    rng = np.random.default_rng(0)
    return {f"visit_{i:02d}": rng.standard_normal((5, 16)) for i in range(visits)}


def test_relative_l2_per_visit(tmp_path):
    np = pytest.importorskip("numpy")
    ref = _states(np)
    assert tf.compare_states(ref, dict(ref))["pass"]

    noisy = {k: v * (1 + 1e-3) for k, v in ref.items()}
    report = tf.compare_states(ref, noisy)
    assert report["pass"]
    assert report["visits"][0]["relative_l2"] == pytest.approx(1e-3)

    # A uniform scale error keeps the cosine at 1 and must still fail.
    doubled = {k: 2 * v for k, v in ref.items()}
    report = tf.compare_states(ref, doubled)
    assert not report["pass"] and report["first_failing"] == "visit_00"
    assert all(v["relative_l2"] == pytest.approx(1.0) for v in report["visits"])

    late = dict(ref)
    for i in range(17, 48):
        late[f"visit_{i:02d}"] = ref[f"visit_{i:02d}"] * 1.015
    report = tf.compare_states(ref, late)
    assert report["first_failing"] == "visit_17"
    assert tf.compare_states(ref, late, tol=tf.TOL_LONG)["pass"]

    missing = {k: v for k, v in ref.items() if k != "visit_03"}
    assert tf.compare_states(ref, missing)["first_failing"] == "visit_03"
    reshaped = {**ref, "visit_04": ref["visit_04"][:, :8]}
    assert tf.compare_states(ref, reshaped)["first_failing"] == "visit_04"
    assert not tf.compare_states({}, {})["pass"]

    paths = {}
    for name, states in (("ref", ref), ("late", late), ("doubled", doubled)):
        paths[name] = str(tmp_path / f"{name}.npz")
        np.savez(paths[name], **states)
    assert tf.main(["states", paths["ref"], paths["ref"]]) == 0
    assert tf.main(["states", paths["ref"], paths["late"]]) == 1
    assert tf.main(["states", paths["ref"], paths["late"], "--long"]) == 0
    assert tf.main(["states", paths["ref"], paths["doubled"], "--long"]) == 1


def test_default_tolerances():
    assert (tf.MIN_AGREEMENT, tf.GAP_MAX) == (0.99, 0.5)
    assert (tf.TOL_SHORT, tf.TOL_LONG) == (1e-2, 2e-2)
