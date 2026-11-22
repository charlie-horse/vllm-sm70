# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/checks/inkling2/first_light.py on synthetic data: the
logprob thresholds, the per-visit state comparison against a dump directory
in the layout of ``sm70.inkling2.visit_dump_dir``, and one end-to-end run
against a fake server. Skipped when the repo's scripts/ tree is not next to
sm70/."""

import importlib.util
import io
import json
import math
import pathlib

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

_FILE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "scripts"
    / "checks"
    / "inkling2"
    / "first_light.py"
)
if not _FILE.is_file():
    pytest.skip("scripts/checks/inkling2 not in this tree", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("inkling2_first_light", _FILE)
fl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fl)

HIDDEN = 16
VISITS = 48
BANDS = fl.parse_visit_tols(fl.VISIT_TOLS)


def _top(tokens, logits):
    logits = np.asarray(logits, dtype=np.float64)
    logprobs = logits - np.log(np.exp(logits).sum())
    order = np.argsort(-logprobs)
    return [[int(tokens[i]), float(logprobs[i])] for i in order]


def _case(case_id, prompt_len, output_len, seed, state_positions=None):
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(output_len):
        logits = rng.normal(size=20) * 3.0
        logits[0] += 6.0
        positions.append({"top": _top(range(100, 120), logits)})
    return {
        "id": case_id,
        "prompt_ids": list(range(1, prompt_len + 1)),
        "output_ids": [p["top"][0][0] for p in positions],
        "positions": positions,
        "prefill_only": False,
        "state_positions": state_positions or [],
    }


def _served(case, edit=None):
    out = {key: case[key] for key in ("id", "prompt_ids", "output_ids")}
    out["positions"] = [
        {"top": [list(pair) for pair in p["top"]]} for p in case["positions"]
    ]
    if edit:
        edit(out["positions"])
    return out


def _run(*cases):
    return {"cases": list(cases)}


# --- thresholds -----------------------------------------------------------


def test_defaults_are_the_suggested_short_context_thresholds():
    assert (fl.MIN_AGREEMENT, fl.GAP_MAX, fl.MAX_REAL_FLIPS) == (0.98, 0.5, 0)
    assert (fl.MAX_MEAN_KL, fl.MAX_ABS_DELTA, fl.P99_ABS_DELTA) == (0.002, 0.75, 0.25)
    assert BANDS == [(0, 6, 2e-3), (7, 17, 2.5e-2), (18, 47, 9e-2)]
    assert [fl.visit_tol(BANDS, v) for v in (0, 6, 7, 17, 18, 47)] == [
        2e-3, 2e-3, 2.5e-2, 2.5e-2, 9e-2, 9e-2
    ]  # fmt: skip
    with pytest.raises(ValueError, match="visit 48"):
        fl.visit_tol(BANDS, 48)


def test_identical_runs_pass_with_zero_kl():
    ref = _run(_case("a", 5, 40, 0), _case("b", 7, 60, 1))
    report = fl.compare_logprobs(ref, _run(*map(_served, ref["cases"])))
    assert report["pass"] and report["failures"] == []
    assert report["positions"] == 100 and report["agreement"] == 1.0
    assert report["mean_kl"] == 0.0 and report["max_abs_delta"] == 0.0
    assert report["kl_tokens_missing"] == 0


def test_kl_of_a_position_is_the_sum_over_the_reference_tokens():
    ref = _top([1, 2, 3], [2.0, 1.0, 0.0])
    cand = _top([1, 2, 3], [1.5, 1.2, 0.1])
    want = sum(math.exp(lp) * (lp - dict(map(tuple, cand))[tok]) for tok, lp in ref)
    got, missing = fl.position_kl(ref, cand)
    assert got == pytest.approx(want) and got > 0 and missing == 0
    # A token the candidate does not list takes its smallest listed logprob.
    got, missing = fl.position_kl(ref, cand[:2])
    low = cand[1][1]
    want = sum(
        math.exp(lp) * (lp - dict(map(tuple, cand[:2])).get(tok, low))
        for tok, lp in ref
    )
    assert got == pytest.approx(want) and missing == 1


def _shift(amount, where):
    def edit(positions):
        for pos in where:
            positions[pos]["top"][0][1] -= amount

    return edit


def test_mean_kl_threshold():
    case = _case("a", 5, 100, 2)

    def blur(positions):
        for p in positions:
            p["top"] = _top([t for t, _ in p["top"]], [lp * 0.9 for _, lp in p["top"]])

    report = fl.compare_logprobs(_run(case), _run(_served(case, blur)))
    assert report["agreement"] == 1.0 and report["mean_kl"] > fl.MAX_MEAN_KL
    assert not report["pass"]
    assert any(f.startswith("mean KL") for f in report["failures"])
    assert fl.compare_logprobs(
        _run(case), _run(_served(case, blur)), max_mean_kl=1.0, max_abs_delta=9.0,
        p99_abs_delta=9.0,
    )["pass"]  # fmt: skip


def test_delta_thresholds_max_and_p99():
    case = _case("a", 5, 200, 3)
    one = fl.compare_logprobs(
        _run(case), _run(_served(case, _shift(0.5, [7]))), max_mean_kl=1.0
    )
    # One position at 0.5: under the max, and the p99 of 200 stays low.
    assert one["max_abs_delta"] == pytest.approx(0.5) and one["pass"]
    big = fl.compare_logprobs(
        _run(case), _run(_served(case, _shift(0.8, [7]))), max_mean_kl=1.0
    )
    assert [f.split(" ")[0] for f in big["failures"]] == ["max"]
    many = fl.compare_logprobs(
        _run(case), _run(_served(case, _shift(0.3, range(0, 200, 20)))), max_mean_kl=1.0
    )
    assert many["max_abs_delta"] < fl.MAX_ABS_DELTA
    assert [f.split(" ")[0] for f in many["failures"]] == ["p99"]


def _swap_top(positions, pos):
    top = positions[pos]["top"]
    top[0][0], top[1][0] = top[1][0], top[0][0]


def test_flips_near_ties_pass_and_real_flips_fail():
    case = _case("a", 5, 100, 4)
    near = [
        i
        for i, p in enumerate(case["positions"])
        if p["top"][0][1] - p["top"][1][1] < 0.5
    ]
    real = [
        i
        for i, p in enumerate(case["positions"])
        if p["top"][0][1] - p["top"][1][1] >= 0.5
    ]
    # Make one near tie explicit.
    tie = real[0]
    case["positions"][tie]["top"][1][1] = case["positions"][tie]["top"][0][1] - 0.01
    loose = {"max_mean_kl": 9.0, "max_abs_delta": 99.0, "p99_abs_delta": 99.0}
    report = fl.compare_logprobs(
        _run(case), _run(_served(case, lambda p: _swap_top(p, tie))), **loose
    )
    assert report["flips"] == 1 and report["real_flips"] == 0 and report["pass"]
    assert report["agreement"] == 0.99
    bad = real[1]
    report = fl.compare_logprobs(
        _run(case), _run(_served(case, lambda p: _swap_top(p, bad))), **loose
    )
    assert report["real_flips"] == 1 and not report["pass"]
    assert "flips with a reference gap >= 0.5" in report["failures"][0]
    assert fl.compare_logprobs(
        _run(case), _run(_served(case, lambda p: _swap_top(p, bad))),
        max_real_flips=1, **loose,
    )["pass"]  # fmt: skip
    del near


def test_agreement_threshold_counts_near_tie_flips():
    case = _case("a", 5, 100, 5)
    for p in case["positions"][:3]:
        p["top"][1][1] = p["top"][0][1] - 0.01

    def flip3(positions):
        for pos in range(3):
            _swap_top(positions, pos)

    loose = {"max_mean_kl": 9.0, "max_abs_delta": 99.0, "p99_abs_delta": 99.0}
    report = fl.compare_logprobs(_run(case), _run(_served(case, flip3)), **loose)
    assert report["real_flips"] == 0 and report["agreement"] == 0.97
    assert report["failures"] == ["top-1 agreement 0.9700 < 0.98"]


def test_reference_top1_outside_the_served_list_fails():
    case = _case("a", 5, 50, 6)

    def drop(positions):
        positions[4]["top"] = positions[4]["top"][1:]

    report = fl.compare_logprobs(_run(case), _run(_served(case, drop)))
    assert report["positions_without_delta"] == 1 and not report["pass"]
    assert any("outside the served top list" in f for f in report["failures"])


def test_mismatched_cases_raise():
    case = _case("a", 5, 10, 7)
    other = _served(case)
    other["output_ids"] = other["output_ids"][::-1]
    with pytest.raises(ValueError, match="output_ids differ"):
        fl.compare_logprobs(_run(case), _run(other))
    with pytest.raises(ValueError, match="case ids differ"):
        fl.compare_logprobs(_run(case), _run())


def test_case_selection():
    run = _run(_case("short", 5, 4, 0), _case("long", 600, 4, 1), _case("pre", 5, 4, 2))
    run["cases"][2]["prefill_only"] = True
    assert [c["id"] for c in fl.select_cases(run, None, 512)] == ["short"]
    assert [c["id"] for c in fl.select_cases(run, None, 600)] == ["short", "long"]
    assert [c["id"] for c in fl.select_cases(run, ["long", "pre"], 1)] == [
        "long",
        "pre",
    ]
    with pytest.raises(ValueError, match="nope"):
        fl.select_cases(run, ["short", "nope"], 512)


# --- states ----------------------------------------------------------------


def _states(num_tokens, seed):
    """visit index -> [num_tokens, HIDDEN] fp32."""
    rng = np.random.default_rng(seed)
    return {
        v: rng.normal(size=(num_tokens, HIDDEN)).astype(np.float32) * 10.0
        for v in range(VISITS)
    }


def _golden(states, positions):
    return {
        f"visit_{v:02d}": rows[positions].astype(np.float16)
        for v, rows in states.items()
    }


def _scaled(states, visit, factor, rows=slice(None)):
    out = {v: rows_.copy() for v, rows_ in states.items()}
    out[visit][rows] *= factor
    return out


def test_states_within_the_bands_pass():
    states = _states(12, 0)
    positions = [0, 3, 4, 11]
    report = fl.compare_states(_golden(states, positions), states, positions, BANDS)
    assert report["pass"] and report["first_failing_visit"] is None
    assert len(report["visits"]) == VISITS and report["positions"] == 4
    # fp16 rounding of the golden rows only.
    assert max(v["median_relative_l2"] for v in report["visits"]) < 1e-3


@pytest.mark.parametrize(
    "visit, factor, fails",
    [
        (3, 1.0015, False),
        (3, 1.004, True),
        (10, 1.02, False),
        (10, 1.03, True),
        (30, 1.08, False),
        (30, 1.1, True),
    ],
)
def test_each_band_has_its_own_tolerance(visit, factor, fails):
    states = _states(12, 1)
    positions = list(range(12))
    golden = _golden(states, positions)
    report = fl.compare_states(golden, _scaled(states, visit, factor), positions, BANDS)
    assert report["pass"] is not fails
    assert report["first_failing_visit"] == (visit if fails else None)


def test_first_failing_visit_is_the_earliest_and_rows_follow_the_positions():
    states = _states(20, 2)
    positions = [1, 5, 6, 7, 19]
    golden = _golden(states, positions)
    port = _scaled(_scaled(states, 40, 2.0), 9, 2.0)
    report = fl.compare_states(golden, port, positions, BANDS)
    assert report["first_failing_visit"] == 9
    assert [v["visit"] for v in report["visits"] if not v["ok"]] == [9, 40]
    # A change at rows the golden has no position for is not seen.
    hidden = _scaled(states, 9, 2.0, rows=[0, 2, 3, 4])
    assert fl.compare_states(golden, hidden, positions, BANDS)["pass"]
    # Rows shifted by one token fail at the first visit.
    shifted = {v: np.roll(rows, 1, axis=0) for v, rows in states.items()}
    assert (
        fl.compare_states(golden, shifted, positions, BANDS)["first_failing_visit"] == 0
    )


def test_the_median_decides_not_one_position():
    states = _states(11, 3)
    positions = list(range(11))
    golden = _golden(states, positions)
    one = _scaled(states, 20, 3.0, rows=[4])
    report = fl.compare_states(golden, one, positions, BANDS)
    assert report["pass"] and report["visits"][20]["max_relative_l2"] > 1.0
    most = _scaled(states, 20, 3.0, rows=list(range(6)))
    assert (
        fl.compare_states(golden, most, positions, BANDS)["first_failing_visit"] == 20
    )


def test_non_finite_rows_and_missing_visits_fail():
    states = _states(8, 4)
    positions = [0, 7]
    golden = _golden(states, positions)
    bad = {v: rows.copy() for v, rows in states.items()}
    bad[5][7, 0] = np.nan
    report = fl.compare_states(golden, bad, positions, BANDS)
    assert report["first_failing_visit"] == 5 and report["visits"][5]["nonfinite"] == 1
    missing = {v: rows for v, rows in states.items() if v != 2}
    report = fl.compare_states(golden, missing, positions, BANDS)
    assert (
        report["first_failing_visit"] == 2 and "no dump" in report["visits"][2]["error"]
    )
    short = {v: rows[:7] for v, rows in states.items()}
    report = fl.compare_states(golden, short, positions, BANDS)
    assert "position 7 outside 7 rows" in report["visits"][0]["error"]
    extra = dict(states)
    extra[48] = states[0]
    assert fl.compare_states(golden, extra, positions, BANDS)[
        "visits_without_golden"
    ] == [48]
    assert not fl.compare_states({}, states, [], BANDS)["pass"]


# --- the dump directory ----------------------------------------------------


def _write_forward(dump_dir, rank, forward, rows_by_visit):
    folder = dump_dir / f"rank{rank}" / f"forward{forward:06d}"
    folder.mkdir(parents=True)
    for visit, rows in rows_by_visit.items():
        layer, pass_idx = visit % 24, visit // 24
        name = f"visit{visit:02d}_layer{layer:02d}_pass{pass_idx}.pt"
        torch.save(torch.from_numpy(rows).half(), folder / name)
    return folder


def test_forwards_are_read_in_order_and_chunks_add_up(tmp_path):
    states = _states(10, 5)
    first = _write_forward(tmp_path, 0, 3, {v: r[:6] for v, r in states.items()})
    second = _write_forward(tmp_path, 0, 4, {v: r[6:] for v, r in states.items()})
    (tmp_path / "rank0" / "partial_maxima.pt").write_bytes(b"")
    assert fl.forward_dirs(tmp_path) == {"rank0": [first, second]}
    port = fl.load_forwards([first, second], 10)
    assert sorted(port) == list(range(VISITS))
    assert np.array_equal(port[7], states[7].astype(np.float16).astype(np.float32))
    with pytest.raises(ValueError, match="10 rows in 2 forwards for a request of 9"):
        fl.load_forwards([first, second], 9)
    with pytest.raises(ValueError, match="6 rows in 1 forwards for a request of 10"):
        fl.load_forwards([first], 10)
    # One forward with padded rows after the request's.
    assert fl.load_forwards([first], 4)[0].shape == (4, HIDDEN)


def test_new_forwards_are_the_ones_after_the_snapshot(tmp_path):
    states = _states(4, 6)
    _write_forward(tmp_path, 0, 0, states)
    _write_forward(tmp_path, 1, 0, states)
    before = fl.forward_dirs(tmp_path)
    assert fl.new_forwards(before, tmp_path, VISITS, 0.0) == {"rank0": [], "rank1": []}
    new0 = _write_forward(tmp_path, 0, 1, states)
    new1 = _write_forward(tmp_path, 1, 1, states)
    assert fl.new_forwards(before, tmp_path, VISITS, 0.0) == {
        "rank0": [new0],
        "rank1": [new1],
    }


# --- end to end against a fake server ----------------------------------------


class _Server:
    """Answers /v1/completions with prompt_logprobs from the golden cases and
    writes each request's forward to the dump directory, as the server does."""

    def __init__(self, cases, dump_dir, states, ranks=2, spoil=None):
        self.cases = {tuple(c["prompt_ids"] + c["output_ids"]): c for c in cases}
        self.dump_dir, self.states, self.ranks = dump_dir, states, ranks
        self.spoil = spoil or {}
        self.forward = 0
        self.bodies = []

    def __call__(self, request, timeout=None):
        body = json.loads(request.data)
        self.bodies.append(body)
        case = self.cases[tuple(body["prompt"])]
        if self.dump_dir is not None:
            self.forward += 1
            for rank in range(self.ranks):
                rows = self.states[case["id"]]
                rows = self.spoil.get((case["id"], rank), rows)
                _write_forward(self.dump_dir, rank, self.forward, rows)
        logprobs = [None] + [{"0": {"logprob": -1.0}}] * (len(case["prompt_ids"]) - 1)
        for position in case["positions"]:
            logprobs.append(
                {
                    str(tok): {"logprob": lp}
                    for tok, lp in position["top"][: body["prompt_logprobs"]]
                }
            )
        return io.BytesIO(
            json.dumps({"choices": [{"prompt_logprobs": logprobs}]}).encode()
        )


@pytest.fixture
def bundle(tmp_path):
    """A goldens directory of two short cases and a long one, with states."""
    goldens, dump = tmp_path / "goldens", tmp_path / "dump"
    (goldens / "cases").mkdir(parents=True)
    dump.mkdir()
    cases = [
        _case("one", 6, 10, 0, state_positions=[0, 1, 2, 3, 4, 5, 6, 14]),
        _case("two", 4, 8, 1, state_positions=[0, 3, 4, 11]),
        _case("long", 600, 3, 2, state_positions=[0]),
    ]
    states = {}
    for seed, case in enumerate(cases[:2]):
        total = len(case["prompt_ids"]) + len(case["output_ids"])
        states[case["id"]] = _states(total, 10 + seed)
        np.savez(
            goldens / "cases" / f"{case['id']}.states.npz",
            **_golden(states[case["id"]], case["state_positions"]),
        )
    (goldens / "run-text.json").write_text(json.dumps({"meta": {}, "cases": cases}))
    # A forward from before the run (startup), which no case may pick up.
    _write_forward(dump, 0, 0, _states(3, 99))
    _write_forward(dump, 1, 0, _states(3, 99))
    return goldens, dump, cases, states


def _main(monkeypatch, server, goldens, dump, tmp_path, *extra):
    monkeypatch.setattr(fl.urllib.request, "urlopen", server)
    out = tmp_path / "report.json"
    argv = ["--base-url", "http://h:1/", "--model", "m", "--goldens", str(goldens)]
    argv += ["--out", str(out), "--settle", "0", *extra]
    if dump is not None:
        argv += ["--dump-dir", str(dump)]
    status = fl.main(argv)
    return status, json.loads(out.read_text())


def test_end_to_end_pass(monkeypatch, bundle, tmp_path, capsys):
    goldens, dump, cases, states = bundle
    server = _Server(cases, dump, states)
    run_file = tmp_path / "served.json"
    status, report = _main(
        monkeypatch, server, goldens, dump, tmp_path, "--save-run", str(run_file)
    )
    assert status == 0 and report["pass"]
    # The short cases only, teacher-forced, with the top 20.
    assert [len(b["prompt"]) for b in server.bodies] == [16, 12]
    assert all(
        b["prompt_logprobs"] == 20 and b["max_tokens"] == 1 for b in server.bodies
    )
    assert report["logprobs"]["positions"] == 18
    assert set(report["states"]) == {"one", "two"}
    for result in report["states"].values():
        assert set(result["ranks"]) == {"rank0", "rank1"}
        assert all(len(r["forwards"]) == 1 for r in result["ranks"].values())
    assert report["states"]["two"]["ranks"]["rank1"]["forwards"] == ["forward000002"]
    assert "first light: pass" in capsys.readouterr().out
    # The saved run compares without a server.
    assert fl.main(["--goldens", str(goldens), "--captured", str(run_file)]) == 0


def test_end_to_end_reports_the_first_failing_visit_of_a_rank(
    monkeypatch, bundle, tmp_path, capsys
):
    goldens, dump, cases, states = bundle
    spoil = {("two", 1): _scaled(_scaled(states["two"], 33, 1.5), 21, 1.5)}
    server = _Server(cases, dump, states, spoil=spoil)
    status, report = _main(monkeypatch, server, goldens, dump, tmp_path)
    assert status == 1 and not report["pass"] and report["logprobs"]["pass"]
    assert report["states"]["one"]["pass"]
    ranks = report["states"]["two"]["ranks"]
    assert ranks["rank0"]["pass"] and ranks["rank1"]["first_failing_visit"] == 21
    out = capsys.readouterr().out
    assert "states two rank1: first failing visit 21: median relative L2" in out
    assert "first light: FAIL" in out


def test_end_to_end_fails_on_logprobs_and_on_missing_files(
    monkeypatch, bundle, tmp_path
):
    goldens, dump, cases, states = bundle
    wrong = json.loads(json.dumps(cases))
    for position in wrong[0]["positions"]:
        position["top"] = _top(
            [t for t, _ in position["top"]], [lp * 0.5 for _, lp in position["top"]]
        )
    by_prompt = _Server(cases, None, states)
    by_prompt.cases = {tuple(c["prompt_ids"] + c["output_ids"]): c for c in wrong}
    status, report = _main(monkeypatch, by_prompt, goldens, None, tmp_path)
    assert status == 1 and report["states"] == {}
    assert any(f.startswith("mean KL") for f in report["logprobs"]["failures"])
    # No golden states for a case, and no dump written by the server.
    (goldens / "cases" / "two.states.npz").unlink()
    status, report = _main(
        monkeypatch, _Server(cases, None, states), goldens, dump, tmp_path
    )
    assert status == 1 and report["logprobs"]["pass"]
    assert "two.states.npz" in report["states"]["two"]["error"]
    assert "no dump" in report["states"]["one"]["ranks"]["rank0"]["error"]
