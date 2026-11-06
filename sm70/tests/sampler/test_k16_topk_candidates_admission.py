# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only admission of the SM70 vocab-parallel top-k candidate sampler.

Port of FORK tests/v1/sample/test_sm70_topk_candidates_admission.py
(cf9492a, 111 lines). Upstream's SamplingMetadata has no `top_k_cpu` /
`temperature_cpu` / `top_p_cpu`, so the host top-k is a separate argument
(the V1 hook reads it from the runner's InputBatch) and the host
temperature/top-p copies are gone (the kernel reads the device tensors).
"""

import importlib.util
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.v1.sample.logits_processor") is not None
    except (ImportError, ValueError):
        return False


requires_vllm = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")

from vllm_sm70.sample import (
    MAX_TOP_K,
    admission_reject_reason,
    argmax_invariant_reject_reason,
    metadata_reject_reason,
)


def _metadata(**overrides):
    fields = dict(
        all_random=True,
        all_greedy=False,
        max_num_logprobs=None,
        logprob_token_ids=None,
        no_penalties=True,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        temperature=object(),
        top_k=object(),
        top_p=object(),
        generators={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize("rows", [1, 2])
def test_served_defaults_are_admitted(rows):
    assert metadata_reject_reason(_metadata(), (20,) * rows, rows) is None


def test_other_temperatures_top_p_and_small_k_are_admitted():
    meta = _metadata(top_p=None)
    assert metadata_reject_reason(meta, (1, MAX_TOP_K), 2) is None


def test_one_seeded_request_is_admitted():
    meta = _metadata(generators={0: object()})
    assert metadata_reject_reason(meta, (20,), 1) is None


def test_numpy_top_k_is_accepted():
    np = pytest.importorskip("numpy")
    top_k = np.array([20, 7], dtype=np.int32)
    assert metadata_reject_reason(_metadata(), top_k.tolist(), 2) is None
    assert metadata_reject_reason(_metadata(), top_k, 2) is None


@pytest.mark.parametrize(
    "rows,overrides,top_k_cpu,reason",
    [
        (3, {}, (20, 20, 20), "batch_size"),
        (0, {}, (), "batch_size"),
        (1, {"all_random": False}, (20,), "sampling_mode"),
        (1, {"all_greedy": True}, (20,), "sampling_mode"),
        (1, {"max_num_logprobs": 0}, (20,), "logprobs"),
        (1, {"logprob_token_ids": {0: [1]}}, (20,), "logprob_token_ids"),
        (1, {"no_penalties": False}, (20,), "penalties"),
        (1, {"allowed_token_ids_mask": object()}, (20,), "allowed_token_ids"),
        (1, {"bad_words_token_ids": {0: [[1]]}}, (20,), "bad_words"),
        (1, {"top_k": None}, None, "top_k"),
        (1, {}, None, "top_k"),
        (1, {}, (MAX_TOP_K + 1,), "top_k"),
        (1, {}, (150144,), "top_k"),
        (1, {}, (0,), "top_k"),
        (2, {}, (20, 150144), "top_k"),
        (2, {}, (20,), "top_k"),
        (1, {"temperature": None}, (20,), "temperature"),
        (2, {"generators": {0: object()}}, (20, 20), "generators"),
    ],
)
def test_rejections(rows, overrides, top_k_cpu, reason):
    assert metadata_reject_reason(_metadata(**overrides), top_k_cpu, rows) == reason


def _min_p(count):
    from vllm.v1.sample.logits_processor.builtin import MinPLogitsProcessor

    processor = object.__new__(MinPLogitsProcessor)
    processor.min_p_count = count
    return processor


def _with_argmax_invariant(*processors):
    return SimpleNamespace(
        logitsprocs=SimpleNamespace(argmax_invariant=list(processors))
    )


@requires_vllm
@pytest.mark.parametrize(
    "spec,reason",
    [
        ((), None),
        ((0,), None),
        ((1,), "argmax_invariant_logits_processor"),
        (("other",), "argmax_invariant_logits_processor"),
        ((0, "other"), "argmax_invariant_logits_processor"),
    ],
)
def test_argmax_invariant_processors(spec, reason):
    processors = [object() if s == "other" else _min_p(s) for s in spec]
    meta = _with_argmax_invariant(*processors)
    assert argmax_invariant_reject_reason(meta) == reason


def test_non_cuda_device_is_rejected_first():
    meta = _metadata()
    assert (
        admission_reject_reason(meta, (20, 20, 20), 3, torch.device("cpu"))
        == "non_cuda"
    )
