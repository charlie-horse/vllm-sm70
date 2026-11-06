# SPDX-License-Identifier: Apache-2.0
"""Tests for M10: FlintSm70 opts into the K16 top-k candidate sampler.

The sampler half of M10. The router half (K13, no model declaration) is
#86's sm70/tests/models/test_m10_flint_router.py. K16 replaces the fork's
per-model sampler hooks (fork cf9492a vllm/models/flint/model.py:570-577,
:605-614); M10 sets ``supports_sm70_topk_sampler = True`` on the Flint model,
the opt-in K16's U4 hook (vllm_sm70/sample_v1.py) checks.

CPU: the M5 model declares the opt-in and passes K16's model gate; the
  Flint TP2 LM head passes K16's rank-uniform LM-head admission; only
  VLLM_SM70_FLINT_SAMPLER=0 makes the platform getter return None, and then
  kernel_warmup does not prepare the sampler.
GPU (``-m gpu``), what K16's contract says is exact, and nothing more
  (Quinn: no exact-token gate between arms that differ numerically):
  K16 samples top-k <= 20 from the same FP16 LM-head logits as upstream,
  restricted to the only tokens upstream can return (FORK
  v1/sample/ops/sm70_topk_candidates.py docstring; vllm_sm70/sample.py).
  So inside ONE engine with K16 on, a top_k=1 request (K16 samples it) gives
  the greedy request's tokens (upstream samples it: K16 rejects temperature
  0 as ``sampling_mode``). Same hidden states, same logits; only an exact
  FP16 tie at the top logit could differ (top-k keeps ties and both K16 and
  upstream top_k=1 race them, greedy takes the lowest id).
  The "off" engine (VLLM_SM70_FLINT_SAMPLER=0) only shows K16 absent; its
  greedy tokens next to the "on" engine's are printed, not gated: two engine
  instances are not bitwise equal on every build (M5 T4c).
  Seeded top-k 20 / top-p 0.95 on vs off stays xfail until K16's RNG parity
  and engine-instance determinism are verified (see ``seeded_parity_unverified``).

K16 is a V1-runner hook (patches/0005, U4); the Flint wrapper sets
VLLM_USE_V2_MODEL_RUNNER=0 (roadmap D2) and so do these tests.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
from vllm_sm70.models import flint_sm70

SAMPLER_SWITCH = "VLLM_SM70_FLINT_SAMPLER"

# Log lines (vllm_sm70/sample.py, sample_v1.py), once per process = per rank.
SAMPLER_ON_LOG = "SM70 top-k candidate sampler enabled."
SAMPLER_GREEDY_REJECT_LOG = "SM70 top-k candidate sampler rejected: sampling_mode"


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------
@pytest.fixture
def platform_cls(monkeypatch):
    """Sm70CudaPlatform on a CPU host (as the K16 CPU tests import it)."""
    import importlib
    import types

    try:
        importlib.import_module("vllm._C_stable_libtorch")
    except ImportError:  # vllm.platforms.cuda imports it (UP :23)
        monkeypatch.setitem(
            sys.modules,
            "vllm._C_stable_libtorch",
            types.ModuleType("vllm._C_stable_libtorch"),
        )
    from vllm_sm70._cuda_platform import Sm70CudaPlatform

    return Sm70CudaPlatform


@pytest.fixture
def tp(monkeypatch):
    """A stand-in TP group, so M5 layers can be built without torch.distributed."""
    import vllm.distributed.parallel_state as ps

    def _set(world_size=1, rank=0):
        group = SimpleNamespace(world_size=world_size, rank_in_group=rank)
        monkeypatch.setattr(ps, "_TP", group)

    _set()
    return _set


@pytest.fixture
def vllm_config(monkeypatch):
    """A current vLLM config with only what LogitsProcessor reads (the model
    config's head_dtype: FP16, i.e. --dtype half with no head_dtype override)."""
    import vllm.config.vllm as vc

    cfg = SimpleNamespace(model_config=SimpleNamespace(head_dtype=torch.float16))
    monkeypatch.setattr(vc, "_current_vllm_config", cfg)
    return cfg


def test_model_declares_the_sampler_opt_in():
    from vllm_sm70 import sample_v1
    from vllm_sm70.models.flint_sm70 import model as m

    assert sample_v1.MODEL_OPT_IN == "supports_sm70_topk_sampler"
    assert m.FlintSm70ForCausalLM.supports_sm70_topk_sampler is True
    # Only the Flint model opts in, not its decoder stack.
    assert not getattr(m.FlintSm70Model, sample_v1.MODEL_OPT_IN, False)


def test_sampler_passes_the_model_gate_on_flint(vllm_config):
    """K16's model gate: the opt-in, lm_head and logits_processor. A step that
    passes it gets the next rejection (spec decode here), not "model"."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm_sm70.models.flint_sm70 import model as m
    from vllm_sm70.sample_v1 import Sm70HiddenStateSampler

    flint = m.FlintSm70ForCausalLM.__new__(m.FlintSm70ForCausalLM)
    torch.nn.Module.__init__(flint)
    flint.lm_head = torch.nn.Linear(8, 8, bias=False)
    flint.logits_processor = LogitsProcessor(8)
    other = torch.nn.Module()
    other.lm_head = flint.lm_head
    other.logits_processor = flint.logits_processor

    class _Runner:  # weak-referenceable stand-in for the V1 runner
        device = "cpu"

    runner = _Runner()
    hook = Sm70HiddenStateSampler()
    hook.prepare(runner, SimpleNamespace(model_config=None))
    args = (torch.zeros(1, 8), SimpleNamespace(), SimpleNamespace(), object())
    assert hook.reject_reason(other, *args) == "model"
    assert hook.reject_reason(flint, *args) == "spec_decode"


def test_k16_admits_the_flint_tp2_lm_head(tp, vllm_config):
    """M5's LM head and logits processor, at Flint's vocabulary on TP2 rank 1
    (the last shard), pass K16's rank-uniform LM-head admission in FP16."""
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.models.flint.configs import FlintConfig
    from vllm_sm70 import sample

    config = FlintConfig()
    tp(world_size=2, rank=1)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        with torch.device("meta"):  # as M5 builds them (model.py __init__)
            lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
    finally:
        torch.set_default_dtype(default)
    logits_processor = LogitsProcessor(config.vocab_size)
    assert lm_head.weight.dtype == torch.float16
    assert logits_processor.scale == 1.0 and logits_processor.soft_cap is None
    assert sample.lm_head_reject_reason(logits_processor, lm_head, 2) is None


@pytest.mark.parametrize(
    "raw,on",
    [
        (None, True),
        ("1", True),
        ("", True),
        ("false", True),
        ("00", True),
        ("0", False),
    ],
)
def test_sampler_kill_switch_only_zero(platform_cls, monkeypatch, raw, on):
    from vllm_sm70.sample_v1 import Sm70HiddenStateSampler

    monkeypatch.delenv(SAMPLER_SWITCH, raising=False)
    if raw is not None:
        monkeypatch.setenv(SAMPLER_SWITCH, raw)
    hook = platform_cls.get_hidden_state_sampler()
    assert isinstance(hook, Sm70HiddenStateSampler) is on
    assert (hook is None) is not on


@pytest.mark.parametrize("sampler_off", [False, True])
def test_kernel_warmup_prepares_the_sampler_only_when_on(
    platform_cls, monkeypatch, sampler_off
):
    """kernel_warmup's step (3) (U1, before capture) binds the sampler via
    sample_v1.prepare_worker; with VLLM_SM70_FLINT_SAMPLER=0 it does not.
    Steps (1) and (2) are covered by tests/plugin/test_kernel_warmup_order.py
    and #86; here (1) is stubbed and (2) skipped (no get_model)."""
    from vllm_sm70 import sample_v1, warmup

    if sampler_off:
        monkeypatch.setenv(SAMPLER_SWITCH, "0")
    else:
        monkeypatch.delenv(SAMPLER_SWITCH, raising=False)
    prepared = []
    monkeypatch.setattr(warmup, "run", lambda worker: None)
    monkeypatch.setattr(sample_v1, "prepare_worker", prepared.append)
    worker = SimpleNamespace()  # no get_model: the router step is skipped
    assert platform_cls.kernel_warmup(worker) is None
    assert prepared == ([] if sampler_off else [worker])


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------
def _gpu_ok(n=1) -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= n
        and torch.cuda.get_device_capability() == (7, 0)
    )


one_gpu = pytest.mark.skipif(not _gpu_ok(1), reason="needs a V100")
two_gpus = pytest.mark.skipif(not _gpu_ok(2), reason="needs 2 V100 (TP2)")
CKPT = os.environ.get("FLINT_SM70_CHECKPOINT")
needs_ckpt = pytest.mark.skipif(CKPT is None, reason="set FLINT_SM70_CHECKPOINT")
ARCH_OVERRIDE = {"architectures": [flint_sm70.ARCH]}
PROMPTS = [
    "def quicksort(xs):",
    "The capital of France is",
    "Explain the CAP theorem in one sentence.",
    "1, 1, 2, 3, 5, 8,",
]
TINY_IDS = [[1, 2, 3, 4, 5], [7, 8, 9]]
V1_RUNNER = {"VLLM_USE_V2_MODEL_RUNNER": "0"}  # U4/K16 are V1-only (D2)


def _gpus_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return [int(x) for x in out.split() if x.strip().isdigit()]


def _shutdown(llm, settle_mib=2048, timeout_s=120):
    """Release an LLM's GPUs before the next one starts (as #36's T-tests):
    stop the engine core, drop the object, tear down the process groups,
    then wait for the memory to come back."""
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    try:
        llm.llm_engine.engine_core.shutdown()
    finally:
        del llm
        gc.collect()
        torch._dynamo.reset()
        cleanup_dist_env_and_memory()
    deadline = time.monotonic() + timeout_s
    used = []
    while time.monotonic() < deadline:
        used = _gpus_used_mib()
        if not used or max(used) < settle_mib:
            return
        time.sleep(2)
    raise RuntimeError(f"GPU memory not released after shutdown: {used} MiB")


def _tiny_flint_dir(tmp_path):
    """#36's tiny random Flint (hidden 2048, E256/K6, vocab 512)."""
    from vllm.models.flint.configs import FlintConfig

    cfg = FlintConfig(
        vocab_size=512,
        hidden_size=2048,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        num_experts=256,
        num_experts_per_tok=6,
        norm_topk_prob=True,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        mlp_only_layers=[0],
        layer_types=["sliding_attention", "full_attention"],
        no_rope_layer_indices=[1],
        sliding_window=64,
        max_position_embeddings=1024,
        torch_dtype="float16",
    )
    cfg.architectures = [flint_sm70.ARCH]
    cfg.save_pretrained(tmp_path)
    return str(tmp_path)


def _tiny_llm(path):
    from vllm import LLM

    return LLM(
        model=path,
        load_format="dummy",
        dtype="half",
        skip_tokenizer_init=True,
        max_model_len=512,
        max_num_seqs=2,
        gpu_memory_utilization=0.4,
        enable_prefix_caching=False,  # the 2nd request must prefill in full
        seed=0,
    )


def _ckpt_llm():
    from vllm import LLM

    return LLM(
        model=CKPT,
        tensor_parallel_size=2,
        dtype="half",
        hf_overrides=ARCH_OVERRIDE,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
        max_num_seqs=2,  # K16 admits batches of 1-2 rows
        enable_prefix_caching=False,  # the 2nd request must prefill in full
        seed=0,
    )


def _greedy_params(n):
    from vllm import SamplingParams

    return SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)


def _top1_params(n):
    from vllm import SamplingParams

    return SamplingParams(temperature=1.0, top_k=1, max_tokens=n, ignore_eos=True)


def _seeded_params(n):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.8, top_k=20, top_p=0.95, seed=1234, max_tokens=n
    )


def _run(make_llm, env, requests, monkeypatch, capfd):
    """One LLM under ``env``: [(prompts, params)] -> per-request token lists,
    plus the worker log (capfd reads the inherited worker stdout/stderr)."""
    import vllm.envs

    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    getattr(vllm.envs, "disable_envs_cache", lambda: None)()
    capfd.readouterr()
    llm = make_llm()
    try:
        toks = []
        for prompts, params in requests:
            outs = llm.generate(prompts, params)
            toks.append([list(o.outputs[0].token_ids) for o in outs])
    finally:
        _shutdown(llm)
    out, err = capfd.readouterr()
    logs = out + err
    sys.stdout.write(logs)  # keep the worker log in the pytest report
    return toks, logs


def _tiny_prompts():
    return [{"prompt_token_ids": p} for p in TINY_IDS]


def _common_prefix(a, b):
    return next(
        (j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )


def _sampler_exact(make_llm, prompts, n, monkeypatch, capfd, *, tp):
    """K16's exact contract inside one engine, plus K16 absent when killed.

    "on" engine: greedy (upstream, K16 rejects sampling_mode) then top_k=1 at
    temperature 1 (K16 samples), same prompts, same batch shapes, prefix
    caching off so both prefill the same rows: identical tokens. "off" engine: K16 never engaged; its greedy is printed next to
    the "on" engine's greedy, not gated (engine-instance noise, M5 T4c)."""
    reqs = [(prompts, _greedy_params(n)), (prompts, _top1_params(n))]
    on, on_logs = _run(
        make_llm, {**V1_RUNNER, SAMPLER_SWITCH: None}, reqs, monkeypatch, capfd
    )
    off, off_logs = _run(
        make_llm, {**V1_RUNNER, SAMPLER_SWITCH: "0"}, reqs[:1], monkeypatch, capfd
    )
    assert on_logs.count(SAMPLER_ON_LOG) == tp, "K16 enabled line not once per rank"
    assert on_logs.count(SAMPLER_GREEDY_REJECT_LOG) == tp, (
        "greedy rows did not take the upstream path on every rank"
    )
    assert SAMPLER_ON_LOG not in off_logs
    assert "SM70 top-k candidate sampler rejected" not in off_logs
    greedy, top1 = on
    assert all(len(t) == n for t in greedy + top1)
    for i, (g, k) in enumerate(zip(greedy, top1)):
        assert g == k, (
            f"request {i}: K16 top_k=1 differs from upstream greedy in the same "
            f"engine at token {_common_prefix(g, k)}/{n}"
        )
    for i, (a, b) in enumerate(zip(greedy, off[0])):
        print(
            f"info request {i}: greedy on-engine vs off-engine common prefix "
            f"{_common_prefix(a, b)}/{n}"
        )


class NotEngagedError(Exception):
    """K16 did not run (or ran while killed): a real failure, not covered by
    ``seeded_parity_unverified``."""


# The seeded on-vs-off comparison spans two engine instances, so it holds only
# if (a) K16 draws upstream's FP32 exponentials from the same Philox state and
# advances the generator by the same offset, verified on GPU by
# sm70/tests/sampler/test_k16_topk_candidates_sampler_gpu.py::
# test_request_generator_matches_reference (a seeded request generator) and
# ::test_same_token_as_reference, and (b) the two engines produce bitwise
# equal logits, verified by M5's T4c (eager vs eager, two instances, tuning
# off). Until both pass on the stack under test only the token equality may
# fail (AssertionError); a NotEngagedError still fails and an XPASS shows.
seeded_parity_unverified = pytest.mark.xfail(
    strict=False,
    raises=AssertionError,
    reason=(
        "needs K16 RNG parity (test_k16_topk_candidates_sampler_gpu.py::"
        "test_request_generator_matches_reference, ::test_same_token_as_"
        "reference) and engine-instance determinism (M5 T4c) on this stack"
    ),
)


def _sampler_seeded_on_off(make_llm, prompt, n, monkeypatch, capfd):
    reqs = [([prompt], _seeded_params(n))]
    on, on_logs = _run(
        make_llm, {**V1_RUNNER, SAMPLER_SWITCH: None}, reqs, monkeypatch, capfd
    )
    off, off_logs = _run(
        make_llm, {**V1_RUNNER, SAMPLER_SWITCH: "0"}, reqs, monkeypatch, capfd
    )
    # Hard failures even under the xfail below (it only covers AssertionError).
    if SAMPLER_ON_LOG not in on_logs:
        raise NotEngagedError("K16 never sampled the seeded request")
    if SAMPLER_ON_LOG in off_logs:
        raise NotEngagedError("K16 sampled with VLLM_SM70_FLINT_SAMPLER=0")
    assert on == off, "seeded tokens differ with VLLM_SM70_FLINT_SAMPLER=0"


@pytest.mark.gpu
@one_gpu
def test_g1_tiny_sampler_top1_equals_greedy(tmp_path, monkeypatch, capfd):
    """G1 (1 GPU): tiny Flint TP1, V1 runner, CUDA graphs: in the K16 engine,
    top_k=1 (K16) gives upstream greedy's tokens; K16 absent when killed."""
    path = _tiny_flint_dir(tmp_path)
    _sampler_exact(
        lambda: _tiny_llm(path), _tiny_prompts(), 16, monkeypatch, capfd, tp=1
    )


@pytest.mark.gpu
@one_gpu
@seeded_parity_unverified
def test_g2_tiny_sampler_seeded_on_off(tmp_path, monkeypatch, capfd):
    """G2 (1 GPU): one seeded top-k 20 / top-p 0.95 request, K16 on vs off."""
    path = _tiny_flint_dir(tmp_path)
    prompt = _tiny_prompts()[0]
    _sampler_seeded_on_off(lambda: _tiny_llm(path), prompt, 32, monkeypatch, capfd)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_g3_ckpt_tp2_sampler_top1_equals_greedy(monkeypatch, capfd):
    """G3 (2 GPUs): checkpoint TP2, V1 runner, CUDA graphs: as G1, with the
    K16 lines once on each rank."""
    _sampler_exact(_ckpt_llm, PROMPTS, 64, monkeypatch, capfd, tp=2)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
@seeded_parity_unverified
def test_g4_ckpt_tp2_sampler_seeded_on_off(monkeypatch, capfd):
    """G4 (2 GPUs): checkpoint TP2: one seeded top-k 20 / top-p 0.95 request,
    K16 on vs off."""
    _sampler_seeded_on_off(_ckpt_llm, PROMPTS[1], 64, monkeypatch, capfd)
