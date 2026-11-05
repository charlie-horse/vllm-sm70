# SPDX-License-Identifier: Apache-2.0
"""Tests for M10: FlintSm70 opts into the K13 fused router and K16 sampler.

CPU: the M5 model declares the K16 opt-in; the MoE block M5 builds (gate and
  router arguments) passes K13's admission with its FP32 gate; the Flint
  TP2 LM head passes K16's LM-head admission; each kill switch ("0" only)
  makes its platform getter return None and keeps kernel_warmup from
  preparing that path.
GPU (``-m gpu``): greedy tokens identical with each switch on and off, on a
  tiny random Flint (TP1, router-shaped: hidden 2048, E256/K6) and on the
  checkpoint at TP2 (FLINT_SM70_CHECKPOINT; both V100s, ``vllm`` stopped).
  Each "on" run also checks the worker log shows the path engaged.

K16 is a V1-runner hook (patches/0005, U4); v0.33 picks Model Runner V2
when nothing rules it out, so the sampler tests set
VLLM_USE_V2_MODEL_RUNNER=0. Greedy requests (temperature 0) always take the
upstream sampler (K16 rejects ``sampling_mode``), so the sampler tests also
decode with top_k=1 at temperature 1, which K16 does sample, and a seeded
top-k 20 / top-p 0.95 request, which K16 draws with the same RNG.
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

ROUTER_SWITCH = "VLLM_SM70_FLINT_FUSED_ROUTER"
SAMPLER_SWITCH = "VLLM_SM70_FLINT_SAMPLER"

# Log lines (vllm_sm70/router.py, vllm_sm70/sample.py) that show a path ran.
ROUTER_ON_LOG = "vllm_sm70: fused E256/K6 sigmoid router enabled"
SAMPLER_ON_LOG = "SM70 top-k candidate sampler enabled."


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------
@pytest.fixture
def platform_cls(monkeypatch):
    """Sm70CudaPlatform on a CPU host (as the K13/K16 CPU tests import it)."""
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
def router_mod(monkeypatch):
    from vllm_sm70 import router

    monkeypatch.delenv(ROUTER_SWITCH, raising=False)
    router._logged.clear()
    router.provider.cache_clear()
    yield router
    router.provider.cache_clear()


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


def _m5_moe_block(monkeypatch, config):
    """M5's FlintSm70SparseMoeBlock with the real GateLinear; the experts and
    shared expert are stubbed and the FusedMoEFactory arguments recorded."""
    from vllm_sm70.models.flint_sm70 import model as m

    factory_kwargs = {}

    def factory(**kw):
        factory_kwargs.update(kw)
        return torch.nn.Identity()

    monkeypatch.setattr(m, "FusedMoEFactory", factory)
    monkeypatch.setattr(m, "FlintSm70MLP", lambda **kw: torch.nn.Identity())
    block = m.FlintSm70SparseMoeBlock(config, None, prefix="model.layers.1.mlp")
    return block, factory_kwargs


def _router_for(monkeypatch, kw):
    """The router UP's factory builds on CUDA from M5's FusedMoEFactory args."""
    from vllm.model_executor.layers.fused_moe.router import router_factory

    monkeypatch.setattr(router_factory.current_platform, "is_cpu", lambda: False)
    return router_factory.create_fused_moe_router(
        top_k=kw["top_k"],
        global_num_experts=kw["num_experts"],
        renormalize=kw["renormalize"],
        use_grouped_topk=kw["use_grouped_topk"],
        scoring_func=kw["scoring_func"],
        e_score_correction_bias=kw["e_score_correction_bias"],
    )


def test_k13_admits_the_m5_gate(monkeypatch, tp, router_mod):
    """The gate M5 builds is FP32 [256, 2048] with FP32 output, and K13's
    admission accepts it together with the router M5's arguments produce."""
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.models.flint.configs import FlintConfig

    monkeypatch.delenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", raising=False)
    block, kw = _m5_moe_block(monkeypatch, FlintConfig())  # Flint's real shapes
    gate = block.gate
    assert kw["gate"] is gate
    assert gate.weight.dtype == torch.float32
    assert tuple(gate.weight.shape) == (256, 2048)
    assert gate.out_dtype == torch.float32 and gate.bias is None
    assert gate.is_unquantized

    router = _router_for(monkeypatch, kw)
    assert type(router) is FusedTopKRouter
    runner = SimpleNamespace(
        layer_name=kw["prefix"],
        gate=gate,
        router=router,
        _fse_fuse_gate=False,  # UP: gate and shared_expert_gate; Flint has none
        routed_input_transform=None,
        enable_dbo=False,
        _quant_method=SimpleNamespace(is_monolithic=False, topk_indices_dtype=None),
        moe_config=SimpleNamespace(dp_size=1, pcp_size=1, use_ep=False),
    )
    calls = []
    monkeypatch.setattr(router_mod, "_is_exact_sm70", lambda device: True)
    monkeypatch.setattr(router_mod, "_flint_route_op", lambda: calls.append)
    assert router_mod._rejection(runner) is None
    route = router_mod.admit_flint_fused_router(runner)
    assert isinstance(route, router_mod.FlintFusedRouter)
    assert route.weight is gate.weight

    monkeypatch.setenv(ROUTER_SWITCH, "0")
    assert router_mod.admit_flint_fused_router(runner) is None


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
def test_router_kill_switch_only_zero(platform_cls, router_mod, monkeypatch, raw, on):
    if raw is not None:
        monkeypatch.setenv(ROUTER_SWITCH, raw)
    provider = platform_cls.get_moe_router_provider()
    assert (provider is router_mod.flint_router_provider) is on
    assert (provider is None) is not on


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


def test_switches_are_independent(platform_cls, router_mod, monkeypatch):
    monkeypatch.setenv(ROUTER_SWITCH, "0")
    monkeypatch.delenv(SAMPLER_SWITCH, raising=False)
    assert platform_cls.get_moe_router_provider() is None
    assert platform_cls.get_hidden_state_sampler() is not None
    router_mod.provider.cache_clear()
    monkeypatch.delenv(ROUTER_SWITCH)
    monkeypatch.setenv(SAMPLER_SWITCH, "0")
    assert platform_cls.get_moe_router_provider() is router_mod.flint_router_provider
    assert platform_cls.get_hidden_state_sampler() is None


@pytest.mark.parametrize("router_off", [False, True])
@pytest.mark.parametrize("sampler_off", [False, True])
def test_kernel_warmup_prepares_only_what_is_on(
    platform_cls, router_mod, monkeypatch, router_off, sampler_off
):
    """kernel_warmup (U1, before capture) decides the routes and binds the
    sampler; a killed path is neither decided nor prepared."""
    from vllm_sm70 import sample_v1

    if router_off:
        monkeypatch.setenv(ROUTER_SWITCH, "0")
    if sampler_off:
        monkeypatch.setenv(SAMPLER_SWITCH, "0")
    else:
        monkeypatch.delenv(SAMPLER_SWITCH, raising=False)
    decided, prepared = [], []
    monkeypatch.setattr(router_mod, "_u3_present", lambda: True)
    model = torch.nn.Module()
    model.gate = torch.nn.Identity()  # stands in for one MoE runner with a gate
    monkeypatch.setattr(router_mod, "_is_moe_runner", lambda mod: mod is model)
    monkeypatch.setattr(
        router_mod, "flint_router_provider", lambda r: decided.append(r)
    )
    monkeypatch.setattr(sample_v1, "prepare_worker", prepared.append)
    worker = SimpleNamespace(get_model=lambda: model)
    assert platform_cls.kernel_warmup(worker) is None
    assert (decided == []) is router_off
    assert (prepared == [worker]) is not sampler_off


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
    """#36's tiny random Flint, router-shaped so K13 admits it: hidden 2048,
    256 experts, sigmoid renormalized top-6. One MoE layer (layer 1)."""
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
        max_num_seqs=2,  # decode at M <= 2, the size both paths admit
        seed=0,
    )


def _greedy_params(n):
    from vllm import SamplingParams

    return SamplingParams(temperature=0, max_tokens=n)


def _top1_params(n):
    from vllm import SamplingParams

    return SamplingParams(temperature=1.0, top_k=1, max_tokens=n)


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


def _router_on_off(make_llm, prompts, n, monkeypatch, capfd):
    reqs = [(prompts, _greedy_params(n))]
    on, on_logs = _run(make_llm, {ROUTER_SWITCH: None}, reqs, monkeypatch, capfd)
    off, off_logs = _run(make_llm, {ROUTER_SWITCH: "0"}, reqs, monkeypatch, capfd)
    assert ROUTER_ON_LOG in on_logs, "K13 was not admitted on any layer"
    assert ROUTER_ON_LOG not in off_logs
    assert all(len(t) == n for t in on[0])
    assert on == off


def _sampler_on_off(make_llm, prompts, n, monkeypatch, capfd):
    # Model Runner V1: K16's hook (U4) is in the V1 runner only.
    reqs = [(prompts, _greedy_params(n)), (prompts, _top1_params(n))]
    base = {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    on, on_logs = _run(
        make_llm, {**base, SAMPLER_SWITCH: None}, reqs, monkeypatch, capfd
    )
    off, off_logs = _run(
        make_llm, {**base, SAMPLER_SWITCH: "0"}, reqs, monkeypatch, capfd
    )
    assert SAMPLER_ON_LOG in on_logs, "K16 never sampled (top_k=1 rows)"
    assert SAMPLER_ON_LOG not in off_logs
    assert all(len(t) == n for t in on[0])
    assert on[0] == off[0], "greedy differs with VLLM_SM70_FLINT_SAMPLER=0"
    assert on[1] == off[1], "top_k=1 differs with VLLM_SM70_FLINT_SAMPLER=0"


def _sampler_seeded_on_off(make_llm, prompt, n, monkeypatch, capfd):
    reqs = [([prompt], _seeded_params(n))]
    base = {"VLLM_USE_V2_MODEL_RUNNER": "0"}
    on, on_logs = _run(
        make_llm, {**base, SAMPLER_SWITCH: None}, reqs, monkeypatch, capfd
    )
    off, _ = _run(make_llm, {**base, SAMPLER_SWITCH: "0"}, reqs, monkeypatch, capfd)
    assert SAMPLER_ON_LOG in on_logs
    assert on == off


@pytest.mark.gpu
@one_gpu
def test_g1_tiny_router_on_off_greedy_identical(tmp_path, monkeypatch, capfd):
    """G1 (1 GPU): tiny Flint TP1, CUDA graphs: K13 on vs VLLM_SM70_FLINT_FUSED_ROUTER=0."""
    path = _tiny_flint_dir(tmp_path)
    _router_on_off(lambda: _tiny_llm(path), _tiny_prompts(), 16, monkeypatch, capfd)


@pytest.mark.gpu
@one_gpu
def test_g2_tiny_sampler_on_off_greedy_identical(tmp_path, monkeypatch, capfd):
    """G2 (1 GPU): tiny Flint TP1, V1 runner: K16 on vs VLLM_SM70_FLINT_SAMPLER=0,
    greedy and top_k=1."""
    path = _tiny_flint_dir(tmp_path)
    _sampler_on_off(lambda: _tiny_llm(path), _tiny_prompts(), 16, monkeypatch, capfd)


@pytest.mark.gpu
@one_gpu
def test_g3_tiny_sampler_seeded_identical(tmp_path, monkeypatch, capfd):
    """G3 (1 GPU): one seeded top-k 20 / top-p 0.95 request samples the same
    tokens with K16 on and off (K16 draws upstream's FP32 exponentials)."""
    path = _tiny_flint_dir(tmp_path)
    prompt = _tiny_prompts()[0]
    _sampler_seeded_on_off(lambda: _tiny_llm(path), prompt, 32, monkeypatch, capfd)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_g4_ckpt_tp2_router_on_off_greedy_identical(monkeypatch, capfd):
    """G4 (2 GPUs): checkpoint TP2, CUDA graphs: K13 on vs off, greedy."""
    _router_on_off(_ckpt_llm, PROMPTS, 64, monkeypatch, capfd)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_g5_ckpt_tp2_sampler_on_off_greedy_identical(monkeypatch, capfd):
    """G5 (2 GPUs): checkpoint TP2, V1 runner: K16 on vs off, greedy and top_k=1."""
    _sampler_on_off(_ckpt_llm, PROMPTS, 64, monkeypatch, capfd)


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_g6_ckpt_tp2_sampler_seeded_identical(monkeypatch, capfd):
    """G6 (2 GPUs): checkpoint TP2: one seeded top-k 20 / top-p 0.95 request,
    K16 on vs off."""
    _sampler_seeded_on_off(_ckpt_llm, PROMPTS[1], 64, monkeypatch, capfd)
