# SPDX-License-Identifier: Apache-2.0
"""Tests for M10a: FlintSm70 takes the K13 fused decode router.

The router-only split of #66 (M10 router + sampler). K13 needs no model
declaration: U3's platform getter (``Sm70CudaPlatform.get_moe_router_provider``,
PR #43) admits each MoE runner by structure (``vllm_sm70.router._rejection``).
It replaces the provider the fork binds per model (fork cf9492a
vllm/models/flint/model.py:227-229).

CPU: the gate M5's FlintSm70SparseMoeBlock builds, with the router M5's
  FusedMoEFactory arguments produce, passes K13's admission; so does the real
  MoERunner M5's block builds through UP's FusedMoEFactory on a mocked SM70
  CUDA platform; the kill switch ("0" only) makes the platform getter return
  None, and kernel_warmup then decides no route.
GPU (``-m gpu``): greedy tokens identical with VLLM_SM70_FLINT_FUSED_ROUTER
  on and off, on a tiny random Flint (TP1, router-shaped: hidden 2048,
  E256/K6) and on the checkpoint at TP2 (FLINT_SM70_CHECKPOINT; both V100s,
  ``vllm`` stopped). Each "on" run also checks the worker log shows K13 was
  admitted. Skipped without SM70 GPUs.
"""

from __future__ import annotations

import gc
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_sm70.models import flint_sm70

ROUTER_SWITCH = "VLLM_SM70_FLINT_FUSED_ROUTER"

# Log line (vllm_sm70/router.py admit_flint_fused_router) that shows K13 ran.
ROUTER_ON_LOG = "vllm_sm70: fused E256/K6 sigmoid router enabled"
# K13 prepare_model (router.py, from kernel_warmup before capture), once per
# rank. The layer count is the strong check: fewer than the model's MoE
# layers means a partial non-admission (per-layer reasons are DEBUG only).
ROUTER_ADMITTED_LOG = re.compile(r"vllm_sm70: fused router admitted on (\d+) layers")
CKPT_MOE_LAYERS = 47  # checkpoint: 49 layers, 47 with MoE (mlp.gate.weight)


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------
@pytest.fixture
def platform_cls(monkeypatch):
    """Sm70CudaPlatform on a CPU host (as the K13 CPU tests import it)."""
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
    with set_default_torch_dtype(torch.float16):  # --dtype half, as the loader
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
    admission accepts it together with the router M5's arguments produce.
    The runner's other fields are written out here; the next test takes them
    from the real MoERunner instead."""
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


def _sm70_cuda_platform():
    """vLLM's platform as a CUDA device at cc 7.0 (no GPU needed), so
    GateLinear and the router factory take their SM70 branches."""
    from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

    class _Sm70Cuda(Platform):
        _enum = PlatformEnum.CUDA
        device_name = "cuda"
        device_type = "cuda"

        @classmethod
        def get_device_capability(cls, device_id: int = 0):
            return DeviceCapability(7, 0)

    return _Sm70Cuda()


@pytest.fixture
def parallel_groups(monkeypatch):
    """TP1 stand-ins for every group FusedMoEFactory asks for."""
    import vllm.distributed.parallel_state as ps

    for name in ("_TP", "_DP", "_PCP", "_EP", "_PP", "_DCP", "_WORLD"):
        group = SimpleNamespace(world_size=1, rank_in_group=0, rank=0, local_rank=0)
        monkeypatch.setattr(ps, name, group)


def test_k13_admits_the_m5_moe_runner_on_sm70(monkeypatch, parallel_groups, router_mod):
    """M5's FlintSm70SparseMoeBlock built unmodified through UP's
    FusedMoEFactory -> MoERunner, on a mocked SM70 CUDA platform: K13's
    _rejection() returns None for the runner it builds (gate, router,
    _fse_fuse_gate, routed_input_transform, enable_dbo, moe_config as UP
    sets them); prepare_model() admits it from the block's module tree, and
    the route passes try_route's own checks on an FP16 [2, 2048] input.

    Stubbed: the routed-expert quant method is a Sm70MoEMethodBase without
    weights (on the checkpoint it is M4's ModelOptNvFp4Sm70MoEMethod, which
    inherits is_monolithic/topk_indices_dtype from that base, checked below),
    the TP1 process groups, U3's presence, the device check (weights are on
    the CPU) and the op.
    """
    import vllm.platforms
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe import routed_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from vllm.models.flint.configs import FlintConfig
    from vllm_sm70.fused_moe.method import Sm70MoEMethodBase
    from vllm_sm70.models.flint_sm70 import model as m
    from vllm_sm70.quant.nvfp4_moe import ModelOptNvFp4Sm70MoEMethod

    # M4's method keeps the base's runner-facing answers (no override).
    for name in ("is_monolithic", "topk_indices_dtype"):
        assert getattr(ModelOptNvFp4Sm70MoEMethod, name) is getattr(
            Sm70MoEMethodBase, name
        )

    class _NoWeightsSm70Method(Sm70MoEMethodBase):
        def create_weights(self, layer, **kw):
            pass

        def apply(self, *args, **kw):
            raise NotImplementedError

    monkeypatch.setattr(
        routed_experts.RoutedExperts,
        "_get_quant_method",
        lambda self, prefix, quant_config, moe: _NoWeightsSm70Method(moe),
    )
    monkeypatch.delenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", raising=False)
    # No CUDA streams on a CPU host: SharedExperts would create one for the
    # shared expert (UP runner/shared_experts.py); irrelevant to admission.
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
    # Mock SM70 only after VllmConfig(), which would otherwise run CUDA
    # platform hooks; the layers below read the platform as they are built.
    monkeypatch.setattr(vllm.platforms, "_current_platform", _sm70_cuda_platform())
    # The loader builds the model under the model dtype (UP model_loader/
    # base_loader.py:73): FP16 here, so only force_fp32_compute keeps FP32.
    with set_current_vllm_config(cfg), set_default_torch_dtype(torch.float16):
        block = m.FlintSm70SparseMoeBlock(
            FlintConfig(), None, prefix="model.layers.1.mlp"
        )

    runner = block.experts
    assert type(runner) is MoERunner
    assert runner.gate is block.gate
    assert type(runner.router) is FusedTopKRouter
    assert block.gate.weight.dtype == torch.float32  # force_fp32_compute at cc 7.0
    assert not block.gate.allow_specialized_router_gemm

    calls = []
    monkeypatch.setattr(router_mod, "_is_exact_sm70", lambda device: True)
    monkeypatch.setattr(
        router_mod, "_flint_route_op", lambda: lambda *a: calls.append(a)
    )
    assert router_mod._rejection(runner) is None
    # kernel_warmup's prepare_model finds the runner in M5's module tree
    # (U3 itself is a patch the CPU venv lacks, hence _u3_present).
    monkeypatch.setattr(router_mod, "_u3_present", lambda: True)
    assert router_mod.prepare_model(block) == 1
    route = router_mod.flint_router_provider(runner)  # the decision cached
    assert isinstance(route, router_mod.FlintFusedRouter)
    assert route.weight is block.gate.weight
    hidden = torch.zeros(2, 2048, dtype=torch.float16)
    weights, ids = route.try_route(hidden)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert tuple(ids.shape) == (2, 6) and len(calls) == 1


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


@pytest.mark.parametrize("router_off", [False, True])
def test_kernel_warmup_decides_routes_only_when_on(
    platform_cls, router_mod, monkeypatch, router_off
):
    """kernel_warmup (U1, before capture) decides the routes through
    prepare_model; with the kill switch it decides none."""
    if router_off:
        monkeypatch.setenv(ROUTER_SWITCH, "0")
    decided = []
    monkeypatch.setattr(router_mod, "_u3_present", lambda: True)
    model = torch.nn.Module()
    model.gate = torch.nn.Identity()  # stands in for one MoE runner with a gate
    monkeypatch.setattr(router_mod, "_is_moe_runner", lambda mod: mod is model)
    monkeypatch.setattr(
        router_mod, "flint_router_provider", lambda r: decided.append(r)
    )
    worker = SimpleNamespace(get_model=lambda: model)
    assert platform_cls.kernel_warmup(worker) is None
    assert decided == ([] if router_off else [model])
    # A worker without get_model is skipped (nothing to prepare).
    assert platform_cls.kernel_warmup(object()) is None


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
        max_num_seqs=2,  # decode at M <= 2, the size K13 routes
        seed=0,
    )


def _greedy_params(n):
    from vllm import SamplingParams

    return SamplingParams(temperature=0, max_tokens=n)


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


def _router_on_off(make_llm, prompts, n, monkeypatch, capfd, *, moe_layers, tp):
    reqs = [(prompts, _greedy_params(n))]
    on, on_logs = _run(make_llm, {ROUTER_SWITCH: None}, reqs, monkeypatch, capfd)
    off, off_logs = _run(make_llm, {ROUTER_SWITCH: "0"}, reqs, monkeypatch, capfd)
    assert on_logs.count(ROUTER_ON_LOG) == tp, "K13 enabled line not once per rank"
    admitted = [int(m) for m in ROUTER_ADMITTED_LOG.findall(on_logs)]
    assert admitted == [moe_layers] * tp, (
        f"K13 admitted per rank {admitted}, expected {moe_layers} on each of {tp}"
    )
    assert ROUTER_ON_LOG not in off_logs
    assert not ROUTER_ADMITTED_LOG.search(off_logs)
    assert all(len(t) == n for t in on[0])
    assert on == off


@pytest.mark.gpu
@one_gpu
def test_g1_tiny_router_on_off_greedy_identical(tmp_path, monkeypatch, capfd):
    """G1 (1 GPU, #66's G1): tiny Flint TP1, CUDA graphs: K13 on vs
    VLLM_SM70_FLINT_FUSED_ROUTER=0, greedy."""
    path = _tiny_flint_dir(tmp_path)
    # 2 layers, mlp_only_layers=[0]: one MoE layer.
    _router_on_off(
        lambda: _tiny_llm(path),
        _tiny_prompts(),
        16,
        monkeypatch,
        capfd,
        moe_layers=1,
        tp=1,
    )


@pytest.mark.gpu
@two_gpus
@needs_ckpt
def test_g2_ckpt_tp2_router_on_off_greedy_identical(monkeypatch, capfd):
    """G2 (2 GPUs, #66's G4): checkpoint TP2, CUDA graphs: K13 on vs off,
    greedy."""
    _router_on_off(
        _ckpt_llm, PROMPTS, 64, monkeypatch, capfd, moe_layers=CKPT_MOE_LAYERS, tp=2
    )
