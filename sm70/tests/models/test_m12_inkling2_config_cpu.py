# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 engine-config hook (inkling2_sm70/config.py),
the ``inkling2`` section of Sm70Config, and the Flint fast paths' answers
for Inkling 2 shapes.

CPU only. The hook's refusals and its attention policy run at a faked cc 7.0
or 8.0. A fake Inkling 2 module tree (hidden 2304, 128 experts, top-8, biased
FP32 gate, soft-capped head: the Inkling 2 Mini config.json values at TP2)
goes through every preparation ``Sm70CudaPlatform.kernel_warmup`` runs and
through each Flint admission; each returns a reason and none throws.
"""

import sys
import types
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")
import torch

import vllm.platforms
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum
from vllm_sm70 import _ext, envs
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.config import Sm70Config, clear_cache, compute_hash, sm70_config
from vllm_sm70.models import inkling2_sm70
from vllm_sm70.models.inkling2_sm70 import config as icfg

# Inkling 2 Mini (tests/models/data/inkling2_mini/config.json), TP2.
HIDDEN = 2304
EXPERTS = 128
TOP_K = 8
LOCAL_INTERMEDIATE = 576
VOCAB = 200058
SOFT_CAP = 30.0


class _FakeCuda(Platform):
    _enum = PlatformEnum.CUDA
    device_name = "cuda"
    device_type = "cuda"
    capability: DeviceCapability | None = None

    @classmethod
    def get_device_capability(cls, device_id: int = 0):
        return cls.capability


@pytest.fixture
def device(monkeypatch):
    def _set(major, minor, ext=True):
        cls = type("Dev", (_FakeCuda,), {})
        cls.capability = DeviceCapability(major, minor)
        monkeypatch.setattr(vllm.platforms, "_current_platform", cls())
        sm70 = (major, minor) == (7, 0)
        monkeypatch.setattr(sm70_platform, "all_visible_gpus_are_sm70", lambda: sm70)
        monkeypatch.setattr(_ext, "is_loaded", lambda: ext)
        sm70_platform.is_sm70_ready.cache_clear()

    yield _set
    sm70_platform.is_sm70_ready.cache_clear()


@pytest.fixture
def route_log(monkeypatch):
    """The hook's info lines, formatted; the once-per-process flag reset."""
    lines: list[str] = []
    monkeypatch.setattr(icfg, "_logged", False)
    monkeypatch.setattr(
        icfg.logger, "info", lambda msg, *args, **kw: lines.append(msg % args)
    )
    clear_cache()
    yield lines
    clear_cache()


def _vllm_config(
    dtype=torch.float16, backend=None, spec=None, prompt_embeds=False, sm70=None
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            enable_prompt_embeds=prompt_embeds,
            architecture=inkling2_sm70.ARCH,
            get_hidden_size=lambda: HIDDEN,
        ),
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=32),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, disable_custom_all_reduce=False
        ),
        speculative_config=spec,
        additional_config=None if sm70 is None else {"sm70": sm70},
        kernel_config=None,
    )


# --- the hook --------------------------------------------------------------


@pytest.mark.parametrize(
    "cc, kw, match",
    [
        ((8, 0), {}, "compute capability 7.0"),
        ((7, 5), {}, "compute capability 7.0"),
        ((7, 0), {"dtype": torch.bfloat16}, "--dtype half"),
        ((7, 0), {"dtype": torch.float32}, "--dtype half"),
        ((7, 0), {"spec": object()}, "speculative decoding"),
        ((7, 0), {"prompt_embeds": True}, "prompt embeddings"),
    ],
)
def test_hook_refuses(device, route_log, cc, kw, match):
    device(*cc)
    with pytest.raises(ValueError, match=match) as e:
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(_vllm_config(**kw))
    assert inkling2_sm70.ARCH in str(e.value)
    assert route_log == []


def test_hook_refuses_on_cpu_platform(monkeypatch, route_log):
    cpu = SimpleNamespace(is_cuda=lambda: False)
    monkeypatch.setattr(vllm.platforms, "_current_platform", cpu)
    with pytest.raises(ValueError, match="compute capability 7.0 GPU.*got None"):
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(_vllm_config())


def test_hook_selects_no_attention_backend(device, route_log):
    """The model owns attention: the hook never picks Flash-V100 (CUSTOM) or
    TRITON_ATTN, and an explicit backend is refused, not carried along."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    device(7, 0)
    vc = _vllm_config()
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(vc)
    assert vc.attention_config.backend is None
    for backend in (AttentionBackendEnum.CUSTOM, AttentionBackendEnum.TRITON_ATTN):
        with pytest.raises(ValueError, match=f"--attention-backend {backend.name}"):
            icfg.Inkling2Sm70ModelConfig.verify_and_update_config(
                _vllm_config(backend=backend)
            )


def test_hook_logs_routes_once(device, route_log):
    device(7, 0)
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(_vllm_config())
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(_vllm_config())
    assert len(route_log) == 1
    line = route_log[0]
    assert "model-owned attention" in line and "state dump off" in line


def test_hook_logs_the_visit_dump(device, route_log):
    device(7, 0)
    vc = _vllm_config(sm70={"inkling2": {"visit_dump_dir": "/dump"}})
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(vc)
    assert "state dump to /dump" in route_log[0]


def test_hook_is_reachable_lazily():
    assert inkling2_sm70.Inkling2Sm70ModelConfig is icfg.Inkling2Sm70ModelConfig
    assert icfg.ARCH == inkling2_sm70.ARCH


# --- Sm70Config section ----------------------------------------------------


def test_inkling2_section_defaults_and_parsing():
    clear_cache()
    assert Sm70Config().inkling2.visit_dump_dir is None
    assert sm70_config(_vllm_config()).inkling2.visit_dump_dir is None
    vc = _vllm_config(sm70={"inkling2": {"visit_dump_dir": "/dump"}})
    assert sm70_config(vc).inkling2.visit_dump_dir == "/dump"
    with pytest.raises(ValueError, match="visit_dump"):
        Sm70Config.model_validate({"inkling2": {"visit_dump": True}})
    clear_cache()


def test_inkling2_section_is_config_only():
    """No environment variable names the section, and the dump directory
    does not enter the graph hash."""
    assert not [n for n in envs.environment_variables() if "INKLING" in n]
    on = Sm70Config.model_validate({"inkling2": {"visit_dump_dir": "/dump"}})
    assert compute_hash(on) == compute_hash(Sm70Config())


# --- a fake Inkling 2 module tree ------------------------------------------


class _FakeGate(torch.nn.Module):
    """use_gate_bias: a biased FP32 [experts, hidden] gate."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(EXPERTS, HIDDEN))
        self.bias = torch.nn.Parameter(torch.zeros(EXPERTS))


class _FakeMoE(torch.nn.Module):
    """Model-owned experts (not a vLLM FusedMoE), TP2 shard, gate/up rows."""

    layer_name = "model.layers.0.mlp"

    def __init__(self):
        super().__init__()
        self.gate = _FakeGate()
        self.router = object()
        self.w13 = torch.nn.Parameter(
            torch.zeros(EXPERTS, 2 * LOCAL_INTERMEDIATE, 1, dtype=torch.float16)
        )
        self.w2 = torch.nn.Parameter(
            torch.zeros(EXPERTS, 1, LOCAL_INTERMEDIATE, dtype=torch.float16)
        )


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.Identity()
        self.mlp = _FakeMoE()


class _FakeInkling2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.lm_head = torch.nn.Linear(HIDDEN, 16, bias=False, dtype=torch.float16)
        self.logits_processor = SimpleNamespace(
            soft_cap=SOFT_CAP, scale=1.0, org_vocab_size=VOCAB
        )


@pytest.fixture
def model():
    return _FakeInkling2()


@pytest.fixture
def router_mod(monkeypatch):
    from vllm_sm70 import router

    monkeypatch.delenv("VLLM_SM70_FLINT_FUSED_ROUTER", raising=False)
    router._logged.clear()
    router.provider.cache_clear()
    # U3 is a patch the CPU venv lacks.
    monkeypatch.setattr(router, "_u3_present", lambda: True)
    yield router
    router.provider.cache_clear()


def test_router_admits_no_inkling2_layer(model, router_mod, monkeypatch):
    # As built: the experts are no MoERunner, so nothing is asked.
    assert router_mod.prepare_model(model) == 0
    # Were they taken for runners, each gate would be rejected with a reason.
    monkeypatch.setattr(
        router_mod, "_is_moe_runner", lambda mod: isinstance(mod, _FakeMoE)
    )
    assert router_mod._rejection(model.layers[0].mlp) == "the gate is not a GateLinear"
    assert router_mod.prepare_model(model) == 0
    for layer in model.layers:
        assert router_mod.flint_router_provider(layer.mlp) is None


def test_router_rejects_an_inkling2_gate_linear(router_mod, monkeypatch):
    """A real GateLinear with Inkling 2's shape and bias is not K13's GEMM."""
    import vllm.distributed.parallel_state as ps
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    # A stand-in TP group: the layer is built without torch.distributed.
    monkeypatch.setattr(ps, "_TP", SimpleNamespace(world_size=1, rank_in_group=0))
    gate = GateLinear(
        HIDDEN,
        EXPERTS,
        bias=True,
        out_dtype=torch.float32,
        params_dtype=torch.float32,
    )
    runner = SimpleNamespace(gate=gate, router=object(), layer_name="mlp")
    assert router_mod._rejection(runner) == (
        "the gate is not an unbiased FP32 [256, 2048] GEMM"
    )
    assert router_mod.admit_flint_fused_router(runner) is None


def test_k12_decode_rejects_the_inkling2_contract():
    from vllm_sm70.ops import flint_moe

    contract = (EXPERTS, HIDDEN, LOCAL_INTERMEDIATE, TOP_K)
    reason = flint_moe.contract_rejection(contract)
    assert reason is not None and "not the compiled Flint TP2 contract" in reason
    x = torch.zeros(1, HIDDEN, dtype=torch.float16)
    weights = torch.zeros(1, TOP_K)
    ids = torch.zeros(1, TOP_K, dtype=torch.int32)
    assert flint_moe.call_rejection(x, weights, ids) is not None


def test_ar_rmsnorm_fusion_rejects_inkling2():
    from vllm_sm70 import ar_rms_fusion

    reason = ar_rms_fusion.rejection(_vllm_config(), 2)
    assert reason is not None and "Flint-only" in reason
    assert inkling2_sm70.ARCH in reason


@pytest.mark.parametrize(
    "k, n",
    [
        (HIDDEN, 2 * LOCAL_INTERMEDIATE),  # expert gate/up
        (LOCAL_INTERMEDIATE, HIDDEN),  # expert down
        (HIDDEN, HIDDEN),
    ],
)
def test_nvfp4_gemv_table_rejects_inkling2_shapes(monkeypatch, k, n):
    from vllm_sm70.quant import nvfp4_linear as nl

    assert (k, n) not in nl.MODELOPT_NVFP4_GEMV_SHAPES
    assert not any(HIDDEN in shape for shape in nl.MODELOPT_NVFP4_GEMV_SHAPES)
    monkeypatch.setattr(nl, "has_tm_op", lambda name: True)
    state = nl.SM70TurboMindLinearState(
        weight=torch.zeros(k, n // 8, dtype=torch.uint8),
        scales=torch.ones(k, n // nl.NVFP4_GROUP_SIZE, dtype=torch.float16),
        group_size=nl.NVFP4_GROUP_SIZE,
        k_ld=32 * k,
        q_ld=n,
        output_size=n,
    )
    assert nl.admit_modelopt_nvfp4_gemv(state, enabled=True) is False
    assert state.tm_gemv is False and state.provider == nl.GEMM_OP


class _Runner:  # weak-referenceable, as the sampler keeps a weakref
    device = "cpu"

    def __init__(self, hook):
        self._hidden_state_sampler = hook


def _worker(model, hook):
    runner = _Runner(hook)
    return SimpleNamespace(
        model_runner=runner, vllm_config=_vllm_config(), get_model=lambda: model
    )


def test_sampler_prepares_and_rejects_inkling2(model):
    from vllm_sm70 import sample, sample_v1

    hook = sample_v1.Sm70HiddenStateSampler()
    worker = _worker(model, hook)
    sample_v1.prepare_worker(worker)
    assert hook.prepared
    hidden = torch.zeros(1, HIDDEN, dtype=torch.float16)
    for all_greedy in (False, True):
        metadata = SimpleNamespace(all_greedy=all_greedy)
        reason = hook.reject_reason(model, hidden, metadata, SimpleNamespace(), None)
        assert reason == "model"
        assert hook.try_sample(model, hidden, metadata, SimpleNamespace(), None) is None
    # Opting in later still leaves the top-k route out: the head is capped.
    assert (
        sample.lm_head_reject_reason(model.logits_processor, model.lm_head, 2)
        == "logits_processor_transform"
    )


def test_kernel_warmup_does_not_raise_for_inkling2(model, router_mod, monkeypatch):
    """Every preparation kernel_warmup runs, on the fake tree: none throws
    (a thrown preparation stops startup) and no route is admitted."""
    from vllm_sm70 import sample_v1

    if "vllm._C_stable_libtorch" not in sys.modules:
        try:
            import vllm._C_stable_libtorch  # noqa: F401
        except ImportError:  # vllm.platforms.cuda imports it (UP :23)
            monkeypatch.setitem(
                sys.modules,
                "vllm._C_stable_libtorch",
                types.ModuleType("vllm._C_stable_libtorch"),
            )
    from vllm_sm70._cuda_platform import Sm70CudaPlatform

    for name in ("VLLM_SM70_FLINT_SAMPLER", "VLLM_SM70_GREEDY_TOKEN_FASTPATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        router_mod, "_is_moe_runner", lambda mod: isinstance(mod, _FakeMoE)
    )
    hook = sample_v1.Sm70HiddenStateSampler()
    assert Sm70CudaPlatform.kernel_warmup(_worker(model, hook)) is None
    assert hook.prepared
    for layer in model.layers:
        assert router_mod.flint_router_provider(layer.mlp) is None
