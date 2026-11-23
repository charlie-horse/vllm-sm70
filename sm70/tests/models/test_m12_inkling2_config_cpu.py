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

import json
import pathlib
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

DATA = pathlib.Path(__file__).parent / "data" / "inkling2_mini"
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
    dtype=torch.float16,
    backend=None,
    spec=None,
    prompt_embeds=False,
    sm70=None,
    cache_dtype="auto",
    block_size=None,
    pp=1,
    async_scheduling=None,
    cudagraph_mode=None,
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=dtype,
            enable_prompt_embeds=prompt_embeds,
            architecture=inkling2_sm70.ARCH,
            get_hidden_size=lambda: HIDDEN,
        ),
        attention_config=SimpleNamespace(backend=backend),
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            block_size=block_size or 16,
            user_specified_block_size=block_size is not None,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp, disable_custom_all_reduce=False
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
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
        ((7, 0), {"cache_dtype": "fp8"}, "--kv-cache-dtype fp8"),
        ((7, 0), {"block_size": 16}, "--block-size 32"),
        ((7, 0), {"pp": 2}, "pipeline parallelism"),
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


@pytest.mark.parametrize("switch", [None, "0"])
def test_hook_takes_a_staged_dspark_draft_only(device, route_log, monkeypatch, switch):
    from vllm_sm70.models.inkling2_sm70 import draft

    device(7, 0)
    monkeypatch.setattr(icfg, "tm_tuning_env", lambda *args: {})
    if switch is None:
        monkeypatch.delenv("VLLM_SM70_INKLING_DSPARK", raising=False)
    else:
        monkeypatch.setenv("VLLM_SM70_INKLING_DSPARK", switch)
    spec = SimpleNamespace(
        method="dspark",
        num_speculative_tokens=7,
        draft_model_config=SimpleNamespace(hf_config=draft.draft_config(DATA)),
    )
    cfg = _vllm_config(spec=spec)
    cfg.use_v2_model_runner = True
    if switch is None:
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
        v1 = _vllm_config(spec=spec)
        with pytest.raises(ValueError, match="VLLM_USE_V2_MODEL_RUNNER=1"):
            icfg.Inkling2Sm70ModelConfig.verify_and_update_config(v1)
        return
    with pytest.raises(ValueError, match="VLLM_SM70_INKLING_DSPARK=0"):
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)


@pytest.mark.parametrize("block_size", [None, 32])
@pytest.mark.parametrize("async_scheduling", [None, False, True])
@pytest.mark.parametrize("cache_dtype", ["auto", "float16"])
def test_hook_fixes_the_kv_plan(
    device, route_log, block_size, async_scheduling, cache_dtype
):
    device(7, 0)
    cfg = _vllm_config(
        block_size=block_size,
        async_scheduling=async_scheduling,
        cache_dtype=cache_dtype,
    )
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.cache_config.block_size == icfg.KV_BLOCK_SIZE == 32
    assert cfg.cache_config.user_specified_block_size is True
    assert cfg.cache_config.cache_dtype == cache_dtype
    # Async scheduling stays upstream's choice.
    assert cfg.scheduler_config.async_scheduling is async_scheduling
    word = {None: "upstream's default", False: "off", True: "on"}[async_scheduling]
    assert f"async scheduling {word};" in route_log[0]


@pytest.mark.parametrize("async_scheduling", [None, False, True])
def test_async_sched_switch_off(device, route_log, monkeypatch, async_scheduling):
    """VLLM_SM70_INKLING_ASYNC_SCHED=0: an unset value becomes off and an
    explicit --async-scheduling is refused."""
    monkeypatch.setenv("VLLM_SM70_INKLING_ASYNC_SCHED", "0")
    monkeypatch.setattr(icfg, "tm_tuning_env", lambda *args: {})
    device(7, 0)
    cfg = _vllm_config(async_scheduling=async_scheduling)
    if async_scheduling:
        with pytest.raises(ValueError, match="--no-async-scheduling") as e:
            icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
        assert inkling2_sm70.ARCH in str(e.value)
        assert route_log == []
        return
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.scheduler_config.async_scheduling is False
    assert "async scheduling off;" in route_log[0]


def test_hook_sets_the_turbomind_tuning_knobs(device, route_log, monkeypatch):
    """With FULL decode graphs and no knob set: tuning on for the expert
    stages up to the decode graph cap x top-k slots (tm_tune test)."""
    import os

    from vllm_sm70.models.inkling2_sm70 import moe_tm

    device(7, 0)
    for name in (*icfg.TM_TUNE_KNOBS, "VLLM_SM70_INKLING_TM_TUNE", DECODE_GRAPH):
        monkeypatch.setenv(name, "x")
        monkeypatch.delenv(name)
    monkeypatch.setattr(moe_tm, "turbomind", lambda: object())
    cfg = _vllm_config()
    cfg.model_config.hf_config = SimpleNamespace(num_experts_per_tok=TOP_K)
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.compilation_config.max_cudagraph_capture_size == 8
    assert {name: os.environ[name] for name in icfg.TM_TUNE_KNOBS} == {
        "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": "1",
        "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": str(8 * TOP_K),
        "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M": "0",
    }
    assert any("expert stages tuned up to 64 routed slots" in x for x in route_log)


BREAKABLE = "VLLM_USE_BREAKABLE_CUDAGRAPH"
DECODE_GRAPH = "VLLM_SM70_INKLING_DECODE_GRAPH"


def _set_env(monkeypatch, name, value):
    if value is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize(
    "switch, breakable, given, want",
    [
        (None, None, None, "FULL_DECODE_ONLY"),
        (None, "1", None, "FULL_DECODE_ONLY"),
        ("1", None, None, "FULL_DECODE_ONLY"),
        ("0", "1", None, "PIECEWISE"),
        ("0", None, None, "NONE"),
        ("0", "0", None, "NONE"),
        ("0", "true", None, "NONE"),
        (None, None, "FULL_DECODE_ONLY", "FULL_DECODE_ONLY"),
        (None, "1", "FULL_AND_PIECEWISE", "FULL_AND_PIECEWISE"),
        (None, None, "PIECEWISE", "PIECEWISE"),
        ("0", "1", "PIECEWISE", "PIECEWISE"),
        (None, "1", "NONE", "NONE"),
        (None, None, "NONE", "NONE"),
        ("0", None, "NONE", "NONE"),
    ],
)
def test_hook_picks_the_cuda_graph_mode(
    device, route_log, monkeypatch, switch, breakable, given, want
):
    """Decode-only FULL graphs by default; the kill switch restores the
    breakable PIECEWISE or NONE default."""
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, DECODE_GRAPH, switch)
    _set_env(monkeypatch, BREAKABLE, breakable)
    device(7, 0)
    cfg = _vllm_config(cudagraph_mode=given and CUDAGraphMode[given])
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode is CUDAGraphMode[want]
    assert f"cudagraph_mode {want};" in route_log[0]


def test_hook_skips_full_graphs_while_dumping_visits(device, route_log, tmp_path):
    from vllm.config.compilation import CUDAGraphMode

    device(7, 0)
    sm70 = {"inkling2": {"visit_dump_dir": str(tmp_path)}}
    cfg = _vllm_config(sm70=sm70)
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.compilation_config.cudagraph_mode is not CUDAGraphMode.FULL_DECODE_ONLY
    assert not cfg.compilation_config.cudagraph_mode.has_full_cudagraphs()


@pytest.mark.parametrize("given", [None, "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
@pytest.mark.parametrize(
    "sizes, max_size, max_num_seqs, want_max",
    [
        (None, None, None, icfg.DECODE_GRAPH_MAX_BATCH),
        (None, None, 256, icfg.DECODE_GRAPH_MAX_BATCH),
        (None, None, 1, 1),
        (None, None, 3, 3),
        ([1, 2], None, 256, None),
        ([1, 2, 8], None, 256, None),
        (None, 8, 256, 8),
        (None, 4, 256, 4),
    ],
)
def test_hook_caps_the_decode_graph_batch(
    device, route_log, monkeypatch, given, sizes, max_size, max_num_seqs, want_max
):
    """A mode with FULL decode graphs captures decode batches up to
    DECODE_GRAPH_MAX_BATCH requests (max_num_seqs if fewer) unless the
    capture sizes are set."""
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, DECODE_GRAPH, None)
    _set_env(monkeypatch, BREAKABLE, "1")
    device(7, 0)
    cfg = _vllm_config(cudagraph_mode=given and CUDAGraphMode[given])
    cfg.scheduler_config.max_num_seqs = max_num_seqs
    cfg.compilation_config.cudagraph_capture_sizes = sizes
    cfg.compilation_config.max_cudagraph_capture_size = max_size
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert icfg.DECODE_GRAPH_MAX_BATCH == 8
    assert cfg.compilation_config.max_cudagraph_capture_size == want_max
    assert cfg.compilation_config.cudagraph_capture_sizes == sizes


@pytest.mark.parametrize("given", [None, "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"])
@pytest.mark.parametrize("sizes, max_size", [(None, 9), (None, 512), ([1, 16], None)])
def test_hook_refuses_decode_graphs_above_the_cap(
    device, route_log, monkeypatch, given, sizes, max_size
):
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, DECODE_GRAPH, None)
    _set_env(monkeypatch, BREAKABLE, "1")
    device(7, 0)
    cfg = _vllm_config(cudagraph_mode=given and CUDAGraphMode[given])
    cfg.scheduler_config.max_num_seqs = 256
    cfg.compilation_config.cudagraph_capture_sizes = sizes
    cfg.compilation_config.max_cudagraph_capture_size = max_size
    with pytest.raises(ValueError, match="FULL decode graphs of at most 8"):
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)


@pytest.mark.parametrize("given", ["PIECEWISE", "NONE"])
def test_hook_leaves_capture_sizes_without_decode_graphs(
    device, route_log, monkeypatch, given
):
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, BREAKABLE, "1")
    device(7, 0)
    cfg = _vllm_config(cudagraph_mode=CUDAGraphMode[given])
    cfg.scheduler_config.max_num_seqs = 256
    cfg.compilation_config.cudagraph_capture_sizes = None
    cfg.compilation_config.max_cudagraph_capture_size = 512
    icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    assert cfg.compilation_config.max_cudagraph_capture_size == 512


@pytest.mark.parametrize(
    "breakable, given, warns",
    [
        (None, "PIECEWISE", True),
        ("0", "PIECEWISE", True),
        ("1", "PIECEWISE", False),
        (None, "FULL_AND_PIECEWISE", True),
        ("1", "FULL_AND_PIECEWISE", False),
        (None, "FULL_DECODE_ONLY", False),
        (None, "NONE", False),
        (None, None, False),
    ],
)
def test_hook_warns_on_piecewise_without_breakable_cuda_graphs(
    device, route_log, monkeypatch, caplog, breakable, given, warns
):
    from vllm.config.compilation import CUDAGraphMode

    if breakable is None:
        monkeypatch.delenv(BREAKABLE, raising=False)
    else:
        monkeypatch.setenv(BREAKABLE, breakable)
    device(7, 0)
    cfg = _vllm_config(cudagraph_mode=given and CUDAGraphMode[given])
    # The "vllm" logger may not propagate to the root, where caplog listens.
    icfg.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level("WARNING", logger=icfg.logger.name):
            icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)
    finally:
        icfg.logger.removeHandler(caplog.handler)
    warnings = list(
        {id(r): r for r in caplog.records if r.levelname == "WARNING"}.values()
    )
    assert len(warnings) == int(warns)
    if warns:
        want = f"cudagraph_mode {given} without VLLM_USE_BREAKABLE_CUDAGRAPH=1 "
        assert want + "captures no" in warnings[0].getMessage()
        assert cfg.compilation_config.cudagraph_mode is CUDAGraphMode[given]


@pytest.fixture
def registered(monkeypatch, device):
    """The architecture, its configs and the quant config registered as at
    cc 7.0; the registries restored afterwards. The modules VllmConfig
    imports lazily are imported first, on the CPU platform."""
    import vllm.model_executor.layers.attention
    import vllm.model_executor.layers.quantization.utils.w8a8_utils
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP
    from vllm_sm70.models.inkling2_sm70 import quant_config

    cpu = vllm.platforms.current_platform
    had_model = inkling2_sm70.ARCH in ModelRegistry.models
    had_hook = inkling2_sm70.ARCH in MODELS_CONFIG_MAP
    monkeypatch.setattr(inkling2_sm70, "_registered", False)
    quant_config.register()
    device(7, 0)
    assert inkling2_sm70.register() is True
    yield cpu
    if not had_model:
        ModelRegistry.models.pop(inkling2_sm70.ARCH, None)
    if not had_hook:
        MODELS_CONFIG_MAP.pop(inkling2_sm70.ARCH, None)


@pytest.mark.parametrize(
    "switch, breakable, given, want",
    [
        (None, None, None, "FULL_DECODE_ONLY"),
        ("0", None, None, "NONE"),
        ("0", "1", None, "PIECEWISE"),
        (None, "1", "NONE", "NONE"),
        (None, None, "FULL", None),
        ("0", None, "FULL_AND_PIECEWISE", None),
    ],
)
def test_real_vllm_config_runs_the_hook_before_the_graph_defaults(
    registered, route_log, monkeypatch, switch, breakable, given, want
):
    """VllmConfig.__post_init__ calls the hook before it applies its own
    cudagraph default, so the hook's choice for an unset mode stands."""
    from vllm.config import CompilationConfig, ModelConfig, VllmConfig
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, DECODE_GRAPH, switch)
    _set_env(monkeypatch, BREAKABLE, breakable)
    cuda = vllm.platforms.current_platform
    # The class inspection imports the model as a CPU process does.
    monkeypatch.setattr(vllm.platforms, "_current_platform", registered)
    model_config = ModelConfig(
        model=str(DATA),
        dtype="float16",
        hf_overrides={"architectures": [inkling2_sm70.ARCH]},
        tokenizer_mode="inkling2_sm70",
        skip_tokenizer_init=True,
    )
    assert model_config.architecture == inkling2_sm70.ARCH
    assert model_config.quantization == "inkling2"
    monkeypatch.setattr(vllm.platforms, "_current_platform", cuda)
    monkeypatch.setattr(type(cuda), "support_static_graph_mode", lambda self: True)
    kwargs = {}
    if given is not None:
        kwargs["compilation_config"] = CompilationConfig(
            cudagraph_mode=CUDAGraphMode[given]
        )
    if want is None:
        with pytest.raises(ValueError, match=f"cudagraph_mode {given};"):
            VllmConfig(model_config=model_config, **kwargs)
        return
    config = VllmConfig(model_config=model_config, **kwargs)
    assert config.compilation_config.cudagraph_mode is CUDAGraphMode[want]
    assert config.cache_config.block_size == icfg.KV_BLOCK_SIZE
    # Upstream turns the unset async scheduling on after the hook.
    assert config.scheduler_config.async_scheduling is True
    assert f"cudagraph_mode {want};" in route_log[0]
    if want == "FULL_DECODE_ONLY":
        compilation = config.compilation_config
        assert compilation.max_cudagraph_capture_size == icfg.DECODE_GRAPH_MAX_BATCH
        assert max(compilation.cudagraph_capture_sizes) <= icfg.DECODE_GRAPH_MAX_BATCH


@pytest.mark.parametrize(
    "switch, dump, mode",
    [
        (None, False, "FULL"),
        ("0", False, "FULL_DECODE_ONLY"),
        ("0", False, "FULL_AND_PIECEWISE"),
        (None, True, "FULL_DECODE_ONLY"),
        (None, True, "FULL_AND_PIECEWISE"),
    ],
)
def test_hook_refuses_full_cuda_graphs(
    device, route_log, monkeypatch, tmp_path, switch, dump, mode
):
    """FULL graphs of mixed batches never; of decode-only batches not with
    the kill switch or a visit dump."""
    from vllm.config.compilation import CUDAGraphMode

    _set_env(monkeypatch, DECODE_GRAPH, switch)
    device(7, 0)
    sm70 = {"inkling2": {"visit_dump_dir": str(tmp_path)}} if dump else None
    cfg = _vllm_config(cudagraph_mode=CUDAGraphMode[mode], sm70=sm70)
    with pytest.raises(ValueError, match=f"cudagraph_mode {mode};"):
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(cfg)


def test_released_index_keeps_the_mtp_file_out_of_the_weights(tmp_path):
    """The default loader drops safetensors files the index does not name
    (UP model_loader/default_loader.py ``_prepare_weights``)."""
    import shutil

    from vllm.model_executor.model_loader.weight_utils import (
        filter_duplicate_safetensors_files,
    )

    index = "model.safetensors.index.json"
    shutil.copy(DATA / index, tmp_path / index)
    names = sorted(set(json.loads((DATA / index).read_text())["weight_map"].values()))
    assert names == ["model-00001-of-00001.safetensors"]
    files = [str(tmp_path / name) for name in [*names, "mtp.safetensors"]]
    kept = filter_duplicate_safetensors_files(files, str(tmp_path), index)
    assert kept == files[:1]


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


def test_hook_hands_the_image_cap_to_the_processor(device, route_log):
    from vllm_sm70.models.inkling2_sm70.sm70_configs import (
        IMAGE_ENCODER_TOKEN_CAP_ATTR,
        Inkling2LoopSm70MMConfig,
    )

    tiny = pathlib.Path(__file__).parent / "fixtures" / "inkling2_tiny" / "config.json"
    tiny = json.loads(tiny.read_text())
    device(7, 0)
    for sm70, want in ((None, 10817), ({"max_image_encoder_tokens": 1601}, 1601)):
        vc = _vllm_config(sm70=None if sm70 is None else {"inkling2": sm70})
        vc.model_config.hf_config = Inkling2LoopSm70MMConfig(**tiny)
        icfg.Inkling2Sm70ModelConfig.verify_and_update_config(vc)
        assert getattr(vc.model_config.hf_config, IMAGE_ENCODER_TOKEN_CAP_ATTR) == want


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
    names = [n for n in envs.environment_variables() if "INKLING" in n]
    assert not [n for n in names if "INKLING2" in n or "DUMP" in n]
    assert all(n in envs._SECTIONS["models"] for n in names)  # route switches
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
