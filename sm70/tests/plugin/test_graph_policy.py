# SPDX-License-Identifier: Apache-2.0
"""T8/T9: SM70 CUDA graph capture sizes and graph-memory policy (CPU).

The size tables are the fork's (FORK vllm/platforms/runtime_defaults.py:
22-33, 200-251); the expected lists below were computed from those
functions at cf9492a.
"""

import os
from types import SimpleNamespace

import pytest
from conftest import requires_vllm

from vllm_sm70 import graph_policy as gp

SWITCHES = (
    gp.SIZES_SWITCH,
    gp.MEMORY_SWITCH,
    gp.UPSTREAM_MEMORY_ENV,
    "VLLM_SM70_FLASH_V100",
    "TORCH_COMPILE_DISABLE",
    "VLLM_USE_BREAKABLE_CUDAGRAPH",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    # setenv first, so monkeypatch restores even what apply() writes itself.
    for name in SWITCHES:
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    gp._LOGGED.clear()
    monkeypatch.setattr(gp, "_memory_env_written", False)


# ---- size tables ----------------------------------------------------------


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        (1, [1, 2]),
        (2, [1, 2]),
        (3, [1, 2, 3]),
        (4, [1, 2, 4]),
        (8, [1, 2, 4, 8]),
        (20, [1, 2, 4, 8, 16, 20]),  # Flint prod (host max-num-seqs 20)
        (32, [1, 2, 4, 8, 16, 32]),
        (64, [1, 2, 4, 8, 16, 32]),  # capped at B32
        (256, [1, 2, 4, 8, 16, 32]),
    ],
)
def test_nomtp_sizes(max_num_seqs, expected):
    assert gp.nomtp_capture_sizes(max_num_seqs) == expected
    assert gp.capture_sizes(max_num_seqs, 0) == expected


def test_nomtp_sizes_never_empty():
    assert gp.nomtp_capture_sizes(0) == [1, 2]


MIXED = [64, 96, 128, 160, 192, 256]


@pytest.mark.parametrize(
    ("max_num_seqs", "expected"),
    [
        # Qwen DFlash2, k=7 -> q=8. Module maxNumSeqs: qwen3.6 20, qwen3.8 8.
        (4, [1, 2, 4, 8, 9, 16, 18, 24, 32] + MIXED),
        (8, [1, 2, 4, 8, 9, 16, 18, 24, 32, 48] + MIXED),
        (20, [1, 2, 4, 8, 9, 16, 18, 24, 32, 48, 160] + MIXED[:4] + [192, 256]),
    ],
)
def test_speculative_sizes_dflash_k7(max_num_seqs, expected):
    want = sorted(set(expected))
    assert gp.speculative_capture_sizes(max_num_seqs, 8) == want
    assert gp.capture_sizes(max_num_seqs, 7) == want


def test_speculative_sizes_at_b32_cap():
    sizes = gp.speculative_capture_sizes(64, 8)
    # verifier rows: q x {1,2,3,4,6,8,12,16} plus q x 32 (the B32 cap)
    for reqs in (1, 2, 3, 4, 6, 8, 12, 16, 32):
        assert 8 * reqs in sizes
    assert max(sizes) == 256


@pytest.mark.parametrize(
    ("sizes", "tokens", "expected"),
    [
        ([1, 2, 4, 8, 16, 20], 8192, 20),
        ([1, 2, 4, 8, 9, 16, 18, 24, 32, 64, 96, 128, 160, 192, 256], 8192, 256),
        ([1, 2, 4, 8, 9, 16, 18, 24, 32, 64, 96, 128, 160, 192, 256], 100, 96),
    ],
)
def test_max_capture_size(sizes, tokens, expected):
    assert gp.max_capture_size(sizes, tokens) == expected


# ---- apply(): fake VllmConfig ---------------------------------------------


def _cfg(
    *,
    max_num_seqs=20,
    num_spec=0,
    sizes=None,
    max_size=None,
    enforce_eager=False,
    backend=None,
    cudagraph_mode=None,
    v2=False,
    tokens=8192,
):
    spec = (
        SimpleNamespace(
            num_speculative_tokens=num_spec,
            uses_dynamic_speculative_decoding=lambda: False,
        )
        if num_spec
        else None
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=enforce_eager),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=sizes,
            max_cudagraph_capture_size=max_size,
            cudagraph_mode=cudagraph_mode,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=tokens
        ),
        speculative_config=spec,
        attention_config=SimpleNamespace(backend=backend),
        use_v2_model_runner=v2,
    )


def test_apply_flint_prod():
    cfg = _cfg(max_num_seqs=20)
    gp.apply(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 20]
    assert cfg.compilation_config.max_cudagraph_capture_size == 20


def test_apply_qwen_spec_v2():
    cfg = _cfg(max_num_seqs=8, num_spec=7, v2=True)
    gp.apply(cfg)
    sizes = cfg.compilation_config.cudagraph_capture_sizes
    assert sizes == gp.speculative_capture_sizes(8, 8)
    assert cfg.compilation_config.max_cudagraph_capture_size == 256


def test_user_sizes_win():
    cfg = _cfg(sizes=[1, 4])
    gp.apply(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 4]
    assert cfg.compilation_config.max_cudagraph_capture_size is None


def test_user_max_size_wins():
    cfg = _cfg(max_size=8)
    gp.apply(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes is None
    assert cfg.compilation_config.max_cudagraph_capture_size == 8


@pytest.mark.parametrize(
    ("kwargs", "env", "reason"),
    [
        ({}, {gp.SIZES_SWITCH: "0"}, "VLLM_SM70_GRAPH_CAPTURE_SIZES=0"),
        ({"enforce_eager": True}, {}, "enforce_eager"),
        ({"cudagraph_mode": SimpleNamespace(name="NONE")}, {}, "cudagraph_mode NONE"),
        ({}, {"TORCH_COMPILE_DISABLE": "1"}, "TORCH_COMPILE_DISABLE=1"),
        ({}, {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}, "VLLM_USE_BREAKABLE_CUDAGRAPH=1"),
        ({}, {"VLLM_SM70_FLASH_V100": "0"}, "VLLM_SM70_FLASH_V100=0"),
        (
            {"backend": SimpleNamespace(name="TRITON_ATTN")},
            {},
            "attention backend TRITON_ATTN selected",
        ),
    ],
)
def test_upstream_sizes_kept(monkeypatch, kwargs, env, reason):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = _cfg(**kwargs)
    assert gp.sizes_rejection(cfg) == reason
    gp.apply(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes is None
    assert cfg.compilation_config.max_cudagraph_capture_size is None


def test_custom_backend_is_flash_v100():
    cfg = _cfg(backend=SimpleNamespace(name="CUSTOM"))
    assert gp.sizes_rejection(cfg) is None


def test_switch_only_zero_acts(monkeypatch):
    monkeypatch.setenv(gp.SIZES_SWITCH, "1")
    assert gp.sizes_rejection(_cfg()) is None


def test_dynamic_speculative_decoding_kept():
    cfg = _cfg(num_spec=7)
    cfg.speculative_config.uses_dynamic_speculative_decoding = lambda: True
    assert gp.sizes_rejection(cfg) == "dynamic speculative decoding"


# ---- T9 memory policy -------------------------------------------------------


def test_memory_estimate_off_on_v1():
    gp.apply(_cfg(v2=False))
    assert os.environ[gp.UPSTREAM_MEMORY_ENV] == "0"


def test_memory_estimate_kept_on_v2():
    gp.apply(_cfg(v2=True))
    assert gp.UPSTREAM_MEMORY_ENV not in os.environ


def test_memory_user_value_wins(monkeypatch):
    monkeypatch.setenv(gp.UPSTREAM_MEMORY_ENV, "1")
    gp.apply(_cfg())
    assert os.environ[gp.UPSTREAM_MEMORY_ENV] == "1"


def test_memory_kill_switch(monkeypatch):
    monkeypatch.setenv(gp.MEMORY_SWITCH, "0")
    gp.apply(_cfg())
    assert gp.UPSTREAM_MEMORY_ENV not in os.environ


def test_memory_kept_under_eager():
    gp.apply(_cfg(enforce_eager=True))
    assert gp.UPSTREAM_MEMORY_ENV not in os.environ


def test_rebuilt_config_keeps_policy(caplog):
    # A second VllmConfig in the same process (e.g. a replace()) sees the
    # SM70 sizes and the env var it wrote; neither reads as a user setting.
    first = _cfg(max_num_seqs=20)
    gp.apply(first)
    again = _cfg(
        max_num_seqs=20,
        sizes=first.compilation_config.cudagraph_capture_sizes,
        max_size=first.compilation_config.max_cudagraph_capture_size,
    )
    with caplog.at_level("INFO", logger="vllm.sm70"):
        gp.apply(again)
    assert again.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 20]
    assert os.environ[gp.UPSTREAM_MEMORY_ENV] == "0"
    assert gp.memory_rejection(again) == f"{gp.UPSTREAM_MEMORY_ENV} set by the user"
    assert f"{gp.UPSTREAM_MEMORY_ENV} set by the user" not in caplog.text


def test_memory_independent_of_user_sizes():
    gp.apply(_cfg(sizes=[1, 2]))
    assert os.environ[gp.UPSTREAM_MEMORY_ENV] == "0"


# ---- envs registry ----------------------------------------------------------


def test_switches_registered():
    from vllm_sm70 import envs

    assert gp.SIZES_SWITCH in envs._SECTIONS["graph"]
    assert gp.MEMORY_SWITCH in envs._SECTIONS["graph"]


# ---- upstream sizing accepts the result (needs vLLM's Python source) -------


def _upstream_sizes(cfg):
    """Run UP VllmConfig._set_cudagraph_sizes on a fake config."""
    from vllm.config import CompilationConfig, VllmConfig
    from vllm.config.compilation import CUDAGraphMode

    # None means unset (UP config/compilation.py:651, :695); not a valid input.
    set_fields = {
        k: getattr(cfg.compilation_config, k)
        for k in ("cudagraph_capture_sizes", "max_cudagraph_capture_size")
        if getattr(cfg.compilation_config, k) is not None
    }
    comp = CompilationConfig(**set_fields)
    comp.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    spec = cfg.speculative_config
    num_spec = spec.num_speculative_tokens if spec else 0
    fake = SimpleNamespace(
        model_config=cfg.model_config,
        compilation_config=comp,
        scheduler_config=cfg.scheduler_config,
        speculative_config=spec,
        uniform_decode_query_len=1 + num_spec,
        num_speculative_tokens=num_spec,
        performance_mode="balanced",
    )
    VllmConfig._set_cudagraph_sizes(fake)
    return comp.cudagraph_capture_sizes, comp.max_cudagraph_capture_size


@requires_vllm
def test_upstream_default_differs_for_flint_prod(stub_stable_libtorch):
    # What vllm-sm70 ran before T8: up to 2 x max_num_seqs, step 8, plus 20.
    assert _upstream_sizes(_cfg(max_num_seqs=20)) == (
        [1, 2, 4, 8, 16, 20, 24, 32, 40],
        40,
    )


@requires_vllm
@pytest.mark.parametrize(
    ("max_num_seqs", "num_spec"), [(2, 0), (20, 0), (64, 0), (8, 7), (20, 7)]
)
def test_upstream_keeps_sm70_sizes(stub_stable_libtorch, max_num_seqs, num_spec):
    cfg = _cfg(max_num_seqs=max_num_seqs, num_spec=num_spec)
    gp.apply(cfg)
    want = gp.capture_sizes(max_num_seqs, num_spec)
    assert _upstream_sizes(cfg) == (want, max(want))


@requires_vllm
def test_upstream_filters_by_batched_tokens(stub_stable_libtorch):
    cfg = _cfg(max_num_seqs=20, num_spec=7, tokens=100)
    gp.apply(cfg)
    sizes, max_size = _upstream_sizes(cfg)
    assert max_size == 96 and sizes[-1] == 96


@requires_vllm
def test_platform_hook_applies_policy(stub_stable_libtorch, monkeypatch):
    from vllm.platforms.cuda import CudaPlatform

    from vllm_sm70 import platform

    monkeypatch.setattr(
        CudaPlatform, "apply_config_platform_defaults", classmethod(lambda c, v: None)
    )
    cfg = _cfg(max_num_seqs=20)
    platform.Sm70CudaPlatform.apply_config_platform_defaults(cfg)
    assert cfg.compilation_config.cudagraph_capture_sizes == [1, 2, 4, 8, 16, 20]
