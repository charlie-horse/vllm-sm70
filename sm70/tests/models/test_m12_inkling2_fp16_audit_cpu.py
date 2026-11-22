# SPDX-License-Identifier: Apache-2.0
"""Tests for the M12 Inkling 2 load-time checks (fp16_audit.py) and their use
by the weight loader (weights.py).

CPU. The tiny checkpoint is loaded in fp16 with one injected value; the
hard-coded config keys are varied one at a time; the post-load walk runs on a
module tree with hooks at several depths.
"""

import json
import pathlib
import sys

import pytest

pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")
import torch
from torch import nn

from vllm_sm70.models.inkling2_sm70 import fp16_audit, sm70_configs, weights
from vllm_sm70.models.inkling2_sm70.fp16_audit import Fp16Audit

HERE = pathlib.Path(__file__).resolve().parent
TINY = HERE / "fixtures" / "inkling2_tiny"
MINI = HERE / "data" / "inkling2_mini"

NATIVE = safetensors_torch.load_file(TINY / "model.safetensors")
# A tensor cast from bf16 to fp16 and one whose parameter is fp32.
CAST = "model.llm.layers.0.attn.wo_ud.weight"
KEPT = "model.llm.layers.0.mlp.gate.weight"


def _config_dict(path=TINY):
    return json.loads((path / "config.json").read_text())


def _config(**text):
    data = _config_dict()
    loop = text.pop("is_varloop", None)
    data["text_config"].update(text)
    if loop is not None:
        data["text_config"]["loop_config"]["is_varloop"] = loop
    return sm70_configs.Inkling2LoopSm70MMConfig(**data)


class _Bag(nn.Module):
    def __init__(self, wmap) -> None:
        super().__init__()
        self.tensors = {
            name: nn.Parameter(torch.zeros(spec.shape, dtype=spec.dtype), False)
            for name, spec in wmap.params.items()
        }

    def named_parameters(self, *args, **kwargs):
        return iter(self.tensors.items())


def _load(native, audit=None, tp_size=1, tp_rank=0):
    config = _config()
    wmap = weights.build_weight_map(config, tp_size=tp_size, tp_rank=tp_rank)
    return weights.load_weights(
        _Bag(wmap),
        native.items(),
        config,
        tp_size=tp_size,
        tp_rank=tp_rank,
        audit=audit,
    )


def _with(name, value, index=0):
    native = dict(NATIVE)
    tensor = native[name].clone()
    tensor.view(-1)[index] = value
    native[name] = tensor
    return native


def test_names_used_here_exist_with_the_expected_dtypes():
    assert NATIVE[CAST].dtype == torch.bfloat16
    wmap = weights.build_weight_map(_config(), tp_size=1, tp_rank=0)
    assert wmap.params[wmap.routes[CAST].param].dtype == torch.float16
    assert wmap.params[wmap.routes[KEPT].param].dtype == torch.float32


def _flushed(name):
    tensor = NATIVE[name]
    return int(((tensor != 0) & (tensor.to(torch.float16) == 0)).sum())


def test_tiny_checkpoint_passes_and_is_counted():
    audit = Fp16Audit()
    _load(NATIVE, audit)
    wmap = weights.build_weight_map(_config(), tp_size=1, tp_rank=0)
    cast = [
        name
        for name, route in wmap.routes.items()
        if wmap.params[route.param].dtype == torch.float16
    ]
    assert audit.tensors == len(cast) > 100
    # Near-zero weights of ordinary tensors round to zero and are counted.
    assert audit.flushed == sum(_flushed(name) for name in cast) > 0
    assert 0 < audit.max_abs <= fp16_audit.FP16_MAX
    assert audit.max_name in NATIVE
    assert f"{audit.flushed} values rounded to zero" in audit.summary()


@pytest.mark.parametrize("value", [7e4, -7e4, float("inf"), float("nan")])
def test_out_of_range_weight_fails_load_with_the_tensor_name(value):
    with pytest.raises(ValueError, match="does not fit in fp16") as e:
        _load(_with(CAST, value))
    assert CAST in str(e.value)


@pytest.mark.parametrize("value", [1e-9, -1e-9])
def test_tensor_of_flushed_weights_fails_load_with_the_tensor_name(value):
    assert torch.tensor(value, dtype=torch.bfloat16).to(torch.float16) == 0
    native = dict(NATIVE)
    native[CAST] = torch.full_like(NATIVE[CAST], value)
    with pytest.raises(ValueError, match="round to zero in fp16") as e:
        _load(native)
    assert CAST in str(e.value)
    assert f"{NATIVE[CAST].numel()} nonzero values" in str(e.value)


def test_one_flushed_weight_in_an_ordinary_tensor_is_counted():
    assert _flushed(CAST) == 0
    base, audit = Fp16Audit(), Fp16Audit()
    _load(NATIVE, base)
    _load(_with(CAST, 1e-9), audit)
    assert audit.flushed == base.flushed + 1


def test_flush_limit_is_relative_to_the_largest_weight():
    limit = fp16_audit.FLUSH_RELATIVE
    lost = 2.0**-26
    # At the limit the flush is tolerated; the tensor is still all subnormal.
    with pytest.raises(ValueError, match="ok: max .* below the smallest normal"):
        Fp16Audit().check("ok", torch.tensor([lost, lost / limit]))
    with pytest.raises(ValueError, match="bad: 1 nonzero values"):
        Fp16Audit().check("bad", torch.tensor([lost, 0.5 * lost / limit]))


def test_default_audit_is_per_load():
    native = dict(NATIVE)
    native[CAST] = torch.full_like(NATIVE[CAST], 1e-9)
    with pytest.raises(ValueError, match="round to zero"):
        _load(native, audit=None)
    _load(NATIVE, audit=None)


def test_only_this_ranks_slice_is_audited():
    tensor = NATIVE[CAST]
    assert tensor.shape[1] % 2 == 0
    last = tensor.numel() - 1  # last column: rank 1 at TP2
    native = _with(CAST, 7e4, index=last)
    _load(native, tp_size=2, tp_rank=0)
    with pytest.raises(ValueError, match="does not fit in fp16"):
        _load(native, tp_size=2, tp_rank=1)


def test_fp32_parameters_are_not_audited():
    _load(_with(KEPT, 7e4))
    native = dict(NATIVE)
    native[KEPT] = torch.full_like(NATIVE[KEPT], 1e-9)
    _load(native)


def test_check_counts_subnormals_and_accepts_exact_zeros():
    audit = Fp16Audit()
    source = torch.tensor([0.0, 1.0, 2.0**-15, -(2.0**-20), 65504.0])
    audit.check("a", source)
    assert (audit.tensors, audit.flushed, audit.subnormal) == (1, 0, 2)
    assert (audit.max_abs, audit.max_name) == (65504.0, "a")
    audit.check("empty", torch.zeros(0))
    audit.check("not fp16", torch.tensor([1e30]), torch.tensor([1e30]))
    assert audit.tensors == 1


@pytest.mark.parametrize("value", [1e-5, -3e-5, 2.0**-15])
def test_tensor_of_subnormal_weights_fails_load_with_the_tensor_name(value):
    cast = torch.tensor(value, dtype=torch.bfloat16).to(torch.float16)
    assert cast != 0 and cast.abs() < fp16_audit.FP16_MIN_NORMAL
    native = dict(NATIVE)
    native[CAST] = torch.full_like(NATIVE[CAST], value)
    with pytest.raises(ValueError, match="below the smallest normal fp16") as e:
        _load(native)
    assert CAST in str(e.value)


def test_subnormal_limit_is_the_smallest_normal_value_and_zeros_pass():
    audit = Fp16Audit()
    low = fp16_audit.FP16_MIN_NORMAL
    audit.check("normal", torch.tensor([0.0, low / 2, -low]))
    audit.check("zeros", torch.zeros(4))
    assert (audit.tensors, audit.subnormal) == (2, 1)
    with pytest.raises(ValueError, match="tiny: max .* below the smallest normal"):
        audit.check("tiny", torch.tensor([0.0, low / 2, -low * 0.99]))
    with pytest.raises(ValueError, match="tiny: max .* below the smallest normal"):
        audit.check("tiny", torch.tensor([low / 2]), torch.tensor([low / 2]).half())


def test_subnormal_share_above_one_percent_is_logged_with_the_name(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(
        fp16_audit.logger, "warning", lambda msg, *args: lines.append(msg % args)
    )
    audit = Fp16Audit()
    source = torch.ones(200)
    source[:2] = 2.0**-16
    audit.check("at the limit", source)
    assert lines == [] and audit.subnormal == 2
    source[2] = -(2.0**-16)
    audit.check("layers.3.weight", source)
    assert len(lines) == 1 and audit.subnormal == 5
    assert "layers.3.weight has 3 of 200 values (1.50%) subnormal" in lines[0]
    _load(NATIVE, audit)
    assert len(lines) == 1


def test_check_of_a_dequantized_block_without_a_cast_copy():
    audit = Fp16Audit()
    with pytest.raises(ValueError, match="experts.3.w2: max .* does not fit"):
        audit.check("experts.3.w2", torch.tensor([[1.0, 65520.0]]))
    with pytest.raises(ValueError, match="experts.3.w2: 2 nonzero values"):
        audit.check("experts.3.w2", torch.tensor([[1e-9, -1e-8, 1e-5]]))
    audit.check("experts.3.w2", torch.tensor([[1e-9, -1e-8, 0.5]]))
    assert audit.flushed == 2


def test_released_and_tiny_configs_are_accepted():
    fp16_audit.check_hard_coded_config(_config())
    fp16_audit.check_hard_coded_config(_config().get_text_config())
    mini = sm70_configs.Inkling2LoopSm70MMConfig(**_config_dict(MINI))
    fp16_audit.check_hard_coded_config(mini)


@pytest.mark.parametrize(
    "key, value",
    [
        ("n_shared_experts", 1),
        ("use_value_norm_no_gamma", False),
        ("use_gate_bias", False),
        ("q_bias", True),
        ("o_bias", True),
        ("is_varloop", True),
    ],
)
def test_config_that_differs_from_the_hard_coded_values_is_refused(key, value):
    try:
        config = _config(**{key: value})
    except ValueError:
        pytest.skip("the config class itself refuses this value")
    with pytest.raises(ValueError, match=f"{key}={value!r}") as e:
        weights.build_weight_map(config, tp_size=1, tp_rank=0)
    assert "supported" in str(e.value)


def test_config_check_names_every_differing_and_missing_key():
    text = _config().get_text_config()
    text.attn_res_normalize_values = False
    text.n_shared_experts = 2
    del text.q_bias
    with pytest.raises(ValueError) as e:
        fp16_audit.check_hard_coded_config(text)
    message = str(e.value)
    assert "attn_res_normalize_values=False" in message
    assert "n_shared_experts=2" in message
    assert "q_bias=missing" in message
    assert "o_bias" not in message


def test_hard_coded_table_covers_the_plan_keys():
    assert set(fp16_audit.HARD_CODED_CONFIG) == {
        "attn_res_normalize_values",
        "use_value_norm_no_gamma",
        "use_gate_bias",
        "n_shared_experts",
        "q_bias",
        "o_bias",
        "is_varloop",
    }


class _Hooked(nn.Module):
    def __init__(self, log, name, *children) -> None:
        super().__init__()
        self.log, self.label = log, name
        self.kids = nn.ModuleList(children)

    def process_weights_after_loading(self) -> None:
        self.log.append(self.label)


def test_walk_reaches_every_hook_once_children_first():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

    class Owner(nn.Module, AttentionLayerBase):
        def __init__(self, log) -> None:
            super().__init__()
            self.log = log

        def get_attn_backend(self):
            raise NotImplementedError

        def get_kv_cache_spec(self, vllm_config):
            return None

        def process_weights_after_loading(self, act_dtype) -> None:
            self.log.append("owner")

    log: list[str] = []
    leaf_a, leaf_b = _Hooked(log, "a"), _Hooked(log, "b")
    plain = nn.Sequential(nn.Linear(2, 2), leaf_b)
    mid = _Hooked(log, "mid", leaf_a, plain, Owner(log))
    model = _Hooked(log, "model", mid, nn.Linear(2, 2))
    reached = fp16_audit.walk_process_weights_after_loading(model)
    # Not the model itself, not the attention layer vLLM calls with a dtype.
    assert sorted(log) == ["a", "b", "mid"]
    assert log.index("mid") > max(log.index("a"), log.index("b"))
    assert sorted(reached) == ["kids.0", "kids.0.kids.0", "kids.0.kids.1.1"]
    expected = {
        name
        for name, module in model.named_modules()
        if module is not model and isinstance(module, _Hooked)
    }
    assert set(reached) == expected


def test_walk_reaches_every_hook_once_children_first_on_the_tiny_model():
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.attention import is_deferred_attention_layer

    sys.path.insert(0, str(HERE))
    import test_m12_inkling2_model_cpu as body

    model = body.Engine().model
    log: list[str] = []
    model.model.layers[0].probe = _Hooked(log, "probe", _Hooked(log, "kid"))
    expected = {
        name
        for name, module in model.named_modules()
        if module is not model
        and callable(getattr(module, "process_weights_after_loading", None))
        and not is_deferred_attention_layer(module)
    }
    assert {"model.layers.0.probe", "model.layers.0.probe.kids.0"} <= expected
    calls = dict.fromkeys(expected, 0)

    def counted(name, hook):
        def wrapper():
            calls[name] += 1
            hook()

        return wrapper

    for name in expected:
        module = model.get_submodule(name)
        module.process_weights_after_loading = counted(
            name, module.process_weights_after_loading
        )
    reached = fp16_audit.walk_process_weights_after_loading(model)
    assert sorted(reached) == sorted(expected)
    assert set(calls.values()) == {1} and log == ["kid", "probe"]
    order = {name: i for i, name in enumerate(reached)}
    for name in reached:
        parent = name.rpartition(".")[0]
        while parent:
            assert order.get(parent, len(order)) > order[name], (name, parent)
            parent = parent.rpartition(".")[0]
    # The model's own hook, the one vLLM calls, runs the same walk.
    model.process_weights_after_loading()
    assert set(calls.values()) == {2} and log == ["kid", "probe"] * 2
