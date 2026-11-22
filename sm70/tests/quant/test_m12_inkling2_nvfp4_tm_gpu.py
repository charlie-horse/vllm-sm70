# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the M12 Inkling 2 NVFP4 TurboMind expert route (moe_tm.py).

One V100 with vllm_sm70._sm70_C built. The TurboMind route is compared with
the fp16 torch path of moe.py (max-abs-rel under 1e-2):
- at the real per-rank shapes (128 experts, top-8; TP2 and TP1) with random
  codes and scales in the checkpoint's range, in both stage call forms;
- on the tiny checkpoint (models/fixtures/inkling2_tiny/) through
  ``Inkling2Sm70Experts.load_expert_layer``, with its shape admitted;
- on real layer weights at both TP2 ranks, when INKLING2_SM70_CHECKPOINT
  names the checkpoint directory (skipped otherwise).

    python -m pytest -q -p no:cacheprovider \
        sm70/tests/quant/test_m12_inkling2_nvfp4_tm_gpu.py -m gpu
"""

from __future__ import annotations

import json
import os
import pathlib
import re
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
safetensors = pytest.importorskip("safetensors")

from vllm_sm70.models.inkling2_sm70 import expert_weights as ew
from vllm_sm70.models.inkling2_sm70 import (
    moe,
    moe_tm,
    quant_config,
    sm70_configs,
)

TINY = pathlib.Path(__file__).resolve().parents[1] / "models/fixtures/inkling2_tiny"
CHECKPOINT_ENV = "INKLING2_SM70_CHECKPOINT"
SWITCH = moe_tm.SWITCH
EXPERTS, HIDDEN, INTERMEDIATE, TOP_K = 128, 2304, 1152, 8
TINY_CONTRACT = (64, 32, 8, 2)
GATE_CAP, UP_CAP = 4.0, 25.0
TOL = 1e-2
REAL_SIZES = [1, 2, 8, 10, 11, 64, 512, 2048]
CHECKPOINT_LAYERS = (0, 10, 19)
CHECKPOINT_SIZES = [1, 2, 8, 64, 512]


def max_abs_rel(out, ref):
    return float((out.float() - ref.float()).abs().max() / ref.float().abs().max())


@pytest.fixture(autouse=True)
def _gpu(monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires an SM70 GPU")
    from vllm_sm70 import _ext

    if not _ext.is_loaded():
        pytest.fail(
            "SM70 GPU present but vllm_sm70._sm70_C is not loaded: "
            f"vllm_sm70 imported from {_ext.__file__}"
        )
    if moe_tm.turbomind() is None:
        pytest.fail("_sm70_C lacks " + ", ".join(moe_tm.REQUIRED_OPS))
    monkeypatch.setenv(SWITCH, "1")
    torch.cuda.set_device(0)
    yield
    moe_tm.Workspace.clear()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Real shapes, random codes
# --------------------------------------------------------------------------
def real_config():
    return SimpleNamespace(
        n_routed_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_experts_per_tok=TOP_K,
        route_scale=8.0,
        mlp_gate_softcap=GATE_CAP,
        mlp_up_softcap=UP_CAP,
    )


def random_codes(name, n, k, seed):
    """Scales in 0.0012..0.141, the checkpoint's range."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (EXPERTS, n, k), generator=g, dtype=torch.uint8)
    block = (torch.rand(EXPERTS, n, k // 16, generator=g) * 139.8 + 1.2).to(
        torch.float8_e4m3fn
    )
    return ew.Nvfp4Codes(name, codes, block, torch.full((EXPERTS,), 1e-3))


def dequant_fp16(codes):
    """fp16 [E, N, K] on the GPU by the Q1 value formula, as the torch path
    holds it."""
    lut = torch.tensor(ew.E2M1_VALUES, device="cuda")
    out = torch.empty(codes.codes.shape, dtype=torch.float16, device="cuda")
    for expert in range(codes.codes.shape[0]):
        block = codes.block_scale[expert].cuda().float().repeat_interleave(16, dim=-1)
        values = lut[codes.codes[expert].cuda().long()] * block
        out[expert] = (values * float(codes.global_scale[expert])).half()
    return out


class RealPair:
    """The TurboMind route and the torch path on the same random experts."""

    def __init__(self, tp_size, seed, monkeypatch):
        local = INTERMEDIATE // tp_size
        w13 = random_codes("w13", 2 * local, HIDDEN, seed)
        w2 = random_codes("w2", HIDDEN, local, seed + 1)
        self.route = moe_tm.build(
            w13, w2,
            device=torch.device("cuda"), gate_cap=GATE_CAP, up_cap=UP_CAP,
            top_k=TOP_K,
        )  # fmt: skip
        with monkeypatch.context() as patch:
            patch.setenv(SWITCH, "0")
            self.oracle = moe.Inkling2Sm70Experts(real_config(), tp_size=tp_size)
        self.oracle.cuda()
        self.oracle.w13.copy_(dequant_fp16(w13))
        self.oracle.w2.copy_(dequant_fp16(w2))
        self.local = local

    def compare(self, m, seed, pad=()):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(m, HIDDEN, generator=g) * 0.5).half().cuda()
        indices = torch.stack(
            [torch.randperm(EXPERTS, generator=g)[:TOP_K] for _ in range(m)]
        ).cuda()
        weights = torch.rand(m, TOP_K, generator=g).cuda()
        if len(pad):
            indices[list(pad)] = EXPERTS
            weights[list(pad)] = 0.0
        out = torch.full_like(x, float("nan"))
        ref = torch.full_like(x, float("nan"))
        self.route.expert_sum(x, indices, weights, out)
        self.oracle.torch_expert_sum(x, indices, weights, ref)
        return out, ref


@pytest.fixture(
    scope="module", params=[(2, 0), (2, 1), (1, 0)], ids=["tp2r0", "tp2r1", "tp1"]
)
def real_pair(request):
    if not torch.cuda.is_available():
        pytest.skip("requires an SM70 GPU")
    tp_size, rank = request.param
    patch = pytest.MonkeyPatch()
    patch.setenv(SWITCH, "1")
    pair = RealPair(tp_size, 100 * tp_size + 10 * rank, patch)
    yield pair
    patch.undo()
    del pair
    torch.cuda.empty_cache()


@pytest.mark.parametrize("m", REAL_SIZES)
def test_real_shape_matches_torch_path(real_pair, m):
    out, ref = real_pair.compare(m, seed=m)
    assert out.dtype == torch.float16 and bool(torch.isfinite(out).all())
    assert max_abs_rel(out, ref) < TOL


@pytest.mark.parametrize("m,pad", [(2, [1]), (10, [0, 9]), (64, [0, 31, 63])])
def test_real_shape_padding_rows(real_pair, m, pad):
    out, ref = real_pair.compare(m, seed=m, pad=pad)
    assert bool((out[pad] == 0).all()) and bool(torch.isfinite(out).all())
    assert max_abs_rel(out, ref) < TOL


@pytest.mark.parametrize("m", [1, 64])
def test_real_shape_all_padding(real_pair, m):
    out, _ = real_pair.compare(m, seed=m, pad=range(m))
    assert bool((out == 0).all())


def test_real_shape_bank_layout_and_bytes(real_pair):
    route, local = real_pair.route, real_pair.local
    assert route.w13.weight.shape == (EXPERTS, HIDDEN, 2 * local // 8)
    assert route.w13.scales.shape == (EXPERTS, HIDDEN // 16, 2 * local)
    assert route.w2.weight.shape == (EXPERTS, local, HIDDEN // 8)
    assert route.w2.scales.shape == (EXPERTS, local // 16, HIDDEN)
    assert route.w13.weight.dtype == route.w2.weight.dtype == torch.int32
    assert route.w13.scales.dtype == route.w2.scales.dtype == torch.float16
    banks = moe_tm.layer_nbytes(EXPERTS, HIDDEN, local)
    assert banks == 318_504_960 * (2 * local // INTERMEDIATE)  # TP2 local = 576
    pointer_tables = route.nbytes() - banks
    assert 0 < pointer_tables <= 4 * EXPERTS * 64


def test_twenty_layers_fit_the_budget(real_pair):
    """20 NVFP4 layers of one TP2 rank: about 5.93 GiB."""
    if real_pair.local != INTERMEDIATE // 2:
        pytest.skip("TP2 budget")
    assert 5.9 < 20 * real_pair.route.nbytes() / 2**30 < 5.96


# --------------------------------------------------------------------------
# Tiny checkpoint through the module
# --------------------------------------------------------------------------
def tiny_module(monkeypatch, *, tp_size, tp_rank, torch_path):
    from safetensors.torch import load_file

    native = load_file(TINY / "model.safetensors")
    formats = quant_config.expert_formats(
        json.loads((TINY / "hf_quant_config.json").read_text())
    )
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((TINY / "config.json").read_text())
    )
    layer = 1
    with monkeypatch.context() as patch:
        if torch_path:
            patch.setenv(SWITCH, "0")
        module = moe.Inkling2Sm70Experts(config, tp_size=tp_size).cuda()
        module.load_expert_layer(
            ew.load_expert_layers(
                native, config, tp_size=tp_size, tp_rank=tp_rank, formats=formats
            )[layer]
        )
    prefix = f"model.llm.layers.{layer}.mlp.gate."
    module.gate.weight.data.copy_(native[prefix + "weight"])
    module.gate.bias.data.copy_(native[prefix + "bias"])
    assert (module.tm_route is None) == torch_path, module.tm_rejection
    return module


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("m", [1, 2, 8, 40, 41, 64])
def test_tiny_checkpoint_matches_torch_path(monkeypatch, tp_size, tp_rank, m):
    from safetensors.torch import load_file

    monkeypatch.setattr(
        moe_tm, "INKLING_CONTRACTS", moe_tm.INKLING_CONTRACTS | {TINY_CONTRACT}
    )
    monkeypatch.setattr(moe_tm, "INKLING_TP_SIZES", (1, 2))
    kwargs = {"tp_size": tp_size, "tp_rank": tp_rank}
    module = tiny_module(monkeypatch, torch_path=False, **kwargs)
    oracle = tiny_module(monkeypatch, torch_path=True, **kwargs)
    recorded = load_file(TINY / "ops.safetensors")["router.0.in.hidden_states"]
    x = recorded[torch.arange(m) % recorded.shape[0]].half().cuda()
    is_padding = torch.zeros(m, dtype=torch.bool, device="cuda")
    out = torch.full_like(x, float("nan"))
    ref = torch.full_like(x, float("nan"))
    module(x, out, is_padding)
    oracle(x, ref, is_padding)
    assert module.w13.numel() == 0 and oracle.w13.shape[0] == 8
    assert max_abs_rel(out, ref) < TOL


# --------------------------------------------------------------------------
# Real layer weights
# --------------------------------------------------------------------------
def checkpoint_layers(directory, config, layers, *, tp_rank):
    from safetensors import safe_open

    wanted = re.compile(
        r"model\.llm\.layers\.(\d+)\.mlp\.(experts\.w(13|2)_weight.*|gate\.(weight|bias))"
    )
    loader = ew.Inkling2ExpertLoader(
        config,
        tp_size=2,
        tp_rank=tp_rank,
        formats=quant_config.load_expert_formats(str(directory)),
    )
    gates: dict[str, torch.Tensor] = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(str(path), framework="pt") as shard:
            for name in shard.keys():
                match = wanted.fullmatch(name)
                if match is None or int(match.group(1)) not in layers:
                    continue
                if ".gate." in name:
                    gates[name] = shard.get_tensor(name)
                else:
                    loader(name, shard.get_tensor(name))
    return loader.layers, gates


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_checkpoint_layers_match_torch_path(monkeypatch, tp_rank):
    directory = os.environ.get(CHECKPOINT_ENV)
    if not directory:
        pytest.skip(f"set {CHECKPOINT_ENV} to the checkpoint directory")
    directory = pathlib.Path(directory)
    config = sm70_configs.Inkling2LoopSm70MMConfig(
        **json.loads((directory / "config.json").read_text())
    )
    layers, gates = checkpoint_layers(
        directory, config, CHECKPOINT_LAYERS, tp_rank=tp_rank
    )
    assert sorted(layers) == sorted(CHECKPOINT_LAYERS)
    for index in CHECKPOINT_LAYERS:
        modules = []
        for torch_path in (False, True):
            with monkeypatch.context() as patch:
                if torch_path:
                    patch.setenv(SWITCH, "0")
                module = moe.Inkling2Sm70Experts(config, tp_size=2).cuda()
                module.load_expert_layer(layers[index])
            prefix = f"model.llm.layers.{index}.mlp.gate."
            module.gate.weight.data.copy_(gates[prefix + "weight"])
            module.gate.bias.data.copy_(gates[prefix + "bias"])
            assert (module.tm_route is None) == torch_path, module.tm_rejection
            modules.append(module)
        module, oracle = modules
        assert module.tm_route.nbytes() - 318_504_960 <= 4 * EXPERTS * 64
        for m in CHECKPOINT_SIZES:
            g = torch.Generator().manual_seed(1000 * index + m)
            x = torch.randn(m, HIDDEN, generator=g).half().cuda()
            is_padding = torch.zeros(m, dtype=torch.bool, device="cuda")
            out = torch.full_like(x, float("nan"))
            ref = torch.full_like(x, float("nan"))
            module(x, out, is_padding)
            oracle(x, ref, is_padding)
            assert max_abs_rel(out, ref) < TOL, (index, m)
        del module, oracle, modules
        torch.cuda.empty_cache()
