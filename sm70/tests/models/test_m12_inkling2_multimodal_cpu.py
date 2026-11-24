# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 multimodal wiring (model.py, processing.py), on CPU.

The registered model class is built on the tiny checkpoint
(fixtures/inkling2_tiny) with its vision encoder. The fixture's text + image
case runs from raw tiles through ``embed_multimodal``, ``embed_input_ids`` and
the paged engine harness; every visit, the final state and the logits are
compared with the dense oracle (inkling2_ref.py) fed the same merged
embeddings, and with the Hugging Face recordings. Also: image rows skip
``embed_norm``, the per-image cap (processor, profiling image, model), the
attention query chunk, and image limit 0 (no encoder, vision weights skipped).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inkling2_harness as hx
from inkling2_ref import Inkling2Oracle

TINY = HERE / "fixtures" / "inkling2_tiny"
BLOCK_SIZE = 8
TOL = 1e-4
# Tiny tower: 8 x 8 pixel tiles of 2 x 2 patches.
TILE, MERGE = 8, 2
MINI_CAP_TOKENS = 2704


def _have_vllm() -> bool:
    try:
        return importlib.util.find_spec("vllm.envs") is not None
    except (ImportError, ValueError):
        return False


pytestmark = pytest.mark.skipif(not _have_vllm(), reason="vLLM not importable")


def _raw_config() -> dict:
    return json.loads((TINY / "config.json").read_text())


def _weights():
    from safetensors.torch import load_file

    return load_file(TINY / "model.safetensors").items()


class Engine:
    """The model on one harness, with ``image_limit`` images per prompt."""

    def __init__(
        self, *, image_limit: int | None = None, sm70=None, dtype=torch.float32
    ):
        from vllm.config import set_current_vllm_config
        from vllm.config.multimodal import MultiModalConfig

        from vllm_sm70.models.inkling2_sm70 import model as model_module
        from vllm_sm70.models.inkling2_sm70.sm70_configs import (
            Inkling2LoopSm70MMConfig,
        )

        self.harness = hx.PagedHarness(1, block_size=BLOCK_SIZE, dtype=dtype)
        config = self.harness.vllm_config(0)
        model_config = config.model_config
        model_config.hf_config = Inkling2LoopSm70MMConfig(**_raw_config())
        model_config.head_dtype = dtype
        model_config.model = str(TINY)
        limits = {} if image_limit is None else {"image": image_limit}
        model_config.multimodal_config = MultiModalConfig(limit_per_prompt=limits)
        config.parallel_config.pipeline_parallel_size = 1
        config.additional_config = None if sm70 is None else {"sm70": sm70}
        with self.harness.tp_rank(0), set_current_vllm_config(config):
            self.model = model_module.Inkling2LoopSm70ForConditionalGeneration(
                vllm_config=config
            )
        self.loaded = self.model.load_weights(_weights())
        self.harness.bind()
        self.num_visits = len(self.model.model.visits)
        self.hidden = self.model.text_config.hidden_size

    def run(self, plan, embeds: torch.Tensor) -> torch.Tensor:
        """Rows [T, visits * H + H + vocab] of one request given its merged
        input embeddings [T, H]."""
        for req_id in list(self.harness.requests):
            self.harness.free_request(req_id)
        body = self.model.model

        def forward(rank, batch):
            states = []
            body.visit_hook = lambda visit, layer, p, state: states.append(
                state.clone()
            )
            try:
                with torch.no_grad():
                    final = self.model(
                        None, batch.positions, inputs_embeds=batch.take({"a": embeds})
                    )
                    logits = self.model.compute_logits(final)
            finally:
                body.visit_hook = None
            return torch.cat([torch.stack(states, 1).flatten(1), final, logits], dim=1)

        return self.harness.run(plan, forward)["a"]


@pytest.fixture(scope="module")
def engine() -> Engine:
    return Engine()


@pytest.fixture(scope="module")
def oracle() -> Inkling2Oracle:
    return Inkling2Oracle.from_checkpoint(TINY)


@pytest.fixture(scope="module")
def case() -> SimpleNamespace:
    """The fixture's text + image case; ``raw`` are the tiles as the image
    processor emits them (0..255)."""
    from safetensors.torch import load_file

    cases = load_file(TINY / "cases.safetensors")
    ids = cases["image.input_ids"][0]
    return SimpleNamespace(
        ids=ids,
        is_image=ids == _raw_config()["image_token_id"],
        raw=(cases["image.pixel_values"] + 1.0) * 127.5,
        grid=cases["image.image_grid_thw"],
        features=cases["image.features"],
        inputs_embeds=cases["image.inputs_embeds"],
        logits=cases["image.logits"],
        last_token_visits=cases["image.visit_states_last_token"],
    )


@pytest.fixture(scope="module")
def merged(engine, case) -> SimpleNamespace:
    with torch.no_grad():
        features = engine.model.embed_multimodal(
            vision_pixels=case.raw, vision_grid_thw=case.grid
        )
        embeds = engine.model.embed_input_ids(
            case.ids, features, is_multimodal=case.is_image
        )
    return SimpleNamespace(features=features, embeds=embeds)


# --- interface and registration ------------------------------------------------


def test_class_is_multimodal_with_a_registered_processor(engine):
    from vllm.model_executor.models.interfaces import supports_multimodal
    from vllm.model_executor.models.interfaces_base import is_text_generation_model

    from vllm_sm70.models.inkling2_sm70 import processing

    cls = type(engine.model)
    assert supports_multimodal(cls) and supports_multimodal(engine.model)
    assert is_text_generation_model(cls)
    factories = cls._processor_factory
    assert factories.processor is processing.Inkling2MultiModalProcessor
    assert factories.info is processing.Inkling2ProcessingInfo
    assert factories.dummy_inputs is processing.Inkling2DummyInputsBuilder
    assert cls.get_placeholder_str("image", 0) == "<|content_image|>"
    with pytest.raises(ValueError, match="image"):
        cls.get_placeholder_str("audio", 0)
    assert engine.model.get_language_model() is engine.model


def test_encoder_is_built_and_every_parameter_loads(engine):
    model = engine.model
    names = {name for name, _ in model.named_parameters()}
    visual = {name for name in names if name.startswith("visual.")}
    assert visual and engine.loaded == names
    assert model.visual.dtype == torch.float32
    assert model.visual.text_hidden_size == engine.hidden
    assert model.max_image_tokens == MINI_CAP_TOKENS


# --- the text + image case -----------------------------------------------------


def test_embed_multimodal_splits_features_by_image(engine, case, merged):
    sizes = case.grid.prod(-1).tolist()
    assert [f.shape for f in merged.features] == [(n, engine.hidden) for n in sizes]
    got = torch.cat(list(merged.features))
    assert (got - case.features).abs().max() <= 1e-5
    assert engine.model.embed_multimodal() == ()


def test_image_rows_skip_embed_norm(engine, oracle, case, merged):
    embeds, features = merged.embeds, torch.cat(list(merged.features))
    assert torch.equal(embeds[case.is_image], features)
    text = ~case.is_image
    assert torch.allclose(embeds[text], oracle.embed(case.ids[text]), atol=1e-6, rtol=0)
    assert (embeds - case.inputs_embeds).abs().max() <= 1e-5
    # The norm would have changed the image rows.
    body = engine.model.model
    from vllm_sm70.models.inkling2_sm70.ops.torch_ref import rms_norm

    normed = rms_norm(features, body.embed_norm.weight[0], body.eps)
    assert (normed - features).abs().max() > 1e-2
    # Text-only calls are unchanged.
    assert torch.equal(
        engine.model.embed_input_ids(case.ids[text]), embeds[text].clone()
    )
    with pytest.raises(ValueError, match="is_multimodal"):
        engine.model.embed_input_ids(case.ids, merged.features)


@pytest.mark.parametrize("plan", ["prefill", "decode", "chunk3"])
def test_text_image_case_matches_oracle_and_reference(
    engine, oracle, case, merged, plan
):
    length = case.ids.shape[0]
    want = oracle.forward(inputs_embeds=merged.embeds[None])
    rows = engine.run(hx.scenarios(length, length, prefix_len=0)[plan], merged.embeds)
    assert rows.shape[0] == length
    split = engine.num_visits * engine.hidden
    visits = rows[:, :split].unflatten(1, (engine.num_visits, -1))
    per_visit = (visits - want.visit_states[:, 0].transpose(0, 1)).abs().amax((0, 2))
    worst = int(per_visit.argmax())
    assert per_visit[worst] <= TOL, f"visit {worst} is off by {per_visit[worst]:.3e}"
    final, logits = (
        rows[:, split : split + engine.hidden],
        rows[:, split + engine.hidden :],
    )
    assert (final - want.final_hidden_state[0]).abs().max() <= TOL
    assert (logits - want.logits[0]).abs().max() <= TOL
    # The Hugging Face recordings of the same prompt and images.
    assert (visits[-1] - case.last_token_visits).abs().max() <= TOL
    assert (logits - case.logits).abs().max() <= TOL


def test_image_features_change_the_output(engine, case, merged):
    """The comparison above is not satisfied by a model that ignores images."""
    length = case.ids.shape[0]
    plan = hx.scenarios(length, length, prefix_len=0)["prefill"]
    text_only = engine.model.embed_input_ids(case.ids)
    with_images = engine.run(plan, merged.embeds)
    without = engine.run(plan, text_only)
    assert (with_images[-1] - without[-1]).abs().max() > 1e-2


# --- the per-image cap -----------------------------------------------------------


def _processor(vision_config=None, cap=None, mode=None):
    from vllm.config.multimodal import MultiModalConfig
    from vllm.multimodal.processing import InputProcessingContext

    from vllm_sm70.models.inkling2_sm70 import processing
    from vllm_sm70.models.inkling2_sm70.sm70_configs import (
        IMAGE_ENCODER_TOKEN_CAP_ATTR,
    )

    mm_config = MultiModalConfig()
    hf_config = SimpleNamespace(vision_config=vision_config)
    if cap is not None:
        setattr(hf_config, IMAGE_ENCODER_TOKEN_CAP_ATTR, cap)
    model_config = SimpleNamespace(
        model="inkling2",
        tokenizer_mode=mode or processing.NATIVE_RENDERER_MODE,
        multimodal_config=mm_config,
        get_multimodal_config=lambda: mm_config,
        hf_config=hf_config,
    )
    info = processing.Inkling2ProcessingInfo(InputProcessingContext(model_config, None))
    return processing.Inkling2MultiModalProcessor(
        info, processing.Inkling2DummyInputsBuilder(info)
    )


def _apply(processor, images):
    from vllm.multimodal.processing import ProcessorInputs, TimingContext

    from vllm_sm70.models.inkling2_sm70.processing import IMAGE_MARKER_ID

    inputs = ProcessorInputs(
        prompt=[IMAGE_MARKER_ID] * len(images),
        mm_data_items=processor.info.parse_mm_data({"image": images}),
    )
    return processor.apply(inputs, TimingContext(enabled=False))


def test_default_cap_is_the_measured_encoder_sequence():
    from vllm_sm70.config import Inkling2Config, Sm70Config

    assert Sm70Config().inkling2.max_image_encoder_tokens == 10817
    with pytest.raises(ValueError):
        Inkling2Config(max_image_encoder_tokens=1)
    info = _processor().info
    # 52 x 52 cells of 2 x 2 patches plus the class token is 10817.
    assert info.max_image_tokens == MINI_CAP_TOKENS == 52 * 52
    assert info.image_processor.max_tokens == MINI_CAP_TOKENS
    info.image_processor.check_size(2080, 2080)
    info.image_processor.check_size(40, 2704 * 40)
    for width, height in ((2081, 2080), (2080, 2081), (41, 2704 * 40)):
        with pytest.raises(ValueError, match="too large"):
            info.image_processor.check_size(width, height)


def test_over_cap_image_is_rejected_with_a_clear_error():
    from PIL import Image

    # 17 encoder tokens: 4 cells of 2 x 2 patches plus the class token.
    vision = {"native_patch_size": 4, "adapter_merge_size": MERGE}
    processor = _processor(vision, cap=17)
    assert processor.info.max_image_tokens == 4
    fits = Image.new("RGB", (4 * TILE, TILE), (1, 2, 3))
    out = _apply(processor, [fits, Image.new("RGB", (2 * TILE, 2 * TILE))])
    assert out["mm_kwargs"].get_data()["vision_grid_thw"].tolist() == [
        [1, 1, 4],
        [1, 2, 2],
    ]
    too_wide = Image.new("RGB", (4 * TILE + 1, TILE), (1, 2, 3))
    with pytest.raises(ValueError) as error:
        _apply(processor, [fits, too_wide])
    message = str(error.value)
    assert f"{4 * TILE + 1}x{TILE} pixels is too large" in message
    assert "5 image tokens" in message and "at most 4 per image" in message
    assert "not resized" in message
    with pytest.raises(ValueError, match="no room"):
        _ = _processor(vision, cap=4).info.max_image_tokens


@pytest.mark.parametrize("mode", [None, "hf"])
def test_profiling_image_has_the_cap_size(mode):
    from vllm.multimodal.processing import TimingContext

    processor = _processor(mode=mode)
    builder = processor.dummy_inputs
    (image,) = builder.get_dummy_mm_data(1_000_000, {"image": 1}, {})["image"]
    assert image.size == (2080, 2080)
    (image,) = builder.get_dummy_mm_data(100, {"image": 1}, {})["image"]
    assert image.size == (400, 400)
    # The largest profiling image passes the processor at exactly the cap.
    inputs = processor.get_dummy_inputs(1_000_000, {"image": 1}, {})
    out = processor.apply(inputs, TimingContext(enabled=False))
    data = out["mm_kwargs"].get_data()
    assert data["vision_grid_thw"].tolist() == [[1, 52, 52]]
    assert data["vision_pixels"].shape[0] == MINI_CAP_TOKENS

    small = _processor({"native_patch_size": 4, "adapter_merge_size": MERGE}, cap=37)
    (image,) = small.dummy_inputs.get_dummy_mm_data(4096, {"image": 1}, {})["image"]
    assert image.size == (3 * TILE, 3 * TILE)


def test_model_refuses_an_image_over_the_cap(case):
    # 17 encoder tokens: at most 4 image tokens; the fixture's images take 6, 2.
    capped = Engine(sm70={"inkling2": {"max_image_encoder_tokens": 17}})
    model = capped.model
    assert model.max_image_tokens == 4
    sizes = case.grid.prod(-1).tolist()
    assert sizes == [6, 2]
    with pytest.raises(ValueError, match="6 image tokens, above the limit of 4"):
        model.embed_multimodal(vision_pixels=case.raw, vision_grid_thw=case.grid)
    with torch.no_grad():
        (second,) = model.embed_multimodal(
            vision_pixels=case.raw[6:], vision_grid_thw=case.grid[1:]
        )
    assert (second - case.features[6:]).abs().max() <= 1e-5


def test_query_chunk_bounds_the_math_kernel(engine):
    from vllm_sm70.models.inkling2_sm70 import model as model_module

    chunk = model_module.vision_query_chunk
    budget = model_module.VISION_ATTENTION_BUDGET
    per_score = model_module.MATH_SDPA_BYTES_PER_SCORE
    # The budget covers 5.59 GiB of scores at 6401 tokens and 16 heads.
    assert 6401 * 6401 * 16 * per_score >= 5.59 * 2**30
    assert chunk(10817, 16) == 620
    for tokens, heads in ((10817, 16), (6401, 16), (1601, 16), (26, 2), (10**9, 16)):
        got = chunk(tokens, heads)
        assert 1 <= got <= model_module.MAX_VISION_QUERY_CHUNK
        assert got == 1 or got * tokens * heads * per_score <= budget
    heads = _raw_config()["vision_config"]["num_attention_heads"]
    assert engine.model.visual.attn_query_chunk == chunk(10817, heads)


# --- image limit 0 -----------------------------------------------------------------


def test_image_limit_zero_builds_no_encoder_and_skips_its_weights(case):
    assert any(name.startswith("model.visual.") for name, _ in _weights())
    text_only = Engine(image_limit=0)
    model = text_only.model
    assert model.visual is None
    names = {name for name, _ in model.named_parameters()}
    assert not [name for name in names if "visual" in name]
    assert text_only.loaded == names
    with pytest.raises(ValueError, match="without its vision encoder"):
        model.embed_multimodal(vision_pixels=case.raw, vision_grid_thw=case.grid)
    assert model.embed_multimodal() == ()


def test_vision_and_expert_casts_go_through_the_load_audit(monkeypatch):
    from vllm_sm70.models.inkling2_sm70.fp16_audit import Fp16Audit

    seen = []
    check = Fp16Audit.check

    def record(self, name, source, cast=None):
        seen.append(name)
        return check(self, name, source, cast)

    monkeypatch.setattr(Fp16Audit, "check", record)
    engine = Engine(dtype=torch.float16)
    half = {
        name
        for name, param in engine.model.visual.named_parameters()
        if param.dtype == torch.float16
    }
    assert half
    assert {name.rpartition("visual.")[2] for name in seen if "visual." in name} == half
    assert any(".mlp.experts.w13_weight" in name for name in seen)
    assert any(".mlp.experts.w2_weight" in name for name in seen)
