# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 cache-plumbing stub (test tree, M12 C4).

A weightless model that owns the cache layers of the real model (one K/V
cache and one conv state per layer visit) and runs no attention. Each
forward writes a record per token into every cache and reads records of
earlier positions back through the block tables:

    record(position q) = (h_q, q, owner, head, stream)
    h_q = sum of mix(token_i, i) for i <= q, modulo 2**30

``h`` continues from the record of the position before the batch, so it is a
function of the whole request and is right only if that record survived in
the cache. The next token is a function of ``h`` and of every value read
back, summed over the TP ranks, so a caller that knows the token ids can
predict each generated token (``Reference``). A record with the wrong
position or owner makes the model emit ``ERROR_TOKEN``.

Reads of the token at position p, per owner (window W, owner index g):

    conv state      p-1 .. p-(W-1)
    sliding K/V     p-1, p-(W-1), p-1-((7p+g) mod (W-1))
    full K/V        p-1, 0, (31p+g) mod p

``register()`` adds the architecture ``STUB_ARCH`` to vLLM; an engine selects
it with ``--hf-overrides '{"architectures": ["Inkling2CachePlumbingStub"]}'``
and ``--load-format dummy``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

STUB_ARCH = "Inkling2CachePlumbingStub"
STUB_TARGET = "inkling2_stub:Inkling2CachePlumbingStub"

MASK = (1 << 30) - 1
ERROR_TOKEN = 1
TOKEN_BASE = 1024
TOKEN_SPAN = 1 << 16

CONV, SLIDING, FULL = "conv", "sliding", "full"

# Record layout: values per (head, stream), each an exact fp16 integer.
_DIGIT = 1024
_HASH, _POS, _OWNER, _HEAD, _STREAM = slice(0, 3), slice(3, 6), 6, 7, 8
RECORD = 9
# Output layout in the hidden state: 4 base-256 digits of the digest, then
# the number of failed checks. Sums over up to 8 ranks stay exact in fp16.
_OUT_DIGITS = 4
_OUT_BAD = 4


def token_span(vocab_size: int) -> int:
    span = min(TOKEN_SPAN, vocab_size - TOKEN_BASE)
    if span < 2:
        raise ValueError(f"vocab {vocab_size} is too small for the stub")
    return span


def cache_plan(text_config: Any) -> list[tuple[str, int]]:
    """(kind, window) of every cache owner, in owner-index order: the K/V
    caches in cache-index order, then the conv states."""
    from vllm_sm70.models.inkling2_sm70.cache import layer_passes

    local = set(text_config.local_layer_ids)
    visits = layer_passes(text_config)
    attn = [
        (SLIDING, int(text_config.sliding_window_size)) if layer in local else (FULL, 0)
        for layer, _ in visits
    ]
    conv = [(CONV, int(text_config.sconv_kernel_size))] * len(visits)
    return attn + conv


def _coef(owner: int, read: int) -> int:
    return 2 * (owner * 8 + read) + 3


class Reference:
    """Host-side prediction of the stub's output (numpy only)."""

    def __init__(self, text_config: Any) -> None:
        self.plan = cache_plan(text_config)
        self.span = token_span(int(text_config.vocab_size))

    @staticmethod
    def mix(tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        x = (tokens * 1000003 + positions * 8191 + 12345) & MASK
        x = (x * 40503) & MASK
        x ^= x >> 13
        return (x * 60493) & MASK

    @staticmethod
    def read_positions(kind: str, window: int, owner: int, p: np.ndarray) -> np.ndarray:
        """[reads, len(p)] positions read by the tokens at ``p``; -1 = none."""
        if kind == CONV:
            return np.stack([p - k for k in range(1, window)])
        if kind == SLIDING:
            return np.stack(
                [p - 1, p - (window - 1), p - 1 - (p * 7 + owner) % (window - 1)]
            )
        some = (p * 31 + owner) % np.maximum(p, 1)
        return np.stack([p - 1, np.where(p > 0, 0, -1), np.where(p > 0, some, -1)])

    def next_tokens(self, token_ids: list[int]) -> np.ndarray:
        """The token the stub emits after each position of ``token_ids``."""
        tokens = np.asarray(token_ids, dtype=np.int64)
        p = np.arange(len(tokens), dtype=np.int64)
        h = np.cumsum(self.mix(tokens, p)) & MASK
        digest = h.copy()
        for owner, (kind, window) in enumerate(self.plan):
            reads = self.read_positions(kind, window, owner, p)
            for read, q in enumerate(reads):
                value = np.where(q >= 0, h[np.maximum(q, 0)], 0)
                digest = (digest + _coef(owner, read) * value) & MASK
        return TOKEN_BASE + digest % self.span

    def generate(self, prompt_ids: list[int], max_tokens: int) -> list[int]:
        """Greedy continuation of ``prompt_ids``."""
        ids = list(prompt_ids)
        for _ in range(max_tokens):
            ids.append(int(self.next_tokens(ids)[-1]))
        return ids[len(prompt_ids) :]


def _mix(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    x = (tokens * 1000003 + positions * 8191 + 12345) & MASK
    x = (x * 40503) & MASK
    x = x ^ (x >> 13)
    return (x * 60493) & MASK


def _read_positions(
    kind: str, window: int, owner: int, p: torch.Tensor
) -> torch.Tensor:
    """[tokens, reads] positions read by the tokens at ``p``; -1 = none."""
    if kind == CONV:
        return torch.stack([p - k for k in range(1, window)], dim=1)
    if kind == SLIDING:
        some = p - 1 - (p * 7 + owner) % (window - 1)
        return torch.stack([p - 1, p - (window - 1), some], dim=1)
    started = p > 0
    some = (p * 31 + owner) % p.clamp(min=1)
    none = torch.full_like(p, -1)
    return torch.stack(
        [p - 1, torch.where(started, 0, none), torch.where(started, some, none)],
        dim=1,
    )


def _digits(value: torch.Tensor, base: int, count: int) -> torch.Tensor:
    return torch.stack([(value // base**k) % base for k in range(count)], dim=-1)


def _undigits(digits: torch.Tensor, base: int) -> torch.Tensor:
    weights = base ** torch.arange(digits.shape[-1], device=digits.device)
    return (digits * weights).sum(-1)


class _Port:
    """One cache owner as the stub reads and writes it."""

    def __init__(self, owner: Any, index: int, kind: str, window: int) -> None:
        self.owner = owner
        self.index = index
        self.kind = kind
        self.window = window
        if kind == CONV:
            ranges = owner.stream_ranges
        else:
            ranges = ((0, owner.head_dim), (owner.head_dim, owner.head_dim))
        if min(width for _, width in ranges) < RECORD:
            raise ValueError(
                f"{owner.prefix}: a stream of {ranges} is narrower than {RECORD}"
            )
        self.offsets = [offset for offset, _ in ranges]

    def _index(self, cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = cache.device
        heads = torch.arange(cache.shape[1], device=device)
        cols = torch.tensor(self.offsets, device=device)[:, None] + torch.arange(
            RECORD, device=device
        )
        return heads, cols.flatten()

    def write(self, md: Any, positions: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Store the records of the batch. Returns [tokens] failed checks."""
        cache = self.owner.kv_cache
        num_heads, tokens_per_block = cache.shape[1], cache.shape[2]
        if md.block_size != tokens_per_block:
            raise ValueError(
                f"{self.owner.prefix}: metadata block size {md.block_size}, "
                f"cache holds {tokens_per_block} tokens per block"
            )
        n = positions.shape[0]
        heads, cols = self._index(cache)
        slots = md.slot_mapping[:n].long()
        row = md.seq_idx[:n].long()
        table = md.block_table[row, positions // tokens_per_block].long()
        bad = slots != table * tokens_per_block + positions % tokens_per_block
        slots = slots.clamp(min=0)
        streams = len(self.offsets)
        record = torch.zeros(
            n, num_heads, streams, RECORD, dtype=torch.int64, device=cache.device
        )
        record[..., _HASH] = _digits(h, _DIGIT, 3)[:, None, None, :]
        record[..., _POS] = _digits(positions, _DIGIT, 3)[:, None, None, :]
        record[..., _OWNER] = self.index
        record[..., _HEAD] = heads[None, :, None]
        record[..., _STREAM] = torch.arange(streams, device=cache.device)
        cache[
            (slots // tokens_per_block)[:, None, None],
            heads[None, :, None],
            (slots % tokens_per_block)[:, None, None],
            cols[None, None, :],
        ] = record.flatten(2).to(cache.dtype)
        return bad.long()

    def read(self, md: Any, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Records at positions ``q`` [tokens, reads] of each token's request.
        Returns (h [tokens, reads], 0 where q < 0; failed checks [tokens])."""
        cache = self.owner.kv_cache
        tokens_per_block = cache.shape[2]
        heads, cols = self._index(cache)
        valid = q >= 0
        position = q.clamp(min=0)
        row = md.seq_idx[: q.shape[0]].long()
        logical = position // tokens_per_block
        in_table = logical < md.block_table.shape[1]
        block = md.block_table[
            row[:, None], logical.clamp(max=md.block_table.shape[1] - 1)
        ].long()
        raw = cache[
            block[:, :, None, None],
            heads[None, None, :, None],
            (position % tokens_per_block)[:, :, None, None],
            cols[None, None, None, :],
        ]
        streams = len(self.offsets)
        record = (
            raw.float()
            .nan_to_num(nan=-1.0, posinf=-1.0, neginf=-1.0)
            .round()
            .long()
            .unflatten(-1, (streams, RECORD))
        )
        h = _undigits(record[..., _HASH], _DIGIT)
        ok = (
            (_undigits(record[..., _POS], _DIGIT) == position[:, :, None, None])
            & (record[..., _OWNER] == self.index)
            & (record[..., _HEAD] == heads[None, None, :, None])
            & (record[..., _STREAM] == torch.arange(streams, device=cache.device))
            & (h == h[:, :, :1, :1])
        ).all(-1).all(-1) & in_table
        bad = (valid & ~ok).sum(1)
        return torch.where(valid, h[:, :, 0, 0], 0), bad


def _run_caches(
    out: torch.Tensor, input_ids: torch.Tensor, positions: torch.Tensor, stub: Any
) -> None:
    """Write this rank's digest digits and failure count into ``out``."""
    from vllm_sm70.models.inkling2_sm70.backend import cache_metadata

    out.zero_()
    ports: list[_Port] = stub.ports
    metadata = [cache_metadata(port.owner) for port in ports]
    if any(md is None for md in metadata):
        return
    n = metadata[0].num_actual_tokens
    if n == 0:
        return
    ids = input_ids[:n].long()
    first = metadata[0]
    pos = first.positions[:n].long()
    bad = (pos != positions[:n].long()).long()

    start = first.query_start[:n].long()
    before, bad_before = ports[0].read(first, (pos[start] - 1)[:, None])
    bad += bad_before
    mixed = _mix(ids, pos)
    total = mixed.cumsum(0)
    h = (before[:, 0] + total - (total - mixed)[start]) & MASK

    for port, md in zip(ports, metadata):
        bad_port = port.write(md, pos, h)
        if md.num_actual_tokens != n:
            bad_port += 1
        stub.note(port, bad_port, pos, "write")
        bad += bad_port
    digest = h.clone()
    for port, md in zip(ports, metadata):
        q = _read_positions(port.kind, port.window, port.index, pos)
        values, bad_port = port.read(md, q)
        stub.note(port, bad_port, pos, "read")
        bad += bad_port
        for read in range(q.shape[1]):
            digest = (digest + _coef(port.index, read) * values[:, read]) & MASK
    out[:n, :_OUT_DIGITS] = _digits(digest, 256, _OUT_DIGITS).to(out.dtype)
    out[:n, _OUT_BAD] = bad.clamp(max=1).to(out.dtype)


def _eager_break(fn):
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    return eager_break_during_capture(fn)


class Inkling2CachePlumbingStub(nn.Module):
    def __init__(self, *, vllm_config: Any, prefix: str = "") -> None:
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.logger import init_logger
        from vllm_sm70.models.inkling2_sm70.cache import build_cache_owners

        self.logger = init_logger("vllm.sm70.inkling2.stub")
        config = vllm_config.model_config.hf_config
        text_config = config.get_text_config()
        self.dtype = vllm_config.model_config.dtype
        self.hidden_size = int(text_config.hidden_size)
        self.vocab_size = int(text_config.vocab_size)
        self.span = token_span(self.vocab_size)
        self.tp_size = get_tensor_model_parallel_world_size()
        owners = build_cache_owners(text_config, vllm_config=vllm_config)
        self.caches = nn.ModuleList(owners.named_owners().values())
        plan = cache_plan(text_config)
        if len(plan) != len(self.caches):
            raise ValueError("cache plan and cache owners disagree")
        self.ports = [
            _Port(owner, index, kind, window)
            for index, (owner, (kind, window)) in enumerate(zip(self.caches, plan))
        ]
        # Failed checks of this rank, by owner name.
        self.failures: dict[str, int] = {}
        self._run = _eager_break(_run_caches)

    def note(
        self, port: _Port, bad: torch.Tensor, pos: torch.Tensor, what: str
    ) -> None:
        count = int(bad.sum())
        if not count:
            return
        name = port.owner.prefix
        if name not in self.failures:
            where = pos[bad > 0][:8].tolist()
            self.logger.error(
                "cache plumbing: %s check failed on %s (%s, window %d) at positions %s",
                what,
                name,
                port.kind,
                port.window,
                where,
            )
        self.failures[name] = self.failures.get(name, 0) + count

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        embeds = torch.zeros(
            *input_ids.shape,
            self.hidden_size,
            dtype=self.dtype,
            device=input_ids.device,
        )
        embeds[..., :3] = _digits(input_ids.long(), _DIGIT, 3).to(self.dtype)
        return embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        from vllm.distributed import tensor_model_parallel_all_reduce

        if input_ids is None:
            input_ids = _undigits(inputs_embeds[..., :3].float().round().long(), _DIGIT)
        hidden = torch.zeros(
            input_ids.shape[0],
            self.hidden_size,
            dtype=self.dtype,
            device=input_ids.device,
        )
        self._run(hidden, input_ids, positions, self)
        return tensor_model_parallel_all_reduce(hidden)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        summed = hidden_states[:, :_OUT_DIGITS].float().round().long()
        value = _undigits(summed, 256)
        bad = (hidden_states[:, _OUT_BAD].float() > 0.5) | (value % self.tp_size != 0)
        token = TOKEN_BASE + (value // self.tp_size) % self.span
        token = torch.where(bad, ERROR_TOKEN, token)
        logits = torch.zeros(
            hidden_states.shape[0],
            self.vocab_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        return logits.scatter_(1, token[:, None], 30.0)

    def load_weights(self, weights: Any) -> set[str]:
        return set()


def register(config_hook: bool = True) -> None:
    """vLLM general plugin: add ``STUB_ARCH``, with the Inkling 2
    engine-config hook unless ``config_hook`` is False (the hook needs a
    cc 7.0 GPU). Runs in every engine process."""
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_sm70.models.inkling2_sm70 import register_configs

    register_configs()
    if STUB_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(STUB_ARCH, STUB_TARGET)
    if config_hook:
        from vllm_sm70.models.inkling2_sm70.config import Inkling2Sm70ModelConfig

        MODELS_CONFIG_MAP[STUB_ARCH] = Inkling2Sm70ModelConfig


# --- scenario driver: `python inkling2_stub.py '<json options>'` -------------
#
# Starts one engine in this process (the scheduler stays here, TP workers are
# subprocesses), runs one scenario set and prints a line "RESULT <json>".
# Options: config (path of a config.json), work_dir, mode ("main", "preempt"
# or "plan": start and report the KV groups), cpu, tp, block_size, batched_tokens, max_model_len, kv_bytes,
# fault (true: the last conv state reads through another request's block
# table row, which the output must expose).


def _cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _prompt(seed: int, length: int, span: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return (TOKEN_BASE + rng.integers(0, span, size=length)).tolist()


def _torch_slot_mapping(
    num_reqs,
    num_tokens,
    max_num_tokens,
    query_start_loc,
    positions,
    block_table,
    block_table_stride,
    slot_mapping,
    kv_cache_block_size,
    total_cp_world_size,
    total_cp_rank,
    cp_kv_cache_interleave_size,
):
    """Torch form of vLLM's Triton slot-mapping kernel for one CP rank
    (UP v1/worker/block_table.py ``ComputeSlotMappingKernel``)."""
    assert total_cp_world_size == 1
    starts = query_start_loc[: num_reqs + 1].long()
    token = torch.arange(num_tokens, device=positions.device)
    req = torch.searchsorted(starts[1:].contiguous(), token, right=True)
    req = req.clamp(max=num_reqs - 1)
    blocks = block_table[req, positions // kv_cache_block_size].long()
    slot_mapping[:num_tokens] = (
        blocks * kv_cache_block_size + positions % kv_cache_block_size
    )
    slot_mapping[num_tokens:max_num_tokens] = -1


_torch_slot_mapping.register_warmup = lambda *args, **kwargs: None


def _use_cpu_platform() -> None:
    """Run vLLM's V1 runner on the CPU of a torch-only install."""
    import vllm.platforms
    from vllm.platforms.cpu import CpuPlatform

    vllm.platforms._current_platform = CpuPlatform()
    from vllm.utils import torch_utils
    from vllm.v1.worker import block_table

    torch_utils.PIN_MEMORY = False
    if not hasattr(torch.ops._C, "init_cpu_memory_env"):
        torch.ops._C.init_cpu_memory_env = lambda nodes: None
    if not hasattr(torch.ops._C, "greedy_argmax"):
        torch.ops._C.greedy_argmax = lambda logits: logits.argmax(dim=-1)
    block_table._COMPUTE_SLOT_MAPPING_KERNEL = _torch_slot_mapping


def _add_import_path(path: str) -> None:
    """For this process and the processes it starts."""
    import os
    import sys

    inherited = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join([path, *inherited])
    if path not in sys.path:
        sys.path.insert(0, path)


def _install_plugin(work_dir: str) -> None:
    """Make ``register`` a vLLM general plugin of this process and of the
    worker processes it starts."""
    import os
    import pathlib

    dist = pathlib.Path(work_dir) / "plugin" / "inkling2_stub-0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: inkling2-stub\nVersion: 0\n"
    )
    (dist / "entry_points.txt").write_text(
        "[vllm.general_plugins]\ninkling2_stub = inkling2_stub:register\n"
    )
    _add_import_path(str(dist.parent))
    if os.environ.get("VLLM_PLUGINS"):
        os.environ["VLLM_PLUGINS"] += ",inkling2_stub"


class _Probe:
    """Peak block use of the scheduler, per KV group and in total."""

    def __init__(self, scheduler: Any) -> None:
        self.scheduler = scheduler
        self.pool = scheduler.kv_cache_manager.block_pool
        self.null = self.pool.null_block.block_id
        self.num_blocks = self.pool.num_gpu_blocks
        self.groups = []
        for group in scheduler.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            spec = next(iter(getattr(spec, "kv_cache_specs", {None: spec}).values()))
            self.groups.append(
                {
                    "spec": type(spec).__name__,
                    "layers": len(group.layer_names),
                    "block_size": int(group.kv_cache_spec.block_size),
                    "page_bytes": int(group.kv_cache_spec.page_size_bytes),
                    "window": getattr(spec, "sliding_window", None),
                    "peak": 0,
                }
            )
        self.peak_used = 0
        self.preemptions = 0
        schedule, preempt = scheduler.schedule, scheduler._preempt_request

        def probed_schedule(*args: Any, **kwargs: Any) -> Any:
            output = schedule(*args, **kwargs)
            self.sample()
            return output

        def probed_preempt(*args: Any, **kwargs: Any) -> Any:
            self.preemptions += 1
            return preempt(*args, **kwargs)

        scheduler.schedule = probed_schedule
        scheduler._preempt_request = probed_preempt

    def sample(self) -> None:
        used = self.num_blocks - self.pool.get_num_free_blocks()
        self.peak_used = max(self.peak_used, used)
        for request in self.scheduler.running:
            ids = self.scheduler.kv_cache_manager.get_block_ids(request.request_id)
            for group, blocks in zip(self.groups, ids):
                held = sum(1 for block in blocks if block != self.null)
                group["peak"] = max(group["peak"], held)

    def report(self, batched_tokens: int) -> dict[str, Any]:
        """Peaks, with the per-request admission cap of each windowed group:
        cdiv(window - 1 + B, block) + 1."""
        for group in self.groups:
            if group["window"] is not None:
                group["cap"] = (
                    _cdiv(group["window"] - 1 + batched_tokens, group["block_size"]) + 1
                )
        return {
            "num_gpu_blocks": self.num_blocks,
            "peak_used_blocks": self.peak_used,
            "preemptions": self.preemptions,
            "groups": self.groups,
        }


def _inject_fault() -> None:
    import copy
    import importlib

    port_cls = importlib.import_module("inkling2_stub")._Port
    read = port_cls.read

    def faulty_read(self: Any, md: Any, q: torch.Tensor) -> Any:
        if self.kind == CONV and self.index % 48 == 47:
            md = copy.copy(md)
            md.block_table = md.block_table.roll(1, 0)
        return read(self, md, q)

    port_cls.read = faulty_read


def _start(options: dict[str, Any], model_dir: str, num_blocks: int | None) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {}
    if num_blocks is not None:
        kwargs["num_gpu_blocks_override"] = num_blocks
    if options.get("kv_bytes"):
        kwargs["kv_cache_memory_bytes"] = int(options["kv_bytes"])
    return LLM(
        model=model_dir,
        dtype="float16",
        hf_overrides={"architectures": [STUB_ARCH]},
        skip_tokenizer_init=True,
        load_format="dummy",
        tensor_parallel_size=int(options.get("tp", 1)),
        block_size=int(options["block_size"]),
        max_model_len=int(options["max_model_len"]),
        max_num_batched_tokens=int(options["batched_tokens"]),
        max_num_seqs=8,
        enable_prefix_caching=True,
        async_scheduling=False,
        enforce_eager=bool(options.get("cpu")),
        **kwargs,
    )


def _generate(llm: Any, reference: Reference, cases: dict[str, tuple[list[int], int]]):
    """Run the cases as one batch; one result row per case."""
    from vllm import SamplingParams

    names = list(cases)
    outputs = llm.generate(
        [{"prompt_token_ids": cases[name][0]} for name in names],
        [
            SamplingParams(
                temperature=0.0,
                max_tokens=cases[name][1],
                ignore_eos=True,
                detokenize=False,
            )
            for name in names
        ],
        use_tqdm=False,
    )
    rows = []
    for name, output in zip(names, outputs):
        prompt, max_tokens = cases[name]
        got = list(output.outputs[0].token_ids)
        want = reference.generate(prompt, max_tokens)
        wrong = [i for i, (a, b) in enumerate(zip(got, want)) if a != b]
        rows.append(
            {
                "case": name,
                "prompt_tokens": len(prompt),
                "ok": got == want,
                "first_wrong": wrong[0] if wrong else None,
                "errors": got.count(ERROR_TOKEN),
                "cached_tokens": output.num_cached_tokens,
            }
        )
    return rows


PREEMPT_REQUESTS = 6


def _pool_for_preemption(
    groups: list[dict[str, Any]], prompt_len: int, batched: int
) -> tuple[int, int]:
    """(pool blocks, max_tokens): a pool that admits ``PREEMPT_REQUESTS``
    requests one after the other and that their full-attention blocks fill
    before they have generated ``max_tokens`` tokens."""
    windowed = [g for g in groups if g["window"] is not None]
    full = [g for g in groups if g["window"] is None]
    steady = sum(_cdiv(g["window"], g["block_size"]) + 1 for g in windowed)
    cap = sum(_cdiv(g["window"] - 1 + batched, g["block_size"]) + 1 for g in windowed)
    context = sum(_cdiv(prompt_len, g["block_size"]) for g in full)
    count = PREEMPT_REQUESTS
    pool = (count - 1) * (steady + context) + cap + context + len(groups)
    slack = pool - count * (steady - len(windowed) + context)
    blocks_per_token = count * sum(1 / g["block_size"] for g in full)
    return pool, int(1.5 * slack / blocks_per_token) + 64


def run_scenarios(options: dict[str, Any]) -> dict[str, Any]:
    import json
    import os
    import pathlib
    import subprocess
    import sys

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    work_dir = pathlib.Path(options["work_dir"])
    model_dir = work_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(pathlib.Path(options["config"]).read_text())
    (model_dir / "config.json").write_text(json.dumps(config))
    _add_import_path(str(pathlib.Path(__file__).resolve().parent))
    if options.get("cpu"):
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        _use_cpu_platform()
        register(config_hook=False)
    else:
        _install_plugin(str(work_dir))
        register()

    from vllm_sm70.models.inkling2_sm70.configs import Inkling2ModelConfig

    text_config = Inkling2ModelConfig(**config["text_config"])
    reference = Reference(text_config)
    span = reference.span
    window = int(text_config.sliding_window_size)
    batched = int(options["batched_tokens"])

    def scheduler_of(llm: Any) -> Any:
        return llm.llm_engine.engine_core.engine_core.scheduler

    if options.get("fault"):
        _inject_fault()
    if options["mode"] == "plan":
        llm = _start(options, str(model_dir), None)
        probe = _Probe(scheduler_of(llm))
        rows = []
    elif options["mode"] == "main":
        llm = _start(options, str(model_dir), None)
        probe = _Probe(scheduler_of(llm))
        # The base request ends 2 tokens past a block boundary, so the conv
        # state before that boundary is still in the cache when it finishes.
        block = int(options["block_size"])
        base_len = window + 2 * batched + 100
        base_len += (2 - base_len - 8) % block
        base = _prompt(3, base_len, span)
        turn = base + reference.generate(base, 8)
        rows = _generate(
            llm,
            reference,
            {"chunked": (_prompt(1, 2 * batched + window + 37, span), 48)},
        )
        rows += _generate(
            llm,
            reference,
            {
                "concurrent_short": (_prompt(2, batched // 2 + 5, span), 64),
                "concurrent_long": (_prompt(4, batched + window // 2 + 11, span), 64),
            },
        )
        rows += _generate(llm, reference, {"prefix_base": (base, 8)})
        rows += _generate(
            llm,
            reference,
            {
                "prefix_resume": (turn + _prompt(5, 70, span), 24),
                "prefix_repeat": (base, 24),
            },
        )
    else:
        prompt_len = batched // 2
        # The KV groups come from an engine of their own, in another process.
        plan = subprocess.run(
            [sys.executable, __file__, json.dumps(dict(options, mode="plan"))],
            capture_output=True,
            text=True,
            check=True,
        )
        groups = json.loads(plan.stdout.rsplit("RESULT ", 1)[1])["groups"]
        num_blocks, max_tokens = _pool_for_preemption(groups, prompt_len, batched)
        llm = _start(options, str(model_dir), num_blocks)
        probe = _Probe(scheduler_of(llm))
        rows = _generate(
            llm,
            reference,
            {
                f"preempt_{i}": (_prompt(10 + i, prompt_len, span), max_tokens)
                for i in range(PREEMPT_REQUESTS)
            },
        )
    return {
        "options": options,
        "cases": rows,
        "kv_cache_layout": llm.llm_engine.vllm_config.cache_config.kv_cache_layout,
        **probe.report(batched),
    }


if __name__ == "__main__":
    import json
    import sys

    print("RESULT " + json.dumps(run_scenarios(json.loads(sys.argv[1]))), flush=True)
