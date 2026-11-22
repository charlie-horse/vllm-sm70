# SPDX-License-Identifier: Apache-2.0
"""CPU engine harness for the Inkling 2 paged caches, for tests only.

`PagedHarness` plays the scheduler and the runner for modules that read and
write the model-owned caches (`vllm_sm70.models.inkling2_sm70.cache`): it
allocates blocks, binds a cache tensor to every owner, builds each owner's
metadata with the owner's own builder (`backend.py`) and calls the module
under a forward context, one call per TP rank.

    harness = PagedHarness(tp_size=2, block_size=8)
    for rank in range(2):
        with harness.tp_rank(rank):            # TP world size and rank
            modules.append(Module(harness.vllm_config(rank)))
    harness.bind()
    outputs = harness.run(scenarios(20, 14, prefix_len=8)["resume"], forward)

`forward(rank, batch)` runs with the rank's forward context set and returns a
tensor with one row per batch token.

Fidelity:
- Every owner has a block table of its own, with block ids no other owner
  uses in the same order. Ranks share the tables and have their own caches.
- A cache is [blocks, heads, tokens per block, head_size + head_size_v] in
  the owner's dtype. Block 0 is the null block.
- Unwritten slots, the null block and freed blocks hold `poison` (NaN).
  Sliding-window owners lose the blocks before the window at the start of a
  step, as upstream's SlidingWindowManager frees them.
- Requests of one token come first in a batch.
- A padded step gives the builders what the v0.33 runner gives them
  (v1/worker/gpu_model_runner.py ``_build_attention_metadata``). `metadata`
  "full", a FULL graph: padded slot mapping (-1), positions and request rows,
  and the padded count as ``num_actual_tokens``. `metadata` "unpadded", a
  PIECEWISE graph: slot mapping, positions and request rows of the scheduled
  tokens only. The batch tensors (`HarnessBatch`) are padded in both.
- At TP > 1 the ranks run in lockstep and `all_reduce` / `all_gather` of the
  TP group combine their tensors.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

__all__ = [
    "HarnessBatch",
    "HarnessError",
    "HarnessRequest",
    "PagedHarness",
    "add",
    "chunked_plan",
    "decode_plan",
    "free",
    "interleaved_plan",
    "prefill_plan",
    "resume_plan",
    "scenarios",
    "step",
]

Schedule = Mapping[str, int] | Sequence[tuple[str, int]]
Forward = Callable[[int, "HarnessBatch"], Any]


class HarnessError(AssertionError):
    """The module under test broke a rule the harness checks."""


class _Desync(RuntimeError):
    pass


# Plans: lists of operations for PagedHarness.run.


def add(req_id: str, resume_from: str | None = None, prefix_len: int = 0) -> list:
    return [("add", req_id, resume_from, prefix_len)]


def free(req_id: str) -> list:
    return [("free", req_id)]


def step(schedule: Schedule) -> list:
    return [("step", _pairs(schedule))]


def prefill_plan(req_id: str, length: int) -> list:
    """The whole sequence in one step."""
    return add(req_id) + step({req_id: length})


def decode_plan(req_id: str, length: int, prompt_len: int) -> list:
    """A prefill of `prompt_len` tokens, then one token per step."""
    plan = add(req_id) + step({req_id: prompt_len})
    return plan + step({req_id: 1}) * (length - prompt_len)


def chunked_plan(req_id: str, length: int, chunk: int) -> list:
    """Prefill in steps of `chunk` tokens."""
    plan = add(req_id)
    for start in range(0, length, chunk):
        plan += step({req_id: min(chunk, length - start)})
    return plan


def resume_plan(req_id: str, length: int, donor: str, prefix_len: int) -> list:
    """`donor` prefills `prefix_len` tokens; `req_id` takes them over as a
    cached prefix and computes the rest. Both sequences start alike."""
    plan = add(donor) + step({donor: prefix_len})
    return plan + add(req_id, donor, prefix_len) + step({req_id: length - prefix_len})


def interleaved_plan(req_a: str, len_a: int, req_b: str, len_b: int) -> list:
    """Two requests in the same batches: `req_b` arrives during the prefill
    of `req_a`, both then decode; the first to finish is freed."""
    if len_a < 7 or len_b < 4:
        raise ValueError("interleaved_plan needs len_a >= 7 and len_b >= 4")
    first = min(3, len_a - 6)
    plan = add(req_a) + step([(req_a, first)])
    plan += add(req_b) + step([(req_b, 2), (req_a, 2)])
    plan += step([(req_a, len_a - first - 4), (req_b, 1)])
    left = {req_a: 2, req_b: len_b - 3}
    order = [req_a, req_b]
    while any(left.values()):
        batch = [(req_id, 1) for req_id in order if left[req_id]]
        plan += step(batch)
        for req_id, _ in batch:
            left[req_id] -= 1
            if not left[req_id]:
                plan += free(req_id)
        order.reverse()
    return plan


def scenarios(
    len_a: int,
    len_b: int,
    *,
    prefix_len: int,
    prompt_len: int | None = None,
    chunks: Sequence[int] = (1, 3, 7),
) -> dict[str, list]:
    """The plans every paged module is driven through, over requests "a" and
    "b". "resume" needs sequence b to start with the first `prefix_len`
    tokens of a; `prefix_len` is a multiple of `PagedHarness.prefix_alignment`.
    """
    prompt_len = len_a // 2 if prompt_len is None else prompt_len
    plans = {
        "prefill": prefill_plan("a", len_a),
        "decode": decode_plan("a", len_a, prompt_len),
    }
    plans.update({f"chunk{c}": chunked_plan("a", len_a, c) for c in chunks})
    plans["resume"] = resume_plan("b", len_b, "a", prefix_len)
    plans["interleaved"] = interleaved_plan("a", len_a, "b", len_b)
    return plans


def _pairs(schedule: Schedule) -> list[tuple[str, int]]:
    items = schedule.items() if isinstance(schedule, Mapping) else schedule
    return [(str(req_id), int(count)) for req_id, count in items]


@dataclasses.dataclass
class HarnessRequest:
    req_id: str
    first: int  # tokens taken over as a cached prefix
    computed: int  # tokens in the caches, prefix included


@dataclasses.dataclass(frozen=True)
class HarnessBatch:
    """One step's batch. P = num_padded tokens, the first num_tokens real."""

    req_ids: tuple[str, ...]
    query_lens: tuple[int, ...]
    starts: tuple[int, ...]  # position of each request's first batch token
    num_tokens: int
    num_padded: int
    num_reqs_padded: int
    positions: torch.Tensor  # [P] int64, 0 on padding
    query_start_loc: torch.Tensor  # [num_reqs_padded + 1] int32

    def rows(self, req_id: str) -> slice:
        """Batch rows of a request."""
        index = self.req_ids.index(req_id)
        start = int(self.query_start_loc[index])
        return slice(start, start + self.query_lens[index])

    def take(self, sequences: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """[P, ...] batch input from each request's whole sequence [T, ...]:
        the positions this step computes, zeros on padding."""
        first = sequences[self.req_ids[0]]
        out = first.new_zeros((self.num_padded, *first.shape[1:]))
        for req_id, start, count in zip(self.req_ids, self.starts, self.query_lens):
            out[self.rows(req_id)] = sequences[req_id][start : start + count]
        return out

    def split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-request rows of a [P, ...] or [num_tokens, ...] tensor."""
        return {req_id: flat[self.rows(req_id)] for req_id in self.req_ids}


class _TpGroup:
    """Stand-in for the TP group coordinator. With more than one rank, each
    rank runs in a thread and only one runs at a time; a collective hands
    over to the next rank and returns when every rank has made the same call.
    """

    def __init__(self, world_size: int) -> None:
        self.world_size = world_size
        self.rank_in_group = 0
        self._cond = threading.Condition()
        self._turn = -1
        self._live: set[int] = set()
        self._inbox: list[dict[int, tuple[str, torch.Tensor]]] = []
        self._calls: list[int] = []
        self._activate: Callable[[int], None] | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return torch.stack(self._collect("all_reduce", input_)).sum(0)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        return torch.cat(self._collect("all_gather", input_), dim=dim)

    def _pass(self, rank: int) -> None:
        order = [(rank + i) % self.world_size for i in range(1, self.world_size + 1)]
        self._turn = next((r for r in order if r in self._live), -1)
        self._cond.notify_all()

    def _collect(self, op: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        if self._activate is None:
            raise HarnessError(f"{op} outside a harness step")
        rank = self.rank_in_group
        with self._cond:
            call = self._calls[rank]
            self._calls[rank] += 1
            if call == len(self._inbox):
                self._inbox.append({})
            entry = self._inbox[call]
            entry[rank] = (op, tensor)
            self._pass(rank)
            self._cond.wait_for(lambda: self._turn == rank)
            if len(entry) != self.world_size or {o for o, _ in entry.values()} != {op}:
                raise _Desync(
                    f"rank {rank}: collective {call} ({op}) has no match on "
                    f"every rank; got {sorted((r, o) for r, (o, _) in entry.items())}"
                )
        self._activate(rank)
        return [entry[r][1] for r in range(self.world_size)]

    def run(self, fn: Callable[[int], Any], activate: Callable[[int], None]) -> list:
        """`fn(rank)` for every rank; `activate(rank)` whenever a rank takes
        over."""
        size = self.world_size
        self._activate = activate
        try:
            if size == 1:
                activate(0)
                return [fn(0)]
            results: list[Any] = [None] * size
            errors: list[BaseException | None] = [None] * size
            self._inbox, self._calls = [], [0] * size
            self._live, self._turn = set(range(size)), 0
            grad = torch.is_grad_enabled()

            def body(rank: int) -> None:
                try:
                    with self._cond:
                        self._cond.wait_for(lambda: self._turn == rank)
                    activate(rank)
                    with torch.set_grad_enabled(grad):
                        results[rank] = fn(rank)
                    if self._calls[rank] < len(self._inbox):
                        raise _Desync(f"rank {rank} made fewer collectives")
                except BaseException as error:  # noqa: BLE001 - re-raised by the caller
                    errors[rank] = error
                finally:
                    with self._cond:
                        self._live.discard(rank)
                        self._pass(rank)

            threads = [threading.Thread(target=body, args=(r,)) for r in range(size)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            raised = [e for e in errors if e is not None]
            for error in sorted(raised, key=lambda e: isinstance(e, _Desync)):
                raise error
            return results
        finally:
            self._activate = None


class _Table:
    """Blocks of one owner: a pool and each request's block ids."""

    def __init__(self, name: str, spec: Any, num_blocks: int, seed: int) -> None:
        self.name = name
        self.block_size = int(spec.block_size)
        window = getattr(spec, "sliding_window", None)
        self.window = None if window is None else int(window)
        self.pool = list(range(1, num_blocks))
        random.Random(seed).shuffle(self.pool)
        self.refs: dict[int, int] = {}
        self.rows: dict[str, list[int]] = {}

    def skipped_blocks(self, computed: int) -> int:
        if self.window is None:
            return 0
        return max(0, computed - self.window + 1) // self.block_size

    def take(self) -> int:
        if not self.pool:
            raise HarnessError(f"{self.name}: out of blocks; raise num_blocks")
        block = self.pool.pop()
        self.refs[block] = 1
        return block

    def release(self, block: int) -> bool:
        """True when the block went back to the pool."""
        if block == 0:
            return False
        self.refs[block] -= 1
        if self.refs[block]:
            return False
        del self.refs[block]
        self.pool.append(block)
        return True


class PagedHarness:
    """See the module docstring.

    block_size: tokens per attention block (`--block-size`).
    conv_block_size: tokens per conv-state block, when the cache is to be
        allocated with more than the owner's `block_size`.
    dtype: model and cache dtype.
    num_blocks: blocks per cache, the null block included.
    poison: fill of slots that hold nothing; None leaves zeros. The write
        checks need NaN.
    check_writes: fail a step in which a rank writes a slot outside the
        batch's slot mapping, or writes only part of an owner's slots.
    vllm_configs: one config per rank to use instead of the stand-ins.
    device: where the caches and the metadata tensors live.
    cudagraph_mode: the stand-ins' CUDAGraphMode name; FULL_DECODE_ONLY
        builds the steps a FULL decode graph would replay as decode-only.
    """

    def __init__(
        self,
        tp_size: int = 1,
        *,
        block_size: int = 32,
        conv_block_size: int | None = None,
        dtype: torch.dtype = torch.float32,
        num_blocks: int = 64,
        max_num_batched_tokens: int = 256,
        poison: float | None = float("nan"),
        check_writes: bool = True,
        seed: int = 0,
        device: torch.device | str = "cpu",
        vllm_configs: Sequence[Any] | None = None,
        cudagraph_mode: str = "NONE",
    ) -> None:
        self.tp_size = tp_size
        self.cudagraph_mode = cudagraph_mode
        self.block_size = block_size
        self.conv_block_size = conv_block_size
        self.dtype = dtype
        self.num_blocks = num_blocks
        self.poison = poison
        self.check_writes = check_writes and poison is not None and math.isnan(poison)
        self.seed = seed
        self.device = torch.device(device)
        if vllm_configs is None:
            vllm_configs = [
                self._stand_in(max_num_batched_tokens) for _ in range(tp_size)
            ]
        if len(vllm_configs) != tp_size:
            raise ValueError("one vllm config per rank")
        self._configs = list(vllm_configs)
        self._group = _TpGroup(tp_size)
        self.requests: dict[str, HarnessRequest] = {}
        self.owners: list[dict[str, Any]] = []  # per rank: name -> owner
        self.caches: list[dict[str, torch.Tensor]] = []  # per rank
        self.tables: dict[str, _Table] = {}
        self.metadata: list[dict[str, Any]] = []  # last step, per rank
        self.batch: HarnessBatch | None = None  # last step
        self._specs: list[dict[str, Any]] = []
        self._builders: list[dict[Any, Any]] = []

    def _stand_in(self, max_num_batched_tokens: int) -> SimpleNamespace:
        """The config fields the owners and the builders read."""
        from vllm.config.compilation import CUDAGraphMode

        from vllm_sm70.models.inkling2_sm70.config import DECODE_GRAPH_MAX_BATCH

        return SimpleNamespace(
            cache_config=SimpleNamespace(
                block_size=self.block_size, cache_dtype="auto"
            ),
            model_config=SimpleNamespace(dtype=self.dtype, max_model_len=1_048_576),
            compilation_config=SimpleNamespace(
                static_forward_context={},
                fast_moe_cold_start=False,
                cudagraph_mode=CUDAGraphMode[self.cudagraph_mode],
                max_cudagraph_capture_size=DECODE_GRAPH_MAX_BATCH,
            ),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=False,
                max_num_batched_tokens=max_num_batched_tokens,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1, tensor_parallel_size=self.tp_size
            ),
            kv_transfer_config=None,
            speculative_config=None,
            max_in_flight_tokens=max_num_batched_tokens,
        )

    def vllm_config(self, rank: int = 0) -> Any:
        """The config to build rank `rank`'s module (and its owners) with."""
        return self._configs[rank]

    @contextlib.contextmanager
    def _installed(self) -> Iterator[None]:
        from vllm import forward_context
        from vllm.distributed import parallel_state

        saved = parallel_state._TP, forward_context._forward_context
        parallel_state._TP = self._group  # type: ignore[assignment]
        try:
            yield
        finally:
            parallel_state._TP, forward_context._forward_context = saved

    @contextlib.contextmanager
    def tp_rank(self, rank: int) -> Iterator[None]:
        """Build a rank's module inside: the TP world size and rank are the
        harness's, and a real VllmConfig is the current one."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(self._installed())
            self._group.rank_in_group = rank
            config = self._configs[rank]
            if not isinstance(config, SimpleNamespace):
                from vllm.config import set_current_vllm_config

                stack.enter_context(set_current_vllm_config(config))
            yield

    # Caches.

    def bind(self) -> None:
        """Allocate and bind the cache of every owner registered in the
        ranks' static forward contexts. Call once, after the modules exist."""
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase

        if self.owners:
            raise HarnessError("bind() was already called")
        for rank, config in enumerate(self._configs):
            context = config.compilation_config.static_forward_context
            owners = {
                name: layer
                for name, layer in context.items()
                if isinstance(layer, AttentionLayerBase)
            }
            if not owners:
                raise HarnessError(f"rank {rank} registered no cache owner")
            specs = {}
            for name, owner in owners.items():
                spec = owner.get_kv_cache_spec(config)
                if self.conv_block_size and hasattr(owner, "stream_ranges"):
                    spec = spec.copy_with_new_block_size(self.conv_block_size)
                specs[name] = spec
            self.owners.append(owners)
            self._specs.append(specs)
            self._builders.append({})
        if any(set(o) != set(self.owners[0]) for o in self.owners):
            raise HarnessError("the ranks registered different cache owners")
        for index, (name, spec) in enumerate(self._specs[0].items()):
            for specs in self._specs[1:]:
                other = specs[name]
                if (type(other), other.block_size) != (type(spec), spec.block_size):
                    raise HarnessError(f"{name}: the ranks' specs disagree")
            self.tables[name] = _Table(name, spec, self.num_blocks, self.seed + index)
        for owners, specs in zip(self.owners, self._specs):
            caches = {}
            for name, owner in owners.items():
                spec = specs[name]
                caches[name] = torch.full(
                    (
                        self.num_blocks,
                        spec.num_kv_heads,
                        spec.block_size,
                        spec.head_size + spec.head_size_v,
                    ),
                    0.0 if self.poison is None else self.poison,
                    dtype=spec.dtype,
                    device=self.device,
                )
                owner.kv_cache = caches[name]
            self.caches.append(caches)

    @property
    def prefix_alignment(self) -> int:
        """A cached prefix is a multiple of this many tokens."""
        return math.lcm(*(table.block_size for table in self.tables.values()))

    def _poison(self, name: str, block: int) -> None:
        if self.poison is not None:
            for caches in self.caches:
                caches[name][block] = self.poison

    def _release(self, table: _Table, blocks: Iterable[int]) -> None:
        for block in blocks:
            if table.release(block):
                self._poison(table.name, block)

    # Requests.

    def add_request(
        self, req_id: str, *, resume_from: str | None = None, prefix_len: int = 0
    ) -> HarnessRequest:
        """A new request. With `resume_from`, its first `prefix_len` tokens
        are the donor's, shared block by block as a prefix-cache hit."""
        if not self.tables:
            raise HarnessError("call bind() first")
        if req_id in self.requests:
            raise HarnessError(f"request {req_id} exists")
        if resume_from is None:
            prefix_len = 0
        else:
            donor = self.requests[resume_from]
            if not 0 < prefix_len <= donor.computed:
                raise HarnessError("the donor has not computed the prefix")
            if prefix_len % self.prefix_alignment:
                raise HarnessError(
                    f"prefix_len must be a multiple of {self.prefix_alignment}"
                )
        for table in self.tables.values():
            row: list[int] = []
            if resume_from is not None:
                donor_row = table.rows[resume_from]
                skipped = table.skipped_blocks(prefix_len)
                row = [0] * skipped + donor_row[
                    skipped : prefix_len // table.block_size
                ]
                if 0 in row[skipped:]:
                    raise HarnessError(
                        f"{table.name}: the donor ran past the prefix and "
                        "freed blocks of the prefix's window"
                    )
                for block in row[skipped:]:
                    table.refs[block] += 1
            table.rows[req_id] = row
        request = HarnessRequest(req_id, prefix_len, prefix_len)
        self.requests[req_id] = request
        return request

    def free_request(self, req_id: str) -> None:
        del self.requests[req_id]
        for table in self.tables.values():
            self._release(table, table.rows.pop(req_id))

    # Steps.

    def _allocate(self, table: _Table, request: HarnessRequest, count: int) -> None:
        row = table.rows[request.req_id]
        skipped = min(table.skipped_blocks(request.computed), len(row))
        self._release(table, row[:skipped])
        row[:skipped] = [0] * skipped
        needed = -(-(request.computed + count) // table.block_size)
        row.extend(table.take() for _ in range(needed - len(row)))

    def _written(self, cache: torch.Tensor) -> torch.Tensor:
        """[blocks * tokens] bool: the slot holds a value."""
        return (~cache.isnan()).any(dim=3).any(dim=1).flatten()

    def step(
        self,
        schedule: Schedule,
        forward: Forward,
        *,
        pad_tokens: int = 0,
        pad_reqs: int = 0,
        metadata: str = "full",
        tamper: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> list[Any]:
        """Compute the next `n` tokens of every scheduled request and return
        `forward(rank, batch)` of every rank.

        pad_tokens, pad_reqs: padding tokens (slot -1) and empty requests
            appended to the batch.
        metadata: "full" or "unpadded", what the builders see of the padding
            (module docstring).
        tamper: called with (rank, owner name -> metadata) before the
            forward, to inject faults.
        """
        from vllm.forward_context import ForwardContext
        from vllm.v1.attention.backend import CommonAttentionMetadata

        if metadata not in ("full", "unpadded"):
            raise ValueError(f"unknown metadata mode {metadata!r}")
        pairs = _pairs(schedule)
        if not pairs or any(count < 1 for _, count in pairs):
            raise ValueError("a step schedules at least one token per request")
        if len({req_id for req_id, _ in pairs}) != len(pairs):
            raise ValueError("a request appears once per step")
        pairs.sort(key=lambda pair: pair[1] != 1)  # decode rows first
        requests = [self.requests[req_id] for req_id, _ in pairs]
        counts = [count for _, count in pairs]
        num_reqs, num_tokens = len(pairs), sum(counts)
        num_padded = num_tokens + pad_tokens

        starts = np.cumsum([0, *counts])
        query_start_loc = torch.tensor(
            [*starts, *[num_tokens] * pad_reqs], dtype=torch.int32
        )
        seq_lens = torch.tensor(
            [r.computed + c for r, c in zip(requests, counts)] + [0] * pad_reqs,
            dtype=torch.int32,
        )
        positions = torch.zeros(num_padded, dtype=torch.int64)
        row_of = np.zeros(num_tokens, dtype=np.int64)
        for row, (request, count) in enumerate(zip(requests, counts)):
            rows = slice(starts[row], starts[row + 1])
            positions[rows] = torch.arange(request.computed, request.computed + count)
            row_of[rows] = row
        batch = HarnessBatch(
            req_ids=tuple(r.req_id for r in requests),
            query_lens=tuple(counts),
            starts=tuple(r.computed for r in requests),
            num_tokens=num_tokens,
            num_padded=num_padded,
            num_reqs_padded=num_reqs + pad_reqs,
            positions=positions,
            query_start_loc=query_start_loc,
        )

        block_tables: dict[str, torch.Tensor] = {}
        slot_mappings: dict[str, torch.Tensor] = {}
        real_positions = positions[:num_tokens].numpy()
        for name, table in self.tables.items():
            for request, count in zip(requests, counts):
                self._allocate(table, request, count)
            width = max(len(table.rows[r.req_id]) for r in requests) + 1
            block_table = np.zeros((num_reqs + pad_reqs, width), dtype=np.int32)
            for row, request in enumerate(requests):
                blocks = table.rows[request.req_id]
                block_table[row, : len(blocks)] = blocks
            slots = np.full(num_padded, -1, dtype=np.int64)
            block = block_table[row_of, real_positions // table.block_size]
            slots[:num_tokens] = (
                block.astype(np.int64) * table.block_size
                + real_positions % table.block_size
            )
            block_tables[name] = torch.from_numpy(block_table).to(self.device)
            slot_mappings[name] = torch.from_numpy(slots).to(self.device)

        seen_reqs, seen_tokens = num_reqs + pad_reqs, num_padded
        if metadata == "unpadded":
            seen_reqs, seen_tokens = num_reqs, num_tokens
            for name in block_tables:
                block_tables[name] = block_tables[name][:num_reqs]
                slot_mappings[name] = slot_mappings[name][:num_tokens]

        device_starts = query_start_loc[: seen_reqs + 1].clone().to(self.device)

        def common(name: str) -> CommonAttentionMetadata:
            return CommonAttentionMetadata(
                query_start_loc=device_starts,
                query_start_loc_cpu=query_start_loc[: seen_reqs + 1],
                seq_lens=seq_lens[:seen_reqs].to(self.device),
                num_reqs=seen_reqs,
                num_actual_tokens=seen_tokens,
                max_query_len=max(counts),
                max_seq_len=int(seq_lens.max()),
                block_table_tensor=block_tables[name],
                slot_mapping=slot_mappings[name],
                seq_lens_cpu_upper_bound=seq_lens[:seen_reqs].clone(),
                positions=positions[:seen_tokens].to(self.device),
            )

        # As the runner: one build per builder, then a block-table update
        # for every owner.
        contexts = []
        self.metadata = []
        for rank, config in enumerate(self._configs):
            built: dict[Any, Any] = {}
            metadata: dict[str, Any] = {}
            for name, owner in self.owners[rank].items():
                spec = self._specs[rank][name]
                builder_cls = owner.get_attn_backend().get_builder_cls()
                key = (builder_cls, spec)
                builder = self._builders[rank].get(key)
                if builder is None:
                    builder = builder_cls(spec, [name], config, self.device)
                    self._builders[rank][key] = builder
                if key not in built:
                    built[key] = builder.build(0, common(name))
                metadata[name] = builder.update_block_table(
                    built[key], block_tables[name], slot_mappings[name]
                )
            if tamper is not None:
                tamper(rank, metadata)
            self.metadata.append(metadata)
            contexts.append(
                ForwardContext(
                    no_compile_layers=config.compilation_config.static_forward_context,
                    attn_metadata=metadata,
                    slot_mapping={n: m.slot_mapping for n, m in metadata.items()},
                )
            )
        self.batch = batch

        before = None
        if self.check_writes:
            before = [
                {name: self._written(cache) for name, cache in caches.items()}
                for caches in self.caches
            ]

        def activate(rank: int) -> None:
            from vllm import forward_context

            self._group.rank_in_group = rank
            forward_context._forward_context = contexts[rank]

        with self._installed():
            results = self._group.run(lambda rank: forward(rank, batch), activate)

        if before is not None:
            self._check_writes(before, slot_mappings, num_tokens)
        for request, count in zip(requests, counts):
            request.computed += count
        return results

    def _check_writes(
        self,
        before: list[dict[str, torch.Tensor]],
        slot_mappings: dict[str, torch.Tensor],
        num_tokens: int,
    ) -> None:
        for rank, caches in enumerate(self.caches):
            for name, cache in caches.items():
                after = self._written(cache)
                new = after & ~before[rank][name]
                if not new.any():
                    continue
                scheduled = torch.zeros_like(after)
                scheduled[slot_mappings[name][:num_tokens]] = True
                where = f"rank {rank}, {name}"
                if after[: cache.shape[2]].any():
                    raise HarnessError(f"{where}: the null block was written")
                if (new & ~scheduled).any():
                    stray = (new & ~scheduled).nonzero().flatten().tolist()
                    raise HarnessError(
                        f"{where}: wrote slots {stray} outside the slot mapping"
                    )
                if not after[scheduled].all():
                    missing = (scheduled & ~after).nonzero().flatten().tolist()
                    raise HarnessError(f"{where}: slots {missing} were not written")

    def run(
        self,
        plan: Iterable[tuple],
        forward: Forward,
        *,
        combine: str | Callable[[list[Any]], torch.Tensor] = "first",
        **step_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run a plan (see `scenarios`) and return each request's outputs,
        one row per position it computed, in position order; a resumed
        request's rows start at its `HarnessRequest.first`.

        combine: "first" takes rank 0's tensor (the ranks of a TP model
            return the same one), "sum" adds the ranks' partials, a callable
            gets the list of rank results.
        """
        parts: dict[str, list[torch.Tensor]] = {}
        for op, *args in plan:
            if op == "add":
                req_id, resume_from, prefix_len = args
                self.add_request(req_id, resume_from=resume_from, prefix_len=prefix_len)
            elif op == "free":
                self.free_request(args[0])
            elif op == "step":
                results = self.step(args[0], forward, **step_kwargs)
                if combine == "first":
                    flat = results[0]
                elif combine == "sum":
                    flat = torch.stack(results).sum(0)
                else:
                    flat = combine(results)  # type: ignore[operator]
                assert self.batch is not None
                for req_id, rows in self.batch.split(flat).items():
                    parts.setdefault(req_id, []).append(rows)
            else:
                raise ValueError(f"unknown plan operation {op!r}")
        return {req_id: torch.cat(rows) for req_id, rows in parts.items()}
