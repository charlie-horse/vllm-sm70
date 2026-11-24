# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the modules that own the paged caches.

Every visit of a decoder layer (layer, pass) has its own attention K/V cache
and short-conv state, each one ``AttentionLayerBase`` owner in vLLM's forward
context, so the KV planner allocates and the runner binds them:

    <prefix>.<layer>.attn[.pass<p>]        Inkling2AttentionCache
    <prefix>.<layer>.conv_state[.pass<p>]  Inkling2ConvState

Pass 0 has no suffix; a name holds one integer, the layer. The conv state
packs four streams per local KV head into one sliding-window spec of
``sconv_kernel_size`` tokens: K, V, attention output, MLP output. The two
output streams hold the full hidden width on every rank; the layout is
upstream's ``InklingConvState(replicate_hidden=True)``, in the model dtype.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backend import AttentionBackend

    from .configs import Inkling2ModelConfig

__all__ = [
    "ATTN",
    "CONV_STATE",
    "STREAM_ATTN",
    "STREAM_K",
    "STREAM_MLP",
    "STREAM_V",
    "Inkling2AttentionCache",
    "Inkling2CacheOwners",
    "Inkling2ConvState",
    "build_cache_owners",
    "cache_index",
    "layer_passes",
    "owner_name",
]

ATTN = "attn"
CONV_STATE = "conv_state"

# Index into Inkling2ConvState.stream_ranges.
STREAM_K, STREAM_V, STREAM_ATTN, STREAM_MLP = range(4)


def layer_passes(config: Inkling2ModelConfig) -> list[tuple[int, int]]:
    """(layer, pass) of every layer visit, in cache-index order: pass 0 of
    every layer, then each further pass of the loop core."""
    if config.loop_share_cache:
        raise ValueError("vllm_sm70: Inkling 2 loop_share_cache is not supported.")
    visits = [(layer, 0) for layer in range(config.num_hidden_layers)]
    for pass_idx in range(1, config.num_loop_passes):
        visits += [(layer, pass_idx) for layer in config.loop_core_layers]
    return visits


def cache_index(config: Inkling2ModelConfig, layer: int, pass_idx: int) -> int:
    """Position of a visit's cache in the reference's cache list: the layer
    for pass 0, then one run of core layers per further pass (HF
    modeling_inkling2_loop.py ``cache_offset``)."""
    core = config.loop_core_layers
    if not 0 <= layer < config.num_hidden_layers:
        raise ValueError(f"layer {layer} is out of range")
    if pass_idx == 0:
        return layer
    if not 0 < pass_idx < config.num_loop_passes or layer not in core:
        raise ValueError(f"layer {layer} has no pass {pass_idx}")
    return config.num_hidden_layers + (pass_idx - 1) * len(core) + layer - core.start


def owner_name(prefix: str, layer: int, kind: str, pass_idx: int = 0) -> str:
    """Forward-context name of a visit's owner; ``kind`` is ATTN or CONV_STATE."""
    return f"{prefix}.{layer}.{kind}" + (f".pass{pass_idx}" if pass_idx else "")


def _resolve(vllm_config: Any, tp_size: int | None) -> tuple[Any, int]:
    if vllm_config is None:
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
    if tp_size is None:
        from vllm.distributed import get_tensor_model_parallel_world_size

        tp_size = get_tensor_model_parallel_world_size()
    return vllm_config, tp_size


def _local_heads(num_kv_heads: int, tp_size: int) -> int:
    if tp_size < 1 or num_kv_heads % tp_size:
        raise ValueError(
            f"vllm_sm70: Inkling 2 caches need a TP size that divides the "
            f"{num_kv_heads} KV heads, got {tp_size}."
        )
    return num_kv_heads // tp_size


class _CacheOwner(nn.Module, AttentionLayerBase):
    """Common part: identity of the visit and forward-context registration."""

    def __init__(
        self, *, prefix: str, layer_idx: int, pass_idx: int, cache_index: int
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.layer_idx = layer_idx
        self.pass_idx = pass_idx
        self.cache_index = cache_index
        # Replaced by bind_kv_cache.
        self.kv_cache = torch.tensor([])

    def _register(self, vllm_config: Any) -> None:
        context = vllm_config.compilation_config.static_forward_context
        if self.prefix in context:
            raise ValueError(f"Duplicate layer name: {self.prefix}")
        context[self.prefix] = self

    def forward(self) -> None: ...


class Inkling2AttentionCache(_CacheOwner):
    """K/V cache of one layer visit: sliding-window for a local layer, full
    attention for a global one. ``num_kv_heads`` is the model's (all ranks)."""

    def __init__(
        self,
        *,
        prefix: str,
        num_kv_heads: int,
        head_dim: int,
        sliding_window: int | None,
        layer_idx: int,
        pass_idx: int = 0,
        cache_index: int | None = None,
        vllm_config: VllmConfig | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            layer_idx=layer_idx,
            pass_idx=pass_idx,
            cache_index=layer_idx if cache_index is None else cache_index,
        )
        from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

        vllm_config, tp_size = _resolve(vllm_config, tp_size)
        self.num_kv_heads = _local_heads(num_kv_heads, tp_size)
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        self.dtype = kv_cache_dtype_str_to_dtype(
            vllm_config.cache_config.cache_dtype, vllm_config.model_config
        )
        self._register(vllm_config)

    @property
    def is_local(self) -> bool:
        return self.sliding_window is not None

    def get_attn_backend(self) -> type[AttentionBackend]:
        from .backend import Inkling2AttentionBackend

        return Inkling2AttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        common = {
            "block_size": vllm_config.cache_config.block_size,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_dim,
            "dtype": self.dtype,
        }
        if self.sliding_window is None:
            return FullAttentionSpec(**common)
        return SlidingWindowSpec(**common, sliding_window=self.sliding_window)


class Inkling2ConvState(_CacheOwner):
    """Short-conv state of one layer visit: the last ``kernel_size`` tokens of
    four streams, packed per local KV head.

    ``stream_ranges[s]`` is the (offset, width) of stream ``s`` inside a
    head's ``head_size`` values; a stream of width W per token is stored as
    ``num_kv_heads x (W / num_kv_heads)``. ``num_kv_heads`` passed in is the
    model's (all ranks); the attribute is the rank's.
    """

    def __init__(
        self,
        *,
        prefix: str,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        kernel_size: int,
        layer_idx: int,
        pass_idx: int = 0,
        cache_index: int | None = None,
        vllm_config: VllmConfig | None = None,
        tp_size: int | None = None,
    ) -> None:
        super().__init__(
            prefix=prefix,
            layer_idx=layer_idx,
            pass_idx=pass_idx,
            cache_index=layer_idx if cache_index is None else cache_index,
        )
        vllm_config, tp_size = _resolve(vllm_config, tp_size)
        self.num_kv_heads = _local_heads(num_kv_heads, tp_size)
        if hidden_size % self.num_kv_heads:
            raise ValueError(
                f"vllm_sm70: hidden size {hidden_size} does not split over "
                f"{self.num_kv_heads} local KV heads."
            )
        hidden_per_head = hidden_size // self.num_kv_heads
        raw_head_size = 2 * head_dim + 2 * hidden_per_head
        # A power of two, so the conv page divides or is divided by the
        # attention page and the planner scales block sizes instead of padding.
        self.head_size = 1 << (raw_head_size - 1).bit_length()
        self.sliding_window = kernel_size
        self.block_size = kernel_size
        self.stream_ranges: tuple[tuple[int, int], ...] = (
            (0, head_dim),
            (head_dim, head_dim),
            (2 * head_dim, hidden_per_head),
            (2 * head_dim + hidden_per_head, hidden_per_head),
        )
        self.dtype = vllm_config.model_config.dtype
        self._register(vllm_config)

    @property
    def cache_block_size(self) -> int:
        """Tokens per block of the bound cache. The runner binds a logical
        (blocks, heads, tokens, width) view in every KV layout
        (UP v1/kv_cache_interface.py ``create_kv_cache_views``). The planner
        may raise it above ``block_size``; index the cache with this value."""
        if self.kv_cache.numel() > 0:
            return self.kv_cache.shape[2]
        return self.block_size

    def get_attn_backend(self) -> type[AttentionBackend]:
        from .backend import Inkling2SconvBackend

        return Inkling2SconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=0,  # the four streams fill head_size
            dtype=self.dtype,
            sliding_window=self.sliding_window,
        )


class Inkling2CacheOwners:
    """All cache owners of a model, keyed by (layer, pass)."""

    def __init__(
        self,
        attn: dict[tuple[int, int], Inkling2AttentionCache],
        conv: dict[tuple[int, int], Inkling2ConvState],
    ) -> None:
        self.attn = attn
        self.conv = conv

    def passes(self, layer: int) -> int:
        """Number of passes that visit ``layer``."""
        return sum(1 for visit in self.attn if visit[0] == layer)

    def named_owners(self) -> dict[str, _CacheOwner]:
        """Forward-context name -> owner: attention owners, then conv."""
        owners: list[_CacheOwner] = [*self.attn.values(), *self.conv.values()]
        return {owner.prefix: owner for owner in owners}

    def kv_cache_specs(self, vllm_config: VllmConfig) -> dict[str, KVCacheSpec]:
        return {
            name: owner.get_kv_cache_spec(vllm_config)
            for name, owner in self.named_owners().items()
        }


def build_cache_owners(
    config: Inkling2ModelConfig,
    *,
    prefix: str = "model.layers",
    vllm_config: VllmConfig | None = None,
    tp_size: int | None = None,
) -> Inkling2CacheOwners:
    """Build and register the owners of every layer visit of ``config`` (the
    text config). ``prefix`` is the name of the decoder layer list and holds
    no integer. Defaults: the current vLLM config and the TP world size."""
    vllm_config, tp_size = _resolve(vllm_config, tp_size)
    local = set(config.local_layer_ids)
    attn: dict[tuple[int, int], Inkling2AttentionCache] = {}
    conv: dict[tuple[int, int], Inkling2ConvState] = {}
    for layer, pass_idx in layer_passes(config):
        is_local = layer in local
        num_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        head_dim = config.swa_head_dim if is_local else config.head_dim
        visit: dict[str, Any] = {
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "layer_idx": layer,
            "pass_idx": pass_idx,
            "cache_index": cache_index(config, layer, pass_idx),
            "vllm_config": vllm_config,
            "tp_size": tp_size,
        }
        attn[layer, pass_idx] = Inkling2AttentionCache(
            prefix=owner_name(prefix, layer, ATTN, pass_idx),
            sliding_window=config.sliding_window_size if is_local else None,
            **visit,
        )
        conv[layer, pass_idx] = Inkling2ConvState(
            prefix=owner_name(prefix, layer, CONV_STATE, pass_idx),
            hidden_size=config.hidden_size,
            kernel_size=config.sconv_kernel_size,
            **visit,
        )
    return Inkling2CacheOwners(attn, conv)
