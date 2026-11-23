# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 on SM70: the packed routed-expert tensors of one TP rank.

Checkpoint layout per layer, under ``model.llm.layers.N.mlp.experts.``
(E experts, H hidden, I intermediate):

- ``w13_weight`` [E, 2I, H] values: row ``2i`` is gate ``i``, row ``2i + 1``
  is up ``i``. ``w2_weight`` [E, H, I] values.
- NVFP4: uint8, two E2M1 codes per byte along the last axis (even element in
  the low nibble); ``_scale`` e4m3, one per 16 values along the last axis;
  ``_amax`` [E] and ``_outer_scale_denominator`` [1] fp32.
  value = E2M1[code] * scale * (amax[e] / denominator).
- MXFP8: e4m3 values; ``_scale`` uint8, one per 32 values along the last
  axis. value = weight * 2 ** (scale - 127).

A rank keeps intermediate units ``[I/tp * rank, I/tp * (rank + 1))``: those
gate/up row pairs of ``w13`` and those columns of ``w2``. Everything returned
is gate-first: ``w13`` rows are the local gates, then the local ups.

``Inkling2ExpertLoader`` is the ``expert_loader`` callback of weights.py; it
hands each complete layer to a sink as an ``ExpertLayer``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "E2M1_VALUES",
    "EXPERTS_PER_CHUNK",
    "MXFP8",
    "MXFP8_BLOCK",
    "MXFP8_MAX_SCALE",
    "NVFP4",
    "NVFP4_BLOCK",
    "ExpertLayer",
    "Inkling2ExpertLoader",
    "Mxfp8Codes",
    "Nvfp4Codes",
    "PackedExpertWeight",
    "check_cast_range",
    "detect_format",
    "load_expert_layers",
]

NVFP4 = "nvfp4"
MXFP8 = "mxfp8"
NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
EXPERTS_PER_CHUNK = 8
# Largest MXFP8 exponent byte whose 2 ** (s - 119) is a finite fp32 number
# (the e4m3 decode of ops/grouped_gemm.py).
MXFP8_MAX_SCALE = 246
# transformers core_model_loading.py Inkling2LoopDequantizeExperts.E2M1_VALUES.
E2M1_VALUES: tuple[float, ...] = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)  # fmt: skip

_BLOCK = {NVFP4: NVFP4_BLOCK, MXFP8: MXFP8_BLOCK}
_SUFFIXES = ("", "_scale", "_amax", "_outer_scale_denominator")
_REQUIRED = {NVFP4: frozenset(_SUFFIXES), MXFP8: frozenset(_SUFFIXES[:2])}
_NAME_RE = re.compile(
    r"(model\.llm\.layers\.(\d+)\.mlp\.experts\.w(?:13|2)_weight)"
    r"(|_scale|_amax|_outer_scale_denominator)"
)

RangeCheck = Callable[[str, torch.Tensor, torch.Tensor], None]


def check_cast_range(name: str, values: torch.Tensor, cast: torch.Tensor) -> None:
    """Raise ValueError naming ``name`` when fp32 ``values`` don't survive the
    cast to ``cast``'s dtype: a non-finite value, a magnitude above the
    dtype's largest finite value, or a nonzero value that became zero."""
    if not values.numel():
        return
    limit = torch.finfo(cast.dtype).max
    amax = values.abs().amax()
    if not bool(amax <= limit):
        raise ValueError(
            f"{name}: max |x| = {float(amax)} does not fit in {cast.dtype} "
            f"(max {limit})"
        )
    flushed = int(((cast == 0) & (values != 0)).sum())
    if flushed:
        raise ValueError(
            f"{name}: {flushed} nonzero values round to zero in {cast.dtype}"
        )


def detect_format(
    name: str, weight: torch.Tensor, scale: torch.Tensor, has_amax: bool
) -> str:
    """ "nvfp4" or "mxfp8" from the dtypes of a weight and its scale and the
    presence of its ``_amax`` tensor; raises when the three disagree."""
    if weight.dtype == torch.uint8 and scale.dtype == torch.float8_e4m3fn and has_amax:
        return NVFP4
    if (
        weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.uint8
        and not has_amax
    ):
        return MXFP8
    raise ValueError(
        f"{name}: weight {weight.dtype}, scale {scale.dtype}, "
        f"{'with' if has_amax else 'without'} amax is neither NVFP4 nor MXFP8"
    )


@dataclass(frozen=True)
class Nvfp4Codes:
    """One NVFP4 weight of one rank, unpacked. value[e, n, k] =
    ``E2M1_VALUES[codes[e, n, k]] * block_scale[e, n, k // 16] *
    global_scale[e]``, the products taken in fp32 in that order.

    ``codes`` uint8 [E, N, K], one code per byte (bit 3 is the sign);
    ``block_scale`` e4m3 [E, N, K/16]; ``global_scale`` fp32 [E] =
    amax / denominator. N is the output axis; ``w13`` rows are gate-first.
    """

    name: str
    codes: torch.Tensor
    block_scale: torch.Tensor
    global_scale: torch.Tensor


@dataclass(frozen=True)
class Mxfp8Codes:
    """One MXFP8 weight of one rank, still 8-bit. value[e, n, k] =
    ``e4m3(codes[e, n, k]) * 2 ** (scale[e, n, (k + offset) // 32] - 127)``.

    ``codes`` uint8 [E, N, K] (the e4m3 bytes); ``scale`` uint8 [E, N, B],
    the blocks that cover this rank's columns; ``offset`` is the first
    column's place in its block. N is the output axis; ``w13`` rows are
    gate-first.
    """

    name: str
    codes: torch.Tensor
    scale: torch.Tensor
    offset: int


@dataclass(frozen=True)
class PackedExpertWeight:
    """One of ``w13`` / ``w2`` for one rank, still packed.

    ``weight`` holds this rank's rows and columns in checkpoint order
    (``w13`` rows interleaved). ``scale`` holds this rank's rows and every
    column block; ``columns`` is the value-column range of ``weight`` within
    the checkpoint tensor, so column ``c`` uses scale block ``c // block``.
    """

    name: str
    format: str
    weight: torch.Tensor
    scale: torch.Tensor
    columns: tuple[int, int]
    amax: torch.Tensor | None = None
    denominator: torch.Tensor | None = None
    interleaved: bool = False

    @property
    def block(self) -> int:
        return _BLOCK[self.format]

    @property
    def shape(self) -> tuple[int, int, int]:
        """[E, rows, value columns] of the dequantized weight."""
        experts, rows = self.weight.shape[:2]
        return experts, rows, self.columns[1] - self.columns[0]

    def global_scale(self) -> torch.Tensor | None:
        """fp32 [E] ``amax / denominator`` (NVFP4), else None."""
        if self.amax is None or self.denominator is None:
            return None
        return self.amax.float() / self.denominator.float()

    def _gate_first(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.interleaved:
            return tensor
        return torch.cat((tensor[:, 0::2], tensor[:, 1::2]), dim=1)

    def _chunk_values(self, experts: slice) -> torch.Tensor:
        """fp32 values of ``experts`` in checkpoint row order."""
        start, stop = self.columns
        blocks = torch.arange(start, stop, device=self.weight.device) // self.block
        scale = self.scale[experts].float()
        if self.format == MXFP8:
            values = self.weight[experts].float()
            return values * torch.exp2(scale - 127).index_select(-1, blocks)
        e2m1 = torch.tensor(E2M1_VALUES, device=self.weight.device)
        byte = torch.arange(256, device=self.weight.device)
        byte_to_values = torch.stack((e2m1[byte & 0xF], e2m1[byte >> 4]), dim=-1)
        values = byte_to_values[self.weight[experts].int()].flatten(-2)
        values = values * scale.index_select(-1, blocks)
        factor = self.global_scale()
        assert factor is not None
        return values * factor[experts, None, None]

    @torch.no_grad()
    def dequantize(
        self,
        dtype: torch.dtype = torch.float16,
        *,
        range_check: RangeCheck | None = check_cast_range,
    ) -> torch.Tensor:
        """[E, rows, columns] in ``dtype``, ``w13`` gate-first. Values are
        computed in fp32, ``EXPERTS_PER_CHUNK`` experts at a time, then cast;
        ``range_check(name, fp32 values, cast values)`` sees every chunk."""
        num_experts = self.weight.shape[0]
        output = torch.empty(self.shape, dtype=dtype, device=self.weight.device)
        for start in range(0, num_experts, EXPERTS_PER_CHUNK):
            experts = slice(start, min(start + EXPERTS_PER_CHUNK, num_experts))
            values = self._chunk_values(experts)
            cast = values.to(dtype)
            if range_check is not None:
                range_check(
                    f"{self.name} (experts {experts.start}-{experts.stop - 1})",
                    values,
                    cast,
                )
            output[experts] = self._gate_first(cast)
        return output

    @torch.no_grad()
    def mxfp8_codes(
        self, dtype: torch.dtype, *, range_check: RangeCheck | None = check_cast_range
    ) -> Mxfp8Codes | None:
        """The codes and scales of an MXFP8 weight, or None when a value is
        not exact in ``dtype`` or a scale exceeds ``MXFP8_MAX_SCALE``: then
        only ``dequantize`` gives the values. ``range_check`` sees the same
        chunks as in ``dequantize``."""
        if self.format != MXFP8:
            raise ValueError(f"{self.name}: {self.format} weight has no MXFP8 codes")
        num_experts = self.weight.shape[0]
        exact = True
        for start in range(0, num_experts, EXPERTS_PER_CHUNK):
            experts = slice(start, min(start + EXPERTS_PER_CHUNK, num_experts))
            values = self._chunk_values(experts)
            cast = values.to(dtype)
            if range_check is not None:
                range_check(
                    f"{self.name} (experts {experts.start}-{experts.stop - 1})",
                    values,
                    cast,
                )
            exact = exact and torch.equal(cast.float(), values)
        if not exact:
            return None
        start, stop = self.columns
        scale = self.scale[..., start // MXFP8_BLOCK : -(-stop // MXFP8_BLOCK)]
        if scale.numel() and int(scale.max()) > MXFP8_MAX_SCALE:
            return None
        return Mxfp8Codes(
            self.name,
            self._gate_first(self.weight.view(torch.uint8)).contiguous(),
            self._gate_first(scale).contiguous(),
            start % MXFP8_BLOCK,
        )

    @torch.no_grad()
    def nvfp4_codes(self) -> Nvfp4Codes:
        """The unpacked codes and scales of an NVFP4 weight. The rank's
        column range must start and end on a block of 16."""
        if self.format != NVFP4:
            raise ValueError(f"{self.name}: {self.format} weight has no NVFP4 codes")
        start, stop = self.columns
        if start % NVFP4_BLOCK or stop % NVFP4_BLOCK:
            raise ValueError(
                f"{self.name}: columns {start}..{stop} don't align to NVFP4 "
                f"blocks of {NVFP4_BLOCK}"
            )
        packed = self.weight
        codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
        scale = self.scale[..., start // NVFP4_BLOCK : stop // NVFP4_BLOCK]
        factor = self.global_scale()
        assert factor is not None
        return Nvfp4Codes(
            self.name,
            self._gate_first(codes).contiguous(),
            self._gate_first(scale).contiguous(),
            factor,
        )


@dataclass(frozen=True)
class ExpertLayer:
    """The routed experts of one layer for one rank."""

    layer: int
    format: str
    w13: PackedExpertWeight
    w2: PackedExpertWeight

    def dequantize(
        self,
        dtype: torch.dtype = torch.float16,
        *,
        range_check: RangeCheck | None = check_cast_range,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(``w13`` [E, 2 * I/tp, H] with the gate rows first,
        ``w2`` [E, H, I/tp]) in ``dtype``."""
        return (
            self.w13.dequantize(dtype, range_check=range_check),
            self.w2.dequantize(dtype, range_check=range_check),
        )

    def nvfp4_codes(self) -> tuple[Nvfp4Codes, Nvfp4Codes]:
        """(``w13``, ``w2``) unpacked codes of an NVFP4 layer."""
        return self.w13.nvfp4_codes(), self.w2.nvfp4_codes()

    def mxfp8_codes(
        self, dtype: torch.dtype, *, range_check: RangeCheck | None = check_cast_range
    ) -> tuple[Mxfp8Codes, Mxfp8Codes] | None:
        """(``w13``, ``w2``) codes of an MXFP8 layer, or None as
        ``PackedExpertWeight.mxfp8_codes``; ``range_check`` sees both."""
        w13 = self.w13.mxfp8_codes(dtype, range_check=range_check)
        w2 = self.w2.mxfp8_codes(dtype, range_check=range_check)
        return None if w13 is None or w2 is None else (w13, w2)


def _text_config(config: Any) -> Any:
    get = getattr(config, "get_text_config", None)
    return config if get is None else get()


class Inkling2ExpertLoader:
    """``expert_loader`` of weights.py: collects the expert tensors by
    checkpoint name and calls ``sink(ExpertLayer)`` when a layer is complete.

    ``formats`` is quant_config.py's ``load_expert_formats`` result; when
    given, each layer's detected format must equal it. Without a sink the
    layers are kept in ``layers``. Call ``finish()`` after the last tensor.
    """

    def __init__(
        self,
        config: Any,
        *,
        tp_size: int = 1,
        tp_rank: int = 0,
        formats: Mapping[int, str] | None = None,
        sink: Callable[[ExpertLayer], None] | None = None,
    ) -> None:
        text = _text_config(config)
        if not 0 <= tp_rank < tp_size:
            raise ValueError(f"Inkling 2: TP rank {tp_rank} outside 0..{tp_size - 1}")
        self.num_layers = int(text.num_hidden_layers)
        self.num_experts = int(text.n_routed_experts)
        self.hidden = int(text.hidden_size)
        self.intermediate = int(text.intermediate_size)
        if self.intermediate % tp_size:
            raise ValueError(
                f"Inkling 2: expert intermediate size {self.intermediate} "
                f"doesn't split over {tp_size} ranks"
            )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.formats = None if formats is None else dict(formats)
        self.layers: dict[int, ExpertLayer] = {}
        self.loaded: set[str] = set()
        self._sink = sink if sink is not None else self._keep
        self._pending: dict[int, dict[str, torch.Tensor]] = {}
        self._done: set[int] = set()

    def _keep(self, layer: ExpertLayer) -> None:
        self.layers[layer.layer] = layer

    def __call__(self, name: str, tensor: torch.Tensor) -> None:
        match = _NAME_RE.fullmatch(name)
        if match is None or int(match.group(2)) >= self.num_layers:
            raise ValueError(f"Inkling 2: unexpected expert tensor {name!r}")
        layer = int(match.group(2))
        if layer in self._done:
            raise ValueError(
                f"Inkling 2: expert tensor {name!r} after layer {layer} was complete"
            )
        pending = self._pending.setdefault(layer, {})
        if name in pending:
            raise ValueError(f"Inkling 2: duplicate expert tensor {name!r}")
        pending[name] = tensor
        built = self._build(layer, pending)
        if built is not None:
            del self._pending[layer]
            self._done.add(layer)
            self.loaded.update(pending)
            self._sink(built)

    def finish(self) -> set[str]:
        """Raise unless every layer was delivered; returns the checkpoint
        names consumed."""
        problems = [
            f"layer {layer}: incomplete, has {sorted(n.rsplit('.', 1)[1] for n in p)}"
            for layer, p in sorted(self._pending.items())
        ]
        missing = sorted(set(range(self.num_layers)) - self._done - set(self._pending))
        if missing:
            problems.append(f"no expert tensors for layers {missing}")
        if problems:
            raise ValueError("Inkling 2 experts: " + "; ".join(problems))
        return set(self.loaded)

    # Assembly of one layer.

    def _build(
        self, layer: int, pending: Mapping[str, torch.Tensor]
    ) -> ExpertLayer | None:
        prefix = f"model.llm.layers.{layer}.mlp.experts."
        names = [prefix + "w13_weight", prefix + "w2_weight"]
        if any(n not in pending for n in names):
            return None
        # A layer's tensors arrive in any order: wait for the set its weight
        # dtype calls for before judging the format.
        for n in names:
            expected = NVFP4 if pending[n].dtype == torch.uint8 else MXFP8
            if any(n + suffix not in pending for suffix in _REQUIRED[expected]):
                return None
        formats = [
            detect_format(n, pending[n], pending[n + "_scale"], n + "_amax" in pending)
            for n in names
        ]
        if formats[0] != formats[1]:
            raise ValueError(
                f"Inkling 2: layer {layer} experts are {formats[0]} (w13) and "
                f"{formats[1]} (w2)"
            )
        fmt = formats[0]
        extra = sorted(
            n + suffix
            for n in names
            for suffix in _SUFFIXES
            if suffix not in _REQUIRED[fmt] and n + suffix in pending
        )
        if extra:
            raise ValueError(f"Inkling 2: {fmt} layer {layer} has {extra}")
        if self.formats is not None and self.formats.get(layer) != fmt:
            raise ValueError(
                f"Inkling 2: layer {layer} experts are {fmt} in the checkpoint "
                f"and {self.formats.get(layer)!r} in hf_quant_config.json"
            )
        local = self.intermediate // self.tp_size
        first = local * self.tp_rank
        w13 = self._packed(
            names[0],
            fmt,
            pending,
            rows=2 * self.intermediate,
            columns=self.hidden,
            row_range=(2 * first, 2 * (first + local)),
            column_range=(0, self.hidden),
            interleaved=True,
        )
        w2 = self._packed(
            names[1],
            fmt,
            pending,
            rows=self.hidden,
            columns=self.intermediate,
            row_range=(0, self.hidden),
            column_range=(first, first + local),
            interleaved=False,
        )
        return ExpertLayer(layer, fmt, w13, w2)

    def _packed(
        self,
        name: str,
        fmt: str,
        pending: Mapping[str, torch.Tensor],
        *,
        rows: int,
        columns: int,
        row_range: tuple[int, int],
        column_range: tuple[int, int],
        interleaved: bool,
    ) -> PackedExpertWeight:
        block = _BLOCK[fmt]
        per_element = 2 if fmt == NVFP4 else 1
        if columns % block:
            raise ValueError(
                f"{name}: {columns} columns are not a multiple of the {fmt} "
                f"block of {block}"
            )
        want = {
            "": (self.num_experts, rows, columns // per_element),
            "_scale": (self.num_experts, rows, columns // block),
            "_amax": (self.num_experts,),
            "_outer_scale_denominator": (1,),
        }
        for suffix in _REQUIRED[fmt]:
            got = tuple(pending[name + suffix].shape)
            if got != want[suffix]:
                raise ValueError(
                    f"{name}{suffix}: checkpoint shape {got} != {want[suffix]}"
                )
        for suffix in _REQUIRED[fmt] - {"", "_scale"}:
            if pending[name + suffix].dtype != torch.float32:
                raise ValueError(
                    f"{name}{suffix}: dtype {pending[name + suffix].dtype} is "
                    "not float32"
                )
        start, stop = column_range
        if start % per_element or stop % per_element:
            raise ValueError(
                f"{name}: columns {start}..{stop} split a packed {fmt} byte"
            )
        rows_ = slice(*row_range)
        weight = pending[name][:, rows_, start // per_element : stop // per_element]
        return PackedExpertWeight(
            name=name,
            format=fmt,
            weight=weight,
            scale=pending[name + "_scale"][:, rows_],
            columns=(start, stop),
            amax=pending.get(name + "_amax"),
            denominator=pending.get(name + "_outer_scale_denominator"),
            interleaved=interleaved,
        )


def load_expert_layers(
    tensors: Mapping[str, torch.Tensor],
    config: Any,
    *,
    tp_size: int = 1,
    tp_rank: int = 0,
    formats: Mapping[int, str] | None = None,
) -> dict[int, ExpertLayer]:
    """Every layer's ``ExpertLayer`` from a name -> tensor mapping; names that
    aren't expert tensors are ignored."""
    loader = Inkling2ExpertLoader(
        config, tp_size=tp_size, tp_rank=tp_rank, formats=formats
    )
    for name, tensor in tensors.items():
        if _NAME_RE.fullmatch(name):
            loader(name, tensor)
    loader.finish()
    return loader.layers
