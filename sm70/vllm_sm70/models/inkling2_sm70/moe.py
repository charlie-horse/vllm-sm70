# SPDX-License-Identifier: Apache-2.0
"""Inkling 2 routed experts in torch, with every expert held in the model dtype.

One module per layer, mounted as ``mlp``: the fp32 router and this rank's
dequantized experts. ``forward`` is one eager-break op that writes this rank's
partial expert sum into a caller buffer. Rounding as the reference: routing,
SwiGLU caps and the weighted sum in fp32, GEMMs in the model dtype.
Switches, read on every eager call (a replayed graph keeps its route):
VLLM_SM70_INKLING_NVFP4_TM=0 keeps NVFP4 layers off TurboMind (moe_tm.py);
VLLM_SM70_INKLING_GROUPED_MOE=0 sends model-dtype layers from
ops/grouped_gemm.py back to the torch sum; VLLM_SM70_INKLING_FUSED_ROUTING=0
keeps the torch router and sort instead of ops/routing.py for batches of at
most ``routing.MAX_SLOTS`` slots; larger batches that are not decode-only
take ``routing.route_blocks`` on the route of fused_small.py (``on_mixed``).
VLLM_SM70_INKLING_MXFP8_EXPERTS=0, read at load, dequantizes MXFP8 layers
instead of keeping their e4m3 codes.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.compilation.breakable_cudagraph import eager_break_during_capture

from vllm_sm70 import envs

from . import act_quant, moe_tm
from .expert_weights import MXFP8, NVFP4, ExpertLayer, RangeCheck, check_cast_range
from .fused_small import on_fused, on_mixed
from .ops import _compat, grouped_gemm, routing, torch_ref
from .visit_fusion import on_decode

SWITCH = "VLLM_SM70_INKLING_GROUPED_MOE"
_enabled = envs.environment_variables()[SWITCH]
FUSED_SWITCH = "VLLM_SM70_INKLING_FUSED_ROUTING"
_fused_enabled = envs.environment_variables()[FUSED_SWITCH]
MXFP8_SWITCH = "VLLM_SM70_INKLING_MXFP8_EXPERTS"
mxfp8_enabled = envs.environment_variables()[MXFP8_SWITCH]


def on_grouped(x: torch.Tensor) -> bool:
    """Whether the experts held in the model dtype run on `x` through
    ops/grouped_gemm.py with its Triton kernels."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    return _enabled()


def on_fused_routing(x: torch.Tensor, top_k: int) -> bool:
    """Whether x [M, H] is routed by ops/routing.py's kernels."""
    if not (_compat.INTERPRET or (x.is_cuda and _compat.AVAILABLE)):
        return False
    if not 0 < x.shape[0] * top_k <= routing.MAX_SLOTS:
        return False
    return _fused_enabled()


def _text_config(config: Any) -> Any:
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else config


class Inkling2Sm70Gate(nn.Module):
    """The top-k router: fp32 ``weight`` [E, H] and selection ``bias`` [E]."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        text = _text_config(config)
        experts, hidden = int(text.n_routed_experts), int(text.hidden_size)
        self.top_k = int(text.num_experts_per_tok)
        self.route_scale = float(text.route_scale)
        self.weight = nn.Parameter(
            torch.empty(experts, hidden, dtype=torch.float32), requires_grad=False
        )
        self.bias = nn.Parameter(
            torch.zeros(experts, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x [M, H] -> fp32 (logits [M, E], weights [M, k], indices [M, k])."""
        return torch_ref.router(x, self.weight, self.bias, self.top_k, self.route_scale)


class Inkling2Sm70Experts(nn.Module):
    """Router and experts of one layer on one TP rank.

    ``w13`` [E, 2 * I / tp, H] holds the gate rows then the up rows and ``w2``
    [E, H, I / tp]; both are filled by ``load_expert_layer``. When the shape
    is admitted to the TurboMind route (``tm_rejection`` is None) they stay
    empty until a layer that needs them is loaded, and an NVFP4 layer is held
    in ``tm_route`` instead, and a held MXFP8 layer in ``mx13`` / ``mx2``
    (``expert_weights()`` gives the pair the routes read).
    """

    def __init__(
        self,
        config: Any,
        *,
        tp_size: int = 1,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        text = _text_config(config)
        self.num_experts = int(text.n_routed_experts)
        self.hidden_size = int(text.hidden_size)
        intermediate = int(text.intermediate_size)
        if intermediate % tp_size:
            raise ValueError(
                f"Inkling 2: expert intermediate size {intermediate} "
                f"doesn't split over {tp_size} ranks"
            )
        self.local_intermediate = intermediate // tp_size
        self.gate_cap = float(text.mlp_gate_softcap)
        self.up_cap = float(text.mlp_up_softcap)
        if self.gate_cap <= 0 or self.up_cap <= 0:
            raise ValueError(
                "Inkling 2: mlp_gate_softcap and mlp_up_softcap must be positive, "
                f"got {self.gate_cap} and {self.up_cap}"
            )
        self.gate = Inkling2Sm70Gate(text)
        self.top_k = int(text.num_experts_per_tok)
        self.tp_size = tp_size
        self.dtype = dtype
        self.contract = (self.hidden_size, intermediate, self.num_experts, self.top_k)
        self.tm_rejection = moe_tm.rejection(self.contract, tp_size, dtype)
        self.tm_route: moe_tm.Nvfp4TmExperts | None = None
        # Expert activation fake-quant (act_quant.install), None when off.
        self.act_quant: act_quant.ActQuant | None = None
        e, h, i = self.num_experts, self.hidden_size, self.local_intermediate
        self.weight_shapes = {"w13": (e, 2 * i, h), "w2": (e, h, i)}
        for name, shape in self.weight_shapes.items():
            if self.tm_rejection is None:
                shape = (0, *shape[1:])
            self.register_buffer(
                name, torch.empty(shape, dtype=dtype), persistent=False
            )
        self.mx13: grouped_gemm.Mxfp8Weight | None = None
        self.mx2: grouped_gemm.Mxfp8Weight | None = None

    def _apply(self, fn, recurse=True):
        # The held MXFP8 codes and scales move with the module (.to, .cuda).
        super()._apply(fn, recurse)
        for name in ("mx13", "mx2"):
            held = getattr(self, name)
            if held is not None:
                setattr(
                    self,
                    name,
                    grouped_gemm.Mxfp8Weight(
                        fn(held.codes), fn(held.scale), held.offset
                    ),
                )
        return self

    def expert_weights(self) -> tuple[Any, Any]:
        """(w13, w2) as the routes read them: tensors in the model dtype, or
        the ``grouped_gemm.Mxfp8Weight`` pair of a held MXFP8 layer."""
        if self.mx13 is not None:
            return self.mx13, self.mx2
        return self.w13, self.w2

    def load_expert_layer(
        self, layer: ExpertLayer, range_check: RangeCheck | None = check_cast_range
    ) -> None:
        """Load `layer` (already sliced to this rank): TurboMind banks for an
        admitted NVFP4 layer, the codes of a held MXFP8 layer, else the
        dequantized w13 and w2. `range_check`
        sees every cast to fp16: the dequantized blocks (expert_weights.py)
        or the banks' scales (moe_tm.py)."""
        device = self.gate.weight.device
        if layer.format == NVFP4 and self.tm_rejection is None:
            self.tm_rejection = moe_tm.rejection(
                self.contract, self.tp_size, self.dtype, device
            )
            if self.tm_rejection is None:
                self.tm_rejection = self._load_tm(layer, device, range_check)
                if self.tm_rejection is None:
                    return
        if layer.format == NVFP4 and moe_tm.enabled():
            moe_tm.logger.warning_once(
                "Inkling 2 NVFP4 experts stay on the fp16 torch path: %s.",
                self.tm_rejection,
            )
        self.mx13 = self.mx2 = None
        if layer.format == MXFP8 and self.dtype == torch.float16 and mxfp8_enabled():
            if self._load_mxfp8(layer, device, range_check):
                return
            range_check = None  # mxfp8_codes saw every value
        w13, w2 = layer.dequantize(self.dtype, range_check=range_check)
        for name, got in (("w13", w13), ("w2", w2)):
            if got.shape != self.weight_shapes[name]:
                raise ValueError(
                    f"Inkling 2: layer {layer.layer} {name} has shape "
                    f"{tuple(got.shape)}, expected {self.weight_shapes[name]}"
                )
            buffer = getattr(self, name)
            if buffer.shape == got.shape:
                buffer.copy_(got)
            else:
                setattr(self, name, got.to(device))
        self.tm_route = None

    def _load_mxfp8(
        self,
        layer: ExpertLayer,
        device: torch.device,
        range_check: RangeCheck | None,
    ) -> bool:
        """Hold `layer` as codes and scales; False when it must be
        dequantized (a value not exact in fp16)."""
        codes = layer.mxfp8_codes(self.dtype, range_check=range_check)
        if codes is None:
            moe_tm.logger.warning_once(
                "Inkling 2 MXFP8 experts of layer %d hold values that are not "
                "exact in fp16; they are dequantized.",
                layer.layer,
            )
            return False
        held = []
        for name, got in zip(("w13", "w2"), codes):
            if got.codes.shape != self.weight_shapes[name]:
                raise ValueError(
                    f"Inkling 2: layer {layer.layer} {name} has shape "
                    f"{tuple(got.codes.shape)}, expected {self.weight_shapes[name]}"
                )
            held.append(
                grouped_gemm.Mxfp8Weight(
                    got.codes.to(device), got.scale.to(device), got.offset
                )
            )
            empty = torch.empty((0, *self.weight_shapes[name][1:]), dtype=self.dtype)
            setattr(self, name, empty.to(device))
        self.mx13, self.mx2 = held
        self.tm_route = None
        moe_tm.logger.info_once(
            "Inkling 2 MXFP8 experts stay 8-bit, decoded in the expert GEMMs "
            "(%s=0 dequantizes them at load).",
            MXFP8_SWITCH,
        )
        return True

    def _load_tm(
        self,
        layer: ExpertLayer,
        device: torch.device,
        range_check: RangeCheck | None,
    ) -> str | None:
        """Build the banks; the reason they were refused, or None."""
        w13, w2 = layer.nvfp4_codes()
        for name, codes in (("w13", w13), ("w2", w2)):
            expected = self.weight_shapes[name]
            if codes.codes.shape != expected:
                raise ValueError(
                    f"Inkling 2: layer {layer.layer} {name} has shape "
                    f"{tuple(codes.codes.shape)}, expected {expected}"
                )
        try:
            self.tm_route = moe_tm.build(
                w13, w2,
                device=device, gate_cap=self.gate_cap, up_cap=self.up_cap,
                top_k=self.top_k, range_check=range_check,
            )  # fmt: skip
        except moe_tm.Rejected as error:
            return str(error)
        for name, shape in self.weight_shapes.items():
            setattr(self, name, torch.empty((0, *shape[1:]), dtype=self.dtype))
        moe_tm.logger.info_once(
            "Inkling 2 NVFP4 experts %s run on the TurboMind dense stage "
            "(%s=0 turns it off).",
            self.contract,
            moe_tm.SWITCH,
        )
        return None

    def route(
        self,
        x: torch.Tensor,
        is_padding: torch.Tensor | None = None,
        decode_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 (logits [M, E], weights [M, k], indices [M, k]) of x [M, H].

        Rows where the bool `is_padding` [M] is set get the expert id E (no
        expert) and weight 0. A batch that is not `decode_only` routes its
        logits in one launch on the route of fused_small.py.
        """
        gate = self.gate
        if not decode_only and on_mixed(x):
            weight = gate.weight.float()
            logits = F.linear(x.float().reshape(-1, weight.shape[-1]), weight)
            pad = None if is_padding is None else is_padding.reshape(-1).bool()
            weights, indices = routing.route_blocks(
                logits, gate.bias, self.top_k, gate.route_scale, pad,
                pad_id=self.num_experts,
            )  # fmt: skip
            return logits, weights, indices
        logits, weights, indices = gate(x)
        if is_padding is not None:
            pad = is_padding.reshape(-1, 1).bool()
            indices = indices.masked_fill(pad, self.num_experts)
            weights = weights.masked_fill(pad, 0.0)
        return logits, weights, indices

    def expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
        rowwise: bool = False,
    ) -> None:
        """Write this rank's weighted expert sum of x [M, H] into `output`.

        top_k_index, top_k_weights [M, k] as `route` returns them; ids outside
        0..E-1 select no expert. `rowwise`: the torch path sums each token
        on its own, without a host sync; the other routes never sync. A
        `rowwise` (decode-only) batch on the TurboMind route runs its SwiGLU
        as one kernel on the route of fused_small.py.
        """
        x = act_quant.apply(self.act_quant, x)
        if self.tm_route is not None:
            self.tm_route.expert_sum(
                x, top_k_index, top_k_weights, output, self.act_quant,
                on_fused(x, rowwise),
            )  # fmt: skip
            return
        if on_grouped(x):
            grouped_gemm.grouped_expert_sum(
                x, top_k_index, top_k_weights, *self.expert_weights(),
                self.gate_cap, self.up_cap, output, act_hook=self.act_quant,
            )  # fmt: skip
            return
        self.torch_expert_sum(x, top_k_index, top_k_weights, output, rowwise)

    def torch_expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
        rowwise: bool = False,
    ) -> None:
        """`expert_sum` on the dequantized w13 and w2. One token, or every
        token with `rowwise`, runs without a host sync; a batch reads its
        per-expert counts on the host once."""
        if x.shape[0] == 1 or rowwise:
            for row in range(x.shape[0]):
                self._token_expert_sum(
                    x[row : row + 1],
                    top_k_index[row : row + 1],
                    top_k_weights[row : row + 1],
                    output[row : row + 1],
                )
        else:
            self._grouped_expert_sum(x, top_k_index, top_k_weights, output)

    def _grouped_expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Each expert's tokens in one GEMM pair, the experts in ascending
        order; a token's experts are summed in that order."""
        k = top_k_index.shape[-1]
        flat = top_k_index.reshape(-1)
        none = self.num_experts
        flat = flat.masked_fill((flat < 0) | (flat > none), none)
        order = torch.argsort(flat, stable=True)
        tokens = order // k
        weights = top_k_weights.reshape(-1)[order].float()
        # torch.bincount syncs on CUDA (min and max); scatter_add_ does not.
        counts = flat.new_zeros(none + 1).scatter_add_(0, flat, torch.ones_like(flat))
        total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        w13, w2 = self.expert_weights()
        start = 0
        for expert, count in enumerate(counts[:none].tolist()):
            if not count:
                continue
            rows = tokens[start : start + count]
            gate_up = F.linear(x[rows], grouped_gemm.expert_weights(w13, expert))
            gate, up = gate_up.chunk(2, dim=-1)
            act = torch_ref.softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)
            act = act_quant.apply(self.act_quant, act)
            out = F.linear(act, grouped_gemm.expert_weights(w2, expert)).float()
            total.index_add_(0, rows, out * weights[start : start + count, None])
            start += count
        output.copy_(total)

    def _token_expert_sum(
        self,
        x: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """`_grouped_expert_sum` of x [1, H] without a host sync, bit-equal to
        it: each slot's expert weights are gathered on the device, a slot
        with no expert is computed on expert E - 1 and zeroed, and the slots
        are summed in ascending expert order."""
        k = top_k_index.shape[-1]
        none = self.num_experts
        ids = top_k_index.reshape(-1)
        ids = ids.masked_fill((ids < 0) | (ids > none), none)
        ids, order = torch.sort(ids, stable=True)
        skip = (ids == none).unsqueeze(1)
        ids = ids.clamp(max=none - 1)
        w13, w2 = (grouped_gemm.expert_weights(w, ids) for w in self.expert_weights())
        row = x.contiguous()
        gate_up = torch.cat([F.linear(row, w13[slot]) for slot in range(k)])
        gate, up = gate_up.chunk(2, dim=-1)
        act = torch_ref.softcapped_swiglu(gate, up, self.gate_cap, self.up_cap)
        act = act_quant.apply(self.act_quant, act)
        out = torch.cat([F.linear(act[slot : slot + 1], w2[slot]) for slot in range(k)])
        weights = top_k_weights.reshape(-1)[order].float()
        out = (out.float() * weights[:, None]).masked_fill_(skip, 0.0)
        total = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for slot in range(k):
            total.add_(out[slot])
        output.copy_(total)

    @eager_break_during_capture
    def forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        is_padding: torch.Tensor | None = None,
        rowwise: bool = False,
    ) -> None:
        """x [M, H] in the model dtype -> `output` [M, H], this rank's partial.

        `output` must not alias `x`. Padding rows are written as zeros.
        `rowwise` as `expert_sum` (a decode-only batch).
        """
        if self._fused_routing(x):
            self.fused_forward(x, output, is_padding, rowwise)
            return
        _, weights, indices = self.route(x, is_padding, rowwise)
        self.expert_sum(x, indices, weights, output, rowwise)

    def _fused_routing(self, x: torch.Tensor) -> bool:
        if not on_fused_routing(x, self.top_k):
            return False
        return self.tm_route is not None or on_grouped(x)

    def fused_forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        is_padding: torch.Tensor | None = None,
        decode_only: bool = False,
    ) -> None:
        """`forward` through ops/routing.py, on the TurboMind or the grouped
        route; the caller checks ``on_fused_routing`` and the route.
        `decode_only` as `expert_sum`'s `rowwise`."""
        gate = self.gate
        if on_decode(x, decode_only):
            # The router GEMV runs inside the route launch (visit_fusion.py).
            logits = routing.RouterInput(x, gate.weight)
        else:
            logits = routing.router_logits(x, gate.weight)
        if is_padding is not None:
            is_padding = is_padding.reshape(-1)
        x = act_quant.apply(self.act_quant, x)
        if self.tm_route is not None:
            self.tm_route.routed_expert_sum(
                x, logits, gate.bias, self.top_k, gate.route_scale, is_padding,
                output, self.act_quant, on_fused(x, decode_only),
            )  # fmt: skip
            return
        routed = routing.route(
            logits, gate.bias, self.top_k, gate.route_scale, is_padding, pad_id=-1
        )
        grouped_gemm.routed_expert_sum(
            x, routed, *self.expert_weights(), self.gate_cap, self.up_cap, output,
            act_hook=self.act_quant,
        )  # fmt: skip
