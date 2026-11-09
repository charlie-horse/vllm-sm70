# SPDX-License-Identifier: Apache-2.0
"""M7 dflash2-fp16-range: BF16-range emulation for DFlash2 drafts on FP16.

What: the incoai/*-DFlash2 drafts (``DFlash2DraftModel``, model_type qwen3,
  ``dtype`` bfloat16) were trained with BF16 activations. We serve them with
  ``--dtype half`` on V100 (cc 7.0, no BF16 arithmetic). This module keeps
  FP16 tensor-core transport and preserves BF16 range at the trained rounding
  points:
  - the residual stream is carried in FP32 and rounded to BF16 (RNE);
  - sublayer outputs travel as value/256 (attention: scaled before o_proj;
    MLP: per-row power-of-two transport scale);
  - the gate/up GEMM input is pre-divided by 32.
  The plugin registers ``DFlash2SM70Qwen3ForCausalLM`` for the
  "DFlash2DraftModel" architecture. Its decoder-layer and model subclasses
  swap in DFlashSM70RMSNorm and DFlashSM70MLP after the upstream
  ``__init__``, before weights load.
Why: in plain FP16 the residual and SwiGLU products of these drafts can pass
  65504 and become inf. The fork notes that the draft "was trained with BF16
  activations, but Volta executes its dense GEMMs in FP16" (FORK cf9492a
  vllm/model_executor/models/dflash_sm70.py:3-7). The fork enabled this by
  default (its DFLASH2_BF16_EMULATION switch was effectively on).
Kill-switch: VLLM_SM70_DFLASH2_EMULATION=0 (envs "models" section). The
  plugin then neither registers the subclass nor applies it, and upstream's
  DFlash2 runs in plain FP16.

Gate: the emulation applies only when compute capability < 8.0, the draft's
  declared dtype is bfloat16 and the runtime dtype is float16 (port of FORK
  qwen3_dflash2.py:52-73 ``_use_sm70_bf16_emulation``).

Port map (FORK = charlie-horse/vLLM cf9492a, vllm/model_executor/models/):
  dflash_sm70.py 1-257, 330-419 -> this file (kernels, wrappers, modules).
  dflash_sm70.py 119-163, 258-329 (layered RMSNorm, perf-only) -> dropped.
  qwen3_dflash2.py 213-231, 465-482, 503-505 -> the layer/model subclasses.
  qwen3_dflash.py 312-314 (o_proj input scale) -> DFlashSM70Qwen3Attention.
  qwen3_dflash.py 675-683 (_normalize_context_states) -> _project_context_kv.

Upstream hooks this rests on (UP = vllm 4ba787e), with no upstream edit:
  - Arch -> class: ModelRegistry "DFlash2DraftModel" (models/registry.py:666),
    resolved for the draft by model_loader/utils.py:85 -> :310 via
    load_dflash_model (v1/worker/gpu/spec_decode/dflash/utils.py:45).
  - Class factories: DFlashQwen3Model.decoder_layer_cls (qwen3_dflash.py:388,
    used at :465) and DFlashQwen3ForCausalLM.model_cls (:737, used at :748).
  - Layer modules are plain attributes set in DFlashQwen3DecoderLayer.__init__
    (qwen3_dflash.py:352-362); DFlash2Qwen3DecoderLayer.__init__
    (qwen3_dflash2.py:230-264) only adds the convs, which are linear in their
    input, so the /256 transport commutes with them.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from vllm.logger import init_logger
from vllm_sm70 import envs

logger = init_logger("vllm.sm70.dflash2")

try:  # vllm.triton_utils gives placeholders when Triton is missing.
    from vllm.triton_utils import HAS_TRITON, tl, triton
except Exception:  # pragma: no cover - vllm not importable
    HAS_TRITON = False
    tl = triton = None  # type: ignore[assignment]

# ported from FORK models/dflash_sm70.py:19-20
DFLASH_SM70_GATE_UP_INPUT_SCALE = 32.0
DFLASH_SM70_WIDE_OUTPUT_SCALE = 256.0
# Largest power-of-two-friendly FP16 magnitude used for the row transport scale.
_FP16_TRANSPORT_MAX = 32752.0

KILL_SWITCH = "VLLM_SM70_DFLASH2_EMULATION"
DRAFT_ARCH = "DFlash2DraftModel"


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------
def _is_bf16_dtype(value: Any) -> bool:
    return value is torch.bfloat16 or str(value).lower() in {
        "bfloat16",
        "bf16",
        "torch.bfloat16",
    }


def draft_declared_bf16(config: Any) -> bool:
    """True if the draft HF config declares bfloat16 (``dtype`` or ``torch_dtype``)."""
    # ported from FORK qwen3_dflash2.py:53-58
    declared = getattr(config, "dtype", None)
    if declared is None:
        declared = getattr(config, "torch_dtype", None)
    return _is_bf16_dtype(declared)


def _device_capability_int() -> int | None:
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        return None
    cap = current_platform.get_device_capability(
        device_id=torch.accelerator.current_device_index()
    )
    return None if cap is None else cap.to_int()


def emulation_enabled(
    config: Any,
    runtime_dtype: torch.dtype,
    capability: int | None = -1,
) -> bool:
    """Gate: not killed, cc < 80, draft declared bf16, runtime fp16.

    ``capability`` -1 means "query the current CUDA device". None means no
    CUDA device, so the emulation is off.
    """
    # ported from FORK qwen3_dflash2.py:52-73 (plus the FP16-runtime check
    # at :215-220, which here gates instead of raising)
    if not envs.VLLM_SM70_DFLASH2_EMULATION:
        return False
    if runtime_dtype != torch.float16 or not draft_declared_bf16(config):
        return False
    if capability == -1:
        capability = _device_capability_int()
    return capability is not None and capability < 80


# --------------------------------------------------------------------------
# Pure-torch references (CPU, tests, and fallback when Triton is absent)
# --------------------------------------------------------------------------
def round_fp32_to_bf16_rne(value: torch.Tensor) -> torch.Tensor:
    """Bit-exact twin of the Triton ``_round_fp32_to_bf16_rne`` (fp32 in/out)."""
    # ported from FORK models/dflash_sm70.py:23-28 (Triton) as torch bit ops
    bits = value.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    rounded = torch.where(rounded >= 2**31, rounded - 2**32, rounded)
    return rounded.to(torch.int32).view(torch.float32)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return round_fp32_to_bf16_rne(x)


def silu_and_mul_reference(
    gate_up: torch.Tensor, gate_up_scale: float = DFLASH_SM70_GATE_UP_INPUT_SCALE
) -> tuple[torch.Tensor, torch.Tensor]:
    # ported from FORK models/dflash_sm70.py:169-180
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate = _bf16(gate * gate_up_scale)
    up = _bf16(up * gate_up_scale)
    activated = _bf16(gate * torch.sigmoid(gate))
    output = _bf16(activated * up)
    max_abs = output.abs().amax(dim=-1)
    row_scales = torch.maximum(max_abs / _FP16_TRANSPORT_MAX, torch.ones_like(max_abs))
    row_scales = torch.pow(2.0, torch.ceil(torch.log2(row_scales)))
    out = (output / row_scales.unsqueeze(-1)).to(gate_up.dtype)
    return out, row_scales.reshape(-1)


def scale_output_reference(
    output: torch.Tensor,
    row_scales: torch.Tensor,
    residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
) -> torch.Tensor:
    # ported from FORK models/dflash_sm70.py:241-244
    n_cols = output.shape[-1]
    flat = output.float().reshape(-1, n_cols)
    scaled = flat * (row_scales.reshape(-1, 1).float() / residual_scale)
    return scaled.reshape(output.shape).to(output.dtype)


def rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
    residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # ported from FORK models/dflash_sm70.py:346-363 (DFlashSM70RMSNorm._reference)
    orig_dtype = x.dtype
    value = x.float()
    residual_output = None
    if residual is not None:
        value = _bf16(value * residual_scale + residual.float())
        residual_output = value
    variance = value.square().mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    output = _bf16(normalized * _bf16(weight.float())).to(orig_dtype)
    if residual_output is None:
        return output
    return output, residual_output


# --------------------------------------------------------------------------
# Triton kernels (GPU). Defined only when Triton is importable.
# --------------------------------------------------------------------------
if HAS_TRITON:

    @triton.jit
    def _round_fp32_to_bf16_rne(value):
        # ported from FORK models/dflash_sm70.py:23-28
        bits = value.to(tl.uint32, bitcast=True)
        rounded_bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        return rounded_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def _dflash_silu_and_mul_sm70_kernel(
        output_ptr,
        output_scales_ptr,
        gate_up_ptr,
        n_cols: tl.constexpr,
        gate_up_scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # ported from FORK models/dflash_sm70.py:31-60
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        input_offsets = row * 2 * n_cols + offsets
        output_offsets = row * n_cols + offsets
        gate = tl.load(gate_up_ptr + input_offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        up = tl.load(gate_up_ptr + input_offsets + n_cols, mask=mask, other=0.0).to(
            tl.float32
        )
        gate = _round_fp32_to_bf16_rne(gate * gate_up_scale)
        up = _round_fp32_to_bf16_rne(up * gate_up_scale)
        activated_gate = _round_fp32_to_bf16_rne(gate * tl.sigmoid(gate))
        output = _round_fp32_to_bf16_rne(activated_gate * up)
        max_abs = tl.max(tl.abs(output), axis=0)
        output_scale = tl.maximum(max_abs / 32752.0, 1.0)
        output_scale = tl.exp2(tl.ceil(tl.log2(output_scale)))
        tl.store(output_scales_ptr + row, output_scale)
        tl.store(output_ptr + output_offsets, output / output_scale, mask=mask)

    @triton.jit
    def _dflash_scale_output_sm70_kernel(
        output_ptr,
        row_scales_ptr,
        n_cols: tl.constexpr,
        residual_scale: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # ported from FORK models/dflash_sm70.py:63-81
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets
        value = tl.load(output_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        row_scale = tl.load(row_scales_ptr + row).to(tl.float32)
        tl.store(
            output_ptr + row_offsets, value * (row_scale / residual_scale), mask=mask
        )

    @triton.jit
    def _dflash_rmsnorm_sm70_kernel(
        output_ptr,
        residual_output_ptr,
        x_ptr,
        residual_ptr,
        weight_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        residual_scale: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        # ported from FORK models/dflash_sm70.py:84-116
        row = tl.program_id(0).to(tl.int64)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets
        value = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(residual_ptr + row_offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            value = _round_fp32_to_bf16_rne(value * residual_scale + residual)
            tl.store(residual_output_ptr + row_offsets, value, mask=mask)
        variance = tl.sum(value * value, axis=0) / n_cols
        normalized = value * tl.rsqrt(variance + eps)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = _round_fp32_to_bf16_rne(weight)
        output = _round_fp32_to_bf16_rne(normalized * weight)
        tl.store(output_ptr + row_offsets, output, mask=mask)


def _use_triton(t: torch.Tensor) -> bool:
    return bool(HAS_TRITON and t.is_cuda)


def _num_warps(n_cols: int) -> int:
    # ported from FORK models/dflash_sm70.py:165-166
    return max(min(triton.next_power_of_2(triton.cdiv(n_cols, 256)), 16), 4)


# --------------------------------------------------------------------------
# Wrappers
# --------------------------------------------------------------------------
def dflash_silu_and_mul_sm70(
    gate_up: torch.Tensor,
    gate_up_scale: float = DFLASH_SM70_GATE_UP_INPUT_SCALE,
) -> tuple[torch.Tensor, torch.Tensor]:
    # ported from FORK models/dflash_sm70.py:183-221
    if gate_up.ndim < 2 or gate_up.shape[-1] % 2:
        raise ValueError(
            "DFlash SM70 SwiGLU expects [..., 2 * intermediate_size], "
            f"got {tuple(gate_up.shape)}."
        )
    if gate_up_scale <= 0:
        raise ValueError(f"gate_up_scale must be positive, got {gate_up_scale}.")
    if not _use_triton(gate_up):
        return silu_and_mul_reference(gate_up, gate_up_scale)
    if gate_up.dtype != torch.float16:
        raise ValueError(
            f"DFlash SM70 SwiGLU requires FP16 CUDA transport, got {gate_up.dtype}."
        )
    if not gate_up.is_contiguous():
        gate_up = gate_up.contiguous()
    n_cols = gate_up.shape[-1] // 2
    n_rows = gate_up.numel() // (2 * n_cols)
    output = torch.empty(
        (*gate_up.shape[:-1], n_cols), dtype=gate_up.dtype, device=gate_up.device
    )
    output_scales = torch.empty(n_rows, dtype=torch.float32, device=gate_up.device)
    _dflash_silu_and_mul_sm70_kernel[(n_rows,)](
        output,
        output_scales,
        gate_up,
        n_cols=n_cols,
        gate_up_scale=gate_up_scale,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=_num_warps(n_cols),
    )
    return output, output_scales


def dflash_scale_output_sm70(
    output: torch.Tensor,
    row_scales: torch.Tensor,
    residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
) -> torch.Tensor:
    # ported from FORK models/dflash_sm70.py:224-255
    if output.ndim < 2:
        raise ValueError(f"DFlash SM70 output must have ndim >= 2, got {output.ndim}.")
    if residual_scale <= 0:
        raise ValueError(f"residual_scale must be positive, got {residual_scale}.")
    n_cols = output.shape[-1]
    n_rows = output.numel() // n_cols
    if row_scales.numel() != n_rows:
        raise ValueError(
            f"DFlash SM70 row-scale mismatch: rows={n_rows}, "
            f"scales={row_scales.numel()}."
        )
    if not _use_triton(output):
        return scale_output_reference(output, row_scales, residual_scale)
    if output.dtype != torch.float16 or not output.is_contiguous():
        raise ValueError("DFlash SM70 W2 scaling requires contiguous CUDA FP16 output.")
    _dflash_scale_output_sm70_kernel[(n_rows,)](
        output,
        row_scales,
        n_cols=n_cols,
        residual_scale=residual_scale,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        num_warps=_num_warps(n_cols),
    )
    return output


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------
class DFlashSM70RMSNorm(nn.Module):
    """RMSNorm with FP32 residual transport and explicit BF16 RNE.

    Same call contract as vllm RMSNorm: ``norm(x)`` returns a tensor, and
    ``norm(x, residual)`` returns ``(out, residual_fp32)``. ``x`` is the
    sublayer output divided by ``residual_scale``.
    """

    # ported from FORK models/dflash_sm70.py:330-404
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: torch.dtype,
        residual_scale: float = DFLASH_SM70_WIDE_OUTPUT_SCALE,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.residual_scale = residual_scale
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not _use_triton(x):
            return rmsnorm_reference(
                x, self.weight, self.variance_epsilon, residual, self.residual_scale
            )
        if x.dtype != torch.float16:
            raise ValueError(
                f"DFlash SM70 RMSNorm requires FP16 CUDA transport, got {x.dtype}."
            )
        x = x.contiguous()
        if residual is not None:
            residual = residual.contiguous()
        n_rows = x.numel() // self.hidden_size
        output = torch.empty_like(x)
        has_residual = residual is not None
        residual_output = (
            torch.empty_like(x, dtype=torch.float32) if has_residual else output
        )
        _dflash_rmsnorm_sm70_kernel[(n_rows,)](
            output,
            residual_output,
            x,
            residual if residual is not None else x,
            self.weight,
            n_cols=self.hidden_size,
            eps=self.variance_epsilon,
            residual_scale=self.residual_scale,
            HAS_RESIDUAL=has_residual,
            BLOCK_SIZE=triton.next_power_of_2(self.hidden_size),
            num_warps=_num_warps(self.hidden_size),
        )
        if has_residual:
            return output, residual_output
        return output


class DFlashSM70MLP(nn.Module):
    """Wrap the Qwen MLP with overflow-safe BF16-equivalent math.

    Reuses the wrapped ``gate_up_proj`` / ``down_proj`` modules, so parameter
    names (``mlp.gate_up_proj.weight`` ...) and weight loading are unchanged.
    Returns ``true_output / DFLASH_SM70_WIDE_OUTPUT_SCALE``.
    """

    # ported from FORK models/dflash_sm70.py:407-419
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.gate_up_proj = mlp.gate_up_proj
        self.down_proj = mlp.down_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x / DFLASH_SM70_GATE_UP_INPUT_SCALE)
        x, row_scales = dflash_silu_and_mul_sm70(gate_up)
        x, _ = self.down_proj(x)
        return dflash_scale_output_sm70(x, row_scales)


def sm70_attention_forward(
    self: nn.Module, positions: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """DFlashQwen3Attention.forward plus the o_proj input scale.

    Copy of UP qwen3_dflash.py:268-295 (28 lines). The only change is the
    division before o_proj, ported from FORK qwen3_dflash.py:312-314.
    """
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q_shape, k_shape = q.shape, k.shape
    q = self.q_norm(
        q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
    ).view(q_shape)
    k = self.k_norm(
        k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
    ).view(k_shape)
    q, k = self.rotary_emb(positions, q, k)
    if self.v_scale is not None:
        v = v * self.v_scale
    attn_output = self.attn(q, k, v)
    attn_output = attn_output / self.output_input_scale
    output, _ = self.o_proj(attn_output)
    return output


_ATTN_CLS_CACHE: dict[type, type] = {}


def sm70_attention_cls(base: type) -> type:
    """Subclass of ``base`` (UP DFlashQwen3Attention) whose forward scales o_proj input."""
    if getattr(base, "_sm70_bf16_attention", False):
        return base
    cls = _ATTN_CLS_CACHE.get(base)
    if cls is None:
        cls = type(
            f"SM70{base.__name__}",
            (base,),
            {"forward": sm70_attention_forward, "_sm70_bf16_attention": True},
        )
        _ATTN_CLS_CACHE[base] = cls
    return cls


def apply_layer_emulation(layer: nn.Module, config: Any, dtype: torch.dtype) -> None:
    """Swap one constructed DFlash decoder layer to the SM70 BF16-range modules.

    Must run before weight loading; the parameter names are unchanged.
    """
    # ported from FORK qwen3_dflash2.py:213-231
    attn = layer.self_attn
    attn.output_input_scale = DFLASH_SM70_WIDE_OUTPUT_SCALE
    # Swap the class to a cached subclass that overrides forward() only.
    # Instance state, the Attention child and the KV-cache layer name stay
    # unchanged, and Dynamo sees an ordinary class-level forward.
    attn.__class__ = sm70_attention_cls(type(attn))
    layer.mlp = DFlashSM70MLP(layer.mlp)
    layer.input_layernorm = DFlashSM70RMSNorm(
        config.hidden_size, config.rms_norm_eps, dtype
    )
    layer.post_attention_layernorm = DFlashSM70RMSNorm(
        config.hidden_size, config.rms_norm_eps, dtype
    )
    layer.use_sm70_bf16_emulation = True


def apply_model_emulation(model: nn.Module, config: Any, dtype: torch.dtype) -> None:
    """Swap the model-level norms (context hidden_norm with no residual scale)."""
    # ported from FORK qwen3_dflash2.py:465-482
    model.hidden_norm = DFlashSM70RMSNorm(
        config.hidden_size, config.rms_norm_eps, dtype, residual_scale=1.0
    )
    model.norm = DFlashSM70RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
    model.use_sm70_bf16_emulation = True


# --------------------------------------------------------------------------
# Upstream subclasses (imported lazily: building them imports vllm models)
# --------------------------------------------------------------------------
def _build_classes() -> dict[str, type]:
    import torch.nn.functional as F

    from vllm.model_executor.models.qwen3_dflash2 import (
        DFlash2Qwen3DecoderLayer,
        DFlash2Qwen3ForCausalLM,
        DFlash2Qwen3Model,
    )

    class DFlash2SM70Qwen3DecoderLayer(DFlash2Qwen3DecoderLayer):
        # UP signature: qwen3_dflash2.py:230-239
        def __init__(
            self,
            vllm_config,
            *,
            config,
            layer_idx: int,
            cache_config=None,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            super().__init__(
                vllm_config,
                config=config,
                layer_idx=layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            )
            dtype = vllm_config.model_config.dtype
            self.use_sm70_bf16_emulation = emulation_enabled(config, dtype)
            if self.use_sm70_bf16_emulation:
                apply_layer_emulation(self, config, dtype)
                # K18: flag the draft projections for the M<=8 FP16 GEMM
                # (packed in load_weights below). FORK qwen3_dflash2.py:280-298.
                # M9a: with the DFlash2 verifier on (k=7), also flag them for
                # the channel-FP8 M8..32 route. FORK qwen3_dflash2.py:295-297.
                from vllm_sm70.ops import dflash2, dflash2_fp8

                dflash2.mark_draft_linears(
                    self,
                    fp8=dflash2_fp8.draft_fp8_wanted(vllm_config),
                    vllm_config=vllm_config,
                )

    class DFlash2SM70Qwen3Model(DFlash2Qwen3Model):
        decoder_layer_cls = DFlash2SM70Qwen3DecoderLayer

        # UP signature: qwen3_dflash2.py:364-370
        def __init__(self, *, vllm_config, start_layer_id: int = 0, prefix: str = ""):
            super().__init__(
                vllm_config=vllm_config, start_layer_id=start_layer_id, prefix=prefix
            )
            dtype = vllm_config.model_config.dtype
            self.use_sm70_bf16_emulation = emulation_enabled(self.config, dtype)
            if self.use_sm70_bf16_emulation:
                apply_model_emulation(self, self.config, dtype)
                logger.info(
                    "Using range-preserving BF16 DFlash2 arithmetic on SM70 "
                    "(FP32 residual, BF16 RNE, scaled FP16 projections)."
                )

        def _project_context_kv(
            self, context_states, num_ctx, num_layers, num_kv_heads, head_dim
        ):
            # UP qwen3_dflash.py:581-619 with the hidden_norm call swapped in
            # (FORK qwen3_dflash.py:659, 675-683). 25 lines copied.
            if not self.use_sm70_bf16_emulation:
                return super()._project_context_kv(
                    context_states, num_ctx, num_layers, num_kv_heads, head_dim
                )
            normed = self.hidden_norm(context_states)
            if self._fused_kv_weight is not None:
                all_kv = F.linear(
                    normed, self._fused_kv_weight, self._fused_kv_bias
                ).view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            else:
                layer_kv = []
                for proj, q_size in zip(
                    self._context_qkv_projs, self._context_q_sizes, strict=True
                ):
                    qkv, _ = proj(normed)
                    layer_kv.append(
                        qkv[:, q_size:].view(num_ctx, 2, num_kv_heads, head_dim)
                    )
                all_kv = torch.stack(layer_kv, dim=1)
            all_kv = all_kv.permute(2, 1, 0, 3, 4).contiguous()
            return all_kv[0], all_kv[1]

    class DFlash2SM70Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
        # UP factory: DFlashQwen3ForCausalLM.model_cls (qwen3_dflash.py:737, :748)
        model_cls = DFlash2SM70Qwen3Model

        def load_weights(self, weights):
            # UP qwen3_dflash.py:851. K18: pack the flagged projections once
            # the weights are final, before profiling and graph capture, so
            # the packed copies count in the model's memory.
            loaded = super().load_weights(weights)
            from vllm_sm70.ops import dflash2

            dflash2.prepare_marked(self)
            return loaded

    return {
        "DFlash2SM70Qwen3DecoderLayer": DFlash2SM70Qwen3DecoderLayer,
        "DFlash2SM70Qwen3Model": DFlash2SM70Qwen3Model,
        "DFlash2SM70Qwen3ForCausalLM": DFlash2SM70Qwen3ForCausalLM,
    }


_CLASSES: dict[str, type] | None = None


def __getattr__(name: str) -> Any:
    global _CLASSES
    if name in (
        "DFlash2SM70Qwen3DecoderLayer",
        "DFlash2SM70Qwen3Model",
        "DFlash2SM70Qwen3ForCausalLM",
    ):
        if _CLASSES is None:
            _CLASSES = _build_classes()
        return _CLASSES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------
LAZY_TARGET = f"{__name__}:DFlash2SM70Qwen3ForCausalLM"


def register() -> bool:
    """Point "DFlash2DraftModel" at the SM70 subclass. Returns True if registered.

    Skips (returns False) when the kill switch is engaged (info log) or no
    visible GPU is cc 7.0 (debug log). Raises when every visible GPU is cc 7.0
    but _sm70_C is missing, and when the registry did not take the override. The
    lazy "<module>:<class>" form keeps CUDA and vllm model imports out of
    plugin load (UP models/registry.py:1176-1212). It overwrites UP's entry
    (registry.py:666), logged at debug. Call it before the first
    get_model_architecture, which caches per model (model_loader/utils.py:340-356).
    Every plugin process runs it, because vllm.general_plugins load in all of
    them.
    """
    from vllm_sm70 import _ext
    from vllm_sm70 import platform as sm70_platform

    if not envs.VLLM_SM70_DFLASH2_EMULATION:
        logger.info("%s=0: DFlash2 SM70 emulation not registered.", KILL_SWITCH)
        return False
    if not sm70_platform.all_visible_gpus_are_sm70():
        logger.debug("No cc 7.0 GPUs visible: DFlash2 SM70 emulation not registered.")
        return False
    if not _ext.is_loaded():
        # A cc 7.0 host without _sm70_C is broken, not "not SM70". Nothing on
        # main fails fast on it: _ext.load() only logs at info (_ext.py:16-28),
        # and Sm70CudaPlatform.check_and_update_config just imports the areas
        # (_cuda_platform.py:72-77). So raise; vllm_sm70._register_area logs
        # it at ERROR (__init__.py:60-66) and the area stays unregistered.
        msg = (
            "vllm_sm70: every visible GPU is cc 7.0 but _sm70_C is not loaded; "
            "the BF16-trained DFlash2 draft would run in plain FP16 (overflow). "
            f"Fix the extension build, or set {KILL_SWITCH}=0 to accept that."
        )
        logger.warning(msg)
        raise RuntimeError(msg)
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(DRAFT_ARCH, LAZY_TARGET)
    entry = ModelRegistry.models.get(DRAFT_ARCH)
    module_name, _, class_name = LAZY_TARGET.partition(":")
    if (
        getattr(entry, "module_name", None),
        getattr(entry, "class_name", None),
    ) != (module_name, class_name):
        raise RuntimeError(
            f"vllm_sm70: {DRAFT_ARCH!r} resolves to {entry!r}, not {LAZY_TARGET}; "
            "upstream registry semantics changed."
        )
    return True
