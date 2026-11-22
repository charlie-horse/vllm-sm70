# SPDX-License-Identifier: Apache-2.0
"""M12 inkling2-sm70-quant: the "inkling2" quantization method on SM70.

What: ``Inkling2Sm70QuantConfig``, registered under "inkling2". It claims a
  checkpoint whose config ``model_type`` is "inkling2_loop", accepts fp16
  activations at capability 70 and returns no per-layer quant method: only
  the routed experts are quantized (NVFP4 or MXFP8) and the model's own
  expert loader handles them. ``expert_formats`` and ``load_expert_formats``
  read the per-layer expert format from hf_quant_config.json.
Why: config.json declares ``{"quant_method": "modelopt", "quant_algo":
  "NVFP4"}``, which upstream resolves to "modelopt_fp4" and its linear and
  MoE methods. Custom methods are probed before the built-in ones and get
  ``hf_config`` (UP config/model.py:1423-1433).
Kill-switch: VLLM_SM70_QUANT=0 skips registration; the checkpoint then
  resolves to "modelopt_fp4" as upstream.

Registration is an import side effect, as in vllm_sm70/quant/configs.py: the
engine core unpickles ``vllm_config.quant_config`` before it loads plugins.
``vllm_sm70.quant.register()`` imports this module in every process and calls
``register()`` below; the package's own ``register()`` stays free of vLLM's
quantization imports.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    QuantizationMethods,
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

from vllm_sm70 import envs

logger = init_logger("vllm.sm70.quant")

QUANT_NAME = "inkling2"
MODEL_TYPE = "inkling2_loop"
HF_QUANT_CONFIG_FILE = "hf_quant_config.json"
SM70_MIN_CAPABILITY = 70

# compute_dtype values of hf_quant_config.json (UP c32251b
# models/inkling2/quant_config.py:21).
COMPUTE_DTYPES = ("float32", "bfloat16", "mxfp8", "nvfp4")
EXPERT_FORMATS = ("nvfp4", "mxfp8")
_EXPERT_TENSOR = re.compile(
    r"^model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13|w2)_weight$"
)

__all__ = [
    "COMPUTE_DTYPES",
    "EXPERT_FORMATS",
    "HF_QUANT_CONFIG_FILE",
    "MODEL_TYPE",
    "QUANT_NAME",
    "REGISTERED",
    "Inkling2Sm70QuantConfig",
    "expert_formats",
    "load_expert_formats",
    "mxfp8_layers",
    "register",
]


class Inkling2Sm70QuantConfig(QuantizationConfig):
    """Marker config: no layer gets a quant method.

    ``from_config`` receives config.json's flat ``quantization_config`` (UP
    model_loader/weight_utils.py:273-277), which carries no per-layer data,
    so the instance holds only that dict.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.original_config: dict[str, Any] = dict(config or {})

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return QUANT_NAME  # type: ignore[return-value]

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return SM70_MIN_CAPABILITY

    @staticmethod
    def get_config_filenames() -> list[str]:
        return [HF_QUANT_CONFIG_FILE]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Inkling2Sm70QuantConfig:
        return cls(config)

    @classmethod
    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        hf_config: Any = None,
    ) -> QuantizationMethods | None:
        # model_type, not architectures: --hf-overrides replaces the latter.
        if getattr(hf_config, "model_type", None) == MODEL_TYPE:
            return QUANT_NAME  # type: ignore[return-value]
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        return None


def expert_formats(hf_quant_config: dict[str, Any]) -> dict[int, str]:
    """Layer index -> "nvfp4" or "mxfp8", from a parsed hf_quant_config.json.

    Reads ``model.parameter_groups`` only. Raises ValueError on an unknown
    ``compute_dtype``, an expert weight in an unquantized group, or a layer
    whose ``w13_weight`` and ``w2_weight`` are missing or differ.
    """
    found: dict[int, dict[str, str]] = {}
    for group in hf_quant_config["model"]["parameter_groups"]:
        compute = group["compute_dtype"]
        if compute not in COMPUTE_DTYPES:
            raise ValueError(f"Unsupported Inkling 2 compute dtype: {compute!r}")
        for name in group["tensors"]:
            match = _EXPERT_TENSOR.match(name)
            if match is None:
                continue
            if compute not in EXPERT_FORMATS:
                raise ValueError(
                    f"{name}: expert weight with compute dtype {compute!r}"
                )
            layer, which = int(match.group(1)), match.group(2)
            if found.setdefault(layer, {}).setdefault(which, compute) != compute:
                raise ValueError(f"{name}: listed as both nvfp4 and mxfp8")
    out: dict[int, str] = {}
    for layer in sorted(found):
        parts = found[layer]
        if set(parts) != {"w13", "w2"} or parts["w13"] != parts["w2"]:
            raise ValueError(
                f"layer {layer}: expert weights w13/w2 are {parts}, expected "
                "both in one format"
            )
        out[layer] = parts["w13"]
    return out


def mxfp8_layers(hf_quant_config: dict[str, Any]) -> frozenset[int]:
    """Indices of the layers whose experts are MXFP8."""
    formats = expert_formats(hf_quant_config)
    return frozenset(layer for layer, fmt in formats.items() if fmt == "mxfp8")


def load_expert_formats(
    model: str | Path, revision: str | None = None
) -> dict[int, str] | None:
    """``expert_formats`` of the checkpoint's hf_quant_config.json.

    ``model`` is a local directory or a Hub repo id. Returns None when the
    file is absent.
    """
    from vllm.transformers_utils.repo_utils import get_hf_file_to_dict

    raw = get_hf_file_to_dict(HF_QUANT_CONFIG_FILE, model, revision)
    if raw is None:
        return None
    return expert_formats(raw)


def _register() -> bool:
    if not envs.VLLM_SM70_QUANT:
        logger.info(
            "VLLM_SM70_QUANT=0: SM70 %r quant config not registered.", QUANT_NAME
        )
        return False
    if QUANT_NAME in get_args(QuantizationMethods):
        # A built-in of that name is probed in upstream's fixed order and
        # must be in its overrides list (UP config/model.py:1440-1444).
        logger.warning(
            "vllm_sm70: quant method %r is built into this vLLM; keeping it, "
            "%s is not used.",
            QUANT_NAME,
            Inkling2Sm70QuantConfig.__name__,
        )
        return False
    register_quantization_config(QUANT_NAME)(Inkling2Sm70QuantConfig)
    return True


REGISTERED = _register()


def register() -> None:
    """Make sure "inkling2" resolves to this module's class.

    Registers it again when the name is gone from the registry; raises when
    another class holds it. No-op when registration was skipped.
    """
    if not (REGISTERED and envs.VLLM_SM70_QUANT):
        return
    try:
        resolved = get_quantization_config(QUANT_NAME)
    except (KeyError, ValueError):
        register_quantization_config(QUANT_NAME)(Inkling2Sm70QuantConfig)
        return
    if resolved is not Inkling2Sm70QuantConfig:
        raise RuntimeError(
            f"vllm_sm70: quant method {QUANT_NAME!r} resolves to {resolved!r}, "
            f"not {Inkling2Sm70QuantConfig.__qualname__}; upstream registry "
            "semantics changed."
        )
