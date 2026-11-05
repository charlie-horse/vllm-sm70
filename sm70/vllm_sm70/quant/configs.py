# SPDX-License-Identifier: Apache-2.0
"""M2 sm70-quant-gates: let SM70 through upstream's quant capability gates.

What: subclasses of upstream's ``Fp8Config`` and ``ModelOptNvFp4Config``,
  registered under the existing names "fp8" and "modelopt_fp4". They
  override only ``get_min_capability``: 70 when ``is_sm70_ready()``,
  otherwise upstream's own value (75 for both at UP 4ba787e,
  quantization/fp8.py:146-147 and modelopt.py:772-774). On every other device
  the result is identical to upstream.
Why: upstream refuses any GPU below ``get_min_capability()``
  (config/vllm.py:926-936). Overwriting the built-in names (UP
  quantization/__init__.py:88-97 and :195) keeps each name's place in the
  override order (config/model.py:1395-1425). Flint's nested ModelOpt config
  (quant_method "modelopt", quant_algo "W4A16_NVFP4") resolves through the
  inherited ``override_quantization_method`` (modelopt.py:776-783) to
  "modelopt_fp4", and so to the class below. ``from_config`` builds the
  instance with ``cls(...)`` (fp8.py:158-174; modelopt.py:403-409 and
  :830-837), so the instance is ours too.
Kill-switch: VLLM_SM70_QUANT=0 skips registration. Upstream's classes stay in
  place and refuse SM70 with their usual ValueError. VLLM_SM70_FP8_TURBOMIND=0
  puts only the "fp8" gate back to upstream's value; "modelopt_fp4" is not
  affected.

Registration is an import side effect: the engine core unpickles
``vllm_config.quant_config`` before it loads plugins (UP v1/engine/core.py
:1462-1464 vs :132-135), and unpickling imports this module by qualname.
Re-importing is safe: registering a name again only overwrites it with the
module's current class (UP quantization/__init__.py:89-95).

``vllm_sm70.quant.register()`` (K01's area hook) imports this module and
calls ``register()`` below. It is not imported from ``vllm_sm70.platform``,
because importing vLLM's quantization package there would re-enter platform
resolution.
"""

from typing import Any

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import (
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

from vllm_sm70 import envs
from vllm_sm70 import platform as sm70_platform
from vllm_sm70.config import fp8_turbomind_enabled

logger = init_logger(__name__)

SM70_MIN_CAPABILITY = 70


def _min_capability(upstream: int) -> int:
    # K01's readiness check: every visible GPU is cc 7.0 (NVML, no CUDA
    # init) and _sm70_C is loaded. Looked up on the module so tests can
    # patch it.
    return SM70_MIN_CAPABILITY if sm70_platform.is_sm70_ready() else upstream


class Sm70Fp8Config(Fp8Config):
    """Upstream ``Fp8Config`` with the SM70 capability gate (Qwen FP8).

    With VLLM_SM70_FP8_TURBOMIND=0 the gate returns upstream's value (75), so
    upstream refuses FP8 on SM70, as the fork did: v1 has no other SM70 FP8
    kernel.
    """

    @classmethod
    def get_min_capability(cls) -> int:
        upstream = super().get_min_capability()
        if not fp8_turbomind_enabled():
            return upstream
        return _min_capability(upstream)

    # HOOK(M3/M4): get_quant_method override routing LinearBase/RoutedExperts
    # to the SM70 FP8 linear and MoE methods goes here.


class Sm70ModelOptNvFp4Config(ModelOptNvFp4Config):
    """Upstream ``ModelOptNvFp4Config`` with the SM70 capability gate (Flint)."""

    @classmethod
    def get_min_capability(cls) -> int:
        return _min_capability(super().get_min_capability())

    # HOOK(M3/M4): LINEAR_METHOD_BUILDERS["W4A16_NVFP4"] (M3) and the
    # FusedMoEMethodCls dispatch (M4) are installed next to this class, at
    # import time, for the same unpickling reason as the registration.


# Registered name -> our class.
SM70_QUANT_CONFIGS: dict[str, type] = {
    "fp8": Sm70Fp8Config,
    "modelopt_fp4": Sm70ModelOptNvFp4Config,
}

# Flint's checkpoint shape (FORK cf9492a vllm/models/flint/convert_gguf.py
# :114-123), used to check that override_quantization_method still names
# our registry slot.
_FLINT_QUANT_CFG: dict[str, Any] = {
    "quant_method": "modelopt",
    "quantization": {
        "quant_algo": "W4A16_NVFP4",
        "group_size": 16,
        "kv_cache_quant_algo": None,
        "exclude_modules": ["model.embed_tokens", "*.mlp.gate", "lm_head"],
    },
}


def _register_all() -> bool:
    if not envs.VLLM_SM70_QUANT:
        logger.info("VLLM_SM70_QUANT=0: SM70 quant configs not registered.")
        return False
    for name, cls in SM70_QUANT_CONFIGS.items():
        register_quantization_config(name)(cls)
    return True


REGISTERED = _register_all()


def register() -> None:
    """Plugin entry hook. Raises if the registration didn't take effect.

    Upstream changes that would otherwise make the override silently not
    apply (D4) fail here instead. It never catches and logs: at cc 7.0 the
    plugin fails startup on an area error (K03). It is idempotent and safe
    to retry, since it only re-checks the registry.
    """
    if not REGISTERED:
        return
    for name, cls in SM70_QUANT_CONFIGS.items():
        resolved = get_quantization_config(name)
        if resolved is not cls:
            raise RuntimeError(
                f"vllm_sm70: quant method {name!r} resolves to {resolved!r}, "
                f"not {cls.__qualname__}; upstream registry semantics changed."
            )
        if "get_min_capability" not in cls.__dict__:
            raise RuntimeError(f"vllm_sm70: {cls.__qualname__} lost its gate.")
    detected = Sm70ModelOptNvFp4Config.override_quantization_method(
        _FLINT_QUANT_CFG, None
    )
    if detected != "modelopt_fp4":
        raise RuntimeError(
            "vllm_sm70: upstream now maps W4A16_NVFP4 ModelOpt checkpoints to "
            f"{detected!r}, not 'modelopt_fp4'; update SM70_QUANT_CONFIGS."
        )
