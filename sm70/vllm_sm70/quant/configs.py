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

logger = init_logger("vllm.sm70.quant")

SM70_MIN_CAPABILITY = 70


def _min_capability(upstream: int) -> int:
    # K01's readiness check: every visible GPU is cc 7.0 (NVML, no CUDA
    # init) and _sm70_C is loaded. Looked up on the module so tests can
    # patch it.
    return SM70_MIN_CAPABILITY if sm70_platform.is_sm70_ready() else upstream


def fp8_route_enabled() -> bool:
    """M6 routing gate: SM70 ready and VLLM_SM70_FP8_TURBOMIND not "0"."""
    return fp8_turbomind_enabled() and sm70_platform.is_sm70_ready()


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

    def get_quant_method(self, layer, prefix: str):
        """M6: Qwen block-FP8 linears and routed experts -> SM70 TurboMind.

        Only when is_sm70_ready() (K01) and VLLM_SM70_FP8_TURBOMIND is not 0.
        Skipped layers (UP fp8.py:180-197, is_layer_skipped) and everything
        else, including KV-cache methods, go to upstream. At TP outside
        (1, 2) the MoE method raises instead of falling back.
        """
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )
        from vllm.model_executor.layers.linear import LinearBase
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            is_layer_skipped,
        )

        if (
            isinstance(layer, (LinearBase, RoutedExperts))
            and fp8_route_enabled()
            and not is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            )
        ):
            if isinstance(layer, LinearBase):
                from vllm_sm70.quant.fp8_linear import Fp8Sm70LinearMethod

                return Fp8Sm70LinearMethod(self)
            from vllm_sm70.quant.fp8_moe import Fp8Sm70MoEMethod

            return Fp8Sm70MoEMethod(self, layer)
        return super().get_quant_method(layer, prefix)


class Sm70ModelOptNvFp4Config(ModelOptNvFp4Config):
    """Upstream ``ModelOptNvFp4Config`` with the SM70 capability gate (Flint)."""

    @classmethod
    def get_min_capability(cls) -> int:
        return _min_capability(super().get_min_capability())

    # M3: LINEAR_METHOD_BUILDERS["W4A16_NVFP4"] is installed below by
    # _install_m3, at import time, for the same unpickling reason as the
    # registration.

    def get_quant_method(self, layer, prefix: str):
        """M4: routed experts -> the SM70 TurboMind NVFP4 MoE method.

        Only when is_sm70_ready() (K01) and the layer is not excluded (UP
        modelopt.py:236-239). Everything else, and every non-SM70 device,
        goes to upstream (modelopt.py:226-262), whose FusedMoEMethodCls
        (:1199) would select an NVFP4 MoE backend that cc 7.0 doesn't have.
        """
        from vllm.model_executor.layers.fused_moe.routed_experts import (
            RoutedExperts,
        )

        if (
            isinstance(layer, RoutedExperts)
            and sm70_platform.is_sm70_ready()
            and not self.is_layer_excluded(prefix)
        ):
            from vllm_sm70.quant.nvfp4_moe import build_moe_method

            return build_moe_method(self, layer.moe_config)
        return super().get_quant_method(layer, prefix)


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


def _install_m3() -> bool:
    """M3: route ModelOpt W4A16_NVFP4 linears to the SM70 TurboMind method
    (vllm_sm70/quant/nvfp4_linear.py). The builder re-checks is_sm70_ready()
    for each layer and otherwise falls through to upstream's
    ModelOptLinearMethod. Kill switch: VLLM_SM70_MODELOPT_NVFP4_GEMV=0 only
    drops the M<=2 GEMV; VLLM_SM70_QUANT=0 skips this together with M2."""
    if not (REGISTERED and sm70_platform.is_sm70_ready()):
        return False
    from vllm_sm70.quant.nvfp4_linear import install_linear_builder

    install_linear_builder()
    return True


M3_INSTALLED = _install_m3()


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
    if M3_INSTALLED:
        from vllm_sm70.quant.nvfp4_linear import builder_installed

        if not builder_installed():
            raise RuntimeError(
                "vllm_sm70: LINEAR_METHOD_BUILDERS['W4A16_NVFP4'] was replaced "
                "after M3 installed it."
            )
    for cls in (Sm70Fp8Config, Sm70ModelOptNvFp4Config):
        if "get_quant_method" not in cls.__dict__:
            raise RuntimeError(f"vllm_sm70: {cls.__qualname__} lost its MoE hook.")
    if "get_quant_method" not in Sm70ModelOptNvFp4Config.__dict__:
        raise RuntimeError("vllm_sm70: Sm70ModelOptNvFp4Config lost its M4 MoE hook.")
    detected = Sm70ModelOptNvFp4Config.override_quantization_method(
        _FLINT_QUANT_CFG, None
    )
    if detected != "modelopt_fp4":
        raise RuntimeError(
            "vllm_sm70: upstream now maps W4A16_NVFP4 ModelOpt checkpoints to "
            f"{detected!r}, not 'modelopt_fp4'; update SM70_QUANT_CONFIGS."
        )
