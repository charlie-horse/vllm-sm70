# SPDX-License-Identifier: Apache-2.0
"""Planted bugs for sm70/scripts/m5_mutation_check.sh (test-only).
T4d and T7 must FAIL with each required one. M5_MUTATION=<name> with this
directory first on PYTHONPATH makes ``sitecustomize`` install it in every
process. Hook mutations wrap ``qk_norm_rope_kv`` in
``vllm_sm70.flint.qk_norm_rope_kv``: the registered op's body looks it up in
the module globals at call time, so the wrapper runs (also at CUDA-graph
capture); the unfused route never calls it.
  rope_pos_decode   positions + 1 on calls with <= DECODE_MAX_TOKENS tokens
  kv_slot_decode    slot_mapping + 1 (padding kept) on such calls
  moe_decode_scale  M<=2 MoE decode output (flint_moe_decode_out) x 1.01;
                    only T4d can catch it (required, by the pooled-mean gate)
  moe_decode_scale_half  x 1.005: INFO probe, never required
  rope_theta_1p01   config mutation (the tests apply it): rope_theta x 1.01;
                    INFO only, too weak at short positions to be required
"""

import functools
import inspect
import json
import os
import sys

TARGET = "vllm_sm70.flint.qk_norm_rope_kv"
MOE_TARGET = "vllm_sm70.ops.flint_moe"
MOE_DECODE_SCALE = 1.01
MOE_DECODE_SCALES = {"moe_decode_scale": MOE_DECODE_SCALE, "moe_decode_scale_half": 1.005}
QK_MUTATIONS = ("rope_pos_decode", "kv_slot_decode")
MOE_MUTATIONS = tuple(MOE_DECODE_SCALES)
HOOK_MUTATIONS = QK_MUTATIONS + MOE_MUTATIONS
CONFIG_MUTATIONS = ("rope_theta_1p01",)
MUTATIONS = HOOK_MUTATIONS + CONFIG_MUTATIONS
DECODE_MAX_TOKENS = 8
ROPE_THETA_FACTOR = 1.01


_FIRED: set = set()


def _mark_fired(mutation: str) -> None:
    """Print once per process the first time a mutation actually changes a
    call (installed is not enough; it must be shown to have fired)."""
    if mutation in _FIRED:
        return
    _FIRED.add(mutation)
    print(f"M5_MUTATION {mutation} fired in pid {os.getpid()}", file=sys.stderr, flush=True)


def mutate_call(mutation: str, bound: inspect.BoundArguments) -> bool:
    """Apply a hook mutation in place to the bound qk_norm_rope_kv args;
    True if the call was changed."""
    a = bound.arguments
    if a["qkv"].shape[0] > DECODE_MAX_TOKENS:
        return False
    if mutation == "rope_pos_decode":
        cs = a.get("cos_sin_cache")
        if cs is not None:
            pos = a["positions"]
            bumped = (pos + 1).clamp(max=cs.shape[0] - 1)
            a["positions"] = pos.where(pos < 0, bumped)
            return True
    elif mutation == "kv_slot_decode":
        slots, kc = a.get("slot_mapping"), a.get("key_cache")
        if slots is not None and kc is not None:
            total = kc.shape[0] * kc.shape[1]
            a["slot_mapping"] = slots.where(slots < 0, (slots + 1) % total)
            return True
    else:
        raise ValueError(f"not a hook mutation: {mutation!r}")
    return False


def make_wrapper(orig, mutation: str):
    sig = inspect.signature(orig)

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if mutate_call(mutation, bound):
            _mark_fired(mutation)
        return orig(*bound.args, **bound.kwargs)

    wrapper._m5_mutation = mutation
    wrapper._m5_original = orig
    return wrapper


def make_moe_wrapper(orig, mutation: str):
    """flint_moe_decode_out returns its ``out`` buffer; scale it in place."""

    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        out = orig(*args, **kwargs)
        out.mul_(MOE_DECODE_SCALES[mutation])
        _mark_fired(mutation)
        return out

    wrapper._m5_mutation = mutation
    wrapper._m5_original = orig
    return wrapper


def _target_for(mutation: str) -> tuple[str, str]:
    if mutation in MOE_MUTATIONS:
        return MOE_TARGET, "flint_moe_decode_out"
    return TARGET, "qk_norm_rope_kv"


def install(module, mutation: str) -> None:
    """Patch the mutation's target function on ``module`` (idempotent)."""
    _, attr = _target_for(mutation)
    cur = getattr(module, attr)
    if getattr(cur, "_m5_mutation", None) == mutation:
        return
    orig = getattr(cur, "_m5_original", cur)
    wrap = make_moe_wrapper if mutation in MOE_MUTATIONS else make_wrapper
    setattr(module, attr, wrap(orig, mutation))
    print(f"M5_MUTATION {mutation} installed in pid {os.getpid()}", file=sys.stderr, flush=True)


class _PatchingLoader:
    def __init__(self, loader, mutation):
        self._loader = loader
        self._mutation = mutation

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        install(module, self._mutation)

    def __getattr__(self, name):
        return getattr(self._loader, name)


class _Finder:
    """Meta-path finder: patch TARGET right after it is first imported."""

    def __init__(self, mutation):
        self.mutation = mutation

    def find_spec(self, name, path=None, target=None):
        if name != _target_for(self.mutation)[0]:
            return None
        import importlib.machinery

        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, self.mutation)
        return spec


def install_import_hook(mutation: str) -> None:
    if mutation not in MUTATIONS:
        raise ValueError(f"M5_MUTATION={mutation!r}; known: {', '.join(MUTATIONS)}")
    if mutation not in HOOK_MUTATIONS:
        return  # config mutation: the tests apply it
    target = _target_for(mutation)[0]
    if target in sys.modules:
        install(sys.modules[target], mutation)
        return
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder(mutation))


def rope_theta_override(model_dir: str, factor: float = ROPE_THETA_FACTOR) -> dict:
    """hf_overrides entries that scale the checkpoint's RoPE theta.

    Flint keeps theta in ``rope_parameters`` (flat, or nested per layer type,
    as _layer_rope_parameters in flint_sm70/model.py reads it) and in the
    legacy ``rope_theta``; FlintConfig defaults to 10000.0 when both are
    absent. Both are overridden, consistently, because vLLM's
    patch_rope_parameters runs after hf_overrides."""
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    base = cfg.get("rope_theta")
    params = cfg.get("rope_parameters")
    if isinstance(params, dict) and params and all(isinstance(v, dict) for v in params.values()):
        new_params = {}
        for layer_type, p in params.items():
            p = dict(p)
            p["rope_theta"] = float(p.get("rope_theta", base if base is not None else 10000.0)) * factor
            new_params[layer_type] = p
    else:
        p = dict(params or {"rope_type": "default"})
        theta = p.get("rope_theta", base if base is not None else 10000.0)
        p["rope_theta"] = float(theta) * factor
        new_params = p
        if base is None:
            base = theta
    out = {"rope_parameters": new_params}
    if base is not None:
        out["rope_theta"] = float(base) * factor
    return out
