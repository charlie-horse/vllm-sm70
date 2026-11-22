# SPDX-License-Identifier: Apache-2.0
"""A CPU stand-in for ``torch.cuda.set_sync_debug_mode("error")``, for tests.

Inside ``no_host_sync()`` every op that would make the host wait for a CUDA
stream raises ``HostSync``: reading a tensor value on the host (``item``,
``bool``/``int``/``float`` of a tensor, ``tolist``, ``numpy``, ``cpu``),
ops whose output shape depends on tensor values (``nonzero``, one-argument
``torch.where``, ``unique``, ``bincount``, ``masked_select``,
``repeat_interleave`` without an output size, boolean-mask indexing) and
host data copied to a tensor (``torch.tensor``, ``torch.as_tensor``,
``torch.from_numpy``). ``allowed`` wraps a function whose syncs are exempt.
``no_host_sync(trace)`` also appends every op it sees to the list `trace`,
with the shapes and dtypes of its tensor outputs: two calls with equal
traces launch the same ops on the same allocation sizes, as a CUDA graph
replays them.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["HostSync", "allowed", "no_host_sync"]

aten = torch.ops.aten
_SYNC_OPS = {
    aten._local_scalar_dense,
    aten.nonzero,
    aten.nonzero_numpy,
    aten.argwhere,
    aten._unique,
    aten._unique2,
    aten.unique_dim,
    aten.unique_consecutive,
    aten.bincount,
    aten.masked_select,
    aten.histc,
}
_INDEX_OPS = {aten.index, aten.index_put, aten.index_put_, aten._index_put_impl_}
_TENSOR_METHODS = ("tolist", "numpy", "cpu")
_TORCH_FUNCTIONS = ("tensor", "as_tensor", "from_numpy")


class HostSync(AssertionError):
    """An op that waits for the device ran inside ``no_host_sync``."""


_state = {"active": False, "exempt": 0}


def _armed() -> bool:
    return _state["active"] and not _state["exempt"]


def _is_bool_index(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.dtype in (torch.bool, torch.uint8)
    if isinstance(value, (list, tuple)):
        return any(_is_bool_index(v) for v in value)
    return False


def _signature(out: Any) -> tuple:
    if isinstance(out, torch.Tensor):
        return ((tuple(out.shape), out.dtype),)
    if isinstance(out, (list, tuple)):
        return tuple(sig for o in out for sig in _signature(o))
    return ()


class _Mode(TorchDispatchMode):
    def __init__(self, trace: list | None = None) -> None:
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _armed():
            packet = func.overloadpacket
            if packet in _SYNC_OPS:
                raise HostSync(f"{func} syncs with the host")
            if packet is aten.repeat_interleave and kwargs.get("output_size") is None:
                raise HostSync(f"{func} without output_size syncs with the host")
            if packet in _INDEX_OPS and _is_bool_index(args[1]):
                raise HostSync(f"{func} with a boolean mask syncs with the host")
        out = func(*args, **kwargs)
        if self.trace is not None:
            self.trace.append((str(func), _signature(out)))
        return out


def _guarded(name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        if _armed():
            raise HostSync(f"{name} syncs with the host")
        return original(*args, **kwargs)

    return wrapper


def allowed(fn: Callable) -> Callable:
    """`fn` with its host syncs exempt."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _state["exempt"] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _state["exempt"] -= 1

    return wrapper


@contextlib.contextmanager
def no_host_sync(trace: list | None = None) -> Iterator[None]:
    saved = [(torch.Tensor, n, getattr(torch.Tensor, n)) for n in _TENSOR_METHODS]
    saved += [(torch, n, getattr(torch, n)) for n in _TORCH_FUNCTIONS]
    for owner, name, original in saved:
        setattr(owner, name, _guarded(name, original))
    _state.update(active=True, exempt=0)
    try:
        with _Mode(trace):
            yield
    finally:
        _state.update(active=False, exempt=0)
        for owner, name, original in saved:
            setattr(owner, name, original)
