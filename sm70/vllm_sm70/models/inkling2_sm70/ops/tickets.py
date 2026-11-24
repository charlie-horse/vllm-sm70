# SPDX-License-Identifier: Apache-2.0
"""Inkling 2: the int32 counters of the kernels that finish a launch in its
last program (routing.py ``route`` of a ``RouterInput``, visit.py
``gate_blend``).

A program adds 1 to its counter after its stores (acq_rel, GPU scope); the
program that draws the last ticket reads what the others stored and puts the
counter back to 0, so a replayed CUDA graph finds it at 0. One buffer of
``COUNTERS`` zeros per device, made on first use and kept: a launch that
uses counters must not overlap another launch of the same counters, which
holds for the kernels of one stream.
"""

from __future__ import annotations

import torch

__all__ = ["COUNTERS", "GATE_ROWS", "GATE_TICKET", "ROUTE_TICKET", "ticket_counter"]

COUNTERS = 128
ROUTE_TICKET = 0
GATE_TICKET = 1
# Two counters per (stream, token) row of a gate_blend launch from here on.
GATE_ROWS = 2

_BUFFERS: dict[torch.device, torch.Tensor] = {}


def ticket_counter(device: torch.device, index: int) -> torch.Tensor:
    """The counters of `device` from `index` on, a view."""
    device = torch.device(device)
    if device not in _BUFFERS:
        _BUFFERS[device] = torch.zeros(COUNTERS, dtype=torch.int32, device=device)
    return _BUFFERS[device][index:]
