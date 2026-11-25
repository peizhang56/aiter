# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-phase INT4 reduce-scatter using QuickReduce IPC state."""

import torch

from aiter.jit.core import compile_ops


@compile_ops("module_quick_reduce_scatter", develop=True)
def qr_reduce_scatter(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    cast_bf2half: bool = True,
) -> None:
    """Write this rank's contiguous sum shard using the QuickReduce INT4 codec.

    ``handle`` is an initialized four- or eight-rank QuickReduce communicator.
    ``input`` and ``output`` are contiguous FP16/BF16 tensors on the same GPU
    with 16-byte-aligned byte counts. Rank ``r`` receives the sum of elements
    ``[r * S, (r + 1) * S)`` of every rank's ``input``, ``S = output.numel()``.
    ``input`` may hold fewer than ``world_size * S`` elements; the rest reads
    as zero. BF16 inputs use FP16 communication arithmetic when
    ``cast_bf2half`` is set. Calls sharing the communicator must be serialized
    on its current stream.
    """


@compile_ops("module_quick_reduce_scatter", develop=True)
def qr_reduce_scatter_route(
    handle: int,
    route: torch.Tensor,
    output: torch.Tensor,
    rows: int,
    topk: int,
    hidden: int,
) -> None:
    """Reduce-scatter an MoE layer's routed term from its per-slot MXFP8 rows.

    ``route`` is the OPUS stage 2 per-slot output (``return_per_slot=True``
    with a uint8 route dtype), ``[rows * topk, hidden + hidden / 8]``: ``hidden``
    e4m3 codes then one e8m0 scale per 8 codes per row, token-major. The
    result is bitwise equal to ``qr_reduce_scatter(handle, x, output, True)``
    with ``x = opus_moe_stage2_reduce_token_slot_route_output_fwd(route,
    topk=topk)``, without materializing ``x``. ``output`` is this rank's bf16
    shard; ``rows * hidden`` may be below ``world_size * output.numel()``.
    ``hidden`` must be a multiple of 64. gfx950 only. Same communicator rules
    as ``qr_reduce_scatter``.
    """
