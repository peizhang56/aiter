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

    ``handle`` is an initialized QuickReduce communicator (2, 4 or 8 ranks).
    ``output`` is this rank's shard of ``S`` elements and ``input`` the rank's
    full tensor: rank ``r`` receives the sum of elements ``[r*S, (r+1)*S)`` of
    every rank's input. ``input`` may hold fewer than ``world_size*S`` elements
    (a ragged tail, e.g. a row count that is not a multiple of the world size):
    the missing elements read as zero, with no padded copy. Both are contiguous
    FP16/BF16 tensors on the same GPU; ``S`` and the input size are multiples of
    16 bytes. BF16 inputs use FP16 communication arithmetic when
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
    """Reduce-scatter the routed MoE term straight from its per-slot MXFP8 rows (gfx950).

    ``route`` is the OPUS stage 2 route output, uint8 ``[rows * topk, hidden + hidden / 8]``
    (``hidden`` fp8 e4m3 codes, then one e8m0 scale per 8 codes); ``topk`` is 6 and
    ``hidden % 128 == 0``. The result equals ``qr_reduce_scatter`` of the bf16 tensor
    ``opus_moe_stage2_reduce_token_slot_route_output`` returns for the same rows,
    bitwise, without materializing that ``[rows, hidden]`` tensor. ``output`` is this rank's
    bf16 shard; ``rows * hidden`` may be below ``world_size * output.numel()`` (the rest reads as
    zero). Same communicator and ordering rules as ``qr_reduce_scatter``.
    """
