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
