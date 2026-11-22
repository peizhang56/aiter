# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Single-phase INT4 all-gather using QuickReduce IPC state."""

import torch

from aiter.jit.core import compile_ops


@compile_ops("module_quick_all_gather", develop=True)
def qr_all_gather(
    handle: int,
    input: torch.Tensor,
    output: torch.Tensor,
    cast_bf2half: bool = True,
) -> None:
    """Gather every rank's shard through the QuickReduce INT4 codec.

    ``handle`` is an initialized QuickReduce communicator (2, 4 or 8 ranks).
    ``input`` is this rank's shard and ``output`` holds ``world_size`` shards
    back to back: contiguous FP16/BF16 tensors on the same GPU, with a
    16-byte-multiple shard. BF16 inputs use FP16 communication arithmetic when
    ``cast_bf2half`` is set. ``qr_reduce_scatter`` followed by ``qr_all_gather``
    over the same tensor equals ``qr_all_reduce`` bitwise. Calls sharing the
    communicator must be serialized on its current stream.
    """
