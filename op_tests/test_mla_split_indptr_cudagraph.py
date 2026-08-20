# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test: get_meta_param's split indptr must survive graph replay.

`get_meta_param` used to build `num_kv_splits_indptr` inside its own
`@functools.lru_cache`, which made the cache entry the tensor's only owner.
A CUDA graph that captured a *cache hit* recorded a bare pointer and no
producing kernel, so once the entry was evicted -- `total_kv` is in the key and
changes almost every decode step, and captured keys are never looked up again
because replay runs no host code -- the storage was recycled and replay read
whatever landed there. `_fwd_kernel_stage2_asm` then derived
`num_valid_kv_splits` from garbage and indexed past the end of `Mid_O`,
surfacing as `Memory access fault ... Reason: Unknown`.

Building the indptr per call fixes it: `torch.arange` on CUDA is a fill kernel,
so capture records the kernel and every replay re-initialises the buffer.
"""

import pytest
import torch

import aiter.mla


def _capture(pool, fn):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = fn()
    return g, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cached_device_tensor_is_corrupted_by_replay():
    """Demonstrates the hazard this fix avoids, independent of aiter."""
    pool = torch.cuda.graph_pool_handle()
    n = 8
    dst = torch.zeros(n, dtype=torch.int32, device="cuda")

    cached = torch.arange(0, n, 1, dtype=torch.int32, device="cuda")
    g, _ = _capture(pool, lambda: dst.copy_(cached))
    del cached  # the eviction
    torch.cuda.synchronize()

    # A later capture in the same pool reuses the reclaimed storage.
    scribble = torch.zeros(4096, dtype=torch.int32, device="cuda")
    g2, _ = _capture(
        pool,
        lambda: scribble.copy_(
            torch.full((4096,), -777, dtype=torch.int32, device="cuda")
        ),
    )
    g2.replay()
    torch.cuda.synchronize()

    dst.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert dst.tolist() != list(range(n)), (
        "expected the pre-capture tensor to be clobbered; if this now passes, "
        "the allocator's graph-pool reuse changed and this test needs revisiting"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("bs", [8, 480])
def test_get_meta_param_indptr_survives_replay(bs):
    """The real regression: a captured get_meta_param must replay correctly."""
    pool = torch.cuda.graph_pool_handle()
    args = (None, bs, bs * 4096, 16, 1, torch.bfloat16, 1, 1)

    # Warm the cache first, so capture takes a HIT -- the exact condition that
    # used to record a pointer with no producing kernel.
    splits, _ = aiter.mla.get_meta_param(*args)
    expected = torch.arange(
        0, (bs + 1) * splits, splits, dtype=torch.int, device="cuda"
    )

    dst = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    g, _ = _capture(pool, lambda: dst.copy_(aiter.mla.get_meta_param(*args)[1]))

    # Churn the cache well past its maxsize so anything it owned is evicted,
    # then reuse the pool so the freed storage is handed out again.
    for i in range(1024):
        aiter.mla.get_meta_param(None, bs, bs * 4096 + i + 1, 16, 1, torch.bfloat16, 1, 1)
    scribble = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    g2, _ = _capture(
        pool,
        lambda: scribble.copy_(
            torch.full((1 << 16,), -777, dtype=torch.int32, device="cuda")
        ),
    )
    g2.replay()
    torch.cuda.synchronize()

    dst.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(dst, expected), (
        f"split indptr corrupted after replay (bs={bs}, splits={splits}): "
        f"{dst.tolist()[:16]} != {expected.tolist()[:16]}"
    )
