# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""qr_reduce_scatter_route: the reduce-scatter that reads the MoE route output directly (gfx950).

The OPUS MoE stage 2 can hand over its per-slot MXFP8 rows ``[rows * 6, H + H / 8]`` instead of
reducing them into a bf16 ``[rows, H]`` tensor. ``qr_reduce_scatter_route`` sums the six slots on
load and must equal ``qr_reduce_scatter`` of that bf16 tensor bitwise, for any row count (the
reduced tensor may hold fewer than world_size shards of data), deterministically, in any order with
the other QuickReduce collectives, and under per-rank skew.
"""

import random
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

H = 7168
TOPK = 6


def make_slots(rank, rows, seed):
    """MXFP8 per-slot rows [rows * 6, H + H / 8]: finite e4m3 codes, e8m0 scales 2^-9 .. 2^3."""
    g = torch.Generator(device="cuda").manual_seed(seed * 7777 + rank)
    codes = torch.randint(
        0, 256, (rows * TOPK, H), device="cuda", generator=g, dtype=torch.int32
    )
    codes = codes.to(torch.uint8)
    nan = (codes & 0x7F) == 0x7F
    codes[nan] = codes[nan] & 0xF0  # e4m3fn NaN (S.1111.111) -> a finite code
    scales = torch.randint(
        127 - 9,
        127 + 4,
        (rows * TOPK, H // 8),
        device="cuda",
        generator=g,
        dtype=torch.int32,
    ).to(torch.uint8)
    return torch.cat([codes, scales], dim=1).contiguous()


def opus_reduce(slots):
    from aiter.ops.opus.moe_stage2_a8w4 import (
        opus_moe_stage2_reduce_token_slot_route_output_fwd,
    )

    return opus_moe_stage2_reduce_token_slot_route_output_fwd(slots, topk=TOPK)


def _worker(rank, world_size, port):
    import aiter as ops
    from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
    from aiter.ops.quick_all_gather import qr_all_gather
    from aiter.ops.quick_reduce_scatter import (
        qr_reduce_scatter,
        qr_reduce_scatter_route,
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=300),
    )
    group = dist.group.WORLD
    ws = world_size
    ptr = ops.init_custom_qr(rank, ws, None)
    qr_exchange_handles(ptr, ws, group)

    def shard_of(rows):
        # each rank owns a whole number of rows; the last ranks may own padding only
        return (rows + ws - 1) // ws * H

    def route_rs(slots, rows):
        out = torch.empty(shard_of(rows), dtype=torch.bfloat16, device=device)
        qr_reduce_scatter_route(ptr, slots, out, rows, TOPK, H)
        return out

    def dense_rs(x, rows):
        out = torch.empty(shard_of(rows), dtype=torch.bfloat16, device=device)
        qr_reduce_scatter(ptr, x, out, True)
        return out

    # 1. route == dense reduce-scatter of the OPUS slot reduce, bitwise; ragged rows; deterministic
    sets = []
    for i, rows in enumerate((1, 7, 8, 9, 1024, 1031, 2049, 4096, 11932, 16384)):
        slots = make_slots(rank, rows, 10 + i)
        routed = opus_reduce(slots)
        assert routed.shape == (rows, H) and routed.dtype == torch.bfloat16
        ref = dense_rs(routed.reshape(-1), rows)
        got = route_rs(slots, rows)
        got2 = route_rs(slots, rows)
        assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), (
            "route != dense",
            rows,
        )
        assert torch.equal(got, got2), ("not deterministic", rows)
        assert torch.isfinite(got.float()).all(), rows
        # the last ranks' padding rows are zero
        lo = rank * (shard_of(rows) // H)
        pad_from = max(rows - lo, 0)
        assert bool((got.view(-1, H)[pad_from:] == 0).all()), (
            "padding rows not zero",
            rows,
        )
        sets.append((rows, slots, routed, ref))

    # 2. mixed with the other collectives under per-rank skew, bitwise against the references
    rnd = random.Random(7)
    skew = random.Random(rank)
    gather_in = torch.randn(2048 * 8, dtype=torch.float16, device=device)
    gather_ref = torch.empty(2048 * 8 * ws, dtype=torch.float16, device=device)
    qr_all_gather(ptr, gather_in, gather_ref, True)
    for it in range(300):
        rows, slots, routed, ref = sets[rnd.randrange(len(sets))]
        if skew.random() < 0.5:
            torch.cuda._sleep(skew.randrange(1, 400) * 5_000)
        kind = rnd.choice(("route", "dense", "gather"))
        if kind == "route":
            assert torch.equal(route_rs(slots, rows), ref), ("route", it, rows)
        elif kind == "dense":
            assert torch.equal(dense_rs(routed.reshape(-1), rows), ref), (
                "dense",
                it,
                rows,
            )
        else:
            out = torch.empty_like(gather_ref)
            qr_all_gather(ptr, gather_in, out, True)
            assert torch.equal(out, gather_ref), ("gather", it)
    torch.cuda.synchronize()
    dist.barrier(group)

    # 3. invalid arguments are rejected and leave the communicator usable
    rows, slots, routed, ref = sets[4]
    out = torch.empty(shard_of(rows), dtype=torch.bfloat16, device=device)

    def rejects(fn, what):
        try:
            fn()
        except Exception:
            return
        raise AssertionError(f"accepted: {what}")

    rejects(lambda: qr_reduce_scatter_route(ptr, slots, out, rows, 8, H), "topk 8")
    rejects(
        lambda: qr_reduce_scatter_route(ptr, slots, out, rows, TOPK, 7100),
        "hidden % 128",
    )
    rejects(
        lambda: qr_reduce_scatter_route(ptr, slots, out, rows + 1, TOPK, H),
        "route size",
    )
    rejects(
        lambda: qr_reduce_scatter_route(
            ptr, slots.view(torch.int8), out, rows, TOPK, H
        ),
        "dtype",
    )
    rejects(
        lambda: qr_reduce_scatter_route(ptr, slots, out.float(), rows, TOPK, H),
        "output dtype",
    )
    rejects(
        lambda: qr_reduce_scatter_route(
            ptr, slots, out[: H * (rows // ws - 1)], rows, TOPK, H
        ),
        "output too small for rows",
    )
    rejects(lambda: qr_reduce_scatter_route(ptr, slots, out, 0, TOPK, H), "zero rows")
    assert torch.equal(
        route_rs(slots, rows), ref
    ), "communicator unusable after rejects"

    torch.cuda.synchronize()
    dist.barrier(group)
    ops.qr_destroy(ptr)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [8])
def test_quick_reduce_scatter_route(world_size):
    if torch.version.hip is None or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs with peer access")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx950":
        pytest.skip("the route source needs gfx950 (OPUS MXFP8 stage 2)")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(8, port), nprocs=8, join=True)
    print("quick reduce-scatter route, 8 ranks: PASS")
