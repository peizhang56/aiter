# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""QuickReduce reduce-scatter / all-gather against qr_all_reduce, on 2, 4 and 8 ranks.

The three collectives share one QuickReduce communicator (IPC buffer, flags, flag colours), so the
tests below exercise them on a single handle, back to back, the way a serving stack does:

* reduce-scatter equals the exact sum within the INT4 codec error, and a ragged input (fewer than
  world_size shards of data) equals the same call on an explicitly zero-padded input, bitwise;
* reduce-scatter followed by all-gather equals qr_all_reduce when the shard is a multiple of the
  codec's 32-element group: bitwise for fp16; for bf16 (computed in fp16 by ``cast_bf2half``) the
  reduced shard is rounded to bf16 between the two, so it matches to bf16 precision;
* a randomized mix of reduce-scatter, all-gather and all-reduce under random per-rank skew
  reproduces the reference of every call bitwise (a missing consumed barrier shows up here as a
  wrong value or a hang);
* CUDA graph capture and replay;
* invalid arguments are rejected.
"""

import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

CODEC_REL_L2 = 0.12  # INT4 QuickReduce, measured 0.086 at 8 ranks
INT4 = 3  # QuickReduceRegime.INT4


def _inputs(case_seed, world_size, n, dtype, device):
    gen = torch.Generator(device=device).manual_seed(case_seed)
    return [
        torch.randn(n, dtype=dtype, device=device, generator=gen)
        for _ in range(world_size)
    ]


def _rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _worker(rank, world_size, port):
    import aiter as ops
    from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
    from aiter.ops.quick_all_gather import qr_all_gather
    from aiter.ops.quick_reduce_scatter import qr_reduce_scatter

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
    ptr = ops.init_custom_qr(rank, world_size, None)
    qr_exchange_handles(ptr, world_size, group)
    ws = world_size

    def rs(x, shard, dtype):
        out = torch.empty(shard, dtype=dtype, device=device)
        qr_reduce_scatter(ptr, x, out, True)
        return out

    def ag(s, dtype):
        out = torch.empty(s.numel() * ws, dtype=dtype, device=device)
        qr_all_gather(ptr, s, out, True)
        return out

    def ar(x):
        out = torch.empty_like(x)
        ops.qr_all_reduce(ptr, x, out, INT4, True)
        return out

    # Shard sizes in elements (all multiples of 8 = 16 bytes): a few atoms, a ragged last tile, and
    # enough tiles for several grid-stride iterations (kMaxNumBlocks = 1216 blocks).
    shards = [2048, 2056, 8 * 1000, 3 * 2048 * 5, 2048 * 1216 * 2 + 1024]
    for dtype in (torch.float16, torch.bfloat16):
        for si, shard in enumerate(shards):
            n = shard * ws
            xs = _inputs(10 + si, ws, n, dtype, device)
            x = xs[rank]
            exact = sum(t.float() for t in xs)
            lo, hi = rank * shard, (rank + 1) * shard

            # 1. reduce-scatter vs the exact sum, deterministic
            r1 = rs(x, shard, dtype)
            r2 = rs(x, shard, dtype)
            assert torch.equal(r1, r2), "reduce-scatter is not deterministic"
            err = _rel_l2(r1, exact[lo:hi])
            assert err < CODEC_REL_L2, (dtype, shard, err)

            # 2. ragged input == explicitly zero-padded input (bitwise): only a prefix is valid
            for valid in (n - 8 * (ws - 1) - 8, n - shard + 8, 8 * ws):
                if not 0 < valid <= n:
                    continue
                short = x[:valid].contiguous()
                padded = torch.zeros(n, dtype=dtype, device=device)
                padded[:valid] = short
                assert torch.equal(rs(short, shard, dtype), rs(padded, shard, dtype)), (
                    dtype,
                    shard,
                    valid,
                )

            # 3. reduce-scatter -> all-gather == all-reduce
            # The codec quantizes groups of 32 consecutive elements. The all-reduce groups them from
            # element 0 and the shard layout from r * S, so the two agree bitwise only when S is a
            # multiple of 32 (fp16). For bf16 the reduced shard is rounded to bf16 between the
            # reduce-scatter and the all-gather, and for S % 32 != 0 the group boundaries differ:
            # there both results are valid INT4 sums, so the check is accuracy parity with the
            # all-reduce against the exact sum.
            rsag = ag(r1, dtype)
            ref = ar(x)
            if shard % 32 == 0 and dtype == torch.float16:
                assert torch.equal(rsag, ref), (shard, "fp16 RS+AG != all-reduce")
            else:
                e_rsag = _rel_l2(rsag, exact)
                e_ar = _rel_l2(ref, exact)
                assert e_rsag < 1.1 * e_ar, (dtype, shard, e_rsag, e_ar)

            # 4. all-gather alone: every shard is its source rank's quantized input
            g = ag(x[:shard].contiguous(), dtype)
            gs = [t[:shard].float() for t in xs]
            for r in range(ws):
                assert _rel_l2(g[r * shard : (r + 1) * shard], gs[r]) < 0.1, (
                    dtype,
                    shard,
                    r,
                )

    # 5. random mix of the three collectives under skew, bitwise vs the first run of each
    dtype = torch.float16
    shard = 2048 * 40
    cases = []
    for k in range(3):
        xs = _inputs(100 + k, ws, shard * ws, dtype, device)
        cases.append(xs[rank])
    ref_rs = [rs(c, shard, dtype) for c in cases]
    ref_ar = [ar(c) for c in cases]
    ref_ag = [ag(c[:shard].contiguous(), dtype) for c in cases]
    torch.cuda.synchronize()
    dist.barrier(group)
    import random

    rnd = random.Random(1234)  # the same op sequence on every rank
    skew = random.Random(rank)  # a different delay on every rank
    for it in range(400):
        k = rnd.randrange(3)
        op = rnd.choice(("rs", "ag", "ar"))
        if skew.random() < 0.5:
            torch.cuda._sleep(skew.randrange(1, 400) * 5_000)
        if op == "rs":
            assert torch.equal(rs(cases[k], shard, dtype), ref_rs[k]), ("rs", it)
        elif op == "ag":
            assert torch.equal(ag(cases[k][:shard].contiguous(), dtype), ref_ag[k]), (
                "ag",
                it,
            )
        else:
            assert torch.equal(ar(cases[k]), ref_ar[k]), ("ar", it)
    torch.cuda.synchronize()
    dist.barrier(group)

    # 6. CUDA graph: reduce-scatter + all-gather replayed with new inputs
    x = cases[0].clone()
    out_s = torch.empty(shard, dtype=dtype, device=device)
    out_g = torch.empty(shard * ws, dtype=dtype, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm up outside the capture
        qr_reduce_scatter(ptr, x, out_s, True)
        qr_all_gather(ptr, out_s, out_g, True)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier(group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qr_reduce_scatter(ptr, x, out_s, True)
        qr_all_gather(ptr, out_s, out_g, True)
    for gen in range(3):
        x.copy_(cases[(gen + 1) % 3])
        for _ in range(4):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out_g, ar(x)), ("graph", gen)

    # 7. invalid arguments are rejected (and leave the communicator usable)
    ok = torch.empty(shard, dtype=dtype, device=device)

    def rejects(fn, what):
        try:
            fn()
        except Exception:
            return
        raise AssertionError(f"accepted: {what}")

    xin = cases[0]
    rejects(lambda: qr_reduce_scatter(ptr, xin.float(), ok.float(), True), "fp32")
    rejects(lambda: qr_reduce_scatter(ptr, xin, ok.bfloat16(), True), "dtype mismatch")
    rejects(lambda: qr_reduce_scatter(ptr, xin[::2], ok, True), "non-contiguous input")
    rejects(
        lambda: qr_reduce_scatter(ptr, xin, ok[: shard - 8], True), "input too large"
    )
    rejects(lambda: qr_reduce_scatter(ptr, xin[:0], ok, True), "empty input")
    rejects(
        lambda: qr_reduce_scatter(ptr, xin[: shard * ws - 1], ok, True),
        "ragged by 1 element",
    )
    rejects(
        lambda: qr_reduce_scatter(ptr, xin[:12], ok[:4], True),
        "shard not a multiple of 16 B",
    )
    rejects(
        lambda: qr_all_gather(
            ptr, ok, torch.empty(shard, dtype=dtype, device=device), True
        ),
        "gather output size",
    )
    rejects(
        lambda: qr_all_gather(
            ptr, ok[:5], torch.empty(5 * ws, dtype=dtype, device=device), True
        ),
        "gather shard not a multiple of 16 B",
    )
    assert torch.equal(
        rs(xin, shard, dtype), ref_rs[0]
    ), "communicator unusable after rejects"

    torch.cuda.synchronize()
    dist.barrier(group)
    ops.qr_destroy(ptr)
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_quick_reduce_scatter_gather(world_size):
    if torch.version.hip is None or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs with peer access")
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip("QuickReduce is validated on gfx942/gfx950")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)


if __name__ == "__main__":
    import sys

    ws = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(ws, port), nprocs=ws, join=True)
    print(f"quick reduce-scatter / all-gather, {ws} ranks: PASS")
