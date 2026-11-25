# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""qr_reduce_scatter_route: reduce-scatter from the OPUS MoE per-slot rows.

The OPUS stage 2 can return its routed term as per-slot MXFP8 rows
``[rows * topk, hidden + hidden / 8]``. ``qr_reduce_scatter_route`` sums the
slots on load and must equal ``qr_reduce_scatter`` of
``opus_moe_stage2_reduce_token_slot_route_output_fwd``'s bf16 output bitwise,
for any row count. gfx950 only. Ranks return their results instead of
asserting, so a failure cannot strand a peer in a spin-wait.
"""

import argparse
import logging
import random

import pandas as pd
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import aiter
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.opus.moe_stage2_a8w4 import (
    opus_moe_stage2_reduce_token_slot_route_output_fwd as opus_reduce,
)
from aiter.ops.quick_reduce_scatter import qr_reduce_scatter, qr_reduce_scatter_route

logger = logging.getLogger("aiter")

INT4 = 3  # QuickReduceRegime.INT4
# (topk, hidden): DeepSeek-V4 (the unrolled topk-6 path), the runtime-topk
# path, and a hidden size that is a multiple of 64 but not of 128.
SHAPES = [(6, 7168), (8, 7168), (6, 2112)]


def _slots(rows, topk, hidden, seed):
    """Per-slot MXFP8 rows: finite e4m3 codes, e8m0 scales 2^-9 .. 2^3."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    codes = torch.randint(0, 256, (rows * topk, hidden), device="cuda", generator=g)
    codes = codes.to(torch.uint8)
    nan = (codes & 0x7F) == 0x7F
    codes[nan] &= 0xF0  # e4m3fn NaN (S.1111.111) -> a finite code
    scales = torch.randint(
        127 - 9, 127 + 4, (rows * topk, hidden // 8), device="cuda", generator=g
    ).to(torch.uint8)
    return torch.cat([codes, scales], dim=1).contiguous()


def _worker(rank, ws, init_method, tokens):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=ws)
    group = dist.group.WORLD
    ptr = aiter.init_custom_qr(rank, ws, None)
    qr_exchange_handles(ptr, ws, group)
    res = {"errors": [], "perf": []}

    def fail(msg):
        res["errors"].append(f"rank {rank}: {msg}")

    try:
        _check(rank, ws, device, group, ptr, fail)
        for t in tokens:
            res["perf"].append(_perf(rank, ws, device, ptr, t))
    finally:
        torch.cuda.synchronize()
        dist.barrier(group)
        aiter.qr_destroy(ptr)
        dist.destroy_process_group()
    return res


def _check(rank, ws, device, group, ptr, fail):
    def route_rs(slots, rows, topk, hidden):
        # NaN-filled, so an element the kernel does not write fails the check.
        shard = -(-rows // ws) * hidden
        out = torch.full((shard,), float("nan"), dtype=torch.bfloat16, device=device)
        qr_reduce_scatter_route(ptr, slots, out, rows, topk, hidden)
        return out

    def dense_rs(x, shard):
        out = torch.empty(shard, dtype=torch.bfloat16, device=device)
        qr_reduce_scatter(ptr, x, out, True)
        return out

    # Row counts: fewer rows than ranks, around a multiple of the world size,
    # and enough for several grid-stride iterations.
    cases = []
    for topk, hidden in SHAPES:
        for i, rows in enumerate((1, ws - 1, ws + 1, 1031, 4096, 16384)):
            tag = f"{rows=} {topk=} {hidden=}"
            slots = _slots(rows, topk, hidden, 1000 * rank + i)
            dense = opus_reduce(slots, topk=topk)
            shard = -(-rows // ws) * hidden
            ref = dense_rs(dense.view(-1), shard)
            got = route_rs(slots, rows, topk, hidden)
            if not torch.equal(got.view(torch.int16), ref.view(torch.int16)):
                fail(f"{tag}: route != dense reduce-scatter of the OPUS reduce")
            if not torch.equal(got, route_rs(slots, rows, topk, hidden)):
                fail(f"{tag}: not deterministic")
            if rows in (ws + 1, 1031):
                cases.append((slots, rows, topk, hidden, ref))

    # Zero rows: the whole shard is padding.
    empty = torch.empty(0, 7168 + 7168 // 8, dtype=torch.uint8, device=device)
    if not bool((route_rs(empty, 0, 6, 7168) == 0).all()):
        fail("rows=0: shard not zero")

    # Interleave with the dense reduce-scatter and the all-reduce under a
    # random per-rank delay: every result must match its reference bitwise.
    x = torch.randn(2048 * 8 * ws, dtype=torch.bfloat16, device=device)
    x_rs = dense_rs(x, 2048 * 8)
    x_ar = torch.empty_like(x)
    aiter.qr_all_reduce(ptr, x, x_ar, INT4, True)
    torch.cuda.synchronize()
    dist.barrier(group)
    order = random.Random(7)  # the same op sequence on every rank
    skew = random.Random(rank)  # a different delay on every rank
    for it in range(150):
        if skew.random() < 0.5:
            torch.cuda._sleep(skew.randrange(1, 400) * 5_000)
        op = order.randrange(3)
        if op == 0:
            slots, rows, topk, hidden, ref = cases[order.randrange(len(cases))]
            if not torch.equal(route_rs(slots, rows, topk, hidden), ref):
                fail(f"interleaved route reduce-scatter {it}: mismatch")
        elif op == 1:
            if not torch.equal(dense_rs(x, 2048 * 8), x_rs):
                fail(f"interleaved reduce-scatter {it}: mismatch")
        else:
            out = torch.empty_like(x)
            aiter.qr_all_reduce(ptr, x, out, INT4, True)
            if not torch.equal(out, x_ar):
                fail(f"interleaved all-reduce {it}: mismatch")
    torch.cuda.synchronize()
    dist.barrier(group)

    # CUDA graph: capture once, replay with new route rows.
    slots, rows, topk, hidden, _ = cases[0]
    src = slots.clone()
    out = torch.empty(-(-rows // ws) * hidden, dtype=torch.bfloat16, device=device)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qr_reduce_scatter_route(ptr, src, out, rows, topk, hidden)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qr_reduce_scatter_route(ptr, src, out, rows, topk, hidden)
    for k in range(3):
        src.copy_(_slots(rows, topk, hidden, 77 + 10 * rank + k))
        graph.replay()
        if not torch.equal(out, route_rs(src, rows, topk, hidden)):
            fail(f"graph replay {k}: mismatch")
    torch.cuda.synchronize()
    dist.barrier(group)

    # Invalid arguments are rejected on the host, on every rank alike.
    ok = torch.empty(-(-rows // ws) * hidden, dtype=torch.bfloat16, device=device)
    rejected = {
        "bf16 route": (slots.view(torch.bfloat16), ok, rows, topk, hidden),
        "fp16 output": (slots, ok.half(), rows, topk, hidden),
        "topk does not match the route size": (slots, ok, rows, topk + 1, hidden),
        # A consistent [6, 96 + 12] route: only the hidden check can fail it.
        "hidden not a multiple of 64": (
            torch.zeros(6, 96 + 12, dtype=torch.uint8, device=device),
            torch.empty(96, dtype=torch.bfloat16, device=device),
            1,
            6,
            96,
        ),
        "rows past world_size shards": (slots, ok[:-hidden], rows, topk, hidden),
        "topk 0": (slots, ok, rows, 0, hidden),
    }
    for what, args in rejected.items():
        try:
            with pytest.raises(ValueError):
                qr_reduce_scatter_route(ptr, *args)
        except pytest.fail.Exception:
            fail(f"accepted: {what}")


def _perf(rank, ws, device, ptr, tokens, topk=6, hidden=7168):
    """The fused route reduce-scatter against the OPUS reduce followed by
    qr_reduce_scatter, on the DeepSeek-V4 shape."""
    slots = _slots(tokens, topk, hidden, rank)
    dense = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=device)
    out = torch.empty(-(-tokens // ws) * hidden, dtype=torch.bfloat16, device=device)

    def unfused():
        opus_reduce(slots, out=dense, topk=topk)
        qr_reduce_scatter(ptr, dense.view(-1), out, True)

    return {
        "tokens": tokens,
        "reduce+qr_rs": _graph_us(unfused),
        "route_rs": _graph_us(
            lambda: qr_reduce_scatter_route(ptr, slots, out, tokens, topk, hidden)
        ),
    }


def _graph_us(op, calls=20, replays=10):
    """Average device time of ``op`` in us from a CUDA graph of ``calls``
    back-to-back calls, so neither launch overhead nor host jitter between
    ranks is measured."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            op()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            op()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()  # start the timed replays aligned across ranks
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (calls * replays)


def _launch(ws, tokens):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = mp.get_context("spawn")
    with ctx.Pool(ws) as pool:
        jobs = [
            pool.apply_async(_worker, (r, ws, init_method, tokens)) for r in range(ws)
        ]
        results = [j.get(timeout=1800) for j in jobs]
    errors = [e for r in results for e in r["errors"]]
    assert not errors, "\n".join(errors)
    rows = []
    for i, p in enumerate(results[0]["perf"]):
        row = {"world_size": ws, "tokens": p["tokens"]}
        for op in ("reduce+qr_rs", "route_rs"):
            row[f"{op}_us"] = round(max(r["perf"][i][op] for r in results), 1)
        row["speedup"] = round(row["reduce+qr_rs_us"] / row["route_rs_us"], 2)
        rows.append(row)
    return rows


def _skip_reason(ws):
    if torch.cuda.device_count() < ws:
        return f"needs {ws} GPUs, found {torch.cuda.device_count()}"
    if get_gfx_runtime() != "gfx950":
        return "the route input (OCP fp8 OPUS stage 2 output) needs gfx950"
    return None


@pytest.mark.parametrize("world_size", [4, 8])
def test_quick_reduce_scatter_route(world_size):
    reason = _skip_reason(world_size)
    if reason:
        pytest.skip(reason)
    _launch(world_size, [2048])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-w", "--world-size", type=int, nargs="+", default=[4, 8])
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="+", default=[64, 512, 2048, 8192]
    )
    args = parser.parse_args()
    rows = []
    for ws in args.world_size:
        reason = _skip_reason(ws)
        if reason:
            logger.warning("skipping world_size %d: %s", ws, reason)
            continue
        rows += _launch(ws, args.tokens)
    if rows:
        logger.info(
            "quick reduce-scatter route summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
