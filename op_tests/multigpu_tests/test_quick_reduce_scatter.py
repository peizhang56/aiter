# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import multiprocessing as mp
import random

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.device_communicators.quick_all_reduce import qr_exchange_handles
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.quick_reduce_scatter import qr_reduce_scatter
from aiter.test_common import run_perftest

SUPPORTED_GFX = ["gfx950"]
INT4 = 3  # QuickReduceRegime.INT4
# Max relative L2 error over 64-element blocks vs the fp32 sum (measured <= 0.17).
MAX_ERR = 0.3


def make_inputs(seed, ws, tokens, hidden, dtype):
    """Every rank's input; the magnitude changes every 256 elements."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    idx = torch.arange(tokens * hidden, device="cuda").view(tokens, hidden)
    scale = torch.exp2((idx // 256 % 9 - 4).float())
    return [
        (torch.randn(tokens, hidden, device="cuda", generator=gen) * scale).to(dtype)
        for _ in range(ws)
    ]


def block_err(out, ref, block=64):
    d = (out.float() - ref).flatten()
    r = ref.flatten()
    pad = -r.numel() % block
    d = torch.nn.functional.pad(d, (0, pad)).view(-1, block).norm(dim=1)
    r = torch.nn.functional.pad(r, (0, pad)).view(-1, block).norm(dim=1)
    return (d / r.clamp_min(1e-6)).max().item()


def run_case(rank, ws, ptr, nccl, tokens, hidden, dtype):
    """Rows of a [tokens, hidden] tensor split across ranks, as in sequence
    parallelism; tokens need not be a multiple of ws."""
    xs = make_inputs(tokens, ws, tokens, hidden, dtype)
    x = xs[rank]
    rows = -(-tokens // ws)
    padded = torch.zeros(rows * ws, hidden, dtype=dtype, device="cuda")
    padded[:tokens] = x
    ref = torch.zeros(rows * ws, hidden, device="cuda")
    ref[:tokens] = sum(t.float() for t in xs)
    ref = ref[rank * rows : (rank + 1) * rows]

    out = torch.full((rows, hidden), float("nan"), dtype=dtype, device="cuda")
    _, qr_us = run_perftest(
        lambda: qr_reduce_scatter(ptr, x, out, True), use_cuda_event=True
    )
    out_padded = torch.empty_like(out)
    qr_reduce_scatter(ptr, padded, out_padded, True)
    out_rccl = torch.empty_like(out)
    _, rccl_us = run_perftest(
        lambda: dist.reduce_scatter_tensor(out_rccl, padded, group=nccl),
        use_cuda_event=True,
    )
    return {
        "qr us": qr_us,
        "qr err": block_err(out, ref),
        "ragged == padded": torch.equal(out, out_padded),
        "rccl us": rccl_us,
        "rccl err": block_err(out_rccl, ref),
    }


def check_comm(rank, ws, ptr, dtype):
    """Reuse of the communicator: interleaving with qr_all_reduce under per-rank
    skew, CUDA graph replay and rejected arguments. Returns failures."""
    fails = []
    rows, hidden = 1024, 7168
    xs = [make_inputs(k, ws, rows * ws, hidden, dtype)[rank] for k in range(3)]

    def rs(x):
        out = torch.empty(rows, hidden, dtype=dtype, device="cuda")
        qr_reduce_scatter(ptr, x, out, True)
        return out

    def ar(x):
        out = torch.empty_like(x)
        aiter.qr_all_reduce(ptr, x, out, INT4, True)
        return out

    ref_rs, ref_ar = [rs(x) for x in xs], [ar(x) for x in xs]
    torch.cuda.synchronize()
    dist.barrier()
    order, skew = random.Random(0), random.Random(rank)
    for it in range(100):
        k = order.randrange(3)
        if skew.random() < 0.5:
            torch.cuda._sleep(skew.randrange(1, 400) * 5000)
        if order.random() < 0.5:
            if not torch.equal(rs(xs[k]), ref_rs[k]):
                fails.append(f"reduce-scatter {it} after mixed calls")
        elif not torch.equal(ar(xs[k]), ref_ar[k]):
            fails.append(f"all-reduce {it} after mixed calls")
    torch.cuda.synchronize()
    dist.barrier()

    x, out = xs[0].clone(), torch.empty(rows, hidden, dtype=dtype, device="cuda")
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        qr_reduce_scatter(ptr, x, out, True)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        qr_reduce_scatter(ptr, x, out, True)
    for k in (1, 2):
        x.copy_(xs[k])
        graph.replay()
        if not torch.equal(out, ref_rs[k]):
            fails.append(f"graph replay {k}")
    torch.cuda.synchronize()
    dist.barrier()

    x, ok = xs[0], torch.empty(rows, hidden, dtype=dtype, device="cuda")
    for what, inp, outp in (
        ("fp32", x.float(), ok.float()),
        ("non-contiguous", x.t(), ok),
        ("input > ws shards", x, ok[:-1]),
        ("not 16-byte aligned", x.view(-1)[:-1], ok),
    ):
        try:
            qr_reduce_scatter(ptr, inp, outp, True)
            fails.append(f"accepted {what}")
        except ValueError:
            pass
    return fails


def worker(rank, ws, init_method, tokens_list, hidden, dtype):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=ws)
    nccl = dist.new_group(backend="nccl")
    ptr = aiter.init_custom_qr(rank, ws, None)
    qr_exchange_handles(ptr, ws, dist.group.WORLD)
    try:
        cases = [run_case(rank, ws, ptr, nccl, t, hidden, dtype) for t in tokens_list]
        return cases, check_comm(rank, ws, ptr, dtype)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        aiter.qr_destroy(ptr)
        dist.destroy_process_group()


def test_quick_reduce_scatter(ws, tokens_list, hidden, dtype):
    """One row per token count; ranks return results so a failure on one rank
    cannot leave the others waiting."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    args = (ws, init_method, tokens_list, hidden, dtype)
    with mp.get_context("spawn").Pool(ws) as pool:
        jobs = [pool.apply_async(worker, (r, *args)) for r in range(ws)]
        results = [j.get(timeout=1800) for j in jobs]
    fails = [f"rank {r}: {f}" for r, (_, fs) in enumerate(results) for f in fs]
    assert not fails, fails
    rows = []
    for i, tokens in enumerate(tokens_list):
        per_rank = [cases[i] for cases, _ in results]
        nbytes = tokens * hidden * dtype.itemsize
        row = {"ws": ws, "tokens": tokens, "hidden": hidden, "dtype": dtype}
        row["gfx"] = get_gfx_runtime()
        for name in ("qr", "rccl"):
            us = max(c[f"{name} us"] for c in per_rank)
            row[f"{name} us"] = us
            row[f"{name} TB/s"] = nbytes / us / 1e6
            row[f"{name} err"] = max(c[f"{name} err"] for c in per_rank)
        assert row["qr err"] < MAX_ERR, row
        assert all(c["ragged == padded"] for c in per_rank), row
        rows.append(row)
    return rows


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("qr_reduce_scatter: skipping on %s", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(description="qr_reduce_scatter test")
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,fp16"
    )
    parser.add_argument("-w", "--world-size", type=int, nargs="*", default=[4, 8])
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=[64, 1023, 4096, 16385]
    )
    parser.add_argument("-n", "--hidden", type=int, default=7168)
    args = parser.parse_args()

    rows = []
    for ws, dtype in itertools.product(args.world_size, args.dtype):
        if torch.cuda.device_count() < ws:
            aiter.logger.warning("qr_reduce_scatter: skipping %d ranks", ws)
            continue
        rows += test_quick_reduce_scatter(ws, args.tokens, args.hidden, dtype)
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "qr_reduce_scatter summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
