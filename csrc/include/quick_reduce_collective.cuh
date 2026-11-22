#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared pieces of the QuickReduce single-phase collectives (quick_reduce_scatter.cu,
// quick_all_gather.cu). Both run on an initialized QuickReduce communicator (DeviceComms): the
// IPC buffer, the per-block flag colours and the flag layout are the all-reduce's own, so the three
// collectives can be mixed on one communicator in stream order.
//
// Communicator layout used here (see DeviceComms::init):
//   flags set 0 [kMaxNumBlocks][W]   arrival: "rank r's data for this block has landed"
//   flags set 1 [kMaxNumBlocks][W]   consumed: "rank r has read this block's data"
//   data        [kMaxNumBlocks][Codec::kTransmittedTileSize]
// A block slot is reused by the next grid-stride iteration, and the next launch reuses all slots.
// The all-reduce is protected by its second phase; a single-phase collective has no second phase,
// so it ends every iteration with the consumed barrier: no rank writes a slot again before every
// rank has read it, within a launch and across launches.
#include <limits>
#include <optional>

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "quick_all_reduce.cuh"

namespace aiter {

__quickreduce_device_inline__ int32x4_t qr_bf16x8_to_half(int32x4_t v)
{
    const nv_bfloat162* b = reinterpret_cast<const nv_bfloat162*>(&v);
    half2 h[4];
#pragma unroll
    for(int j = 0; j < 4; ++j)
        h[j] = __float22half2_rn(__bfloat1622float2(b[j]));
    return *reinterpret_cast<const int32x4_t*>(h);
}

__quickreduce_device_inline__ int32x4_t qr_half_to_bf16x8(int32x4_t v)
{
    const half2* h = reinterpret_cast<const half2*>(&v);
    nv_bfloat162 b[4];
#pragma unroll
    for(int j = 0; j < 4; ++j)
        b[j] = __float22bfloat162_rn(__half22float2(h[j]));
    return *reinterpret_cast<const int32x4_t*>(b);
}

// Slot reuse protocol of a single-phase collective (see the layout at the top). After reading its
// slot a block publishes "consumed" (colour c) and goes on to load its next tile; before it writes
// the slot again (iteration c + 1) it waits for every rank's consumed(c), so the wait hides behind the
// load latency. The last iteration of a launch waits at its end instead: the next kernel on the
// communicator, an all-reduce included, writes the slot without checking.
__quickreduce_device_inline__ void qr_wait_consumed(
    uint8_t** buffers, uint32_t consumed, int rank, int thread, int world_size, uint32_t color)
{
    if(thread == 0)
    {
        auto done = reinterpret_cast<uint32_t*>(buffers[rank] + consumed);
        for(int r = 0; r < world_size; ++r)
            wait_sync_flag(done + r, color);
    }
    __syncthreads();
}

// Publish `color` to every peer's flag slot `flag_off + rank * 4` (thread t < W targets rank t).
// The stores this wave issued are drained first (vmcnt(0)); a relaxed system-scope flag store then
// orders after them, which is all the protocol needs: the data is written to the peers' uncached
// IPC buffer, so it is visible once acknowledged. A release store would add a full L2 writeback
// on gfx950 for no gain. Callers use it for the arrival flag (after the sends) and for the consumed
// flag (after the last read of the slot).
__quickreduce_device_inline__ void qr_publish_flag(
    uint8_t** buffers, uint32_t flag_off, int rank, int thread, int world_size, uint32_t color)
{
    __builtin_amdgcn_s_waitcnt(0x0F70);
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
    __syncthreads();
    if(thread < world_size)
        __hip_atomic_store(reinterpret_cast<uint32_t*>(buffers[thread] + flag_off + rank * sizeof(uint32_t)),
                           color,
                           __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}

// Per-collective kernel arguments beyond the common ones (none for the dense collectives).
struct QuickNoArgs
{
};

// Grid-stride driver shared by the collectives. `Kernel::run` handles one block (atom tile) and
// uses one flag colour per iteration. The colour lives in device memory and is advanced on the
// device, so a CUDA-graph replay uses a fresh colour (same scheme as allreduce_prototype_twoshot).
// in_bytes: bytes of `input` that are valid; reads past it return zero (see QuickReduceScatter).
template <typename Kernel, typename T, typename Args = QuickNoArgs>
__global__ __quickreduce_launch_bounds_two_shot__ void
quick_collective_kernel(T const* input,
                        T* output,
                        uint32_t shard_bytes,
                        uint32_t in_bytes,
                        uint32_t num_blocks,
                        int rank,
                        uint8_t** buffers,
                        uint32_t data_offset,
                        uint32_t* d_flag_color,
                        Args args = Args())
{
    int block           = blockIdx.x;
    uint32_t flag_color = d_flag_color[blockIdx.x];
    while(block < num_blocks)
    {
        const bool first = block == static_cast<int>(blockIdx.x);
        const bool last  = block + static_cast<int>(gridDim.x) >= static_cast<int>(num_blocks);
        Kernel::run(input, output, shard_bytes, in_bytes, block, rank, buffers, data_offset,
                    flag_color, first, last, args);
        block += gridDim.x;
        flag_color++;
    }
    if(threadIdx.x == 0 && threadIdx.y == 0)
        d_flag_color[blockIdx.x] = flag_color;
}

// Host-side checks common to the collectives.
inline DeviceComms* quick_collective_comm(int64_t handle, const char* name)
{
    auto comm = reinterpret_cast<DeviceComms*>(handle);
    if(!comm || !comm->initialized)
        throw std::invalid_argument(std::string(name) + ": uninitialized communicator");
    if(comm->world_size != 2 && comm->world_size != 4 && comm->world_size != 8)
        throw std::invalid_argument(std::string(name) + ": world size must be 2, 4 or 8");
    return comm;
}

inline void quick_collective_check_tensors(const char* name,
                                           const aiter_tensor_t& input,
                                           const aiter_tensor_t& output,
                                           const DeviceComms* comm,
                                           size_t shard_numel)
{
    if(!input.is_gpu() || input.device_id != output.device_id || input.dtype() != output.dtype() ||
       !input.is_contiguous() || !output.is_contiguous())
        throw std::invalid_argument(std::string(name) + ": device/dtype/contiguity mismatch");
    if(input.dtype() != AITER_DTYPE_fp16 && input.dtype() != AITER_DTYPE_bf16)
        throw std::invalid_argument(std::string(name) + ": only fp16/bf16 supported");
    // The buffer loads and stores move 16 bytes per lane.
    if(shard_numel == 0 || shard_numel * input.element_size() % 16 != 0)
        throw std::invalid_argument(std::string(name) +
                                    ": the shard must be a non-empty multiple of 16 bytes");
    // Byte offsets inside the kernels are 32-bit; the communicator's problem size is below 2^31.
    if(shard_numel * comm->world_size * input.element_size() >
       static_cast<size_t>(comm->kMaxProblemSize))
        throw std::invalid_argument(std::string(name) + ": tensor exceeds the communicator's max size");
}

} // namespace aiter
