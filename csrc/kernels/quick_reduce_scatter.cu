// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// quick_all_reduce.cuh requires these standard-library declarations first.
#include <limits>
#include <optional>

#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "quick_all_reduce.cuh"

namespace aiter {

// The first communication phase of QuickReduce, with contiguous rank shards.
// The existing all-reduce assigns a stripe of every tile to each rank instead.
// Reuse its codec and accumulation order, but omit the entire data all-gather
// and the second quantization. A consumption barrier still protects IPC buffer
// reuse by the next grid-stride iteration or CUDA graph replay.
// The input may be shorter than W shards; the missing tail reads as zero.
template <typename T, class Codec, bool cast_bf2half>
struct QuickReduceScatter
{
    static constexpr int kWorldSize      = Codec::kWorldSize;
    static constexpr int kLocalTileBytes = kTileSize / kWorldSize;

    __quickreduce_device_inline__ static void
    publish_flag(uint8_t** buffers, uint32_t flag, int rank, int thread, uint32_t color)
    {
#if defined(__gfx950__)
        // The IPC buffer is uncached: draining vmcnt is enough, and avoids
        // the L2 writeback of a release store.
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        __syncthreads();
        if(thread < kWorldSize)
            __hip_atomic_store(
                reinterpret_cast<uint32_t*>(buffers[thread] + flag + rank * sizeof(uint32_t)),
                color,
                __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_SYSTEM);
#else
        __syncthreads();
        if(thread < kWorldSize)
            set_sync_flag(
                reinterpret_cast<uint32_t*>(buffers[thread] + flag + rank * sizeof(uint32_t)),
                color);
#endif
    }

    __device__ static void run(T const* __restrict__ input,
                               T* __restrict__ output,
                               uint32_t N,
                               int block,
                               int rank,
                               uint8_t** __restrict__ buffers,
                               uint32_t data_offset,
                               uint32_t color,
                               int64_t shard_size)
    {
        int thread = threadIdx.x + threadIdx.y * kWavefront;
        Codec codec(thread, rank);
        // Cache the peer pointers: uint8_t* stores may alias the table.
        uint8_t* bp[kWorldSize];
        for(int r = 0; r < kWorldSize; ++r)
            bp[r] = buffers[r];
        uint32_t shard_bytes = shard_size;
        uint32_t in_bytes    = N * sizeof(T);
        uint32_t data        = data_offset + blockIdx.x * Codec::kTransmittedTileSize;
        uint32_t arrival     = blockIdx.x * kWorldSize * sizeof(uint32_t);
        uint32_t consumed    = data_offset / 2 + arrival;

        int32x4_t values[kWorldSize][Codec::kRankAtoms];
        for(int r = 0; r < kWorldSize; ++r)
        {
            // A separate bounded resource per shard makes padded tail atoms
            // zero, without reading the next rank's token shard.
            uint32_t begin = r * shard_bytes;
            uint32_t valid = in_bytes > begin ? min(in_bytes - begin, shard_bytes) : 0;
            BufferResource src(const_cast<T*>(input) + begin / sizeof(T), valid);
            uint32_t offset = block * kLocalTileBytes + thread * sizeof(int32x4_t);
            for(int i = 0; i < Codec::kRankAtoms; ++i)
            {
                values[r][i] = buffer_load_dwordx4(src.descriptor, offset, 0, 0);
                offset += kAtomStride * sizeof(int32x4_t);
            }
        }

        for(int r = 0; r < kWorldSize; ++r)
        {
            if constexpr(cast_bf2half)
                for(int i = 0; i < Codec::kRankAtoms; ++i)
                {
                    auto bf = reinterpret_cast<const nv_bfloat162*>(&values[r][i]);
                    half2 fp[4];
#pragma unroll
                    for(int j = 0; j < 4; ++j)
                        fp[j] = __float22half2_rn(__bfloat1622float2(bf[j]));
                    values[r][i] = *reinterpret_cast<int32x4_t*>(fp);
                }
            auto dst =
                reinterpret_cast<int32x4_t*>(bp[r] + data + rank * Codec::kRankTransmittedTileSize);
            codec.send(dst, values[r]);
        }

        publish_flag(buffers, arrival, rank, thread, color);

        int32x4_t sum[Codec::kRankAtoms] = {};
        auto recv                        = reinterpret_cast<int32x4_t*>(buffers[rank] + data);
        auto flags                       = reinterpret_cast<uint32_t*>(buffers[rank] + arrival);
        for(int r = 0; r < kWorldSize; ++r)
        {
            if(thread == 0)
                wait_sync_flag(flags + r, color);
            __syncthreads();
            int32x4_t recvd[Codec::kRankAtoms];
            codec.recv(&recv, recvd);
            for(int i = 0; i < Codec::kRankAtoms; ++i)
                packed_assign_add<T>(&sum[i], &recvd[i]);
        }

        // Acknowledge that all threads have consumed the input data. Unlike
        // all-reduce, this phase sends only flags, never reduced payloads.
        publish_flag(buffers, consumed, rank, thread, color);

        BufferResource dst(output, shard_bytes);
        uint32_t offset = block * kLocalTileBytes + thread * sizeof(int32x4_t);
        for(int i = 0; i < Codec::kRankAtoms; ++i)
        {
            if constexpr(cast_bf2half)
            {
                auto fp = reinterpret_cast<const half2*>(&sum[i]);
                nv_bfloat162 bf[4];
#pragma unroll
                for(int j = 0; j < 4; ++j)
                    bf[j] = __float22bfloat162_rn(__half22float2(fp[j]));
                sum[i] = *reinterpret_cast<int32x4_t*>(bf);
            }
            buffer_store_dwordx4(sum[i], dst.descriptor, offset, 0, 0);
            offset += kAtomStride * sizeof(int32x4_t);
        }

        auto done = reinterpret_cast<uint32_t*>(buffers[rank] + consumed);
        if(thread == 0)
            for(int r = 0; r < kWorldSize; ++r)
                wait_sync_flag(done + r, color);
        __syncthreads();
    }
};

template <typename T, int W, bool cast_bf2half>
void launch_quick_rs(DeviceComms* comm,
                     const aiter_tensor_t& input,
                     const aiter_tensor_t& output,
                     hipStream_t stream)
{
    using Kernel         = QuickReduceScatter<T, CodecQ4<T, W>, cast_bf2half>;
    uint32_t N           = input.numel();
    uint32_t shard_bytes = output.numel() * output.element_size();
    uint32_t blocks      = divceil(shard_bytes, kTileSize / W);
    uint32_t grid        = std::min(static_cast<uint32_t>(kMaxNumBlocks), blocks);
    hipLaunchKernelGGL((allreduce_prototype_twoshot<Kernel, T>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       reinterpret_cast<const T*>(input.data_ptr()),
                       reinterpret_cast<T*>(output.data_ptr()),
                       N,
                       blocks,
                       comm->rank,
                       comm->dbuffer_list,
                       comm->data_offset,
                       comm->d_flag_color,
                       shard_bytes);
    HIP_CHECK(hipGetLastError());
}

template <typename T, bool cast_bf2half>
void dispatch_quick_rs(DeviceComms* comm,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       hipStream_t stream)
{
    switch(comm->world_size)
    {
    case 4: launch_quick_rs<T, 4, cast_bf2half>(comm, input, output, stream); break;
    case 8: launch_quick_rs<T, 8, cast_bf2half>(comm, input, output, stream); break;
    default: throw std::invalid_argument("qr_reduce_scatter: requires 4 or 8 ranks");
    }
}

void qr_reduce_scatter(int64_t handle,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       bool cast_bf2half)
{
    auto comm = reinterpret_cast<DeviceComms*>(handle);
    if(!comm || !comm->initialized)
        throw std::invalid_argument("qr_reduce_scatter: uninitialized communicator");
    if(comm->world_size != 4 && comm->world_size != 8)
        throw std::invalid_argument("qr_reduce_scatter: requires 4 or 8 ranks");
    if(!input.is_gpu() || input.device_id != output.device_id || input.dtype() != output.dtype() ||
       !input.is_contiguous() || !output.is_contiguous())
        throw std::invalid_argument("qr_reduce_scatter: device/dtype/contiguity mismatch");
    const size_t full_numel = output.numel() * comm->world_size;
    if(input.numel() > full_numel || output.numel() * output.element_size() % 16 != 0 ||
       input.numel() * input.element_size() % 16 != 0 ||
       full_numel * output.element_size() > static_cast<size_t>(comm->kMaxProblemSize))
        throw std::invalid_argument("qr_reduce_scatter: invalid or unsupported shard size");
    if(output.numel() == 0)
        return;
    HipDeviceGuard guard(input.device_id);
    auto stream = getCurrentHIPStream();
    if(input.dtype() == AITER_DTYPE_fp16)
        dispatch_quick_rs<half, false>(comm, input, output, stream);
    else if(input.dtype() == AITER_DTYPE_bf16 && cast_bf2half)
        dispatch_quick_rs<half, true>(comm, input, output, stream);
    else if(input.dtype() == AITER_DTYPE_bf16)
        dispatch_quick_rs<__hip_bfloat16, false>(comm, input, output, stream);
    else
        throw std::invalid_argument("qr_reduce_scatter: only fp16/bf16 supported");
}

} // namespace aiter
