// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "quick_reduce_collective.cuh"

namespace aiter {

// The second communication phase of QuickReduce on its own: every rank quantizes its contiguous
// shard with the QuickReduce codec and broadcasts it, and every rank dequantizes the W shards into
// the output. It is the all-gather that follows quick_reduce_scatter: reduce-scatter followed by
// all-gather over the same tensor produces the all-reduce result bitwise (the codec, the shard
// layout and the accumulation order are the all-reduce's own).
//
// Rank r contributes elements [0, S) of its input, S = input.numel(), and receives
// output[r' * S, (r' + 1) * S) = dequantize(quantize(rank r' input)).
template <typename T, class Codec, bool cast_bf2half>
struct QuickAllGather
{
    static constexpr int kWorldSize      = Codec::kWorldSize;
    static constexpr int kLocalTileBytes = kTileSize / kWorldSize;

    __device__ static void run(T const* __restrict__ input,
                               T* __restrict__ output,
                               uint32_t shard_bytes,
                               uint32_t /*in_bytes: the whole shard is valid*/,
                               int block,
                               int rank,
                               uint8_t** __restrict__ buffers,
                               uint32_t data_offset,
                               uint32_t color,
                               bool first,
                               bool last,
                               QuickNoArgs)
    {
        int thread = threadIdx.x + threadIdx.y * kWavefront;
        Codec codec(thread, rank);
        uint8_t* bp[kWorldSize];
        for(int r = 0; r < kWorldSize; ++r)
            bp[r] = buffers[r];
        uint32_t data     = data_offset + blockIdx.x * Codec::kTransmittedTileSize;
        uint32_t arrival  = blockIdx.x * kWorldSize * sizeof(uint32_t);
        uint32_t consumed = data_offset / 2 + arrival;

        // Tail atoms past the shard read as zero (bounded descriptor) and are dropped on store.
        BufferResource src(const_cast<T*>(input), shard_bytes);
        int32x4_t values[Codec::kRankAtoms];
        uint32_t offset = block * kLocalTileBytes + thread * sizeof(int32x4_t);
        for(int i = 0; i < Codec::kRankAtoms; ++i)
        {
            values[i] = buffer_load_dwordx4(src.descriptor, offset, 0, 0);
            offset += kAtomStride * sizeof(int32x4_t);
            if constexpr(cast_bf2half)
                values[i] = qr_bf16x8_to_half(values[i]);
        }
        // Do not overwrite the slot before every rank has consumed the previous tile that used it.
        if(!first)
            qr_wait_consumed(buffers, consumed, rank, thread, kWorldSize, color - 1);
        for(int r = 0; r < kWorldSize; ++r)
        {
            auto dst = reinterpret_cast<int32x4_t*>(bp[r] + data +
                                                    rank * Codec::kRankTransmittedTileSize);
            codec.send(dst, values);
        }

        qr_publish_flag(buffers, arrival, rank, thread, kWorldSize, color);

        auto recv  = reinterpret_cast<int32x4_t*>(buffers[rank] + data);
        auto flags = reinterpret_cast<uint32_t*>(buffers[rank] + arrival);
        int32x4_t got[kWorldSize][Codec::kRankAtoms];
        for(int r = 0; r < kWorldSize; ++r)
        {
            if(thread == 0)
                wait_sync_flag(flags + r, color);
            __syncthreads();
            codec.recv(&recv, got[r]);
        }

        // Every rank's data is in registers: the slot is consumed. Acknowledge, then store.
        qr_publish_flag(buffers, consumed, rank, thread, kWorldSize, color);
        for(int r = 0; r < kWorldSize; ++r)
        {
            // One resource per source rank: a tail atom must not spill into the next rank's rows.
            BufferResource dst(output + r * (shard_bytes / sizeof(T)), shard_bytes);
            uint32_t o = block * kLocalTileBytes + thread * sizeof(int32x4_t);
            for(int i = 0; i < Codec::kRankAtoms; ++i)
            {
                if constexpr(cast_bf2half)
                    got[r][i] = qr_half_to_bf16x8(got[r][i]);
                buffer_store_dwordx4(got[r][i], dst.descriptor, o, 0, 0);
                o += kAtomStride * sizeof(int32x4_t);
            }
        }

        if(last)
            qr_wait_consumed(buffers, consumed, rank, thread, kWorldSize, color);
    }
};

template <typename T, int W, bool cast_bf2half>
void launch_quick_ag(DeviceComms* comm,
                     const aiter_tensor_t& input,
                     const aiter_tensor_t& output,
                     hipStream_t stream)
{
    using Kernel = QuickAllGather<T, CodecQ4<T, W>, cast_bf2half>;
    uint32_t shard_bytes = input.numel() * input.element_size();
    uint32_t blocks      = divceil(shard_bytes, kTileSize / W);
    uint32_t grid        = std::min(static_cast<uint32_t>(kMaxNumBlocks), blocks);
    hipLaunchKernelGGL((quick_collective_kernel<Kernel, T>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       reinterpret_cast<const T*>(input.data_ptr()),
                       reinterpret_cast<T*>(output.data_ptr()),
                       shard_bytes,
                       shard_bytes,
                       blocks,
                       comm->rank,
                       comm->dbuffer_list,
                       comm->data_offset,
                       comm->d_flag_color);
    HIP_CHECK(hipGetLastError());
}

template <typename T, bool cast_bf2half>
void dispatch_quick_ag(DeviceComms* comm,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       hipStream_t stream)
{
    switch(comm->world_size)
    {
    case 2: launch_quick_ag<T, 2, cast_bf2half>(comm, input, output, stream); break;
    case 4: launch_quick_ag<T, 4, cast_bf2half>(comm, input, output, stream); break;
    default: launch_quick_ag<T, 8, cast_bf2half>(comm, input, output, stream); break;
    }
}

void qr_all_gather(int64_t handle,
                   const aiter_tensor_t& input,
                   const aiter_tensor_t& output,
                   bool cast_bf2half)
{
    auto comm = quick_collective_comm(handle, "qr_all_gather");
    quick_collective_check_tensors("qr_all_gather", input, output, comm, input.numel());
    if(output.numel() != input.numel() * comm->world_size)
        throw std::invalid_argument("qr_all_gather: output must hold world_size * input.numel() elements");
    HipDeviceGuard guard(input.device_id);
    auto stream = getCurrentHIPStream();
    if(input.dtype() == AITER_DTYPE_fp16)
        dispatch_quick_ag<half, false>(comm, input, output, stream);
    else if(cast_bf2half)
        dispatch_quick_ag<half, true>(comm, input, output, stream);
    else
        dispatch_quick_ag<__hip_bfloat16, false>(comm, input, output, stream);
}

} // namespace aiter
