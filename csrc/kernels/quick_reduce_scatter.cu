// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <type_traits>

#include "quick_reduce_collective.cuh"

namespace aiter {

#if defined(__gfx950__)
// The routed term of an MoE layer as the OPUS stage 2 leaves it: per-slot MXFP8 rows
// [rows * topk, hidden fp8 e4m3 | hidden / 8 e8m0 scales] uint8, row-major by token then slot.
// Summing the slots on load is exactly what opus_moe_stage2_reduce_token_slot_route_output
// <ROUTE_FP8=true> does: fp32 accumulators from 0, slots in order, acc += fp8 * 2^e (a power-of-two
// scale makes the product exact, so FMA or mul + add round the same) and one v_cvt_pk_bf16_f32 at the
// end. The 8 elements a thread owns are 8 consecutive columns of one token row (hidden % 128 == 0 and
// 16-byte aligned), i.e. exactly one scale group per slot. Rows >= rows read as zero.
__device__ __forceinline__ uint32_t qr_cvt_pk_bf16_f32(float lo, float hi)
{
    uint32_t packed;
    asm volatile("v_cvt_pk_bf16_f32 %0, %1, %2" : "=v"(packed) : "v"(lo), "v"(hi));
    return packed;
}

template <int TOPK>
__device__ __forceinline__ int32x4_t qr_route_slots_bf16x8(const uint8_t* __restrict__ route,
                                                           uint32_t elem,
                                                           uint32_t rows,
                                                           uint32_t hidden)
{
    using f32x2          = float __attribute__((ext_vector_type(2)));
    const uint32_t token = elem / hidden;
    if(token >= rows)
        return int32x4_t{0, 0, 0, 0};
    const uint32_t col  = elem - token * hidden;
    const uint64_t rstr = static_cast<uint64_t>(hidden) + hidden / 8;
    f32x2 acc[4]        = {f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}};
#pragma unroll
    for(int slot = 0; slot < TOPK; ++slot)
    {
        const uint8_t* rp = route + (static_cast<uint64_t>(token) * TOPK + slot) * rstr;
        const uint64_t f8 = __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(rp + col));
        const float sf =
            __builtin_bit_cast(float, static_cast<uint32_t>(rp[hidden + (col >> 3)]) << 23);
        const f32x2 s2      = f32x2{sf, sf};
        const uint32_t lo32 = static_cast<uint32_t>(f8);
        const uint32_t hi32 = static_cast<uint32_t>(f8 >> 32);
        acc[0] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, false) * s2;
        acc[1] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, true) * s2;
        acc[2] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, false) * s2;
        acc[3] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, true) * s2;
    }
    int32x4_t v;
    v[0] = static_cast<int32_t>(qr_cvt_pk_bf16_f32(acc[0][0], acc[0][1]));
    v[1] = static_cast<int32_t>(qr_cvt_pk_bf16_f32(acc[1][0], acc[1][1]));
    v[2] = static_cast<int32_t>(qr_cvt_pk_bf16_f32(acc[2][0], acc[2][1]));
    v[3] = static_cast<int32_t>(qr_cvt_pk_bf16_f32(acc[3][0], acc[3][1]));
    return v;
}
#endif

// Arguments of the route source: the real row count and the row width.
struct QuickRouteArgs
{
    uint32_t rows;
    uint32_t hidden;
};

// The first communication phase of QuickReduce, with contiguous rank shards.
// The existing all-reduce assigns a stripe of every tile to each rank instead.
// Reuse its codec and accumulation order, but omit the entire data all-gather
// and the second quantization. The consumption barrier at the end of each block
// protects IPC buffer reuse by the next grid-stride iteration, the next launch
// or a CUDA graph replay (see quick_reduce_collective.cuh).
//
// Rank r receives the sum of shard r of every rank's input, shard r being
// elements [r * S, (r + 1) * S) of the (virtually zero-padded) input, S = output.numel().
// Only in_bytes of the input are read; everything past it reads as zero. A ragged input
// (numel < W * S, e.g. a row count that is not a multiple of W) therefore needs no padded copy.
// kRoute: the input is the route source above (half codec with a bf16 cast, TOPK 6) instead of a
// dense tensor; `input` then points at the MXFP8 slot rows and in_bytes is the bf16 byte size of the
// reduced tensor (rows * hidden * 2).
template <typename T, class Codec, bool cast_bf2half, bool kRoute = false>
struct QuickReduceScatter
{
    static_assert(!kRoute || cast_bf2half, "the route source produces bf16 rows for the half codec");
    using Args = std::conditional_t<kRoute, QuickRouteArgs, QuickNoArgs>;

    static constexpr int kWorldSize      = Codec::kWorldSize;
    static constexpr int kLocalTileBytes = kTileSize / kWorldSize;

    __device__ static void run(T const* __restrict__ input,
                               T* __restrict__ output,
                               uint32_t shard_bytes,
                               uint32_t in_bytes,
                               int block,
                               int rank,
                               uint8_t** __restrict__ buffers,
                               uint32_t data_offset,
                               uint32_t color,
                               bool first,
                               bool last,
                               Args args)
    {
        int thread = threadIdx.x + threadIdx.y * kWavefront;
        Codec codec(thread, rank);
        uint8_t* bp[kWorldSize];
        for(int r = 0; r < kWorldSize; ++r)
            bp[r] = buffers[r];
        uint32_t data     = data_offset + blockIdx.x * Codec::kTransmittedTileSize;
        uint32_t arrival  = blockIdx.x * kWorldSize * sizeof(uint32_t);
        uint32_t consumed = data_offset / 2 + arrival;

        // Issue every load first (one bounded descriptor per shard makes the padded tail atoms read
        // as zero without touching the next rank's shard), then quantize and send shard by shard.
        int32x4_t values[kWorldSize][Codec::kRankAtoms];
        for(int r = 0; r < kWorldSize; ++r)
        {
            uint32_t begin = r * shard_bytes;
            uint32_t valid = in_bytes > begin ? min(in_bytes - begin, shard_bytes) : 0;
            BufferResource src(const_cast<T*>(input) + r * (shard_bytes / sizeof(T)), valid);
            uint32_t offset = block * kLocalTileBytes + thread * sizeof(int32x4_t);
            for(int i = 0; i < Codec::kRankAtoms; ++i)
            {
                if constexpr(kRoute)
                {
#if defined(__gfx950__)
                    // the bounded descriptor of the dense path becomes an explicit range check
                    values[r][i] = offset < valid
                                       ? qr_route_slots_bf16x8<6>(reinterpret_cast<const uint8_t*>(input),
                                                                  (begin + offset) / sizeof(T),
                                                                  args.rows,
                                                                  args.hidden)
                                       : int32x4_t{0, 0, 0, 0};
#else
                    __builtin_trap();
#endif
                }
                else
                {
                    values[r][i] = buffer_load_dwordx4(src.descriptor, offset, 0, 0);
                }
                offset += kAtomStride * sizeof(int32x4_t);
            }
        }
        if(!first)
            qr_wait_consumed(buffers, consumed, rank, thread, kWorldSize, color - 1);
        for(int r = 0; r < kWorldSize; ++r)
        {
            if constexpr(cast_bf2half)
                for(int i = 0; i < Codec::kRankAtoms; ++i)
                    values[r][i] = qr_bf16x8_to_half(values[r][i]);
            auto dst = reinterpret_cast<int32x4_t*>(bp[r] + data +
                                                    rank * Codec::kRankTransmittedTileSize);
            codec.send(dst, values[r]);
        }

        qr_publish_flag(buffers, arrival, rank, thread, kWorldSize, color);

        int32x4_t sum[Codec::kRankAtoms] = {};
        auto recv                        = reinterpret_cast<int32x4_t*>(buffers[rank] + data);
        auto flags                       = reinterpret_cast<uint32_t*>(buffers[rank] + arrival);
        for(int r = 0; r < kWorldSize; ++r)
        {
            if(thread == 0)
                wait_sync_flag(flags + r, color);
            __syncthreads();
            int32x4_t values[Codec::kRankAtoms];
            codec.recv(&recv, values);
            for(int i = 0; i < Codec::kRankAtoms; ++i)
                packed_assign_add<T>(&sum[i], &values[i]);
        }

        // Acknowledge that all threads have consumed the input data. Unlike
        // all-reduce, this phase sends only flags, never reduced payloads.
        qr_publish_flag(buffers, consumed, rank, thread, kWorldSize, color);

        BufferResource dst(output, shard_bytes);
        uint32_t offset = block * kLocalTileBytes + thread * sizeof(int32x4_t);
        for(int i = 0; i < Codec::kRankAtoms; ++i)
        {
            if constexpr(cast_bf2half)
                sum[i] = qr_half_to_bf16x8(sum[i]);
            buffer_store_dwordx4(sum[i], dst.descriptor, offset, 0, 0);
            offset += kAtomStride * sizeof(int32x4_t);
        }

        if(last)
            qr_wait_consumed(buffers, consumed, rank, thread, kWorldSize, color);
    }
};

template <typename T, int W, bool cast_bf2half, bool kRoute = false>
void launch_quick_rs(DeviceComms* comm,
                     const void* input,
                     uint32_t in_bytes,
                     void* output,
                     uint32_t shard_bytes,
                     QuickRouteArgs route,
                     hipStream_t stream)
{
    using Kernel = QuickReduceScatter<T, CodecQ4<T, W>, cast_bf2half, kRoute>;
    using Args   = typename Kernel::Args;
    uint32_t blocks = divceil(shard_bytes, kTileSize / W);
    uint32_t grid   = std::min(static_cast<uint32_t>(kMaxNumBlocks), blocks);
    Args args{};
    if constexpr(kRoute)
        args = route;
    hipLaunchKernelGGL((quick_collective_kernel<Kernel, T, Args>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       reinterpret_cast<const T*>(input),
                       reinterpret_cast<T*>(output),
                       shard_bytes,
                       in_bytes,
                       blocks,
                       comm->rank,
                       comm->dbuffer_list,
                       comm->data_offset,
                       comm->d_flag_color,
                       args);
    HIP_CHECK(hipGetLastError());
}

template <typename T, bool cast_bf2half, bool kRoute = false>
void dispatch_quick_rs(DeviceComms* comm,
                       const void* input,
                       uint32_t in_bytes,
                       void* output,
                       uint32_t shard_bytes,
                       QuickRouteArgs route,
                       hipStream_t stream)
{
    switch(comm->world_size)
    {
    case 2:
        launch_quick_rs<T, 2, cast_bf2half, kRoute>(comm, input, in_bytes, output, shard_bytes, route, stream);
        break;
    case 4:
        launch_quick_rs<T, 4, cast_bf2half, kRoute>(comm, input, in_bytes, output, shard_bytes, route, stream);
        break;
    default:
        launch_quick_rs<T, 8, cast_bf2half, kRoute>(comm, input, in_bytes, output, shard_bytes, route, stream);
        break;
    }
}

void qr_reduce_scatter(int64_t handle,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       bool cast_bf2half)
{
    auto comm = quick_collective_comm(handle, "qr_reduce_scatter");
    quick_collective_check_tensors("qr_reduce_scatter", input, output, comm, output.numel());
    // input may be shorter than W shards (ragged tail, read as zero) but must reach into the
    // last shard's rank range, and must be 16-byte sized for the vector loads at its end.
    const size_t shard = output.numel();
    if(input.numel() == 0 || input.numel() > shard * comm->world_size ||
       input.numel() * input.element_size() % 16 != 0)
        throw std::invalid_argument(
            "qr_reduce_scatter: input must hold 1..world_size*output.numel() elements and a "
            "multiple of 16 bytes");
    HipDeviceGuard guard(input.device_id);
    auto stream               = getCurrentHIPStream();
    const uint32_t in_bytes   = input.numel() * input.element_size();
    const uint32_t shard_bytes = shard * output.element_size();
    if(input.dtype() == AITER_DTYPE_fp16)
        dispatch_quick_rs<half, false>(
            comm, input.data_ptr(), in_bytes, output.data_ptr(), shard_bytes, {}, stream);
    else if(cast_bf2half)
        dispatch_quick_rs<half, true>(
            comm, input.data_ptr(), in_bytes, output.data_ptr(), shard_bytes, {}, stream);
    else
        dispatch_quick_rs<__hip_bfloat16, false>(
            comm, input.data_ptr(), in_bytes, output.data_ptr(), shard_bytes, {}, stream);
}

// Reduce-scatter of the routed term of an MoE layer straight from the OPUS stage 2's per-slot MXFP8
// rows (see qr_route_slots_bf16x8): bitwise equal to qr_reduce_scatter of the bf16 tensor
// opus_moe_stage2_reduce_token_slot_route_output returns for the same rows, without writing and
// re-reading that [rows, hidden] tensor. route: uint8 [rows * 6, hidden + hidden / 8];
// output: bf16, this rank's shard; the reduced tensor [rows, hidden] may hold fewer than
// world_size shards of data (ragged: the rest reads as zero). gfx950 only. The INT4 wire runs in
// fp16 (cast_bf2half), as the all-reduce does for bf16.
void qr_reduce_scatter_route(int64_t handle,
                             const aiter_tensor_t& route,
                             const aiter_tensor_t& output,
                             int64_t rows,
                             int64_t topk,
                             int64_t hidden)
{
    auto comm = quick_collective_comm(handle, "qr_reduce_scatter_route");
    if(!route.is_gpu() || route.device_id != output.device_id || !route.is_contiguous() ||
       !output.is_contiguous() || route.dtype() != AITER_DTYPE_u8 || output.dtype() != AITER_DTYPE_bf16)
        throw std::invalid_argument(
            "qr_reduce_scatter_route: route must be contiguous uint8 and output contiguous bf16 on one GPU");
    if(topk != 6)
        throw std::invalid_argument("qr_reduce_scatter_route: only topk 6 is supported");
    if(hidden <= 0 || hidden % 128 != 0 || rows <= 0)
        throw std::invalid_argument("qr_reduce_scatter_route: needs rows > 0 and hidden % 128 == 0");
    if(route.numel() != static_cast<size_t>(rows) * topk * (hidden + hidden / 8))
        throw std::invalid_argument(
            "qr_reduce_scatter_route: route must be [rows * topk, hidden + hidden / 8] bytes");
    const size_t shard = output.numel();
    if(shard * output.element_size() % 16 != 0 || shard == 0 ||
       static_cast<size_t>(rows) * hidden > shard * comm->world_size ||
       shard * comm->world_size * output.element_size() > static_cast<size_t>(comm->kMaxProblemSize))
        throw std::invalid_argument(
            "qr_reduce_scatter_route: output must be a 16-byte-multiple shard with rows * hidden <= "
            "world_size * output.numel() and the whole tensor within the communicator's max size");
    HipDeviceGuard guard(route.device_id);
    dispatch_quick_rs<half, true, true>(comm,
                                        route.data_ptr(),
                                        static_cast<uint32_t>(rows * hidden * 2),
                                        output.data_ptr(),
                                        static_cast<uint32_t>(shard * 2),
                                        QuickRouteArgs{static_cast<uint32_t>(rows),
                                                       static_cast<uint32_t>(hidden)},
                                        getCurrentHIPStream());
}

} // namespace aiter
