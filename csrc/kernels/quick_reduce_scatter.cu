// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// quick_all_reduce.cuh requires these standard-library declarations first.
#include <limits>
#include <optional>

#include <array>
#include <atomic>
#include <string>

#include "../opus_moe/include/gfx950/opus_moe_stage2_utils_gfx950.cuh"
#include "aiter_stream.h"
#include "aiter_tensor.h"
#include "quick_all_reduce.cuh"

namespace aiter {

// Inputs of QuickReduceScatter. load() reads this thread's atoms of rank r's
// shard of a virtual [W * S] tensor at byte `offset` within the shard; atoms
// past the valid data read as zero.

// A dense tensor of `bytes` bytes (at most W shards).
template <typename T>
struct QrDenseInput
{
    const T* ptr;
    uint32_t bytes;

    template <int kAtoms>
    __device__ void
    load(int r, uint32_t shard_bytes, uint32_t offset, int32x4_t (&values)[kAtoms]) const
    {
        // A separate bounded resource per shard makes padded tail atoms zero,
        // without reading the next rank's token shard.
        uint32_t begin = r * shard_bytes;
        uint32_t valid = bytes > begin ? min(bytes - begin, shard_bytes) : 0;
        BufferResource src(const_cast<T*>(ptr) + begin / sizeof(T), valid);
        for(int i = 0; i < kAtoms; ++i)
        {
            values[i] = buffer_load_dwordx4(src.descriptor, offset, 0, 0);
            offset += kAtomStride * sizeof(int32x4_t);
        }
    }
};

// The routed term of an MoE layer as the OPUS stage 2 leaves it with
// return_per_slot: per-slot MXFP8 rows, uint8 [rows * topk, hidden + hidden / 8]
// (hidden e4m3 codes, then one e8m0 scale per 8 codes), token-major. load()
// sums the topk slots of a token into the bf16 value
// opus_moe_stage2_reduce_token_slot_route_output_fwd writes, bitwise: the
// same fp32 accumulation, slots in order, and one v_cvt_pk_bf16_f32 (see
// opus_moe_stage2_route_output_reduce_gfx950.cuh). The 8 elements of an atom
// are 8 consecutive columns of one row (hidden % 64 == 0), so one scale each.
// OCP e4m3 and the conversion need gfx950; the host rejects other GPUs.
// TOPK > 0 unrolls the slot loop, TOPK == 0 reads topk at run time.
template <int TOPK>
struct QrRouteInput
{
    const uint8_t* route;
    uint32_t rows;
    uint32_t hidden;
    uint32_t topk;

    __device__ int32x4_t sum_slots(uint32_t elem) const
    {
#if defined(__gfx950__)
        using f32x2          = float __attribute__((ext_vector_type(2)));
        const uint32_t token = elem / hidden;
        if(token >= rows)
            return int32x4_t{0, 0, 0, 0};
        const uint32_t col    = elem - token * hidden;
        const int topk_loop   = TOPK > 0 ? TOPK : topk;
        const uint64_t stride = hidden + hidden / 8;
        const uint8_t* row0   = route + static_cast<uint64_t>(token) * topk_loop * stride;
        f32x2 acc[4] = {f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}, f32x2{0.0f, 0.0f}};
#pragma unroll
        for(int slot = 0; slot < topk_loop; ++slot)
        {
            const uint8_t* rp = row0 + slot * stride;
            const uint64_t f8 =
                __builtin_nontemporal_load(reinterpret_cast<const uint64_t*>(rp + col));
            // e8m0 scale 2^(e - 127): a power of two, so fp8 * scale is exact
            // and FMA or mul + add round alike.
            const float sf =
                __builtin_bit_cast(float, static_cast<uint32_t>(rp[hidden + col / 8]) << 23);
            const f32x2 s2      = f32x2{sf, sf};
            const uint32_t lo32 = static_cast<uint32_t>(f8);
            const uint32_t hi32 = static_cast<uint32_t>(f8 >> 32);
            acc[0] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, false) * s2;
            acc[1] += __builtin_amdgcn_cvt_pk_f32_fp8(lo32, true) * s2;
            acc[2] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, false) * s2;
            acc[3] += __builtin_amdgcn_cvt_pk_f32_fp8(hi32, true) * s2;
        }
        int32x4_t v;
        for(int j = 0; j < 4; ++j)
            v[j] = static_cast<int32_t>(opus_moe_gfx950_cvt_pk_bf16_f32(acc[j][0], acc[j][1]));
        return v;
#else
        __builtin_trap();
        return int32x4_t{0, 0, 0, 0};
#endif
    }

    template <int kAtoms>
    __device__ void
    load(int r, uint32_t shard_bytes, uint32_t offset, int32x4_t (&values)[kAtoms]) const
    {
        const uint32_t begin = r * shard_bytes;
        for(int i = 0; i < kAtoms; ++i)
        {
            values[i] = offset < shard_bytes ? sum_slots((begin + offset) / sizeof(__hip_bfloat16))
                                             : int32x4_t{0, 0, 0, 0};
            offset += kAtomStride * sizeof(int32x4_t);
        }
    }
};

// Grid-stride driver: like allreduce_prototype_twoshot, but hands the kernel
// its own argument struct. The per-block flag colour lives in device memory
// and is advanced on the device, so each CUDA graph replay uses a fresh one.
template <typename Kernel, typename Args>
__global__ __quickreduce_launch_bounds_two_shot__ static void
quick_rs_kernel(Args args,
                uint32_t num_blocks,
                int rank,
                uint8_t** buffers,
                uint32_t data_offset,
                uint32_t* d_flag_color)
{
    int block           = blockIdx.x;
    uint32_t flag_color = d_flag_color[blockIdx.x];
    while(block < num_blocks)
    {
        Kernel::run(args, block, rank, buffers, data_offset, flag_color);
        block += gridDim.x;
        flag_color++;
    }
    if(threadIdx.x == 0 && threadIdx.y == 0)
        d_flag_color[blockIdx.x] = flag_color;
}

// The first communication phase of QuickReduce, with contiguous rank shards.
// The existing all-reduce assigns a stripe of every tile to each rank instead.
// Reuse its codec and accumulation order, but omit the entire data all-gather
// and the second quantization. A consumption barrier still protects IPC buffer
// reuse by the next grid-stride iteration or CUDA graph replay.
// The input may be shorter than W shards; the missing tail reads as zero.
// `Input` supplies the shards (QrDenseInput, QrRouteInput).
template <typename T, class Codec, bool cast_bf2half, class Input>
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

    struct Args
    {
        Input input;
        T* output;
        uint32_t shard_bytes;
    };

    __device__ static void run(const Args& args,
                               int block,
                               int rank,
                               uint8_t** __restrict__ buffers,
                               uint32_t data_offset,
                               uint32_t color)
    {
        int thread = threadIdx.x + threadIdx.y * kWavefront;
        Codec codec(thread, rank);
        // Cache the peer pointers: uint8_t* stores may alias the table.
        uint8_t* bp[kWorldSize];
        for(int r = 0; r < kWorldSize; ++r)
            bp[r] = buffers[r];
        const uint32_t shard_bytes = args.shard_bytes;
        uint32_t data              = data_offset + blockIdx.x * Codec::kTransmittedTileSize;
        uint32_t arrival           = blockIdx.x * kWorldSize * sizeof(uint32_t);
        uint32_t consumed          = data_offset / 2 + arrival;

        int32x4_t values[kWorldSize][Codec::kRankAtoms];
        for(int r = 0; r < kWorldSize; ++r)
            args.input.load(
                r, shard_bytes, block * kLocalTileBytes + thread * sizeof(int32x4_t), values[r]);

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

        BufferResource dst(args.output, shard_bytes);
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

template <typename T, int W, bool cast_bf2half, class Input>
void launch_quick_rs(
    DeviceComms* comm, const Input& input, void* output, uint32_t shard_bytes, hipStream_t stream)
{
    using Kernel    = QuickReduceScatter<T, CodecQ4<T, W>, cast_bf2half, Input>;
    using Args      = typename Kernel::Args;
    uint32_t blocks = divceil(shard_bytes, kTileSize / W);
    uint32_t grid   = std::min(static_cast<uint32_t>(kMaxNumBlocks), blocks);
    hipLaunchKernelGGL((quick_rs_kernel<Kernel, Args>),
                       dim3(grid),
                       dim3(kBlockTwoShot),
                       0,
                       stream,
                       Args{input, reinterpret_cast<T*>(output), shard_bytes},
                       blocks,
                       comm->rank,
                       comm->dbuffer_list,
                       comm->data_offset,
                       comm->d_flag_color);
    HIP_CHECK(hipGetLastError());
}

template <typename T, bool cast_bf2half, class Input>
void dispatch_quick_rs(
    DeviceComms* comm, const Input& input, void* output, uint32_t shard_bytes, hipStream_t stream)
{
    switch(comm->world_size)
    {
    case 4: launch_quick_rs<T, 4, cast_bf2half>(comm, input, output, shard_bytes, stream); break;
    case 8: launch_quick_rs<T, 8, cast_bf2half>(comm, input, output, shard_bytes, stream); break;
    default: throw std::invalid_argument("qr_reduce_scatter: requires 4 or 8 ranks");
    }
}

// Host checks shared by the reduce-scatter entry points.
static DeviceComms* quick_rs_comm(int64_t handle, const char* name)
{
    auto comm = reinterpret_cast<DeviceComms*>(handle);
    if(!comm || !comm->initialized)
        throw std::invalid_argument(std::string(name) + ": uninitialized communicator");
    if(comm->world_size != 4 && comm->world_size != 8)
        throw std::invalid_argument(std::string(name) + ": requires 4 or 8 ranks");
    return comm;
}

void qr_reduce_scatter(int64_t handle,
                       const aiter_tensor_t& input,
                       const aiter_tensor_t& output,
                       bool cast_bf2half)
{
    auto comm = quick_rs_comm(handle, "qr_reduce_scatter");
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
    auto stream                = getCurrentHIPStream();
    const uint32_t in_bytes    = input.numel() * input.element_size();
    const uint32_t shard_bytes = output.numel() * output.element_size();
    void* out                  = output.data_ptr();
    if(input.dtype() == AITER_DTYPE_fp16)
        dispatch_quick_rs<half, false>(
            comm,
            QrDenseInput<half>{static_cast<const half*>(input.data_ptr()), in_bytes},
            out,
            shard_bytes,
            stream);
    else if(input.dtype() == AITER_DTYPE_bf16 && cast_bf2half)
        dispatch_quick_rs<half, true>(
            comm,
            QrDenseInput<half>{static_cast<const half*>(input.data_ptr()), in_bytes},
            out,
            shard_bytes,
            stream);
    else if(input.dtype() == AITER_DTYPE_bf16)
        dispatch_quick_rs<__hip_bfloat16, false>(
            comm,
            QrDenseInput<__hip_bfloat16>{static_cast<const __hip_bfloat16*>(input.data_ptr()),
                                         in_bytes},
            out,
            shard_bytes,
            stream);
    else
        throw std::invalid_argument("qr_reduce_scatter: only fp16/bf16 supported");
}

// gfx950, checked once per device: the route input needs OCP e4m3 and
// v_cvt_pk_bf16_f32, and its device code traps on other GPUs.
static bool quick_rs_is_gfx950(int device)
{
    static std::array<std::atomic<int8_t>, 64> cache{}; // 0 unknown, 1 yes, -1 no
    if(device < 0 || device >= static_cast<int>(cache.size()))
        throw std::invalid_argument("qr_reduce_scatter_route: invalid device id");
    int8_t known = cache[device].load(std::memory_order_relaxed);
    if(known == 0)
    {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, device));
        known = std::string(prop.gcnArchName).rfind("gfx950", 0) == 0 ? 1 : -1;
        cache[device].store(known, std::memory_order_relaxed);
    }
    return known > 0;
}

// Reduce-scatter of the routed term of an MoE layer straight from the OPUS
// stage 2's per-slot MXFP8 rows (see QrRouteInput): equal to qr_reduce_scatter
// (cast_bf2half) of opus_moe_stage2_reduce_token_slot_route_output_fwd's bf16
// [rows, hidden] output, bitwise, without writing and re-reading that tensor.
void qr_reduce_scatter_route(int64_t handle,
                             const aiter_tensor_t& route,
                             const aiter_tensor_t& output,
                             int64_t rows,
                             int64_t topk,
                             int64_t hidden)
{
    auto comm = quick_rs_comm(handle, "qr_reduce_scatter_route");
    if(!route.is_gpu() || route.device_id != output.device_id || !route.is_contiguous() ||
       !output.is_contiguous() || route.dtype() != AITER_DTYPE_u8 ||
       output.dtype() != AITER_DTYPE_bf16)
        throw std::invalid_argument("qr_reduce_scatter_route: route must be contiguous uint8 and "
                                    "output contiguous bf16, on one GPU");
    if(!quick_rs_is_gfx950(route.device_id))
        throw std::invalid_argument("qr_reduce_scatter_route: requires gfx950");
    // hidden % 64: an atom (8 columns, one scale) stays inside a row, and the
    // 8-byte code loads stay aligned (row stride hidden * 9 / 8 bytes).
    if(rows < 0 || topk <= 0 || hidden <= 0 || hidden % 64 != 0)
        throw std::invalid_argument(
            "qr_reduce_scatter_route: needs rows >= 0, topk > 0 and hidden % 64 == 0");
    if(route.numel() != static_cast<size_t>(rows) * topk * (hidden + hidden / 8))
        throw std::invalid_argument(
            "qr_reduce_scatter_route: route must be [rows * topk, hidden + hidden / 8] bytes");
    const size_t full_numel = output.numel() * comm->world_size;
    if(static_cast<size_t>(rows) * hidden > full_numel ||
       output.numel() * output.element_size() % 16 != 0 ||
       full_numel * output.element_size() > static_cast<size_t>(comm->kMaxProblemSize))
        throw std::invalid_argument("qr_reduce_scatter_route: invalid or unsupported shard size");
    if(output.numel() == 0)
        return;
    HipDeviceGuard guard(route.device_id);
    auto stream                = getCurrentHIPStream();
    const uint32_t shard_bytes = output.numel() * output.element_size();
    auto src                   = static_cast<const uint8_t*>(route.data_ptr());
    const uint32_t r = rows, h = hidden, k = topk;
    // The INT4 codec runs in fp16, as qr_reduce_scatter's default cast_bf2half.
    if(topk == 6)
        dispatch_quick_rs<half, true>(
            comm, QrRouteInput<6>{src, r, h, k}, output.data_ptr(), shard_bytes, stream);
    else
        dispatch_quick_rs<half, true>(
            comm, QrRouteInput<0>{src, r, h, k}, output.data_ptr(), shard_bytes, stream);
}

} // namespace aiter
