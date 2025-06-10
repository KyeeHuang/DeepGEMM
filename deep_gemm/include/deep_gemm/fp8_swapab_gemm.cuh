#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include "mma_utils.cuh"
#include "scheduler.cuh"
#include "tma_utils.cuh"
#include "utils.cuh"

namespace deep_gemm {

template <uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t NUM_WARPS_PER_BLOCK>
static __device__ __forceinline__ void write_result_to_gmem(__nv_bfloat16* gmem_d_this_block, const __nv_bfloat16* smem_d, 
                                                            const uint32_t m_offset, const uint32_t m_boundary,
                                                            const uint32_t n_offset, const uint32_t shape_n,
                                                            const uint32_t ld_output) {
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;
    constexpr int int4_per_tile_line = BLOCK_N * sizeof(__nv_bfloat16) / sizeof(int4);
    int int4_per_global_line = shape_n * sizeof(__nv_bfloat16) / sizeof(int4);
    constexpr auto num_lines = BLOCK_M;
    constexpr auto num_warps = NUM_WARPS_PER_BLOCK;
    const int4* smem_d_int4 = reinterpret_cast<const int4*>(smem_d);
    bool is_last_n_block = n_offset + BLOCK_N > shape_n;
    int int4_per_line =
        is_last_n_block ? int4_per_global_line % int4_per_tile_line : int4_per_tile_line;

    for (int line_idx = warp_idx; line_idx < num_lines; line_idx += num_warps) {
        if (m_offset + line_idx >= m_boundary) {
            break;
        }
        for (int elem_idx = lane_idx; elem_idx < int4_per_line; elem_idx += 32) {
            uint64_t idx = (uint64_t)line_idx * ld_output + n_offset;
            int4* g_data_addr =
                reinterpret_cast<int4*>(&gmem_d_this_block[idx]) + elem_idx;
            const int4* s_data_addr =
                &smem_d_int4[line_idx * (int4_per_tile_line) + elem_idx];
            *g_data_addr = *s_data_addr;
        }
        __syncwarp();
    }   
}

template <uint32_t SHAPE_M, uint32_t SHAPE_K,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleDMode,
          uint32_t kNumGroups, uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreadsPerGroup,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnAct,
          GemmType kGemmType>
__global__ void __launch_bounds__(get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M), 1)
fp8_gemm_kernel_swapab(__nv_bfloat16* gmem_d, float* scales_a, int* grouped_layout,
                        uint32_t shape_n,
                      const __grid_constant__ CUtensorMap tensor_map_a,  // weight (previously act)
                      const __grid_constant__ CUtensorMap tensor_map_b,  // act (previously weight)
                      const __grid_constant__ CUtensorMap tensor_map_scales_b,  // act scales (previously tensor_map_scales_a)
                      const __grid_constant__ CUtensorMap tensor_map_d) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(ceil_div(BLOCK_M, BLOCK_K) <= 2, "Too much A scales in a single block");
    DG_STATIC_ASSERT(ceil_div(BLOCK_M, BLOCK_K) == 1 or (BLOCK_K == BLOCK_M - BLOCK_K), "BLOCK_M should be 64, 128, 256");
    DG_STATIC_ASSERT(kGemmType == GemmType::GroupedMasked || kGemmType == GemmType::Normal, "SwapAB GEMM only supports grouped masked or normal");

    // Types
    using WGMMA = typename FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0, "Invalid block size");

    // Shared memory
    static constexpr uint32_t SMEM_D_SIZE = BLOCK_N * BLOCK_M * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE = BLOCK_N * sizeof(float);  // B matrix (act) scales
    static constexpr uint32_t SMEM_SCALES_B_SIZE_PER_STAGE_PADDED = ceil_div<uint32_t>(BLOCK_N * sizeof(float), 128)*128;  // B matrix (act) scales, 128B aligned
    static constexpr uint32_t SHAPE_K_SCALES = ceil_div(SHAPE_K, BLOCK_K);
    static constexpr uint32_t BLOCK_M_A_SCALES = ceil_div(BLOCK_M, BLOCK_K);
    static constexpr uint32_t SMEM_SCALES_A_SIZE = ceil_div<uint32_t>(BLOCK_M_A_SCALES * SHAPE_K_SCALES * sizeof(float), sizeof(Barrier)) * sizeof(Barrier);  // renamed to A (weight)

    // Configs
    constexpr uint32_t kFullKOfAllStages = kNumStages * BLOCK_K;
    constexpr uint32_t kNumThreads = get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M);
    constexpr uint32_t kNumMathThreads = kNumThreads - kNumTMAThreads;
    constexpr uint32_t kNumIterations = ceil_div(SHAPE_K, kFullKOfAllStages);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = get_lane_id();

    // Prefetch TMA descriptors at very beginning
    if (threadIdx.x == kNumMathThreads) {
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_a));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_scales_b));
        cute::prefetch_tma_descriptor(reinterpret_cast<cute::TmaDescriptor const*>(&tensor_map_d));
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    __nv_fp8_e4m3* smem_a[kNumStages];  // weight
    __nv_fp8_e4m3* smem_b[kNumStages];  // act
    float* smem_scales_b[kNumStages];   // act scales
    float* smem_scales_a;               // weight scales

    // TMA Barrier for both divisible and non-divisible cases
    Barrier* full_barriers[kNumStages];
    Barrier* empty_barriers[kNumStages];

    // Fill shared memory pointers
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        smem_a[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
        smem_b[i] = reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
        smem_scales_b[i] = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) + i * SMEM_SCALES_B_SIZE_PER_STAGE_PADDED);
    }
    smem_scales_a = reinterpret_cast<float*>(smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE_PADDED));

    // Fill barriers
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_scales_a) + SMEM_SCALES_A_SIZE);
    #pragma unroll
    for (int i = 0; i < kNumStages; ++ i) {
        full_barriers[i] = barrier_start_ptr + i;
        empty_barriers[i] = barrier_start_ptr + kNumStages + i;
    }

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (threadIdx.x == kNumMathThreads) {
        #pragma unroll
        for (int i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_view_async_shared();
        (kNumTMAMulticast > 1) ? cutlass::arch::fence_barrier_init() : void();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // For pipeline unrolling
    struct DivisibleK {};
    struct NotDivisibleK {};
    auto launch_k_iterations = [](const auto& func) {
        if constexpr (SHAPE_K % kFullKOfAllStages == 0) {
            for (int k_iter = 0; k_iter < kNumIterations; ++ k_iter)
                func(k_iter, DivisibleK{});
        } else {
            for (int k_iter = 0; k_iter < kNumIterations - 1; ++ k_iter)
                func(k_iter, DivisibleK{});
            func(kNumIterations - 1, NotDivisibleK{});
        }
    };

    // Register reconfigurations
    constexpr int kNumTMARegisters = 40;
    constexpr int kNumMathRegisters = 232;

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    // auto scheduler = SchedulerType(problem_input);
    auto scheduler = Scheduler<kGemmType, SHAPE_M, BLOCK_N, BLOCK_M, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnAct>(shape_n, grouped_layout);

    if (threadIdx.x >= kNumMathThreads) {
        // TMA warp-group for loading data
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();

        // NOTES: only one thread (or warp) will be used
        if (threadIdx.x == kNumMathThreads) {
            // Persistently schedule over blocks
            while (scheduler.get_next_block(n_block_idx, m_block_idx)) {
                launch_k_iterations([&](int k_iter, auto type) {
                    constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                    constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                    DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                    // Assign TMA multicast number into A and B
                    // NOTES: there may be additional odd rows/columns or cases where multicast is not possible.
                    const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);  // swapab only support normal gemm and grouped masked gemm
                    const uint32_t num_tma_multicast_a = (not kIsTMAMulticastOnAct and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    const uint32_t num_tma_multicast_b = (kIsTMAMulticastOnAct and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                    DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                    #pragma unroll
                    for (uint32_t s = 0; s < kNumInnerStages; ++ s) {
                        // Wait consumer release
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);

                        // Issue TMA A (weight) 
                        auto& full_barrier = *full_barriers[s];
                        int k_idx = k_iter * kFullKOfAllStages + s * BLOCK_K;
                        tma_copy(&tensor_map_a, reinterpret_cast<uint64_t*>(&full_barrier),
                                 smem_a[s], k_idx, scheduler.get_global_idx<false>(SHAPE_M, BLOCK_M, m_block_idx, n_block_idx), num_tma_multicast_a);

                        // Issue TMA B (act)
                        tma_copy(&tensor_map_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                  smem_b[s], k_idx, scheduler.get_global_idx(shape_n, BLOCK_N, n_block_idx), num_tma_multicast_b);
                        // Issue TMA scales_b (act scales) for B matrix
                        tma_copy(&tensor_map_scales_b, reinterpret_cast<uint64_t*>(&full_barrier),
                                                      smem_scales_b[s], n_block_idx * BLOCK_N,
                                                      scheduler.get_global_idx(SHAPE_K_SCALES, 1, k_idx / BLOCK_K), num_tma_multicast_b);
                        
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE + SMEM_SCALES_B_SIZE_PER_STAGE);
                    }

                    // Wait unaligned cases
                    #pragma unroll
                    for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                        empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter + 1) & 1);
                        full_barriers[s]->arrive();
                    }
                });
            }

            // To safely deconstruct distributed shared barriers, we need another round of empty waits
            if constexpr (kNumTMAMulticast > 1) {
                #pragma unroll
                for (uint32_t s = 0; s < kNumStages; ++ s)
                    empty_barriers[s]->wait((scheduler.current_iter * kNumIterations + 1) & 1);
            }
        }
    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / kNumMathThreadsPerGroup, 0);

        // Each thread loads consecutive 2 scales
        const uint32_t scale_offset = (lane_idx % 4) * 2;

        // Persistently schedule over blocks
        while (scheduler.get_next_block(n_block_idx, m_block_idx)) {
            // Load weight scales (scales_a) - these are associated with tensor_map_a (weight)
            // Decide the number of scales A to load
            DG_STATIC_ASSERT(SHAPE_M % 8 == 0, "Invalid shape M");
            uint32_t num_scales_a = BLOCK_M_A_SCALES * SHAPE_K_SCALES;

            // Load A scales with math warp-groups (weight scales)
            if (threadIdx.x >= 32) {
                auto num_previous_lines = scheduler.get_global_idx<false>(ceil_div(SHAPE_M, BLOCK_K), 0, 0, n_block_idx);
                auto local_scales_a = scales_a + (num_previous_lines + ((m_block_idx * BLOCK_M) / BLOCK_K)) * SHAPE_K_SCALES;
                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_scales_a; i += kNumMathThreads - 32)
                    st_shared(smem_scales_a + i, __ldg(local_scales_a + i));
            }
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Accumulation for WGMMA or CUDA promotion
            constexpr int WAVE_BLOCK_M = WGMMA::M * get_num_math_warpgroups(BLOCK_M);
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};

            // Empty barrier arrival
            auto empty_barrier_arrive = [&](int s) {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[s]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[s]->arrive(target_cta) : void();
                }
            };

            // Launch MMAs
            launch_k_iterations([&](int k_iter, auto type) {
                constexpr bool kHasDivisibleStages = std::is_same_v<decltype(type), DivisibleK>;
                constexpr int kNumInnerStages = kHasDivisibleStages ? kNumStages : (SHAPE_K % kFullKOfAllStages) / BLOCK_K;
                DG_STATIC_ASSERT(kNumInnerStages != 0, "Invalid number of inner stages");

                #pragma unroll
                for (int s = 0; s < kNumInnerStages; ++ s) {
                    // Read weight scales (A scales)
                    float scale_a_0 = ld_shared(smem_scales_a + k_iter * kNumStages + s), scale_a_1;
                    if constexpr (BLOCK_M_A_SCALES == 2) {
                        scale_a_1 = ld_shared(smem_scales_a + k_iter * kNumStages + s + SHAPE_K_SCALES);
                    }

                    // Wait TMA arrivals
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);

                     // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                    // Each thread reads consecutive two b scales, each thread needs to read WGMMA::N / 4 * 2 b scales
                    float scale_0_0[WGMMA::kNumAccum / 4], scale_0_1[WGMMA::kNumAccum / 4], scale_1_0[WGMMA::kNumAccum / 4], scale_1_1[WGMMA::kNumAccum / 4];
                    #pragma unroll
                    for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        float2 scale_b = ld_shared(reinterpret_cast<const float2*>(smem_scales_b[s] + i * 8 + scale_offset));
                        scale_0_0[i] = scale_a_0 * scale_b.x;
                        scale_0_1[i] = scale_a_0 * scale_b.y;
                        if constexpr (BLOCK_M_A_SCALES == 2) {
                            scale_1_0[i] = scale_a_1 * scale_b.x;
                            scale_1_1[i] = scale_a_1 * scale_b.y;
                        }
                    }

                    #pragma unroll
                    for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                      	auto m_offset = local_idx * WAVE_BLOCK_M;
                        // Commit WGMMA instructions
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_arrive();
                        #pragma unroll
                        for (int k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                            // auto desc_a = make_smem_desc(smem_a[s] + math_wg_idx * WGMMA::M * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_a = make_smem_desc(smem_a[s] + (math_wg_idx * WGMMA::M + m_offset) * BLOCK_K + k * WGMMA::K, 1);
                            auto desc_b = make_smem_desc(smem_b[s] + k * WGMMA::K, 1);
                            WGMMA::wgmma(desc_a, desc_b, accum, k);
                        }
                        warpgroup_commit_batch();
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum; ++ i)
                            warpgroup_fence_operand(accum[i]);
                        warpgroup_wait<0>();

                        // Notify barrier arrival at the last warpgroup wave
                        if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                            empty_barrier_arrive(s);

                        auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                        bool predicate = local_idx == 0;
                        // Promote with scales
                        #pragma unroll
                        for (int i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                            shifted_accum[i * 4 + 0] += (predicate ? scale_0_0[i] : scale_1_0[i]) * accum[i * 4 + 0];
                            shifted_accum[i * 4 + 1] += (predicate ? scale_0_1[i] : scale_1_1[i]) * accum[i * 4 + 1];
                            shifted_accum[i * 4 + 2] += (predicate ? scale_0_0[i] : scale_1_0[i]) * accum[i * 4 + 2];
                            shifted_accum[i * 4 + 3] += (predicate ? scale_0_1[i] : scale_1_1[i]) * accum[i * 4 + 3];
                        }
                    }
                }

                // Wait unaligned cases
                #pragma unroll
                for (uint32_t s = kNumInnerStages; s < kNumStages; ++ s) {
                    full_barriers[s]->wait((scheduler.current_iter * kNumIterations + k_iter) & 1);
                    empty_barrier_arrive(s);
                }
            });

            // TMA swizzle
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_M = kSwizzleDMode == 0 ? BLOCK_M : (kSwizzleDMode / kNumElemBytes);
            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_M / TMA_D_BLOCK_M)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier(kNumMathThreads).sync();

            // Write back to shared memory using STSM
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            // TMA swizzle 128B  bm=64,128,256
            DG_STATIC_ASSERT(kSwizzleDMode==128 || kSwizzleDMode==0, "swizzle 128B  bm=64,128,256");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                if constexpr (kSwizzleDMode == 128) {
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        uint8_t* smem_ptr = nullptr;
                        constexpr int kNumBankGroupBytes = 16;

                        auto row = lane_idx % 8;  // thread 0 - 15
                        auto col = (warp_idx % 4) * 2 + lane_idx / 8;
                        col ^= row % (kSwizzleDMode / 16);

                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                                m_offset * BLOCK_N * kNumElemBytes +                     // Wave offset
                                warp_idx / 4 * BLOCK_N * kSwizzleDMode +                         // Warp offset
                                i * 8 * kSwizzleDMode +              // Swizzle atom offset (constants)
                                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset

                        SM90_U32x2_STSM_T<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
                else
                {
                    // no swizzle
                    int tid = 0;
                    if(lane_idx<8){
                        tid = lane_idx * BLOCK_M;
                    }else if(lane_idx<16){
                        tid = (lane_idx - 8) * BLOCK_M + 8;
                    }else if(lane_idx<24){
                        tid = (lane_idx - 8) * BLOCK_M;
                    }else{
                        tid = (lane_idx - 16) * BLOCK_M + 8;
                    }
                    #pragma unroll
                    for (auto i = 0; i < WGMMA::kNumAccum / 8; ++ i) {
                        SM90_U32x4_STSM_T<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 8 + 0], shifted_accum[i * 8 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 8 + 2], shifted_accum[i * 8 + 3]}),
                            __float22bfloat162_rn({shifted_accum[i * 8 + 4], shifted_accum[i * 8 + 5]}),
                            __float22bfloat162_rn({shifted_accum[i * 8 + 6], shifted_accum[i * 8 + 7]}),
                            smem_d + warp_idx * 16 + i * 16 * BLOCK_M + tid + m_offset
                        );
                    }
                    if constexpr (WGMMA::kNumAccum % 8 != 0) {
                        SM90_U32x2_STSM_T<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[WGMMA::kNumAccum / 8 * 8 + 0], shifted_accum[WGMMA::kNumAccum / 8 * 8 + 1]}),
                            __float22bfloat162_rn({shifted_accum[WGMMA::kNumAccum / 8 * 8 + 2], shifted_accum[WGMMA::kNumAccum / 8 * 8 + 3]}),
                            smem_d + warp_idx * 16 + WGMMA::kNumAccum / 8 * 16 * BLOCK_M + tid + m_offset
                        );
                    }
                }
            }

            if constexpr (kGemmType == GemmType::GroupedMasked) {
                DG_STATIC_ASSERT(kSwizzleDMode==0, "write_result_to_gmem doesn't support swizzle");
                auto n_idx = BLOCK_N * n_block_idx;
                auto n_global_idx = scheduler.get_global_idx(shape_n, BLOCK_N, n_block_idx);
                bool cross_boundary = (n_idx + BLOCK_N) > shape_n;
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                if (!cross_boundary){
                    // Use TMA store to write back to global memory
                    DG_STATIC_ASSERT(kNumMathThreads >= BLOCK_M / TMA_D_BLOCK_M, "Too many TMA blocks");
                    if (threadIdx.x < BLOCK_M / TMA_D_BLOCK_M) {
                        auto in_block_m_offset = threadIdx.x * TMA_D_BLOCK_M;
                        auto smem_ptr = smem_d + in_block_m_offset * BLOCK_N;
                        cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                                    m_block_idx * BLOCK_M + in_block_m_offset, n_global_idx);
                        cute::tma_store_arrive();
                    }
                } else {
                    __nv_bfloat16* gmem_d_this_block = gmem_d + n_global_idx * SHAPE_M;
                    constexpr int NUM_WARPS =
                        (get_num_threads_per_sm<kNumTMAThreads, kNumMathThreadsPerGroup>(BLOCK_M) - 128) / 32;
                    write_result_to_gmem<BLOCK_N, BLOCK_M, NUM_WARPS>(gmem_d_this_block, smem_d, n_global_idx,
                                                                        (scheduler.curr_group_idx + 1) * shape_n, m_block_idx * BLOCK_M,
                                                                        SHAPE_M, SHAPE_M);
                }
            } else {  // normal gemm
                cute::tma_store_fence();
                cutlass::arch::NamedBarrier(kNumMathThreads).sync();
                // TMA swizzle128B or no swizzle
                DG_STATIC_ASSERT(kNumMathThreads >= BLOCK_M / TMA_D_BLOCK_M, "Too many TMA blocks");
                if (threadIdx.x < BLOCK_M / TMA_D_BLOCK_M) {
                    auto in_block_m_offset = threadIdx.x * TMA_D_BLOCK_M;
                    auto smem_ptr = smem_d + in_block_m_offset * BLOCK_N;
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr,
                                                m_block_idx * BLOCK_M + in_block_m_offset, scheduler.get_global_idx(shape_n, BLOCK_N, n_block_idx));
                    cute::tma_store_arrive();
                }

            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}
};  // namespace deep_gemm

#pragma clang diagnostic pop
