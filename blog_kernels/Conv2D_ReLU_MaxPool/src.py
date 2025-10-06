import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math # For torch.finfo

fused_conv_relu_maxpool_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // For at::cuda::getCurrentCUDAStream()
#include <mma.h>
#include <cuda_fp16.h>
#include <float.h> // For FLT_MAX, though std::numeric_limits is preferred in C++
#include <limits>  // For std::numeric_limits

using namespace nvcuda;

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Skew padding for shared memory to avoid bank conflicts
#define SKEW_HALF 8 // 8 half elements (16 bytes)

// CUDA built-in warpSize is 32 for supported architectures (sm_70+)
#define CUDA_WARP_SIZE_CONST 32 

// Threadblock configuration
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * CUDA_WARP_SIZE_CONST) 

// Macro-tile dimensions computed by a threadblock
#define BLOCK_M_TILES_WMMA 8
#define BLOCK_N_TILES_WMMA 8

#define TILE_M_PER_BLOCK (BLOCK_M_TILES_WMMA * WMMA_M) // e.g., 8 * 16 = 128 (for C_out dimension)
#define TILE_N_PER_BLOCK (BLOCK_N_TILES_WMMA * WMMA_N) // e.g., 8 * 16 = 128 (for N_batch * H_conv_out * W_conv_out dimension)

// Vector size for shared memory writes (4 half elements = 64 bits)
#define VECTOR_SIZE_H_EFFECTIVE 4 

// Struct to hold precomputed N-dimension GEMM indices for convolution output
struct NDecomposedConv {
    int ow_conv_eff;
    int oh_conv_eff;
    int n_batch_idx;
    bool isValidPixel; // True if this pixel_idx is within N_gemm_conv bounds
    int h_in_base; 
    int w_in_base; 
};

// Optimized atomic max for float, leveraging integer atomics.
// Assumes val >= 0.0f and *addr is initialized to a very negative number or a non-negative float.
__device__ inline void atomicMax_float_optimized(float* addr, float val) {
    // Preconditions:
    // 1. val >= 0.0f (guaranteed by ReLU).
    // 2. The initial value at *addr can be -std::numeric_limits<float>::max() or any other float.
    //    The comparison works correctly due to how float bit patterns map to signed integers:
    //    - Positive floats map to positive integers (MSB 0).
    //    - Negative floats map to negative integers (MSB 1).
    //    - For non-negative floats, their natural order is preserved in their int representation.
    //    - A non-negative float's int representation will be greater than a negative float's int representation.
    
    // Cast the float pointer to an int pointer.
    // Use __float_as_int to get the bitwise integer representation of val.
    // atomicMax for signed integers will then perform the correct operation.
    atomicMax((int*)addr, __float_as_int(val));
}

// Helper to pack 4 halfs into an int2 (64-bit)
// Stores h0, h1 in x component, h2, h3 in y component of int2
// us0 (from h0) -> int2.x low 16 bits
// us1 (from h1) -> int2.x high 16 bits
// us2 (from h2) -> int2.y low 16 bits
// us3 (from h3) -> int2.y high 16 bits
__device__ inline int2 __pack_half4_to_int2(half h0, half h1, half h2, half h3) {
    unsigned short us0 = __half_as_ushort(h0);
    unsigned short us1 = __half_as_ushort(h1);
    unsigned short us2 = __half_as_ushort(h2);
    unsigned short us3 = __half_as_ushort(h3);
    return make_int2(
        static_cast<int>(((static_cast<unsigned int>(us1)) << 16) | static_cast<unsigned int>(us0)),
        static_cast<int>(((static_cast<unsigned int>(us3)) << 16) | static_cast<unsigned int>(us2))
    );
}


__global__ void fused_conv_relu_maxpool_wmma_kernel(
    const float* __restrict__ input_ptr,    // Input: (N, Cin, Hin, Win)
    const float* __restrict__ weight_ptr,   // Weights: (Cout, Cin, Kh, Kw)
    const float* __restrict__ bias_ptr,     // Bias: (Cout) or nullptr
    float* __restrict__ output_final_ptr,   // Output: (N, Cout, H_pool_out, W_pool_out), PRE-INITIALIZED TO -std::numeric_limits<float>::max()
    const int N_batch, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int H_conv_out, const int W_conv_out, // Convolution output dimensions
    const int M_gemm_conv, // C_out
    const int N_gemm_conv, // N_batch * H_conv_out * W_conv_out
    const int K_gemm_conv, // C_in * K_h * K_w
    const int pool_kernel_h, const int pool_kernel_w,
    const int pool_stride_h, const int pool_stride_w,
    const int H_pool_out, const int W_pool_out      // Final pooled output dimensions (effective, >0)
) {
    // Thread identification
    const int warp_id = threadIdx.x / warpSize;        // 0 .. WARPS_PER_BLOCK-1
    const int lane_id = threadIdx.x % warpSize;        // 0 .. 31 (or warpSize-1)

    // Top-left corner of the macro-tile this block is responsible for in GEMM terms (for convolution)
    const int block_row_gemm_start = TILE_M_PER_BLOCK * blockIdx.y; // Output channel base for this block
    const int block_col_gemm_start = TILE_N_PER_BLOCK * blockIdx.x; // Flattened conv output spatial base

    // Shared memory for tiles of A (weights) and B (input/im2col) - Double Buffered for K-loop pipelining
    __shared__ half Asub_pipe[2][TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_pipe[2][TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

    // Shared memory for precomputed N-indices (for convolution output)
    __shared__ NDecomposedConv n_params_conv_sh[TILE_N_PER_BLOCK];

    // Shared memory for output stage (per-warp buffers for conv results before ReLU and pooling)
    __shared__ float C_shmem_conv_buffers[WARPS_PER_BLOCK][WMMA_M][WMMA_N];

    // Accumulator fragments per warp.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[BLOCK_N_TILES_WMMA];
    #pragma unroll
    for (int i = 0; i < BLOCK_N_TILES_WMMA; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    // Populate n_params_conv_sh once at the beginning of the kernel
    if (threadIdx.x < TILE_N_PER_BLOCK) {
        int r_b_tile_idx = threadIdx.x; 
        int current_conv_pixel_idx_flat = block_col_gemm_start + r_b_tile_idx;

        if (current_conv_pixel_idx_flat < N_gemm_conv) {
            n_params_conv_sh[r_b_tile_idx].ow_conv_eff = current_conv_pixel_idx_flat % W_conv_out;
            int temp_div_wout_conv = current_conv_pixel_idx_flat / W_conv_out;
            n_params_conv_sh[r_b_tile_idx].oh_conv_eff = temp_div_wout_conv % H_conv_out;
            n_params_conv_sh[r_b_tile_idx].n_batch_idx = temp_div_wout_conv / H_conv_out;
            n_params_conv_sh[r_b_tile_idx].isValidPixel = true;
            n_params_conv_sh[r_b_tile_idx].h_in_base = n_params_conv_sh[r_b_tile_idx].oh_conv_eff * stride_h - pad_h;
            n_params_conv_sh[r_b_tile_idx].w_in_base = n_params_conv_sh[r_b_tile_idx].ow_conv_eff * stride_w - pad_w;
        } else {
            n_params_conv_sh[r_b_tile_idx].isValidPixel = false;
            n_params_conv_sh[r_b_tile_idx].ow_conv_eff = 0; 
            n_params_conv_sh[r_b_tile_idx].oh_conv_eff = 0;
            n_params_conv_sh[r_b_tile_idx].n_batch_idx = 0;
            n_params_conv_sh[r_b_tile_idx].h_in_base = 0; 
            n_params_conv_sh[r_b_tile_idx].w_in_base = 0;
        }
    }
    __syncthreads();

    // Constants for data loading strategy
    const int NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP = WMMA_K / VECTOR_SIZE_H_EFFECTIVE; 
    const int NUM_ROW_GROUPS_PER_BLOCK_DIM = THREADS_PER_BLOCK / NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP; 

    // --- Cache NDecomposedConv data for Bsub_pipe loading ---
    const int MAX_R_B_TILES_PER_THREAD_B_LOAD = (TILE_N_PER_BLOCK + NUM_ROW_GROUPS_PER_BLOCK_DIM - 1) / NUM_ROW_GROUPS_PER_BLOCK_DIM;
    NDecomposedConv local_cnp_cache[MAX_R_B_TILES_PER_THREAD_B_LOAD];
    int num_r_b_tiles_this_thread_loads = 0;
    const int n_row_group_id_B_for_cache = threadIdx.x / NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;

    for (int r_b_idx_iter = n_row_group_id_B_for_cache; r_b_idx_iter < TILE_N_PER_BLOCK; r_b_idx_iter += NUM_ROW_GROUPS_PER_BLOCK_DIM) {
        if (num_r_b_tiles_this_thread_loads < MAX_R_B_TILES_PER_THREAD_B_LOAD) {
            local_cnp_cache[num_r_b_tiles_this_thread_loads] = n_params_conv_sh[r_b_idx_iter];
            num_r_b_tiles_this_thread_loads++;
        } else {
            // This should not be hit if MAX_R_B_TILES_PER_THREAD_B_LOAD is calculated correctly
            // as the maximum number of tiles any thread group will process.
            break; 
        }
    }
    // --- End Cache NDecomposedConv data ---

    int num_k_tiles = (K_gemm_conv + WMMA_K - 1) / WMMA_K;
    
    // --- Prologue: Load first k-tile (k_tile_iter = 0) into pipe_idx = 0 ---
    if (num_k_tiles > 0) { 
        int k_tile_start_prologue = 0; 
        int current_pipe_idx_prologue = 0; 

        // Load Asub_pipe[0]
        {
            int h_group_idx_in_k_A = threadIdx.x % NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
            int shmem_k_start_for_h_group_A = h_group_idx_in_k_A * VECTOR_SIZE_H_EFFECTIVE;
            int k_global_A_base = k_tile_start_prologue + shmem_k_start_for_h_group_A;
            int k_global_A_0 = k_global_A_base + 0;
            int k_global_A_1 = k_global_A_base + 1;
            int k_global_A_2 = k_global_A_base + 2;
            int k_global_A_3 = k_global_A_base + 3;

            // --- START OPTIMIZED K-INDEX DECOMPOSITION FOR A (PROLOGUE) ---
            bool is_valid_k_A_0, is_valid_k_A_1, is_valid_k_A_2, is_valid_k_A_3;
            int kw_eff_reg_A_0=0, kh_eff_reg_A_0=0, ic_eff_reg_A_0=0;
            int kw_eff_reg_A_1=0, kh_eff_reg_A_1=0, ic_eff_reg_A_1=0;
            int kw_eff_reg_A_2=0, kh_eff_reg_A_2=0, ic_eff_reg_A_2=0;
            int kw_eff_reg_A_3=0, kh_eff_reg_A_3=0, ic_eff_reg_A_3=0;
            int k_val;

            k_val = k_global_A_0; is_valid_k_A_0 = (k_val < K_gemm_conv); if (is_valid_k_A_0) { kw_eff_reg_A_0 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_0 = td % K_h; ic_eff_reg_A_0 = td / K_h; }
            k_val = k_global_A_1; is_valid_k_A_1 = (k_val < K_gemm_conv); if (is_valid_k_A_1) { if (is_valid_k_A_0) { kw_eff_reg_A_1 = kw_eff_reg_A_0 + 1; kh_eff_reg_A_1 = kh_eff_reg_A_0; ic_eff_reg_A_1 = ic_eff_reg_A_0; if (kw_eff_reg_A_1 == K_w) { kw_eff_reg_A_1 = 0; kh_eff_reg_A_1++; if (kh_eff_reg_A_1 == K_h) { kh_eff_reg_A_1 = 0; ic_eff_reg_A_1++; } } } else { kw_eff_reg_A_1 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_1 = td % K_h; ic_eff_reg_A_1 = td / K_h; } }
            k_val = k_global_A_2; is_valid_k_A_2 = (k_val < K_gemm_conv); if (is_valid_k_A_2) { if (is_valid_k_A_1) { kw_eff_reg_A_2 = kw_eff_reg_A_1 + 1; kh_eff_reg_A_2 = kh_eff_reg_A_1; ic_eff_reg_A_2 = ic_eff_reg_A_1; if (kw_eff_reg_A_2 == K_w) { kw_eff_reg_A_2 = 0; kh_eff_reg_A_2++; if (kh_eff_reg_A_2 == K_h) { kh_eff_reg_A_2 = 0; ic_eff_reg_A_2++; } } } else { kw_eff_reg_A_2 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_2 = td % K_h; ic_eff_reg_A_2 = td / K_h; } }
            k_val = k_global_A_3; is_valid_k_A_3 = (k_val < K_gemm_conv); if (is_valid_k_A_3) { if (is_valid_k_A_2) { kw_eff_reg_A_3 = kw_eff_reg_A_2 + 1; kh_eff_reg_A_3 = kh_eff_reg_A_2; ic_eff_reg_A_3 = ic_eff_reg_A_2; if (kw_eff_reg_A_3 == K_w) { kw_eff_reg_A_3 = 0; kh_eff_reg_A_3++; if (kh_eff_reg_A_3 == K_h) { kh_eff_reg_A_3 = 0; ic_eff_reg_A_3++; } } } else { kw_eff_reg_A_3 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_3 = td % K_h; ic_eff_reg_A_3 = td / K_h; } }
            // --- END OPTIMIZED K-INDEX DECOMPOSITION FOR A (PROLOGUE) ---
            
            int m_row_group_id_A = threadIdx.x / NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
            #pragma unroll
            for (int r_a_tile_base = m_row_group_id_A; r_a_tile_base < TILE_M_PER_BLOCK; r_a_tile_base += NUM_ROW_GROUPS_PER_BLOCK_DIM) {
                int oc_idx = block_row_gemm_start + r_a_tile_base;
                float w0=0.0f, w1=0.0f, w2=0.0f, w3=0.0f;
                if (oc_idx < C_out) {
                    if(is_valid_k_A_0) w0 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_0 * K_h * K_w + kh_eff_reg_A_0 * K_w + kw_eff_reg_A_0];
                    if(is_valid_k_A_1) w1 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_1 * K_h * K_w + kh_eff_reg_A_1 * K_w + kw_eff_reg_A_1];
                    if(is_valid_k_A_2) w2 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_2 * K_h * K_w + kh_eff_reg_A_2 * K_w + kw_eff_reg_A_2];
                    if(is_valid_k_A_3) w3 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_3 * K_h * K_w + kh_eff_reg_A_3 * K_w + kw_eff_reg_A_3];
                }
                reinterpret_cast<int2*>(&Asub_pipe[current_pipe_idx_prologue][r_a_tile_base][shmem_k_start_for_h_group_A])[0] = 
                    __pack_half4_to_int2(__float2half(w0), __float2half(w1), __float2half(w2), __float2half(w3));
            }
        }
        // Load Bsub_pipe[0]
        {
            int h_group_idx_in_k_B = threadIdx.x % NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
            int shmem_k_start_for_h_group_B = h_group_idx_in_k_B * VECTOR_SIZE_H_EFFECTIVE;
            int k_global_B_base = k_tile_start_prologue + shmem_k_start_for_h_group_B;
            int k_global_B_0 = k_global_B_base + 0;
            int k_global_B_1 = k_global_B_base + 1;
            int k_global_B_2 = k_global_B_base + 2;
            int k_global_B_3 = k_global_B_base + 3;

            // --- START OPTIMIZED K-INDEX DECOMPOSITION FOR B (PROLOGUE) ---
            bool is_valid_k_B_0, is_valid_k_B_1, is_valid_k_B_2, is_valid_k_B_3;
            int kw_eff_reg_B_0=0, kh_eff_reg_B_0=0, ic_eff_reg_B_0=0;
            int kw_eff_reg_B_1=0, kh_eff_reg_B_1=0, ic_eff_reg_B_1=0;
            int kw_eff_reg_B_2=0, kh_eff_reg_B_2=0, ic_eff_reg_B_2=0;
            int kw_eff_reg_B_3=0, kh_eff_reg_B_3=0, ic_eff_reg_B_3=0;
            int k_val;

            k_val = k_global_B_0; is_valid_k_B_0 = (k_val < K_gemm_conv); if (is_valid_k_B_0) { kw_eff_reg_B_0 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_0 = td % K_h; ic_eff_reg_B_0 = td / K_h; }
            k_val = k_global_B_1; is_valid_k_B_1 = (k_val < K_gemm_conv); if (is_valid_k_B_1) { if (is_valid_k_B_0) { kw_eff_reg_B_1 = kw_eff_reg_B_0 + 1; kh_eff_reg_B_1 = kh_eff_reg_B_0; ic_eff_reg_B_1 = ic_eff_reg_B_0; if (kw_eff_reg_B_1 == K_w) { kw_eff_reg_B_1 = 0; kh_eff_reg_B_1++; if (kh_eff_reg_B_1 == K_h) { kh_eff_reg_B_1 = 0; ic_eff_reg_B_1++; } } } else { kw_eff_reg_B_1 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_1 = td % K_h; ic_eff_reg_B_1 = td / K_h; } }
            k_val = k_global_B_2; is_valid_k_B_2 = (k_val < K_gemm_conv); if (is_valid_k_B_2) { if (is_valid_k_B_1) { kw_eff_reg_B_2 = kw_eff_reg_B_1 + 1; kh_eff_reg_B_2 = kh_eff_reg_B_1; ic_eff_reg_B_2 = ic_eff_reg_B_1; if (kw_eff_reg_B_2 == K_w) { kw_eff_reg_B_2 = 0; kh_eff_reg_B_2++; if (kh_eff_reg_B_2 == K_h) { kh_eff_reg_B_2 = 0; ic_eff_reg_B_2++; } } } else { kw_eff_reg_B_2 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_2 = td % K_h; ic_eff_reg_B_2 = td / K_h; } }
            k_val = k_global_B_3; is_valid_k_B_3 = (k_val < K_gemm_conv); if (is_valid_k_B_3) { if (is_valid_k_B_2) { kw_eff_reg_B_3 = kw_eff_reg_B_2 + 1; kh_eff_reg_B_3 = kh_eff_reg_B_2; ic_eff_reg_B_3 = ic_eff_reg_B_2; if (kw_eff_reg_B_3 == K_w) { kw_eff_reg_B_3 = 0; kh_eff_reg_B_3++; if (kh_eff_reg_B_3 == K_h) { kh_eff_reg_B_3 = 0; ic_eff_reg_B_3++; } } } else { kw_eff_reg_B_3 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_3 = td % K_h; ic_eff_reg_B_3 = td / K_h; } }
            // --- END OPTIMIZED K-INDEX DECOMPOSITION FOR B (PROLOGUE) ---

            #pragma unroll
            for (int cache_idx = 0; cache_idx < num_r_b_tiles_this_thread_loads; ++cache_idx) {
                int r_b_tile_base = n_row_group_id_B_for_cache + cache_idx * NUM_ROW_GROUPS_PER_BLOCK_DIM;
                const NDecomposedConv& cnp = local_cnp_cache[cache_idx]; // Use cached data

                float i0=0.0f, i1=0.0f, i2=0.0f, i3=0.0f;
                if (cnp.isValidPixel) {
                    if (is_valid_k_B_0) {
                        int h_in_eff_0 = cnp.h_in_base + kh_eff_reg_B_0; int w_in_eff_0 = cnp.w_in_base + kw_eff_reg_B_0;
                        if (h_in_eff_0>=0 && h_in_eff_0<H_in && w_in_eff_0>=0 && w_in_eff_0<W_in) 
                            i0 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_0*H_in*W_in + h_in_eff_0*W_in + w_in_eff_0];
                    }
                    if (is_valid_k_B_1) {
                        int h_in_eff_1 = cnp.h_in_base + kh_eff_reg_B_1; int w_in_eff_1 = cnp.w_in_base + kw_eff_reg_B_1;
                        if (h_in_eff_1>=0 && h_in_eff_1<H_in && w_in_eff_1>=0 && w_in_eff_1<W_in)
                            i1 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_1*H_in*W_in + h_in_eff_1*W_in + w_in_eff_1];
                    }
                    if (is_valid_k_B_2) {
                        int h_in_eff_2 = cnp.h_in_base + kh_eff_reg_B_2; int w_in_eff_2 = cnp.w_in_base + kw_eff_reg_B_2;
                        if (h_in_eff_2>=0 && h_in_eff_2<H_in && w_in_eff_2>=0 && w_in_eff_2<W_in)
                            i2 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_2*H_in*W_in + h_in_eff_2*W_in + w_in_eff_2];
                    }
                    if (is_valid_k_B_3) {
                        int h_in_eff_3 = cnp.h_in_base + kh_eff_reg_B_3; int w_in_eff_3 = cnp.w_in_base + kw_eff_reg_B_3;
                        if (h_in_eff_3>=0 && h_in_eff_3<H_in && w_in_eff_3>=0 && w_in_eff_3<W_in)
                            i3 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_3*H_in*W_in + h_in_eff_3*W_in + w_in_eff_3];
                    }
                }
                reinterpret_cast<int2*>(&Bsub_pipe[current_pipe_idx_prologue][r_b_tile_base][shmem_k_start_for_h_group_B])[0] = 
                    __pack_half4_to_int2(__float2half(i0), __float2half(i1), __float2half(i2), __float2half(i3));
            }
        }
    }

    // Loop over the K_gemm_conv dimension in tiles of WMMA_K
    for (int k_tile_iter = 0; k_tile_iter < num_k_tiles; ++k_tile_iter) {
        __syncthreads(); 

        int compute_pipe_idx = k_tile_iter % 2;
        int load_pipe_idx = (k_tile_iter + 1) % 2;

        // --- Load Stage for next k-tile (k_tile_iter + 1) into load_pipe_idx ---
        int k_tile_start_for_load = (k_tile_iter + 1) * WMMA_K;
        if (k_tile_start_for_load < K_gemm_conv) { 
            // Load Asub_pipe[load_pipe_idx]
            { 
                int h_group_idx_in_k_A = threadIdx.x % NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
                int shmem_k_start_for_h_group_A = h_group_idx_in_k_A * VECTOR_SIZE_H_EFFECTIVE;
                int k_global_A_base = k_tile_start_for_load + shmem_k_start_for_h_group_A;
                int k_global_A_0 = k_global_A_base + 0;
                int k_global_A_1 = k_global_A_base + 1;
                int k_global_A_2 = k_global_A_base + 2;
                int k_global_A_3 = k_global_A_base + 3;

                // --- START OPTIMIZED K-INDEX DECOMPOSITION FOR A (MAIN LOOP) ---
                bool is_valid_k_A_0, is_valid_k_A_1, is_valid_k_A_2, is_valid_k_A_3;
                int kw_eff_reg_A_0=0, kh_eff_reg_A_0=0, ic_eff_reg_A_0=0;
                int kw_eff_reg_A_1=0, kh_eff_reg_A_1=0, ic_eff_reg_A_1=0;
                int kw_eff_reg_A_2=0, kh_eff_reg_A_2=0, ic_eff_reg_A_2=0;
                int kw_eff_reg_A_3=0, kh_eff_reg_A_3=0, ic_eff_reg_A_3=0;
                int k_val;

                k_val = k_global_A_0; is_valid_k_A_0 = (k_val < K_gemm_conv); if (is_valid_k_A_0) { kw_eff_reg_A_0 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_0 = td % K_h; ic_eff_reg_A_0 = td / K_h; }
                k_val = k_global_A_1; is_valid_k_A_1 = (k_val < K_gemm_conv); if (is_valid_k_A_1) { if (is_valid_k_A_0) { kw_eff_reg_A_1 = kw_eff_reg_A_0 + 1; kh_eff_reg_A_1 = kh_eff_reg_A_0; ic_eff_reg_A_1 = ic_eff_reg_A_0; if (kw_eff_reg_A_1 == K_w) { kw_eff_reg_A_1 = 0; kh_eff_reg_A_1++; if (kh_eff_reg_A_1 == K_h) { kh_eff_reg_A_1 = 0; ic_eff_reg_A_1++; } } } else { kw_eff_reg_A_1 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_1 = td % K_h; ic_eff_reg_A_1 = td / K_h; } }
                k_val = k_global_A_2; is_valid_k_A_2 = (k_val < K_gemm_conv); if (is_valid_k_A_2) { if (is_valid_k_A_1) { kw_eff_reg_A_2 = kw_eff_reg_A_1 + 1; kh_eff_reg_A_2 = kh_eff_reg_A_1; ic_eff_reg_A_2 = ic_eff_reg_A_1; if (kw_eff_reg_A_2 == K_w) { kw_eff_reg_A_2 = 0; kh_eff_reg_A_2++; if (kh_eff_reg_A_2 == K_h) { kh_eff_reg_A_2 = 0; ic_eff_reg_A_2++; } } } else { kw_eff_reg_A_2 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_2 = td % K_h; ic_eff_reg_A_2 = td / K_h; } }
                k_val = k_global_A_3; is_valid_k_A_3 = (k_val < K_gemm_conv); if (is_valid_k_A_3) { if (is_valid_k_A_2) { kw_eff_reg_A_3 = kw_eff_reg_A_2 + 1; kh_eff_reg_A_3 = kh_eff_reg_A_2; ic_eff_reg_A_3 = ic_eff_reg_A_2; if (kw_eff_reg_A_3 == K_w) { kw_eff_reg_A_3 = 0; kh_eff_reg_A_3++; if (kh_eff_reg_A_3 == K_h) { kh_eff_reg_A_3 = 0; ic_eff_reg_A_3++; } } } else { kw_eff_reg_A_3 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_A_3 = td % K_h; ic_eff_reg_A_3 = td / K_h; } }
                // --- END OPTIMIZED K-INDEX DECOMPOSITION FOR A (MAIN LOOP) ---
                
                int m_row_group_id_A = threadIdx.x / NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
                #pragma unroll
                for (int r_a_tile_base = m_row_group_id_A; r_a_tile_base < TILE_M_PER_BLOCK; r_a_tile_base += NUM_ROW_GROUPS_PER_BLOCK_DIM) {
                    int oc_idx = block_row_gemm_start + r_a_tile_base;
                    float w0=0.0f, w1=0.0f, w2=0.0f, w3=0.0f;
                    if (oc_idx < C_out) {
                        if(is_valid_k_A_0) w0 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_0 * K_h * K_w + kh_eff_reg_A_0 * K_w + kw_eff_reg_A_0];
                        if(is_valid_k_A_1) w1 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_1 * K_h * K_w + kh_eff_reg_A_1 * K_w + kw_eff_reg_A_1];
                        if(is_valid_k_A_2) w2 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_2 * K_h * K_w + kh_eff_reg_A_2 * K_w + kw_eff_reg_A_2];
                        if(is_valid_k_A_3) w3 = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff_reg_A_3 * K_h * K_w + kh_eff_reg_A_3 * K_w + kw_eff_reg_A_3];
                    }
                    reinterpret_cast<int2*>(&Asub_pipe[load_pipe_idx][r_a_tile_base][shmem_k_start_for_h_group_A])[0] = 
                        __pack_half4_to_int2(__float2half(w0), __float2half(w1), __float2half(w2), __float2half(w3));
                }
            } 
            // Load Bsub_pipe[load_pipe_idx]
            { 
                int h_group_idx_in_k_B = threadIdx.x % NUM_H_ELEMENTS_PER_K_TILE_PER_THREAD_GROUP;
                int shmem_k_start_for_h_group_B = h_group_idx_in_k_B * VECTOR_SIZE_H_EFFECTIVE;
                int k_global_B_base = k_tile_start_for_load + shmem_k_start_for_h_group_B;
                int k_global_B_0 = k_global_B_base + 0;
                int k_global_B_1 = k_global_B_base + 1;
                int k_global_B_2 = k_global_B_base + 2;
                int k_global_B_3 = k_global_B_base + 3;

                // --- START OPTIMIZED K-INDEX DECOMPOSITION FOR B (MAIN LOOP) ---
                bool is_valid_k_B_0, is_valid_k_B_1, is_valid_k_B_2, is_valid_k_B_3;
                int kw_eff_reg_B_0=0, kh_eff_reg_B_0=0, ic_eff_reg_B_0=0;
                int kw_eff_reg_B_1=0, kh_eff_reg_B_1=0, ic_eff_reg_B_1=0;
                int kw_eff_reg_B_2=0, kh_eff_reg_B_2=0, ic_eff_reg_B_2=0;
                int kw_eff_reg_B_3=0, kh_eff_reg_B_3=0, ic_eff_reg_B_3=0;
                int k_val;

                k_val = k_global_B_0; is_valid_k_B_0 = (k_val < K_gemm_conv); if (is_valid_k_B_0) { kw_eff_reg_B_0 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_0 = td % K_h; ic_eff_reg_B_0 = td / K_h; }
                k_val = k_global_B_1; is_valid_k_B_1 = (k_val < K_gemm_conv); if (is_valid_k_B_1) { if (is_valid_k_B_0) { kw_eff_reg_B_1 = kw_eff_reg_B_0 + 1; kh_eff_reg_B_1 = kh_eff_reg_B_0; ic_eff_reg_B_1 = ic_eff_reg_B_0; if (kw_eff_reg_B_1 == K_w) { kw_eff_reg_B_1 = 0; kh_eff_reg_B_1++; if (kh_eff_reg_B_1 == K_h) { kh_eff_reg_B_1 = 0; ic_eff_reg_B_1++; } } } else { kw_eff_reg_B_1 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_1 = td % K_h; ic_eff_reg_B_1 = td / K_h; } }
                k_val = k_global_B_2; is_valid_k_B_2 = (k_val < K_gemm_conv); if (is_valid_k_B_2) { if (is_valid_k_B_1) { kw_eff_reg_B_2 = kw_eff_reg_B_1 + 1; kh_eff_reg_B_2 = kh_eff_reg_B_1; ic_eff_reg_B_2 = ic_eff_reg_B_1; if (kw_eff_reg_B_2 == K_w) { kw_eff_reg_B_2 = 0; kh_eff_reg_B_2++; if (kh_eff_reg_B_2 == K_h) { kh_eff_reg_B_2 = 0; ic_eff_reg_B_2++; } } } else { kw_eff_reg_B_2 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_2 = td % K_h; ic_eff_reg_B_2 = td / K_h; } }
                k_val = k_global_B_3; is_valid_k_B_3 = (k_val < K_gemm_conv); if (is_valid_k_B_3) { if (is_valid_k_B_2) { kw_eff_reg_B_3 = kw_eff_reg_B_2 + 1; kh_eff_reg_B_3 = kh_eff_reg_B_2; ic_eff_reg_B_3 = ic_eff_reg_B_2; if (kw_eff_reg_B_3 == K_w) { kw_eff_reg_B_3 = 0; kh_eff_reg_B_3++; if (kh_eff_reg_B_3 == K_h) { kh_eff_reg_B_3 = 0; ic_eff_reg_B_3++; } } } else { kw_eff_reg_B_3 = k_val % K_w; int td = k_val / K_w; kh_eff_reg_B_3 = td % K_h; ic_eff_reg_B_3 = td / K_h; } }
                // --- END OPTIMIZED K-INDEX DECOMPOSITION FOR B (MAIN LOOP) ---

                #pragma unroll
                for (int cache_idx = 0; cache_idx < num_r_b_tiles_this_thread_loads; ++cache_idx) {
                    int r_b_tile_base = n_row_group_id_B_for_cache + cache_idx * NUM_ROW_GROUPS_PER_BLOCK_DIM;
                    const NDecomposedConv& cnp = local_cnp_cache[cache_idx]; // Use cached data

                    float i0=0.0f, i1=0.0f, i2=0.0f, i3=0.0f;
                    if (cnp.isValidPixel) {
                        if (is_valid_k_B_0) {
                            int h_in_eff_0 = cnp.h_in_base + kh_eff_reg_B_0; int w_in_eff_0 = cnp.w_in_base + kw_eff_reg_B_0;
                            if (h_in_eff_0>=0 && h_in_eff_0<H_in && w_in_eff_0>=0 && w_in_eff_0<W_in) 
                                i0 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_0*H_in*W_in + h_in_eff_0*W_in + w_in_eff_0];
                        }
                        if (is_valid_k_B_1) {
                            int h_in_eff_1 = cnp.h_in_base + kh_eff_reg_B_1; int w_in_eff_1 = cnp.w_in_base + kw_eff_reg_B_1;
                            if (h_in_eff_1>=0 && h_in_eff_1<H_in && w_in_eff_1>=0 && w_in_eff_1<W_in)
                                i1 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_1*H_in*W_in + h_in_eff_1*W_in + w_in_eff_1];
                        }
                        if (is_valid_k_B_2) {
                            int h_in_eff_2 = cnp.h_in_base + kh_eff_reg_B_2; int w_in_eff_2 = cnp.w_in_base + kw_eff_reg_B_2;
                            if (h_in_eff_2>=0 && h_in_eff_2<H_in && w_in_eff_2>=0 && w_in_eff_2<W_in)
                                i2 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_2*H_in*W_in + h_in_eff_2*W_in + w_in_eff_2];
                        }
                        if (is_valid_k_B_3) {
                            int h_in_eff_3 = cnp.h_in_base + kh_eff_reg_B_3; int w_in_eff_3 = cnp.w_in_base + kw_eff_reg_B_3;
                            if (h_in_eff_3>=0 && h_in_eff_3<H_in && w_in_eff_3>=0 && w_in_eff_3<W_in)
                                i3 = input_ptr[cnp.n_batch_idx*C_in*H_in*W_in + ic_eff_reg_B_3*H_in*W_in + h_in_eff_3*W_in + w_in_eff_3];
                        }
                    }
                    reinterpret_cast<int2*>(&Bsub_pipe[load_pipe_idx][r_b_tile_base][shmem_k_start_for_h_group_B])[0] = 
                        __pack_half4_to_int2(__float2half(i0), __float2half(i1), __float2half(i2), __float2half(i3));
                }
            } 
        }

        // --- Compute Stage for current k-tile (k_tile_iter) using compute_pipe_idx ---
        int a_row_start_in_tile = warp_id * WMMA_M; 
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &Asub_pipe[compute_pipe_idx][a_row_start_in_tile][0], WMMA_K + SKEW_HALF);

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_inner_pipe[2];
        if (BLOCK_N_TILES_WMMA > 0) {
            wmma::load_matrix_sync(b_frag_inner_pipe[0], &Bsub_pipe[compute_pipe_idx][0][0], WMMA_K + SKEW_HALF);
        }
        
        int current_inner_pipe_idx = 0;
        #pragma unroll
        for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
            int next_inner_pipe_idx = 1 - current_inner_pipe_idx;
            if (n_tile < BLOCK_N_TILES_WMMA - 1) {
                int b_col_start_in_tile_next = (n_tile + 1) * WMMA_N;
                wmma::load_matrix_sync(b_frag_inner_pipe[next_inner_pipe_idx], &Bsub_pipe[compute_pipe_idx][b_col_start_in_tile_next][0], WMMA_K + SKEW_HALF);
            }
            wmma::mma_sync(acc_frag[n_tile], a_frag, b_frag_inner_pipe[current_inner_pipe_idx], acc_frag[n_tile]);
            current_inner_pipe_idx = next_inner_pipe_idx;
        }
    }
    __syncthreads(); 

    // Store results: Conv -> Bias -> ReLU -> MaxPool (using atomicMax)
    #pragma unroll
    for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
        // Store accumulator fragment to per-warp shared memory buffer
        wmma::store_matrix_sync(&C_shmem_conv_buffers[warp_id][0][0], acc_frag[n_tile], WMMA_N, wmma::mem_row_major);

        // Each thread in the warp processes its share of the WMMA_M x WMMA_N tile
        for (int elem_idx_in_frag = lane_id; elem_idx_in_frag < WMMA_M * WMMA_N; elem_idx_in_frag += warpSize) {
            int r_frag = elem_idx_in_frag / WMMA_N; // Row in the WMMA_M x WMMA_N tile
            int c_frag = elem_idx_in_frag % WMMA_N; // Col in the WMMA_M x WMMA_N tile

            // Global output channel for this element
            int oc_idx = block_row_gemm_start + (warp_id * WMMA_M) + r_frag;
            
            // Index into n_params_conv_sh, corresponds to a specific (n_batch, oh_conv, ow_conv)
            int offset_in_block_N_processing = (n_tile * WMMA_N) + c_frag;

            if (oc_idx < C_out && offset_in_block_N_processing < TILE_N_PER_BLOCK && 
                n_params_conv_sh[offset_in_block_N_processing].isValidPixel) {
                
                const NDecomposedConv& current_conv_params = n_params_conv_sh[offset_in_block_N_processing];
                int n_b_idx = current_conv_params.n_batch_idx;
                int oh_conv = current_conv_params.oh_conv_eff;
                int ow_conv = current_conv_params.ow_conv_eff;

                // Get convolution result from shared memory
                float conv_val = C_shmem_conv_buffers[warp_id][r_frag][c_frag];

                // Add bias
                if (bias_ptr != nullptr) {
                    conv_val += bias_ptr[oc_idx];
                }

                // Apply ReLU
                float relu_val = fmaxf(0.0f, conv_val);

                // Max Pooling:
                int num_h = oh_conv - pool_kernel_h + 1;
                int min_hp_val_intermediate;
                if (num_h <= 0) {
                    min_hp_val_intermediate = num_h / pool_stride_h; 
                } else {
                    min_hp_val_intermediate = (num_h + pool_stride_h - 1) / pool_stride_h; 
                }
                int actual_min_hp = (0 > min_hp_val_intermediate) ? 0 : min_hp_val_intermediate;

                int max_hp_val_intermediate = oh_conv / pool_stride_h; 
                int actual_max_hp = ((H_pool_out - 1) < max_hp_val_intermediate) ? (H_pool_out - 1) : max_hp_val_intermediate;

                int num_w = ow_conv - pool_kernel_w + 1;
                int min_wp_val_intermediate;
                if (num_w <= 0) {
                    min_wp_val_intermediate = num_w / pool_stride_w;
                } else {
                    min_wp_val_intermediate = (num_w + pool_stride_w - 1) / pool_stride_w;
                }
                int actual_min_wp = (0 > min_wp_val_intermediate) ? 0 : min_wp_val_intermediate;

                int max_wp_val_intermediate = ow_conv / pool_stride_w;
                int actual_max_wp = ((W_pool_out - 1) < max_wp_val_intermediate) ? (W_pool_out - 1) : max_wp_val_intermediate;

                for (int hp = actual_min_hp; hp <= actual_max_hp; ++hp) {
                    for (int wp = actual_min_wp; wp <= actual_max_wp; ++wp) {
                        unsigned long long final_out_idx = 
                            (unsigned long long)n_b_idx * C_out * H_pool_out * W_pool_out +
                            (unsigned long long)oc_idx * H_pool_out * W_pool_out +
                            (unsigned long long)hp * W_pool_out +
                            wp;
                        atomicMax_float_optimized(&output_final_ptr[final_out_idx], relu_val);
                    }
                }
            }
        }
    }
}


torch::Tensor fused_conv_relu_maxpool_cuda_op(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int pool_kernel_h, int pool_kernel_w, int pool_stride_h, int pool_stride_w) {

    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.device().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias has wrong shape");
    }

    const int H_conv_out = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const int W_conv_out = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    const int H_pool_out = (H_conv_out - pool_kernel_h) / pool_stride_h + 1;
    const int W_pool_out = (W_conv_out - pool_kernel_w) / pool_stride_w + 1;
    
    int H_pool_out_eff = H_pool_out > 0 ? H_pool_out : 0;
    int W_pool_out_eff = W_pool_out > 0 ? W_pool_out : 0;

    auto output_final = torch::full(
        {N_batch, C_out, H_pool_out_eff, W_pool_out_eff}, 
        -std::numeric_limits<float>::max(), 
        input.options()
    );

    if (H_conv_out <= 0 || W_conv_out <= 0 || H_pool_out <= 0 || W_pool_out <= 0) {
        return output_final; 
    }

    const int M_gemm_conv = C_out;
    const int N_gemm_conv = N_batch * H_conv_out * W_conv_out;
    const int K_gemm_conv = C_in * K_h * K_w;

    if (M_gemm_conv == 0 || N_gemm_conv == 0) { 
        return output_final; 
    }
    
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(
        (N_gemm_conv + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK, 
        (M_gemm_conv + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK  
    );

    const float* bias_ptr_data = bias.defined() ? bias.data_ptr<float>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fused_conv_relu_maxpool_wmma_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr_data,
        output_final.data_ptr<float>(),
        N_batch, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_conv_out, W_conv_out,
        M_gemm_conv, N_gemm_conv, K_gemm_conv,
        pool_kernel_h, pool_kernel_w, pool_stride_h, pool_stride_w,
        H_pool_out_eff, W_pool_out_eff 
    );
    
    AT_CUDA_CHECK(cudaGetLastError());
    
    return output_final;
}
"""

fused_conv_relu_maxpool_cuda_declaration = r"""
torch::Tensor fused_conv_relu_maxpool_cuda_op(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int pool_kernel_h, int pool_kernel_w, int pool_stride_h, int pool_stride_w);
"""

# JIT compile the CUDA kernel
custom_fused_ops = load_inline(
    name="custom_fused_conv_relu_maxpool_ops_v5", # Kept version as per user's original code
    cpp_sources=fused_conv_relu_maxpool_cuda_declaration,
    cuda_sources=fused_conv_relu_maxpool_cuda_source,
    functions=["fused_conv_relu_maxpool_cuda_op"],
    verbose=True, 
    extra_cuda_cflags=["-arch=sm_70", "--use_fast_math", "-std=c++17", "-Xptxas", "-v"]
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000): 
        super(ModelNew, self).__init__()
        
        self.in_channels_conv1 = 3
        self.out_channels_conv1 = 96
        self.kernel_size_conv1 = 11
        self.stride_conv1 = 4
        self.padding_conv1 = 2

        self.kernel_size_pool1 = 3
        self.stride_pool1 = 2

        temp_conv1 = nn.Conv2d(
            in_channels=self.in_channels_conv1, 
            out_channels=self.out_channels_conv1, 
            kernel_size=self.kernel_size_conv1, 
            stride=self.stride_conv1, 
            padding=self.padding_conv1,
            bias=True 
        )
        self.conv1_weight = nn.Parameter(temp_conv1.weight.detach().clone())
        if temp_conv1.bias is not None:
            self.conv1_bias = nn.Parameter(temp_conv1.bias.detach().clone())
        else:
            self.register_parameter('conv1_bias', None) 

        self.fused_op = custom_fused_ops.fused_conv_relu_maxpool_cuda_op

    def forward(self, x):
        N_batch = x.size(0)
        H_in = x.size(2)
        W_in = x.size(3)
        
        bias_tensor = self.conv1_bias if self.conv1_bias is not None else torch.Tensor()

        out = self.fused_op(
            x, self.conv1_weight, bias_tensor,
            N_batch, self.in_channels_conv1, H_in, W_in,
            self.out_channels_conv1, 
            self.kernel_size_conv1, self.kernel_size_conv1, 
            self.stride_conv1, self.stride_conv1,          
            self.padding_conv1, self.padding_conv1,        
            self.kernel_size_pool1, self.kernel_size_pool1, 
            self.stride_pool1, self.stride_pool1            
        )
        
        if out.numel() > 0 : 
             out.masked_fill_(out == -torch.finfo(out.dtype).max, 0.0)

        return out