import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv2d_implicit_gemm_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // For at::cuda::getCurrentCUDAStream()
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Skew padding for shared memory to avoid bank conflicts
#define SKEW_HALF 8 // 8 half elements (16 bytes)

// CUDA built-in warpSize is 32 for supported architectures (sm_70+)
// This constant is used for host-side configuration (e.g. blockDim)
#define CUDA_WARP_SIZE_CONST 32 

// Threadblock configuration
#define WARPS_PER_BLOCK 8
// THREADS_PER_BLOCK must be evaluatable by host compiler for blockDim configuration
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * CUDA_WARP_SIZE_CONST) 

// Macro-tile dimensions computed by a threadblock
// BLOCK_M_TILES_WMMA * WMMA_M = output channels processed by a block
// BLOCK_N_TILES_WMMA * WMMA_N = output spatial elements processed by a block
#define BLOCK_M_TILES_WMMA 8
#define BLOCK_N_TILES_WMMA 8

#define TILE_M_PER_BLOCK (BLOCK_M_TILES_WMMA * WMMA_M) // e.g., 8 * 16 = 128 (for C_out dimension)
#define TILE_N_PER_BLOCK (BLOCK_N_TILES_WMMA * WMMA_N) // e.g., 8 * 16 = 128 (for N_batch * H_out * W_out dimension)

// Vector size for shared memory writes (half2)
#define VECTOR_SIZE_H2 2

// Struct to hold precomputed N-dimension GEMM indices
struct NDecomposed {
    int ow_eff;
    int oh_eff;
    int n_batch_idx;
    bool isValidPixel; // True if this pixel_idx is within N_gemm bounds
    int h_in_base; 
    int w_in_base; 
};

__global__ void conv2d_implicit_gemm_wmma_kernel(
    const float* __restrict__ input_ptr,    // Input: (N, Cin, Hin, Win)
    const float* __restrict__ weight_ptr,   // Weights: (Cout, Cin, Kh, Kw)
    const float* __restrict__ bias_ptr,     // Bias: (Cout) or nullptr
    float* __restrict__ output_ptr,         // Output: (N, Cout, Hout, Wout)
    const int N_batch, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int H_out, const int W_out,
    const int M_gemm, // C_out
    const int N_gemm, // N_batch * H_out * W_out
    const int K_gemm  // C_in * K_h * K_w
) {
    // Thread identification
    const int warp_id = threadIdx.x / warpSize;        // 0 .. WARPS_PER_BLOCK-1
    const int lane_id = threadIdx.x % warpSize;        // 0 .. 31 (or warpSize-1)

    // Top-left corner of the macro-tile this block is responsible for in GEMM terms
    const int block_row_gemm_start = TILE_M_PER_BLOCK * blockIdx.y;
    const int block_col_gemm_start = TILE_N_PER_BLOCK * blockIdx.x;

    // Shared memory for tiles of A (weights) and B (input/im2col) - Double Buffered for K-loop pipelining
    __shared__ half Asub_pipe[2][TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_pipe[2][TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

    // Shared memory for precomputed N-indices
    __shared__ NDecomposed n_params_sh[TILE_N_PER_BLOCK];

    // Shared memory for output stage (per-warp buffers)
    __shared__ float C_shmem_output_buffers[WARPS_PER_BLOCK][WMMA_M][WMMA_N];

    // Accumulator fragments per warp.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[BLOCK_N_TILES_WMMA];
    #pragma unroll
    for (int i = 0; i < BLOCK_N_TILES_WMMA; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    // Populate n_params_sh once at the beginning of the kernel
    if (threadIdx.x < TILE_N_PER_BLOCK) {
        int r_b_tile_idx = threadIdx.x; 
        int current_pixel_idx = block_col_gemm_start + r_b_tile_idx;

        if (current_pixel_idx < N_gemm) {
            n_params_sh[r_b_tile_idx].ow_eff = current_pixel_idx % W_out;
            int temp_div_wout = current_pixel_idx / W_out;
            n_params_sh[r_b_tile_idx].oh_eff = temp_div_wout % H_out;
            n_params_sh[r_b_tile_idx].n_batch_idx = temp_div_wout / H_out;
            n_params_sh[r_b_tile_idx].isValidPixel = true;

            n_params_sh[r_b_tile_idx].h_in_base = n_params_sh[r_b_tile_idx].oh_eff * stride_h - pad_h;
            n_params_sh[r_b_tile_idx].w_in_base = n_params_sh[r_b_tile_idx].ow_eff * stride_w - pad_w;
        } else {
            n_params_sh[r_b_tile_idx].isValidPixel = false;
            n_params_sh[r_b_tile_idx].ow_eff = 0; 
            n_params_sh[r_b_tile_idx].oh_eff = 0;
            n_params_sh[r_b_tile_idx].n_batch_idx = 0;
            n_params_sh[r_b_tile_idx].h_in_base = 0; 
            n_params_sh[r_b_tile_idx].w_in_base = 0;
        }
    }
    __syncthreads();

    // Constants for vectorized shared memory loading
    // Number of half2 elements along K-dim for a shared memory tile row
    const int NUM_H2_ELEMENTS_IN_K_DIM = WMMA_K / VECTOR_SIZE_H2;
    // Number of thread groups, where each group has NUM_H2_ELEMENTS_IN_K_DIM threads.
    // Each group is responsible for loading the K-dimension for one M-row (for A) or N-row (for B) at a time,
    // iterating over M-rows or N-rows with this step size.
    const int NUM_ROW_PROCESSING_GROUPS = THREADS_PER_BLOCK / NUM_H2_ELEMENTS_IN_K_DIM;


    // --- K-Loop Pipelining ---
    int num_k_tiles = (K_gemm + WMMA_K - 1) / WMMA_K;
    
    // --- Prologue: Load first k-tile (k_tile_iter = 0) into pipe_idx = 0 ---
    if (num_k_tiles > 0) { 
        int k_tile_start_prologue = 0; 
        int current_pipe_idx_prologue = 0; 

        // Load Asub_pipe[0] for k_tile_iter = 0
        {
            // This thread is responsible for the 'h2_idx_in_k_dim_A'-th half2 element
            // in the K-dimension of the shared memory tile.
            int h2_idx_in_k_dim_A = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
            // Starting 'half' index in shared memory for this half2 write.
            int shmem_k_start_for_h2_A = h2_idx_in_k_dim_A * VECTOR_SIZE_H2;

            // Global k-indices for the two half elements.
            int k_global_A_0 = k_tile_start_prologue + shmem_k_start_for_h2_A;
            int k_global_A_1 = k_tile_start_prologue + shmem_k_start_for_h2_A + 1;

            // Decompose k_global_A_0
            int kw_eff_reg_A_0 = 0, kh_eff_reg_A_0 = 0, ic_eff_reg_A_0 = 0;
            bool is_valid_k_A_0 = (k_global_A_0 < K_gemm);
            if (is_valid_k_A_0) {
                kw_eff_reg_A_0 = k_global_A_0 % K_w;
                int temp_div_kw_A_0 = k_global_A_0 / K_w;
                kh_eff_reg_A_0 = temp_div_kw_A_0 % K_h;
                ic_eff_reg_A_0 = temp_div_kw_A_0 / K_h;
            }

            // Decompose k_global_A_1
            int kw_eff_reg_A_1 = 0, kh_eff_reg_A_1 = 0, ic_eff_reg_A_1 = 0;
            bool is_valid_k_A_1 = (k_global_A_1 < K_gemm);
            if (is_valid_k_A_1) {
                kw_eff_reg_A_1 = k_global_A_1 % K_w;
                int temp_div_kw_A_1 = k_global_A_1 / K_w;
                kh_eff_reg_A_1 = temp_div_kw_A_1 % K_h;
                ic_eff_reg_A_1 = temp_div_kw_A_1 / K_h;
            }
            
            // This thread belongs to 'm_row_group_id_A'-th group of threads.
            // This group iterates over M-rows of the Asub_pipe tile.
            int m_row_group_id_A = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
            for (int r_a_tile_base = m_row_group_id_A; r_a_tile_base < TILE_M_PER_BLOCK; r_a_tile_base += NUM_ROW_PROCESSING_GROUPS) {
                int oc_idx = block_row_gemm_start + r_a_tile_base;
                float weight_val_0 = 0.0f;
                if (oc_idx < C_out && is_valid_k_A_0) {
                    weight_val_0 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                              ic_eff_reg_A_0 * K_h * K_w +
                                              kh_eff_reg_A_0 * K_w +
                                              kw_eff_reg_A_0];
                }
                float weight_val_1 = 0.0f;
                if (oc_idx < C_out && is_valid_k_A_1) {
                    weight_val_1 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                              ic_eff_reg_A_1 * K_h * K_w +
                                              kh_eff_reg_A_1 * K_w +
                                              kw_eff_reg_A_1];
                }
                half2* smem_ptr_h2_A = reinterpret_cast<half2*>(
                    &Asub_pipe[current_pipe_idx_prologue][r_a_tile_base][shmem_k_start_for_h2_A]
                );
                *smem_ptr_h2_A = make_half2(__float2half(weight_val_0), __float2half(weight_val_1));
            }
        }

        // Load Bsub_pipe[0] for k_tile_iter = 0
        {
            int h2_idx_in_k_dim_B = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
            int shmem_k_start_for_h2_B = h2_idx_in_k_dim_B * VECTOR_SIZE_H2;

            int k_global_B_0 = k_tile_start_prologue + shmem_k_start_for_h2_B;
            int k_global_B_1 = k_tile_start_prologue + shmem_k_start_for_h2_B + 1;

            int kw_eff_reg_B_0 = 0, kh_eff_reg_B_0 = 0, ic_eff_reg_B_0 = 0;
            bool is_valid_k_B_0 = (k_global_B_0 < K_gemm);
            if (is_valid_k_B_0) {
                kw_eff_reg_B_0 = k_global_B_0 % K_w;
                int temp_div_kw_B_0 = k_global_B_0 / K_w;
                kh_eff_reg_B_0 = temp_div_kw_B_0 % K_h;
                ic_eff_reg_B_0 = temp_div_kw_B_0 / K_h;
            }

            int kw_eff_reg_B_1 = 0, kh_eff_reg_B_1 = 0, ic_eff_reg_B_1 = 0;
            bool is_valid_k_B_1 = (k_global_B_1 < K_gemm);
            if (is_valid_k_B_1) {
                kw_eff_reg_B_1 = k_global_B_1 % K_w;
                int temp_div_kw_B_1 = k_global_B_1 / K_w;
                kh_eff_reg_B_1 = temp_div_kw_B_1 % K_h;
                ic_eff_reg_B_1 = temp_div_kw_B_1 / K_h;
            }

            int n_row_group_id_B = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
            for (int r_b_tile_base = n_row_group_id_B; r_b_tile_base < TILE_N_PER_BLOCK; r_b_tile_base += NUM_ROW_PROCESSING_GROUPS) {
                float input_val_0 = 0.0f;
                if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_0) {
                    const NDecomposed& current_n_params = n_params_sh[r_b_tile_base];
                    int h_in_eff_0 = current_n_params.h_in_base + kh_eff_reg_B_0;
                    int w_in_eff_0 = current_n_params.w_in_base + kw_eff_reg_B_0;
                    if (h_in_eff_0 >= 0 && h_in_eff_0 < H_in && w_in_eff_0 >= 0 && w_in_eff_0 < W_in) {
                        input_val_0 = input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                                              ic_eff_reg_B_0 * H_in * W_in +
                                              h_in_eff_0 * W_in +
                                              w_in_eff_0];
                    }
                }
                float input_val_1 = 0.0f;
                 if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_1) {
                    const NDecomposed& current_n_params = n_params_sh[r_b_tile_base];
                    int h_in_eff_1 = current_n_params.h_in_base + kh_eff_reg_B_1;
                    int w_in_eff_1 = current_n_params.w_in_base + kw_eff_reg_B_1;
                     if (h_in_eff_1 >= 0 && h_in_eff_1 < H_in && w_in_eff_1 >= 0 && w_in_eff_1 < W_in) {
                        input_val_1 = input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                                              ic_eff_reg_B_1 * H_in * W_in +
                                              h_in_eff_1 * W_in +
                                              w_in_eff_1];
                    }
                }
                half2* smem_ptr_h2_B = reinterpret_cast<half2*>(
                    &Bsub_pipe[current_pipe_idx_prologue][r_b_tile_base][shmem_k_start_for_h2_B]
                );
                *smem_ptr_h2_B = make_half2(__float2half(input_val_0), __float2half(input_val_1));
            }
        }
    }


    // Loop over the K_gemm dimension in tiles of WMMA_K
    for (int k_tile_iter = 0; k_tile_iter < num_k_tiles; ++k_tile_iter) {
        __syncthreads(); // Sync point for pipelining

        int compute_pipe_idx = k_tile_iter % 2;
        int load_pipe_idx = (k_tile_iter + 1) % 2;

        // --- Load Stage for next k-tile (k_tile_iter + 1) into load_pipe_idx ---
        int k_tile_start_for_load = (k_tile_iter + 1) * WMMA_K;
        if (k_tile_start_for_load < K_gemm) { 
            // Load Asub_pipe[load_pipe_idx]
            { 
                int h2_idx_in_k_dim_A = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
                int shmem_k_start_for_h2_A = h2_idx_in_k_dim_A * VECTOR_SIZE_H2;

                int k_global_A_0 = k_tile_start_for_load + shmem_k_start_for_h2_A;
                int k_global_A_1 = k_tile_start_for_load + shmem_k_start_for_h2_A + 1;

                int kw_eff_reg_A_0 = 0, kh_eff_reg_A_0 = 0, ic_eff_reg_A_0 = 0;
                bool is_valid_k_A_0 = (k_global_A_0 < K_gemm);
                if (is_valid_k_A_0) {
                    kw_eff_reg_A_0 = k_global_A_0 % K_w;
                    int temp_div_kw_A_0 = k_global_A_0 / K_w;
                    kh_eff_reg_A_0 = temp_div_kw_A_0 % K_h;
                    ic_eff_reg_A_0 = temp_div_kw_A_0 / K_h;
                }

                int kw_eff_reg_A_1 = 0, kh_eff_reg_A_1 = 0, ic_eff_reg_A_1 = 0;
                bool is_valid_k_A_1 = (k_global_A_1 < K_gemm);
                if (is_valid_k_A_1) {
                    kw_eff_reg_A_1 = k_global_A_1 % K_w;
                    int temp_div_kw_A_1 = k_global_A_1 / K_w;
                    kh_eff_reg_A_1 = temp_div_kw_A_1 % K_h;
                    ic_eff_reg_A_1 = temp_div_kw_A_1 / K_h;
                }
                
                int m_row_group_id_A = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
                for (int r_a_tile_base = m_row_group_id_A; r_a_tile_base < TILE_M_PER_BLOCK; r_a_tile_base += NUM_ROW_PROCESSING_GROUPS) {
                    int oc_idx = block_row_gemm_start + r_a_tile_base;
                    float weight_val_0 = 0.0f;
                    if (oc_idx < C_out && is_valid_k_A_0) {
                        weight_val_0 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                                  ic_eff_reg_A_0 * K_h * K_w +
                                                  kh_eff_reg_A_0 * K_w +
                                                  kw_eff_reg_A_0];
                    }
                    float weight_val_1 = 0.0f;
                    if (oc_idx < C_out && is_valid_k_A_1) {
                        weight_val_1 = weight_ptr[oc_idx * C_in * K_h * K_w +
                                                  ic_eff_reg_A_1 * K_h * K_w +
                                                  kh_eff_reg_A_1 * K_w +
                                                  kw_eff_reg_A_1];
                    }
                    half2* smem_ptr_h2_A = reinterpret_cast<half2*>(
                        &Asub_pipe[load_pipe_idx][r_a_tile_base][shmem_k_start_for_h2_A]
                    );
                    *smem_ptr_h2_A = make_half2(__float2half(weight_val_0), __float2half(weight_val_1));
                }
            } 

            // Load Bsub_pipe[load_pipe_idx]
            { 
                int h2_idx_in_k_dim_B = threadIdx.x % NUM_H2_ELEMENTS_IN_K_DIM;
                int shmem_k_start_for_h2_B = h2_idx_in_k_dim_B * VECTOR_SIZE_H2;

                int k_global_B_0 = k_tile_start_for_load + shmem_k_start_for_h2_B;
                int k_global_B_1 = k_tile_start_for_load + shmem_k_start_for_h2_B + 1;

                int kw_eff_reg_B_0 = 0, kh_eff_reg_B_0 = 0, ic_eff_reg_B_0 = 0;
                bool is_valid_k_B_0 = (k_global_B_0 < K_gemm);
                if (is_valid_k_B_0) {
                    kw_eff_reg_B_0 = k_global_B_0 % K_w;
                    int temp_div_kw_B_0 = k_global_B_0 / K_w;
                    kh_eff_reg_B_0 = temp_div_kw_B_0 % K_h;
                    ic_eff_reg_B_0 = temp_div_kw_B_0 / K_h;
                }

                int kw_eff_reg_B_1 = 0, kh_eff_reg_B_1 = 0, ic_eff_reg_B_1 = 0;
                bool is_valid_k_B_1 = (k_global_B_1 < K_gemm);
                if (is_valid_k_B_1) {
                    kw_eff_reg_B_1 = k_global_B_1 % K_w;
                    int temp_div_kw_B_1 = k_global_B_1 / K_w;
                    kh_eff_reg_B_1 = temp_div_kw_B_1 % K_h;
                    ic_eff_reg_B_1 = temp_div_kw_B_1 / K_h;
                }

                int n_row_group_id_B = threadIdx.x / NUM_H2_ELEMENTS_IN_K_DIM;
                for (int r_b_tile_base = n_row_group_id_B; r_b_tile_base < TILE_N_PER_BLOCK; r_b_tile_base += NUM_ROW_PROCESSING_GROUPS) {
                    float input_val_0 = 0.0f;
                    if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_0) {
                        const NDecomposed& current_n_params = n_params_sh[r_b_tile_base];
                        int h_in_eff_0 = current_n_params.h_in_base + kh_eff_reg_B_0;
                        int w_in_eff_0 = current_n_params.w_in_base + kw_eff_reg_B_0;
                        if (h_in_eff_0 >= 0 && h_in_eff_0 < H_in && w_in_eff_0 >= 0 && w_in_eff_0 < W_in) {
                            input_val_0 = input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                                                  ic_eff_reg_B_0 * H_in * W_in +
                                                  h_in_eff_0 * W_in +
                                                  w_in_eff_0];
                        }
                    }
                    float input_val_1 = 0.0f;
                    if (n_params_sh[r_b_tile_base].isValidPixel && is_valid_k_B_1) {
                        const NDecomposed& current_n_params = n_params_sh[r_b_tile_base];
                        int h_in_eff_1 = current_n_params.h_in_base + kh_eff_reg_B_1;
                        int w_in_eff_1 = current_n_params.w_in_base + kw_eff_reg_B_1;
                        if (h_in_eff_1 >= 0 && h_in_eff_1 < H_in && w_in_eff_1 >= 0 && w_in_eff_1 < W_in) {
                            input_val_1 = input_ptr[current_n_params.n_batch_idx * C_in * H_in * W_in +
                                                  ic_eff_reg_B_1 * H_in * W_in +
                                                  h_in_eff_1 * W_in +
                                                  w_in_eff_1];
                        }
                    }
                    half2* smem_ptr_h2_B = reinterpret_cast<half2*>(
                        &Bsub_pipe[load_pipe_idx][r_b_tile_base][shmem_k_start_for_h2_B]
                    );
                    *smem_ptr_h2_B = make_half2(__float2half(input_val_0), __float2half(input_val_1));
                }
            } 
        }

        // --- Compute Stage for current k-tile (k_tile_iter) using compute_pipe_idx ---
        int a_row_start_in_tile = warp_id * WMMA_M; 

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, &Asub_pipe[compute_pipe_idx][a_row_start_in_tile][0], WMMA_K + SKEW_HALF);

        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_inner_pipe[2];

        if (BLOCK_N_TILES_WMMA > 0) {
            int b_col_start_in_tile_current = 0 * WMMA_N; 
            wmma::load_matrix_sync(b_frag_inner_pipe[0], &Bsub_pipe[compute_pipe_idx][b_col_start_in_tile_current][0], WMMA_K + SKEW_HALF);
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

    // Store results from accumulator fragments to global memory
    #pragma unroll
    for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
        wmma::store_matrix_sync(&C_shmem_output_buffers[warp_id][0][0], acc_frag[n_tile], WMMA_N, wmma::mem_row_major);

        for (int elem_idx_in_frag = lane_id; elem_idx_in_frag < WMMA_M * WMMA_N; elem_idx_in_frag += warpSize) {
            int r_frag = elem_idx_in_frag / WMMA_N;
            int c_frag = elem_idx_in_frag % WMMA_N;

            int oc_idx = block_row_gemm_start + (warp_id * WMMA_M) + r_frag;
            
            int offset_in_block_N_processing = (n_tile * WMMA_N) + c_frag;

            if (oc_idx < C_out && offset_in_block_N_processing < TILE_N_PER_BLOCK && 
                n_params_sh[offset_in_block_N_processing].isValidPixel) {
                const NDecomposed& current_n_params = n_params_sh[offset_in_block_N_processing];
                int ow_eff = current_n_params.ow_eff;
                int oh_eff = current_n_params.oh_eff;
                int n_batch_idx = current_n_params.n_batch_idx;

                float val = C_shmem_output_buffers[warp_id][r_frag][c_frag];

                if (bias_ptr != nullptr) {
                    val += bias_ptr[oc_idx];
                }

                output_ptr[n_batch_idx * C_out * H_out * W_out +
                           oc_idx * H_out * W_out +
                           oh_eff * W_out +
                           ow_eff] = val;
            }
        }
    }
}


torch::Tensor conv2d_implicit_gemm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out) {

    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.device().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias has wrong shape");
    }

    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(input.size(0) == N_batch, "Input N_batch mismatch");
    TORCH_CHECK(input.size(1) == C_in, "Input C_in mismatch");
    TORCH_CHECK(input.size(2) == H_in, "Input H_in mismatch");
    TORCH_CHECK(input.size(3) == W_in, "Input W_in mismatch");
    TORCH_CHECK(weight.size(0) == C_out, "Weight C_out mismatch");
    TORCH_CHECK(weight.size(1) == C_in, "Weight C_in mismatch");
    TORCH_CHECK(weight.size(2) == K_h, "Weight K_h mismatch");
    TORCH_CHECK(weight.size(3) == K_w, "Weight K_w mismatch");

    auto output = torch::zeros({N_batch, C_out, H_out, W_out}, input.options());

    const int M_gemm = C_out;
    const int N_gemm = N_batch * H_out * W_out;
    const int K_gemm = C_in * K_h * K_w;

    if (M_gemm == 0 || N_gemm == 0) { 
        return output;
    }
    if (K_gemm == 0) { 
         if (bias.defined()) { 
            output = output + bias.reshape({1, C_out, 1, 1});
        }
        return output; 
    }

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(
        (N_gemm + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK, 
        (M_gemm + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK  
    );

    const float* bias_ptr_data = bias.defined() ? bias.data_ptr<float>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_implicit_gemm_wmma_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr_data,
        output.data_ptr<float>(),
        N_batch, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        M_gemm, N_gemm, K_gemm
    );
    
    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}
"""

conv2d_implicit_gemm_cuda_declaration = r"""
torch::Tensor conv2d_implicit_gemm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out);
"""

# JIT compile the CUDA kernel
custom_conv2d_wmma_ops = load_inline(
    name="custom_conv2d_wmma_ops_optimized_k_pipe_vec_smem", # Changed name to avoid collision
    cpp_sources=conv2d_implicit_gemm_cuda_declaration,
    cuda_sources=conv2d_implicit_gemm_cuda_source,
    functions=["conv2d_implicit_gemm_cuda"],
    verbose=True, 
    extra_cuda_cflags=["-arch=sm_70", "--use_fast_math", "-std=c++17"] 
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000): # num_classes is part of original signature, kept for consistency
        super(ModelNew, self).__init__()
        
        # Define Conv1 parameters (matching the original model)
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size_val = 11 # Assuming square kernel
        self.stride_val = 4       # Assuming square stride
        self.padding_val = 2      # Assuming square padding

        # Create a temporary Conv2d layer to initialize weights and bias
        temp_conv = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size_val, 
            stride=self.stride_val, 
            padding=self.padding_val,
            bias=True # nn.Conv2d has bias=True by default
        )
        self.conv1_weight = nn.Parameter(temp_conv.weight.detach().clone())
        if temp_conv.bias is not None:
            self.conv1_bias = nn.Parameter(temp_conv.bias.detach().clone())
        else:
            # Correctly register 'conv1_bias' as None if not present
            self.register_parameter('conv1_bias', None) 


        self.custom_conv_op = custom_conv2d_wmma_ops.conv2d_implicit_gemm_cuda

    def forward(self, x):
        N_batch = x.size(0)
        # C_in_runtime = x.size(1) # Should match self.in_channels
        H_in = x.size(2)
        W_in = x.size(3)

        # Calculate output dimensions
        H_out = (H_in + 2 * self.padding_val - self.kernel_size_val) // self.stride_val + 1
        W_out = (W_in + 2 * self.padding_val - self.kernel_size_val) // self.stride_val + 1
        
        # Bias tensor handling: pass an undefined tensor if bias is None.
        # The C++ TORCH_CHECK(bias.defined()) handles this by providing nullptr to kernel.
        bias_tensor = self.conv1_bias if self.conv1_bias is not None else torch.Tensor()


        x = self.custom_conv_op(
            x, self.conv1_weight, bias_tensor,
            N_batch, self.in_channels, H_in, W_in,
            self.out_channels, self.kernel_size_val, self.kernel_size_val, # K_h, K_w
            self.stride_val, self.stride_val, # stride_h, stride_w
            self.padding_val, self.padding_val, # pad_h, pad_w
            H_out, W_out
        )
        return x