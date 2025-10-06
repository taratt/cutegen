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
    // These use the CUDA built-in 'warpSize' variable, available in device code.
    const int warp_id = threadIdx.x / warpSize;        // 0 .. WARPS_PER_BLOCK-1
    const int lane_id = threadIdx.x % warpSize;        // 0 .. 31 (or warpSize-1)

    // Top-left corner of the macro-tile this block is responsible for in GEMM terms
    const int block_row_gemm_start = TILE_M_PER_BLOCK * blockIdx.y;
    const int block_col_gemm_start = TILE_N_PER_BLOCK * blockIdx.x;

    // Shared memory for tiles of A (weights) and B (input/im2col)
    __shared__ half Asub_hi[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Asub_lo[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_lo[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

    // Accumulator fragments per warp. Each warp handles a TILE_M_PER_BLOCK / WARPS_PER_BLOCK row-stripe of A,
    // and all BLOCK_N_TILES_WMMA column-tiles of B.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[BLOCK_N_TILES_WMMA];
    #pragma unroll
    for (int i = 0; i < BLOCK_N_TILES_WMMA; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    // Loop over the K_gemm dimension in tiles of WMMA_K
    for (int k_tile_start = 0; k_tile_start < K_gemm; k_tile_start += WMMA_K) {
        __syncthreads(); // Ensure previous MMA ops are done before loading new data

        // Load tile of A (weights) into shared memory
        for (int i = threadIdx.x; i < TILE_M_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_a_tile = i / WMMA_K; 
            int c_a_tile = i % WMMA_K; 

            int oc_idx = block_row_gemm_start + r_a_tile; 
            int k_idx = k_tile_start + c_a_tile;      

            float weight_val = 0.0f;
            if (oc_idx < C_out && k_idx < K_gemm) {
                int kw_eff = k_idx % K_w;
                int kh_eff = (k_idx / K_w) % K_h;
                int ic_eff = k_idx / (K_h * K_w);
                weight_val = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff * K_h * K_w + kh_eff * K_w + kw_eff];
            }
            Asub_hi[r_a_tile][c_a_tile] = __float2half(weight_val);
            Asub_lo[r_a_tile][c_a_tile] = __float2half(weight_val - __half2float(Asub_hi[r_a_tile][c_a_tile]));
        }

        // Load tile of B (input/im2col) into shared memory
        for (int i = threadIdx.x; i < TILE_N_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_b_tile = i / WMMA_K; 
            int c_b_tile = i % WMMA_K; 

            int pixel_idx_offset_in_block = r_b_tile;
            int k_idx_offset_in_ktile = c_b_tile;

            int k_idx = k_tile_start + k_idx_offset_in_ktile; 
            int pixel_idx = block_col_gemm_start + pixel_idx_offset_in_block; 

            float input_val = 0.0f;
            if (k_idx < K_gemm && pixel_idx < N_gemm) {
                int kw_eff = k_idx % K_w;
                int kh_eff = (k_idx / K_w) % K_h;
                int ic_eff = k_idx / (K_h * K_w);

                int ow_eff = pixel_idx % W_out;
                int oh_eff = (pixel_idx / W_out) % H_out;
                int n_batch_idx = pixel_idx / (H_out * W_out);

                int h_in_eff = oh_eff * stride_h + kh_eff - pad_h;
                int w_in_eff = ow_eff * stride_w + kw_eff - pad_w;

                if (n_batch_idx < N_batch && ic_eff < C_in &&
                    h_in_eff >= 0 && h_in_eff < H_in &&
                    w_in_eff >= 0 && w_in_eff < W_in) {
                    input_val = input_ptr[n_batch_idx * C_in * H_in * W_in +
                                          ic_eff * H_in * W_in +
                                          h_in_eff * W_in +
                                          w_in_eff];
                }
            }
            Bsub_hi[r_b_tile][c_b_tile] = __float2half(input_val);
            Bsub_lo[r_b_tile][c_b_tile] = __float2half(input_val - __half2float(Bsub_hi[r_b_tile][c_b_tile]));
        }
        __syncthreads(); // Ensure shared memory is populated

        // Perform MMA operations
        int a_row_start_in_tile = warp_id * WMMA_M; 

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_hi, a_frag_lo;
        wmma::load_matrix_sync(a_frag_hi, &Asub_hi[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);
        wmma::load_matrix_sync(a_frag_lo, &Asub_lo[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);

        #pragma unroll
        for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
            int b_col_start_in_tile = n_tile * WMMA_N; 

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_hi, b_frag_lo;
            wmma::load_matrix_sync(b_frag_hi, &Bsub_hi[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);
            wmma::load_matrix_sync(b_frag_lo, &Bsub_lo[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);

            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_hi, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_lo, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_hi, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_lo, acc_frag[n_tile]);
        }
    }
    __syncthreads(); // Ensure all MMA operations for the block are complete

    // Store results from accumulator fragments to global memory
    // Define a shared memory tile for one fragment to allow reordering before global write.
    __shared__ float C_shmem_frag_tile[WMMA_M][WMMA_N];

    #pragma unroll
    for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
        // Iterate over warps in the block to serialize access to C_shmem_frag_tile
        // This prevents race conditions when multiple warps try to use the same C_shmem_frag_tile.
        for (int w_iter = 0; w_iter < WARPS_PER_BLOCK; ++w_iter) {
            if (warp_id == w_iter) { // Only threads of the current active warp (w_iter) execute this
                // Active warp (w_iter) stores its fragment acc_frag[n_tile] to C_shmem_frag_tile
                // wmma::store_matrix_sync is a collective operation for threads within a warp.
                wmma::store_matrix_sync(&C_shmem_frag_tile[0][0], acc_frag[n_tile], WMMA_N, wmma::mem_row_major);
                // C_shmem_frag_tile is now filled with data from acc_frag[n_tile] of warp w_iter.
                // No __syncthreads() is needed here because only one warp (w_iter) is writing to C_shmem_frag_tile,
                // and its threads will read from it next. Other warps are in the 'else' branch or waiting at __syncthreads() below.

                // Threads of the active warp (w_iter) read from C_shmem_frag_tile and write to global memory.
                // Each thread in the warp handles a subset of elements from the fragment.
                // Loop uses 'lane_id' (threadIdx.x % warpSize) and 'warpSize' (CUDA built-in).
                for (int elem_idx_in_frag = lane_id; elem_idx_in_frag < WMMA_M * WMMA_N; elem_idx_in_frag += warpSize) {
                    int r_frag = elem_idx_in_frag / WMMA_N; // Row within the WMMA_M x WMMA_N fragment
                    int c_frag = elem_idx_in_frag % WMMA_N; // Col within the WMMA_M x WMMA_N fragment

                    // Determine global output coordinates for the element (r_frag, c_frag) in the current fragment
                    int c_frag_row_start_for_this_warp = w_iter * WMMA_M; // Row start in block's M-tile for this warp
                    int c_frag_col_start_for_this_n_tile = n_tile * WMMA_N;  // Col start in block's N-tile for this fragment column

                    int oc_idx = block_row_gemm_start + c_frag_row_start_for_this_warp + r_frag;
                    int pixel_idx = block_col_gemm_start + c_frag_col_start_for_this_n_tile + c_frag;

                    if (oc_idx < C_out && pixel_idx < N_gemm) {
                        // Decompose pixel_idx into (n_batch_idx, oh, ow)
                        int ow_eff = pixel_idx % W_out;
                        int oh_eff = (pixel_idx / W_out) % H_out;
                        int n_batch_idx = pixel_idx / (H_out * W_out);

                        float val = C_shmem_frag_tile[r_frag][c_frag];
                        if (bias_ptr != nullptr) {
                            // Bias is added here, once per output element after all K-tiles are accumulated.
                            val += bias_ptr[oc_idx];
                        }

                        output_ptr[n_batch_idx * C_out * H_out * W_out +
                                   oc_idx * H_out * W_out +
                                   oh_eff * W_out +
                                   ow_eff] = val;
                    }
                }
            }
            __syncthreads(); // Crucial synchronization:
                             // 1. Ensures that warp w_iter completes its use of C_shmem_frag_tile
                             //    (both wmma::store_matrix_sync and subsequent global writes from C_shmem_frag_tile)
                             //    before the next warp (w_iter+1) starts using C_shmem_frag_tile for the SAME n_tile.
                             // 2. Ensures all global memory writes from warp w_iter for the current n_tile are complete
                             //    and visible to other threads/warps (though not strictly needed for correctness here, good practice).
        } // End loop over w_iter (serialized warps for C_shmem_frag_tile access within an n_tile iteration)
        // After this point (end of w_iter loop, all __syncthreads() passed), all warps have processed
        // their respective acc_frag[n_tile] and written results to global memory.
        // C_shmem_frag_tile has been used by each warp in turn for the current n_tile.
    } // End loop over n_tile
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

    // Handle cases where some dimensions are zero, which would lead to no computation.
    // If M_gemm or N_gemm is zero, the output tensor area is zero.
    // If K_gemm is zero (e.g. Cin=0), the result of matmul is zero, then bias is added.
    if (M_gemm == 0 || N_gemm == 0) {
        // If output spatial dimensions or Cout is 0, output is already zeros.
        // Bias might still be applicable if C_out > 0 but H_out/W_out is 0 (M_gemm > 0, N_gemm = 0).
        // However, if N_gemm = 0, output tensor has 0 elements in spatial/batch dims, so adding bias is moot.
        // If M_gemm = 0 (C_out = 0), output tensor has 0 channels, bias is also 0-sized.
        // So, if M_gemm or N_gemm is 0, just return the zero-initialized output.
        return output;
    }
    if (K_gemm == 0) { // No actual multiplication to do, output is effectively 0 before bias.
         if (bias.defined()) { // bias is (C_out)
            // Add bias to the zero tensor. Unsqueeze bias to (1,C_out,1,1) for broadcasting.
            output = output + bias.reshape({1, C_out, 1, 1});
        }
        return output;
    }

    // THREADS_PER_BLOCK is defined using CUDA_WARP_SIZE_CONST, so it's known by host compiler.
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(
        (N_gemm + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK, // Blocks for N_gemm dimension
        (M_gemm + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK  // Blocks for M_gemm dimension
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
    name="custom_conv2d_wmma_ops",
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