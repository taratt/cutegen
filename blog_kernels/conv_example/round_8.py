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

// Struct to hold precomputed k-dimension indices
struct KDecomposed {
    int kw;
    int kh;
    int ic;
    bool isValid; // True if current_k_idx < K_gemm
};

// Struct to hold precomputed N-dimension GEMM indices
struct NDecomposed {
    int ow_eff;
    int oh_eff;
    int n_batch_idx;
    bool isValidPixel; // True if this pixel_idx is within N_gemm bounds
    int h_in_base; // New: oh_eff * stride_h - pad_h
    int w_in_base; // New: ow_eff * stride_w - pad_w
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

    // Shared memory for tiles of A (weights) and B (input/im2col)
    __shared__ half Asub_hi[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Asub_lo[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_lo[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

    // Shared memory for precomputed k-indices
    __shared__ KDecomposed k_params[WMMA_K];
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

    // Loop over the K_gemm dimension in tiles of WMMA_K
    for (int k_tile_start = 0; k_tile_start < K_gemm; k_tile_start += WMMA_K) {
        __syncthreads(); 

        if (threadIdx.x < WMMA_K) {
            int c_tile_offset = threadIdx.x; 
            int current_k_idx = k_tile_start + c_tile_offset;

            if (current_k_idx < K_gemm) {
                k_params[c_tile_offset].kw = current_k_idx % K_w;
                int temp_div_kw = current_k_idx / K_w;
                k_params[c_tile_offset].kh = temp_div_kw % K_h;
                k_params[c_tile_offset].ic = temp_div_kw / K_h;
                k_params[c_tile_offset].isValid = true;
            } else {
                k_params[c_tile_offset].isValid = false;
                k_params[c_tile_offset].kw = 0;
                k_params[c_tile_offset].kh = 0;
                k_params[c_tile_offset].ic = 0;
            }
        }
        __syncthreads(); 

        // Load tile of A (weights) into shared memory
        for (int i = threadIdx.x; i < TILE_M_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_a_tile = i / WMMA_K; 
            int c_a_tile = i % WMMA_K; 
            int oc_idx = block_row_gemm_start + r_a_tile; 
            
            float weight_val = 0.0f;
            if (oc_idx < C_out && k_params[c_a_tile].isValid) {
                int kw_eff = k_params[c_a_tile].kw;
                int kh_eff = k_params[c_a_tile].kh;
                int ic_eff = k_params[c_a_tile].ic;
                weight_val = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff * K_h * K_w + kh_eff * K_w + kw_eff];
            }
            Asub_hi[r_a_tile][c_a_tile] = __float2half(weight_val);
            Asub_lo[r_a_tile][c_a_tile] = __float2half(weight_val - __half2float(Asub_hi[r_a_tile][c_a_tile]));
        }

        // Load tile of B (input/im2col) into shared memory
        for (int i = threadIdx.x; i < TILE_N_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_b_tile = i / WMMA_K; 
            int c_b_tile = i % WMMA_K; 

            float input_val = 0.0f;
            if (n_params_sh[r_b_tile].isValidPixel && k_params[c_b_tile].isValid) {
                int kw_eff = k_params[c_b_tile].kw;
                int kh_eff = k_params[c_b_tile].kh;
                int ic_eff = k_params[c_b_tile].ic;

                int n_batch_idx = n_params_sh[r_b_tile].n_batch_idx;
                int h_in_base = n_params_sh[r_b_tile].h_in_base;
                int w_in_base = n_params_sh[r_b_tile].w_in_base;

                int h_in_eff = h_in_base + kh_eff;
                int w_in_eff = w_in_base + kw_eff;

                if (h_in_eff >= 0 && h_in_eff < H_in &&
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
        __syncthreads(); 

        // Perform MMA operations
        int a_row_start_in_tile = warp_id * WMMA_M; 

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_hi, a_frag_lo;
        wmma::load_matrix_sync(a_frag_hi, &Asub_hi[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);
        wmma::load_matrix_sync(a_frag_lo, &Asub_lo[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);

        // Software pipelining for B fragments
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_hi_pipe[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_lo_pipe[2];

        // Initial load for the first B-tile (n_tile = 0)
        if (BLOCK_N_TILES_WMMA > 0) {
            int b_col_start_in_tile_current = 0 * WMMA_N; // For n_tile = 0
            wmma::load_matrix_sync(b_frag_hi_pipe[0], &Bsub_hi[b_col_start_in_tile_current][0], WMMA_K + SKEW_HALF);
            wmma::load_matrix_sync(b_frag_lo_pipe[0], &Bsub_lo[b_col_start_in_tile_current][0], WMMA_K + SKEW_HALF);
        }
        
        int current_pipe_idx = 0;

        #pragma unroll
        for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
            int next_pipe_idx = 1 - current_pipe_idx;

            // Prefetch B fragments for the *next* iteration (n_tile + 1)
            if (n_tile < BLOCK_N_TILES_WMMA - 1) {
                int b_col_start_in_tile_next = (n_tile + 1) * WMMA_N;
                wmma::load_matrix_sync(b_frag_hi_pipe[next_pipe_idx], &Bsub_hi[b_col_start_in_tile_next][0], WMMA_K + SKEW_HALF);
                wmma::load_matrix_sync(b_frag_lo_pipe[next_pipe_idx], &Bsub_lo[b_col_start_in_tile_next][0], WMMA_K + SKEW_HALF);
            }

            // Perform MMA operations using A fragments and B fragments from current_pipe_idx
            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_hi_pipe[current_pipe_idx], acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_lo_pipe[current_pipe_idx], acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_hi_pipe[current_pipe_idx], acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_lo_pipe[current_pipe_idx], acc_frag[n_tile]);
            
            current_pipe_idx = next_pipe_idx;
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
            int pixel_idx = block_col_gemm_start + (n_tile * WMMA_N) + c_frag;

            if (oc_idx < C_out && pixel_idx < N_gemm) {
                int ow_eff = pixel_idx % W_out;
                int oh_eff = (pixel_idx / W_out) % H_out;
                int n_batch_idx = pixel_idx / (H_out * W_out);

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