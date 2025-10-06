import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ------------------------------------------------------------------
# CUDA implementation of a vectorized (float4) row-wise softmax
# ------------------------------------------------------------------
cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>

__global__ void softmax_forward_kernel_vec4(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            const int num_features) {
    extern __shared__ float sdata[];

    const int  row    = blockIdx.x;      // Each block handles one row
    const int  tid    = threadIdx.x;
    const int  stride = blockDim.x;

    const float* row_in  = input  + row * num_features;
    float*       row_out = output + row * num_features;

    // Cast to float4 pointers for vectorized access
    const float4* in4  = reinterpret_cast<const float4*>(row_in);
    float4*       out4 = reinterpret_cast<float4*>(row_out);

    const int num_vec = num_features >> 2;      // num_features / 4
    const int rem     = num_features & 3;       // leftover elements (0-3)

    // ------------------------------------------------------------------
    // 1. Find maximum value in the row (for numerical stability)
    // ------------------------------------------------------------------
    float thread_max = -FLT_MAX;
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 v = in4[vec_idx];
        thread_max = fmaxf(thread_max, v.x);
        thread_max = fmaxf(thread_max, v.y);
        thread_max = fmaxf(thread_max, v.z);
        thread_max = fmaxf(thread_max, v.w);
    }

    // Handle leftovers (if any)
    if (tid == 0 && rem) {
        int base = num_vec << 2;        // num_vec * 4
        for (int i = 0; i < rem; ++i) {
            thread_max = fmaxf(thread_max, row_in[base + i]);
        }
    }

    // Reduce within block to get the row maximum
    sdata[tid] = thread_max;
    __syncthreads();
    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = sdata[tid + offset];
            if (other > sdata[tid])
                sdata[tid] = other;
        }
        __syncthreads();
    }
    float row_max = sdata[0];

    // ------------------------------------------------------------------
    // 2. Compute exponentials and their sum
    // ------------------------------------------------------------------
    float thread_sum = 0.0f;
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 v  = in4[vec_idx];
        float4 ev = {__expf(v.x - row_max),
                     __expf(v.y - row_max),
                     __expf(v.z - row_max),
                     __expf(v.w - row_max)};

        out4[vec_idx] = ev;
        thread_sum += ev.x + ev.y + ev.z + ev.w;
    }

    // Leftovers
    if (tid == 0 && rem) {
        int base = num_vec << 2;
        for (int i = 0; i < rem; ++i) {
            float val = __expf(row_in[base + i] - row_max);
            row_out[base + i] = val;
            thread_sum += val;
        }
    }

    // Block-wide reduction to obtain the sum of exponentials
    sdata[tid] = thread_sum;
    __syncthreads();
    for (int offset = stride >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    float sum_exp = sdata[0] + 1e-8f;      // Avoid divide-by-zero

    // ------------------------------------------------------------------
    // 3. Normalize to obtain softmax output
    // ------------------------------------------------------------------
    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += stride) {
        float4 ev = out4[vec_idx];
        ev.x /= sum_exp;
        ev.y /= sum_exp;
        ev.z /= sum_exp;
        ev.w /= sum_exp;
        out4[vec_idx] = ev;
    }

    if (tid == 0 && rem) {
        int base = num_vec << 2;
        for (int i = 0; i < rem; ++i) {
            row_out[base + i] /= sum_exp;
        }
    }
}

torch::Tensor softmax_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "only float tensors are supported");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D (batch_size, num_features)");

    const int batch_size   = input.size(0);
    const int num_features = input.size(1);

    // Allocate output
    auto output = torch::empty_like(input);

    // One block per row, 1024 threads per block
    const int threads        = 1024;
    const int shared_mem_sz  = threads * sizeof(float);

    softmax_forward_kernel_vec4<<<batch_size, threads, shared_mem_sz>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        num_features);

    return output;
}
"""

cpp_source = "torch::Tensor softmax_forward_cuda(torch::Tensor input);"

# Compile and load the CUDA extension
softmax_ops = load_inline(
    name="row_softmax_cuda",
    cpp_sources=[cpp_source],
    cuda_sources=[cuda_source],
    functions=["softmax_forward_cuda"],
    verbose=False,
)

# ------------------------------------------------------------------
# PyTorch module that uses the custom softmax operator
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.softmax_fn = softmax_ops

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax_fn.softmax_forward_cuda(x)