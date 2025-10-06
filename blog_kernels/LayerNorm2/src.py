import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------------
# Single-kernel, numerically-stable LayerNorm (true Welford + 3-pass) CUDA implementation
# ----------------------------------------------------------------------
fused_layernorm_cuda_source = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <climits>
#include <cmath>

namespace cg = cooperative_groups;

// -----------------------------------------------------------------------------
// Tunables
// -----------------------------------------------------------------------------
#ifndef CTA_PER_SAMPLE
#define CTA_PER_SAMPLE 4         // number of CTAs that work on the same sample
#endif

#ifndef THREADS_PER_CTA
#define THREADS_PER_CTA 256      // threads per CTA
#endif

#ifndef MAX_N
#define MAX_N 131072             // maximum batch size we can handle
#endif

// -----------------------------------------------------------------------------
// Small global buffers –  fp32 atomicAdd’s are fast on sm70+
// -----------------------------------------------------------------------------
__device__ float g_sum[MAX_N];        // running ∑x
__device__ float g_scratch[MAX_N];    // running M2 ( ∑(x-mean)^2 )
__device__ float g_mean[MAX_N];       // final mean
__device__ float g_inv_std[MAX_N];    // final inverse std

// -----------------------------------------------------------------------------
// Helper : combine two Welford tuples
// -----------------------------------------------------------------------------
__device__ inline void welford_combine(float      &mean_a,
                                       float      &M2_a,
                                       float      &count_a,
                                       const float mean_b,
                                       const float M2_b,
                                       const float count_b)
{
    if (count_b == 0.f) return;            // nothing to do
    float delta = mean_b - mean_a;
    float count_tot = count_a + count_b;

    mean_a  += delta * (count_b / count_tot);
    M2_a    += M2_b + delta * delta * (count_a * count_b / count_tot);
    count_a  = count_tot;
}

// -----------------------------------------------------------------------------
// Fused cooperative-groups kernel (single-pass Welford + normalize + affine)
//   Vectorised version : float4 traffic
// -----------------------------------------------------------------------------
__global__ __launch_bounds__(THREADS_PER_CTA, 4)
void fused_layernorm_kernel(const float* __restrict__ input,
                            const float* __restrict__ gamma,
                            const float* __restrict__ beta,
                            float* __restrict__ output,
                            int   D,
                            float eps)
{
    // Cooperative grid
    cg::grid_group grid = cg::this_grid();

    const int tid           = threadIdx.x;
    const int cta_global    = blockIdx.x;
    const int sample        = cta_global / CTA_PER_SAMPLE;   // which sample
    const int cta_in_sample = cta_global % CTA_PER_SAMPLE;   // CTA index inside sample
    const long long base    = static_cast<long long>(sample) * D;

    // -------------------------------------------------------------------------
    // Zero the per-sample accumulators once
    // -------------------------------------------------------------------------
    if (cta_in_sample == 0 && tid == 0) {
        g_sum    [sample] = 0.f;
        g_scratch[sample] = 0.f;
    }
    grid.sync(); // make sure zeros are visible to every CTA in the grid

    // -------------------------------------------------------------------------
    // Pointers casted to float4
    // -------------------------------------------------------------------------
    const float4* __restrict__ input4  = reinterpret_cast<const float4*>(input)  + (base >> 2);
    float4*       __restrict__ output4 = reinterpret_cast<float4*>(output)       + (base >> 2);
    const float4* __restrict__ gamma4  = reinterpret_cast<const float4*>(gamma);
    const float4* __restrict__ beta4   = reinterpret_cast<const float4*>(beta);

    const int D4      = D >> 2;          // # float4’s
    const int D_tail  = D & 3;           // remaining scalars (0–3)

    // -------------------------------------------------------------------------
    // Pass #1 : Per-thread Welford while reading its strided slice (float4)
    // -------------------------------------------------------------------------
    float thread_mean = 0.f;
    float thread_M2   = 0.f;
    float thread_cnt  = 0.f;

    for (int idx4 = tid + cta_in_sample * blockDim.x;
         idx4 < D4;
         idx4 += blockDim.x * CTA_PER_SAMPLE)
    {
        float4 v = input4[idx4];

        // Unroll the four scalars
        float x[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float val = x[k];
            thread_cnt += 1.f;
            float delta  = val - thread_mean;
            thread_mean += delta / thread_cnt;
            float delta2 = val - thread_mean;
            thread_M2   += delta * delta2;
        }
    }

    // Handle tail (if D not multiple of 4)
    for (int idx = (D4 << 2) + tid + cta_in_sample * blockDim.x;
         idx < D;
         idx += blockDim.x * CTA_PER_SAMPLE)
    {
        float val = input[base + idx];
        thread_cnt += 1.f;
        float delta  = val - thread_mean;
        thread_mean += delta / thread_cnt;
        float delta2 = val - thread_mean;
        thread_M2   += delta * delta2;
    }

    // -------------------------------------------------------------------------
    // CTA reduction – hierarchical Welford over the (cnt, mean, M2) tuples
    // -------------------------------------------------------------------------
    extern __shared__ float shmem[];                // size: 3 * blockDim.x floats
    float* s_mean  = shmem;                         // blockDim.x
    float* s_M2    = shmem + blockDim.x;            // blockDim.x
    float* s_count = shmem + 2 * blockDim.x;        // blockDim.x

    s_mean [tid] = thread_mean;
    s_M2   [tid] = thread_M2;
    s_count[tid] = thread_cnt;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float mean_b  = s_mean [tid + stride];
            float M2_b    = s_M2   [tid + stride];
            float cnt_b   = s_count[tid + stride];

            float mean_a  = s_mean [tid];
            float M2_a    = s_M2   [tid];
            float cnt_a   = s_count[tid];

            welford_combine(mean_a, M2_a, cnt_a, mean_b, M2_b, cnt_b);

            s_mean [tid] = mean_a;
            s_M2   [tid] = M2_a;
            s_count[tid] = cnt_a;
        }
        __syncthreads();
    }

    if (tid == 0) {
        // s_mean[0] now holds mean_cta, s_M2[0] holds M2_cta, s_count[0] == n_cta
        float cta_sum = s_mean[0] * s_count[0];
        atomicAdd(&g_sum    [sample], cta_sum);
        atomicAdd(&g_scratch[sample], s_M2[0]);
    }

    // Grid-wide barrier – make sure g_sum & g_scratch are fully accumulated
    grid.sync();

    // -------------------------------------------------------------------------
    // One CTA per sample computes final mean, variance & inv_std
    // -------------------------------------------------------------------------
    if (cta_in_sample == 0 && tid == 0) {
        float sum = g_sum    [sample];          // ∑x
        float M2  = g_scratch[sample];          // ∑(x-mean)^2
        float mean = sum / static_cast<float>(D);
        float var  = M2  / static_cast<float>(D);
        float inv_std = rsqrtf(var + eps);

        g_mean   [sample] = mean;
        g_inv_std[sample] = inv_std;
    }

    // Barrier – mean & inv_std ready for everyone
    grid.sync();

    const float mean_f     = g_mean   [sample];
    const float inv_std_f  = g_inv_std[sample];

    // -------------------------------------------------------------------------
    // Pass #2 : normalize + (optional) affine – vectorised
    // -------------------------------------------------------------------------
    for (int idx4 = tid + cta_in_sample * blockDim.x;
         idx4 < D4;
         idx4 += blockDim.x * CTA_PER_SAMPLE)
    {
        float4 v_in = input4[idx4];
        float4 v_out;

        float gamma_vals[4] = {1.f, 1.f, 1.f, 1.f};
        float beta_vals [4] = {0.f, 0.f, 0.f, 0.f};

        if (gamma4 != nullptr) {
            float4 g = gamma4[idx4];
            gamma_vals[0] = g.x; gamma_vals[1] = g.y; gamma_vals[2] = g.z; gamma_vals[3] = g.w;
        }
        if (beta4 != nullptr) {
            float4 b = beta4[idx4];
            beta_vals[0] = b.x; beta_vals[1] = b.y; beta_vals[2] = b.z; beta_vals[3] = b.w;
        }

        float in_vals[4] = {v_in.x, v_in.y, v_in.z, v_in.w};
        float out_vals[4];

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float y = (in_vals[k] - mean_f) * inv_std_f;
            y = y * gamma_vals[k] + beta_vals[k];
            out_vals[k] = y;
        }

        v_out.x = out_vals[0];
        v_out.y = out_vals[1];
        v_out.z = out_vals[2];
        v_out.w = out_vals[3];

        output4[idx4] = v_out;
    }

    // Tail elements (if any)
    for (int idx = (D4 << 2) + tid + cta_in_sample * blockDim.x;
         idx < D;
         idx += blockDim.x * CTA_PER_SAMPLE)
    {
        float x = input[base + idx];
        float y = (x - mean_f) * inv_std_f;
        if (gamma != nullptr) y *= gamma[idx];
        if (beta  != nullptr) y += beta[idx];
        output[base + idx] = y;
    }
}

// -----------------------------------------------------------------------------
// C++ interface
// -----------------------------------------------------------------------------
torch::Tensor layernorm_forward_cuda(torch::Tensor input,
                                     torch::Tensor gamma,
                                     torch::Tensor beta,
                                     double eps_double)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32,
                "only float32 is supported");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    const int64_t N64 = input.size(0);
    TORCH_CHECK(N64 <= MAX_N,
                "Batch dimension exceeds MAX_N. Re-compile with larger MAX_N.");
    const int N = static_cast<int>(N64);

    const int64_t D64 = input.numel() / N64;
    TORCH_CHECK(D64 <= INT_MAX, "Feature dimension too large");
    const int D = static_cast<int>(D64);

    const dim3 block(THREADS_PER_CTA);
    const dim3 grid(N * CTA_PER_SAMPLE);
    const size_t shmem_bytes = block.x * 3 * sizeof(float); // (mean, M2, count)

    auto output = torch::empty_like(input);

    // Prepare kernel arguments
    const float* input_ptr  = input.data_ptr<float>();
    const float* gamma_ptr  = gamma.defined() ? gamma.data_ptr<float>() : nullptr;
    const float* beta_ptr   = beta .defined() ? beta .data_ptr<float>()  : nullptr;
    float*       output_ptr = output.data_ptr<float>();
    int   D_int   = D;
    float eps     = static_cast<float>(eps_double);

    void* args[] = {
        (void*)&input_ptr,
        (void*)&gamma_ptr,
        (void*)&beta_ptr,
        (void*)&output_ptr,
        (void*)&D_int,
        (void*)&eps
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch cooperatively
    cudaLaunchCooperativeKernel(
        (void*)fused_layernorm_kernel,
        grid,
        block,
        args,
        shmem_bytes,
        stream);

    return output;
}"""

fused_layernorm_cpp_source = r"""
torch::Tensor layernorm_forward_cuda(torch::Tensor input,
                                     torch::Tensor gamma,
                                     torch::Tensor beta,
                                     double eps);
"""

_layernorm = load_inline(
    name="fused_layernorm",
    cpp_sources=fused_layernorm_cpp_source,
    cuda_sources=fused_layernorm_cuda_source,
    functions=["layernorm_forward_cuda"],
    extra_cuda_cflags=["-Xptxas", "-maxrregcount=64"],
    verbose=False
)

# ----------------------------------------------------------------------
# Optimized PyTorch module using the fused CUDA LayerNorm
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Layer Normalization powered by a single, numerically-stable CUDA kernel.
    Normalization is performed over all dimensions except the first (batch).
    """
    def __init__(self, normalized_shape, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(self.normalized_shape, dtype=torch.float32)
        )
        self.bias = nn.Parameter(
            torch.zeros(self.normalized_shape, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        N = x.size(0)

        # Flatten all non-batch dimensions
        x_flat      = x.contiguous().view(N, -1)
        weight_flat = self.weight.contiguous().view(-1)
        bias_flat   = self.bias.contiguous().view(-1)

        y_flat = _layernorm.layernorm_forward_cuda(
            x_flat, weight_flat, bias_flat, self.eps
        )
        return y_flat.view(orig_shape)