import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Inline CUDA kernel: optimized convolution using read-only cache (__ldg)
# ---------------------------------------------------------------------
cuda_src = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

//////////////////////////////////////////////////////////////////
// Optimized CUDA kernel for NCHW 2-D convolution (stride, padding = user)
// Uses __ldg() to exploit the read-only (texture) cache
//////////////////////////////////////////////////////////////////
__global__ void conv2d_forward_kernel(
        const float* __restrict__ input,   // N, C_in, H_in, W_in
        const float* __restrict__ weight,  // C_out, C_in, K_h, K_w
        const float* __restrict__ bias,    // C_out
        float* __restrict__ output,        // N, C_out, H_out, W_out
        int N,
        int C_in,
        int H_in,
        int W_in,
        int C_out,
        int H_out,
        int W_out,
        int K_h,
        int K_w,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w) {

    // Each thread computes one (n, c_out, h_out, w_out) element
    int hw = blockIdx.x * blockDim.x + threadIdx.x;   // flatten h_out * w_out
    int c_out = blockIdx.y;                           // output channel
    int n      = blockIdx.z;                          // batch index

    if (hw >= H_out * W_out) return;

    int h_out = hw / W_out;
    int w_out = hw %  W_out;

    // Fetch bias through read-only cache if present
    float val = bias != nullptr ? __ldg(bias + c_out) : 0.0f;

    // Iterate over C_in and kernel window
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {

                int h_in = h_out * stride_h - pad_h + kh;
                int w_in = w_out * stride_w - pad_w + kw;

                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in)
                    continue;

                int in_idx  = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                int wt_idx  = (((c_out * C_in + c_in) * K_h + kh) * K_w + kw);

                // Exploit read-only cache
                float in_val = __ldg(input  + in_idx);
                float wt_val = __ldg(weight + wt_idx);

                val += in_val * wt_val;
            }
        }
    }

    int out_idx = ((n * C_out + c_out) * H_out + h_out) * W_out + w_out;
    output[out_idx] = val;
}

//////////////////////////////////////////////////////////////
// C++/CUDA launcher
//////////////////////////////////////////////////////////////
torch::Tensor conv2d_cuda_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int stride_h,
                                  int stride_w,
                                  int pad_h,
                                  int pad_w) {

    TORCH_CHECK(input.is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(),   "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype()  == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype()   == torch::kFloat32, "bias must be float32");

    // Ensure contiguous tensors
    input  = input.contiguous();
    weight = weight.contiguous();
    bias   = bias.contiguous();

    const int N      = input.size(0);
    const int C_in   = input.size(1);
    const int H_in   = input.size(2);
    const int W_in   = input.size(3);

    const int C_out  = weight.size(0);
    const int K_h    = weight.size(2);
    const int K_w    = weight.size(3);

    const int H_out  = (H_in + 2 * pad_h - K_h) / stride_h + 1;
    const int W_out  = (W_in + 2 * pad_w - K_w) / stride_w + 1;

    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    auto output  = torch::empty({N, C_out, H_out, W_out}, options);

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((H_out * W_out + threads - 1) / threads, C_out, N);

    conv2d_forward_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        K_h, K_w,
        stride_h, stride_w,
        pad_h, pad_w);

    return output;
}
"""

# Function declaration so load_inline knows what to export
cpp_src = r"""
torch::Tensor conv2d_cuda_forward(torch::Tensor input,
                                  torch::Tensor weight,
                                  torch::Tensor bias,
                                  int stride_h,
                                  int stride_w,
                                  int pad_h,
                                  int pad_w);
"""

# Build the extension (compiled only once and cached afterwards)
conv2d_ext = load_inline(
    name="custom_conv2d",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["conv2d_cuda_forward"],
    verbose=False,
)

# ---------------------------------------------------------------------
# Replacement model using the optimized custom CUDA kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        # Keep the same learnable parameters as the original Conv2d layer
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=11,
            stride=4,
            padding=2,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move data & parameters to CUDA if not already
        if not x.is_cuda:
            x = x.cuda(non_blocking=True)
        weight = self.conv1.weight.to(x.device, non_blocking=True)
        bias   = self.conv1.bias.to(x.device,   non_blocking=True)

        return conv2d_ext.conv2d_cuda_forward(
            x,
            weight,
            bias,
            4, 4,   # stride_h, stride_w
            2, 2    # pad_h, pad_w
        )