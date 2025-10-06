import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

cutlass_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The user must provide the path to the CUTLASS `include` directory during compilation.
#include <cute/tensor.hpp>
#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/tools/util/include/cutlass/util/helper_cuda.hpp"
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"

// A single-configuration CUDA kernel for a specific GEMM.
// All shapes, layouts, and tiling strategies are hardcoded inside for simplicity.
// This kernel computes C = A * B.T where A is (M, K) and B is (N, K).
template <class TA, class TB, class TC, class Alpha, class Beta>
__global__ static void gemm_kernel(int M, int N, int K, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
                                  Beta beta, TC* C, int ldC) {
    using namespace cute;

    // 1. DEFINE THE SHAPES AND LAYOUTS (The "One Shape, One Layout" part)
    // The CTA tile size, this is our primary "shape" knob
    using CtaTiler = Shape<_128, _128, _8>;

    // The layout for threads within a CTA, our primary "layout" knob
    using CThreadLayout = Layout<Shape<_16, _16>>; // 256 threads per block (16x16)

    // Shared memory layouts
    using ASmemLayout = Layout<Shape<_128, _8>>;    // (M, K)
    using BSmemLayout = Layout<Shape<_128, _8>>;    // (N, K)

    // Thread layouts for loading data
    using AThreadLayout = Layout<Shape<_32, _8>>;   // (M, K)
    using BThreadLayout = Layout<Shape<_32, _8>>;   // (N, K)

    // Define strides for global memory tensors.
    // NOTE: This is the key change from the original example. PyTorch uses row-major
    // tensors, so the stride for the 'row' dimension (M, N, or M) is the number of
    // columns, and the stride for the 'column' dimension is 1.
    // Original (for column-major): auto dA = make_stride(Int<1>{}, ldA);
    // New (for row-major):
    auto dA = make_stride(ldA, Int<1>{}); // Stride for A(M,K) is (K, 1)
    auto dB = make_stride(ldB, Int<1>{}); // Stride for B(N,K) is (K, 1)
    auto dC = make_stride(ldC, Int<1>{}); // Stride for C(M,N) is (N, 1)

    // 2. KERNEL LOGIC (mostly unchanged from the original)
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, CtaTiler{}, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, CtaTiler{}, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, CtaTiler{}, cta_coord, Step<_1, _1, X>{});

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), ASmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), BSmemLayout{});

    auto tA = AThreadLayout{};
    auto tB = BThreadLayout{};
    auto tC = CThreadLayout{};

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }
    axpby(alpha, tCrC, beta, tCgC);
}

// A single, self-contained host function to launch the GEMM.
// This function is hardcoded for the 'NT' case (C = A * B.T).
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_simplified(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    // Define the CTA tile size (must match the kernel's definition)
    using CtaTiler = Shape<_128, _128, _8>;
    // Define the thread layout (must match the kernel's definition)
    using CThreadLayout = Layout<Shape<_16, _16>>;

    // Determine grid and block dimensions from our fixed shapes
    dim3 dimBlock(size(CThreadLayout{}));
    dim3 dimGrid(size(ceil_div(m, size<0>(CtaTiler{}))),
                 size(ceil_div(n, size<1>(CtaTiler{}))));

    // Launch the single, simplified kernel
    gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}


// PyTorch C++ wrapper function
// This function takes PyTorch tensors, performs checks, and calls the host launcher.
torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B) {
    // This kernel is hardcoded for C = A * B.T, with A(M,K) and B(N,K)
    // It's also hardcoded for float type.
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Input tensor B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Input tensor A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "Input tensor B must be float32");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimension K (A.size(1)) must match B.size(1)");

    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(0);

    // For a contiguous row-major tensor (M, K), the leading dimension stride is K.
    const int ldA = A.stride(0);
    const int ldB = B.stride(0);
    TORCH_CHECK(ldA == k, "A is not contiguous (row-major), expected stride(0) == K");
    TORCH_CHECK(ldB == k, "B is not contiguous (row-major), expected stride(0) == K");

    auto C = torch::empty({m, n}, A.options());
    const int ldC = C.stride(0); // For a new contiguous (M, N) tensor, this is N.

    using TA = float;
    using TB = float;
    using TC = float;
    using Alpha = float;
    using Beta = float;

    Alpha alpha = 1.0f;
    Beta beta = 0.0f;

    // Get the current CUDA stream from PyTorch
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Call the host launcher function
    cute_gemm_simplified<TA, TB, TC, Alpha, Beta>(
        m, n, k,
        alpha,
        A.data_ptr<TA>(), ldA,
        B.data_ptr<TB>(), ldB,
        beta,
        C.data_ptr<TC>(), ldC,
        stream
    );

    // Check for any CUDA errors after kernel launch
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}
"""

# Define the C++ function signature for the build system
# This is a forward declaration of the function we want to expose to Python.
cutlass_gemm_cpp_source = "torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the code using torch.utils.cpp_extension.load_inline
CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

# JIT (Just-In-Time) compilation of our CUDA kernel
# This will compile the source code and load it as a Python module.
cutlass_gemm_lib = load_inline(
    name="cutlass_gemm_lib",
    cpp_sources=cutlass_gemm_cpp_source,
    cuda_sources=cutlass_gemm_source,
    functions=["cutlass_gemm_nt_cuda"],
    verbose=True,
    # CUTLASS requires C++17 and the path to its headers.
    extra_cflags=['-std=c++17', '-O3'],
    extra_cuda_cflags=[f'-I{CUTLASS_PATH} -I{CUTLASS_PATH}/include -I{CUTLASS_PATH}/../', '-gencode arch=compute_90a,code=sm_90a'],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # Store the loaded library containing our CUDA function
        self.cutlass_gemm_lib = cutlass_gemm_lib

    def forward(self, A, B):
        """
        Performs C = A * B.T using the custom CUTLASS kernel.
        Args:
            A (torch.Tensor): A 2D tensor of shape (M, K).
            B (torch.Tensor): A 2D tensor of shape (N, K).
        Returns:
            torch.Tensor: The result tensor C of shape (M, N).
        """
        return self.cutlass_gemm_lib.cutlass_gemm_nt_cuda(A, B)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B.T)

# --- Example Usage ---
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping example.")
    else:
        # Instantiate the model
        model = ModelNew().cuda()
        model_old = Model().cuda()

        # Define input tensors. The kernel is hardcoded for a specific problem size range.
        M, N, K = 5120, 5120, 4096
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(N, K, device="cuda", dtype=torch.float32)

        # Run the forward pass to execute the CUTLASS kernel
        print("Running custom CUTLASS kernel...")
        C_cutlass = model.forward(A, B)
        print("CUTLASS kernel finished.")

        # For verification, compute the same operation using PyTorch's native matmul
        print("Running PyTorch native matmul for verification...")
        C_pytorch = model_old.forward(A, B)
        print("PyTorch matmul finished.")

        # Check for correctness
        is_close = torch.allclose(C_cutlass, C_pytorch, atol=1e-3, rtol=1e-4)
        print(f"\nVerification check: {'SUCCESS' if is_close else 'FAILURE'}")
        print("Output tensor shape:", C_cutlass.shape)
        
        # Simple performance comparison
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warmup
        for _ in range(10):
            model.forward(A, B)
        torch.cuda.synchronize()
        
        start_event.record()
        for _ in range(100):
            model.forward(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        cutlass_time_ms = start_event.elapsed_time(end_event) / 100
        print(f"Average CUTLASS kernel time: {cutlass_time_ms:.4f} ms")
        
        start_event.record()
        for _ in range(100):
            model_old.forward(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        pytorch_time_ms = start_event.elapsed_time(end_event) / 100
        print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")