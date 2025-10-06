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

#include <cute/tensor.hpp>
#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/tools/util/include/cutlass/util/helper_cuda.hpp"
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"

// ============================================================================
//  GEMM kernel: double-buffered cp_async (2-stage ping-pong)
// ============================================================================

template <class TA, class TB, class TC, class Alpha, class Beta>
__global__ static void gemm_kernel(int M, int N, int K,
                                   Alpha alpha,
                                   TA const* __restrict__ A, int ldA,
                                   TB const* __restrict__ B, int ldB,
                                   Beta  beta,
                                   TC* __restrict__ C, int ldC) {
    using namespace cute;

    // ------------------------------------
    // CTA, thread, and shared-memory shapes
    // ------------------------------------
    using CtaTiler      = Shape<_128, _128, _8>;   // 128×128×8 GEMM tile
    using CThreadLayout = Layout<Shape<_16, _16>>; // 256 threads/CTA

    using ASmemLayout = Layout<Shape<_128, _8>>;
    using BSmemLayout = Layout<Shape<_128, _8>>;

    using AThreadLayout = Layout<Shape<_32, _8>>;
    using BThreadLayout = Layout<Shape<_32, _8>>;

    // ------------------
    // Global-memory views
    // ------------------
    auto dA = make_stride(ldA, Int<1>{}); // row-major (K,1)
    auto dB = make_stride(ldB, Int<1>{}); // row-major (K,1)
    auto dC = make_stride(ldC, Int<1>{}); // row-major (N,1)

    Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
    Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
    Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor tAgA = local_tile(gA, CtaTiler{}, cta_coord, Step<_1, X, _1>{});
    Tensor tBgB = local_tile(gB, CtaTiler{}, cta_coord, Step<X, _1, _1>{});
    Tensor tCgC = local_tile(gC, CtaTiler{}, cta_coord, Step<_1, _1, X>{});

    // ------------------
    // Shared memory (2×)
    // ------------------
    __shared__ TA smemA[2][cosize_v<ASmemLayout>];
    __shared__ TB smemB[2][cosize_v<BSmemLayout>];

    auto tA = AThreadLayout{};
    auto tB = BThreadLayout{};
    auto tC = CThreadLayout{};

    // Thread partitioning for loads
    Tensor tAgA_partitioned = local_partition(tAgA, tA, threadIdx.x);
    Tensor tBgB_partitioned = local_partition(tBgB, tB, threadIdx.x);
    
    // Thread partitioning for the accumulator
    Tensor tCgC_partitioned = local_partition(tCgC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC_partitioned);
    clear(tCrC);

    // ------------------
    // Stage 0 prefetch
    // ------------------
    {
        Tensor sA0 = make_tensor(make_smem_ptr(smemA[0]), ASmemLayout{});
        Tensor sB0 = make_tensor(make_smem_ptr(smemB[0]), BSmemLayout{});

        Tensor tAsA0 = local_partition(sA0, tA, threadIdx.x);
        Tensor tBsB0 = local_partition(sB0, tB, threadIdx.x);

        copy(tAgA_partitioned(_, _, Int<0>{}), tAsA0);
        copy(tBgB_partitioned(_, _, Int<0>{}), tBsB0);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
    }

    // ------------------
    // Main loop
    // ------------------
    
    int K_TILE_MAX = size<2>(tAgA_partitioned);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        int stage      = k_tile & 1;      // stage to COMPUTE from
        int next_stage = stage ^ 1;       // stage to LOAD into

        // ---- Asynchronously load next tile ----
        if (k_tile + 1 < K_TILE_MAX) {
            Tensor sA_next = make_tensor(make_smem_ptr(smemA[next_stage]), ASmemLayout{});
            Tensor sB_next = make_tensor(make_smem_ptr(smemB[next_stage]), BSmemLayout{});

            Tensor tAsA_next = local_partition(sA_next, tA, threadIdx.x);
            Tensor tBsB_next = local_partition(sB_next, tB, threadIdx.x);

            copy(tAgA_partitioned(_, _, k_tile + 1), tAsA_next);
            copy(tBgB_partitioned(_, _, k_tile + 1), tBsB_next);
            cp_async_fence();
        }

        // Wait until the tile we need is ready
        cp_async_wait<1>();
        __syncthreads();

        // ---- Tensor Core MMA on current stage ----
        Tensor sA_curr = make_tensor(make_smem_ptr(smemA[stage]), ASmemLayout{});
        Tensor sB_curr = make_tensor(make_smem_ptr(smemB[stage]), BSmemLayout{});

        Tensor tCsA = local_partition(sA_curr, tC, threadIdx.x, Step<_1, X>{});
        Tensor tCsB = local_partition(sB_curr, tC, threadIdx.x, Step<X, _1>{});

        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

    // ------------------
    // Epilogue: C = alpha*acc + beta*C
    // ------------------
    axpby(alpha, tCrC, beta, tCgC_partitioned);
}

// ============================================================================
//  Host launcher (unchanged except for including the new kernel)
// ============================================================================

template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_simplified(int m, int n, int k,
                          Alpha alpha,
                          TA const* A, int ldA,
                          TB const* B, int ldB,
                          Beta beta,
                          TC* C, int ldC,
                          cudaStream_t stream = 0) {
    using namespace cute;

    using CtaTiler      = Shape<_128, _128, _8>;
    using CThreadLayout = Layout<Shape<_16, _16>>;

    dim3 dimBlock(size(CThreadLayout{}));
    dim3 dimGrid(size(ceil_div(m, size<0>(CtaTiler{}))),
                 size(ceil_div(n, size<1>(CtaTiler{}))));

    gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(m, n, k,
                                                  alpha,
                                                  A, ldA,
                                                  B, ldB,
                                                  beta,
                                                  C, ldC);
}

// ============================================================================
//  PyTorch front-end
// ============================================================================

torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B) {
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

    const int ldA = A.stride(0);
    const int ldB = B.stride(0);
    TORCH_CHECK(ldA == k, "A is not contiguous (row-major), expected stride(0) == K");
    TORCH_CHECK(ldB == k, "B is not contiguous (row-major), expected stride(0) == K");

    auto C = torch::empty({m, n}, A.options());
    const int ldC = C.stride(0);

    using TA = float;
    using TB = float;
    using TC = float;
    using Alpha = float;
    using Beta = float;

    Alpha alpha = 1.0f;
    Beta  beta  = 0.0f;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    cute_gemm_simplified<TA, TB, TC, Alpha, Beta>(
        m, n, k,
        alpha,
        A.data_ptr<TA>(), ldA,
        B.data_ptr<TB>(), ldB,
        beta,
        C.data_ptr<TC>(), ldC,
        stream);

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}
"""

# Define the C++ function signature for the build system
cutlass_gemm_cpp_source = "torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the code using torch.utils.cpp_extension.load_inline
CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

# JIT compilation of our CUDA kernel
cutlass_gemm_lib = load_inline(
    name="cutlass_gemm_lib_optimized",
    cpp_sources=cutlass_gemm_cpp_source,
    cuda_sources=cutlass_gemm_source,
    functions=["cutlass_gemm_nt_cuda"],
    verbose=True,
    extra_cflags=['-std=c++17', '-O3'],
    extra_cuda_cflags=[f'-I{CUTLASS_PATH} -I{CUTLASS_PATH}/include -I{CUTLASS_PATH}/../', '-gencode arch=compute_90a,code=sm_90a'],
)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cutlass_gemm_lib = cutlass_gemm_lib

    def forward(self, A, B):
        """
        Performs C = A * B.T using the custom CUTLASS kernel with double buffering.
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
        # Instantiate the models
        model = ModelNew().cuda()
        model_old = Model().cuda()

        # Define input tensors
        M, N, K = 5120, 5120, 4096
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(N, K, device="cuda", dtype=torch.float32)

        # Run the forward pass to execute the CUTLASS kernel
        print("Running custom CUTLASS kernel with double buffering...")
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
        print(f"Average CUTLASS kernel time (double buffered): {cutlass_time_ms:.4f} ms")
        
        start_event.record()
        for _ in range(100):
            model_old.forward(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        pytorch_time_ms = start_event.elapsed_time(end_event) / 100
        print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")
        print(f"Speedup: {pytorch_time_ms/cutlass_time_ms:.2f}x")