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

using namespace cute;

// ----------------------------------------------------------------------------
//  Kernel - double-buffered cp_async
// ----------------------------------------------------------------------------
template <class TA, class TB, class TC, class Alpha, class Beta>
__global__ static void gemm_kernel(int M, int N, int K,
                                   Alpha alpha,
                                   TA const* __restrict__ A, int ldA,
                                   TB const* __restrict__ B, int ldB,
                                   Beta  beta,
                                   TC*          C, int ldC)
{
    // 1.  Shapes / Layouts (unchanged)
    using CtaTiler      = Shape<_128, _128, _8>;          // CTA tile  (M,N,Ktile)
    using CThreadLayout = Layout<Shape<_16, _16>>;        // 256 threads (16x16)

    using ASmemLayout   = Layout<Shape<_128, _8>>;        // (M,Ktile)
    using BSmemLayout   = Layout<Shape<_128, _8>>;        // (N,Ktile)

    using AThreadLayout = Layout<Shape<_32, _8>>;         // (M,Ktile)
    using BThreadLayout = Layout<Shape<_32, _8>>;         // (N,Ktile)

    // ------------------------------------------------------------------------
    //  Global memory tensors (row-major => stride(col)=1, stride(row)=ld)
    // ------------------------------------------------------------------------
    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(ldC, Int<1>{});

    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M,N)

    // Tile tensors for this CTA
    auto  cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, CtaTiler{}, cta_coord, Step<_1,  X , _1>{});
    Tensor gB = local_tile(mB, CtaTiler{}, cta_coord, Step< X , _1, _1>{});
    Tensor gC = local_tile(mC, CtaTiler{}, cta_coord, Step<_1, _1,  X >{});

    // ------------------------------------------------------------------------
    //  Shared memory - double buffer (ping-pong)
    // ------------------------------------------------------------------------
    __shared__ TA smemA[2][cosize_v<ASmemLayout>];
    __shared__ TB smemB[2][cosize_v<BSmemLayout>];

    Tensor sA0 = make_tensor(make_smem_ptr(smemA[0]), ASmemLayout{});
    Tensor sA1 = make_tensor(make_smem_ptr(smemA[1]), ASmemLayout{});
    Tensor sB0 = make_tensor(make_smem_ptr(smemB[0]), BSmemLayout{});
    Tensor sB1 = make_tensor(make_smem_ptr(smemB[1]), BSmemLayout{});

    // ------------------------------------------------------------------------
    //  Thread partitions for GMEM->SMEM copies
    // ------------------------------------------------------------------------
    auto tA = AThreadLayout{};
    auto tB = BThreadLayout{};
    auto tC = CThreadLayout{};

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);

    Tensor tAsA0 = local_partition(sA0, tA, threadIdx.x);
    Tensor tAsA1 = local_partition(sA1, tA, threadIdx.x);
    Tensor tBsB0 = local_partition(sB0, tB, threadIdx.x);
    Tensor tBsB1 = local_partition(sB1, tB, threadIdx.x);

    // Thread partitions for SMEM->register GEMM
    Tensor tCsA0 = local_partition(sA0, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsA1 = local_partition(sA1, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB0 = local_partition(sB0, tC, threadIdx.x, Step< X, _1>{});
    Tensor tCsB1 = local_partition(sB1, tC, threadIdx.x, Step< X, _1>{});

    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    // ------------------------------------------------------------------------
    //  Double-buffered mainloop
    // ------------------------------------------------------------------------
    int  K_TILE_MAX = size<2>(tAgA);         // number of K-tiles inside CTA
    int  next_tile  = 0;

    // Prefetch tile 0 into buffer 0
    copy(tAgA(_, _, next_tile), tAsA0);
    copy(tBgB(_, _, next_tile), tBsB0);
    cp_async_fence();                        // commit cp_async
    // (Do NOT wait yet - overlap with later compute)

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // --------------------------------------------------------------------
        //  Wait until the data for CURRENT tile is in shared memory
        // --------------------------------------------------------------------
        cp_async_wait<0>();
        __syncthreads();

        // Choose current buffers
        bool use_buffer0 = (k_tile & 1) == 0;

        // --------------------------------------------------------------------
        //  GEMM on CURRENT tile (compute in registers)
        // --------------------------------------------------------------------
        if (use_buffer0)
            gemm(tCsA0, tCsB0, tCrC);
        else
            gemm(tCsA1, tCsB1, tCrC);

        // --------------------------------------------------------------------
        //  Prefetch NEXT tile into the OTHER buffer (if any)
        // --------------------------------------------------------------------
        next_tile = k_tile + 1;
        if (next_tile < K_TILE_MAX)
        {
            if (use_buffer0)   // current 0, so load next into buffer 1
            {
                copy(tAgA(_, _, next_tile), tAsA1);
                copy(tBgB(_, _, next_tile), tBsB1);
            }
            else               // current 1, load next into buffer 0
            {
                copy(tAgA(_, _, next_tile), tAsA0);
                copy(tBgB(_, _, next_tile), tBsB0);
            }
            cp_async_fence();  // commit outstanding cp_async for next tile
        }
    }

    // ------------------------------------------------------------------------
    //  Epilogue:  C = alpha * Acc + beta * C
    // ------------------------------------------------------------------------
    axpby(alpha, tCrC, beta, tCgC);
}

// ----------------------------------------------------------------------------
//  Host launcher (unchanged API)
// ----------------------------------------------------------------------------
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_simplified(int m, int n, int k,
                          Alpha alpha,
                          TA const* A, int ldA,
                          TB const* B, int ldB,
                          Beta beta,
                          TC*       C, int ldC,
                          cudaStream_t stream = 0)
{
    using CtaTiler      = Shape<_128, _128, _8>;
    using CThreadLayout = Layout<Shape<_16, _16>>;

    dim3 dimBlock(size(CThreadLayout{}));
    dim3 dimGrid(size(ceil_div(m, size<0>(CtaTiler{}))),
                 size(ceil_div(n, size<1>(CtaTiler{}))));

    gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(
        m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}

// ----------------------------------------------------------------------------
//  PyTorch glue
// ----------------------------------------------------------------------------
torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(),         "Input tensor A must be on CUDA");
    TORCH_CHECK(B.is_cuda(),         "Input tensor B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(),   "Input tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(),   "Input tensor B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,       "Tensors must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(1),
                "Inner dimension K must match:  A(M,K)  B(N,K)");

    const int m   = A.size(0);
    const int k   = A.size(1);
    const int n   = B.size(0);
    const int ldA = A.stride(0);
    const int ldB = B.stride(0);

    TORCH_CHECK(ldA == k, "A not row-major contiguous");
    TORCH_CHECK(ldB == k, "B not row-major contiguous");

    auto C  = torch::empty({m, n}, A.options());
    const int ldC = C.stride(0);

    using TA = float;
    using TB = float;
    using TC = float;
    using Alpha = float;
    using Beta  = float;

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
    name="cutlass_gemm_lib_double_buffered",
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
        print("Running custom CUTLASS kernel (double-buffered)...")
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
        print(f"Average CUTLASS kernel time (double-buffered): {cutlass_time_ms:.4f} ms")
        
        start_event.record()
        for _ in range(100):
            model_old.forward(A, B)
        end_event.record()
        torch.cuda.synchronize()
        
        pytorch_time_ms = start_event.elapsed_time(end_event) / 100
        print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")
        print(f"Speedup: {pytorch_time_ms/cutlass_time_ms:.2f}x")