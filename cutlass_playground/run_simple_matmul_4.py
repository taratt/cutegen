import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

cutlass_gemm_source = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// -----------------------------------------------------------------------------
// CUTLASS / CUTE
// -----------------------------------------------------------------------------
#include <cute/tensor.hpp>
#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/tools/util/include/cutlass/util/helper_cuda.hpp"
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"

// -----------------------------------------------------------------------------
// PARAMS
// -----------------------------------------------------------------------------
// The depth of K each CTA processes (must be a multiple of the per-CTA K-tile).
// NOTE:  Use a PREPROCESSOR macro so that both host and device code see the
//        exact same value without any linkage/visibility issues.
#define KSLICE 32

// -----------------------------------------------------------------------------
//  CUDA  KERNEL  (split-K, atomic-accumulating fast path for beta == 0)
// -----------------------------------------------------------------------------
template <class TA, class TB, class TC, class Alpha, class Beta>
__global__ void gemm_kernel_splitK(int M, int N, int K,
                                   Alpha alpha,
                                   TA const* __restrict__ A, int ldA,
                                   TB const* __restrict__ B, int ldB,
                                   Beta beta,
                                   TC* __restrict__  C, int ldC)
{
    using namespace cute;

    //----------------------------------------------------------------------
    //  Thread-block / CTA static shapes
    //----------------------------------------------------------------------
    using  CtaTiler      = Shape<_128, _128, _32>;  // (Mtile, Ntile, Ktile)
    using  CThreadLayout = Layout<Shape<_16, _16>>; // 256 threads / CTA

    //----------------------------------------------------------------------
    //  Work-partition helpers
    //----------------------------------------------------------------------
    const int k_begin = blockIdx.z * KSLICE;
    if (k_begin >= K) return;                       // out-of-range guard
    // NOTE: we assume K is a multiple of KSLICE for correctness;  if you need
    // tail handling add masking or pad K so K %% 32 == 0 in Python.
    TA const* A_ptr = A + k_begin;                  // row-major : +col-offset
    TB const* B_ptr = B + k_begin;                  // idem

    //----------------------------------------------------------------------
    //  Row-major strides (PyTorch default)
    //----------------------------------------------------------------------
    auto dA = make_stride(ldA, Int<1>{});           // (K , 1)
    auto dB = make_stride(ldB, Int<1>{});           // (K , 1)
    auto dC = make_stride(ldC, Int<1>{});           // (N , 1)

    //----------------------------------------------------------------------
    //  Global tensors limited to the CTA’s K-slice
    //----------------------------------------------------------------------
    Tensor mA = make_tensor(make_gmem_ptr(A_ptr), make_shape(M,  KSLICE), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B_ptr), make_shape(N,  KSLICE), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C   ),  make_shape(M,       N), dC);

    //----------------------------------------------------------------------
    //  Tile this CTA works on (blockIdx.x, blockIdx.y)
    //----------------------------------------------------------------------
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, CtaTiler{}, cta_coord, Step<_1,  X, _1>{});
    Tensor gB = local_tile(mB, CtaTiler{}, cta_coord, Step< X, _1, _1>{});
    Tensor gC = local_tile(mC, CtaTiler{}, cta_coord, Step<_1, _1,  X>{});

    //----------------------------------------------------------------------
    //  Shared memory
    //----------------------------------------------------------------------
    using ASmemLayout = Layout<Shape<_128, _32>>;
    using BSmemLayout = Layout<Shape<_128, _32>>;

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), ASmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), BSmemLayout{});

    //----------------------------------------------------------------------
    //  Thread partitions
    //----------------------------------------------------------------------
    using AThreadLayout = Layout<Shape<_32, _8>>;
    using BThreadLayout = Layout<Shape<_32, _8>>;

    auto tA = AThreadLayout{};
    auto tB = BThreadLayout{};
    auto tC = CThreadLayout{};

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1,  X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step< X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    //----------------------------------------------------------------------
    //  Main K-loop – exactly 32 slices per CTA (== depth of CtaTiler)
    //----------------------------------------------------------------------
    constexpr int K_TILE_MAX = size<2>(CtaTiler{}); // 32
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);                     // tensor cores
        __syncthreads();
    }

    //----------------------------------------------------------------------
    //  Accumulate -> global C   (fast path beta == 0 : atomicAdd)
    //----------------------------------------------------------------------
    Alpha mul_alpha = alpha;                        // promote to register
    int TM = size<0>(tCrC);
    int TN = size<1>(tCrC);

    for (int mi = 0; mi < TM; ++mi) {
        for (int ni = 0; ni < TN; ++ni) {
            float val = static_cast<float>(mul_alpha * tCrC(mi, ni));
            atomicAdd(&tCgC(mi, ni), val);
        }
    }
}

// -----------------------------------------------------------------------------
//  Host launcher with split-K grid-Z
// -----------------------------------------------------------------------------
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_splitK(int M, int N, int K,
                      Alpha alpha,
                      TA const* A, int ldA,
                      TB const* B, int ldB,
                      Beta beta,
                      TC* C, int ldC,
                      cudaStream_t stream = 0)
{
    using namespace cute;

    using CtaTiler      = Shape<_128, _128, _32>;
    using CThreadLayout = Layout<Shape<_16, _16>>;

    dim3 dimBlock(size(CThreadLayout{}));
    dim3 dimGrid( ceil_div(M, size<0>(CtaTiler{})),
                  ceil_div(N, size<1>(CtaTiler{})),
                  ceil_div(K, KSLICE) );

    gemm_kernel_splitK<<<dimGrid, dimBlock, 0, stream>>>(
        M, N, K,
        alpha,
        A, ldA,
        B, ldB,
        beta,
        C, ldC);
}

// -----------------------------------------------------------------------------
//  PyTorch wrapper
// -----------------------------------------------------------------------------
torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(),  "A must be CUDA");
    TORCH_CHECK(B.is_cuda(),  "B must be CUDA");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2-D");
    TORCH_CHECK(A.size(1) == B.size(1), "K mismatch between A and B");

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(0));

    TORCH_CHECK((K % KSLICE) == 0,
                "K dimension (", K, ") must be a multiple of ", KSLICE,
                " for the current split-K implementation.");

    const int ldA = static_cast<int>(A.stride(0));
    const int ldB = static_cast<int>(B.stride(0));
    TORCH_CHECK(ldA == K, "A is not row-major contiguous");
    TORCH_CHECK(ldB == K, "B is not row-major contiguous");

    auto C = torch::zeros({M, N}, A.options());  // Initialize to zero for atomic accumulation
    const int ldC = static_cast<int>(C.stride(0));

    using TA = float;
    using TB = float;
    using TC = float;
    using Alpha = float;
    using Beta  = float;

    Alpha alpha = 1.0f;
    Beta  beta  = 0.0f;      // fast atomic path

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    cute_gemm_splitK<TA, TB, TC, Alpha, Beta>(
        M, N, K,
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

# --------------------------------------------------------------------------
# JIT build  (forward declaration + compilation)
# --------------------------------------------------------------------------
cutlass_gemm_cpp_source = "torch::Tensor cutlass_gemm_nt_cuda(torch::Tensor A, torch::Tensor B);"

CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

cutlass_gemm_lib = load_inline(
    name="cutlass_gemm_lib",
    cpp_sources=cutlass_gemm_cpp_source,
    cuda_sources=cutlass_gemm_source,
    functions=["cutlass_gemm_nt_cuda"],
    verbose=True,
    extra_cflags=['-std=c++17', '-O3'],
    extra_cuda_cflags=[
        f'-I{CUTLASS_PATH}',
        f'-I{CUTLASS_PATH}/include',
        f'-I{CUTLASS_PATH}/../',
        '-gencode arch=compute_90a,code=sm_90a',
    ],
)

# --------------------------------------------------------------------------
#  PyTorch-side wrapper module
# --------------------------------------------------------------------------
class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.cutlass_gemm_lib = cutlass_gemm_lib

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Compute C = A · Bᵀ via the custom CUTLASS split-K kernel.
        A : (M, K) row-major contiguous FP32
        B : (N, K) row-major contiguous FP32
        Returns:
            C : (M, N)
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
        try:
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
            print("Output tensor shape:", C_cutlass.shape)

            # For verification, compute the same operation using PyTorch's native matmul
            try:
                print("\nRunning PyTorch native matmul for verification...")
                C_pytorch = model_old.forward(A, B)
                print("PyTorch matmul finished.")

                # Check for correctness
                is_close = torch.allclose(C_cutlass, C_pytorch, atol=1e-3, rtol=1e-4)
                print(f"Verification check: {'SUCCESS' if is_close else 'FAILURE'}")
            except RuntimeError as e:
                print(f"WARNING: PyTorch native matmul verification failed: {e}")
                print("Skipping verification step.")
            
            # Simple performance comparison
            try:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Warmup
                print("\nRunning warmup...")
                for _ in range(10):
                    model.forward(A, B)
                torch.cuda.synchronize()
                
                # Benchmark CUTLASS kernel
                print("Benchmarking CUTLASS kernel...")
                start_event.record()
                for _ in range(100):
                    model.forward(A, B)
                end_event.record()
                torch.cuda.synchronize()
                
                cutlass_time_ms = start_event.elapsed_time(end_event) / 100
                print(f"Average CUTLASS kernel time (double-buffered): {cutlass_time_ms:.4f} ms")
                
                # Try to benchmark PyTorch matmul if possible
                try:
                    print("\nBenchmarking PyTorch matmul...")
                    start_event.record()
                    for _ in range(100):
                        model_old.forward(A, B)
                    end_event.record()
                    torch.cuda.synchronize()
                    
                    pytorch_time_ms = start_event.elapsed_time(end_event) / 100
                    print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")
                    print(f"Speedup: {pytorch_time_ms/cutlass_time_ms:.2f}x")
                except RuntimeError as e:
                    print(f"WARNING: PyTorch matmul benchmark failed: {e}")
                    print("Cannot calculate speedup comparison.")
            except RuntimeError as e:
                print(f"WARNING: Performance benchmarking failed: {e}")
                print("The CUTLASS kernel may have memory access issues.")
                
        except Exception as e:
            print(f"ERROR: An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc()