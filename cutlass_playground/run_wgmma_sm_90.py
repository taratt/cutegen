import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

wgmma_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cute/tensor.hpp>
#include "cutlass/cluster_launch.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

using namespace cute;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;
};

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), ASmemLayout{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), BSmemLayout{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor sA_ = as_position_independent_swizzle_tensor(sA);
  Tensor tAsA = thr_copy_a.partition_D(sA_);                           // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor sB_ = as_position_independent_swizzle_tensor(sB);
  Tensor tBsB = thr_copy_b.partition_D(sB_);                           // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCsA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCsB)));              // MMA_N
  CUTE_STATIC_ASSERT_V((size<2>(tCsA) == size<2>(tCsB)));              // MMA_K

  // Clear the accumulators
  clear(tCrC);

  // Total number of k-tiles
  auto K_TILE_MAX  = size<3>(tAgA);
  // Number of pipelined k-tiles in smem
  auto K_PIPE_MAX  = size<3>(tAsA);

  //
  // PREFETCH
  //

  // Prefetch all but the last
  CUTE_UNROLL
  for (int k = 0; k < K_PIPE_MAX-1; ++k)
  {
    copy(copy_a, tAgA(_,_,_,k), tAsA(_,_,_,k));
    copy(copy_b, tBgB(_,_,_,k), tBsB(_,_,_,k));
    cp_async_fence();
  }

  // Clear the accumulators
  clear(tCrC);

  __syncthreads();

  //
  // PIPELINED MAIN LOOP
  //

  // Current pipe to read from
  int k_pipe_read  = 0;
  // Current pipe to write to
  int k_pipe_write = K_PIPE_MAX-1;

  CUTE_NO_UNROLL
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    int k_tile_next = k_tile + (K_PIPE_MAX-1);
    k_tile_next = (k_tile_next >= K_TILE_MAX) ? K_TILE_MAX-1 : k_tile_next;

    //
    // Copy gmem to smem for k_tile_write
    //

    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe_write));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe_write));
    cp_async_fence();

    // Advance k_pipe_write
    ++k_pipe_write;
    k_pipe_write = (k_pipe_write == K_PIPE_MAX) ? 0 : k_pipe_write;

    //
    // Compute on k_tile
    //

    // Wait on all cp.async -- optimize by pipelining to overlap GMEM reads
    cp_async_wait<0>();

    warpgroup_fence_operand(tCrC);
    warpgroup_arrive();
    // (V,M,K) x (V,N,K) => (V,M,N)
    cute::gemm(mma, tCrA(_,_,_,k_pipe_read), tCrB(_,_,_,k_pipe_read), tCrC);
    warpgroup_commit_batch();
    /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);

    // Advance k_pipe_read
    ++k_pipe_read;
    k_pipe_read = (k_pipe_read == K_PIPE_MAX) ? 0 : k_pipe_read;
  }

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the thread layouts (static)
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TA>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 32x4 m-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, TB>{},
                                    Layout<Shape<_16,_8>>{}, // Thr layout 32x4 n-major
                                    Layout<Shape< _8,_1>>{});// Val layout  8x1 n-major

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  //
  // Setup and Launch
  //

  // Launch parameter setup
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(1, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  int  smemBytes = sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>);

  auto* kernel_ptr = &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                  TA, decltype(dA), decltype(sA), decltype(copyA),
                                  TB, decltype(dB), decltype(sB), decltype(copyB),
                                  TC, decltype(dC), decltype(tiled_mma),
                                  decltype(alpha), decltype(beta)>;

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(kernel_ptr,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        smemBytes));

  // Kernel Launch
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smemBytes, stream};
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, (void const*) kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, dA, sA, copyA,
                                                             B, dB, sB, copyB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

// PyTorch wrapper - adapted for row-major layout
torch::Tensor wgmma_gemm_nt_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Input tensor B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16, "Input tensor A must be float16");
    TORCH_CHECK(B.scalar_type() == torch::kFloat16, "Input tensor B must be float16");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimension K (A.size(1)) must match B.size(1)");

    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(0);

    // Create output tensor
    auto C = torch::empty({m, n}, A.options());

    // For PyTorch row-major tensors, we need to transpose the logic
    // PyTorch: A is (M, K), B is (N, K), C is (M, N)
    // We want: C = A @ B.T
    // For column-major CUTLASS, we need to think of this as: C.T = B @ A.T
    // So we pass B as the first matrix and A as the second matrix to compute C.T
    
    using ElementA = cute::half_t;
    using ElementB = cute::half_t;
    using ElementC = cute::half_t;
    using ElementCompute = cute::half_t;

    ElementCompute alpha = ElementCompute(1.0f);
    ElementCompute beta = ElementCompute(0.0f);

    // Get the current CUDA stream from PyTorch
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Since PyTorch uses row-major and CUTLASS expects column-major,
    // we swap the roles: Pass B as first matrix, A as second matrix
    // This computes C.T = B @ A.T, which gives us C = A @ B.T in row-major
    gemm_nt<ElementB, ElementA, ElementC>(
        n, m, k,  // Swapped dimensions
        alpha,
        reinterpret_cast<ElementB const*>(B.data_ptr<at::Half>()),
        B.stride(0),  // For row-major (N,K), stride is K
        reinterpret_cast<ElementA const*>(A.data_ptr<at::Half>()),
        A.stride(0),  // For row-major (M,K), stride is K
        beta,
        reinterpret_cast<ElementC*>(C.data_ptr<at::Half>()),
        C.stride(0),  // For row-major (M,N), stride is N
        stream
    );

    // Check for any CUDA errors after kernel launch
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}
"""

# Define the C++ function signature
wgmma_gemm_cpp_source = "torch::Tensor wgmma_gemm_nt_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the code
CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

# JIT compilation
wgmma_gemm_lib = load_inline(
    name="wgmma_gemm_lib",
    cpp_sources=wgmma_gemm_cpp_source,
    cuda_sources=wgmma_gemm_source,
    functions=["wgmma_gemm_nt_cuda"],
    verbose=True,
    extra_cflags=['-std=c++17', '-O3'],
    extra_cuda_cflags=[
        f'-I{CUTLASS_PATH}', 
        f'-I{CUTLASS_PATH}/include',
        f'-I{CUTLASS_PATH}/../',
        f'-I{CUTLASS_PATH}/tools/util/include',
        '-gencode', 'arch=compute_90a,code=sm_90a',
        '-DCUTLASS_ARCH_MMA_SM90_SUPPORTED'
    ],
)


class WGMMAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wgmma_gemm_lib = wgmma_gemm_lib

    def forward(self, A, B):
        """
        Performs C = A * B.T using the WGMMA CUTLASS kernel.
        Args:
            A (torch.Tensor): A 2D tensor of shape (M, K) in fp16.
            B (torch.Tensor): A 2D tensor of shape (N, K) in fp16.
        Returns:
            torch.Tensor: The result tensor C of shape (M, N) in fp16.
        """
        return self.wgmma_gemm_lib.wgmma_gemm_nt_cuda(A, B)


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, A, B):
        return torch.matmul(A, B.T)


# Example Usage and Benchmarking
if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping example.")
    else:
        # Check if we have SM90 GPU
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major < 9:
            print(f"GPU compute capability {device_props.major}.{device_props.minor} detected.")
            print("This example requires NVIDIA Hopper Architecture (SM90). Skipping.")
        else:
            # Instantiate models
            wgmma_model = WGMMAModel().cuda()
            torch_model = TorchModel().cuda()

            # Define input tensors in fp16
            M, N, K = 5120, 5120, 4096
            A = torch.randn((M, K), device="cuda", dtype=torch.float16)
            B = torch.randn((N, K), device="cuda", dtype=torch.float16)
            # A = torch.ones((M, K), device="cuda", dtype=torch.float16)
            # B = torch.randn((N, K), device="cuda", dtype=torch.float16)

            # Run the forward pass
            print("Running WGMMA CUTLASS kernel...")
            C_wgmma = wgmma_model.forward(A, B)
            print("WGMMA kernel finished.")

            # Run PyTorch reference
            print("Running PyTorch native matmul for verification...")
            C_pytorch = torch_model.forward(A, B)
            print("PyTorch matmul finished.")

            # Check for correctness
            is_close = torch.allclose(C_wgmma, C_pytorch, atol=1e-2, rtol=1e-3)
            print(f"\nVerification check: {'SUCCESS' if is_close else 'FAILURE'}")
            if not is_close:
                max_diff = torch.max(torch.abs(C_wgmma - C_pytorch)).item()
                print(f"Max difference: {max_diff}")
                print("First 100 elements of the output tensor:")
                print(C_wgmma.flatten()[:100])
                print("First 100 elements of the reference tensor:")
                print(C_pytorch.flatten()[:100])
            print("Output tensor shape:", C_wgmma.shape)
            
            # Performance comparison
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            print("\nWarming up...")
            for _ in range(10):
                wgmma_model.forward(A, B)
                torch_model.forward(A, B)
            torch.cuda.synchronize()
            
            # Benchmark WGMMA
            print("Benchmarking WGMMA kernel...")
            start_event.record()
            for _ in range(100):
                wgmma_model.forward(A, B)
            end_event.record()
            torch.cuda.synchronize()
            
            wgmma_time_ms = start_event.elapsed_time(end_event) / 100
            print(f"Average WGMMA kernel time: {wgmma_time_ms:.4f} ms")
            
            # Benchmark PyTorch
            print("Benchmarking PyTorch matmul...")
            start_event.record()
            for _ in range(100):
                torch_model.forward(A, B)
            end_event.record()
            torch.cuda.synchronize()
            
            pytorch_time_ms = start_event.elapsed_time(end_event) / 100
            print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")
            
            # Calculate GFLOPS
            gflops = (2.0 * M * N * K) * 1e-9
            wgmma_gflops = gflops / (wgmma_time_ms * 1e-3)
            pytorch_gflops = gflops / (pytorch_time_ms * 1e-3)
            
            print(f"\nWGMMA Performance: {wgmma_gflops:.2f} GFLOP/s")
            print(f"PyTorch Performance: {pytorch_gflops:.2f} GFLOP/s")
            print(f"Speedup: {pytorch_time_ms/wgmma_time_ms:.2f}x")