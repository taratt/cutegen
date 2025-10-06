import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

wgmma_tma_gemm_source = """
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
#include "cutlass/arch/barrier.h"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/device_kernel.h"

using namespace cute;

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (M,K,P)
          class SmemLayoutB>  // (N,K,P)
struct SharedStorage
{
  alignas(128) cute::ArrayEngine<ElementA, cosize_v<SmemLayoutA>> A;
  alignas(128) cute::ArrayEngine<ElementB, cosize_v<SmemLayoutB>> B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //
  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int tma_transaction_bytes = sizeof(make_tensor_like(tensor<0>(tAsA)))
                                      + sizeof(make_tensor_like(tensor<0>(tBsB)));

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<1>(tAsA);

  // Total count of tiles
  int k_tile_count = size<1>(tAgA);
  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], 128);
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Start async loads for all pipes
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  //
  // PIPELINED MAIN LOOP
  //

  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive();
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], tma_transaction_bytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Epilogue (unpredicated)
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for an NT GEMM
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
  auto bP = Int<  3>{};  // Pipeline

  // Define the smem layouts (static)
  auto sA = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TA>{}, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(GMMA::Layout_MN_SW128_Atom<TB>{}, make_shape(bN,bK,bP));

  // Define the MMA
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN,GMMA::Major::MN>{});

  // Define the TMAs
  // Create Global memory tensors for TMA inspection
  Tensor mA = make_tensor(A, make_shape(M,K), dA);
  Tensor mB = make_tensor(B, make_shape(N,K), dB);

  // Create TMA Atoms with the desired copy operation on the source and destination
  Copy_Atom tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_,_,0), make_shape(bM,bK));
  Copy_Atom tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_,_,0), make_shape(bN,bK));

  //
  // Setup and Launch
  //

  // Launch parameter setup
  int smem_size = int(sizeof(SharedStorage<TA, TB, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(tiled_mma));
  dim3 dimCluster(2, 1, 1);
  dim3 dimGrid(round_up(size(ceil_div(m, bM)), dimCluster.x),
               round_up(size(ceil_div(n, bN)), dimCluster.y));
  cutlass::ClusterLaunchParams params = {dimGrid, dimBlock, dimCluster, smem_size, stream};

  void const* kernel_ptr = reinterpret_cast<void const*>(
                              &gemm_device<decltype(prob_shape), decltype(cta_tiler),
                                           TA, decltype(sA), decltype(tmaA),
                                           TB, decltype(sB), decltype(tmaB),
                                           TC, decltype(dC), decltype(tiled_mma),
                                           decltype(alpha), decltype(beta)>);

  CUTE_CHECK_ERROR(cudaFuncSetAttribute(
    kernel_ptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    smem_size));

  // Kernel Launch
  cutlass::Status status = cutlass::launch_kernel_on_cluster(params, kernel_ptr,
                                                             prob_shape, cta_tiler,
                                                             A, tmaA,
                                                             B, tmaB,
                                                             C, dC, tiled_mma,
                                                             alpha, beta);
  CUTE_CHECK_LAST();

  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Error: Failed at kernel Launch" << std::endl;
  }
}

// PyTorch wrapper - adapted for row-major layout
torch::Tensor wgmma_tma_gemm_nt_cuda(torch::Tensor A, torch::Tensor B) {
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
wgmma_tma_gemm_cpp_source = "torch::Tensor wgmma_tma_gemm_nt_cuda(torch::Tensor A, torch::Tensor B);"

# Compile the code
CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

# JIT compilation
wgmma_tma_gemm_lib = load_inline(
    name="wgmma_tma_gemm_lib",
    cpp_sources=wgmma_tma_gemm_cpp_source,
    cuda_sources=wgmma_tma_gemm_source,
    functions=["wgmma_tma_gemm_nt_cuda"],
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


class WGMMATMAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wgmma_tma_gemm_lib = wgmma_tma_gemm_lib

    def forward(self, A, B):
        """
        Performs C = A * B.T using the WGMMA TMA CUTLASS kernel.
        Args:
            A (torch.Tensor): A 2D tensor of shape (M, K) in fp16.
            B (torch.Tensor): A 2D tensor of shape (N, K) in fp16.
        Returns:
            torch.Tensor: The result tensor C of shape (M, N) in fp16.
        """
        return self.wgmma_tma_gemm_lib.wgmma_tma_gemm_nt_cuda(A, B)


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
            wgmma_tma_model = WGMMATMAModel().cuda()
            torch_model = TorchModel().cuda()

            # Define input tensors in fp16
            M, N, K = 5120, 5120, 4096
            A = torch.randn(M, K, device="cuda", dtype=torch.float16)
            B = torch.randn(N, K, device="cuda", dtype=torch.float16)

            # Run the forward pass
            print("Running WGMMA TMA CUTLASS kernel...")
            C_wgmma_tma = wgmma_tma_model.forward(A, B)
            print("WGMMA TMA kernel finished.")

            # Run PyTorch reference
            print("Running PyTorch native matmul for verification...")
            C_pytorch = torch_model.forward(A, B)
            print("PyTorch matmul finished.")

            # Check for correctness
            is_close = torch.allclose(C_wgmma_tma, C_pytorch, atol=1e-2, rtol=1e-3)
            print(f"\nVerification check: {'SUCCESS' if is_close else 'FAILURE'}")
            if not is_close:
                max_diff = torch.max(torch.abs(C_wgmma_tma - C_pytorch)).item()
                print(f"Max difference: {max_diff}")
            print("Output tensor shape:", C_wgmma_tma.shape)
            
            # Performance comparison
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            print("\nWarming up...")
            for _ in range(10):
                wgmma_tma_model.forward(A, B)
                torch_model.forward(A, B)
            torch.cuda.synchronize()
            
            # Benchmark WGMMA TMA
            print("Benchmarking WGMMA TMA kernel...")
            start_event.record()
            for _ in range(100):
                wgmma_tma_model.forward(A, B)
            end_event.record()
            torch.cuda.synchronize()
            
            wgmma_tma_time_ms = start_event.elapsed_time(end_event) / 100
            print(f"Average WGMMA TMA kernel time: {wgmma_tma_time_ms:.4f} ms")
            
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
            wgmma_tma_gflops = gflops / (wgmma_tma_time_ms * 1e-3)
            pytorch_gflops = gflops / (pytorch_time_ms * 1e-3)
            
            print(f"\nWGMMA TMA Performance: {wgmma_tma_gflops:.2f} GFLOP/s")
            print(f"PyTorch Performance: {pytorch_gflops:.2f} GFLOP/s")
            print(f"Speedup: {pytorch_time_ms/wgmma_tma_time_ms:.2f}x")