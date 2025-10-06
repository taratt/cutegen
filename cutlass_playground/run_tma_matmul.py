import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

cutlass_tma_gemm_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/arch/barrier.h>
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"

// Inlined helper functions
namespace cfk {

__device__ void barrierInit(uint64_t &tma_load_mbar, int numThreads) {
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  if (warp_idx == 0 and lane_predicate) {
    tma_load_mbar = 0;
    cute::initialize_barrier(tma_load_mbar, numThreads);
  }
  __syncthreads();
  cutlass::arch::fence_barrier_init();
}

template <typename SrcEngineA, typename SrcLayoutA, typename SrcEngineB,
          typename SrcLayoutB, typename DstEngineA, typename DstLayoutA,
          typename DstEngineB, typename DstLayoutB, typename AtomA,
          typename AtomB, class... ArgsA, class... ArgsB>
__device__ void
copy(cute::Tensor<SrcEngineA, SrcLayoutA> const &gA,
     cute::Tensor<SrcEngineB, SrcLayoutB> const &gB,
     cute::Tensor<DstEngineA, DstLayoutA> &&sA, cute::Tensor<DstEngineB, DstLayoutB> &&sB,
     cute::TiledCopy<AtomA, ArgsA...> const &tma_load_a,
     cute::TiledCopy<AtomB, ArgsB...> const &tma_load_b, uint64_t &tma_load_mbar,
     uint16_t mcast_mask_a = 0, uint16_t mcast_mask_b = 0) {

  using SrcTypeA = typename AtomA::ValType;
  using SrcTypeB = typename AtomB::ValType;
  __syncthreads();
  constexpr int kTmaTransactionBytes =
      cute::size(SrcLayoutA{}) * cute::sizeof_bits_v<SrcTypeA> / 8 +
      cute::size(SrcLayoutB{}) * cute::sizeof_bits_v<SrcTypeB> / 8;

  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  if (warp_idx == 0 and lane_predicate) {
    cute::set_barrier_transaction_bytes(tma_load_mbar, kTmaTransactionBytes);
    cute::copy(tma_load_a.with(tma_load_mbar, mcast_mask_a), gA, sA);
    cute::copy(tma_load_b.with(tma_load_mbar, mcast_mask_b), gB, sB);
  }
  __syncthreads();
}

template <typename TA, typename LayoutA, typename TB, typename LayoutB,
          typename TC, typename LayoutC, typename TiledMma>
__device__ void gemm(TiledMma &tiled_mma, const cute::Tensor<TA, LayoutA> &tCrA,
                     const cute::Tensor<TB, LayoutB> &tCrB,
                     cute::Tensor<TC, LayoutC> &tCrC) {
  cute::warpgroup_fence_operand(tCrC);
  cute::warpgroup_arrive();
  cute::gemm(tiled_mma, tCrA, tCrB, tCrC);
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();
  cute::warpgroup_fence_operand(tCrC);
}

namespace utils {
void set_smem_size(int smem_size, void const* kernel) {
  if (smem_size >= (48 << 10)) {
    cudaError_t result = cudaFuncSetAttribute(
        kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (cudaSuccess != result) {
      result = cudaGetLastError();
      throw std::runtime_error("Failed to set shared memory size");
    }
  }
}
} // namespace utils

} // namespace cfk

template <class ElementTypeA, class ElementTypeB, class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage {
  cute::array_aligned<ElementTypeA, cute::cosize_v<SmemLayoutA>> smem_a;
  cute::array_aligned<ElementTypeB, cute::cosize_v<SmemLayoutB>> smem_b;
  cute::uint64_t tma_load_mbar[1];
};

template <class TiledMma, class ClusterShape, class TA, class TiledCopyA,
          class TileShapeA, class GmemLayoutA, class SmemLayoutA, class TB,
          class TiledCopyB, class TileShapeB, class GmemLayoutB,
          class SmemLayoutB, class TC, class TileShapeC, class GmemLayoutC>
__global__ static void gemm_device(
    TA const *A, CUTE_GRID_CONSTANT TiledCopyA const tma_load_a,
    TileShapeA tile_shape_a, GmemLayoutA gmem_layout_a,
    SmemLayoutA smem_layout_a, TB const *B,
    CUTE_GRID_CONSTANT TiledCopyB const tma_load_b, TileShapeB tile_shape_b,
    GmemLayoutB gmem_layout_b, SmemLayoutB smem_layout_b, TC *C,
    TileShapeC tile_shape_c, GmemLayoutC gmem_layout_c) {
  using namespace cute;
  using X = Underscore;

  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;

  uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                  block_rank_in_cluster / cluster_shape_x};

  Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()),
                          smem_layout_a);
  Tensor mA = tma_load_a.get_tma_tensor(shape(gmem_layout_a));
  auto blk_coord_a = make_coord(uint64_t(blockIdx.x), _, uint64_t(blockIdx.z));
  Tensor gA = local_tile(mA, tile_shape_a, blk_coord_a);

  auto cta_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
  Tensor tAgA_x = cta_tma_a.partition_S(gA);
  Tensor tAsA_x = cta_tma_a.partition_D(sA);

  Tensor tAgA = group_modes<1, rank(tAgA_x)>(tAgA_x);
  Tensor tAsA = group_modes<1, rank(tAsA_x)>(tAsA_x);

  Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()),
                          smem_layout_b);
  Tensor mB = tma_load_b.get_tma_tensor(shape(gmem_layout_b));
  auto blk_coord_b = make_coord(uint64_t(blockIdx.y), _, uint64_t(blockIdx.z));
  Tensor gB = local_tile(mB, tile_shape_b, blk_coord_b);

  auto cta_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);
  Tensor tBgB_x = cta_tma_b.partition_S(gB);
  Tensor tBsB_x = cta_tma_b.partition_D(sB);

  Tensor tBgB = group_modes<1, rank(tBgB_x)>(tBgB_x);
  Tensor tBsB = group_modes<1, rank(tBsB_x)>(tBsB_x);

  TiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);

  Tensor tCsA = thread_mma.partition_A(sA);
  Tensor tCsB = thread_mma.partition_B(sB);
  Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  Tensor tCrB = thread_mma.make_fragment_B(tCsB);

  Tensor mC = make_tensor(make_gmem_ptr(C), gmem_layout_c);
  auto blk_coord_c = make_coord(uint64_t(blockIdx.x), uint64_t(blockIdx.y),
                                uint64_t(blockIdx.z));
  Tensor gC = local_tile(mC, tile_shape_c, blk_coord_c);
  Tensor tCgC = thread_mma.partition_C(gC);
  auto tCrC = partition_fragment_C(tiled_mma, tile_shape_c);

  uint16_t mcast_mask_a = 0;
  uint16_t mcast_mask_b = 0;

  if constexpr (cute::is_same_v<TiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
    auto block_layout = Layout<ClusterShape>{};
    for (int n = 0; n < size<1>(block_layout); ++n) {
      mcast_mask_a |=
          (uint16_t(1) << block_layout(cluster_local_block_id.x, n, Int<0>{}));
    }
  }

  if constexpr (cute::is_same_v<TiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
    auto block_layout = Layout<ClusterShape>{};
    for (int m = 0; m < size<0>(block_layout); ++m) {
      mcast_mask_b |=
          (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
    }
  }

  cute::prefetch_tma_descriptor(tma_load_a.get_tma_descriptor());
  cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
  
  cfk::barrierInit(*tma_load_mbar, size(TiledMma{}));
  __syncthreads();

#pragma unroll
  for (int stage = 0; stage < size<1>(tAgA); ++stage) {
    cfk::copy(tAgA(_, stage), tBgB(_, stage), tAsA(_, 0), tBsB(_, 0),
              tma_load_a, tma_load_b, *tma_load_mbar, mcast_mask_a,
              mcast_mask_b);
    cfk::gemm(tiled_mma, tCrA, tCrB, tCrC);
  }

#pragma unroll
  for (int i = 0; i < size(tCrC); ++i) {
    tCgC(i) = tCrC(i);
  }

  __syncthreads();
}

template <typename TA, typename TB, typename TC>
void tma_gemm_launch(int m, int n, int k, TA const *A, TB const *B,
                     TC *C, cudaStream_t stream = 0) {
  using namespace cute;

  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto L = 1; // Single batch

  using ClusterShape = Shape<_1, _1, _1>;
  using bM = Int<128>;
  using bN = Int<128>;
  using bK = Int<32>;

  using MmaA = cute::conditional_t<cute::is_same_v<TA, float>, tfloat32_t, TA>;
  using MmaB = cute::conditional_t<cute::is_same_v<TB, float>, tfloat32_t, TB>;

  // PyTorch tensors are row-major
  auto ptr_A = reinterpret_cast<MmaA const *>(A);
  auto ptr_B = reinterpret_cast<MmaB const *>(B);
  auto tile_shape_a = make_shape(bM{}, bK{});
  auto smem_layout_a = tile_to_shape(GMMA::Layout_K_SW64_Atom<MmaA>{}, tile_shape_a);
  Layout gmem_layout_a = make_layout(make_shape(M, K, L), make_stride<uint64_t>(K, 1, M * K));
  Tensor gA = make_tensor(ptr_A, gmem_layout_a);
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gA, smem_layout_a, tile_shape_a, Int<1>{});

  auto tile_shape_b = make_shape(bN{}, bK{});
  auto smem_layout_b = tile_to_shape(GMMA::Layout_K_SW64_Atom<MmaB>{}, tile_shape_b);
  Layout gmem_layout_b = make_layout(make_shape(N, K, L), make_stride<uint64_t>(K, 1, N * K));
  Tensor gB = make_tensor(ptr_B, gmem_layout_b);
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gB, smem_layout_b, tile_shape_b, Int<1>{});

  auto tile_shape_c = make_shape(bM{}, bN{});
  Layout gmem_layout_c = make_layout(make_shape(M, N, L), make_stride<uint64_t>(N, 1, M * N));

  int smem_size = int(sizeof(SharedStorage<MmaA, MmaB, decltype(smem_layout_a),
                                           decltype(smem_layout_b)>));

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, TC, Shape<bM, bN, bK>>(),
      Layout<Shape<_1, _1, _1>>()));

  void const *kernel = (void const *)gemm_device<
      TiledMma, ClusterShape, MmaA, decltype(tma_a), decltype(tile_shape_a),
      decltype(gmem_layout_a), decltype(smem_layout_a), MmaB, decltype(tma_b),
      decltype(tile_shape_b), decltype(gmem_layout_b),
      decltype(smem_layout_b), TC, decltype(tile_shape_c),
      decltype(gmem_layout_c)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 block_dims(size(TiledMma{}));
  dim3 grid_dims(ceil_div(size(M), size(bM{})), ceil_div(size(N), size(bN{})), L);
  dim3 cluster_dims(cute::size<0>(ClusterShape{}),
                    cute::size<1>(ClusterShape{}),
                    cute::size<2>(ClusterShape{}));
  cutlass::ClusterLaunchParams params{grid_dims, block_dims, cluster_dims,
                                      smem_size};

  cutlass::Status status = cutlass::launch_kernel_on_cluster(
      params, kernel, ptr_A, tma_a, tile_shape_a, gmem_layout_a,
      smem_layout_a, ptr_B, tma_b, tile_shape_b, gmem_layout_b,
      smem_layout_b, C, tile_shape_c, gmem_layout_c);
}

torch::Tensor cutlass_tma_gemm_cuda(torch::Tensor A, torch::Tensor B) {
    // Performs C = A * B.T using TMA
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be on CUDA");
    TORCH_CHECK(A.is_contiguous(), "Input tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Input tensor B must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32, "Input tensor A must be float32");
    TORCH_CHECK(B.scalar_type() == torch::kFloat32, "Input tensor B must be float32");
    TORCH_CHECK(A.dim() == 2, "Input tensor A must be 2D");
    TORCH_CHECK(B.dim() == 2, "Input tensor B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "Inner dimension K must match");

    const int m = A.size(0);
    const int k = A.size(1);
    const int n = B.size(0);

    auto C = torch::empty({m, n}, A.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    tma_gemm_launch<float, float, float>(
        m, n, k,
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        stream
    );

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}
"""

cutlass_tma_gemm_cpp_source = "torch::Tensor cutlass_tma_gemm_cuda(torch::Tensor A, torch::Tensor B);"

CUTLASS_PATH = os.getenv("CUTLASS_PATH")
if CUTLASS_PATH is None:
    raise RuntimeError(
        "CUTLASS_PATH environment variable not set. "
        "Please set it to the root of your CUTLASS repository."
    )

cutlass_tma_gemm_lib = load_inline(
    name="cutlass_tma_gemm_lib",
    cpp_sources=cutlass_tma_gemm_cpp_source,
    cuda_sources=cutlass_tma_gemm_source,
    functions=["cutlass_tma_gemm_cuda"],
    verbose=True,
    extra_cflags=['-std=c++17', '-O3'],
    extra_cuda_cflags=[f'-I {CUTLASS_PATH}/include/ -I {CUTLASS_PATH}/ -I {CUTLASS_PATH}/../ -I {CUTLASS_PATH}/tools/util/include/ -I {CUTLASS_PATH}/tools/library/include/ -I {CUTLASS_PATH}/../', '-gencode', 'arch=compute_90a,code=sm_90a'],
)


class TMAMatmulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cutlass_tma_gemm_lib = cutlass_tma_gemm_lib

    def forward(self, A, B):
        """
        Performs C = A * B.T using the custom CUTLASS TMA kernel.
        Args:
            A (torch.Tensor): A 2D tensor of shape (M, K).
            B (torch.Tensor): A 2D tensor of shape (N, K).
        Returns:
            torch.Tensor: The result tensor C of shape (M, N).
        """
        return self.cutlass_tma_gemm_lib.cutlass_tma_gemm_cuda(A, B)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping example.")
    else:
        # Check if we have SM90+ capability
        device_props = torch.cuda.get_device_properties(0)
        major, minor = device_props.major, device_props.minor
        if major < 9:
            print(f"TMA requires SM90+ but found SM{major}{minor}. Skipping.")
        else:
            model = TMAMatmulModel().cuda()

            M, N, K = 4096, 4096, 4096
            A = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B = torch.randn(N, K, device="cuda", dtype=torch.float32)

            print("Running custom CUTLASS TMA kernel...")
            C_cutlass = model.forward(A, B)
            print("CUTLASS TMA kernel finished.")

            print("Running PyTorch native matmul for verification...")
            C_pytorch = torch.matmul(A, B.T)
            print("PyTorch matmul finished.")

            is_close = torch.allclose(C_cutlass, C_pytorch, atol=1e-3, rtol=1e-4)
            print(f"\nVerification check: {'SUCCESS' if is_close else 'FAILURE'}")
            print("Output tensor shape:", C_cutlass.shape)
            
            # Performance comparison
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
            print(f"Average CUTLASS TMA kernel time: {cutlass_time_ms:.4f} ms")
            
            start_event.record()
            for _ in range(100):
                torch.matmul(A, B.T)
            end_event.record()
            torch.cuda.synchronize()
            
            pytorch_time_ms = start_event.elapsed_time(end_event) / 100
            print(f"Average PyTorch matmul time: {pytorch_time_ms:.4f} ms")
            print(f"Speedup: {pytorch_time_ms/cutlass_time_ms:.2f}x")