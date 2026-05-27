import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
from cutegen.config import CUTLASS_BASE_PATH, CUTLASS_INCLUDE_PATH
mm_cpp_decl = r"""
#include <torch/extension.h>
torch::Tensor mm_gemm(torch::Tensor A, torch::Tensor B);
"""
mm_cuda_src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/axpby.hpp>
// Simple helpers for cp.async pipelining (SM80+).
  __device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src, bool pred) {
    unsigned smem_addr = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
    if (pred) {
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(gmem_src));
    } else {
      // Zero-fill destination if pred is false (write 16B)
      uint4 z = {0, 0, 0, 0};
      *reinterpret_cast<uint4*>(smem_dst) = z;
    }
  }
  __device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
  __device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }
template <class TA, class TB, class TC>
__global__ __launch_bounds__(256, 2) void
gemm_nn_kernel(cute::Shape<int,int,int> shape_MNK,
               TA const* __restrict__ A, cute::Stride<int,int> dA,
               TB const* __restrict__ B, cute::Stride<int,int> dB,
               TC      * __restrict__ C, cute::Stride<int,int> dC)
{
  using namespace cute;
  int M = get<0>(shape_MNK);
  int N = get<1>(shape_MNK);
  int K = get<2>(shape_MNK);
  // Global memory tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA); // (M,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC); // (M,N)
  // CTA tiler: (M,N,K) = (128,128,8)
  auto cta_tiler = make_shape(Int<128>{}, Int<128>{}, Int<8>{});
  // Map blockIdx to (M,N) tiles
  auto cta_coord = make_coord(blockIdx.y, blockIdx.x, _);
  // Local tiles for this CTA
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (128,8) slice from A
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (128,128) slice from C
  // Shared memory tiles
  // A layout: (Mtile=128, Ktile=8) with K-dimension padding (stride_M = 9, stride_K = 1), double-buffered
  // Shared memory tiles
  // A layout: (Mtile=128, Ktile=8) with 16B-aligned row stride (stride_M = 12 floats), double-buffered
  __shared__ TA smemA[2 * 128 * 12];
  // B layout: (Ntile=128, Ktile=8) with N contiguous (stride_N = 1) and 16B-aligned K stride (stride_K = 132), double-buffered
  __shared__ TB smemB[2 * 8 * 132];
  // Make shared-memory tensors with explicit strides
  // A pages: stride row = 12, col = 1
  Tensor sA0 = make_tensor(
      make_smem_ptr(smemA + 0),
      make_shape(Int<128>{}, Int<8>{}),
      make_stride(Int<12>{}, Int<1>{}));   // (Mtile, Ktile) with 16B alignment
  Tensor sA1 = make_tensor(
      make_smem_ptr(smemA + 128 * 12),
      make_shape(Int<128>{}, Int<8>{}),
      make_stride(Int<12>{}, Int<1>{}));   // (Mtile, Ktile) with 16B alignment
  // B pages: stride_N = 1, stride_K = 132
  Tensor sB0 = make_tensor(
      make_smem_ptr(smemB + 0),
      make_shape(Int<128>{}, Int<8>{}),
      make_stride(Int<1>{}, Int<132>{})); // (Ntile, Ktile) with 16B alignment and N contiguous
  Tensor sB1 = make_tensor(
      make_smem_ptr(smemB + 8 * 132),
      make_shape(Int<128>{}, Int<8>{}),
      make_stride(Int<1>{}, Int<132>{})); // (Ntile, Ktile) with 16B alignment and N contiguous
  // Thread-level layouts
  auto tA = make_layout(make_shape(Int<32>{}, Int<8>{}));   // 256 threads
  auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));
  // Partition global and shared tiles to threads
  Tensor tAgA = local_partition(gA, tA, threadIdx.x);
  Tensor tAsA0 = local_partition(sA0, tA, threadIdx.x);
  Tensor tAsA1 = local_partition(sA1, tA, threadIdx.x);
  // Partitions for GEMM and output
  // Warp-level tiling: split CTA tile (128x128) into 8 warp tiles (64x32) and map lanes to 8x8 thread tiles
  int warp_id = threadIdx.x >> 5;
  int lane_id = threadIdx.x & 31;
  int warp_m = warp_id % 2;
  int warp_n = warp_id / 2;
  // Warp tiles cut from shared-memory and output tiles
  auto warp_tiler_A = make_shape(Int<64>{}, Int<8>{});   // (Mtile_warp, Ktile)
  auto warp_tiler_B = make_shape(Int<32>{}, Int<8>{});   // (Ntile_warp, Ktile)
  auto warp_tiler_C = make_shape(Int<64>{}, Int<32>{});  // (Mtile_warp, Ntile_warp)
  Tensor wAsA0 = local_tile(sA0, warp_tiler_A, make_coord(warp_m, 0));
  Tensor wAsA1 = local_tile(sA1, warp_tiler_A, make_coord(warp_m, 0));
  Tensor wBsB0 = local_tile(sB0, warp_tiler_B, make_coord(warp_n, 0));
  Tensor wBsB1 = local_tile(sB1, warp_tiler_B, make_coord(warp_n, 0));
  Tensor wCgC  = local_tile(gC,  warp_tiler_C, make_coord(warp_m, warp_n));
  // Map lanes within a warp to 8x8 per-thread tiles inside the (64x32) warp tile
  auto tC_warp = make_layout(make_shape(Int<8>{}, Int<4>{}));  // 8*4 = 32 threads
  Tensor tCsA0 = local_partition(wAsA0, tC_warp, lane_id, Step<_1, X>{}); // (m,k)
  Tensor tCsA1 = local_partition(wAsA1, tC_warp, lane_id, Step<_1, X>{}); // (m,k)
  Tensor tCsB0 = local_partition(wBsB0, tC_warp, lane_id, Step<X, _1>{}); // (n,k)
  Tensor tCsB1 = local_partition(wBsB1, tC_warp, lane_id, Step<X, _1>{}); // (n,k)
  Tensor tCgC  = local_partition(wCgC,  tC_warp, lane_id, Step<_1, _1>{}); // (m,n)
  Tensor tCrC = make_tensor_like(tCgC);
  clear(tCrC);
  int K_TILE_MAX = (K + 8 - 1) / 8;  // Number of K tiles (ceil-div)
  // Helper lambda: cp.async-based global->shared copy for A tile (128x8), 16B granularity, zero-fill on edges
  auto copy_A_tile_async = [&](int k_tile_idx, TA* __restrict__ sA_page_ptr) {
    int lda_m = get<0>(dA);
    int lda_k = get<1>(dA);
    int m_tile = static_cast<int>(blockIdx.y) * 128;
    int k_base = k_tile_idx * 8;
    int vec = static_cast<int>(threadIdx.x);
    int row = vec / 2;         // 2 vectors per row
    int kvec = vec % 2;        // which 4-wide vector along K
    int m_g = m_tile + row;
    int k_g0 = k_base + kvec * 4;
    // Destination base in shared (stride along row = 12, col = 1)
    int s_row_stride = 12;
    int s_off_base = row * s_row_stride + kvec * 4;
    float* s_ptr = reinterpret_cast<float*>(sA_page_ptr);
    bool full = (m_g < M) && (k_g0 + 3 < K);
    const float* g_ptr = reinterpret_cast<const float*>(&A[m_g * lda_m + k_g0 * lda_k]);
    void* s_dst = reinterpret_cast<void*>(&s_ptr[s_off_base]);
    if (full) {
      cp_async_16B(s_dst, g_ptr, true);
    } else {
      // Zero block then fixup the valid scalars
      cp_async_16B(s_dst, g_ptr, false);
      if (m_g < M) {
        if (k_g0 + 0 < K) s_ptr[s_off_base + 0] = A[m_g * lda_m + (k_g0 + 0) * lda_k];
        if (k_g0 + 1 < K) s_ptr[s_off_base + 1] = A[m_g * lda_m + (k_g0 + 1) * lda_k];
        if (k_g0 + 2 < K) s_ptr[s_off_base + 2] = A[m_g * lda_m + (k_g0 + 2) * lda_k];
        if (k_g0 + 3 < K) s_ptr[s_off_base + 3] = A[m_g * lda_m + (k_g0 + 3) * lda_k];
      }
    }
  };
  // Helper lambda: cp.async-based global->shared copy for B tile (128x8) with zero-fill on edges
  // Helper lambda: cp.async-based global->shared copy for B tile (128x8) using 16B transactions; scalar fallback for tails
  auto copy_B_tile_async = [&](int k_tile_idx, TB* __restrict__ sB_page_ptr) {
    int ldb_k = get<0>(dB); // stride along K
    int ldb_n = get<1>(dB); // stride along N (1 for contiguous)
    int n_tile = static_cast<int>(blockIdx.x) * 128;
    int k_base = k_tile_idx * 8;
    int vec = static_cast<int>(threadIdx.x);
    int k_local = vec / 32;     // 8 K rows per tile
    int nvec   = vec % 32;      // 32 vectors of 4 along N per K row
    int n_g0 = n_tile + nvec * 4;
    int k_g  = k_base + k_local;
    // Shared layout: stride_N = 1, stride_K = 132 (16B aligned)
    constexpr int s_stride_k = 132;
    int s_off_base = k_local * s_stride_k + nvec * 4;
    float* s_ptr = reinterpret_cast<float*>(sB_page_ptr);
    bool full = (k_g < K) && (n_g0 + 3 < N);
    const float* g_ptr = reinterpret_cast<const float*>(&B[k_g * ldb_k + n_g0 * ldb_n]);
    void* s_dst = reinterpret_cast<void*>(&s_ptr[s_off_base]);
    if (full) {
      cp_async_16B(s_dst, g_ptr, true);
    } else {
      // Zero block then fixup the valid scalars
      cp_async_16B(s_dst, g_ptr, false);
      if (k_g < K) {
        if (n_g0 + 0 < N) s_ptr[s_off_base + 0] = B[k_g * ldb_k + (n_g0 + 0) * ldb_n];
        if (n_g0 + 1 < N) s_ptr[s_off_base + 1] = B[k_g * ldb_k + (n_g0 + 1) * ldb_n];
        if (n_g0 + 2 < N) s_ptr[s_off_base + 2] = B[k_g * ldb_k + (n_g0 + 2) * ldb_n];
        if (n_g0 + 3 < N) s_ptr[s_off_base + 3] = B[k_g * ldb_k + (n_g0 + 3) * ldb_n];
      }
    }
  };
  // Preload first K-slice into page 0
  if (K_TILE_MAX > 0) {
    copy_A_tile_async(0, smemA + 0);
    copy_B_tile_async(0, smemB + 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
  }
  int smem_page = 0;
  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
    // Issue prefetch for next K-slice into the alternate shared-memory page
    int next_k = k_tile + 1;
    if (next_k < K_TILE_MAX) {
      if (smem_page == 0) {
        copy_A_tile_async(next_k, smemA + 128 * 12);
        copy_B_tile_async(next_k, smemB + 8 * 132);
      } else {
        copy_A_tile_async(next_k, smemA + 0);
        copy_B_tile_async(next_k, smemB + 0);
      }
      cp_async_commit();
    }
    // Compute partial GEMM on the current K-slice
    if (smem_page == 0) {
      gemm(tCsA0, tCsB0, tCrC);
    } else {
      gemm(tCsA1, tCsB1, tCrC);
    }
    // Wait for the next stage to arrive before swapping pages
    if (next_k < K_TILE_MAX) {
      cp_async_wait_all();
      __syncthreads();
      smem_page ^= 1; // flip between 0 and 1
    }
  }
  // Epilogue: write back
  axpby(1.0f, tCrC, 0.0f, tCgC);
}
torch::Tensor mm_gemm(torch::Tensor A, torch::Tensor B) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
  TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
  TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
              "mm_gemm expects float32 tensors");
  int64_t M = A.size(0);
  int64_t K = A.size(1);
  int64_t Kb = B.size(0);
  int64_t N = B.size(1);
  TORCH_CHECK(K == Kb, "Inner dimensions must match: A(M,K) x B(K,N)");
  auto A_c = A.contiguous();
  auto B_c = B.contiguous();
  auto C   = torch::empty({M, N}, A.options());
  auto ptrA = A_c.data_ptr<float>();
  auto ptrB = B_c.data_ptr<float>();
  auto ptrC = C.data_ptr<float>();
  auto dA = cute::make_stride(static_cast<int>(A_c.stride(0)), static_cast<int>(A_c.stride(1))); // (M,K)
  auto dB = cute::make_stride(static_cast<int>(B_c.stride(0)), static_cast<int>(B_c.stride(1))); // (K,N)
  auto dC = cute::make_stride(static_cast<int>(C.stride(0)),   static_cast<int>(C.stride(1)));   // (M,N)
  auto shape_MNK = cute::make_shape(static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
  dim3 dimBlock(256);
  dim3 dimGrid((N + 127) / 128, (M + 127) / 128);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  gemm_nn_kernel<<<dimGrid, dimBlock, 0, stream>>>(
      shape_MNK, ptrA, dA, ptrB, dB, ptrC, dC);
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUTE GEMM launch failed: ", cudaGetErrorString(err));
  return C;
}
"""
cute_mm = load_inline(
    name="cute_mm_inline",
    cpp_sources=mm_cpp_decl,
    cuda_sources=mm_cuda_src,
    functions=["mm_gemm"],
    extra_include_paths=[CUTLASS_BASE_PATH, CUTLASS_INCLUDE_PATH],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
    verbose=True,
)
class ModelNew(nn.Module):
    """
    Custom model using a single CUTE-based CUDA kernel for matrix multiplication (C = A @ B).
    """
    def __init__(self):
        super().__init__()
        self._ext = cute_mm
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure float32 dtype
        A = A.to(dtype=torch.float32)
        B = B.to(dtype=torch.float32)
        # If inputs are on CPU, move to CUDA for computation and then move result back
        inputs_on_cpu = (A.device.type == "cpu") and (B.device.type == "cpu")
        if inputs_on_cpu:
            A_cuda = A.cuda(non_blocking=True)
            B_cuda = B.cuda(non_blocking=True)
            C_cuda = self._ext.mm_gemm(A_cuda, B_cuda)
            return C_cuda.cpu()
        else:
            # Put both tensors on the same CUDA device
            device = A.device if A.is_cuda else B.device
            A_cuda = A.to(device, non_blocking=True)
            B_cuda = B.to(device, non_blocking=True)
            return self._ext.mm_gemm(A_cuda, B_cuda)


if __name__ == "__main__":
    torch.manual_seed(0)

    # Problem size (pick whatever you actually care about)
    M, K, N = 4096, 4096, 4096

    model = ModelNew().cuda()
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    # Warmup (so JIT / caches settle)
    for _ in range(3):
        C = model(A, B)
    torch.cuda.synchronize()

    # This call is what we’ll profile
    C = model(A, B)
    torch.cuda.synchronize()
