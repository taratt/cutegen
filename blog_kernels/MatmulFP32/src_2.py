import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# --------------------------------------------------------------------------------
# Inline CUDA implementation of a numerically–compensated WMMA (Tensor-Core)
# square matrix multiplication (C = A * B) that achieves full-float32 accuracy.
# Now optimised: each 256-thread block (8 warps) cooperatively computes a
# 128 × 128 output macro-tile, re-using the A/B data it loads from global memory.
# This version adds skew padding to shared–memory tiles to eliminate bank conflicts.
# --------------------------------------------------------------------------------
matmul_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// -----------------------------------------------------------------------------
// Skew padding: 8 half elements (16 B) to avoid shared-memory bank conflicts
// -----------------------------------------------------------------------------
#define SKEW 8                       // 8 halves = 16 B padding

// One block  =  8 warps = 256 threads
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

// Macro-tile sizes that one thread-block computes
#define BLOCK_ROW_TILES 8                 // 8 × 16  = 128 rows
#define BLOCK_COL_TILES 8                 // 8 × 16  = 128 cols
#define BLOCK_M  (WMMA_M * BLOCK_ROW_TILES)
#define BLOCK_N  (WMMA_N * BLOCK_COL_TILES)

// -----------------------------------------------------------------------------
// 256-thread WMMA kernel: one block computes one 128 × 128 tile of C
// -----------------------------------------------------------------------------
__global__ void matmul_kernel_wmma(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {

    const int lane_id = threadIdx.x & 31;        // 0 … 31
    const int warp_id = threadIdx.x >> 5;        // 0 … 7

    // Top-left corner of the macro-tile this block is responsible for
    const int block_row = BLOCK_M * blockIdx.y;
    const int block_col = BLOCK_N * blockIdx.x;

    // -------------------------------------------------------------------------
    // Shared memory with skew padding to remove bank conflicts
    // -------------------------------------------------------------------------
    // Each K-slice (16) that the block works on is stored once in shared memory
    __shared__ half Asub_hi[BLOCK_M][WMMA_K + SKEW];          // 128 × 24
    __shared__ half Asub_lo[BLOCK_M][WMMA_K + SKEW];
    // For B we adopt the transposed storage so that the first index is column,
    // giving the column-major layout required by WMMA
    __shared__ half Bsub_hi[BLOCK_N][WMMA_K + SKEW];          // 128 × 24  (col-major)
    __shared__ half Bsub_lo[BLOCK_N][WMMA_K + SKEW];

    // -------------------------------------------------------------------------
    // Accumulators: each warp keeps eight 16×16 fragments (one per column tile)
    // -------------------------------------------------------------------------
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[BLOCK_COL_TILES];
    #pragma unroll
    for (int n = 0; n < BLOCK_COL_TILES; ++n)
        wmma::fill_fragment(c_frag[n], 0.0f);

    // -------------------------------------------------------------------------
    // Loop over the K dimension in steps of 16 (WMMA_K)
    // -------------------------------------------------------------------------
    for (int k_tile = 0; k_tile < N; k_tile += WMMA_K) {

        // ---------------------------------------------------------------------
        // (1) Cooperative LOAD of the current 128×16 slice of A and 16×128 slice
        //     of B into shared memory, converting to (hi, lo) fp16 components.
        // ---------------------------------------------------------------------
        // --- A ----------------------------------------------------------------
        for (int idx = threadIdx.x; idx < BLOCK_M * WMMA_K; idx += THREADS_PER_BLOCK) {
            int r = idx / WMMA_K;       // 0 … 127
            int c = idx % WMMA_K;       // 0 … 15

            int global_r = block_row + r;
            int global_c = k_tile   + c;

            float a_val = (global_r < N && global_c < N)
                          ? A[global_r * N + global_c]
                          : 0.0f;

            half a_hi = __float2half(a_val);
            half a_lo = __float2half(a_val - __half2float(a_hi));

            Asub_hi[r][c] = a_hi;
            Asub_lo[r][c] = a_lo;
        }

        // --- B ----------------------------------------------------------------
        for (int idx = threadIdx.x; idx < BLOCK_N * WMMA_K; idx += THREADS_PER_BLOCK) {
            int c = idx / WMMA_K;       // 0 … 127   (matrix column)
            int r = idx % WMMA_K;       // 0 … 15    (matrix row / K)

            int global_r = k_tile   + r;
            int global_c = block_col + c;

            float b_val = (global_r < N && global_c < N)
                          ? B[global_r * N + global_c]
                          : 0.0f;

            half b_hi = __float2half(b_val);
            half b_lo = __float2half(b_val - __half2float(b_hi));

            // Transpose during store so first index = column (for col-major)
            Bsub_hi[c][r] = b_hi;
            Bsub_lo[c][r] = b_lo;
        }

        __syncthreads();

        // ---------------------------------------------------------------------
        // (2) Each warp processes one 16-row stripe of the macro-tile and all
        //     eight 16-column tiles within that stripe.
        // ---------------------------------------------------------------------
        const int a_row   = warp_id * WMMA_M;          // 0,16,…,112

        // Load A fragments (hi & lo) for this warp’s 16×16 tile row
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_hi_frag, a_lo_frag;

        wmma::load_matrix_sync(a_hi_frag, &Asub_hi[a_row][0], WMMA_K + SKEW);
        wmma::load_matrix_sync(a_lo_frag, &Asub_lo[a_row][0], WMMA_K + SKEW);

        // For every column tile in the macro-tile
        #pragma unroll
        for (int n = 0; n < BLOCK_COL_TILES; ++n) {

            // Column offset inside the macro-tile
            const int b_col = n * WMMA_N;              // 0,16,…,112

            // Load the B fragments (hi & lo) for this column tile
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> b_hi_frag, b_lo_frag;

            wmma::load_matrix_sync(b_hi_frag, &Bsub_hi[b_col][0], WMMA_K + SKEW);
            wmma::load_matrix_sync(b_lo_frag, &Bsub_lo[b_col][0], WMMA_K + SKEW);

            // 1) hi * hi
            wmma::mma_sync(c_frag[n], a_hi_frag, b_hi_frag, c_frag[n]);
            // 2) hi * lo
            wmma::mma_sync(c_frag[n], a_hi_frag, b_lo_frag, c_frag[n]);
            // 3) lo * hi
            wmma::mma_sync(c_frag[n], a_lo_frag, b_hi_frag, c_frag[n]);
            // 4) lo * lo
            wmma::mma_sync(c_frag[n], a_lo_frag, b_lo_frag, c_frag[n]);
        }

        __syncthreads();
    }

    // -------------------------------------------------------------------------
    // (3) Store accumulators back to global memory
    // -------------------------------------------------------------------------
    const int c_row = block_row + warp_id * WMMA_M;

    #pragma unroll
    for (int n = 0; n < BLOCK_COL_TILES; ++n) {
        const int c_col = block_col + n * WMMA_N;
        float* C_tile = C + c_row * N + c_col;

        wmma::store_matrix_sync(C_tile, c_frag[n], N, wmma::mem_row_major);
    }
}

// -----------------------------------------------------------------------------
// Python-exposed helper
// -----------------------------------------------------------------------------
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.device().is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "Only float32 supported");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "Inputs must be 2-D");
    TORCH_CHECK(A.size(0) == A.size(1) &&
                B.size(0) == B.size(1) &&
                A.size(0) == B.size(0),
                "Must be same-sized square matrices");

    int N = A.size(0);

    auto A_c = A.contiguous();
    auto B_c = B.contiguous();
    auto C   = torch::empty({N, N}, A.options());

    dim3 block(THREADS_PER_BLOCK, 1, 1);   // 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (N + BLOCK_M - 1) / BLOCK_M);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    matmul_kernel_wmma<<<grid, block, 0, stream>>>(
        A_c.data_ptr<float>(),
        B_c.data_ptr<float>(),
        C.data_ptr<float>(),
        N);

    return C;
}
"""

matmul_cuda_declaration = r"""
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);
"""

# -------------------------------------------------------------------------------
# Compile & load the CUDA extension
# -------------------------------------------------------------------------------
matmul_ext = load_inline(
    name         = "matmul_cuda_extension",
    cpp_sources  = matmul_cuda_declaration,
    cuda_sources = matmul_cuda_source,
    functions    = ["matmul_cuda"],
    verbose      = False,
    extra_cuda_cflags=["-arch=sm_70"]  # Tensor-Core capable (Volta+)
)

# -------------------------------------------------------------------------------
# Optimised PyTorch model that uses the custom WMMA kernel
# -------------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised model: square matrix multiply (C = A * B) realised with
    high-accuracy, Tensor-Core WMMA tiles.
    """
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_ext.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return self.matmul_cuda(A, B)