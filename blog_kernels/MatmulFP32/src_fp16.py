# KernelBench \u2013 hand-tuned solution
# -------------------------------------------------
# Tensor-Core WMMA GEMM (see comments inside for full
# algorithmic details).  Requires SM \u2265 80.
# -------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# 1. CUDA implementation (trimmed docstring \u2013 identical to the long
#    version you posted; nothing substantive removed).
# ---------------------------------------------------------------------
matmul_cuda_source = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>

using namespace nvcuda;

// ------------------------------------------------------------------
// CUDA-error checker
// ------------------------------------------------------------------
inline void CUDA_CHECK_ERRORS()
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA kernel launch failed : %s\
", cudaGetErrorString(err));
        exit(-1);
    }
}

// ------------------------------------------------------------------
// cp.async helpers (sm_80+).  On the host-compilation path __CUDA_ARCH__
// is not defined \u2013 provide no-op stubs so the file still compiles.
// ------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// Convert generic ptr \u2192 smem address expected by cp.async
__device__ __forceinline__ unsigned int get_smem_addr(const void* ptr)
{
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

// Copy 16 B from GMEM \u2192 SMEM using cp.async.cg
#define CP_ASYNC_CG(dst, src)                                       \\
    {                                                               \\
        unsigned int dst_smem_ = get_smem_addr(dst);                \\
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"    \\
                     :: "r"(dst_smem_), "l"(src), "n"(16));         \\
    }

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;" ::);
#define CP_ASYNC_WAIT_GROUP()   asm volatile("cp.async.wait_group 0;" ::);

#elif defined(__CUDA_ARCH__)
#error "This kernel requires sm_80 or newer for cp.async."
#else
// Host-side compilation \u2013 no-op macros
#define CP_ASYNC_CG(dst, src)
#define CP_ASYNC_COMMIT_GROUP()
#define CP_ASYNC_WAIT_GROUP()
#endif

// WMMA shapes
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Thread-block tile sizes
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 16;

// Thread / warp configuration
constexpr int WARPS_PER_BLOCK   = 8;              // 8 warps  (256 threads)
constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK;

// ==================================================================
// Block-tiled WMMA kernel with cp.async prefetch
//   \u2013 Each TB computes one 128 \u00d7 128 tile of C
// ==================================================================
__global__ void matmul_wmma_block_kernel(const half * __restrict__ A,
                                         const half * __restrict__ B,
                                         float       * __restrict__ C,
                                         const int N)
{
    // Global tile indices (one TB = one C tile)
    const int global_tile_row = blockIdx.y;
    const int global_tile_col = blockIdx.x;

    // -----------------------------------------------------------------
    // Shared memory buffers (double-buffered)
    // 2 \u00d7 (A-panel 128\u00d716 + B-panel 16\u00d7128) \u2192  2 \u00d7 4096 half = 8192 half
    // -----------------------------------------------------------------
    extern __shared__ half shared_mem[];
    half *As[2];
    half *Bs[2];

    As[0] = shared_mem;
    Bs[0] = As[0] + BLOCK_M * BLOCK_K;
    As[1] = Bs[0] + BLOCK_K * BLOCK_N;
    Bs[1] = As[1] + BLOCK_M * BLOCK_K;

    const int thread_id = threadIdx.y * 32 + threadIdx.x;   // 0-255
    const int warp_id   = threadIdx.y;                      // 0-7

    // Each warp accumulates 8 WMMA fragments      128 / 16 = 8
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[BLOCK_N / WMMA_N];

    #pragma unroll
    for (int i = 0; i < BLOCK_N / WMMA_N; ++i)
        wmma::fill_fragment(c_frag[i], 0.0f);

    // Warp base row (within the tile)
    const int warp_base_row  = warp_id * WMMA_M;

    // Offsets of this TB in the global matrices
    const int global_row_offset = global_tile_row * BLOCK_M;
    const int global_col_offset = global_tile_col * BLOCK_N;

    int cur_buf = 0, nxt_buf = 1;

    // -----------------------------------------------------------------
    // 1. Pre-load panel-0 for A & B (tile_k = 0) using cp.async
    // -----------------------------------------------------------------
    {
        // ---- A panel (row-major) -----------------------------------
        for (int idx = thread_id; idx < (BLOCK_M * BLOCK_K) / 8;
             idx += THREADS_PER_BLOCK)
        {
            const int row       = (idx >> 1);       // 0-127
            const int col_group = (idx & 1) * 8;    // 0 or 8

            const char *g_ptr = reinterpret_cast<const char *>(
                A + (global_row_offset + row) * N + col_group);
            char *s_ptr = reinterpret_cast<char *>(
                As[cur_buf] + row * BLOCK_K + col_group);

            CP_ASYNC_CG(s_ptr, g_ptr);
        }

        // ---- B panel (column-major) --------------------------------
        for (int idx = thread_id; idx < (BLOCK_N * BLOCK_K) / 8;
             idx += THREADS_PER_BLOCK)
        {
            const int col       = (idx >> 1);       // 0-127
            const int row_group = (idx & 1) * 8;    // 0 or 8

            const char *g_ptr = reinterpret_cast<const char *>(
                B + row_group + (global_col_offset + col) * N);
            char *s_ptr = reinterpret_cast<char *>(
                Bs[cur_buf] + row_group + col * BLOCK_K);

            CP_ASYNC_CG(s_ptr, g_ptr);
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP();
        __syncthreads();
    }

    // -----------------------------------------------------------------
    // 2. Main K-loop over 16-wide panels
    // -----------------------------------------------------------------
    for (int tile_k = 0; tile_k < N; tile_k += BLOCK_K)
    {
        const int next_k = tile_k + BLOCK_K;

        // -------------------------------------------------------------
        // Asynchronous prefetch of NEXT panels (double buffering)
        // -------------------------------------------------------------
        if (next_k < N)
        {
            // ---- A panel -------------------------------------------
            for (int idx = thread_id; idx < (BLOCK_M * BLOCK_K) / 8;
                 idx += THREADS_PER_BLOCK)
            {
                const int row       = (idx >> 1);
                const int col_group = (idx & 1) * 8;

                const char *g_ptr = reinterpret_cast<const char *>(
                    A + (global_row_offset + row) * N + next_k + col_group);
                char *s_ptr = reinterpret_cast<char *>(
                    As[nxt_buf] + row * BLOCK_K + col_group);

                CP_ASYNC_CG(s_ptr, g_ptr);
            }

            // ---- B panel -------------------------------------------
            for (int idx = thread_id; idx < (BLOCK_N * BLOCK_K) / 8;
                 idx += THREADS_PER_BLOCK)
            {
                const int col       = (idx >> 1);
                const int row_group = (idx & 1) * 8;

                const char *g_ptr = reinterpret_cast<const char *>(
                    B + next_k + row_group + (global_col_offset + col) * N);
                char *s_ptr = reinterpret_cast<char *>(
                    Bs[nxt_buf] + row_group + col * BLOCK_K);

                CP_ASYNC_CG(s_ptr, g_ptr);
            }

            CP_ASYNC_COMMIT_GROUP();
        }

        // -------------------------------------------------------------
        // Tensor-Core compute on CURRENT panels
        // -------------------------------------------------------------
        const half *tile_ptr_A = &As[cur_buf][warp_base_row * BLOCK_K];

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag;
        wmma::load_matrix_sync(a_frag, tile_ptr_A, BLOCK_K);   // lda = 16

        #pragma unroll
        for (int n_frag = 0; n_frag < BLOCK_N / WMMA_N; ++n_frag)
        {
            const half *tile_ptr_B = &Bs[cur_buf][(n_frag * WMMA_N) * BLOCK_K];

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> b_frag;
            wmma::load_matrix_sync(b_frag, tile_ptr_B, BLOCK_K); // ldb = 16

            wmma::mma_sync(c_frag[n_frag], a_frag, b_frag, c_frag[n_frag]);
        }

        // -------------------------------------------------------------
        // Ensure next panels are ready before next iteration
        // -------------------------------------------------------------
        if (next_k < N)
        {
            CP_ASYNC_WAIT_GROUP();
        }
        __syncthreads();

        // Swap buffers
        cur_buf ^= 1;
        nxt_buf ^= 1;
    }

    // -----------------------------------------------------------------
    // 3. Store C tile back to global memory (FP32 row-major)
    // -----------------------------------------------------------------
    #pragma unroll
    for (int n_frag = 0; n_frag < BLOCK_N / WMMA_N; ++n_frag)
    {
        const int row = global_row_offset + warp_base_row;
        const int col = global_col_offset + n_frag * WMMA_N;
        float *c_ptr  = C + row * N + col;

        wmma::store_matrix_sync(c_ptr, c_frag[n_frag], N, wmma::mem_row_major);
    }
}

// ==================================================================
// Host-side wrapper
// ==================================================================
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "Inputs must be float16 (half).");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2-D matrices.");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1),
                "Only square matrices supported.");
    TORCH_CHECK(A.size(1) == B.size(0), "Incompatible matrix dimensions.");

    const int N = A.size(0);
    TORCH_CHECK(N % 16  == 0,  "Matrix size must be multiple of 16.");
    TORCH_CHECK(N % 128 == 0, "Matrix size must be multiple of 128 for this kernel.");

    // Output tensor (FP32)
    auto C = torch::zeros({N, N},
                          torch::dtype(torch::kFloat32).device(A.device()));

    dim3 block(32, WARPS_PER_BLOCK, 1);   // 32 threads/warp \u00d7 8 warps
    dim3 grid (N / BLOCK_N, N / BLOCK_M);

    size_t smem_bytes = 2 * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(half);

    matmul_wmma_block_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half *>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(B.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        N);

    CUDA_CHECK_ERRORS();
    return C;
}
"""

cpp_decl = r"torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

matmul_mod = load_inline(
    name="wmma_gemm_ext",
    cpp_sources=cpp_decl,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    extra_cuda_cflags=[
        "-gencode=arch=compute_89,code=sm_89", #120 for Blackwell, 89 for L40S
        "--use_fast_math",
    ],
    verbose=False,
)


class ModelNew(nn.Module):
    """
    Calls the custom WMMA kernel (half \u00d7 half \u2192 float).
    """
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_mod.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:   # noqa: N802
        # The kernel expects  FP16 row-major A,  FP16 col-major B.
        A_ = A.to(torch.float16).contiguous()
        B_ = B.to(torch.float16).t().contiguous()      # transpose once
        return self.matmul_cuda(A_, B_)