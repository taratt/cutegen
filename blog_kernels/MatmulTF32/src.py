import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# ---------------------------------------------------------------------
# Inline CUDA kernel for block-level Tensor-Core WMMA GEMM (128×128×16)
#  – Double-buffered cp.async GMEM→SMEM prefetch overlapped with compute
#  – A, C : row-major
#  – B    : column-major (host converts once, no on-the-fly transpose)
# ---------------------------------------------------------------------
matmul_cuda_source = r"""#include <torch/extension.h>
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
        printf("CUDA kernel launch failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// ------------------------------------------------------------------
// cp.async helpers (sm_80+) – provide host-side stubs when needed
// ------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// Convert generic ptr → SMEM address expected by cp.async
__device__ __forceinline__ unsigned int get_smem_addr(const void* ptr)
{
    return static_cast<unsigned int>(__cvta_generic_to_shared(ptr));
}

#define CP_ASYNC_CG(dst, src)                                       \
    {                                                               \
        unsigned int _dst_smem = get_smem_addr(dst);                \
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"    \
                     :: "r"(_dst_smem), "l"(src), "n"(16));         \
    }

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;" ::);
#define CP_ASYNC_WAIT_GROUP()   asm volatile("cp.async.wait_group 0;" ::);

#elif defined(__CUDA_ARCH__)
#error "This kernel requires sm_80 or newer for cp.async."
#else
// Host-side compilation – no-op macros
#define CP_ASYNC_CG(dst, src)
#define CP_ASYNC_COMMIT_GROUP()
#define CP_ASYNC_WAIT_GROUP()
#endif


// ------------------------------------------------------------------
// WMMA shapes / tiling configuration
// ------------------------------------------------------------------
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;                 // 32-wide K-tiles

// ------------------------------------------------------------------
// Padded stride for matrix A in SMEM
//   – must be a multiple of 8  (for 16-byte cp.async alignment)
//   – different from 32 to avoid ldmatrix bank conflicts
// ------------------------------------------------------------------
constexpr int BLOCK_K_PADDED = BLOCK_K + 16;   // 48 (32 + 16)

constexpr int WARPS_PER_BLOCK   = 8;           // 8 warps  (256 threads)
constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK;

constexpr int GROUPS_PER_K = BLOCK_K / 8;      // 4 groups of 8-half (16 B) copies

// ==================================================================
// Block-tiled WMMA kernel (one TB = one 128×128 C-tile)
// ==================================================================
__global__ void matmul_wmma_block_kernel(const half * __restrict__ A,
                                         const half * __restrict__ B,   // column-major!
                                         float       * __restrict__ C,
                                         const int N)
{
    // Tile indices
    const int global_tile_row = blockIdx.y;
    const int global_tile_col = blockIdx.x;

    // -------------------------------------------------------------
    // Double-buffered shared memory (A is padded)
    // -------------------------------------------------------------
    extern __shared__ half shared_mem[];
    half *As[2];
    half *Bs[2];

    const int size_A_panel = BLOCK_M * BLOCK_K_PADDED; // with padding
    const int size_B_panel = BLOCK_K * BLOCK_N;        // column-major, no padding

    As[0] = shared_mem;
    Bs[0] = As[0] + size_A_panel;
    As[1] = Bs[0] + size_B_panel;
    Bs[1] = As[1] + size_A_panel;

    const int thread_id = threadIdx.y * 32 + threadIdx.x;  // 0-255
    const int warp_id   = threadIdx.y;                     // 0-7

    // Each warp holds 8 accumulator fragments (128 / 16 = 8)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[BLOCK_N / WMMA_N];

    #pragma unroll
    for (int i = 0; i < BLOCK_N / WMMA_N; ++i)
        wmma::fill_fragment(c_frag[i], 0.0f);

    // Offsets of this TB in the global matrices
    const int global_row_offset = global_tile_row * BLOCK_M;
    const int global_col_offset = global_tile_col * BLOCK_N;

    // Warp base row (within the tile)
    const int warp_base_row = warp_id * WMMA_M;

    int cur_buf = 0, nxt_buf = 1;

    // -------------------------------------------------------------
    // 1. Pre-load first panels (tile_k = 0)
    // -------------------------------------------------------------
    {
        // ---- A panel (row-major, padded) -------------------------
        for (int idx = thread_id; idx < (BLOCK_M * BLOCK_K) / 8;
             idx += THREADS_PER_BLOCK)
        {
            const int row       = idx / GROUPS_PER_K;           // 0-127
            const int col_group = (idx % GROUPS_PER_K) * 8;     // 0,8,16,24

            const char *g_ptr = reinterpret_cast<const char *>(
                A + (global_row_offset + row) * N + col_group);

            char *s_ptr = reinterpret_cast<char *>(
                As[cur_buf] + row * BLOCK_K_PADDED + col_group);

            CP_ASYNC_CG(s_ptr, g_ptr);
        }

        // ---- B panel (COLUMN-MAJOR in GMEM & SMEM) ---------------
        for (int idx = thread_id; idx < (BLOCK_N * BLOCK_K) / 8;
             idx += THREADS_PER_BLOCK)
        {
            const int col       = idx / GROUPS_PER_K;           // 0-127
            const int k_group   = (idx % GROUPS_PER_K) * 8;     // 0,8,16,24
            const int k         = k_group;

            const char *g_ptr = reinterpret_cast<const char *>(
                B + (global_col_offset + col) * N + k);         // column-major load

            char *s_ptr = reinterpret_cast<char *>(
                Bs[cur_buf] + k + col * BLOCK_K);               // keep column-major

            CP_ASYNC_CG(s_ptr, g_ptr);
        }

        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP();
        __syncthreads();
    }

    // -------------------------------------------------------------
    // 2. Main K-loop
    // -------------------------------------------------------------
    for (int tile_k = 0; tile_k < N; tile_k += BLOCK_K)
    {
        const int next_k = tile_k + BLOCK_K;

        // Asynchronous prefetch of NEXT panels
        if (next_k < N)
        {
            // ---- A panel ----------------------------------------
            for (int idx = thread_id; idx < (BLOCK_M * BLOCK_K) / 8;
                 idx += THREADS_PER_BLOCK)
            {
                const int row       = idx / GROUPS_PER_K;
                const int col_group = (idx % GROUPS_PER_K) * 8;

                const char *g_ptr = reinterpret_cast<const char *>(
                    A + (global_row_offset + row) * N + next_k + col_group);

                char *s_ptr = reinterpret_cast<char *>(
                    As[nxt_buf] + row * BLOCK_K_PADDED + col_group);

                CP_ASYNC_CG(s_ptr, g_ptr);
            }

            // ---- B panel ----------------------------------------
            for (int idx = thread_id; idx < (BLOCK_N * BLOCK_K) / 8;
                 idx += THREADS_PER_BLOCK)
            {
                const int col       = idx / GROUPS_PER_K;
                const int k_group   = (idx % GROUPS_PER_K) * 8;
                const int k         = k_group;

                const char *g_ptr = reinterpret_cast<const char *>(
                    B + (global_col_offset + col) * N + next_k + k);   // column-major

                char *s_ptr = reinterpret_cast<char *>(
                    Bs[nxt_buf] + k + col * BLOCK_K);

                CP_ASYNC_CG(s_ptr, g_ptr);
            }

            CP_ASYNC_COMMIT_GROUP();
        }

        // ---------------------------------------------------------
        // Tensor-Core compute on CURRENT panels (32-wide)
        // ---------------------------------------------------------
        const half *warp_base_A = &As[cur_buf][warp_base_row * BLOCK_K_PADDED];

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       half, wmma::row_major> a_frag0, a_frag1;

        wmma::load_matrix_sync(a_frag0, warp_base_A,      BLOCK_K_PADDED); // k 0-15
        wmma::load_matrix_sync(a_frag1, warp_base_A + 16, BLOCK_K_PADDED); // k 16-31

        #pragma unroll
        for (int n_frag = 0; n_frag < BLOCK_N / WMMA_N; ++n_frag)
        {
            const half *tile_ptr_B0 = &Bs[cur_buf][(n_frag * WMMA_N) * BLOCK_K];
            const half *tile_ptr_B1 = tile_ptr_B0 + 16; // advance 16 in K

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           half, wmma::col_major> b_frag0, b_frag1;

            wmma::load_matrix_sync(b_frag0, tile_ptr_B0, BLOCK_K); // k 0-15
            wmma::load_matrix_sync(b_frag1, tile_ptr_B1, BLOCK_K); // k 16-31

            // Tensor-Core MACs
            wmma::mma_sync(c_frag[n_frag], a_frag0, b_frag0, c_frag[n_frag]);
            wmma::mma_sync(c_frag[n_frag], a_frag1, b_frag1, c_frag[n_frag]);
        }

        // Wait for next panels
        if (next_k < N)
            CP_ASYNC_WAIT_GROUP();
        __syncthreads();

        // Swap double buffers
        cur_buf ^= 1;
        nxt_buf ^= 1;
    }

    // -------------------------------------------------------------
    // 3. Store C tile back to global memory (FP32 row-major)
    // -------------------------------------------------------------
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
// Host-side wrapper – takes FP32 inputs, converts internally
// ==================================================================
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors.");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "Inputs must be float32.");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2-D matrices.");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1),
                "Only square matrices supported.");
    TORCH_CHECK(A.size(1) == B.size(0),
                "Incompatible matrix dimensions.");

    const int N = A.size(0);
    TORCH_CHECK(N % 128 == 0,
                "Matrix size must be multiple of 128.");

    // -------------------------------------------------------------
    // Prepare inputs:
    //   – A : row-major → FP16 contiguous
    //   – B : convert to FP16 + materialize COLUMN-MAJOR layout
    // -------------------------------------------------------------
    at::Tensor A_half = A.to(at::kHalf).contiguous();         // row-major
    at::Tensor B_half = B.to(at::kHalf).transpose(0, 1).contiguous(); // column-major

    // Output tensor (FP32)
    auto C = torch::zeros({N, N},
                          torch::dtype(torch::kFloat32).device(A.device()));

    dim3 block(32, WARPS_PER_BLOCK, 1);   // 32 threads/warp × 8 warps
    dim3 grid (N / BLOCK_N, N / BLOCK_M);

    size_t smem_bytes =
        2 * (BLOCK_M * BLOCK_K_PADDED + BLOCK_K * BLOCK_N) * sizeof(half); // with padding

    matmul_wmma_block_kernel<<<grid, block, smem_bytes>>>(
        reinterpret_cast<const half *>(A_half.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(B_half.data_ptr<at::Half>()),
        C.data_ptr<float>(),
        N);

    CUDA_CHECK_ERRORS();
    return C;
}"""

# ---------------------------------------------------------------------
# C++ header stub
# ---------------------------------------------------------------------
matmul_cpp_source = "torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B);"

# ---------------------------------------------------------------------
# Build the extension
# ---------------------------------------------------------------------
matmul_module = load_inline(
    name="custom_matmul_fp32_fix",
    cpp_sources=matmul_cpp_source,
    cuda_sources=matmul_cuda_source,
    functions=["matmul_cuda"],
    extra_cuda_cflags=[
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "--use_fast_math"
    ],
    verbose=False
)

# ---------------------------------------------------------------------
# Optimized PyTorch model leveraging the custom kernel
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized model that performs a single square matrix multiplication
    using the custom Tensor-Core WMMA kernel.  Accepts FP32 inputs and
    returns an FP32 result; all conversions are handled inside the
    compiled CUDA extension.
    """
    def __init__(self):
        super().__init__()
        self.matmul_cuda = matmul_module.matmul_cuda

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Raw CUDA kernel – no PyTorch ops allowed here.
        return self.matmul_cuda(A, B)