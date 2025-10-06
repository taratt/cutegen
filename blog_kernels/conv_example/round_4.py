import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

conv2d_implicit_gemm_cuda_source = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h> // For at::cuda::getCurrentCUDAStream()
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// WMMA tile dimensions
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Skew padding for shared memory to avoid bank conflicts
#define SKEW_HALF 8 // 8 half elements (16 bytes)

// CUDA built-in warpSize is 32 for supported architectures (sm_70+)
// This constant is used for host-side configuration (e.g. blockDim)
#define CUDA_WARP_SIZE_CONST 32 

// Threadblock configuration
#define WARPS_PER_BLOCK 8
// THREADS_PER_BLOCK must be evaluatable by host compiler for blockDim configuration
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * CUDA_WARP_SIZE_CONST) 

// Macro-tile dimensions computed by a threadblock
// BLOCK_M_TILES_WMMA * WMMA_M = output channels processed by a block
// BLOCK_N_TILES_WMMA * WMMA_N = output spatial elements processed by a block
#define BLOCK_M_TILES_WMMA 8
#define BLOCK_N_TILES_WMMA 8

#define TILE_M_PER_BLOCK (BLOCK_M_TILES_WMMA * WMMA_M) // e.g., 8 * 16 = 128 (for C_out dimension)
#define TILE_N_PER_BLOCK (BLOCK_N_TILES_WMMA * WMMA_N) // e.g., 8 * 16 = 128 (for N_batch * H_out * W_out dimension)

// Struct to hold precomputed k-dimension indices
struct KDecomposed {
    int kw;
    int kh;
    int ic;
    bool isValid; // True if current_k_idx < K_gemm
};

__global__ void conv2d_implicit_gemm_wmma_kernel(
    const float* __restrict__ input_ptr,    // Input: (N, Cin, Hin, Win)
    const float* __restrict__ weight_ptr,   // Weights: (Cout, Cin, Kh, Kw)
    const float* __restrict__ bias_ptr,     // Bias: (Cout) or nullptr
    float* __restrict__ output_ptr,         // Output: (N, Cout, Hout, Wout)
    const int N_batch, const int C_in, const int H_in, const int W_in,
    const int C_out, const int K_h, const int K_w,
    const int stride_h, const int stride_w,
    const int pad_h, const int pad_w,
    const int H_out, const int W_out,
    const int M_gemm, // C_out
    const int N_gemm, // N_batch * H_out * W_out
    const int K_gemm  // C_in * K_h * K_w
) {
    // Thread identification
    const int warp_id = threadIdx.x / warpSize;        // 0 .. WARPS_PER_BLOCK-1
    const int lane_id = threadIdx.x % warpSize;        // 0 .. 31 (or warpSize-1)

    // Top-left corner of the macro-tile this block is responsible for in GEMM terms
    const int block_row_gemm_start = TILE_M_PER_BLOCK * blockIdx.y;
    const int block_col_gemm_start = TILE_N_PER_BLOCK * blockIdx.x;

    // Shared memory for tiles of A (weights) and B (input/im2col)
    __shared__ half Asub_hi[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Asub_lo[TILE_M_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];
    __shared__ half Bsub_lo[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF];

    // Shared memory for precomputed k-indices
    __shared__ KDecomposed k_params[WMMA_K];

    // Accumulator fragments per warp.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[BLOCK_N_TILES_WMMA];
    #pragma unroll
    for (int i = 0; i < BLOCK_N_TILES_WMMA; ++i) {
        wmma::fill_fragment(acc_frag[i], 0.0f);
    }

    // Loop over the K_gemm dimension in tiles of WMMA_K
    for (int k_tile_start = 0; k_tile_start < K_gemm; k_tile_start += WMMA_K) {
        __syncthreads(); // Ensure previous MMA ops are done before loading new data or populating k_params

        // Populate k_params: First WMMA_K threads compute for the current k_tile
        if (threadIdx.x < WMMA_K) {
            int c_tile_offset = threadIdx.x; // Offset within the current K-tile (0 to WMMA_K-1)
            int current_k_idx = k_tile_start + c_tile_offset;

            if (current_k_idx < K_gemm) {
                // K_h > 0 and K_w > 0 is assumed here because K_gemm > 0 (checked in host code)
                // K_gemm = C_in * K_h * K_w
                k_params[c_tile_offset].kw = current_k_idx % K_w;
                int temp_div_kw = current_k_idx / K_w;
                k_params[c_tile_offset].kh = temp_div_kw % K_h;
                k_params[c_tile_offset].ic = temp_div_kw / K_h;
                k_params[c_tile_offset].isValid = true;
            } else {
                k_params[c_tile_offset].isValid = false;
                // Initialize to 0 to avoid using uninitialized values if logic error, though isValid should prevent access
                k_params[c_tile_offset].kw = 0;
                k_params[c_tile_offset].kh = 0;
                k_params[c_tile_offset].ic = 0;
            }
        }
        __syncthreads(); // Ensure k_params is fully populated before use by all threads

        // Load tile of A (weights) into shared memory
        for (int i = threadIdx.x; i < TILE_M_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_a_tile = i / WMMA_K; 
            int c_a_tile = i % WMMA_K; // Index for k_params (0 to WMMA_K-1)

            int oc_idx = block_row_gemm_start + r_a_tile; 
            
            float weight_val = 0.0f;
            if (oc_idx < C_out && k_params[c_a_tile].isValid) {
                int kw_eff = k_params[c_a_tile].kw;
                int kh_eff = k_params[c_a_tile].kh;
                int ic_eff = k_params[c_a_tile].ic;
                // If k_params[c_a_tile].isValid, then ic_eff < C_in is guaranteed.
                weight_val = weight_ptr[oc_idx * C_in * K_h * K_w + ic_eff * K_h * K_w + kh_eff * K_w + kw_eff];
            }
            Asub_hi[r_a_tile][c_a_tile] = __float2half(weight_val);
            Asub_lo[r_a_tile][c_a_tile] = __float2half(weight_val - __half2float(Asub_hi[r_a_tile][c_a_tile]));
        }

        // Load tile of B (input/im2col) into shared memory
        for (int i = threadIdx.x; i < TILE_N_PER_BLOCK * WMMA_K; i += THREADS_PER_BLOCK) {
            int r_b_tile = i / WMMA_K; // Corresponds to pixel_idx_offset_in_block
            int c_b_tile = i % WMMA_K; // Corresponds to k_idx_offset_in_ktile, index for k_params

            int pixel_idx = block_col_gemm_start + r_b_tile; 

            float input_val = 0.0f;
            if (pixel_idx < N_gemm && k_params[c_b_tile].isValid) {
                int kw_eff = k_params[c_b_tile].kw;
                int kh_eff = k_params[c_b_tile].kh;
                int ic_eff = k_params[c_b_tile].ic; // If k_params[c_b_tile].isValid, then ic_eff < C_in.

                int ow_eff = pixel_idx % W_out;
                int oh_eff = (pixel_idx / W_out) % H_out;
                int n_batch_idx = pixel_idx / (H_out * W_out);

                int h_in_eff = oh_eff * stride_h + kh_eff - pad_h;
                int w_in_eff = ow_eff * stride_w + kw_eff - pad_w;

                if (n_batch_idx < N_batch && // n_batch_idx check is still needed
                    h_in_eff >= 0 && h_in_eff < H_in &&
                    w_in_eff >= 0 && w_in_eff < W_in) {
                    input_val = input_ptr[n_batch_idx * C_in * H_in * W_in +
                                          ic_eff * H_in * W_in +
                                          h_in_eff * W_in +
                                          w_in_eff];
                }
            }
            Bsub_hi[r_b_tile][c_b_tile] = __float2half(input_val);
            Bsub_lo[r_b_tile][c_b_tile] = __float2half(input_val - __half2float(Bsub_hi[r_b_tile][c_b_tile]));
        }
        __syncthreads(); // Ensure shared memory (Asub, Bsub) is populated

        // Perform MMA operations
        int a_row_start_in_tile = warp_id * WMMA_M; 

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag_hi, a_frag_lo;
        wmma::load_matrix_sync(a_frag_hi, &Asub_hi[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);
        wmma::load_matrix_sync(a_frag_lo, &Asub_lo[a_row_start_in_tile][0], WMMA_K + SKEW_HALF);

        #pragma unroll
        for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
            int b_col_start_in_tile = n_tile * WMMA_N; 

            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag_hi, b_frag_lo;
            // Note: Bsub is indexed by [TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF]
            // load_matrix_sync for matrix_b (col_major in shared mem for WMMA) expects a pointer to the start of the tile.
            // Bsub_hi[b_col_start_in_tile][0] is correct if Bsub is conceptually (K_gemm_tile, N_gemm_tile_block)
            // but it's declared as Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF].
            // TILE_N_PER_BLOCK is for the N dimension of GEMM, WMMA_K for K dimension.
            // So Bsub_hi[b_col_start_in_tile] is selecting a row in shared memory.
            // This means Bsub is effectively (N_elements_in_block, K_elements_in_tile_shmem)
            // For col_major fragment, it expects shared memory to be K rows, N cols.
            // The current Bsub layout is [N_idx_in_block_tile][K_idx_in_K_tile].
            // If wmma::col_major for matrix_b means it expects data stored as if it were column major in global memory
            // (i.e. K is the leading dimension in shared memory for that tile), then Bsub should be Bsub[K][N].
            // Current: Bsub_hi[N_elements_for_block][K_elements_for_tile_shmem]
            // For wmma::load_matrix_sync(frag_b, &Bsub_hi[b_col_start_in_tile][0], stride)
            // where frag_b is wmma::col_major, it means the fragment elements b[k][j] are mapped to memory
            // such that elements of the same column j are contiguous.
            // The shared memory Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF] is (rows, cols)
            // A tile for B is WMMA_K rows and WMMA_N columns.
            // Bsub_hi[b_col_start_in_tile] gives a pointer to the start of the b_col_start_in_tile'th row of Bsub_hi.
            // This is fine if Bsub_hi is treated as K rows by N columns.
            // The current Bsub_hi is N rows by K columns.
            // If matrix_b is col_major, it means elements b(k, n) are stored at memory location base_ptr + k + n*ldm.
            // The current shared memory layout for Bsub is Bsub[N_idx][K_idx].
            // A tile of B is K_dim x N_dim (WMMA_K x WMMA_N).
            // If Bsub is [N_idx_in_block_tile][K_idx_in_K_tile], then &Bsub_hi[b_col_start_in_tile][0] points to the
            // (b_col_start_in_tile)-th row of the shared memory. This row corresponds to a specific output pixel's features
            // across the K dimension.
            // For wmma::col_major, the fragment expects data to be laid out column-wise in shared memory.
            // That is, for a KxN tile, element (k,n) is at base + k + n*K_dim_stride.
            // The current Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF] is effectively (N_dim_elements, K_dim_elements_padded).
            // So, Bsub_hi[row_idx_N][col_idx_K].
            // A tile for matrix B is K rows and N columns.
            // To load a KxN tile for wmma::col_major, we need to pick WMMA_N columns from Bsub_hi, where each column is WMMA_K elements deep.
            // This means we need to load from Bsub_hi[b_col_start_in_tile ... b_col_start_in_tile + WMMA_N-1] for the N dimension,
            // and for each of these, we take WMMA_K elements.
            // The current load `wmma::load_matrix_sync(b_frag_hi, &Bsub_hi[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);`
            // treats `&Bsub_hi[b_col_start_in_tile][0]` as the base pointer.
            // `b_col_start_in_tile` iterates from `0` to `(BLOCK_N_TILES_WMMA-1)*WMMA_N`. This is an index along the N dimension.
            // `WMMA_K + SKEW_HALF` is the stride (ldm).
            // For wmma::col_major, fragment(k,n) = ptr[k + n * ldm].
            // Here, ptr = &Bsub_hi[b_col_start_in_tile][0]. This points to the K-elements for the b_col_start_in_tile-th N-element.
            // This seems to be loading a N_rows x K_cols tile from shared memory if Bsub_hi is [N][K].
            // But fragment B is K_rows x N_cols.
            // This was likely an error in the original code's comments or my understanding.
            // If Bsub is (N,K) in shared memory, and we want to load a (K,N) tile for fragment B.
            // wmma::col_major means fragment[k][n] = shmem[k + n * ldm]
            // If Bsub is [N_idx_smem][K_idx_smem], then shmem[k + n * ldm] would mean Bsub_hi[k_coord][n_coord] where k_coord depends on n.
            // This is tricky. Let's assume the original code's load was correct for its intent.
            // The common way for GEMM A*B where A is (M,K) row-major and B is (K,N) col-major (or (N,K) row-major)
            // A_frag (row_major): frag_a[i][k] = shmem_A[i_offset + i][k_offset + k]
            // B_frag (col_major): frag_b[k][j] = shmem_B[k_offset + k][j_offset + j] (if shmem_B is KxN)
            // If shmem_B is (N,K) as it is (Bsub[N_elements][K_elements]), then for B_frag (col_major)
            // frag_b[k][j] = shmem_B[j_offset + j][k_offset + k]
            // The load `wmma::load_matrix_sync(b_frag, &Bsub[n_start][k_start], lds)`
            // For col_major B: base pointer is &Bsub[n_start][k_start], lds is K-dim stride.
            // This means Bsub is indexed [N][K].
            // `wmma::load_matrix_sync(b_frag_hi, &Bsub_hi[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);`
            // Here, b_col_start_in_tile is an offset in N. So this is `&Bsub_hi[N_offset][0]`.
            // The stride `WMMA_K + SKEW_HALF` is the distance between "columns" in shared memory.
            // For `wmma::col_major` fragment, `frag[k][n] = ptr[k + n * ldm]`.
            // `ptr` is `&Bsub_hi[b_col_start_in_tile][0]`. `ldm` is `WMMA_K + SKEW_HALF`.
            // This means `frag[k][n]` maps to `Bsub_hi[b_col_start_in_tile][k]` for `n=0`, and `Bsub_hi[b_col_start_in_tile][k + ldm]` for `n=1`.
            // This is not right. `ldm` should be the leading dimension of the matrix in shared memory.
            // If Bsub is N rows, K cols in memory: `Bsub[N_idx][K_idx]`.
            // A KxN tile for B means we need K rows, N columns.
            // If we want to load it as col_major fragment, we need `shmem[k][n]`.
            // The current Bsub_hi is `Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF]`.
            // This is `N_shmem_rows` x `K_shmem_cols`.
            // `b_col_start_in_tile` is an index into the N_shmem_rows.
            // `&Bsub_hi[b_col_start_in_tile][0]` is the start of a row in shared memory.
            // This row has `WMMA_K` elements (plus skew).
            // If this is passed to `load_matrix_sync` for a `wmma::col_major` fragment,
            // it means the fragment expects the data pointed by `ptr` to be the first column of the KxN tile.
            // And `ptr + ldm` is the second column, etc.
            // This implies that shared memory should be organized as `B_sh[K_dim][N_dim]`.
            // The current `Bsub_hi` is `[N_dim][K_dim]`.
            // This was likely a bug or misunderstanding in the original code or my interpretation of its shared memory layout vs wmma requirements.
            // However, the problem asks to *only* implement the precomputation of indices.
            // So, I should not change this loading logic, assuming it worked as intended in the original context.
            // The most common use is: Asub is MxK (row_major fragment, row_major shared mem), Bsub is KxN (col_major fragment, col_major shared mem).
            // If Bsub_sh is (K, N_padded), then load_matrix_sync(b_frag, &Bsub_sh[0][n_tile_offset_N], N_padded)
            // If Bsub_sh is (N, K_padded) as it is here, and we want to load a KxN tile.
            // For col_major fragment: frag[k][j] = PtrToTileInShared[k + j * StrideK]
            // PtrToTileInShared would be &Bsub_sh[n_tile_offset_N][0] if we consider a sub-matrix of Bsub_sh that is N rows, K cols.
            // This means we are loading Bsub_sh[n_tile_offset_N + j][k] into frag[k][j]. This is a transpose.
            // This is a common pattern: load B as row-major from global, store to shared as row-major (N,K), then load fragment B by transposing.
            // If `wmma::col_major` for fragment means it's KxN, and shared memory is N_rows x K_cols (stride K_cols_padded).
            // Then `wmma::load_matrix_sync(..., Bsub[n_start_row_idx][0], K_cols_padded, wmma::mem_row_major)`
            // Or if the shared memory is K_rows x N_cols (stride N_cols_padded).
            // `wmma::load_matrix_sync(..., Bsub[0][n_start_col_idx], N_cols_padded, wmma::mem_col_major)`
            // The current code uses `wmma::load_matrix_sync(b_frag_hi, &Bsub_hi[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);`
            // The third argument is `ldm`. For `wmma::col_major` fragment, this `ldm` is the number of rows of the matrix in shared memory (stride between columns).
            // So, it expects `Bsub_hi` to be K rows, and `WMMA_K + SKEW_HALF` is the stride to get to the next column.
            // This means `Bsub_hi` should be `Bsub_hi[WMMA_K][TILE_N_PER_BLOCK_PADDED]`.
            // But it's `Bsub_hi[TILE_N_PER_BLOCK][WMMA_K + SKEW_HALF]`.
            // This implies the shared memory is N rows, K columns.
            // And `b_col_start_in_tile` is a row index (in N dim). `WMMA_K + SKEW_HALF` is the length of a row (K dim).
            // This is consistent with loading it as if it's `wmma::mem_row_major`.
            // The default for `load_matrix_sync` is `layout_t = wmma::mem_row_major` if fragment is `wmma::col_major` for B.
            // This means `frag[i][j] = shmem[i * ldm + j]` for row_major fragment, or `frag[i][j] = shmem[j * ldm + i]` for col_major fragment.
            // For `matrix_b` (col_major fragment) and `mem_row_major` (default layout for load):
            // `b_frag.x[el]` maps to `shmem[row_b_frag][col_b_frag]` where `row_b_frag` and `col_b_frag` are derived from `el`.
            // Specifically, `b_frag[k][n]` (k=row, n=col in fragment) is `shmem[n][k]` effectively, when loading from row-major shared mem.
            // `shmem_ptr + n * ldm + k`. Here `ldm` is `WMMA_K + SKEW_HALF`. `shmem_ptr` is `&Bsub_hi[b_col_start_in_tile][0]`.
            // So `b_frag[k][n]` is loaded from `Bsub_hi[b_col_start_in_tile + n][k]`.
            // This means it loads a tile of N rows (WMMA_N) and K columns (WMMA_K) from shared memory.
            // This is correct for B_transpose. (N,K) tile. But fragment B is (K,N).
            // So `b_frag[k][n]` (fragment indexing) gets `Bsub_hi[b_col_start_in_tile + n][k]` (shared memory indexing).
            // This is loading B_transpose into B_fragment. This is the standard way if B is row-major in shared memory.
            // So, the original code is likely correct and follows this standard pattern. My optimization does not touch this part.

            wmma::load_matrix_sync(b_frag_hi, &Bsub_hi[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);
            wmma::load_matrix_sync(b_frag_lo, &Bsub_lo[b_col_start_in_tile][0], WMMA_K + SKEW_HALF);

            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_hi, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_hi, b_frag_lo, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_hi, acc_frag[n_tile]);
            wmma::mma_sync(acc_frag[n_tile], a_frag_lo, b_frag_lo, acc_frag[n_tile]);
        }
    }
    __syncthreads(); // Ensure all MMA operations for the block are complete

    // Store results from accumulator fragments to global memory
    __shared__ float C_shmem_frag_tile[WMMA_M][WMMA_N];

    #pragma unroll
    for (int n_tile = 0; n_tile < BLOCK_N_TILES_WMMA; ++n_tile) {
        for (int w_iter = 0; w_iter < WARPS_PER_BLOCK; ++w_iter) {
            if (warp_id == w_iter) {
                wmma::store_matrix_sync(&C_shmem_frag_tile[0][0], acc_frag[n_tile], WMMA_N, wmma::mem_row_major);
                
                for (int elem_idx_in_frag = lane_id; elem_idx_in_frag < WMMA_M * WMMA_N; elem_idx_in_frag += warpSize) {
                    int r_frag = elem_idx_in_frag / WMMA_N; 
                    int c_frag = elem_idx_in_frag % WMMA_N; 

                    int c_frag_row_start_for_this_warp = w_iter * WMMA_M; 
                    int c_frag_col_start_for_this_n_tile = n_tile * WMMA_N;  

                    int oc_idx = block_row_gemm_start + c_frag_row_start_for_this_warp + r_frag;
                    int pixel_idx = block_col_gemm_start + c_frag_col_start_for_this_n_tile + c_frag;

                    if (oc_idx < C_out && pixel_idx < N_gemm) {
                        int ow_eff = pixel_idx % W_out;
                        int oh_eff = (pixel_idx / W_out) % H_out;
                        int n_batch_idx = pixel_idx / (H_out * W_out);

                        float val = C_shmem_frag_tile[r_frag][c_frag];
                        if (bias_ptr != nullptr) {
                            val += bias_ptr[oc_idx];
                        }

                        output_ptr[n_batch_idx * C_out * H_out * W_out +
                                   oc_idx * H_out * W_out +
                                   oh_eff * W_out +
                                   ow_eff] = val;
                    }
                }
            }
            __syncthreads(); 
        } 
    } 
}


torch::Tensor conv2d_implicit_gemm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out) {

    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.device().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "Bias has wrong shape");
    }

    TORCH_CHECK(input.dim() == 4, "Input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D");
    TORCH_CHECK(input.size(0) == N_batch, "Input N_batch mismatch");
    TORCH_CHECK(input.size(1) == C_in, "Input C_in mismatch");
    TORCH_CHECK(input.size(2) == H_in, "Input H_in mismatch");
    TORCH_CHECK(input.size(3) == W_in, "Input W_in mismatch");
    TORCH_CHECK(weight.size(0) == C_out, "Weight C_out mismatch");
    TORCH_CHECK(weight.size(1) == C_in, "Weight C_in mismatch");
    TORCH_CHECK(weight.size(2) == K_h, "Weight K_h mismatch");
    TORCH_CHECK(weight.size(3) == K_w, "Weight K_w mismatch");

    auto output = torch::zeros({N_batch, C_out, H_out, W_out}, input.options());

    const int M_gemm = C_out;
    const int N_gemm = N_batch * H_out * W_out;
    const int K_gemm = C_in * K_h * K_w;

    if (M_gemm == 0 || N_gemm == 0) {
        return output;
    }
    if (K_gemm == 0) { 
         if (bias.defined()) { 
            output = output + bias.reshape({1, C_out, 1, 1});
        }
        return output;
    }

    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(
        (N_gemm + TILE_N_PER_BLOCK - 1) / TILE_N_PER_BLOCK, 
        (M_gemm + TILE_M_PER_BLOCK - 1) / TILE_M_PER_BLOCK  
    );

    const float* bias_ptr_data = bias.defined() ? bias.data_ptr<float>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    conv2d_implicit_gemm_wmma_kernel<<<grid_dim, block_dim, 0, stream>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr_data,
        output.data_ptr<float>(),
        N_batch, C_in, H_in, W_in,
        C_out, K_h, K_w,
        stride_h, stride_w, pad_h, pad_w,
        H_out, W_out,
        M_gemm, N_gemm, K_gemm
    );
    
    AT_CUDA_CHECK(cudaGetLastError());

    return output;
}
"""

conv2d_implicit_gemm_cuda_declaration = r"""
torch::Tensor conv2d_implicit_gemm_cuda(
    torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
    int N_batch, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int H_out, int W_out);
"""

# JIT compile the CUDA kernel
custom_conv2d_wmma_ops = load_inline(
    name="custom_conv2d_wmma_ops",
    cpp_sources=conv2d_implicit_gemm_cuda_declaration,
    cuda_sources=conv2d_implicit_gemm_cuda_source,
    functions=["conv2d_implicit_gemm_cuda"],
    verbose=True, 
    extra_cuda_cflags=["-arch=sm_70", "--use_fast_math", "-std=c++17"] 
)


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000): # num_classes is part of original signature, kept for consistency
        super(ModelNew, self).__init__()
        
        # Define Conv1 parameters (matching the original model)
        self.in_channels = 3
        self.out_channels = 96
        self.kernel_size_val = 11 # Assuming square kernel
        self.stride_val = 4       # Assuming square stride
        self.padding_val = 2      # Assuming square padding

        # Create a temporary Conv2d layer to initialize weights and bias
        temp_conv = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size_val, 
            stride=self.stride_val, 
            padding=self.padding_val,
            bias=True # nn.Conv2d has bias=True by default
        )
        self.conv1_weight = nn.Parameter(temp_conv.weight.detach().clone())
        if temp_conv.bias is not None:
            self.conv1_bias = nn.Parameter(temp_conv.bias.detach().clone())
        else:
            # Correctly register 'conv1_bias' as None if not present
            self.register_parameter('conv1_bias', None) 


        self.custom_conv_op = custom_conv2d_wmma_ops.conv2d_implicit_gemm_cuda

    def forward(self, x):
        N_batch = x.size(0)
        # C_in_runtime = x.size(1) # Should match self.in_channels
        H_in = x.size(2)
        W_in = x.size(3)

        # Calculate output dimensions
        H_out = (H_in + 2 * self.padding_val - self.kernel_size_val) // self.stride_val + 1
        W_out = (W_in + 2 * self.padding_val - self.kernel_size_val) // self.stride_val + 1
        
        # Bias tensor handling: pass an undefined tensor if bias is None.
        # The C++ TORCH_CHECK(bias.defined()) handles this by providing nullptr to kernel.
        bias_tensor = self.conv1_bias if self.conv1_bias is not None else torch.Tensor()


        x = self.custom_conv_op(
            x, self.conv1_weight, bias_tensor,
            N_batch, self.in_channels, H_in, W_in,
            self.out_channels, self.kernel_size_val, self.kernel_size_val, # K_h, K_w
            self.stride_val, self.stride_val, # stride_h, stride_w
            self.padding_val, self.padding_val, # pad_h, pad_w
            H_out, W_out
        )
        return x