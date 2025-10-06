/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * This file has been modified to demonstrate a simplified, single-configuration GEMM.
 **************************************************************************************************/

// Standard Library
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

// CUDA
#include <cuda_runtime.h>

// CuTe
#include <cute/tensor.hpp>

// Utility headers
#include "cutlass/tools/util/include/cutlass/util/GPU_Clock.hpp"
#include "cutlass/tools/util/include/cutlass/util/helper_cuda.hpp"
#include "cutlass/tools/util/include/cutlass/util/print_error.hpp"

// A single-configuration CUDA kernel for a specific GEMM.
// All shapes, layouts, and tiling strategies are hardcoded inside for simplicity.
template <class TA, class TB, class TC, class Alpha, class Beta>
__global__ static void gemm_kernel(int M, int N, int K, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB,
                                  Beta beta, TC* C, int ldC) {
    using namespace cute;

    // 1. DEFINE THE SHAPES AND LAYOUTS (The "One Shape, One Layout" part)
    // The CTA tile size, this is our primary "shape" knob
    using CtaTiler = Shape<_128, _128, _8>;

    // The layout for threads within a CTA, our primary "layout" knob
    using CThreadLayout = Layout<Shape<_16, _16>>; // 256 threads per block (16x16)

    // Shared memory layouts
    using ASmemLayout = Layout<Shape<_128, _8>>;    // (M, K)
    using BSmemLayout = Layout<Shape<_128, _8>>;    // (N, K)

    // Thread layouts for loading data
    using AThreadLayout = Layout<Shape<_32, _8>>;   // (M, K)
    using BThreadLayout = Layout<Shape<_32, _8>>;   // (N, K)

    // Define strides for the 'NT' case (A is NxK, B is NxK)
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    // 2. KERNEL LOGIC (mostly unchanged from the original)
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), dA);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(N, K), dB);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), dC);

    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, CtaTiler{}, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, CtaTiler{}, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, CtaTiler{}, cta_coord, Step<_1, _1, X>{});

    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), ASmemLayout{});
    Tensor sB = make_tensor(make_smem_ptr(smemB), BSmemLayout{});

    auto tA = AThreadLayout{};
    auto tB = BThreadLayout{};
    auto tC = CThreadLayout{};

    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);
    Tensor tBgB = local_partition(gB, tB, threadIdx.x);
    Tensor tBsB = local_partition(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    auto K_TILE_MAX = size<2>(tAgA);
    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile) {
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);
        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();
        gemm(tCsA, tCsB, tCrC);
        __syncthreads();
    }
    axpby(alpha, tCrC, beta, tCgC);
}

// A single, self-contained host function to launch the GEMM.
// This function is hardcoded for the 'NT' case.
template <class TA, class TB, class TC, class Alpha, class Beta>
void cute_gemm_simplified(int m, int n, int k, Alpha alpha, TA const* A, int ldA, TB const* B, int ldB, Beta beta, TC* C, int ldC, cudaStream_t stream = 0) {
    using namespace cute;

    // Define the CTA tile size (must match the kernel's definition)
    using CtaTiler = Shape<_128, _128, _8>;
    // Define the thread layout (must match the kernel's definition)
    using CThreadLayout = Layout<Shape<_16, _16>>;

    // Determine grid and block dimensions from our fixed shapes
    dim3 dimBlock(size(CThreadLayout{}));
    dim3 dimGrid(size(ceil_div(m, size<0>(CtaTiler{}))),
                 size(ceil_div(n, size<1>(CtaTiler{}))));

    // Launch the single, simplified kernel
    gemm_kernel<<<dimGrid, dimBlock, 0, stream>>>(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
}

// Main function to initialize data, run the GEMM, and report performance.
int main(int argc, char** argv) {
    int m = 5120;
    if (argc >= 2) sscanf(argv[1], "%d", &m);

    int n = 5120;
    if (argc >= 3) sscanf(argv[2], "%d", &n);

    int k = 4096;
    if (argc >= 4) sscanf(argv[3], "%d", &k);

    // This version is hardcoded for transA='N', transB='T'
    const char transA = 'N';
    const char transB = 'T';

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    std::cout << "Running Simplified GEMM (Hardcoded for C = A * B^T)" << std::endl;
    std::cout << "M = " << m << std::endl;
    std::cout << "N = " << n << std::endl;
    std::cout << "K = " << k << std::endl;

    cute::device_init(0);

    // Allocate and initialize host memory
    std::vector<TA> h_A(m * k);
    std::vector<TB> h_B(n * k);
    std::vector<TC> h_C(m * n);

    for (size_t j = 0; j < h_A.size(); ++j) h_A[j] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    for (size_t j = 0; j < h_B.size(); ++j) h_B[j] = static_cast<TB>(2 * (rand() / double(RAND_MAX)) - 1);
    for (size_t j = 0; j < h_C.size(); ++j) h_C[j] = static_cast<TC>(-1);

    // Allocate device memory
    TA* d_A = nullptr;
    TB* d_B = nullptr;
    TC* d_C = nullptr;
    cudaMalloc(&d_A, h_A.size() * sizeof(TA));
    cudaMalloc(&d_B, h_B.size() * sizeof(TB));
    cudaMalloc(&d_C, h_C.size() * sizeof(TC));
    CUTE_CHECK_LAST();

    // Copy data from host to device
    cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(TA), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), h_B.size() * sizeof(TB), cudaMemcpyHostToDevice);
    CUTE_CHECK_LAST();

    // Calculate leading dimensions for the 'NT' case
    int ldA = m;
    int ldB = n;
    int ldC = m;

    // Run once for warmup
    cudaMemcpy(d_C, h_C.data(), h_C.size() * sizeof(TC), cudaMemcpyHostToDevice);
    cute_gemm_simplified(m, n, k, alpha, d_A, ldA, d_B, ldB, beta, d_C, ldC);
    CUTE_CHECK_LAST();

    // Time the execution
    const int timing_iterations = 100;
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        cute_gemm_simplified(m, n, k, alpha, d_A, ldA, d_B, ldB, beta, d_C, ldC);
    }
    double cute_time = timer.seconds() / timing_iterations;
    CUTE_CHECK_LAST();

    double gflops = (2.0 * m * n * k) * 1e-9;
    printf("CUTE_GEMM:     [%6.1f] GFLOP/s  (%6.4f ms)\n", gflops / cute_time, cute_time * 1000);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}