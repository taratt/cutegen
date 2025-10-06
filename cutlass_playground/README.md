# CUTLASS Playground

## Dependencies
- CUTLASS (TODO: build with `-DCUTLASS_ENABLE_CUBLAS=1`)
- Colfax cutlass-kernels (https://github.com/ColfaxResearch/cutlass-kernels/tree/master)

## Simple Matmul
```bash
nvcc -std=c++17 -O3 -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -gencode arch=compute_90a,code=sm_90a -o simple_matmul simple_matmul.cu
```

## TMA Matmul
```bash
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -I $CUTLASS_PATH/tools/util/include/ -I $CUTLASS_PATH/tools/library/include/ -I $COLFAX_PATH/ -I $COLFAX_PATH/include/ -I $COLFAX_PATH/lib/ -gencode arch=compute_90a,code=sm_90a -o tma_matmul tma_matmul.cu
```

```bash
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -DCUTLASS_ENABLE_CUBLAS=1 -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -I $CUTLASS_PATH/tools/util/include/ -I $CUTLASS_PATH/tools/library/include/ -lcublas -gencode arch=compute_90a,code=sm_90a -o tma_matmul_2 tma_matmul_2.cu
```

## WGMMA SM90
```bash
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -I $CUTLASS_PATH/tools/util/include/ -I $CUTLASS_PATH/tools/library/include/ -gencode arch=compute_90a,code=sm_90a -o wgmma_sm_90 wgmma_sm_90.cu
```

## WGMMA TMA SM90
```bash
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -I $CUTLASS_PATH/tools/util/include/ -I $CUTLASS_PATH/tools/library/include/ -gencode arch=compute_90a,code=sm_90a -o wgmma_tma_sm_90 wgmma_tma_sm_90.cu
```