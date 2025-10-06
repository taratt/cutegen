# compile
```bash
nvcc -std=c++17 -O3 -I $CUTLASS_PATH/include/ -I $CUTLASS_PATH/ -I $CUTLASS_PATH/../ -gencode arch=compute_90a,code=sm_90a -o integers integers.cu
```

# run
```bash
./integers
```