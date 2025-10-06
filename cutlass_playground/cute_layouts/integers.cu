#include <cute/tensor.hpp>
#include <iostream>
#include <type_traits>

using namespace cute;

template <typename T>
__host__ void print_integer_info(const char* name, T value) {
    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Value: " << value << std::endl;
    std::cout << "is_integral: " << is_integral<T>::value << std::endl;
    std::cout << "is_std_integral: " << is_std_integral<T>::value << std::endl;
    std::cout << "is_static: " << is_static<T>::value << std::endl;
    std::cout << "is_constant<5>: " << is_constant<5, T>::value << std::endl;
    std::cout << "sizeof(T): " << sizeof(T) << " bytes" << std::endl;
}

template <typename T1, typename T2>
__host__ void demonstrate_arithmetic(const char* op_name, T1 a, T2 b) {
    std::cout << "\n" << op_name << " operation:" << std::endl;

    auto sum = a + b;
    std::cout << a << " + " << b << " = " << sum;
    std::cout << " (result is_static: " << is_static<decltype(sum)>::value << ")" << std::endl;

    auto prod = a * b;
    std::cout << a << " * " << b << " = " << prod;
    std::cout << " (result is_static: " << is_static<decltype(prod)>::value << ")" << std::endl;
}

__global__ void kernel_with_static_ints() {
    constexpr auto static_3 = Int<3>{};
    constexpr auto static_4 = Int<4>{};

    constexpr auto compile_time_sum = static_3 + static_4;

    if (threadIdx.x == 0) {
        printf("GPU: Static 3 + Static 4 = %d (computed at compile time)\n", int(compile_time_sum));
        printf("GPU: Size of static int: %zu bytes\n", sizeof(static_3));
        printf("GPU: Size of dynamic int: %zu bytes\n", sizeof(int));
    }
}

__global__ void kernel_mixed_arithmetic(int dynamic_value) {
    constexpr auto static_2 = Int<2>{};

    auto mixed_result = static_2 * dynamic_value;

    if (threadIdx.x == 0) {
        printf("GPU: Static 2 * Dynamic %d = %d\n", dynamic_value, mixed_result);
        printf("GPU: Mixed result type is dynamic (sizeof = %zu)\n", sizeof(mixed_result));
    }
}

template <int N>
__host__ __device__ void compile_time_loop() {
    printf("Compile-time unrolled iteration %d\n", N);
    if constexpr (N > 1) {
        compile_time_loop<N-1>();
    }
}

__global__ void kernel_compile_time_loop() {
    if (threadIdx.x == 0) {
        printf("\nGPU: Compile-time loop unrolling:\n");
        compile_time_loop<5>();
    }
}

int main() {
    std::cout << "=====================================" << std::endl;
    std::cout << "CuTe Integer Types Teaching Example" << std::endl;
    std::cout << "=====================================" << std::endl;

    std::cout << "\n--- Part 1: Dynamic Integers ---" << std::endl;
    int dynamic_int = 42;
    size_t dynamic_size = 100;
    uint16_t dynamic_uint16 = 255;

    print_integer_info("dynamic int", dynamic_int);
    print_integer_info("dynamic size_t", dynamic_size);
    print_integer_info("dynamic uint16_t", dynamic_uint16);

    std::cout << "\n--- Part 2: Static Integers ---" << std::endl;
    auto static_1 = Int<1>{};
    auto static_5 = Int<5>{};
    auto static_10 = Int<10>{};

    print_integer_info("static Int<1>", static_1);
    print_integer_info("static Int<5>", static_5);
    print_integer_info("static Int<10>", static_10);

    std::cout << "\n--- Part 3: Arithmetic Operations ---" << std::endl;

    demonstrate_arithmetic("Static + Static", Int<3>{}, Int<4>{});
    demonstrate_arithmetic("Dynamic + Dynamic", 3, 4);
    demonstrate_arithmetic("Static + Dynamic", Int<3>{}, 4);

    std::cout << "\n--- Part 4: Compile-time Optimizations ---" << std::endl;

    constexpr auto compile_time_calc = Int<10>{} * Int<20>{} + Int<30>{};
    std::cout << "Compile-time calculation: 10 * 20 + 30 = " << compile_time_calc << std::endl;
    std::cout << "Result type is static: " << is_static<decltype(compile_time_calc)>::value << std::endl;

    int runtime_value = 10;
    auto mixed_calc = compile_time_calc * runtime_value;
    std::cout << "Mixed calculation: " << compile_time_calc << " * " << runtime_value
              << " = " << mixed_calc << std::endl;
    std::cout << "Mixed result is static: " << is_static<decltype(mixed_calc)>::value << std::endl;

    std::cout << "\n--- Part 5: Type Traits in Action ---" << std::endl;

    using StaticType = Int<42>;
    using DynamicType = int;

    std::cout << "Checking Int<42>:" << std::endl;
    std::cout << "  is_integral: " << is_integral<StaticType>::value << std::endl;
    std::cout << "  is_std_integral: " << is_std_integral<StaticType>::value << std::endl;
    std::cout << "  is_static: " << is_static<StaticType>::value << std::endl;
    std::cout << "  is_constant<42>: " << is_constant<42, StaticType>::value << std::endl;
    std::cout << "  is_constant<41>: " << is_constant<41, StaticType>::value << std::endl;

    std::cout << "\n--- Part 6: GPU Kernel Demonstrations ---" << std::endl;

    kernel_with_static_ints<<<1, 32>>>();
    cudaDeviceSynchronize();

    kernel_mixed_arithmetic<<<1, 32>>>(15);
    cudaDeviceSynchronize();

    kernel_compile_time_loop<<<1, 1>>>();
    cudaDeviceSynchronize();

    std::cout << "\n--- Part 7: Practical Use Cases ---" << std::endl;

    constexpr auto block_size = Int<256>{};
    auto grid_size = (1000 + block_size - 1) / block_size;
    std::cout << "Block size (static): " << block_size << std::endl;
    std::cout << "Grid size (dynamic): " << grid_size << std::endl;
    std::cout << "Grid size type is static: " << is_static<decltype(grid_size)>::value << std::endl;

    std::cout << "\n--- Part 8: Template Specialization ---" << std::endl;

    auto static_shape = make_shape(Int<4>{}, Int<8>{});
    std::cout << "Static shape (4, 8): " << static_shape << std::endl;
    std::cout << "Shape size (compile-time): " << size(static_shape) << std::endl;
    std::cout << "Size is static: " << is_static<decltype(size(static_shape))>::value << std::endl;

    auto dynamic_shape = make_shape(4, 8);
    std::cout << "Dynamic shape (4, 8): " << dynamic_shape << std::endl;
    std::cout << "Shape size (runtime): " << size(dynamic_shape) << std::endl;
    std::cout << "Size is static: " << is_static<decltype(size(dynamic_shape))>::value << std::endl;

    std::cout << "\n=====================================" << std::endl;
    std::cout << "Example completed successfully!" << std::endl;
    std::cout << "=====================================" << std::endl;

    return 0;
}