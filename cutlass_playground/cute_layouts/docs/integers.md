Integers
CuTe makes great use of dynamic (known only at run-time) and static (known at compile-time) integers.

Dynamic integers (or “run-time integers”) are just ordinary integral types like int or size_t or uint16_t. Anything that is accepted by std::is_integral<T> is considered a dynamic integer in CuTe.

Static integers (or “compile-time integers”) are instantiations of types like std::integral_constant<Value>. These types encode the value as a static constexpr member. They also support casting to their underlying dynamic types, so they can be used in expressions with dynamic integers. CuTe defines its own CUDA-compatibe static integer types cute::C<Value> along with overloaded math operators so that math on static integers results in static integers. CuTe defines shortcut aliases Int<1>, Int<2>, Int<3> and _1, _2, _3 as conveniences, which you should see often within examples.

CuTe attempts to handle static and dynamic integers identically. In the examples that follow, all dynamic integers could be replaced with static integers and vice versa. When we say “integer” in CuTe, we almost always mean a static OR dynamic integer.

CuTe provides a number of traits to work with integers.

cute::is_integral<T>: Checks whether T is a static or dynamic integer type.

cute::is_std_integral<T>: Checks whether T is a dynamic integer type. Equivalent to std::is_integral<T>.

cute::is_static<T>: Checks whether T is an empty type (so instantiations cannot depend on any dynamic information). Equivalent to std::is_empty.

cute::is_constant<N,T>: Checks that T is a static integer AND its value is equivalent to N.