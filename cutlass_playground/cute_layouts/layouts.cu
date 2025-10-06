#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
}

template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
  printf("\n");
}

int main()
{

  Layout s8 = make_layout(Int<8>{});
  Layout d8 = make_layout(8);
  // print1D(s8);
  // print1D(d8);

  Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  Layout s2xd4 = make_layout(make_shape(Int<2>{},4));
  // print1D(s2xs4);
  // print1D(s2xd4);
  // print2D(s2xs4);
  // print2D(s2xd4);
  // print_layout(s2xs4);
  // print_layout(s2xd4);

  Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                              make_stride(Int<12>{},Int<1>{}));
  Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                                LayoutLeft{});
  Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                                LayoutRight{});
  // print1D(s2xd4_a);
  // print1D(s2xd4_col);
  // print1D(s2xd4_row);
  // print2D(s2xd4_a);
  // print2D(s2xd4_col);
  // print2D(s2xd4_row);
  // print_layout(s2xd4_a);
  // print_layout(s2xd4_col);
  // print_layout(s2xd4_row);

  // Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
  //                           make_stride(4,make_stride(2,1)));
  // Layout s2xh4_col = make_layout(shape(s2xh4),
  //                               LayoutLeft{});
  // print1D(s2xh4);
  // print2D(s2xh4);
  // print_layout(s2xh4);
  // print1D(s2xh4_col);
  // print2D(s2xh4_col);
  // print_layout(s2xh4_col);

  Layout l = make_layout(make_shape((24)));
  print1D(l);
  // print2D(l);
  // print_layout(l);


  // std::cout << "=== CuTe Layout Examples ===" << std::endl << std::endl;

  // // Basic Layout construction
  // std::cout << "1. Basic Layouts:" << std::endl;
  // Layout s8 = make_layout(Int<8>{});
  // Layout d8 = make_layout(8);

  // std::cout << "s8 (static 8):  "; print(s8); std::cout << std::endl;
  // std::cout << "s8 (static 8):  "; print_layout(s8); std::cout << std::endl;
  // std::cout << "d8 (dynamic 8): "; print(d8); std::cout << std::endl;
  // std::cout << "d8 (dynamic 8): "; print_layout(d8); std::cout << std::endl;
  // std::cout << std::endl;

  // // 2D Layouts with different types
  // std::cout << "2. 2D Layouts:" << std::endl;
  // Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
  // Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

  // std::cout << "s2xs4 (2x4 static):  "; print(s2xs4); std::cout << std::endl;
  // std::cout << "s2xd4 (2x4 mixed):   "; print(s2xd4); std::cout << std::endl;
  // std::cout << std::endl;

  // // Layouts with custom strides
  // std::cout << "3. Layouts with Custom Strides:" << std::endl;
  // Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
  //                              make_stride(Int<12>{},Int<1>{}));
  // Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
  //                               LayoutLeft{});
  // Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
  //                               LayoutRight{});

  // std::cout << "s2xd4_a (custom stride): "; print(s2xd4_a); std::cout << std::endl;
  // std::cout << "s2xd4_col (column-major): "; print(s2xd4_col); std::cout << std::endl;
  // std::cout << "s2xd4_row (row-major): "; print(s2xd4_row); std::cout << std::endl;
  // std::cout << std::endl;

  // // Hierarchical layouts
  // std::cout << "4. Hierarchical Layouts:" << std::endl;
  // Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
  //                           make_stride(4,make_stride(2,1)));
  // Layout s2xh4_col = make_layout(shape(s2xh4),
  //                               LayoutLeft{});

  // std::cout << "s2xh4 (hierarchical): "; print(s2xh4); std::cout << std::endl;
  // std::cout << "s2xh4_col (hier col-major): "; print(s2xh4_col); std::cout << std::endl;
  // std::cout << std::endl;

  // // Layout properties
  // std::cout << "5. Layout Properties:" << std::endl;
  // std::cout << "rank(s2xs4) = " << rank(s2xs4) << std::endl;
  // std::cout << "depth(s2xs4) = " << depth(s2xs4) << std::endl;
  // std::cout << "size(s2xs4) = " << size(s2xs4) << std::endl;
  // std::cout << "cosize(s2xs4) = " << cosize(s2xs4) << std::endl;
  // std::cout << "size<0>(s2xs4) = " << size<0>(s2xs4) << std::endl;
  // std::cout << "size<1>(s2xs4) = " << size<1>(s2xs4) << std::endl;
  // std::cout << std::endl;

  // std::cout << "rank(s2xh4) = " << rank(s2xh4) << std::endl;
  // std::cout << "depth(s2xh4) = " << depth(s2xh4) << std::endl;
  // std::cout << "size(s2xh4) = " << size(s2xh4) << std::endl;
  // std::cout << std::endl;

  // // 2D printing of layouts
  // std::cout << "6. 2D Layout Visualization:" << std::endl;

  // std::cout << "s2xs4 (column-major):" << std::endl;
  // print2D(s2xs4);
  // std::cout << std::endl;

  // std::cout << "s2xd4_a (custom stride):" << std::endl;
  // print2D(s2xd4_a);
  // std::cout << std::endl;

  // std::cout << "s2xd4_row (row-major):" << std::endl;
  // print2D(s2xd4_row);
  // std::cout << std::endl;

  // std::cout << "s2xh4_col (hierarchical column-major):" << std::endl;
  // print2D(s2xh4_col);
  // std::cout << std::endl;

  // std::cout << "s2xh4 (hierarchical mixed):" << std::endl;
  // print2D(s2xh4);
  // std::cout << std::endl;

  // // 1D printing of layouts
  // std::cout << "7. 1D Layout Visualization (flattened):" << std::endl;

  // std::cout << "s2xs4: ";
  // print1D(s2xs4);

  // std::cout << "s2xd4_a: ";
  // print1D(s2xd4_a);

  // std::cout << "s2xh4_col: ";
  // print1D(s2xh4_col);

  // std::cout << "s2xh4: ";
  // print1D(s2xh4);
  // std::cout << std::endl;

  // // Using print_layout for formatted output
  // std::cout << "8. Formatted Layout Printing:" << std::endl;
  // print_layout(s2xh4);
  // std::cout << std::endl;

  // // Demonstrate layout mapping
  // std::cout << "9. Layout Mapping Examples:" << std::endl;
  // std::cout << "For layout s2xs4 = " ; print(s2xs4); std::cout << std::endl;
  // std::cout << "  s2xs4(0,0) = " << s2xs4(0,0) << std::endl;
  // std::cout << "  s2xs4(1,0) = " << s2xs4(1,0) << std::endl;
  // std::cout << "  s2xs4(0,1) = " << s2xs4(0,1) << std::endl;
  // std::cout << "  s2xs4(1,3) = " << s2xs4(1,3) << std::endl;
  // std::cout << std::endl;

  // std::cout << "For layout s2xd4_row = " ; print(s2xd4_row); std::cout << std::endl;
  // std::cout << "  s2xd4_row(0,0) = " << s2xd4_row(0,0) << std::endl;
  // std::cout << "  s2xd4_row(0,1) = " << s2xd4_row(0,1) << std::endl;
  // std::cout << "  s2xd4_row(1,0) = " << s2xd4_row(1,0) << std::endl;
  // std::cout << "  s2xd4_row(1,3) = " << s2xd4_row(1,3) << std::endl;
  // std::cout << std::endl;

  // // Verify congruence
  // std::cout << "10. Shape and Stride Congruence:" << std::endl;
  // auto my_shape = make_shape(2, make_shape(2,2));
  // auto my_stride = make_stride(4, make_stride(2,1));
  // static_assert(congruent(my_shape, my_stride), "Shape and Stride must be congruent!");
  // std::cout << "Shape and Stride are congruent for hierarchical layout" << std::endl;

  return 0;
}