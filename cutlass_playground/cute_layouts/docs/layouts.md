Layout Creation and Use
A Layout is a pair of IntTuples: the Shape and the Stride. The first element defines the abstract shape of the Layout, and the second element defines the strides, which map from coordinates within the shape to the index space.

We define many operations on Layouts analogous to those defined on IntTuple.

rank(Layout): The number of modes in a Layout. Equivalent to the tuple size of the Layout’s shape.

get<I>(Layout): The Ith sub-layout of the Layout, with I < rank.

depth(Layout): The depth of the Layout’s shape. A single integer has depth 0, a tuple of integers has depth 1, a tuple of tuples of integers has depth 2, etc.

shape(Layout): The shape of the Layout.

stride(Layout): The stride of the Layout.

size(Layout): The size of the Layout function’s domain. Equivalent to size(shape(Layout)).

cosize(Layout): The size of the Layout function’s codomain (not necessarily the range). Equivalent to A(size(A) - 1) + 1.

Hierarchical access functions
IntTuples and Layouts can be arbitrarily nested. For convenience, we define versions of some of the above functions that take a sequence of integers, instead of just one integer. This makes it possible to access elements inside of nested IntTuple or Layout more easily. For example, we permit get<I...>(x), where I... is a “C++ parameter pack” that denotes zero or more (integer) template arguments. These hierarchical access functions include the following.

get<I0,I1,...,IN>(x) := get<IN>(...(get<I1>(get<I0>(x)))...). Extract the INth of the … of the I1st of the I0th element of x.

rank<I...>(x)  := rank(get<I...>(x)). The rank of the I...th element of x.

depth<I...>(x) := depth(get<I...>(x)). The depth of the I...th element of x.

shape<I...>(x)  := shape(get<I...>(x)). The shape of the I...th element of x.

size<I...>(x)  := size(get<I...>(x)). The size of the I...th element of x.

In the following examples, you’ll see use of size<0> and size<1> to determine loops bounds for the 0th and 1st mode of a layout or tensor.

Constructing a Layout
A Layout can be constructed in many different ways. It can include any combination of compile-time (static) integers or run-time (dynamic) integers.

Layout s8 = make_layout(Int<8>{});
Layout d8 = make_layout(8);

Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                             make_stride(Int<12>{},Int<1>{}));
Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                               LayoutLeft{});
Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                               LayoutRight{});

Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                           make_stride(4,make_stride(2,1)));
Layout s2xh4_col = make_layout(shape(s2xh4),
                               LayoutLeft{});
The make_layout function returns a Layout. It deduces the types of the function’s arguments and returns a Layout with the appropriate template arguments. Similarly, the make_shape and make_stride functions return a Shape resp. Stride. CuTe often uses these make_* functions due to restrictions around constructor template argument deduction (CTAD) and to avoid having to repeat static or dynamic integer types.

When the Stride argument is omitted, it is generated from the provided Shape with LayoutLeft as default. The LayoutLeft tag constructs strides as an exclusive prefix product of the Shape from left to right, without regard to the Shape’s hierarchy. This can be considered a “generalized column-major stride generation”. The LayoutRight tag constructs strides as an exclusive prefix product of the Shape from right to left, without regard to the Shape’s hierarchy. For shapes of depth one, this can be considered a “row-major stride generation”, but for hierarchical shapes the resulting strides may be surprising. For example, the strides of s2xh4 above could be generated with LayoutRight.

Calling print on each layout above results in the following

s8        :  _8:_1
d8        :  8:_1
s2xs4     :  (_2,_4):(_1,_2)
s2xd4     :  (_2,4):(_1,_2)
s2xd4_a   :  (_2,4):(_12,_1)
s2xd4_col :  (_2,4):(_1,_2)
s2xd4_row :  (_2,4):(4,_1)
s2xh4     :  (2,(2,2)):(4,(2,1))
s2xh4_col :  (2,(2,2)):(_1,(2,4))
The Shape:Stride notation is used quite often for Layout. The _N notation is shorthand for a static integer while other integers are dynamic integers. Observe that both Shape and Stride may be composed of both static and dynamic integers.

Also note that the Shape and Stride are assumed to be congruent. That is, Shape and Stride have the same tuple profiles. For every integer in Shape, there is a corresponding integer in Stride. This can be asserted with

static_assert(congruent(my_shape, my_stride));
Using a Layout
The fundamental use of a Layout is to map between coordinate space(s) defined by the Shape and an index space defined by the Stride. For example, to print an arbitrary rank-2 layout in a 2-D table, we can write the function

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
which produces the following output for the above examples.

> print2D(s2xs4)
  0    2    4    6
  1    3    5    7
> print2D(s2xd4_a)
  0    1    2    3
 12   13   14   15
> print2D(s2xh4_col)
  0    2    4    6
  1    3    5    7
> print2D(s2xh4)
  0    2    1    3
  4    6    5    7
We can see static, dynamic, row-major, column-major, and hierarchical layouts printed here. The statement layout(m,n) provides the mapping of the logical 2-D coordinate (m,n) to the 1-D index.

Interestingly, the s2xh4 example isn’t row-major or column-major. Furthermore, it has three modes but is still interpreted as rank-2 and we’re using a 2-D coordinate. Specifically, s2xh4 has a 2-D multi-mode in the second mode, but we’re still able to use a 1-D coordinate for that mode. More on this in the next section, but first we can generalize this another step. Let’s use a 1-D coordinate and treat all of the modes of each layout as a single multi-mode. For instance, the following print1D function

template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
produces the following output for the above examples.

> print1D(s2xs4)
  0    1    2    3    4    5    6    7
> print1D(s2xd4_a)
  0   12    1   13    2   14    3   15
> print1D(s2xh4_col)
  0    1    2    3    4    5    6    7
> print1D(s2xh4)
  0    4    2    6    1    5    3    7
Any multi-mode of a layout, including the entire layout itself, can accept a 1-D coordinate. More on this in the following sections.

CuTe provides more printing utilities for visualizing Layouts. The print_layout function produces a formatted 2-D table of the Layout’s mapping.

> print_layout(s2xh4)
(2,(2,2)):(4,(2,1))
      0   1   2   3
    +---+---+---+---+
 0  | 0 | 2 | 1 | 3 |
    +---+---+---+---+
 1  | 4 | 6 | 5 | 7 |
    +---+---+---+---+
The print_latex function generates LaTeX that can be compiled with pdflatex into a color-coded vector graphics image of the same 2-D table.