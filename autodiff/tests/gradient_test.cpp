#include "gradient.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, addition) {
    var a(10, 'a');
    var b(10, 'b');

    auto x = a * a * a * a + b;
    auto X = gradient(x);
    ASSERT_EQ(X['a'], 4000);
    ASSERT_EQ(X['b'], 1);

    var c(1, 'c');
    var d(10, 'd');

    auto y = c / d;
    auto Y = gradient(y);
    ASSERT_EQ(Y['c'], 0.1);
    ASSERT_EQ(Y['d'], -0.01);
}
