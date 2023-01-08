#include "gradient.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(no_brackets, simple_binary_ops) {
    var c(1, 'c');
    var d(10, 'd');
    auto y = c / d;
    auto Y = gradient(y);
    ASSERT_EQ(Y['c'], 0.1);
    ASSERT_EQ(Y['d'], -0.01);
}

TEST(brackets, simple_binary_ops) {
    var c(1, 'c');
    var d(1, 'd');
    auto y = (c + d) * d;
    auto Y = gradient(y);
    ASSERT_EQ(Y['c'], 1);
    ASSERT_EQ(Y['d'], 3);
}

TEST(no_brackets, complex_binary_ops) {
    var a(10, 'a');
    var b(10, 'b');
    auto x = a * a * a * a + b;
    auto X = gradient(x);
    ASSERT_EQ(X['a'], 4000);
    ASSERT_EQ(X['b'], 1);
}

TEST(brackets, complex_binary_ops) {
    var a(10, 'a');
    var b(10, 'b');
    auto z = (a + b) * (a + b);
    auto Z = gradient(z);
    ASSERT_EQ(Z['a'], 40);
    ASSERT_EQ(Z['b'], 40);

    var c(5, 'c');
    auto w = (a + b) * (a + b) + c * a * b;
    auto W = gradient(w);
    ASSERT_EQ(W['a'], 90);
    ASSERT_EQ(W['b'], 90);
    ASSERT_EQ(W['c'], 100);
}

