#include "gradient.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

/*
TEST(no_brackets, simple_binary_ops) {
    var a(10, 'a');
    auto x = -a;
    auto X = gradient(x);
    ASSERT_EQ(X['a'], -1);

    var c(1, 'c');
    var d(10, 'd');
    auto y = c / d;
    auto Y = gradient(y);
    ASSERT_EQ(Y['c'], 0.1);
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

TEST(functions, exp) {
    var a(2, 'a');
    auto exp_ = autodiff::functions::exp();
    auto c = exp_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C['a'], 218.39, 0.1);

    var b(3, 'b');
    auto d = exp_(a * b);
    auto D = gradient(d);
    ASSERT_NEAR(D['a'], 1210.29, 0.1);
    ASSERT_NEAR(D['b'], 806.85, 0.1);

    auto e = exp_(a * b) + exp_(a * b);
    auto E = gradient(e);
    ASSERT_NEAR(E['a'], 2420.57, 0.1);
    ASSERT_NEAR(E['b'], 1613.71, 0.1);
}

TEST(functions, exp_complex) {
    var x(2, 'x');
    var y(1, 'y');
    auto exp_ = autodiff::functions::exp();
    auto z = y / (y + exp_(-x));
    auto Z = gradient(z);
    ASSERT_NEAR(Z['x'], 0.104994, 0.0001);
}

TEST(functions, sin) {
    var a(2, 'a');
    auto sin_ = autodiff::functions::sin();
    auto c = sin_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C['a'], -2.61457, 0.001);
}

TEST(functions, cos) {
    var a(2, 'a');
    auto cos_ = autodiff::functions::cos();
    auto c = cos_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C['a'], 3.0272, 0.001);

    var b(1, 'b');
    auto d = cos_(b);
    auto D = gradient(d);
    ASSERT_NEAR(D['b'], -0.8414709, 0.001);
}

TEST(functions, ln) {
    var a(2, 'a');
    auto ln_ = autodiff::functions::ln();
    auto c = ln_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C['a'], 1.0, 0.001);
}

TEST(functions, log) {
    var a(2, 'a');
    auto log_ = autodiff::functions::log();
    auto c = log_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C['a'], 1.44269504, 0.001);
}
*/
