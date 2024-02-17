#include "gradient.hpp"
#include "gtest/gtest.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

using namespace autodiff;
using namespace base;

/*
TEST(no_brackets, simple_binary_ops) {
    var a(10);
    auto x = -a;
    auto X = gradient(x);
    ASSERT_EQ(X[a], -1);

    var c(1);
    var d(10);
    auto y = c / d;
    auto Y = gradient(y);
    ASSERT_EQ(Y[c], 0.1);
}

TEST(brackets, simple_binary_ops) {
    var c(2);
    var d(5);
    auto y = (c + d) * d;
    auto Y = gradient(y);
    ASSERT_EQ(Y[c], 5);
    ASSERT_EQ(Y[d], 12);
}

TEST(no_brackets, complex_binary_ops) {
    var a(10);
    var b(10);
    auto x = a * a * a * a + b;
    auto X = gradient(x);
    ASSERT_EQ(X[a], 4000);
    ASSERT_EQ(X[b], 1);
}

TEST(brackets, complex_binary_ops) {
    var a(10);
    var b(10);
    auto z = (a + b) * (a + b);
    auto Z = gradient(z);
    ASSERT_EQ(Z[a], 40);
    ASSERT_EQ(Z[b], 40);

    var c(5);
    auto w = (a + b) * (a + b) + c * a * b;
    auto W = gradient(w);
    ASSERT_EQ(W[a], 90);
    ASSERT_EQ(W[b], 90);
    ASSERT_EQ(W[c], 100);
}

TEST(functions, exp) {
    var a(2);
    auto exp_ = autodiff::functions::exp();
    auto c = exp_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], 218.39, 0.1);

    var b(3);
    auto d = exp_(a * b);
    auto D = gradient(d);
    ASSERT_NEAR(D[a], 1210.29, 0.1);
    ASSERT_NEAR(D[b], 806.85, 0.1);

    auto e = exp_(a * b) + exp_(a * b);
    auto E = gradient(e);
    ASSERT_NEAR(E[a], 2420.57, 0.1);
    ASSERT_NEAR(E[b], 1613.71, 0.1);
}

TEST(functions, exp_complex) {
    var x(2);
    var y(1);
    auto exp_ = autodiff::functions::exp();
    auto z = y / (y + exp_(-x));
    auto Z = gradient(z);
    ASSERT_NEAR(Z[x], 0.104994, 0.0001);
}

TEST(functions, sin) {
    var a(2);
    auto sin_ = autodiff::functions::sin();
    auto c = sin_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], -2.61457, 0.001);
}

TEST(functions, cos) {
    var a(2);
    auto cos_ = autodiff::functions::cos();
    auto c = cos_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], 3.0272, 0.001);

    var b(1);
    auto d = cos_(b);
    auto D = gradient(d);
    ASSERT_NEAR(D[b], -0.8414709, 0.001);
}

TEST(functions, ln) {
    var a(2);
    auto ln_ = autodiff::functions::ln();
    auto c = ln_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], 1.0, 0.001);
}

TEST(functions, log) {
    var a(2);
    auto log_ = autodiff::functions::log();
    auto c = log_(a * a);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], 1.44269504, 0.001);
}

TEST(functions, pow) {
    var a(2);
    var b(2);
    auto pow_ = autodiff::functions::pow();
    auto c = pow_(a, b);
    auto C = gradient(c);
    ASSERT_NEAR(C[a], 4.0, 0.01);
    ASSERT_NEAR(C[b], 2.77259, 0.01);

    var x(3);
    var y(4);
    auto d = pow_(x, y);
    auto D = gradient(d);
    ASSERT_NEAR(D[x], 108.0, 0.01);
    ASSERT_NEAR(D[y], 88.9876, 0.01);
}

TEST(changing_values, addition_multiplication) {
    var a(9);
    var b(8);
    auto x = a * a + b * b;
    auto X = gradient(x);
    ASSERT_EQ(X[a], 18);
    ASSERT_EQ(X[b], 16);

    set_value(a, 1);
    set_value(b, 2);
    X = gradient(x);
    ASSERT_EQ(X[a], 2);
    ASSERT_EQ(X[b], 4);

    set_value(a, 3);
    set_value(b, 3);
    X = gradient(x);
    ASSERT_EQ(X[a], 6);
    ASSERT_EQ(X[b], 6);
}

TEST(changing_values, division) {
    var a(9);
    var b(8);
    auto x = a / b;
    auto X = gradient(x);
    ASSERT_EQ(X[a], 0.125);
    ASSERT_EQ(X[b], -0.140625);

    set_value(a, 1);
    set_value(b, 2);
    X = gradient(x);
    ASSERT_EQ(X[a], 0.5);
    ASSERT_EQ(X[b], -0.25);

    set_value(a, 3);
    set_value(b, 3);
    X = gradient(x);
    ASSERT_NEAR(X[a], 0.3333, 0.01);
    ASSERT_NEAR(X[b], -0.333, 0.01);
}

TEST(changing_values, subtraction) {
    {
        var a(9);
        var b(8);
        auto x = a * a - b * b;
        auto X = gradient(x);
        ASSERT_EQ(x.value(), 17);
        ASSERT_EQ(X[a], 18);
        ASSERT_EQ(X[b], -16);

        set_value(a, 1);
        set_value(b, 2);
        X = gradient(x);
        ASSERT_EQ(X[a], 2);
        ASSERT_EQ(X[b], -4);

        set_value(a, 3);
        set_value(b, 3);
        X = gradient(x);
        ASSERT_NEAR(X[a], 6, 0.01);
        ASSERT_NEAR(X[b], -6, 0.01);
    }
    {
        var a(9);
        var b(8);
        auto x = a - b;
        auto X = gradient(x);
        ASSERT_EQ(x.value(), 1);
        ASSERT_EQ(X[a], 1);
        ASSERT_EQ(X[b], -1);

        set_value(a, 1);
        set_value(b, 2);
        X = gradient(x);
        ASSERT_EQ(X[a], 1);
        ASSERT_EQ(X[b], -1);

        set_value(a, 3);
        set_value(b, 3);
        X = gradient(x);
        ASSERT_NEAR(X[a], 1, 0.01);
        ASSERT_NEAR(X[b], -1, 0.01);
    }
}

TEST(changing_values, exp) {
    var x(3);
    auto exp_ = autodiff::functions::exp();
    auto d = exp_(x);
    auto D = gradient(d);
    ASSERT_NEAR(D[x], 20.08, 0.01);
    set_value(x, 9);
    D = gradient(d);
    ASSERT_NEAR(D[x], 8103.08, 0.01);

    var y(2);
    auto f = exp_(x + y * y);
    auto F = gradient(f);
    ASSERT_NEAR(f.value(), 442413.392, 0.01);
    ASSERT_NEAR(F[x], 442413.392, 0.01);
    ASSERT_NEAR(F[y], 1769653.568, 0.01);
}

TEST(changing_values, sin) {
    var x(2);
    auto sin_ = autodiff::functions::sin();
    auto d = sin_(x);
    ASSERT_NEAR(d.value(), 0.909, 0.01);
    auto D = gradient(d);
    ASSERT_NEAR(D[x], -0.416, 0.01);

    var y(10.8);
    d = sin_(x + y);
    D = gradient(d);
    ASSERT_NEAR(d.value(), 0.231, 0.01);
    ASSERT_NEAR(D[x], 0.972, 0.01);
    ASSERT_NEAR(D[y], 0.972, 0.01);

    set_value(x, 9);
    set_value(y, 3);
    D = gradient(d);
    ASSERT_NEAR(d.value(), -0.536, 0.01);
    ASSERT_NEAR(D[x], 0.843, 0.01);
    ASSERT_NEAR(D[y], 0.843, 0.01);
}

*/
void test(std::array<var, 2>& v, var& eqn) {
    std::cout << "begin test" << std::endl;
    auto G = gradient(eqn);
    std::cout << G[v[0]] << std::endl;
    std::cout << G[v[1]] << std::endl;
}
TEST(array_of_var, 2D) {
    var x(3.4); 
    var y(2.2);
    auto eqn = x * y + x * x;
    std::cout << "calling const ===============" << std::endl;
    std::array<var, 2> vars = {x, y};
    std::cout << "calling const ===============" << std::endl;
    std::cout << vars[0].value() << std::endl;
    test(vars, eqn);
}

