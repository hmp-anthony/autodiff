#include "var.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;
TEST(basic, addition) {
    var a(1);
    var b(10);

    auto x_ = a + b;
    auto a_ = x_.left();
    auto b_ = x_.right();

    ASSERT_EQ(x_.value(), 11);
    ASSERT_EQ(a_->value(), 1);
    ASSERT_EQ(b_->value(), 10);
}

TEST(functions, exp) {
    // testing functions
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);
    var y_3 = y_1 + y_2;

    auto z_1 = exp_(y_1 + y_2);

    auto z_2 = exp_(y_3);
    ASSERT_NEAR(z_1.value(), 59874.1, 0.1);
    ASSERT_NEAR(z_2.value(), 59874.1, 0.1);
}

TEST(functions, exp_2) {
    // testing functions
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);
    var y_3(10);
    var y_4(2);

    auto z_1 = exp_(y_1 + y_2) + y_3 * y_4;
    ASSERT_NEAR(z_1.value(), 59894.1, 0.1);
    ASSERT_EQ(z_1.to_string(), "+");
}

TEST(basic, computation_graph_1) {
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);

    auto z = y_1 + y_2;

    ASSERT_TRUE(z.to_string() == "+");
    ASSERT_TRUE(z.left()->value() == 10);
    ASSERT_TRUE(z.right()->value() == 1);

    auto f = exp_(y_1 + y_2);
    auto g = f.left();

    ASSERT_TRUE(g->to_string() == "+");
    ASSERT_TRUE(g->left()->value() == 10);
    ASSERT_TRUE(g->right()->value() == 1);
}

TEST(basic, computation_graph_2) {
    var x(10);
    var y(12);

    auto z = x * x + y * y;
    ASSERT_EQ(z.to_string(), "+");

    auto zl = z.left();
    auto zr = z.right();
    ASSERT_EQ(zl->to_string(), "*");
    ASSERT_EQ(zr->to_string(), "*");

    auto zll = zl->left();
    auto zlr = zl->right();
    ASSERT_EQ(stod(zll->to_string()), 10.0);
    ASSERT_EQ(stod(zlr->to_string()), 10.0);

    auto zrl = zr->left();
    auto zrr = zr->right();
    ASSERT_EQ(stod(zrl->to_string()), 12.0);
    ASSERT_EQ(stod(zrr->to_string()), 12.0);
}

TEST(basic, change_values) {
    var x(10);
    var y(12);

    auto z = x * x + y * y;
    ASSERT_EQ(z.eval(), 244);

    set_value(&x, 1);
    set_value(&y, 2);
    ASSERT_EQ(z.eval(), 5);

    set_value(&x, 5);
    set_value(&y, 5);
    ASSERT_EQ(z.eval(), 50);
}
