#include "autodiff/var.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "autodiff/functions.hpp"
#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;
/*
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
    auto exp_ = autodiff::function::exp();

    var y_1(10);
    var y_2(1);
    var y_3 = y_1 + y_2;

    auto z_1 = exp_(y_1 + y_2);

    auto z_2 = exp_(y_3);
    ASSERT_NEAR(z_1.value(), 59874.1, 0.1);
    ASSERT_NEAR(z_2.value(), 59874.1, 0.1);
}
TEST(basic, computation_graph_1) {
    auto exp_ = autodiff::function::exp();

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
    std::cout << z.to_string() << std::endl;

    auto zl = z.left();
    auto zr = z.right();
    std::cout << zl->to_string() << std::endl;
    std::cout << zr->to_string() << std::endl;

    auto zll = zl->left();
    auto zlr = zl->right();
    std::cout << zll->to_string() << std::endl;
    std::cout << zlr->to_string() << std::endl;

    auto zrl = zr->left();
    auto zrr = zr->right();
    std::cout << zrl->to_string() << std::endl;
    std::cout << zrr->to_string() << std::endl;
}*/
