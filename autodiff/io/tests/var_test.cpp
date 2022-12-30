#include "io/var.hpp"
#include "autodiff/functions.hpp"

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
    auto a_ = x_.get_left();
    auto b_ = x_.get_right();
/*
    ASSERT_EQ(x_.value(), 11);
    ASSERT_EQ(a_->value(), 1);
  ASSERT_EQ(b_->value(), 10);*/
    std::cout << x_.value() << std::endl;
    std::cout << a_->value() << std::endl;
    std::cout << b_->value() << std::endl;
}
/*
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

TEST(basic, computation_graph) {
    auto exp_ = autodiff::function::exp();

    var y_1(10);
    var y_2(1);

    auto z = exp_(y_1 + y_2);

    auto a = z.get_left();
    ASSERT_TRUE(a->is_binary_operation());
    ASSERT_TRUE(a->to_string() == "+");
    ASSERT_TRUE(a->get_left()->value() == 10);
    ASSERT_TRUE(a->get_right()->value() == 1);
}*/
