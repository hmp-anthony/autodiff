#include "gtest/gtest.h"
#include "io/var.hpp"

using autodiff::var;

TEST(basic, addition) {
    var a(1);
    var b(10);

    auto x_ = a + b;
    auto a_ = x_.get_left();
    auto b_ = x_.get_right();

    std::cout << x_ << std::endl;
    std::cout << *b_ << std::endl;
    std::cout << *a_ << std::endl;

    // testing functions
    auto exp_ = autodiff::function::exp();

    var y_1(10);
    var y_2(1);
    var y_3 = y_1 + y_2;

    auto z_1 = exp_(y_1 + y_2);
    auto z_2 = exp_(y_3);
    // Obtains a pointer to var of which the value is 11
    auto w_1 = z_1.get_left();

    // attempt to get children of w;
    auto k_1 = w_1->get_left();
    auto l_1 = w_1->get_right();

    std::cout << *w_1 << std::endl;
    std::cout << *k_1 << std::endl;
    std::cout << *l_1 << std::endl;

    // Obtains a pointer to var of which the value is 11
    auto w_2 = z_2.get_left();

    // attempt to get children of w;
    auto k_2 = w_2->get_left();
    auto l_2 = w_2->get_right();

    std::cout << *w_2 << std::endl;
    std::cout << *k_2 << std::endl;
    std::cout << *l_2 << std::endl;
}
