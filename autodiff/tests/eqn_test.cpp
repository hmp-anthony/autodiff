#include "eqn.hpp"

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, eqn) {
    var x(10);
    var y(11);

    std::cout << &x << "is the address of x" << std::endl;
    std::cout << &y << "is the address of y" << std::endl;

    std::cout << "-------------" << std::endl;

    auto e = x * x + y * y;
    eqn e1(&e);
    e1.print_aliases();
    std::cout << "value " << e1.eval() << std::endl;
    set_value(&x, 1);
    set_value(&y, 2);
    std::cout << "value " << e1.eval() << std::endl;

    auto f = x * (x + y);
    eqn e2(&f);
    e2.print_aliases();
}
