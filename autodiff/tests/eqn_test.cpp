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

    auto e1 = eqn(x * x + y * y);
    e1.print_aliases();
    std::cout << "value " << e1.value() << std::endl;
    x.set_value(1);
    y.set_value(2);
    std::cout << "value " << e1.value() << std::endl;
    /*
    auto e2 = eqn(x * (x + y));
    std::cout << e2.value() << std::endl;
    e2.print_aliases();
    */
}
