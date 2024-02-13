#include "eqn.hpp"

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, eqn) {
    var x(10);
    var y(11);

    std::cout << &x << "is the address of x" << std::endl;
    std::cout << &y << "is the address of y" << std::endl;

    auto e1 = eqn(x * x + y * y);
    auto e2 = eqn(x * (x + y));
}
