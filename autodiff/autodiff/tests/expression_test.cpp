#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "autodiff/var.hpp"
#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, addition) {
    var a(1, 'a');
    var b(10, 'b');

    auto x = a * a * a + b;
    auto X = expression(x);
    X.grad();
    X.print_grad();

    auto vars = X.variables();
}
