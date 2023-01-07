#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "var.hpp"
#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, addition) {
    var a(10, 'a');
    var b(10, 'b');

    auto x = a * a * a * a + b;
    auto X = expression(x);
    X.grad();
    X.print_grad();

    var c(1, 'c');
    var d(10, 'd');
    
    auto y = c / d;
    auto Y = expression(y);
    Y.grad();
    Y.print_grad();


    auto vars = X.variables();
    for (const auto& v : vars) {
        std::cout << v->name() << v->parents().size() << std::endl;
    }
}
