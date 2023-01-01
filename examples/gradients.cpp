#include "autodiff/var.hpp"

using autodiff::base::expression;
using autodiff::base::var;

using point = std::array<double, 3>;

int main() {
    var x(10);
    var y(12);

    auto z = x * x * x + y * y * y;
    auto Z = expression(z);
    Z.grad();
}
