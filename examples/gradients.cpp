#include "autodiff/var.hpp"

using autodiff::base::expression;
using autodiff::base::var;

using point = std::array<double, 3>;

int main() {
    var x(10, 'x');
    var y(12, 'y');
    var z(3, 'z');

    auto w = x * x * x;
    auto W = expression(w);
    W.grad();
    W.print_grad();
}
