#include "autodiff/var.hpp"

using autodiff::base::var;


using point = std::array<double, 3>;

int main() {
    var x(10);
    var y(12);

    auto z = x * x + y * y;
    std::cout << z.eval() << std::endl;
    z.print();

}
