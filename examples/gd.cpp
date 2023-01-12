#include <iostream>

#include "gradient.hpp"

using autodiff::base::var;

int main() {
    var a(10);
    var b(20);

    auto c = a * a + b + a * a;
    auto C = autodiff::base::gradient(c);
    std::cout << C[a] << std::endl;
    std::cout << C[b] << std::endl;
    return 0;
}
