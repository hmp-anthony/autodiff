#include <iostream>

#include "gradient.hpp"
#include "var.hpp"

using autodiff::base::var;
using autodiff::functions::cos;

int main() {
    double a_ = 1;
    double b_ = 1;

    double gamma_ = 0.1;

    auto cos_ = autodiff::functions::cos();

    for(int i = 0; i < 100; ++i){
        var a(a_);
        var b(b_);
        // define loss function.
        auto c = a * a + b * b + cos_(a * a);
        // obtain gradient of loff function.
        auto C = autodiff::base::gradient(c);
        // take a step
        a_ = a_ - gamma_*C[a];
        b_ = b_ - gamma_*C[b];
        std::cout << a_ << " " << b_ << std::endl;
    }
    return 0;
}
