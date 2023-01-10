#include <iostream>

#include "gradient.hpp"

using autodiff::base::var;

int main() {
    var a(10, 'a');
    var b(20, 'b');

    auto c = a + b * a;
    auto C = autodiff::base::gradient(c);
    return 0;
}
