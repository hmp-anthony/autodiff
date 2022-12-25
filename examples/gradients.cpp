#include "autodiff/autodiff/rad.hpp"

using namespace autodiff;

using point = std::array<double, 3>;

int main() {
    auto exp = rad::expression("x*x*x");
    exp = rad::expression("exp(x) + y");
    exp = rad::expression("exp(x + y)");
    exp = rad::expression("sin(x+y)*z");
    exp = rad::expression("sin(x+y)");
    exp = rad::expression("sin(x) + y");
    exp = rad::expression("1/(1+exp(-x))");
}
