#include "autodiff/autodiff/rad.hpp"

using namespace autodiff;

using point = std::array<double, 3>;

int main() {
    auto e = rad::expression("x*x*y*z + z*z");
    auto f = rad::expression("exp(x) + y");
    auto g = rad::expression("x");
    auto h = rad::expression("sin(x) + ln(x)");
    auto i = rad::expression("sin(x*ln(x))");
    state s = {{"x", 1}, {"y", 1}, {"z", 1}};
}
