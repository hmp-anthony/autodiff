#include "autodiff/autodiff/rad.hpp"

using namespace autodiff;

using point = std::array<double, 3>;

int main() {
    auto exp1 = rad::expression("exp(x) + y");
    auto exp2 = rad::expression("sin(x) + ln(x)");
    auto exp3 = rad::expression("sin(x*ln(x))");
    auto exp4 = rad::expression("1/(1+exp(-x))");
}
