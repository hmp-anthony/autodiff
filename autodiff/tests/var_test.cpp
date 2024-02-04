#include "var.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

using namespace autodiff;
using namespace base;

int basic_addition() {
    var a(1);
    var b(10);

    auto x_ = a + b;
    auto a_ = x_.left();
    auto b_ = x_.right();

    if((x_.value() == 11) && (a_->value() == 1) && (b_->value() == 10)) {
        return 0;
    }
    return 1;
}

int functions_exp() {
    // testing functions
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);
    var y_3 = y_1 + y_2;

    auto z_1 = exp_(y_1 + y_2);
    auto z_2 = exp_(y_3);

    if(std::abs(z_1.value() - 59874.1) < 0.1
    && std::abs(z_2.value() - 59874.1) < 0.1) {
        return 0;
    }
    return 1;
}

int functions_exp2() {
    // testing functions
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);
    var y_3(10);
    var y_4(2);

    auto z_1 = exp_(y_1 + y_2) + y_3 * y_4;

    if(std::abs(z_1.value() - 59894.1) < 0.1
    && z_1.to_string() == "+") {
        return 0;
    }
    return 1;
}

int computation_graph() {
    auto exp_ = autodiff::functions::exp();

    var y_1(10);
    var y_2(1);

    auto z = y_1 + y_2;
    auto f = exp_(y_1 + y_2);
    auto g = f.left();

    if(z.to_string() == "+" && z.left()->value() == 10
    && z.right()->value() == 1 && g->to_string() == "+"
    && g->left()->value() == 10 && g->right()->value() == 1) {
        return 0;
    }
    return 1;
}

int computation_graph2() {
    var x(10);
    var y(12);

    auto z = x * x + y * y;
    bool b1 = (z.to_string() == "+");

    auto zl = z.left();
    auto zr = z.right();
    bool b2 = (zl->to_string() == "*");
    bool b3 = (zr->to_string() == "*");

    auto zll = zl->left();
    auto zlr = zl->right();
    bool b4 = (stod(zll->to_string()) == 10.0);
    bool b5 = (stod(zlr->to_string()) == 10.0);

    auto zrl = zr->left();
    auto zrr = zr->right();
    bool b6 = (stod(zrl->to_string()) == 12.0);
    bool b7 = (stod(zrr->to_string()) == 12.0);

    if(b1 && b2 && b3 && b4 && b5 && b6 && b7){
        return 0;
    }
    return 1;
}

int main() {
    if(basic_addition() == 1) { 
        std::cout << "basic_addition() failed" << std::endl;
        return 1;
    }
    
    if(functions_exp() == 1) {
        std::cout << "functions_exp() failed" << std::endl;
        return 1;
    }

    if(functions_exp2() == 1) {
        std::cout << "functions_exp2() failed" << std::endl;
        return 1;
    }

    if(computation_graph() == 1) {
        std::cout << "computation_graph() failed" << std::endl;
        return 1;
    }

    if(computation_graph2() == 1) {
        std::cout << "computation_graph2() failed" << std::endl;
        return 1;
    }
    return 0;
}
