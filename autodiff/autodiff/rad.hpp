#pragma once

#include <cmath>

#include "io/var.hpp"

namespace autodiff {

namespace rad {
class var : public autodiff::base::var {
public:
    explicit var(token t) : autodiff::base::var(t), grad_(0), visit_count_(0) {}

    void reset() {
        reset_value();
        grad_ = parents().empty() ? 1 : 0;
        visit_count_ = 0;
        if (left()) std::static_pointer_cast<var>(left())->reset();
        if (right()) std::static_pointer_cast<var>(right())->reset();
    }

    void grad(const state& s) {
        // Do not propagate untill all parents have
        // touched their child. I do not advocate this.
        if (++visit_count_ < parents().size() && !parents().empty()) {
            return;
        }

        // carry out grad
        if (type() == token::token_type::binary_operation) {
            std::string str = to_string();
            if (str == "*") {
                multiplication(s);
            } else if (str == "+") {
                addition(s);
            } else if (str == "/") {
                division(s);
            } else if (str == "-") {
                subtraction(s);
            }
            std::static_pointer_cast<var>(left())->grad(s);
            std::static_pointer_cast<var>(right())->grad(s);
            return;
        }
        // if(type() == unary_operation) ...
        // return
    }

    void multiplication(const state& s) {
        std::static_pointer_cast<var>(left())->grad_ += grad_ * (*right())[s];
        std::static_pointer_cast<var>(right())->grad_ += grad_ * (*left())[s];
    }

    void addition(const state& s) {
        std::static_pointer_cast<var>(left())->grad_ += grad_;
        std::static_pointer_cast<var>(right())->grad_ += grad_;
    }

    void division(const state& s) {
        double r = (*right())[s];
        double l = (*left())[s];
        std::static_pointer_cast<var>(right())->grad_ += grad_ * (1.0 / l);
        std::static_pointer_cast<var>(left())->grad_ -= grad_ * (r / (l * l));
    }

    void subtraction(const state& s) {
        std::static_pointer_cast<var>(left())->grad_ -= grad_;
        std::static_pointer_cast<var>(right())->grad_ += grad_;
    }

    double grad_;
    int visit_count_;
};
}  // namespace rad

}  // namespace autodiff
