#pragma once
#include <cmath>

#include "io/var.hpp"

namespace autodiff {
namespace function {

using autodiff::base::var;

struct exp {
    exp() {}
    var operator()(var& e) {
        var result(std::exp(e.value()));
        result.set_left(std::make_shared<var>(e));
        return result;
    }
    var operator()(var&& e) {
        var result(std::exp(e.value()));
        result.set_left(std::make_shared<var>(e));
        return result;
    }
};
}  // namespace function
}  // namespace autodiff
