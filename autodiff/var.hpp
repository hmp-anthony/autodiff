#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "token.hpp"
#include "var.hpp"

namespace autodiff {
namespace base {
class var : public std::enable_shared_from_this<var> {
public:
    var(const var& v)
        : t_(v.t_),
          grad_(0),
          left_(std::move(v.left_)),
          right_(std::move(v.right_)),
          v_(v.v_) {}
    explicit var(std::string s, double v)
        : t_(s), grad_(0), v_(v) {}
    explicit var(double v) : t_(v), grad_(0) {
        v_ = v;
    }
    explicit var(token t)
        : t_(std::move(t)), grad_(0) {};
    explicit var(var&& n)
        : t_(std::move(n.t_)),
          grad_(0),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)),
          v_(n.v_) {}

    static std::map<const var*,std::vector<std::shared_ptr<var>>> aliases;
    
    std::shared_ptr<var> getptr() {
        return shared_from_this();
    }


    friend void update_aliases(const var& l, const var& r, var& result) {
        if(!(result.left_->is_binary_operation())) {
            aliases[&l].push_back(result.left_);
        }
        if(!(result.right_->is_binary_operation())) {
            aliases[&r].push_back(result.right_);
        }
    }

    friend void set_value(var& v, double value) { 
        auto a = aliases[&v];
        (&v)->v_ = value;
        (&v)->set_gradient(0); 
        for(auto& e: a) {
            e->v_ = value;
            e->set_gradient(0); 
        }
    }

    friend var operator+(const var& l, const var& r) {
        var result(token(std::string("+")));
        result.v_ = l.v_.value() + r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        update_aliases(l, r, result);
        return result;
    }

    friend var operator-(const var& l, const var& r) {
        var result(token(std::string("-")));
        result.v_ = l.v_.value() - r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        update_aliases(l, r, result);
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        var result(token(std::string("*")));
        result.v_ = l.v_.value() * r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        update_aliases(l, r, result);
        return result;
    }

    friend var operator/(const var& l, const var& r) {
        var result(token(std::string("/")));
        result.v_ = l.v_.value() / r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        update_aliases(l, r, result);
        return result;
    }

    friend var operator-(const var& v) {
        var result(token(std::string("0-")));
        result.v_ = -1 * v.v_.value();
        result.left_ = std::make_shared<var>(v);
        if(!(result.left_->is_binary_operation())) {
            aliases[&v].push_back(result.left_);
        }
        return result;
    }

    var operator=(const var& v) {
        t_ = v.t_;
        v_ = v.v_;
        left_ = v.left_;
        right_ = v.right_;
        return *this;
    }

    bool is_binary_operation() { return t_.is_binary_operation(); }
    bool is_function() { return t_.is_function(); }
    bool is_variable() { return t_.is_variable(); }
    bool is_constant() { return t_.is_constant(); }

    void set_left(std::shared_ptr<var> lc) { left_ = lc; }
    void set_right(std::shared_ptr<var> rc) { right_ = rc; }

    std::shared_ptr<var>& left() { return left_; }
    std::shared_ptr<var>& right() { return right_; }

    void clean_grad() {
        grad_ = 0;
        if(left_) left_->clean_grad();
        if(right_) right_->clean_grad();
    }

    token& get_token() { return t_; }

    double value() { 
        if(t_.to_string() == "+") {
            return left_->value() + right_->value();
        } else if(t_.to_string() == "*") {
            return left_->value() * right_->value();
        } else if(t_.to_string() == "-") {
            return left_->value() - right_->value();
        } else if(t_.to_string() == "/") {
            return left_->value() / right_->value();
        } else if(t_.to_string() == "exp") {
            return std::exp(left_->value());
        } else if(t_.to_string() == "sin") {
            return std::sin(left_->value());
        } else if(t_.to_string() == "cos") {
            return std::cos(left_->value());
        } else if(t_.to_string() == "ln") {
            return std::log(left_->value());
        } else if(t_.to_string() == "log") {
            return std::log(left_->value()) / std::log(2);
        } else {
            return v_.value();
        }
    }

    double get_gradient() { return grad_; }
    void set_gradient(double grad) { grad_ = grad; }

    void grad() {
        if (is_binary_operation()) {
            auto str = to_string();
            if (str == "*") {
                multiplication();
            } else if (str == "+") {
                addition();
            } else if (str == "/") {
                division();
            } else if (str == "-") {
                subtraction();
            }
            left()->grad();
            right()->grad();
            return;
        }
        if (is_function()) {
            auto str = to_string();
            if (str == "exp") {
                exp();
            } else if (str == "sin") {
                sin();
            } else if (str == "cos") {
                cos();
            } else if (str == "ln") {
                ln();
            } else if (str == "log") {
                log();
            } else if (str == "0-") {
                neg();
            } else if (str == "pow") {
                pow();
            }
            left()->grad();
            return;
        }
    }

    void addition() {
        left_->grad_ += grad_;
        right_->grad_ += grad_;
    }

    void multiplication() {
        left_->grad_ += grad_ * right_->value();
        right_->grad_ += grad_ * left_->value();
    }

    void division() {
        double r = right()->value();
        double l = left()->value();
        (left())->grad_ += grad_ * (1.0 / r);
        (right())->grad_ -= grad_ * (l / (r * r));
    }

    void subtraction() {
        (left())->grad_ += grad_;
        (right())->grad_ -= grad_;
    }

    void exp() { left_->grad_ += grad_ * std::exp(left_->value()); }

    void sin() { left_->grad_ += std::cos(left_->value()); }

    void cos() { left_->grad_ -= std::sin(left_->value()); }

    void ln() { left_->grad_ += grad_ * (1 / left_->value()); }

    void log() { left_->grad_ += grad_ * (1 / (left_->value() * std::log(2))); }

    void neg() { left_->grad_ += (-1) * grad_; }

    void pow() {
        auto r = right()->value();
        auto l = left()->value();
        left()->grad_ += grad_ * r * std::pow(l, r -1);
        right()->grad_ += grad_ * value() * std::log(l);
    }

    const std::string& to_string() const { return t_.to_string(); }

    double grad_;

private:
    token t_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::optional<double> v_;
};

std::map<const var*,std::vector<std::shared_ptr<var>>> var::aliases;

}  // namespace base

namespace functions {

using autodiff::base::var;

struct pow {
    pow() {}
    var operator()(var& x, var& y) {
        var result("pow", std::pow(x.value(), y.value()));
        result.set_left(std::make_shared<var>(x));
        result.set_right(std::make_shared<var>(y));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&x].push_back(result.left());
        }
        if(!(result.right()->is_binary_operation())) {
            var::aliases[&y].push_back(result.right());
        }
        return result;
    }
};

struct exp {
    exp() {}
    var operator()(var& e) {
        var result("exp", std::exp(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
    var operator()(var&& e) {
        var result("exp", std::exp(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
};

struct sin {
    sin() {}

    var operator()(var& e) {
        var result("sin", std::sin(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
    var operator()(var&& e) {
        var result("sin", std::sin(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
};

struct cos {
    cos() {}

    var operator()(var& e) {
        var result("cos", std::cos(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
    var operator()(var&& e) {
        var result("cos", std::cos(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
};

struct ln {
    ln() {}

    var operator()(var& e) {
        var result("ln", std::log(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
    var operator()(var&& e) {
        var result("ln", std::log(e.value()));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
};

struct log {
    log() {}

    var operator()(var& e) {
        var result("log", std::log(e.value()) / std::log(2));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
    var operator()(var&& e) {
        var result("log", std::log(e.value()) / std::log(2));
        result.set_left(std::make_shared<var>(e));
        if(!(result.left()->is_binary_operation())) {
            var::aliases[&e].push_back(result.left());
        }
        return result;
    }
};

}  // namespace functions
}  // namespace autodiff
