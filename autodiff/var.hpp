#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "token.hpp"
#include "var.hpp"

namespace autodiff {
namespace base {
class var {
public:
    var(const var& v)
        : t_(v.t_),
          name_(v.name_),
          left_(std::move(v.left_)),
          right_(std::move(v.right_)),
          v_(v.v_),
          grad_(0),
          visit_count_(0) {}
    explicit var(std::string s, char name = ' ')
        : t_(s), name_(name), grad_(0), visit_count_(0) {}
    explicit var(double v, char name = ' ')
        : t_(v), name_(name), grad_(0), visit_count_(0) {
        v_ = v;
    }
    explicit var(token t, char name = ' ')
        : t_(std::move(t)), name_(name), grad_(0), visit_count_(0){};
    explicit var(var&& n, char name = ' ')
        : t_(std::move(n.t_)),
          name_(name),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)),
          v_(n.v_),
          grad_(0),
          visit_count_(0) {}

    friend var operator+(const var& l, const var& r) {
        var result(token(std::string("+")));
        result.v_ = l.v_.value() + r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        auto result_ptr = std::make_shared<var>(result);
        return result;
    }

    friend var operator-(const var& l, const var& r) {
        var result(token(std::string("+")));
        result.v_ = l.v_.value() - r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        auto result_ptr = std::make_shared<var>(result);
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        var result(token(std::string("*")));
        result.v_ = l.v_.value() * r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        auto result_ptr = std::make_shared<var>(result);
        return result;
    }

    friend var operator/(const var& l, const var& r) {
        var result(token(std::string("/")));
        result.v_ = l.v_.value() / r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        auto result_ptr = std::make_shared<var>(result);
        return result;
    }

    var operator=(const var& v) {
        t_ = v.t_;
        name_ = v.name_;
        v_ = v.v_;
        left_ = v.left_;
        right_ = v.right_;
        return *this;
    }

    bool is_binary_operation() { return t_.is_binary_operation(); }
    bool is_variable() { return t_.is_variable(); }
    bool is_constant() { return t_.is_constant(); }

    void set_left(std::shared_ptr<var> lc) { left_ = lc; }
    void set_right(std::shared_ptr<var> rc) { right_ = rc; }

    std::shared_ptr<var>& left() { return left_; }
    std::shared_ptr<var>& right() { return right_; }

    char name() { return name_; }
    double value() { return v_.value(); }
    void reset_value() { v_.reset(); }

    token& get_token() { return t_; }

    double eval() { return v_.value(); }

    void grad() {
        if (is_binary_operation()) {
            std::string str = to_string();
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
    }

    void addition() {
        left_->grad_ += grad_;
        right_->grad_ += grad_;
    }

    void multiplication() {
        left_->grad_ += grad_ * right_->eval();
        right_->grad_ += grad_ * left_->eval();
    }

    void division() {
        double r = right()->eval();
        double l = left()->eval();
        (left())->grad_ += grad_ * (1.0 / r);
        (right())->grad_ -= grad_ * (l / (r * r));
    }

    void subtraction() {
        (left())->grad_ -= grad_;
        (right())->grad_ += grad_;
    }

    double grad_;
    int visit_count_;

    const std::string& to_string() { return t_.to_string(); }

private:
    token t_;
    char name_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::optional<double> v_;
};

}  // namespace base
}  // namespace autodiff
