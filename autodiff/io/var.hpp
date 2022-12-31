#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>

#include "io/token.hpp"
#include "io/unique_stack.hpp"
#include "io/var.hpp"

namespace autodiff {

using state = std::map<std::string, double>;

namespace base {
class var {
public:
    var(const var& v)
        : t_(v.t_),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_){};
    var(var& v)
        : t_(v.t_),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_){};
    var(std::string s) : t_(s){};
    var(double v) : t_(v) { v_ = v; }
    var(token t) : t_(std::move(t)){};
    var(token t, token l, token r)
        : t_(t),
          left_(std::make_shared<var>(l)),
          right_(std::make_shared<var>(r)) {}

    var(var&& n)
        : t_(std::move(n.t_)),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)),
          v_(n.v_) {}

    friend var operator+(const var& l, const var& r) {
        var result(token(std::string("+")));
        result.v_ = l.v_.value() + r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        var result(token(std::string("*")));
        result.v_ = l.v_.value() * r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        return result;
    }

    bool operator<(const var& n) const {
        if (left_ == nullptr && n.left_ == nullptr && right_ == nullptr &&
            n.right_ == nullptr) {
            // i.e if this var and the other are NOT
            // operations. More specifically they are
            // variable or constant vars.
            return t_.to_string() < n.t_.to_string();
        }
        return (t_.to_string() < n.t_.to_string() ||
                (t_.to_string() == n.t_.to_string() &&
                 left_->t_.to_string() < n.left_->t_.to_string()) ||
                (t_.to_string() == n.t_.to_string() &&
                 left_->t_.to_string() == n.left_->t_.to_string() &&
                 right_->t_.to_string() < n.right_->t_.to_string()));
    }

    var operator=(const var& v) {
        t_ = v.t_;
        v_ = v.v_;
        left_ = v.left_;
        right_ = v.right_;
        parents_ = v.parents_;
        return *this;
    }

    //! Checks that vars have the same data.
    //! This does not mean they are the SAME var.
    bool operator==(const var& n) {
        if (t_ == n.t_ && left_->t_ == n.left_->t_ &&
            right_->t_ == n.right_->t_)
            return true;
        return false;
    }

    bool is_binary_operation() {
        return t_.type() == token::token_type::binary_operation;
    }

    void add_parent(const std::shared_ptr<var>& p) { parents_.push_back(p); }

    void set_left(std::shared_ptr<var> lc) { left_ = lc; }
    void set_right(std::shared_ptr<var> rc) { right_ = rc; }

    std::shared_ptr<var>& get_left() { return left_; }
    std::shared_ptr<var>& get_right() { return right_; }

    const std::vector<std::shared_ptr<var>>& parents() { return parents_; }
    std::shared_ptr<var>& left() { return left_; }
    std::shared_ptr<var>& right() { return right_; }

    double value() { return v_.value(); }
    void reset_value() { v_.reset(); }

    token get_token() { return t_; }

    virtual double eval() {
        return v_.value();
    }

    void print() {
        std::cout << "------------" << std::endl;
        std::cout << t_.to_string() << std::endl;
        std::cout << "------------" << std::endl;
        if (left_) {
            std::cout << left_->value() << std::endl;
            left_->print();
        }
        std::cout << t_.to_string() << std::endl;
        if (right_) {
            std::cout << right_->value() << std::endl;
            right_->print();
        }
    }

    token::token_type type() { return t_.type(); }
    const std::string& to_string() { return t_.to_string(); }

private:
    token t_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::vector<std::shared_ptr<var>> parents_;
    std::optional<double> v_;
};

}  // namespace base
}  // namespace autodiff
