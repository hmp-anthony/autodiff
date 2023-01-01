#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>

#include "autodiff/token.hpp"
#include "autodiff/unique_stack.hpp"
#include "autodiff/var.hpp"

namespace autodiff {

namespace base {
class var {
public:
    var(const var& v)
        : t_(v.t_),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_),
          grad_(0),
          visit_count_(0){};
    var(var& v)
        : t_(v.t_),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_),
          grad_(0),
          visit_count_(0){};
    var(std::string s) : t_(s), grad_(0), visit_count_(0) {}
    var(double v) : t_(v), grad_(0), visit_count_(0) { v_ = v; }
    var(token t) : t_(std::move(t)), grad_(0), visit_count_(0){};
    var(var&& n)
        : t_(std::move(n.t_)),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)),
          v_(n.v_),
          grad_(0),
          visit_count_(0) {}

    friend var operator+(const var& l, const var& r) {
        var result(token(std::string("+")));
        result.v_ = l.v_.value() + r.v_.value();

        result.left_ = std::make_shared<var>(l);
        items_.push_back(result.left_);

        result.right_ = std::make_shared<var>(r);
        items_.push_back(result.right_);

        auto result_ptr = std::make_shared<var>(result);
        result.left_->add_parent(result_ptr);
        result.right_->add_parent(result_ptr);
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        var result(token(std::string("*")));
        result.v_ = l.v_.value() * r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        auto result_ptr = std::make_shared<var>(result);
        result.left_->add_parent(result_ptr);
        result.right_->add_parent(result_ptr);
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

    bool is_binary_operation() { return t_.is_binary_operation(); }
    bool is_variable() { return t_.is_variable(); }

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

    token& get_token() { return t_; }

    virtual double eval() { return v_.value(); }

    void print() {
        std::cout << "---" << std::endl;
        if (left_) {
            std::cout << left_->value() << " " << left_->parents_.size()
                      << std::endl;
            left_->print();
        }
        std::cout << t_.to_string() << std::endl;
        if (right_) {
            std::cout << right_->value() << " " << right_->parents_.size()
                      << std::endl;
            right_->print();
        }
    }

    void grad() {
        // Do not propagate untill all parents have
        // touched their child. I do not advocate this.
        if (++visit_count_ < parents().size() && !parents().empty()) {
            return;
        }

        // carry out grad
        if (is_binary_operation()) {
            std::string str = to_string();
            if (str == "*") {
                multiplication();
            } else if (str == "+") {
                addition();
                /*
            } else if (str == "/") {
                division(s);
            } else if (str == "-") {
                subtraction(s);
                */
            }
            left_->grad();
            right_->grad();
            return;
        }
    }

    void addition() {
        left_->grad_ += grad_;
        right_->grad_ += grad_;
    }

    void multiplication() {
        left_->grad_ += grad_ * right_->v_.value();
        right_->grad_ += grad_ * left_->v_.value();
    }

    double grad_;
    int visit_count_;

    const std::string& to_string() { return t_.to_string(); }

    static std::vector<std::shared_ptr<var>> items_;

private:
    token t_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::vector<std::shared_ptr<var>> parents_;
    std::optional<double> v_;
};

class expression {
public:
    expression(var v) {
        head_ = std::make_shared<var>(v);
        for (const auto& i : var::items_) {
            std::cout << i->to_string() << std::endl;
        }
    }
    auto grad() {
        head_->grad_ = 1;
        head_->grad();
    }

private:
    std::shared_ptr<var> head_;
    std::list<std::shared_ptr<var>> variables_;
};

}  // namespace base
}  // namespace autodiff
