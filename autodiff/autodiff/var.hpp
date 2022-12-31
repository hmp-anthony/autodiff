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

    void populate() {
        if (left_) {
            items_.push_back(left_);
            left_->populate();
        }
        items_.push_back(std::make_shared<var>(t_));
        if (right_) {
            items_.push_back(right_);
            right_->populate();
        }
    }

    void prepare() {
        populate();
        // now use unique stack to get variables.
        
    }

    token::token_type type() { return t_.type(); }
    const std::string& to_string() { return t_.to_string(); }
        

private:
    token t_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::vector<std::shared_ptr<var>> parents_;
    std::optional<double> v_;

    std::list<std::shared_ptr<var>> items_;
};

}  // namespace base
}  // namespace autodiff
