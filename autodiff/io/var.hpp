#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <stack>
#include <vector>

#include "io/token.hpp"
#include "io/var.hpp"

namespace autodiff {

using state = std::map<std::string, double>;

template <typename T>
class unique_stack {
public:
    struct cmp {
        bool operator()(const std::shared_ptr<T>& a,
                        const std::shared_ptr<T>& b) const {
            return *a < *b;
        }
    };

    void push(const std::shared_ptr<T>& n) {
        auto it = uniques_.insert(n);
        if (!it.second) {
            // i.e the var already exists;
            s_.push(*it.first);
            return;
        }
        s_.push(n);
    }

    void pop() { s_.pop(); }
    std::shared_ptr<T>& top() { return s_.top(); }
    std::set<std::shared_ptr<T>, cmp> unique_items() { return uniques_; }

private:
    std::set<std::shared_ptr<T>, cmp> uniques_;
    std::stack<std::shared_ptr<T>> s_;
};

namespace base {
//! Absorbs token to form a expression var
class var {
public:
    var(const var& v)
        : t_(t_.to_string()),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_){};
    explicit var(var& v)
        : t_(t_.to_string()),
          left_(v.left_),
          right_(v.right_),
          parents_(v.parents_),
          v_(v.v_){};
    explicit var(std::string s) : t_(s){};
    explicit var(token t) : t_(std::move(t)){};
    explicit var(token t, token l, token r)
        : t_(std::move(t)),
          left_(std::make_shared<var>(l)),
          right_(std::make_shared<var>(r)) {}

    explicit var(var&& n)
        : t_(std::move(n.t_)),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)) {}

    friend var operator+(const var& l, const var& r) {
        auto tk = token("+");
        var result(tk);
        result.v_ = l.v_.value() + r.v_.value();
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        auto tk = token("*");
        var result(tk);
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

    void add_parent(const std::shared_ptr<var>& p) { parents_.push_back(p); }

    void set_left(std::shared_ptr<var> lc) { left_ = lc; }
    void set_right(std::shared_ptr<var> rc) { right_ = rc; }

    std::shared_ptr<var> get_left() { return left_; }
    std::shared_ptr<var> get_right() { return right_; }

    const std::vector<std::shared_ptr<var>>& parents() { return parents_; }
    std::shared_ptr<var>& left() { return left_; }
    std::shared_ptr<var>& right() { return right_; }

    double value() { return v_.value(); }
    void reset_value() { v_.reset(); }

    token get_token() { return t_; }

    double operator[](const state& s) {
        if (v_) {
            return v_.value();
        }
        v_ = eval(s);
        return v_.value();
    }

    virtual double eval(const state& s) {
        if (t_.type() == token::token_type::binary_operation) {
            if (t_.to_string() == "+") return right_->eval(s) + left_->eval(s);
            if (t_.to_string() == "-") return right_->eval(s) - left_->eval(s);
            if (t_.to_string() == "/") {
                double z;
                if ((z = left_->eval(s)) == 0.0) {
                    throw std::invalid_argument("dividing by zero!");
                }
                return right_->eval(s) / left_->eval(s);
            };
            if (t_.to_string() == "*") return right_->eval(s) * left_->eval(s);
            return 0;
        }
        if (t_.type() == token::token_type::variable) {
            return s.at(t_.to_string());
        }
        if (t_.type() == token::token_type::constant) {
            return std::stod(t_.to_string());
        }
        return 0;
    }

    void print() {
        std::cout << "------------" << std::endl;
        std::cout << t_.to_string() << std::endl;
        std::cout << "------------" << std::endl;
        if (left_) {
            std::cout << left_ << std::endl;
            left_->print();
        }
        if (right_) {
            std::cout << right_ << std::endl;
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
