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
            // i.e the node already exists;
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
//! Absorbs token to form a expression node
class node {
public:
    explicit node(token t) : t_(std::move(t)){};
    explicit node(token t, token l, token r)
        : t_(std::move(t)),
          left_(std::make_shared<node>(l)),
          right_(std::make_shared<node>(r)) {}

    explicit node(node&& n)
        : t_(std::move(n.t_)),
          left_(std::move(n.left_)),
          right_(std::move(n.right_)) {}
    node() = delete;

    bool operator<(const node& n) const {
        if (left_ == nullptr && n.left_ == nullptr && right_ == nullptr &&
            n.right_ == nullptr) {
            // i.e if this node and the other are NOT
            // operations. More specifically they are
            // variable or constant nodes.
            return t_.to_string() < n.t_.to_string();
        }
        return (t_.to_string() < n.t_.to_string() ||
                (t_.to_string() == n.t_.to_string() &&
                 left_->t_.to_string() < n.left_->t_.to_string()) ||
                (t_.to_string() == n.t_.to_string() &&
                 left_->t_.to_string() == n.left_->t_.to_string() &&
                 right_->t_.to_string() < n.right_->t_.to_string()));
    }

    //! Checks that nodes have the same data.
    //! This does not mean they are the SAME node.
    bool operator==(const node& n) {
        if (t_ == n.t_ && left_->t_ == n.left_->t_ &&
            right_->t_ == n.right_->t_)
            return true;
        return false;
    }

    void add_parent(const std::shared_ptr<node>& p) { parents_.push_back(p); }

    void set_left_child(std::shared_ptr<node> lc) { left_ = lc; }
    void set_right_child(std::shared_ptr<node> rc) { right_ = rc; }

    const std::vector<std::shared_ptr<node>>& parents() { return parents_; }
    std::shared_ptr<node>& left() { return left_; }
    std::shared_ptr<node>& right() { return right_; }

    void reset_value() { v_.reset(); }

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
    std::shared_ptr<node> left_;
    std::shared_ptr<node> right_;
    std::vector<std::shared_ptr<node>> parents_;
    std::optional<double> v_;
};

template <typename NODE>
class expression {
public:
    expression(var v) {
        // Here we convert the "var" computational graph
        // to an expression which implements back_prop.
    }

    //! Evaluates the expression. Note that we do not utilise the
    //! cached evaluation functionality present in the node class.
    //! This is quite simply because it provides no real advantage
    //! if we are just evaluating an expression.
    double operator[](const state& s) { return head_->eval(s); }
    void print() {
        std::cout << head_ << std::endl;
        head_->print();
    }

protected:
    std::shared_ptr<NODE> head_;
    std::list<std::shared_ptr<NODE>> variables_;

private:
};
}  // namespace base

}  // namespace autodiff
