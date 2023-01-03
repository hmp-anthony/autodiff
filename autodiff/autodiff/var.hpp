#pragma once

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <utility>

#include "autodiff/token.hpp"
#include "autodiff/var.hpp"

namespace autodiff {
namespace base {
class var {
public:
    var(const var& v) = default;
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

    var operator=(const var& v) {
        t_ = v.t_;
        name_ = v.name_;
        v_ = v.v_;
        left_ = v.left_;
        right_ = v.right_;
        parents_ = v.parents_;
        return *this;
    }

    bool is_binary_operation() { return t_.is_binary_operation(); }
    bool is_variable() { return t_.is_variable(); }

    void add_parent(const std::shared_ptr<var>& p) { parents_.push_back(p); }

    void set_left(std::shared_ptr<var> lc) { left_ = lc; }
    void set_right(std::shared_ptr<var> rc) { right_ = rc; }

    std::vector<std::shared_ptr<var>>& parents() { return parents_; }
    std::shared_ptr<var>& left() { return left_; }
    std::shared_ptr<var>& right() { return right_; }

    char name() { return name_; }
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

private:
    token t_;
    char name_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;
    std::vector<std::shared_ptr<var>> parents_;
    std::optional<double> v_;
};

class expression {
public:
    expression(var v) {
        head_ = std::make_shared<var>(v);
        populate_variables(head_);

        std::vector<std::pair<char, std::shared_ptr<var>>> unique;
        for (const auto& v : variables_) {
            auto it = std::find_if(
                unique.begin(), unique.end(),
                [&v](const std::pair<char, std::shared_ptr<var>>& e) {
                    return e.first == v->name();
                });

            if (it == std::end(unique)) {
                // if not found, add it
                auto p = std::make_pair(v->name(), v);
                unique.push_back(p);
            } else {
                // if found
                (*it).second->add_parent(v->parents()[0]);
                v->parents()[0]->left() = nullptr;
                v->parents()[0]->right() = nullptr;
            }
        }
        variables_.clear();
        for (const auto& u : unique) {
            variables_.push_back(u.second);
        }
    }

    auto grad() {
        head_->grad_ = 1;
        head_->grad();
    }

    void print_grad() {
        for (const auto& v : variables_) {
            std::cout << v->grad_ << std::endl;
        }
    }

private:
    void populate_variables(const std::shared_ptr<var>& root) {
        if (!root) return;
        if (!root->left() && !root->right()) {
            variables_.push_back(root);
            return;
        }
        if (root->left()) populate_variables(root->left());
        if (root->right()) populate_variables(root->right());
    }
    std::shared_ptr<var> head_;
    std::list<std::shared_ptr<var>> variables_;
};

}  // namespace base
}  // namespace autodiff
