#include "var.hpp"

namespace autodiff {
namespace base {

class gradient {
public:
    gradient(var v) {
        // set head
        head_ = std::make_shared<var>(v);
        // populate variables
        populate_variables(head_);
        // fire off gradient
        grad();
    }

    double operator[](char x) { return gradients_[x]; }

    void grad() {
        head_->grad_ = 1;
        head_->grad();
        // collect contributions
        for (const auto& v : variables_) {
            gradients_[v->name()] += v->grad_;
        }
    }

    void print_grad() {
        for (const auto& v : gradients_) {
            std::cout << v.first << " " << v.second << std::endl;
        }
    }

    std::list<std::shared_ptr<var>> variables() { return variables_; }

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
    std::map<char, double> gradients_;
};
}  // namespace base
}  // namespace autodiff
