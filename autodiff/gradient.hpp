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

    double operator[](var& x) { return gradients_[&x]; }

    double grad() {
        head_->set_gradient(1.0);
        head_->grad();
        // collect contributions
        for (const auto& v : variables_) {
            for(const auto& w : var::aliases) {
                for(const auto& r : w.second) {
                    if(r == v) {
                        gradients_[w.first] += r->grad_;
                    }
                }
            }
        }
    }

    void print_grad() {
        for (const auto& v : gradients_) {
            std::cout << "-..........." << std::endl;
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
    std::map<const var *, double> gradients_;
};
}  // namespace base
}  // namespace autodiff
