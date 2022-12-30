#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <vector>
namespace autodiff {
class var {
private:
    double value_;
    std::string op_;
    std::shared_ptr<var> left_;
    std::shared_ptr<var> right_;

public:
    var(const var& v)
        : value_(v.value_), left_(v.left_), right_(v.right_), op_(v.op_) {}
    var(const double& v) : value_(v){};
    void set_left(std::shared_ptr<var> v) { left_ = v; }
    void set_right(std::shared_ptr<var> v) { right_ = v; }

    std::shared_ptr<var> get_left() { return left_; }
    std::shared_ptr<var> get_right() { return right_; }

    double value() const { return value_; }

    bool is_operation() {
        if (op_ == "") return false;
        return true;
    }

    std::string operation() { return op_; }

    var operator=(const var& v) {
        value_ = v.value_;
        left_ = v.left_;
        right_ = v.right_;
        op_ = v.op_;
        return *this;
    }

    friend var operator+(const var& l, const var& r) {
        var result(l.value_ + r.value_);
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        result.op_ = "+";
        return result;
    }

    friend var operator*(const var& l, const var& r) {
        var result(l.value_ * r.value_);
        result.left_ = std::make_shared<var>(l);
        result.right_ = std::make_shared<var>(r);
        result.op_ = "*";
        return result;
    }

    friend std::ostream& operator<<(std::ostream& os, const var& var) {
        os << var.value_;
        return os;
    }
};

namespace function {
struct exp {
    exp() {}
    var operator()(const var e) {
        var result(std::exp(e.value()));
        result.set_left(std::make_shared<var>(e));
        return result;
    }
};
}  // namespace function
}  // namespace autodiff
