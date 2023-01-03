#pragma once

#include <string.h>

#include <algorithm>
#include <cctype>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>

namespace autodiff {

const std::string ops("(^-+*/)");
const std::map<std::string, int> priority_map = {{"(", -1}, {"^", 0}, {"-", 1},
                                                 {"+", 1},  {"*", 2}, {"/", 2}};

class token {
public:
    token(double v, bool is_constant = false) : s_(std::to_string(v)) {
        is_variable_ = !is_constant;
        is_constant_ = is_constant;
        is_binary_operation_ = false;
    }

    token(std::string s, bool is_constant = false) : s_(s) {
        is_binary_operation_ = is_binary_operation();
        if (is_binary_operation_) {
            is_variable_ = false;
            is_constant_ = false;
        } else {
            is_variable_ = !is_constant;
            is_constant_ = is_constant;
        }
    }

    token(const char* c_s, bool is_constant = false) : s_(c_s) {
        is_binary_operation_ = is_binary_operation();
        if (is_binary_operation_) {
            is_variable_ = false;
            is_constant_ = false;
        } else {
            is_variable_ = !is_constant;
            is_constant_ = is_constant;
        }
    }

    token(const token& t) {
        s_ = t.s_;
        is_variable_ = t.is_variable_;
        is_constant_ = t.is_constant_;
        is_binary_operation_ = t.is_binary_operation_;
    }

    bool operator<(const token& t) const { return s_ < t.s_; }
    bool operator==(const token& t) const { return s_ == t.s_; }
    bool operator!=(const token& t) const { return s_ != t.s_; }

    token operator=(const token& t) {
        s_ = t.s_;
        is_variable_ = t.is_variable_;
        is_constant_ = t.is_constant_;
        is_binary_operation_ = t.is_binary_operation_;
        return *this;
    }

    const std::string& to_string() const { return s_; }

    bool is_binary_operation() {
        if (s_.find_first_of(ops) != std::string::npos) {
            return true;
        }
        return false;
    }
    bool is_constant() { return is_constant_; }
    bool is_open_paren() { return s_ == "("; }
    bool is_closed_paren() { return s_ == ")"; }
    bool is_variable() { return is_variable_; }
    bool is_comma() { return s_ == ","; }

private:
    std::string s_;
    bool is_variable_;
    bool is_constant_;
    bool is_binary_operation_;
};

const std::map<char, token> ops_map = {{'(', token('(')}, {'+', token('+')},
                                       {'-', token('-')}, {'*', token('*')},
                                       {')', token(')')}, {'/', token('/')}};

}  // namespace autodiff
