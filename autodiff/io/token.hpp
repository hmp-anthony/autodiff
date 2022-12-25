#pragma once

#include <algorithm>
#include <cctype>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

namespace autodiff {

const std::string ops("(^-+*/)");
const std::map<std::string, int> priority_map = {{"(", -1}, {"^", 0}, {"-", 1},
                                                 {"+", 1},  {"*", 2}, {"/", 2}};
const std::vector<std::string> functions = {"exp", "sin", "ln"};

class token {
public:
    enum class token_type { variable, constant, binary_operation, function };
    enum class op_priority { bracket, other, division_multiplcation };

    token(std::string s) : s_(std::move(s)), t_(set_type()) {}
    token(char c) : s_(std::string(1, c)), t_(set_type()) {}
    token(const char* c_s) : s_(c_s), t_(set_type()) {}

    token(const token& t) : s_(t.s_), t_(t.t_) {}
    token(token&& t) : s_(std::move(t.s_)), t_(std::move(t.t_)) {}

    bool operator<(const token& t) const { return s_ < t.s_; }
    bool operator==(const token& t) const { return s_ == t.s_; }
    bool operator!=(const token& t) const { return s_ != t.s_; }

    const std::string& to_string() const { return s_; }

    token_type type() const { return t_; }

    int priority() const {
        if (t_ == token_type::function) {
            return 40;
        }
        if (t_ != token_type::binary_operation) {
            return ops.size();
        }
        return priority_map.at(s_);
    }

    bool is_binary_operation() {
        if (s_.find_first_of(ops) != std::string::npos) {
            return true;
        }
        return false;
    }
    bool is_function() {
        auto p = std::find(functions.begin(), functions.end(), s_);
        if (p != functions.end()) {
            return true;
        }
        return false;
    }
    bool is_constant() {
        return !s_.empty() &&
               std::find_if(s_.begin(), s_.end(), [](unsigned char c) {
                   return !std::isdigit(c);
               }) == s_.end();
    }
    bool is_open_paren() { return s_ == "("; }
    bool is_closed_paren() { return s_ == ")"; }
    bool is_variable() { return t_ == token_type::variable ? true : false; }
    bool is_comma() { return s_ == ","; }

private:
    token_type set_type() {
        if (is_binary_operation()) {
            return token_type::binary_operation;
        }
        if (is_function()) {
            return token_type::function;
        }
        if (is_constant()) {
            return token_type::constant;
        }
        return token_type::variable;
    }

    std::string s_;
    token_type t_;
};

const std::map<char, token> ops_map = {{'(', token('(')}, {'+', token('+')},
                                       {'-', token('-')}, {'*', token('*')},
                                       {')', token(')')}, {'/', token('/')}};

template <typename T>
class token_container {
public:
    token_container() = default;

    void add_token(std::shared_ptr<token> t) {
        if (t->to_string() == "") return;
        tks_.push_back(t);
    }

    std::list<std::shared_ptr<token>>&& move_tokens() {
        return std::move(tks_);
    }

    std::list<std::shared_ptr<token>> copy_tokens() { return tks_; }

    std::string to_string() const {
        std::string ts;
        for (const auto& t : tks_) {
            ts += t->to_string();
        }
        return ts;
    }

    void reset() { tks_.clear(); }

private:
    std::list<std::shared_ptr<token>> tks_;
};

}  // namespace autodiff
