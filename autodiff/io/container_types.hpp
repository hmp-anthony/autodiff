#pragma once

#include <stack>

#include "io/token.hpp"

namespace autodiff {

namespace type_rule {
struct infix {
    // add "valid" method to check validity
};
struct postfix {
    // add "valid" method to check validity
};
}  // namespace type_rule

using infix = token_container<type_rule::infix>;
using postfix = token_container<type_rule::postfix>;

infix to_infix(std::string s) {
    infix ifx;
    size_t pos = 0;
    std::string t;
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    while ((pos = s.find_first_of("+-/*()")) != std::string::npos) {
        ifx.add_token(std::make_shared<token>(s.substr(0, pos)));
        ifx.add_token(std::make_shared<token>(s[pos]));
        s.erase(0, pos + 1);
    }
    if (s.size() > 0) {
        ifx.add_token(std::make_shared<token>(s));
    }
    return ifx;
}

infix to_infix(postfix&& pfx) {
    std::stack<std::string> s;
    auto ts = pfx.move_tokens();
    for (const auto& t : ts) {
        auto tp = t->type();
        // Check type. If symbol is operand push it on the stack
        if (tp == token::token_type::variable ||
            tp == token::token_type::constant) {
            s.push(t->to_string());
        } else {
            auto s1 = s.top();
            s.pop();
            auto s2 = s.top();
            s.pop();
            auto str = s2 + t->to_string() + s1;
            s.push(str);
        }
    }
    auto ifx = to_infix(std::move(s.top()));
    return ifx;
}

void function_pass(std::list<std::shared_ptr<token>>& ts) {
    for (const auto& t : ts) {
    }
}

//! convert infix to postfix
postfix to_postfix(infix&& ifx) {
    std::stack<std::shared_ptr<token>> s;
    auto ts = ifx.move_tokens();
    function_pass(ts);
    postfix pfx;
    for (const auto& t : ts) {
        if (t->is_constant() || t->is_variable()) {
            pfx.add_token(t);
            continue;
        }

        if (t->is_open_paren()) {
            s.push(t);
            continue;
        }

        if (t->is_closed_paren()) {
            while (*s.top() != ops_map.at('(')) {
                pfx.add_token(s.top());
                s.pop();
            }
            s.pop();
            continue;
        }

        while (!s.empty() && (s.top())->priority() >= t->priority()) {
            pfx.add_token(s.top());
            s.pop();
        }
        s.push(t);
    }

    while (!s.empty()) {
        pfx.add_token(s.top());
        s.pop();
    }
    std::cout << pfx.to_string() << std::endl;
    return pfx;
}

postfix to_postfix(const std::string& s) { return to_postfix(to_infix(s)); }
}  // namespace autodiff
