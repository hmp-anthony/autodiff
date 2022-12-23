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
    while ((pos = s.find_first_of(autodiff::ops)) != std::string::npos) {
        auto str1 = s.substr(0, pos);
        auto str2 = s[pos];
        ifx.add_token(std::make_shared<token>(str1));
        ifx.add_token(std::make_shared<token>(str2));
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

// --------------------------------------- THIS IS DIRTY.

int priority(std::string s) {
    if (s == "^") return 15;
    if (s == "*") return 14;
    if (s == "/") return 13;
    if (s == "+") return 12;
    if (s == "-") return 11;
    if (s == ">" || s == "<" || s == ">=" || s == "<=") return 10;
    if (s == "==" || s == "!=") return 9;
    if (s == "&&") return 8;
    if (s == "||") return 7;
    if (s == ",") return 1;
    // function
    return 20;
}

// This is really dirty as we have two sources of priority.
// One within the class def for token and one here.

bool has_higher_priority(const std::shared_ptr<token>& t1,
                         const std::shared_ptr<token>& t2) {
    int t1_p = priority(t1->to_string());
    int t2_p = priority(t2->to_string());
    std::cout << t1_p << " " << t2_p << std::endl;
    return t1_p >= t2_p;
}

bool has_equal_priority(const std::shared_ptr<token>& t1,
                        const std::shared_ptr<token>& t2) {
    int t1_p = priority(t1->to_string());
    int t2_p = priority(t2->to_string());
    std::cout << t1->to_string() << " " << t1_p << std::endl;
    std::cout << t2->to_string() << " " << t2_p << std::endl;
    return t1_p == t2_p;
}

bool is_left_associative(const std::shared_ptr<token>& t) {
    return !("^" == t->to_string());
}

// ---------------------------------------- END

postfix to_postfix(infix&& ifx) {
    std::stack<std::shared_ptr<token>> s;
    std::cout << ifx.to_string() << std::endl;
    auto ts = ifx.move_tokens();
    postfix pfx;

    for (const auto& t : ts) {
        if (t->is_constant() || t->is_variable()) {
            pfx.add_token(t);
            continue;
        }
        if (t->is_function()) {
            s.push(t);
            continue;
        }
        if (t->is_open_paren()) {
            s.push(t);
            continue;
        }

        if (t->is_closed_paren()) {
            auto c = s.top();
            s.pop();
            while (!c->is_open_paren()) {
                if (!c->is_comma()) {
                    pfx.add_token(s.top());
                }
                c = s.top();
                s.pop();
            }
            continue;
        }

        // error here somewhere
        if (t->is_binary_operation()) {
            if (!s.empty()) {
                auto c = s.top();
                std::cout << "before bool" << std::endl;
                std::cout << c->to_string() << " " << t->to_string()
                          << std::endl;
                bool is_bin_op = c->is_binary_operation();
                bool has_high_p = has_higher_priority(c, t);
                bool has_equal_p = has_equal_priority(c, t);
                bool is_left = is_left_associative(t);
                std::cout << is_bin_op << has_high_p << has_equal_p << is_left
                          << std::endl;

                // separate the argument to while to look for bugs
        

                while ((!is_bin_op || (is_bin_op && has_high_p) ||
                        (has_equal_p && is_left)) &&
                       !c->is_open_paren()) {
                    if (!c->is_comma()) {
                        pfx.add_token(c);
                    }
                    std::cout << "before pop  " << s.top()->to_string()
                              << std::endl;
                    s.pop();
                    std::cout << "after pop  " << s.top()->to_string()
                              << std::endl;
                    if (!s.empty()) {
                        c = s.top();
                    }
                }
            }
            s.push(t);
            continue;
        }
        /*
        if (t->is_comma()) {
            auto c = s.top();
            while (!(c->is_open_paren() || c->is_comma())) {
                pfx.add_token(c);
                s.pop();
                if (!s.empty()) {
                    c = s.top();
                }
            }
            s.push(t);
        }*/
    }
    /*
    while (!s.empty()) {
        auto pop = s.top();
        s.pop();
        if (pop->is_comma()) {
            pfx.add_token(pop);
        }
    }
    */
    std::cout << pfx.to_string() << std::endl;
    return pfx;
}

postfix to_postfix(const std::string& s) { return to_postfix(to_infix(s)); }

}  // namespace autodiff
