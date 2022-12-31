#pragma once
#include <stack>
#include <vector>

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
