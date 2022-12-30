#include "io/var.hpp"
#include "autodiff/functions.hpp"

#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "gtest/gtest.h"

using namespace autodiff;
using namespace base;

TEST(basic, addition) {
    var a(1);
    var b(10);

    auto x_ = a + b;
    auto a_ = x_.get_left();
    auto b_ = x_.get_right();

    ASSERT_EQ(x_.value(), 11);
    ASSERT_EQ(a_->value(), 1);
    ASSERT_EQ(b_->value(), 10);
}
TEST(functions, exp) {
    // testing functions
    auto exp_ = autodiff::function::exp();

    var y_1(10);
    var y_2(1);
    var y_3 = y_1 + y_2;

    auto z_1 = exp_(y_1 + y_2);
    auto z_2 = exp_(y_3);

    ASSERT_NEAR(z_1.value(), 59874.1, 0.1);
    ASSERT_NEAR(z_2.value(), 59874.1, 0.1);
}

TEST(basic, computation_graph) {
    auto exp_ = autodiff::function::exp();

    var y_1(10);
    var y_2(1);

    auto z = exp_(y_1 + y_2);

    auto a = z.get_left();
    ASSERT_TRUE(a->is_binary_operation());
    ASSERT_TRUE(a->to_string() == "+");
    ASSERT_TRUE(a->get_left()->value() == 10);
    ASSERT_TRUE(a->get_right()->value() == 1);
}
/* while unique_stack is in expression.hpp, we test in here*/
/*
TEST(unique_stack, basic) {
    unique_stack<node> s;
    s.push(std::make_shared<node>(token("A")));
    auto uis = s.unique_items();
    std::list<std::shared_ptr<node>> l;

    for (auto n : uis) {
        l.push_back(n);
    }

    ASSERT_EQ(l.size(), 1);
}

void fill_count_map(std::map<std::shared_ptr<node>, int>& nodes,
                    const std::shared_ptr<node>& head) {
    auto it_h = nodes.insert(std::make_pair(head, 1));
    if (!it_h.second) {
        it_h.first->second++;
    }

    if (head->left() != nullptr) {
        fill_count_map(nodes, head->left());
    }
    if (head->right() != nullptr) {
        fill_count_map(nodes, head->right());
    }
}

auto unique_tester(const std::shared_ptr<node>& head) {
    std::map<std::shared_ptr<node>, int> nodes;
    fill_count_map(nodes, head);
    return nodes;
}

class test_expression : public base::expression<node> {
public:
    test_expression(postfix&& pfx) : base::expression<node>(std::move(pfx)) {}
    std::shared_ptr<node> get() { return head_; }
};

TEST(expression, unique) {
    std::string e1 = "(A+B)+A*(A+B)";
    auto g0 = test_expression(to_postfix(e1));
    // gets the shared pointer of the head of the expression
    auto nodes = unique_tester(g0.get());
    node a_plus_b = node(token("+"), token("A"), token("B"));
    for (const auto& n : nodes) {
        if (n.first->to_string() == "A") {
            ASSERT_EQ(n.second, 3);
            continue;
        }
        if (n.first->to_string() == "B") {
            ASSERT_EQ(n.second, 2);
            continue;
        }
        if (*n.first == a_plus_b) {
            ASSERT_EQ(n.second, 2);
            continue;
        }
    }
}
*/
