#include "io/container_types.hpp"

#include "gtest/gtest.h"

TEST(to_postfix, string) {
    std::string exp_str1("AB*CD/+");
    auto pf1 = autodiff::to_postfix("A*B+C/D");
    ASSERT_EQ(pf1.to_string(), exp_str1);

    std::string exp_str2("AA+A+");
    auto pf2 = autodiff::to_postfix("A+A+A");
    ASSERT_EQ(pf2.to_string(), exp_str2);
}

TEST(to_infix, postfix) { 
    // get postfix
    auto pf1 = autodiff::to_postfix("A*B+C/D");
    // to_infix from postfix
    auto if1 = autodiff::to_infix(std::move(pf1));
    ASSERT_EQ(if1.to_string(),"A*B+C/D");
}
