#include "io/postfix.hpp"

#include "gtest/gtest.h"

TEST(to_postfix, postfix) {
    std::string exp_str1("AB*CD/+");
    auto pf1 = autodiff::to_postfix("A*B+C/D");
    ASSERT_EQ(pf1.to_string(), exp_str1);

    std::string exp_str2("AA+A+");
    auto pf2 = autodiff::to_postfix("A+A+A");
    ASSERT_EQ(pf2.to_string(), exp_str2);
}

