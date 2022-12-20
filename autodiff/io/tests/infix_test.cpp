#include "io/infix.hpp"

#include "gtest/gtest.h"

TEST(as_infix, infix) {
    std::string exp_str1("A*B+C/D");
    auto if1 = autodiff::to_infix("A*B+C/D");
    ASSERT_EQ(if1.to_string(), exp_str1);
}

