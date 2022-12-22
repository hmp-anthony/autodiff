#include "io/token.hpp"

#include "gtest/gtest.h"

using namespace autodiff;

TEST(token, basic) {
    token t1 = "this_is_a_variable";
    std::string e1 = "this_is_a_variable";
    ASSERT_EQ(t1.to_string(), e1);
    ASSERT_EQ(t1.type(), token::token_type::variable);
    ASSERT_NE(t1.type(), token::token_type::constant);
    ASSERT_NE(t1.type(), token::token_type::binary_operation);
    ASSERT_EQ(t1.is_variable(), true);
    ASSERT_EQ(t1.is_constant(), false);
    ASSERT_EQ(t1.is_binary_operation(), false);
    ASSERT_EQ(t1.is_function(), false);

    token t2 = "+";
    ASSERT_NE(t2.type(), token::token_type::variable);
    ASSERT_NE(t2.type(), token::token_type::constant);
    ASSERT_EQ(t2.type(), token::token_type::binary_operation);
    ASSERT_EQ(t2.is_variable(), false);
    ASSERT_EQ(t2.is_constant(), false);
    ASSERT_EQ(t2.is_binary_operation(), true);
    ASSERT_EQ(t2.is_function(), false);

    token t3 = "exp";
    ASSERT_NE(t3.type(), token::token_type::variable);
    ASSERT_NE(t3.type(), token::token_type::constant);
    ASSERT_NE(t3.type(), token::token_type::binary_operation);
    ASSERT_EQ(t3.type(), token::token_type::function);
    ASSERT_EQ(t3.is_variable(), false);
    ASSERT_EQ(t3.is_constant(), false);
    ASSERT_EQ(t3.is_binary_operation(), false);
    ASSERT_EQ(t3.is_function(), true);
}

