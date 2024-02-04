#include "token.hpp"

using namespace autodiff;

TEST(token, basic) {
    token t1 = "this_is_a_variable";
    std::string e1 = "this_is_a_variable";
    ASSERT_EQ(t1.to_string(), e1);
    ASSERT_EQ(t1.is_variable(), true);
    ASSERT_EQ(t1.is_constant(), false);
    ASSERT_EQ(t1.is_binary_operation(), false);

    token t2 = "+";
    ASSERT_EQ(t2.is_variable(), false);
    ASSERT_EQ(t2.is_constant(), false);
    ASSERT_EQ(t2.is_binary_operation(), true);

    token t4 = "(";
    ASSERT_TRUE(t4.is_open_paren());
    ASSERT_FALSE(t4.is_closed_paren());

    token t5 = ")";
    ASSERT_FALSE(t5.is_open_paren());
    ASSERT_TRUE(t5.is_closed_paren());

    token t6 = ",";
    ASSERT_TRUE(t6.is_comma());

    token t7 = "x";
    ASSERT_EQ(t7.is_variable(), true);
    ASSERT_EQ(t7.is_constant(), false);

    token t8 = "y";
    ASSERT_EQ(t8.is_variable(), true);
    ASSERT_EQ(t8.is_constant(), false);
}

