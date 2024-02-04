set(_tests
    var_test
    )

foreach(_test IN LISTS _tests)
  add_executable(${_test} ${_test}.cpp)
  add_test(NAME ${_test} COMMAND ${_test})
endforeach()
