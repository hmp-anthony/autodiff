cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(autodiff)

if (MSVC)
    # Solves complications that arrise in the runtime when compiling
    # gtest using msvc.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (MSVC)


include_directories(autodiff) 

enable_testing()

include(cmake/googletest.cmake)
fetch_googletest(
    ${PROJECT_SOURCE_DIR}/cmake
    ${PROJECT_BINARY_DIR}/googletest
    )

add_subdirectory(autodiff)
add_subdirectory(docs)
add_subdirectory(examples)
