cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 17)
project(autodiff CXX)

if (MSVC)
    # Solves complications that arrise in the runtime when compiling
    # gtest using msvc.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
endif (MSVC)

enable_testing()

include_directories(autodiff) 

add_subdirectory(autodiff)
add_subdirectory(docs)
add_subdirectory(examples)
