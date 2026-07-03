cmake_minimum_required(VERSION 3.16)
project(superfaiss CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(superfaiss STATIC
	src/kernels.cpp
	src/alloc.cpp
	src/validate.cpp
	src/bake.cpp
	src/query.cpp
)
target_include_directories(superfaiss PUBLIC include)

# Determinism: forbid FP contraction (fused multiply-add would change rounding and
# break scalar/SIMD bit-equality).
if(MSVC)
	target_compile_options(superfaiss PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superfaiss PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

add_executable(superfaiss_tests tests/test_main.cpp)
target_link_libraries(superfaiss_tests PRIVATE superfaiss)
if(MSVC)
	target_compile_options(superfaiss_tests PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superfaiss_tests PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

enable_testing()
add_test(NAME superfaiss_tests COMMAND superfaiss_tests)
