cmake_minimum_required(VERSION 3.16)
project(superfaiss CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(superfaiss STATIC
	src/kernels.cpp
	src/kernels_avx2.cpp
	src/alloc.cpp
	src/validate.cpp
	src/bake.cpp
	src/query.cpp
)
target_include_directories(superfaiss PUBLIC include)

# Determinism: forbid FP contraction (implicit fused multiply-add would change rounding
# and break scalar/SIMD bit-equality; the AVX2 path uses FMA *explicitly*, mirrored by
# std::fma).
if(MSVC)
	target_compile_options(superfaiss PRIVATE /W4 /fp:precise)
	# VEX-encode the whole AVX2 TU so no SSE/AVX transition penalties occur. The
	# functions are runtime-dispatched; the TU is never executed on non-AVX2 CPUs.
	set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
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
