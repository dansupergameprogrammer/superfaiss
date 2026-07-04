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
	src/compose.cpp
	src/pca.cpp
	src/scratch.cpp
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
	# GCC/Clang default to the SSE2 baseline on x86; the SSE kernels are SSE4.1.
	# (The AVX2 TU needs no flags — its functions carry target("avx2,fma") attributes.)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686")
		set_source_files_properties(src/kernels.cpp PROPERTIES COMPILE_OPTIONS "-msse4.2")
	endif()
endif()

add_executable(superfaiss_tests tests/test_main.cpp)
find_package(Threads REQUIRED)
target_link_libraries(superfaiss_tests PRIVATE superfaiss Threads::Threads)
if(MSVC)
	target_compile_options(superfaiss_tests PRIVATE /W4 /fp:precise)
else()
	target_compile_options(superfaiss_tests PRIVATE -Wall -Wextra -ffp-contract=off)
endif()

enable_testing()
add_test(NAME superfaiss_tests COMMAND superfaiss_tests)
