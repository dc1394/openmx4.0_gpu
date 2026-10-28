cmake_minimum_required(VERSION 3.25.2)
project(OpenMXXcExample LANGUAGES C CXX)
find_package(OpenMXXc 0.1 CONFIG REQUIRED)
add_executable(pbe_example main.cpp)
target_link_libraries(pbe_example PRIVATE OpenMX::xc)
# The package-consumer test also builds the C interface test from an
# installed package; a plain example build leaves it out.
option(OPENMX_XC_EXAMPLE_C_TEST "Also build tests/test_c.c copied next to this file" OFF)
if(OPENMX_XC_EXAMPLE_C_TEST)
  add_executable(c_consumer test_c.c)
  target_link_libraries(c_consumer PRIVATE OpenMX::xc $<$<PLATFORM_ID:Linux>:m>)
  # The inlined kernels agree bitwise with the library only under the same
  # contraction policy (see kernels.h).
  target_compile_options(c_consumer PRIVATE
    "$<$<C_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-ffp-contract=off>")
  set_target_properties(c_consumer PROPERTIES LINKER_LANGUAGE CXX)
endif()
