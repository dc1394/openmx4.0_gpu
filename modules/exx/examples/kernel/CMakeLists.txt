cmake_minimum_required(VERSION 3.25.2)
project(OpenMXExxExample LANGUAGES CXX)
find_package(OpenMXExx 0.1 CONFIG REQUIRED)
add_executable(kernel_example main.cpp)
target_link_libraries(kernel_example PRIVATE OpenMX::exx)
# The package-consumer test also builds the test of the C interface from an
# installed package; a plain example build leaves it out.
option(OPENMX_EXX_EXAMPLE_C_TEST "Also build tests/test_c.c copied next to this file" OFF)
if(OPENMX_EXX_EXAMPLE_C_TEST)
  enable_language(C)
  add_executable(c_consumer test_c.c)
  target_link_libraries(c_consumer PRIVATE OpenMX::exx $<$<PLATFORM_ID:Linux>:m>)
  set_target_properties(c_consumer PROPERTIES LINKER_LANGUAGE CXX)
endif()
