cmake_minimum_required(VERSION 3.25.2)
project(OpenMXIoExample LANGUAGES C CXX)
find_package(OpenMXIo 0.1 CONFIG REQUIRED)
add_executable(keywords_example main.cpp)
target_link_libraries(keywords_example PRIVATE OpenMX::io)
# The package-consumer test also builds the C interface test from an
# installed package; a plain example build leaves it out.
option(OPENMX_IO_EXAMPLE_C_TEST "Also build tests/test_c.c copied next to this file" OFF)
if(OPENMX_IO_EXAMPLE_C_TEST)
  add_executable(c_consumer test_c.c)
  target_link_libraries(c_consumer PRIVATE OpenMX::io)
  set_target_properties(c_consumer PROPERTIES LINKER_LANGUAGE CXX)
endif()
