# Exercise the public boundary with no source/build tree available at the
# original location. A copied module must build without the rest of OpenMX.
function(run)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "${ARGV}\n${output}\n${error}")
  endif()
endfunction()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")
file(COPY "${BASIS_SOURCE}/" DESTINATION "${TEST_ROOT}/source")
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/source" -B "${TEST_ROOT}/build"
  "-DCMAKE_CXX_COMPILER=${CXX}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  "-DCMAKE_INSTALL_PREFIX=${TEST_ROOT}/prefix" -DCMAKE_INSTALL_LIBDIR=lib)
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/build" --config Release --parallel 2)
run("${CMAKE_COMMAND}" --install "${TEST_ROOT}/build" --config Release)
file(RENAME "${TEST_ROOT}/prefix" "${TEST_ROOT}/relocated")
file(RENAME "${TEST_ROOT}/source" "${TEST_ROOT}/hidden-source")
file(RENAME "${TEST_ROOT}/build" "${TEST_ROOT}/hidden-build")

file(COPY "${BASIS_SOURCE}/examples/radial/" DESTINATION "${TEST_ROOT}/consumer")
file(COPY "${BASIS_SOURCE}/tests/test_c.c" DESTINATION "${TEST_ROOT}/consumer")
file(APPEND "${TEST_ROOT}/consumer/CMakeLists.txt" [=[
enable_language(C)
add_executable(c_consumer test_c.c)
target_link_libraries(c_consumer PRIVATE OpenMX::basis)
get_target_property(dependencies OpenMX::basis INTERFACE_LINK_LIBRARIES)
if(dependencies)
  message(FATAL_ERROR "Basis unexpectedly requires other libraries: ${dependencies}")
endif()
enable_testing()
add_test(NAME cpp-consumer COMMAND radial_example)
add_test(NAME c-consumer COMMAND c_consumer)
]=])
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/consumer" -B "${TEST_ROOT}/consumer-build"
  "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_C_COMPILER=${CC}" -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/relocated")
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/consumer-build" --config Release --parallel 2)
run("${CMAKE_CTEST_COMMAND}" --test-dir "${TEST_ROOT}/consumer-build" -C Release --output-on-failure)
message(STATUS "PASS copied basis package, relocated installation, C and C++ consumers")
