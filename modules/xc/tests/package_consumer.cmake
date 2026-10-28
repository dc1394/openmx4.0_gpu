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
file(COPY "${XC_SOURCE}/" DESTINATION "${TEST_ROOT}/source")
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/source" -B "${TEST_ROOT}/build"
  "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_C_COMPILER=${CC}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  "-DCMAKE_INSTALL_PREFIX=${TEST_ROOT}/prefix" -DCMAKE_INSTALL_LIBDIR=lib)
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/build" --config Release --parallel 2)
run("${CMAKE_COMMAND}" --install "${TEST_ROOT}/build" --config Release)
file(RENAME "${TEST_ROOT}/prefix" "${TEST_ROOT}/relocated")
file(RENAME "${TEST_ROOT}/source" "${TEST_ROOT}/hidden-source")
file(RENAME "${TEST_ROOT}/build" "${TEST_ROOT}/hidden-build")

file(COPY "${XC_SOURCE}/examples/pbe/" DESTINATION "${TEST_ROOT}/consumer")
file(COPY "${XC_SOURCE}/tests/test_c.c" DESTINATION "${TEST_ROOT}/consumer")
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/consumer" -B "${TEST_ROOT}/consumer-build"
  "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_C_COMPILER=${CC}" -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/relocated" -DOPENMX_XC_EXAMPLE_C_TEST=ON)
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/consumer-build" --config Release --parallel 2)
run("${TEST_ROOT}/consumer-build/pbe_example")
run("${TEST_ROOT}/consumer-build/c_consumer")
