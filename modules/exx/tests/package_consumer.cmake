# Exercise the public boundary with no source/build tree available at the
# original location. A copied module must build without the rest of OpenMX,
# against an installed basis package.
function(run)
  execute_process(COMMAND ${ARGV} RESULT_VARIABLE status
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status STREQUAL "0")
    message(FATAL_ERROR "${ARGV}\n${output}\n${error}")
  endif()
endfunction()

file(REMOVE_RECURSE "${TEST_ROOT}")
file(MAKE_DIRECTORY "${TEST_ROOT}")

# The basis package: build and install a copy of its sources, or use the
# installed package this build already found.
set(basis_arguments "")
if(BASIS_SOURCE)
  file(COPY "${BASIS_SOURCE}/" DESTINATION "${TEST_ROOT}/basis-source")
  run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/basis-source" -B "${TEST_ROOT}/basis-build"
    "-DCMAKE_CXX_COMPILER=${CXX}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
    "-DCMAKE_INSTALL_PREFIX=${TEST_ROOT}/prefix" -DCMAKE_INSTALL_LIBDIR=lib)
  run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/basis-build" --config Release --parallel 2)
  run("${CMAKE_COMMAND}" --install "${TEST_ROOT}/basis-build" --config Release)
  file(RENAME "${TEST_ROOT}/basis-source" "${TEST_ROOT}/hidden-basis-source")
  file(RENAME "${TEST_ROOT}/basis-build" "${TEST_ROOT}/hidden-basis-build")
elseif(BASIS_PACKAGE_DIR)
  set(basis_arguments "-DOpenMXBasis_DIR=${BASIS_PACKAGE_DIR}")
else()
  message(FATAL_ERROR "the package-consumer test needs the basis sources or an installed basis package")
endif()

file(COPY "${EXX_SOURCE}/" DESTINATION "${TEST_ROOT}/source")
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/source" -B "${TEST_ROOT}/build"
  "-DCMAKE_CXX_COMPILER=${CXX}" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
  "-DCMAKE_INSTALL_PREFIX=${TEST_ROOT}/prefix" -DCMAKE_INSTALL_LIBDIR=lib
  "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/prefix" ${basis_arguments})
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/build" --config Release --parallel 2)
run("${CMAKE_COMMAND}" --install "${TEST_ROOT}/build" --config Release)
file(RENAME "${TEST_ROOT}/prefix" "${TEST_ROOT}/relocated")
file(RENAME "${TEST_ROOT}/source" "${TEST_ROOT}/hidden-source")
file(RENAME "${TEST_ROOT}/build" "${TEST_ROOT}/hidden-build")

file(COPY "${EXX_SOURCE}/examples/kernel/" DESTINATION "${TEST_ROOT}/consumer")
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/consumer" -B "${TEST_ROOT}/consumer-build"
  "-DCMAKE_CXX_COMPILER=${CXX}" -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/relocated" ${basis_arguments})
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/consumer-build" --config Release --parallel 2)
run("${TEST_ROOT}/consumer-build/kernel_example")
