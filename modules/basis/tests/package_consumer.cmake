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
file(COPY "${BASIS_SOURCE}/tests/test_density_c.c" DESTINATION "${TEST_ROOT}/consumer")
file(COPY "${BASIS_SOURCE}/tests/test_quadrature_c.c" DESTINATION "${TEST_ROOT}/consumer")
configure_file("${BASIS_SOURCE}/examples/pao/main.cpp" "${TEST_ROOT}/consumer/pao_example.cpp" COPYONLY)
configure_file("${BASIS_SOURCE}/examples/vps/main.cpp" "${TEST_ROOT}/consumer/vps_example.cpp" COPYONLY)
configure_file("${BASIS_SOURCE}/examples/density/main.cpp" "${TEST_ROOT}/consumer/density_example.cpp" COPYONLY)
file(WRITE "${TEST_ROOT}/consumer/fixture.vps" [=[
AtomSpecies 1
grid.num.output 6
<project.energies
1
0 -0.5
project.energies>
<Pseudo.Potentials
0 1 -1 2
1 2 -1 2
2 3 -1 2
3 4 -1 2
4 5 -1 2
5 6 -1 2
Pseudo.Potentials>
]=])
file(WRITE "${TEST_ROOT}/consumer/fixture.pao" [=[
AtomSpecies 1
grid.num.output 8
PAO.Lmax 0
PAO.Mul 1
<valence.charge.density
0 1 1
1 2.718281828459045 1
2 7.38905609893065 1
3 20.085536923187668 1
4 54.598150033144236 1
5 148.4131591025766 1
6 403.4287934927351 1
7 1096.6331584284585 1
valence.charge.density>
<pseudo.atomic.orbitals.L=0
0 1 2
1 2.718281828459045 2
2 7.38905609893065 2
3 20.085536923187668 2
4 54.598150033144236 2
5 148.4131591025766 2
6 403.4287934927351 2
7 1096.6331584284585 2
pseudo.atomic.orbitals.L=0>
]=])
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
add_executable(pao_consumer pao_example.cpp)
target_link_libraries(pao_consumer PRIVATE OpenMX::basis)
add_test(NAME pao-consumer COMMAND pao_consumer "${CMAKE_CURRENT_SOURCE_DIR}/fixture.pao")
add_executable(vps_consumer vps_example.cpp)
target_link_libraries(vps_consumer PRIVATE OpenMX::basis)
add_test(NAME vps-consumer COMMAND vps_consumer "${CMAKE_CURRENT_SOURCE_DIR}/fixture.vps")
add_executable(density_consumer density_example.cpp)
target_link_libraries(density_consumer PRIVATE OpenMX::basis)
add_test(NAME density-consumer COMMAND density_consumer "${CMAKE_CURRENT_SOURCE_DIR}/fixture.pao")
add_executable(density_c_consumer test_density_c.c)
target_link_libraries(density_c_consumer PRIVATE OpenMX::basis)
add_test(NAME density-c-consumer COMMAND density_c_consumer)
add_executable(quadrature_c_consumer test_quadrature_c.c)
target_link_libraries(quadrature_c_consumer PRIVATE OpenMX::basis)
add_test(NAME quadrature-c-consumer COMMAND quadrature_c_consumer)
]=])
run("${CMAKE_COMMAND}" -S "${TEST_ROOT}/consumer" -B "${TEST_ROOT}/consumer-build"
  "-DCMAKE_CXX_COMPILER=${CXX}" "-DCMAKE_C_COMPILER=${CC}" -DCMAKE_BUILD_TYPE=Release
  "-DCMAKE_PREFIX_PATH=${TEST_ROOT}/relocated")
run("${CMAKE_COMMAND}" --build "${TEST_ROOT}/consumer-build" --config Release --parallel 2)
run("${CMAKE_CTEST_COMMAND}" --test-dir "${TEST_ROOT}/consumer-build" -C Release --output-on-failure)
message(STATUS "PASS copied basis package, relocated installation, C and C++ consumers")
