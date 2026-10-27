find_package(Threads REQUIRED)
add_executable(openmx_basis_density_tests test_density.cpp)
target_link_libraries(openmx_basis_density_tests PRIVATE OpenMX::basis Threads::Threads)
foreach(_case interpolation hartree invalid ownership concurrent allocation)
  add_test(NAME basis-density-${_case} COMMAND openmx_basis_density_tests ${_case})
  set_tests_properties(basis-density-${_case} PROPERTIES LABELS "unit;basis" TIMEOUT 15)
endforeach()

add_executable(openmx_basis_vps_tests test_vps_file.cpp)
target_link_libraries(openmx_basis_vps_tests PRIVATE OpenMX::basis Threads::Threads)
foreach(_case contents metadata invalid lifetime concurrent allocation)
  add_test(NAME basis-vps-${_case} COMMAND openmx_basis_vps_tests ${_case})
  set_tests_properties(basis-vps-${_case} PROPERTIES LABELS "unit;basis" TIMEOUT 15)
endforeach()

add_executable(openmx_basis_pao_tests test_pao_file.cpp)
target_link_libraries(openmx_basis_pao_tests PRIVATE OpenMX::basis Threads::Threads)
foreach(_case contents metadata invalid lifetime concurrent allocation)
  add_test(NAME basis-pao-${_case} COMMAND openmx_basis_pao_tests ${_case})
  set_tests_properties(basis-pao-${_case} PROPERTIES LABELS "unit;basis" TIMEOUT 15)
endforeach()

add_executable(openmx_basis_tests test_radial.cpp)
target_link_libraries(openmx_basis_tests PRIVATE OpenMX::basis Threads::Threads)
foreach(_case radial reciprocal ownership invalid concurrent allocation)
  add_test(NAME basis-${_case} COMMAND openmx_basis_tests ${_case})
  set_tests_properties(basis-${_case} PROPERTIES LABELS "unit;basis" TIMEOUT 15)
endforeach()

# Compile a real C caller, rather than relying on a C++ compiler accepting it.
enable_language(C)
add_executable(openmx_basis_density_c_tests test_density_c.c)
target_link_libraries(openmx_basis_density_c_tests PRIVATE OpenMX::basis)
add_test(NAME basis-density-c-abi COMMAND openmx_basis_density_c_tests)
set_tests_properties(basis-density-c-abi PROPERTIES LABELS "unit;basis" TIMEOUT 15)
add_executable(openmx_basis_c_tests test_c.c)
target_link_libraries(openmx_basis_c_tests PRIVATE OpenMX::basis)
add_test(NAME basis-c-api COMMAND openmx_basis_c_tests)
set_tests_properties(basis-c-api PROPERTIES LABELS "unit;basis" TIMEOUT 15)

add_test(NAME basis-package-consumer COMMAND "${CMAKE_COMMAND}"
  "-DBASIS_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/package-test"
  "-DCXX=${CMAKE_CXX_COMPILER}" "-DCC=${CMAKE_C_COMPILER}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
set_tests_properties(basis-package-consumer PROPERTIES LABELS "package;basis" TIMEOUT 120)
