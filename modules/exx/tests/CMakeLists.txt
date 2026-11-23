# One executable holds the unit tests; the first argument selects the case and
# the second one names a fixture when the case reads one.
add_executable(openmx_exx_tests test_main.cpp test_kernel.cpp test_two_center.cpp)
target_link_libraries(openmx_exx_tests PRIVATE OpenMX::exx OpenMX::basis $<$<PLATFORM_ID:Linux>:m>)
target_compile_options(openmx_exx_tests PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>")

set(_exx_fixtures "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# 50-digit references written by tools/exx_reference.py --kernels.
add_test(NAME exx-kernel-analytic COMMAND openmx_exx_tests kernel-analytic
  "${_exx_fixtures}/kernel_reference.txt")
set_tests_properties(exx-kernel-analytic PROPERTIES LABELS "unit;exx" TIMEOUT 60)

# Equations (11) and (12) against analytic values for Gaussian functions, and
# the symmetries of the two-centre blocks.
foreach(_case two-center-gaussian two-center-symmetry)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case})
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 300)
endforeach()

add_test(NAME exx-package-consumer COMMAND "${CMAKE_COMMAND}"
  "-DEXX_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-DBASIS_SOURCE=${_exx_basis_source}"
  "-DBASIS_PACKAGE_DIR=${_exx_basis_package_dir}"
  "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/package-consumer"
  "-DCXX=${CMAKE_CXX_COMPILER}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
set_tests_properties(exx-package-consumer PROPERTIES LABELS "integration;exx" TIMEOUT 900)
