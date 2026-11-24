# One executable holds the unit tests; the first argument selects the case and
# the second one names a fixture when the case reads one.  The tests of the
# loop of the function table start threads; the library itself does not.
find_package(Threads REQUIRED)
add_executable(openmx_exx_tests test_main.cpp test_kernel.cpp test_two_center.cpp test_overlap.cpp
  test_abf.cpp test_lri.cpp test_system.cpp test_dense_backend.cpp test_exchange.cpp overlap_fixture.cpp
  kbuild_fixture.cpp)
target_link_libraries(openmx_exx_tests PRIVATE OpenMX::exx OpenMX::basis Threads::Threads
  $<$<PLATFORM_ID:Linux>:m>)
target_compile_options(openmx_exx_tests PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>")

set(_exx_fixtures "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")

# 50-digit references written by tools/exx_reference.py --kernels.
add_test(NAME exx-kernel-analytic COMMAND openmx_exx_tests kernel-analytic
  "${_exx_fixtures}/kernel_reference.txt")
set_tests_properties(exx-kernel-analytic PROPERTIES LABELS "unit;exx" TIMEOUT 60)

# Equations (11) and (12) against analytic values for Gaussian functions, the
# symmetries of the two-centre blocks, and their derivative with respect to
# the displacement against central differences.
foreach(_case two-center-gaussian two-center-symmetry two-center-gradient)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case})
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 300)
endforeach()

# The overlap matrix of the main program (tools/exx_overlap_fixture.py).
add_test(NAME exx-two-center-overlap COMMAND openmx_exx_tests two-center-overlap
  "${_exx_fixtures}/overlap_reference.txt")
set_tests_properties(exx-two-center-overlap PROPERTIES LABELS "unit;exx" TIMEOUT 600)

# The auxiliary basis of the PAOs of the overlap fixture.
add_test(NAME exx-abf-pca COMMAND openmx_exx_tests abf-pca "${_exx_fixtures}/overlap_reference.txt")
set_tests_properties(exx-abf-pca PROPERTIES LABELS "unit;exx" TIMEOUT 300)

# The right-hand side of equation (7) against analytic integrals of Gaussian
# functions.
add_test(NAME exx-three-center-gaussian COMMAND openmx_exx_tests three-center-gaussian)
set_tests_properties(exx-three-center-gaussian PROPERTIES LABELS "unit;exx" TIMEOUT 300)

# The local RI with the PAOs of the overlap fixture, and the derivative of
# its coefficients with respect to the displacement of a pair.
foreach(_case lri-identity lri-convergence lri-gradient)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case} "${_exx_fixtures}/overlap_reference.txt")
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 600)
endforeach()

# Neighbour lists and the T and V blocks of the structure of the overlap
# fixture.
foreach(_case system-neighbours system-blocks)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case} "${_exx_fixtures}/overlap_reference.txt")
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 600)
endforeach()

# The matrix products of the function table (section 7.2 of the guide).
add_test(NAME exx-dense-backend COMMAND openmx_exx_tests dense-backend)
set_tests_properties(exx-dense-backend PROPERTIES LABELS "unit;exx" TIMEOUT 60)

# The exchange matrix of the model of tools/exx_reference.py --dump: the
# acceptance tests of milestone M3.
foreach(_case exchange-fixture exchange-layout exchange-gamma exchange-gradient exchange-centre-gradient
    exchange-backend)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case} "${_exx_fixtures}/kbuild_reference.txt")
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 300)
endforeach()

# The exchange matrix with T and V blocks of the PAOs of the overlap fixture,
# and the C interface against the C++ interface; the derivative of E_x with
# respect to the positions of the atoms against central differences.
foreach(_case exchange-system exchange-forces)
  add_test(NAME exx-${_case} COMMAND openmx_exx_tests ${_case} "${_exx_fixtures}/overlap_reference.txt")
  set_tests_properties(exx-${_case} PROPERTIES LABELS "unit;exx" TIMEOUT 600)
endforeach()

# The tests whose loops start threads have the label "threads" as well:
# "ctest -L threads" selects them, for example in a build with
# OPENMX_EXX_THREAD_SANITIZER.
set_property(TEST exx-dense-backend exx-exchange-backend exx-exchange-system APPEND PROPERTY LABELS threads)

# The C interface from a program compiled as C.
add_executable(openmx_exx_c_tests test_c.c)
target_link_libraries(openmx_exx_c_tests PRIVATE OpenMX::exx $<$<PLATFORM_ID:Linux>:m>)
target_compile_options(openmx_exx_c_tests PRIVATE
  "$<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>")
set_target_properties(openmx_exx_c_tests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME exx-c-api COMMAND openmx_exx_c_tests)
set_tests_properties(exx-c-api PROPERTIES LABELS "unit;exx" TIMEOUT 120)

add_test(NAME exx-package-consumer COMMAND "${CMAKE_COMMAND}"
  "-DEXX_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-DBASIS_SOURCE=${_exx_basis_source}"
  "-DBASIS_PACKAGE_DIR=${_exx_basis_package_dir}"
  "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/package-consumer"
  "-DCXX=${CMAKE_CXX_COMPILER}" "-DCC=${CMAKE_C_COMPILER}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
set_tests_properties(exx-package-consumer PROPERTIES LABELS "integration;exx" TIMEOUT 900)
