# The frozen Inputtools.c of OpenMX 4.0 is the behavioural reference; it is
# compiled as is (its warnings are its own).
add_executable(openmx_io_tests test_keyword_file.cpp fixtures/Inputtools.c)
set_source_files_properties(fixtures/Inputtools.c PROPERTIES COMPILE_OPTIONS
  "$<$<C_COMPILER_ID:GNU,Clang,AppleClang,IntelLLVM>:-w>")
target_include_directories(openmx_io_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/legacy_include")
target_link_libraries(openmx_io_tests PRIVATE OpenMX::io)
foreach(_case legacy semantics invalid)
  add_test(NAME io-${_case} COMMAND openmx_io_tests ${_case})
  set_tests_properties(io-${_case} PROPERTIES LABELS "unit;io" TIMEOUT 60)
endforeach()

add_executable(openmx_io_basis_spec_tests test_basis_spec.cpp)
target_link_libraries(openmx_io_basis_spec_tests PRIVATE OpenMX::io)
add_test(NAME io-basis-spec COMMAND openmx_io_basis_spec_tests)
set_tests_properties(io-basis-spec PROPERTIES LABELS "unit;io" TIMEOUT 30)

add_executable(openmx_io_c_tests test_c.c)
target_link_libraries(openmx_io_c_tests PRIVATE OpenMX::io)
set_target_properties(openmx_io_c_tests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME io-c-api COMMAND openmx_io_c_tests)
set_tests_properties(io-c-api PROPERTIES LABELS "unit;io" TIMEOUT 30)

add_test(NAME io-package-consumer COMMAND "${CMAKE_COMMAND}"
  "-DIO_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/package-consumer"
  "-DCXX=${CMAKE_CXX_COMPILER}" "-DCC=${CMAKE_C_COMPILER}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
set_tests_properties(io-package-consumer PROPERTIES LABELS "integration;io" TIMEOUT 600)
