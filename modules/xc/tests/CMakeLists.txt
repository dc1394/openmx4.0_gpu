# The frozen OpenMX 4.0 routines are the reference of the bitwise tests; they
# are compiled under the library's floating-point contract (no contraction).
set(_xc_legacy fixtures/XC_Ceperly_Alder.c fixtures/XC_CA_LSDA.c fixtures/XC_PW92C.c
  fixtures/XC_EX.c fixtures/XC_PBE.c fixtures/legacy_support.c)
set_source_files_properties(${_xc_legacy} PROPERTIES COMPILE_OPTIONS
  "$<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off;-Wno-unused-but-set-variable;-Wno-unused-variable>;$<$<C_COMPILER_ID:IntelLLVM>:-fp-model=precise;-ffp-contract=off>;$<$<C_COMPILER_ID:MSVC>:/fp:strict>")
add_executable(openmx_xc_tests test_functionals.cpp ${_xc_legacy})
target_include_directories(openmx_xc_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fixtures/legacy_include")
target_link_libraries(openmx_xc_tests PRIVATE OpenMX::xc $<$<PLATFORM_ID:Linux>:m>)
target_compile_options(openmx_xc_tests PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>")
foreach(_case legacy analytic derivatives invalid)
  add_test(NAME xc-${_case} COMMAND openmx_xc_tests ${_case})
  set_tests_properties(xc-${_case} PROPERTIES LABELS "unit;xc" TIMEOUT 60)
endforeach()

add_executable(openmx_xc_c_tests test_c.c)
target_link_libraries(openmx_xc_c_tests PRIVATE OpenMX::xc $<$<PLATFORM_ID:Linux>:m>)
set_target_properties(openmx_xc_c_tests PROPERTIES LINKER_LANGUAGE CXX)
add_test(NAME xc-c-api COMMAND openmx_xc_c_tests)
set_tests_properties(xc-c-api PROPERTIES LABELS "unit;xc" TIMEOUT 30)

add_test(NAME xc-package-consumer COMMAND "${CMAKE_COMMAND}"
  "-DXC_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}/.."
  "-DTEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/package-consumer"
  "-DCXX=${CMAKE_CXX_COMPILER}" "-DCC=${CMAKE_C_COMPILER}"
  -P "${CMAKE_CURRENT_SOURCE_DIR}/package_consumer.cmake")
set_tests_properties(xc-package-consumer PROPERTIES LABELS "integration;xc" TIMEOUT 600)
