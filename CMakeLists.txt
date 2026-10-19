cmake_minimum_required(VERSION 3.25.2)
# The production makefiles contain deliberate per-translation-unit compiler
# choices, OpenACC/OpenMP target flags and dependency patches. Keep those rules
# as the build engine, with CMake providing selection, isolation and installation.
project(OpenMXGPU VERSION 4.0 LANGUAGES NONE)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/cuda")
endif()

include(CTest)
option(OPENMX_BUILD_APPLICATION "Build the legacy application and its dependencies" ON)
# The core can be developed and tested on a plain C++ host without the native
# application's MPI, ELPA, CUDA or ROCm stack.
add_subdirectory(core)
if(NOT OPENMX_BUILD_APPLICATION)
  return()
endif()

set(OPENMX_GPU_BACKEND "CUDA" CACHE STRING "GPU backend: CUDA (NVIDIA) or HIP (AMD)")
set_property(CACHE OPENMX_GPU_BACKEND PROPERTY STRINGS CUDA HIP)
string(TOUPPER "${OPENMX_GPU_BACKEND}" _backend)
if(_backend STREQUAL "NVIDIA")
  set(_backend CUDA)
elseif(_backend STREQUAL "AMD")
  set(_backend HIP)
endif()
if(NOT _backend MATCHES "^(CUDA|HIP)$")
  message(FATAL_ERROR "OPENMX_GPU_BACKEND must be CUDA or HIP")
endif()
string(TOUPPER "${OPENMX_CORE_BACKEND}" _core_backend)
if(_core_backend MATCHES "^(CUDA|HIP)$" AND NOT _core_backend STREQUAL _backend)
  message(FATAL_ERROR "The Kokkos accelerator must match OPENMX_GPU_BACKEND")
endif()

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
find_program(OPENMX_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
set(OPENMX_BUILD_JOBS "4" CACHE STRING "Parallel jobs in the native build")
set(OPENMX_GPU_ARCH "" CACHE STRING "GPU architecture (CUDA: 120, 90; HIP: gfx942, gfx1201)")
set(OPENMX_MAKE_OPTIONS "" CACHE STRING "Native make variable assignments separated by semicolons")

set(_native_options "GPUSOLVER2_JOBS=${OPENMX_BUILD_JOBS}" ${OPENMX_MAKE_OPTIONS})
list(APPEND _native_options "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>")
if(_backend STREQUAL "HIP")
  list(APPEND _native_options "OPENMX_SOURCE_ROOT=${CMAKE_SOURCE_DIR}/source")
endif()
if(OPENMX_GPU_ARCH)
  if(_backend STREQUAL "CUDA")
    list(APPEND _native_options "GEMMUL8_GPU_ARCH=${OPENMX_GPU_ARCH}")
  else()
    list(APPEND _native_options "ARCH=${OPENMX_GPU_ARCH}")
  endif()
endif()
foreach(_option IN LISTS _native_options)
  if(NOT _option MATCHES "^[A-Za-z_][A-Za-z_0-9]*=.*$")
    message(FATAL_ERROR "Invalid OPENMX_MAKE_OPTIONS entry: ${_option}. Expected NAME=value.")
  endif()
endforeach()
if(NOT OPENMX_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "OPENMX_BUILD_JOBS must be a positive integer")
endif()

set(_prepare "${CMAKE_SOURCE_DIR}/tools/prepare_build.py")
set(_prepare_command "${Python3_EXECUTABLE}" "${_prepare}"
    --source-root "${CMAKE_SOURCE_DIR}" --build-root "${CMAKE_BINARY_DIR}"
    --backend "${_backend}")
execute_process(COMMAND ${_prepare_command} RESULT_VARIABLE _result
    OUTPUT_VARIABLE _prepared OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_VARIABLE _error)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "${_error}")
endif()
set(_native "${CMAKE_BINARY_DIR}/native/source")
add_custom_target(openmx ALL
    COMMAND ${_prepare_command}
    COMMAND "${OPENMX_MAKE_EXECUTABLE}" -C "${_native}" -j${OPENMX_BUILD_JOBS}
            ${_native_options} openmx
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_native}/openmx" "${CMAKE_BINARY_DIR}/openmx"
    BYPRODUCTS "${CMAKE_BINARY_DIR}/openmx"
    USES_TERMINAL VERBATIM)
add_dependencies(openmx openmx_core)
add_custom_target(native-clean
    COMMAND "${OPENMX_MAKE_EXECUTABLE}" -C "${_native}" ${_native_options} clean
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_BINARY_DIR}/openmx"
    USES_TERMINAL VERBATIM)
install(PROGRAMS "${CMAKE_BINARY_DIR}/openmx" DESTINATION bin)
install(TARGETS openmx_core LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  add_test(NAME build-backend-isolation
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_unified_build.py")
  if(_backend STREQUAL "CUDA")
    add_test(NAME vna-archive-recovery
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_setpro_archived_cpu.py")
  endif()
  if(_backend STREQUAL "HIP")
    foreach(_test IN ITEMS gemmul8_arch_policy nc_mi300a_policy_cpu band_noncol_refine_flow_cpu mi300a_rank_launcher runtest_paths_cpu rocm_runtime_paths_cpu elpa_cptr_interop_cpu deviceless_cleanup_cpu)
      add_test(NAME "hip-${_test}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_BINARY_DIR}/native/tests/run_${_test}.py")
    endforeach()
  endif()
endif()
message(STATUS "OpenMX backend: ${_backend}; native build: ${_native}")
message(STATUS "Native make options: ${_native_options}")
