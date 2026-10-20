cmake_minimum_required(VERSION 3.25.2)
# The production makefiles contain deliberate per-translation-unit compiler
# choices, OpenACC/OpenMP target flags and dependency patches. Keep those rules
# as the build engine, with CMake providing selection, isolation and installation.
project(OpenMXGPU VERSION 4.0 LANGUAGES NONE)
include(cmake/OpenMXBuildType.cmake)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/cuda")
endif()

include(CTest)
option(OPENMX_BUILD_APPLICATION "Build the legacy application and its dependencies" ON)
set(OPENMX_WORK_DIR "${CMAKE_SOURCE_DIR}/work" CACHE PATH "Directory receiving the built openmx executable")
# A local application install puts openmx directly in work/. Explicit CMake
# prefixes retain the standard bin/ and lib/ layout, including --prefix at
# install time. Keep the managed default in step with OPENMX_WORK_DIR.
set(_openmx_work_install_default OFF)
if(OPENMX_BUILD_APPLICATION)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OR
     (DEFINED OPENMX_MANAGED_WORK_INSTALL_PREFIX AND
      CMAKE_INSTALL_PREFIX STREQUAL OPENMX_MANAGED_WORK_INSTALL_PREFIX))
    set(CMAKE_INSTALL_PREFIX "${OPENMX_WORK_DIR}" CACHE PATH "OpenMX installation prefix" FORCE)
    set(OPENMX_MANAGED_WORK_INSTALL_PREFIX "${OPENMX_WORK_DIR}" CACHE INTERNAL "Local install prefix selected by OpenMX")
    set(_openmx_work_install_default ON)
  else()
    unset(OPENMX_MANAGED_WORK_INSTALL_PREFIX CACHE)
  endif()
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
set(OPENMX_GPU_ARCH "" CACHE STRING "GPU architecture (CUDA: auto via nvidia-smi, sm_80, 90, sm_120; HIP: gfx942)")
include(cmake/OpenMXGPUArchitecture.cmake)
if(OPENMX_BUILD_APPLICATION AND _backend STREQUAL "CUDA")
  openmx_resolve_cuda_architecture()
endif()
# Resolve the architecture before CUDA language / Kokkos configuration, so
# cross-compilation on a login node never needs a device to infer its target.
add_subdirectory(core)
if(NOT OPENMX_BUILD_APPLICATION)
  return()
endif()
string(TOUPPER "${OPENMX_CORE_BACKEND}" _core_backend)
if(_core_backend MATCHES "^(CUDA|HIP)$" AND NOT _core_backend STREQUAL _backend)
  message(FATAL_ERROR "The Kokkos accelerator must match OPENMX_GPU_BACKEND")
endif()

find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
find_program(OPENMX_MAKE_EXECUTABLE NAMES gmake make REQUIRED)
set(OPENMX_BUILD_JOBS "4" CACHE STRING "Parallel jobs in the native build")
set(OPENMX_MAKE_OPTIONS "" CACHE STRING "Native make variable assignments separated by semicolons")
option(OPENMX_COPY_TO_WORK "Copy the successful application build to OPENMX_WORK_DIR" ON)

set(_native_options "GPUSOLVER2_JOBS=${OPENMX_BUILD_JOBS}" ${OPENMX_MAKE_OPTIONS})
list(APPEND _native_options "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>")
if(_backend STREQUAL "HIP")
  list(APPEND _native_options "OPENMX_SOURCE_ROOT=${CMAKE_SOURCE_DIR}/source")
endif()
if(_backend STREQUAL "CUDA" AND OPENMX_CUDA_ARCH)
  list(APPEND _native_options "GEMMUL8_GPU_ARCH=${OPENMX_CUDA_ARCH}")
elseif(_backend STREQUAL "HIP" AND OPENMX_GPU_ARCH)
  list(APPEND _native_options "ARCH=${OPENMX_GPU_ARCH}")
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
set(_copy_to_work)
if(OPENMX_COPY_TO_WORK)
  list(APPEND _copy_to_work COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/copy_openmx.py"
       "${CMAKE_BINARY_DIR}/openmx" "${OPENMX_WORK_DIR}/openmx")
endif()
add_custom_target(openmx ALL
    COMMAND ${_prepare_command}
    COMMAND "${OPENMX_MAKE_EXECUTABLE}" -C "${_native}" -j${OPENMX_BUILD_JOBS}
            ${_native_options} openmx
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_native}/openmx" "${CMAKE_BINARY_DIR}/openmx"
    ${_copy_to_work}
    BYPRODUCTS "${CMAKE_BINARY_DIR}/openmx"
    USES_TERMINAL VERBATIM)
add_dependencies(openmx openmx_core)
add_custom_target(native-clean
    COMMAND "${OPENMX_MAKE_EXECUTABLE}" -C "${_native}" ${_native_options} clean
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${CMAKE_BINARY_DIR}/openmx"
    USES_TERMINAL VERBATIM)
set(_openmx_install_program [=[
  set(_openmx_install_bindir "${CMAKE_INSTALL_PREFIX}/bin")
  if(@_openmx_work_install_default@ AND CMAKE_INSTALL_PREFIX STREQUAL [==[@OPENMX_WORK_DIR@]==])
    set(_openmx_install_bindir "${CMAKE_INSTALL_PREFIX}")
  endif()
  set(_openmx_installed_program "${_openmx_install_bindir}/openmx")
  execute_process(COMMAND [==[@Python3_EXECUTABLE@]==]
      [==[@CMAKE_SOURCE_DIR@/tools/copy_openmx.py]==]
      [==[@CMAKE_BINARY_DIR@/openmx]==] "$ENV{DESTDIR}${_openmx_installed_program}"
      RESULT_VARIABLE _openmx_copy_result ERROR_VARIABLE _openmx_copy_error)
  if(NOT "${_openmx_copy_result}" STREQUAL "0")
    message(FATAL_ERROR "Cannot install openmx: ${_openmx_copy_error}")
  endif()
  message(STATUS "Installing: $ENV{DESTDIR}${_openmx_installed_program}")
  list(APPEND CMAKE_INSTALL_MANIFEST_FILES "${_openmx_installed_program}")
]=])
string(CONFIGURE "${_openmx_install_program}" _openmx_install_program @ONLY)
install(CODE "${_openmx_install_program}")
install(TARGETS openmx_core LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  add_test(NAME build-backend-isolation
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_unified_build.py")
  add_test(NAME cmake-driver
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_cmake_driver.py")
  add_test(NAME atomic-hartree-interpolation COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_atomic_hartree_cpu.py")
  set_tests_properties(atomic-hartree-interpolation PROPERTIES TIMEOUT 120)
  add_test(NAME dftd3-interpolation
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_dftd3_interpolation_cpu.py")
  set(_dense_kernel_test "${CMAKE_SOURCE_DIR}/tests/run_gpu_dense_workers_cpu.py")
  set(_worker_admission_test "${CMAKE_SOURCE_DIR}/tests/run_gpu_workers_cpu.py")
  set(_integral_thread_test "${CMAKE_SOURCE_DIR}/tests/run_integral_threading_cpu.py")
  set(_vna_workspace_test "${CMAKE_SOURCE_DIR}/tests/run_vna_workspace_cpu.py")
  if(_backend STREQUAL "HIP")
    set(_dense_kernel_test "${CMAKE_BINARY_DIR}/native/tests/run_gpu_dense_workers_cpu.py")
    set(_worker_admission_test "${CMAKE_BINARY_DIR}/native/tests/run_gpu_workers_cpu.py")
    set(_integral_thread_test "${CMAKE_BINARY_DIR}/native/tests/run_integral_threading_cpu.py")
  endif()
  add_test(NAME dense-kernel-indexing COMMAND "${Python3_EXECUTABLE}" "${_dense_kernel_test}")
  add_test(NAME gpu-worker-admission COMMAND "${Python3_EXECUTABLE}" "${_worker_admission_test}")
  add_test(NAME integral-threading COMMAND "${Python3_EXECUTABLE}" "${_integral_thread_test}")
  set_tests_properties(integral-threading PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
  add_test(NAME vna-workspace-integrals COMMAND "${Python3_EXECUTABLE}" "${_vna_workspace_test}")
  set_tests_properties(vna-workspace-integrals PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;OPENMX_VNA_SOURCE=${_native}/Set_ProExpn_VNA.c" TIMEOUT 120)
  add_test(NAME vna-preparation COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_vna_preparation_cpu.py")
  set_tests_properties(vna-preparation PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;OPENMX_VNA_SOURCE=${_native}/Set_ProExpn_VNA.c" TIMEOUT 120)
  add_test(NAME dsvna-workspace-integrals COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_dsvna_workspace_cpu.py")
  set_tests_properties(dsvna-workspace-integrals PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;OPENMX_VNA_SOURCE=${_native}/Set_ProExpn_VNA.c" TIMEOUT 120)
  add_test(NAME vna-stage-status COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_vna_status_cpu.py")
  set_tests_properties(vna-stage-status PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
  add_test(NAME hvna-workspace-integrals COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_hvna_workspace_cpu.py")
  set_tests_properties(hvna-workspace-integrals PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;OPENMX_VNA_SOURCE=${_native}/Set_ProExpn_VNA.c" TIMEOUT 180)
  find_program(OPENMX_TEST_MPICC NAMES mpicc)
  if(OPENMX_TEST_MPICC)
    add_test(NAME projector-exchange-mpi COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_projector_exchange_mpi.py")
    set_tests_properties(projector-exchange-mpi PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;MPICC=${OPENMX_TEST_MPICC}"
      TIMEOUT 200 PROCESSORS 4)
    add_test(NAME vna-exchange-mpi COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_vna_exchange_mpi.py")
    set_tests_properties(vna-exchange-mpi PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;MPICC=${OPENMX_TEST_MPICC}"
      TIMEOUT 380 PROCESSORS 4)
    add_test(NAME reciprocal-archive-mpi COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_reciprocal_archive_mpi.py")
    set_tests_properties(reciprocal-archive-mpi PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>;MPICC=${OPENMX_TEST_MPICC}"
      TIMEOUT 200 PROCESSORS 4)
  endif()
  if(_backend STREQUAL "CUDA")
    add_test(NAME force-archive-recovery
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force4b_archived_cpu.py")
    add_test(NAME vna-archive-recovery
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_setpro_archived_cpu.py")
    add_test(NAME vna-batch-budget
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_setpro_batch_cpu.py")
  endif()
  if(_backend STREQUAL "HIP")
    if(OPENMX_TEST_MPICC AND _core_backend STREQUAL "HIP")
      string(JOIN "," _vector_test_arch ${CMAKE_HIP_ARCHITECTURES})
      add_test(NAME hip-vector-reductions
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_vector_reductions_hip.py"
          --rocm "${CMAKE_HIP_COMPILER_ROCM_ROOT}" --cc "${CMAKE_HIP_COMPILER}"
          --arch "${_vector_test_arch}" --mpicc "${OPENMX_TEST_MPICC}")
      set_tests_properties(hip-vector-reductions PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_DYNAMIC=FALSE"
        LABELS "gpu;integration" TIMEOUT 240)
      add_test(NAME hip-setham-capacity
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_setham_capacity_hip.py"
          --rocm "${CMAKE_HIP_COMPILER_ROCM_ROOT}" --cc "${CMAKE_HIP_COMPILER}"
          --arch "${_vector_test_arch}" --mpicc "${OPENMX_TEST_MPICC}")
      set_tests_properties(hip-setham-capacity PROPERTIES
        ENVIRONMENT "OMP_NUM_THREADS=1;OMP_DYNAMIC=FALSE"
        LABELS "gpu;integration" TIMEOUT 240)
    endif()
    add_test(NAME hip-vna-archive
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_BINARY_DIR}/native/tests/run_setpro_hvna_cpu.py" --sanitize)
    set_tests_properties(hip-vna-archive PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
    foreach(_test IN ITEMS gemmul8_arch_policy nc_mi300a_policy_cpu band_noncol_refine_flow_cpu mi300a_rank_launcher runtest_paths_cpu rocm_runtime_paths_cpu elpa_cptr_interop_cpu deviceless_cleanup_cpu)
      add_test(NAME "hip-${_test}"
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_BINARY_DIR}/native/tests/run_${_test}.py")
    endforeach()
  endif()
endif()
message(STATUS "OpenMX backend: ${_backend}; native build: ${_native}")
message(STATUS "Native make options: ${_native_options}")
