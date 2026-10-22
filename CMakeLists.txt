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
if(_core_backend MATCHES "^(CUDA|HIP|SYCL)$" AND NOT _core_backend STREQUAL _backend)
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
  add_test(NAME cluster-col-calculation-state-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_cluster_col_state_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  set_tests_properties(cluster-col-calculation-state-lifecycle PROPERTIES LABELS "unit;gpu;core" TIMEOUT 180)
  add_test(NAME cluster-col-cpu-density
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_cluster_dm_cpu.py")
  set_tests_properties(cluster-col-cpu-density PROPERTIES LABELS "unit;gpu" TIMEOUT 120)
  add_test(NAME cusolver-refinement-architecture-policy
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_cusolver_policy_cpu.py")
  set_tests_properties(cusolver-refinement-architecture-policy PROPERTIES LABELS "unit;gpu" TIMEOUT 120)
  add_test(NAME gpu-vector-packet-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_vector_packet_memory.py")
  set_tests_properties(gpu-vector-packet-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gpu-task-packet-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_task_packet_memory.py")
  set_tests_properties(gpu-task-packet-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gpu-task-cpu-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_task_cpu_memory.py")
  set_tests_properties(gpu-task-cpu-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gpu-vector-capacity-recovery
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_vector_cpu.py")
  set_tests_properties(gpu-vector-capacity-recovery PROPERTIES LABELS "unit;gpu" TIMEOUT 120)
  add_test(NAME gpu-vector-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_vector_memory_cpu.py")
  set_tests_properties(gpu-vector-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gpu-task-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_task_memory_cpu.py")
  set_tests_properties(gpu-task-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gemmul8-cache-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gemmul8_cache_cpu.py")
  set_tests_properties(gemmul8-cache-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gemmul8-architecture-policy
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gemmul8_policy_cpu.py")
  set_tests_properties(gemmul8-architecture-policy PROPERTIES LABELS "unit;gpu" TIMEOUT 120)
  add_test(NAME gpu-worker-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_worker_memory_cpu.py")
  set_tests_properties(gpu-worker-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME gpu-view-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_views_cpu.py")
  set_tests_properties(gpu-view-resource-lifecycle PROPERTIES LABELS "unit;gpu" TIMEOUT 180)
  add_test(NAME eigen-refine-resource-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_eigen_refine_faults.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  set_tests_properties(eigen-refine-resource-lifecycle PROPERTIES LABELS "unit" TIMEOUT 120)
  add_test(NAME build-backend-isolation
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_unified_build.py")
  add_test(NAME environment-registry
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_env_registry_cpu.py")
  set_tests_properties(environment-registry PROPERTIES LABELS "unit" TIMEOUT 120)
  add_test(NAME force-invocation-lifetime
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force_lifetime_cpu.py")
  set_tests_properties(force-invocation-lifetime PROPERTIES LABELS "unit;force"
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
  add_test(NAME force-spin-trace-calls
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force_traces_cpu.py")
  set_tests_properties(force-spin-trace-calls PROPERTIES LABELS "unit;force"
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
  add_test(NAME core-hole-force-views COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_core_hole_force_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(core-hole-force-views PROPERTIES LABELS "unit;force" TIMEOUT 180)
  add_test(NAME projector-force-views COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_projector_force_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(projector-force-views PROPERTIES LABELS "unit;force" TIMEOUT 180)
  add_test(NAME force-hnl-gpu-archives COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_force_hnl_gpu_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(force-hnl-gpu-archives PROPERTIES LABELS "unit;force" TIMEOUT 180)
  add_test(NAME force-hnl-gpu-batches COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_force_hnl_batch_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(force-hnl-gpu-batches PROPERTIES LABELS "unit;force" TIMEOUT 180)
  add_test(NAME force-vna-gpu-resources COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_force_vna_gpu_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(force-vna-gpu-resources PROPERTIES LABELS "unit;force" TIMEOUT 180)
  foreach(_force_fixture IN ITEMS stream case1 case1_turns)
    add_test(NAME "force-vna-${_force_fixture}-packing" COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_force4b_${_force_fixture}_cpu.py")
    set_tests_properties("force-vna-${_force_fixture}-packing" PROPERTIES LABELS "unit;force" TIMEOUT 120)
  endforeach()
  add_test(NAME projector-capacity-recovery COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_projector_recovery_cpu.py")
  set_tests_properties(projector-capacity-recovery PROPERTIES LABELS "unit;force" TIMEOUT 120)
  set_tests_properties(projector-capacity-recovery PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>")
  add_test(NAME nonlocal-invocation-lifetime
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_nonlocal_lifetime_cpu.py")
  set_tests_properties(nonlocal-invocation-lifetime PROPERTIES LABELS "unit;integrals"
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
  add_test(NAME cmake-driver
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_cmake_driver.py")
  add_test(NAME gpu-error-boundary
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_gpu_error_cpu.py")
  add_test(NAME force-recovery-failures
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force_recovery_failures.py")
  set_tests_properties(force-recovery-failures PROPERTIES
    ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" LABELS "unit;force" TIMEOUT 120)
  add_test(NAME mgpu-runtime
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_mgpu_runtime_cpu.py")
  add_test(NAME mgpu-gemm
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_mgpu_gemm_cpu.py")
  set_tests_properties(gpu-error-boundary mgpu-runtime mgpu-gemm PROPERTIES TIMEOUT 180)
  add_test(NAME atomic-hartree-interpolation COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_atomic_hartree_cpu.py")
  set_tests_properties(atomic-hartree-interpolation PROPERTIES TIMEOUT 120)
  add_test(NAME orbital-evaluation COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_orbital_evaluation_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(orbital-evaluation PROPERTIES TIMEOUT 180 LABELS "unit;orbitals")
  add_test(NAME local-basis-contraction COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_local_contraction.py")
  set_tests_properties(local-basis-contraction PROPERTIES TIMEOUT 60 LABELS "unit;orbitals")
  add_test(NAME suite-output COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/test_suite_output.py")
  set_tests_properties(suite-output PROPERTIES TIMEOUT 30 LABELS "unit")
  add_test(NAME hubbard-force-views COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tests/run_hubbard_force_cpu.py" "$<TARGET_FILE:openmx_core>")
  set_tests_properties(hubbard-force-views PROPERTIES TIMEOUT 180 LABELS "unit")
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
    find_program(OPENMX_TEST_MPICXX NAMES mpicxx mpic++)
    find_program(OPENMX_TEST_MPIEXEC NAMES mpirun mpiexec)
    if(OPENMX_TEST_MPICXX AND OPENMX_TEST_MPIEXEC)
      add_test(NAME gpu-vector-packet-mpi COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/tests/run_gpu_vector_packet_mpi.py")
      set_tests_properties(gpu-vector-packet-mpi PROPERTIES
        ENVIRONMENT "MPICC=${OPENMX_TEST_MPICC};MPICXX=${OPENMX_TEST_MPICXX};MPIEXEC=${OPENMX_TEST_MPIEXEC}"
        LABELS "unit;gpu;mpi" TIMEOUT 150 PROCESSORS 6)
      add_test(NAME gpu-task-packet-mpi COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_SOURCE_DIR}/tests/run_gpu_task_packet_mpi.py")
      set_tests_properties(gpu-task-packet-mpi PROPERTIES
        ENVIRONMENT "MPICC=${OPENMX_TEST_MPICC};MPICXX=${OPENMX_TEST_MPICXX};MPIEXEC=${OPENMX_TEST_MPIEXEC}"
        LABELS "unit;gpu;mpi" TIMEOUT 150 PROCESSORS 6)
    endif()
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
    add_test(NAME orbital-spline-boundaries COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_orbital_spline_cpu.py")
    set_tests_properties(orbital-spline-boundaries PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" TIMEOUT 120)
    add_test(NAME orbital-device-tables COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_orbs_grid_boundaries.py")
    add_test(NAME orbital-compact-gathers COMMAND "${Python3_EXECUTABLE}"
      "${CMAKE_SOURCE_DIR}/tests/run_orbs_grid_chunks_cpu.py")
    set_tests_properties(orbital-device-tables orbital-compact-gathers PROPERTIES TIMEOUT 120)
    add_test(NAME force-archive-recovery
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force4b_archived_cpu.py")
    set_tests_properties(force-archive-recovery PROPERTIES
      ENVIRONMENT "OPENMX_CORE_LIBRARY=$<TARGET_FILE:openmx_core>" LABELS "unit;force" TIMEOUT 120)
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
      add_test(NAME hip-force-reductions
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/run_force_reductions_hip.py"
          --rocm "${CMAKE_HIP_COMPILER_ROCM_ROOT}" --cc "${CMAKE_HIP_COMPILER}"
          --arch "${_vector_test_arch}" --mpicc "${OPENMX_TEST_MPICC}")
      set_tests_properties(hip-force-reductions PROPERTIES
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
