# Exercise production C metadata/control/cleanup with the actual C++ owner.
# SDK calls are modeled; MPI tests use real four-rank subcommunicators.
function(openmx_add_native_state_tests)
  find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
  get_filename_component(_repo "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  set(_compiler "\"${CMAKE_CXX_COMPILER}\"")
  if(CMAKE_CXX_COMPILER_ARG1)
    string(APPEND _compiler " ${CMAKE_CXX_COMPILER_ARG1}")
  endif()
  # Match the core's toolchain (including its sanitizer runtime/stdlib).
  # The energy runner forces C11 only for its .c input, even with a C++ driver.
  set(_environment "CXX=${_compiler}" "CC=${_compiler}"
    "CXXFLAGS=${CMAKE_CXX_FLAGS}" "CFLAGS=${CMAKE_CXX_FLAGS}")
  if(APPLE)
    list(APPEND _environment "OPENMX_TEST_SANITIZERS=${OPENMX_CORE_SANITIZERS}")
  endif()

  foreach(_solver dc dclno krylov)
    add_test(NAME ${_solver}-calculation-state-lifecycle
      COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_local_solver_state_cpu.py"
        --kind "${_solver}" --core-library "$<TARGET_FILE:openmx_core>")
  endforeach()
  add_test(NAME mixing-h-calculation-state-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_h_state_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  add_test(NAME mixing-rhok-calculation-state-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_rhok_state_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  add_test(NAME mixing-rhok-workspace-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_rhok_workspace_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  add_test(NAME mixing-dm-workspace-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_dm_workspace_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  add_test(NAME mixing-v-workspace-lifecycle
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_v_workspace_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  add_test(NAME mixing-history-native
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_mixing_history_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>")
  foreach(_band_kind col noncol)
    add_test(NAME band-${_band_kind}-calculation-state-lifecycle
      COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_band_state_cpu.py"
        --kind "${_band_kind}" --core-library "$<TARGET_FILE:openmx_core>")
  endforeach()
  foreach(_cluster_kind col noncol)
    add_test(NAME cluster-${_cluster_kind}-calculation-state-lifecycle
      COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_cluster_${_cluster_kind}_state_cpu.py"
        --core-library "$<TARGET_FILE:openmx_core>")
  endforeach()
  set_tests_properties(mixing-history-native mixing-h-calculation-state-lifecycle mixing-rhok-calculation-state-lifecycle mixing-rhok-workspace-lifecycle mixing-dm-workspace-lifecycle mixing-v-workspace-lifecycle dc-calculation-state-lifecycle dclno-calculation-state-lifecycle
    krylov-calculation-state-lifecycle band-col-calculation-state-lifecycle
    band-noncol-calculation-state-lifecycle cluster-col-calculation-state-lifecycle
    cluster-noncol-calculation-state-lifecycle PROPERTIES
      ENVIRONMENT "${_environment}" LABELS "unit;gpu;core;native-state" TIMEOUT 180)

  add_test(NAME solver-noncollinear-energy-output
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_solver_energy_output_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>" --negative-control)
  set_tests_properties(solver-noncollinear-energy-output PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME cwf-noncollinear-projection
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_cwf_projection_cpu.py"
      --negative-control)
  set_tests_properties(cwf-noncollinear-projection PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME cwf-band-helpers
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_cwf_band_helpers_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>" --negative-control)
  set_tests_properties(cwf-band-helpers PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME linear-algebra-link-boundary
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_linear_algebra_link_cpu.py")
  set_tests_properties(linear-algebra-link-boundary PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME total-energy-workspace
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_energy_workspace_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>" --negative-control)
  set_tests_properties(total-energy-workspace PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME total-energy-coulomb
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_energy_coulomb_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>" --negative-control)
  set_tests_properties(total-energy-coulomb PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  add_test(NAME total-energy-xc
    COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_energy_xc_cpu.py"
      --core-library "$<TARGET_FILE:openmx_core>" --negative-control)
  set_tests_properties(total-energy-xc PROPERTIES
    ENVIRONMENT "${_environment}" LABELS "unit;solver;core;native-state" TIMEOUT 180)

  find_program(OPENMX_TEST_MPICXX NAMES mpicxx mpic++)
  find_program(OPENMX_TEST_MPIEXEC NAMES mpirun mpiexec)
  if(OPENMX_TEST_MPICXX AND OPENMX_TEST_MPIEXEC)
    foreach(_solver dclno krylov)
      add_test(NAME ${_solver}-calculation-state-mpi
        COMMAND "${Python3_EXECUTABLE}" "${_repo}/tests/run_local_solver_state_cpu.py"
          --kind "${_solver}" --core-library "$<TARGET_FILE:openmx_core>"
          --mpi-cxx "${OPENMX_TEST_MPICXX}" --mpiexec "${OPENMX_TEST_MPIEXEC}")
      set_tests_properties(${_solver}-calculation-state-mpi PROPERTIES
        ENVIRONMENT "${_environment};OMPI_CXX=${CMAKE_CXX_COMPILER};MPICH_CXX=${CMAKE_CXX_COMPILER};I_MPI_CXX=${CMAKE_CXX_COMPILER}"
        LABELS "unit;gpu;mpi;core;native-state-mpi" TIMEOUT 180 PROCESSORS 4)
    endforeach()
  endif()
endfunction()
