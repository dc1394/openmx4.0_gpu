cmake_minimum_required(VERSION 3.26)
project(OpenMXHIPHostToolchain LANGUAGES NONE)
get_filename_component(OPENMX_REPOSITORY "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${OPENMX_REPOSITORY}/cmake/OpenMXHIPOptions.cmake")
set(OPENMX_BUILD_JOBS 4 CACHE STRING "ScaLAPACK build parallelism")
include("${OPENMX_REPOSITORY}/cmake/OpenMXHIP.cmake")
# Use production discovery and generated make flags; exercise both real and
# complex distributed Cholesky and the C++/Fortran ABI with two MPI ranks.
add_executable(hip_host_toolchain "${OPENMX_REPOSITORY}/tests/cpu_numerics.c"
  "${OPENMX_REPOSITORY}/tests/cpu_toolchain/compiler.cpp"
  "${OPENMX_REPOSITORY}/source/backends/cpu/fortran_runtime.f90")
target_compile_definitions(hip_host_toolchain PRIVATE OPENMX_TEST_SCALAPACK_ONLY OPENMX_TEST_CXX_BOUNDARY)
target_compile_features(hip_host_toolchain PRIVATE cxx_std_20)
target_include_directories(hip_host_toolchain PRIVATE "${OPENMX_REPOSITORY}/source")
target_link_libraries(hip_host_toolchain PRIVATE MPI::MPI_C MPI::MPI_CXX MPI::MPI_Fortran
  ${_hip_complete_math} m ${CMAKE_DL_LIBS} -Wl,--disable-new-dtags)
target_link_options(hip_host_toolchain PRIVATE -Wl,--disable-new-dtags)
set_property(TARGET hip_host_toolchain PROPERTY BUILD_RPATH "${_hip_runtime_dirs}")
if(TARGET openmx_hip_scalapack_build)
  add_dependencies(hip_host_toolchain openmx_hip_scalapack_build)
endif()
enable_testing()
add_test(NAME hip-host-mpi-scalapack COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
  ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:hip_host_toolchain>" ${MPIEXEC_POSTFLAGS})
set_tests_properties(hip-host-mpi-scalapack PROPERTIES TIMEOUT 120 PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")

# CMake's imported MPI target can reorder rpaths and mask a native link error.
# Compile this probe with the generated make flags, in the application's order.
separate_arguments(_native_mpi_compile UNIX_COMMAND "${_hip_C_flags}")
separate_arguments(_native_mpi_link UNIX_COMMAND "${_hip_rpaths} ${_hip_mpi_link} ${_hip_library_dirs} ${_hip_fortran_link}")
set(_native_expected_mpi "")
foreach(library IN LISTS MPI_C_LIBRARIES)
  get_filename_component(name "${library}" NAME)
  if(name MATCHES "^libmpi[.]so([.].*)?$")
    get_filename_component(_native_expected_mpi "${library}" REALPATH)
  endif()
endforeach()
set(_native_mpi_probe "${CMAKE_CURRENT_BINARY_DIR}/native_mpi_runtime")
add_custom_command(OUTPUT "${_native_mpi_probe}"
  COMMAND "${CMAKE_C_COMPILER}" ${_native_mpi_compile}
    "${CMAKE_CURRENT_LIST_DIR}/mpi_runtime.c" ${_native_mpi_link}
    -ldl -Wl,--disable-new-dtags -o "${_native_mpi_probe}"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/mpi_runtime.c" "${_hip_make_config}"
  VERBATIM)
add_custom_target(hip_native_mpi_runtime ALL DEPENDS "${_native_mpi_probe}")
add_test(NAME hip-native-mpi-runtime COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
  ${MPIEXEC_PREFLAGS} "${_native_mpi_probe}" "${_native_expected_mpi}" ${MPIEXEC_POSTFLAGS})
set_tests_properties(hip-native-mpi-runtime PROPERTIES TIMEOUT 60 PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1"
  ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=unset:")
