cmake_minimum_required(VERSION 3.25.2)
project(OpenMXCPUToolchain LANGUAGES NONE)
get_filename_component(OPENMX_REPOSITORY "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
include("${OPENMX_REPOSITORY}/cmake/OpenMXCPUOptions.cmake")
find_package(Threads REQUIRED)
set(OPENMX_BUILD_JOBS 4 CACHE STRING "ScaLAPACK build parallelism")
include("${OPENMX_REPOSITORY}/cmake/OpenMXCPULinalg.cmake")
# A small real numerical probe can test a whole toolchain matrix without
# rebuilding the application or needing FFTW, ELPA, Kokkos or a GPU SDK.
add_executable(cpu_toolchain "${OPENMX_REPOSITORY}/tests/cpu_numerics.c" compiler.cpp
  "${OPENMX_REPOSITORY}/source/backends/cpu/fortran_runtime.f90")
target_compile_definitions(cpu_toolchain PRIVATE OPENMX_TEST_SCALAPACK_ONLY OPENMX_TEST_CXX_BOUNDARY)
target_compile_features(cpu_toolchain PRIVATE cxx_std_20)
target_include_directories(cpu_toolchain PRIVATE "${OPENMX_REPOSITORY}/source")
set_target_properties(cpu_toolchain PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(cpu_toolchain PRIVATE MPI::MPI_C MPI::MPI_CXX MPI::MPI_Fortran
  ${OPENMX_CPU_LINALG_LIBRARIES} m ${CMAKE_DL_LIBS})
if(TARGET openmx_cpu_scalapack_build)
  add_dependencies(cpu_toolchain openmx_cpu_scalapack_build)
endif()
set(_runtime_dirs ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES} ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
foreach(library IN LISTS OPENMX_CPU_LINALG_LIBRARIES MPI_C_LIBRARIES MPI_Fortran_LIBRARIES)
  if(IS_ABSOLUTE "${library}")
    get_filename_component(directory "${library}" DIRECTORY)
    list(APPEND _runtime_dirs "${directory}")
  endif()
endforeach()
set_property(TARGET cpu_toolchain PROPERTY BUILD_RPATH "${_runtime_dirs}")
enable_testing()
add_test(NAME cpu-mpi-scalapack COMMAND "${MPIEXEC_EXECUTABLE}" ${MPIEXEC_NUMPROC_FLAG} 2
  ${MPIEXEC_PREFLAGS} "$<TARGET_FILE:cpu_toolchain>" ${MPIEXEC_POSTFLAGS})
set_tests_properties(cpu-mpi-scalapack PROPERTIES TIMEOUT 120 PROCESSORS 2
  ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1")
