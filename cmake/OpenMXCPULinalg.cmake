# Concrete LP64 library paths are shared by the application and ELPA's
# Autoconf probes. Do not let FindBLAS choose a provider from the login shell.
set(OPENMX_CPU_LINALG_LIBRARIES "" CACHE STRING "CUSTOM LP64 ScaLAPACK, LAPACK and BLAS libraries (semicolon-separated)")
set(OPENMX_CPU_MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "oneMKL prefix")
set(OPENMX_CPU_OPENBLAS_ROOT "" CACHE PATH "OpenBLAS prefix (empty searches system paths)")
set(OPENMX_CPU_SCALAPACK_ROOT "" CACHE PATH "ScaLAPACK prefix built for the selected MPI")
option(OPENMX_CPU_BUILD_SCALAPACK "Build matching Netlib ScaLAPACK if no compatible installation is found" ON)
set(OPENMX_CPU_SCALAPACK_ARCHIVE "" CACHE FILEPATH "Offline ScaLAPACK 2.2.3 source archive; empty downloads the pinned release")
set(_cpu_math_roots "${OPENMX_CPU_MKL_ROOT}|${OPENMX_CPU_OPENBLAS_ROOT}|${OPENMX_CPU_SCALAPACK_ROOT}")
if(DEFINED OPENMX_CPU_CONFIGURED_MATH_ROOTS AND
   NOT OPENMX_CPU_CONFIGURED_MATH_ROOTS STREQUAL _cpu_math_roots)
  message(FATAL_ERROR "CPU numerical-library prefixes changed. Use a new build directory to avoid reusing libraries from the previous prefix.")
endif()
set(OPENMX_CPU_LINALG_RESOLVED "${OPENMX_CPU_LINALG}")
if(OPENMX_CPU_LINALG STREQUAL "AUTO" AND OPENMX_CPU_LINALG_LIBRARIES)
  set(OPENMX_CPU_LINALG_RESOLVED CUSTOM)
elseif(OPENMX_CPU_LINALG STREQUAL "AUTO")
  if(OPENMX_CPU_OPENBLAS_ROOT)
    find_library(OPENMX_CPU_OPENBLAS_LIBRARY NAMES openblas
      PATHS "${OPENMX_CPU_OPENBLAS_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
  else()
    find_library(OPENMX_CPU_OPENBLAS_LIBRARY NAMES openblas)
  endif()
  if(OPENMX_CPU_OPENBLAS_LIBRARY)
    set(OPENMX_CPU_LINALG_RESOLVED OPENBLAS)
  else()
    set(OPENMX_CPU_LINALG_RESOLVED MKL)
  endif()
endif()
if(OPENMX_CPU_LINALG_RESOLVED STREQUAL "CUSTOM")
  if(NOT OPENMX_CPU_LINALG_LIBRARIES)
    message(FATAL_ERROR "OPENMX_CPU_LINALG=CUSTOM requires OPENMX_CPU_LINALG_LIBRARIES")
  endif()
elseif(OPENMX_CPU_LINALG_LIBRARIES)
  message(FATAL_ERROR "Use OPENMX_CPU_LINALG=CUSTOM for OPENMX_CPU_LINALG_LIBRARIES, or leave that variable empty for MKL/OPENBLAS selection.")
elseif(OPENMX_CPU_LINALG_RESOLVED STREQUAL "MKL")
  # The Fortran ABI follows the Fortran compiler, independently of C/C++.
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(_cpu_mkl_interface mkl_gf_lp64)
  else()
    set(_cpu_mkl_interface mkl_intel_lp64)
  endif()
  if(OPENMX_CPU_MPI_RESOLVED STREQUAL "OPENMPI")
    set(_cpu_mkl_blacs mkl_blacs_openmpi_lp64)
  else()
    set(_cpu_mkl_blacs mkl_blacs_intelmpi_lp64 mkl_blacs_lp64)
  endif()
  foreach(component SCALAPACK INTERFACE SEQUENTIAL CORE BLACS)
    if(component STREQUAL "INTERFACE")
      set(names "${_cpu_mkl_interface}")
    elseif(component STREQUAL "BLACS")
      set(names ${_cpu_mkl_blacs})
    elseif(component STREQUAL "SCALAPACK")
      set(names mkl_scalapack_lp64)
    else()
      string(TOLOWER "${component}" lower)
      set(names "mkl_${lower}")
    endif()
    if(OPENMX_CPU_MKL_ROOT)
      find_library(OPENMX_CPU_MKL_${component}_LIBRARY NAMES ${names}
        PATHS "${OPENMX_CPU_MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib64
        NO_DEFAULT_PATH REQUIRED)
    else()
      find_library(OPENMX_CPU_MKL_${component}_LIBRARY NAMES ${names} REQUIRED)
    endif()
  endforeach()
  # A rescan group also supports static MKL's circular BLAS dependencies.
  # Sequential MKL avoids loading libgomp and libiomp5 in the same process;
  # parallelism remains in OpenMX's MPI/OpenMP and ELPA kernels.
  set(OPENMX_CPU_LINALG_LIBRARIES
    "-Wl,--start-group" "${OPENMX_CPU_MKL_SCALAPACK_LIBRARY}" "${OPENMX_CPU_MKL_INTERFACE_LIBRARY}"
    "${OPENMX_CPU_MKL_SEQUENTIAL_LIBRARY}" "${OPENMX_CPU_MKL_CORE_LIBRARY}"
    "${OPENMX_CPU_MKL_BLACS_LIBRARY}" "-Wl,--end-group" pthread m ${CMAKE_DL_LIBS})
else()
  if(OPENMX_CPU_OPENBLAS_ROOT)
    find_library(OPENMX_CPU_OPENBLAS_LIBRARY NAMES openblas
      PATHS "${OPENMX_CPU_OPENBLAS_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
  else()
    find_library(OPENMX_CPU_OPENBLAS_LIBRARY NAMES openblas REQUIRED)
  endif()
  if(NOT CMAKE_CROSSCOMPILING)
    include(CheckCSourceRuns)
    set(CMAKE_REQUIRED_LIBRARIES "${OPENMX_CPU_OPENBLAS_LIBRARY}")
    check_c_source_runs("#include <string.h>\nextern char *openblas_get_config(void);\nint main(void) { return strstr(openblas_get_config(), \"USE64BITINT\") != 0; }"
      OPENMX_CPU_OPENBLAS_IS_LP64)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(NOT OPENMX_CPU_OPENBLAS_IS_LP64)
      message(FATAL_ERROR "OPENMX_CPU_OPENBLAS_LIBRARY must be a working LP64 OpenBLAS (32-bit Fortran integers), not an ILP64 library. See the configure test log.")
    endif()
  endif()
  # Unqualified system ScaLAPACK may have been built against another MPI.
  # Only use it with an explicit prefix/library supplied by the caller.
  if(OPENMX_CPU_SCALAPACK_ROOT)
    find_library(OPENMX_CPU_SCALAPACK_LIBRARY NAMES scalapack scalapack-openmpi scalapack-intelmpi
      PATHS "${OPENMX_CPU_SCALAPACK_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
  elseif(OPENMX_CPU_MPI_RESOLVED STREQUAL "OPENMPI")
    find_library(OPENMX_CPU_SCALAPACK_LIBRARY NAMES scalapack-openmpi)
  endif()
  if(NOT OPENMX_CPU_SCALAPACK_LIBRARY)
    if(NOT OPENMX_CPU_BUILD_SCALAPACK)
      message(FATAL_ERROR "No ScaLAPACK for ${OPENMX_CPU_MPI_RESOLVED}. Set OPENMX_CPU_SCALAPACK_LIBRARY/ROOT or enable OPENMX_CPU_BUILD_SCALAPACK.")
    endif()
    if(CMAKE_VERSION VERSION_LESS 3.26)
      message(FATAL_ERROR "Building ScaLAPACK 2.2.3 requires CMake 3.26 or later. Supply an existing OPENMX_CPU_SCALAPACK_LIBRARY or update CMake.")
    endif()
    include(ExternalProject)
    set(_cpu_scalapack_url https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.3.tar.gz)
    if(OPENMX_CPU_SCALAPACK_ARCHIVE)
      set(_cpu_scalapack_url "${OPENMX_CPU_SCALAPACK_ARCHIVE}")
    endif()
    set(_cpu_scalapack_build "${CMAKE_BINARY_DIR}/cpu-scalapack")
    set(OPENMX_CPU_SCALAPACK_LIBRARY "${_cpu_scalapack_build}/lib/libscalapack.a")
    ExternalProject_Add(openmx_cpu_scalapack_build
      URL "${_cpu_scalapack_url}"
      URL_HASH SHA256=5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE UPDATE_COMMAND ""
      BINARY_DIR "${_cpu_scalapack_build}"
      CMAKE_ARGS "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
        "-DCMAKE_Fortran_COMPILER:FILEPATH=${CMAKE_Fortran_COMPILER}"
        "-DMPI_C_COMPILER:FILEPATH=${MPI_C_COMPILER}" "-DMPI_Fortran_COMPILER:FILEPATH=${MPI_Fortran_COMPILER}"
        "-DMPIEXEC_EXECUTABLE:FILEPATH=${MPIEXEC_EXECUTABLE}"
        "-DCMAKE_BUILD_TYPE:STRING=Release" "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
        "-DBUILD_SHARED_LIBS:BOOL=OFF" "-DSCALAPACK_BUILD_TESTS:BOOL=OFF"
        "-DBLAS_LIBRARIES:STRING=${OPENMX_CPU_OPENBLAS_LIBRARY}"
        "-DLAPACK_LIBRARIES:STRING=${OPENMX_CPU_OPENBLAS_LIBRARY}"
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target scalapack --parallel ${OPENMX_BUILD_JOBS}
      INSTALL_COMMAND "" BUILD_BYPRODUCTS "${OPENMX_CPU_SCALAPACK_LIBRARY}")
    message(STATUS "Building ScaLAPACK 2.2.3 for ${OPENMX_CPU_MPI_RESOLVED}")
  endif()
  set(OPENMX_CPU_LINALG_LIBRARIES "${OPENMX_CPU_SCALAPACK_LIBRARY}" "${OPENMX_CPU_OPENBLAS_LIBRARY}")
endif()
set(_cpu_math_signature "${OPENMX_CPU_LINALG_RESOLVED}|${OPENMX_CPU_LINALG_LIBRARIES}")
if(DEFINED OPENMX_CPU_CONFIGURED_MATH_LIBRARIES AND
   NOT OPENMX_CPU_CONFIGURED_MATH_LIBRARIES STREQUAL _cpu_math_signature)
  message(FATAL_ERROR "CPU numerical libraries changed. Use a new build directory so ELPA and ScaLAPACK are rebuilt for the selected libraries.")
endif()
set(OPENMX_CPU_CONFIGURED_MATH_ROOTS "${_cpu_math_roots}" CACHE INTERNAL "Configured CPU numerical-library prefixes")
set(OPENMX_CPU_CONFIGURED_MATH_LIBRARIES "${_cpu_math_signature}" CACHE INTERNAL "Configured CPU numerical libraries")
message(STATUS "CPU numerical libraries: ${OPENMX_CPU_LINALG_RESOLVED} (LP64), MPI ${OPENMX_CPU_MPI_RESOLVED}")
