# Shared LP64 providers. Keep the CPU and HIP public cache namespaces separate.
macro(openmx_configure_linalg _math_prefix _math_target _math_directory)
  # Concrete LP64 library paths are shared by the application and ELPA's
  # Autoconf probes. Do not let FindBLAS choose a provider from the login shell.
  set(${_math_prefix}_LINALG_LIBRARIES "" CACHE STRING "CUSTOM LP64 ScaLAPACK, LAPACK and BLAS libraries (semicolon-separated)")
  set(${_math_prefix}_MKL_ROOT "$ENV{MKLROOT}" CACHE PATH "oneMKL prefix")
  set(${_math_prefix}_OPENBLAS_ROOT "" CACHE PATH "OpenBLAS prefix (empty searches system paths)")
  set(${_math_prefix}_SCALAPACK_ROOT "" CACHE PATH "ScaLAPACK prefix built for the selected MPI")
  option(${_math_prefix}_BUILD_SCALAPACK "Build matching Netlib ScaLAPACK if no compatible installation is found" ON)
  set(${_math_prefix}_SCALAPACK_ARCHIVE "" CACHE FILEPATH "Offline ScaLAPACK 2.2.3 source archive; empty downloads the pinned release")
  set(${_math_prefix}_AOCL_ROOT "$ENV{AOCL_ROOT}" CACHE PATH "AOCL LP64 installation prefix")
  set(${_math_prefix}_MKL_THREADING "SEQUENTIAL" CACHE STRING
      "MKL threading layer: SEQUENTIAL, or INTEL (mkl_intel_thread) when C, C++ and Fortran all use Intel OpenMP")
  set_property(CACHE ${_math_prefix}_MKL_THREADING PROPERTY STRINGS SEQUENTIAL INTEL)
  set(_host_math_roots "${${_math_prefix}_MKL_ROOT}|${${_math_prefix}_OPENBLAS_ROOT}|${${_math_prefix}_SCALAPACK_ROOT}")
  if("${_math_prefix}" STREQUAL "OPENMX_HIP" OR ${_math_prefix}_LINALG STREQUAL "AOCL")
    string(APPEND _host_math_roots "|${${_math_prefix}_AOCL_ROOT}")
  endif()
  if(DEFINED ${_math_prefix}_CONFIGURED_MATH_ROOTS AND
     NOT ${_math_prefix}_CONFIGURED_MATH_ROOTS STREQUAL _host_math_roots)
    message(FATAL_ERROR "Host numerical-library prefixes changed. Use a new build directory to avoid reusing libraries from the previous prefix.")
  endif()
  set(${_math_prefix}_LINALG_RESOLVED "${${_math_prefix}_LINALG}")
  if(${_math_prefix}_LINALG STREQUAL "AUTO" AND ${_math_prefix}_LINALG_LIBRARIES)
    set(${_math_prefix}_LINALG_RESOLVED CUSTOM)
  elseif(${_math_prefix}_LINALG STREQUAL "AUTO")
    if(${_math_prefix}_OPENBLAS_ROOT)
      find_library(${_math_prefix}_OPENBLAS_LIBRARY NAMES openblas
        PATHS "${${_math_prefix}_OPENBLAS_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
    else()
      find_library(${_math_prefix}_OPENBLAS_LIBRARY NAMES openblas)
    endif()
    if(${_math_prefix}_OPENBLAS_LIBRARY)
      set(${_math_prefix}_LINALG_RESOLVED OPENBLAS)
    else()
      set(${_math_prefix}_LINALG_RESOLVED MKL)
    endif()
  endif()
  if(${_math_prefix}_LINALG_RESOLVED STREQUAL "CUSTOM")
    if(NOT ${_math_prefix}_LINALG_LIBRARIES)
      message(FATAL_ERROR "${_math_prefix}_LINALG=CUSTOM requires ${_math_prefix}_LINALG_LIBRARIES")
    endif()
  elseif(${_math_prefix}_LINALG_LIBRARIES)
    message(FATAL_ERROR "Use ${_math_prefix}_LINALG=CUSTOM for ${_math_prefix}_LINALG_LIBRARIES, or leave that variable empty for MKL/OPENBLAS selection.")
  elseif(${_math_prefix}_LINALG_RESOLVED STREQUAL "MKL")
    # The Fortran ABI follows the Fortran compiler, independently of C/C++.
    if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
      set(_host_math_mkl_interface mkl_gf_lp64)
    else()
      set(_host_math_mkl_interface mkl_intel_lp64)
    endif()
    if(${_math_prefix}_MPI_RESOLVED STREQUAL "OPENMPI")
      set(_host_math_mkl_blacs mkl_blacs_openmpi_lp64)
    else()
      set(_host_math_mkl_blacs mkl_blacs_intelmpi_lp64 mkl_blacs_lp64)
    endif()
    # Sequential MKL avoids loading libgomp and libiomp5 in the same process.
    # An all-Intel toolchain already uses libiomp5, so INTEL links
    # mkl_intel_thread with it: the BLAS that ScaLAPACK and ELPA call outside
    # OpenMX's parallel regions then uses the rank's OpenMP threads, while
    # calls inside those regions stay single-threaded (MKL's nested default).
    set(_host_math_mkl_runtime)
    if(${_math_prefix}_MKL_THREADING STREQUAL "INTEL")
      foreach(_host_math_language C CXX Fortran)
        if(NOT CMAKE_${_host_math_language}_COMPILER_ID MATCHES "^Intel")
          message(FATAL_ERROR "${_math_prefix}_MKL_THREADING=INTEL needs Intel OpenMP in every language; "
            "the ${_host_math_language} compiler is ${CMAKE_${_host_math_language}_COMPILER_ID}")
        endif()
      endforeach()
      set(_host_math_mkl_threading INTEL_THREAD)
      # ELPA's Autoconf probes link without OpenMP flags, so name the runtime.
      get_filename_component(_host_math_compiler_bin "${CMAKE_C_COMPILER}" DIRECTORY)
      find_library(${_math_prefix}_MKL_IOMP5_LIBRARY NAMES iomp5
        HINTS "${_host_math_compiler_bin}/../lib" "${_host_math_compiler_bin}/../compiler/lib/intel64_lin" REQUIRED)
      set(_host_math_mkl_runtime "${${_math_prefix}_MKL_IOMP5_LIBRARY}")
    elseif(${_math_prefix}_MKL_THREADING STREQUAL "SEQUENTIAL")
      set(_host_math_mkl_threading SEQUENTIAL)
    else()
      message(FATAL_ERROR "${_math_prefix}_MKL_THREADING must be SEQUENTIAL or INTEL")
    endif()
    foreach(component SCALAPACK INTERFACE ${_host_math_mkl_threading} CORE BLACS)
      if(component STREQUAL "INTERFACE")
        set(names "${_host_math_mkl_interface}")
      elseif(component STREQUAL "BLACS")
        set(names ${_host_math_mkl_blacs})
      elseif(component STREQUAL "SCALAPACK")
        set(names mkl_scalapack_lp64)
      else()
        string(TOLOWER "${component}" lower)
        set(names "mkl_${lower}")
      endif()
      if(${_math_prefix}_MKL_ROOT)
        find_library(${_math_prefix}_MKL_${component}_LIBRARY NAMES ${names}
          PATHS "${${_math_prefix}_MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 lib64
          NO_DEFAULT_PATH REQUIRED)
      else()
        find_library(${_math_prefix}_MKL_${component}_LIBRARY NAMES ${names} REQUIRED)
      endif()
    endforeach()
    set(${_math_prefix}_BLAS_LIBRARIES "${${_math_prefix}_MKL_INTERFACE_LIBRARY}"
      "${${_math_prefix}_MKL_${_host_math_mkl_threading}_LIBRARY}" "${${_math_prefix}_MKL_CORE_LIBRARY}"
      ${_host_math_mkl_runtime} pthread m ${CMAKE_DL_LIBS})
    set(${_math_prefix}_LAPACK_LIBRARIES ${${_math_prefix}_BLAS_LIBRARIES})
    # A rescan group also supports static MKL's circular BLAS dependencies.
    set(${_math_prefix}_LINALG_LIBRARIES
      "-Wl,--start-group" "${${_math_prefix}_MKL_SCALAPACK_LIBRARY}" "${${_math_prefix}_MKL_INTERFACE_LIBRARY}"
      "${${_math_prefix}_MKL_${_host_math_mkl_threading}_LIBRARY}" "${${_math_prefix}_MKL_CORE_LIBRARY}"
      "${${_math_prefix}_MKL_BLACS_LIBRARY}" "-Wl,--end-group" ${_host_math_mkl_runtime} pthread m ${CMAKE_DL_LIBS})
  else()
    if(${_math_prefix}_LINALG_RESOLVED STREQUAL "AOCL")
      if(NOT ${_math_prefix}_AOCL_ROOT)
        message(FATAL_ERROR "AOCL requires ${_math_prefix}_AOCL_ROOT (the LP64 installation prefix)")
      endif()
      foreach(component flame blis aoclutils)
        find_library(${_math_prefix}_AOCL_${component}_LIBRARY NAMES ${component}
          PATHS "${${_math_prefix}_AOCL_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
      endforeach()
      if(NOT CMAKE_CROSSCOMPILING)
        find_path(${_math_prefix}_AOCL_INCLUDE_DIR blis.h
          PATHS "${${_math_prefix}_AOCL_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
        include(CheckCSourceRuns)
        set(CMAKE_REQUIRED_INCLUDES "${${_math_prefix}_AOCL_INCLUDE_DIR}")
        set(CMAKE_REQUIRED_LIBRARIES "${${_math_prefix}_AOCL_blis_LIBRARY}")
        check_c_source_runs("#include <blis.h>\nint main(void) { return bli_info_get_blas_int_type_size() != 32; }"
          ${_math_prefix}_AOCL_IS_LP64)
        unset(CMAKE_REQUIRED_INCLUDES)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NOT ${_math_prefix}_AOCL_IS_LP64)
          message(FATAL_ERROR "${_math_prefix}_AOCL_ROOT must provide working LP64 BLAS (32-bit Fortran integers)")
        endif()
      endif()
      set(${_math_prefix}_BLAS_LIBRARIES "${${_math_prefix}_AOCL_blis_LIBRARY}" "${${_math_prefix}_AOCL_aoclutils_LIBRARY}")
      set(${_math_prefix}_LAPACK_LIBRARIES "${${_math_prefix}_AOCL_flame_LIBRARY}" ${${_math_prefix}_BLAS_LIBRARIES})
    else()
      if(${_math_prefix}_OPENBLAS_ROOT)
        find_library(${_math_prefix}_OPENBLAS_LIBRARY NAMES openblas
          PATHS "${${_math_prefix}_OPENBLAS_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
      else()
        find_library(${_math_prefix}_OPENBLAS_LIBRARY NAMES openblas REQUIRED)
      endif()
      if(NOT CMAKE_CROSSCOMPILING)
        include(CheckCSourceRuns)
        set(CMAKE_REQUIRED_LIBRARIES "${${_math_prefix}_OPENBLAS_LIBRARY}")
        check_c_source_runs("#include <string.h>\nextern char *openblas_get_config(void);\nint main(void) { return strstr(openblas_get_config(), \"USE64BITINT\") != 0; }"
          ${_math_prefix}_OPENBLAS_IS_LP64)
        unset(CMAKE_REQUIRED_LIBRARIES)
        if(NOT ${_math_prefix}_OPENBLAS_IS_LP64)
          message(FATAL_ERROR "${_math_prefix}_OPENBLAS_LIBRARY must be a working LP64 OpenBLAS (32-bit Fortran integers), not an ILP64 library. See the configure test log.")
        endif()
      endif()
      set(${_math_prefix}_BLAS_LIBRARIES "${${_math_prefix}_OPENBLAS_LIBRARY}")
      set(${_math_prefix}_LAPACK_LIBRARIES "${${_math_prefix}_OPENBLAS_LIBRARY}")
    endif()
    # Unqualified system ScaLAPACK may have been built against another MPI.
    # Only use it with an explicit prefix/library supplied by the caller.
    if(${_math_prefix}_SCALAPACK_ROOT)
      find_library(${_math_prefix}_SCALAPACK_LIBRARY NAMES scalapack scalapack-openmpi scalapack-intelmpi
        PATHS "${${_math_prefix}_SCALAPACK_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH REQUIRED)
    elseif(${_math_prefix}_LINALG_RESOLVED STREQUAL "AOCL" AND ${_math_prefix}_MPI_RESOLVED STREQUAL "OPENMPI")
      find_library(${_math_prefix}_SCALAPACK_LIBRARY NAMES scalapack
        PATHS "${${_math_prefix}_AOCL_ROOT}" PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
    elseif(${_math_prefix}_MPI_RESOLVED STREQUAL "OPENMPI")
      find_library(${_math_prefix}_SCALAPACK_LIBRARY NAMES scalapack-openmpi)
    endif()
    if(NOT ${_math_prefix}_SCALAPACK_LIBRARY)
      if(NOT ${_math_prefix}_BUILD_SCALAPACK)
        message(FATAL_ERROR "No ScaLAPACK for ${${_math_prefix}_MPI_RESOLVED}. Set ${_math_prefix}_SCALAPACK_LIBRARY/ROOT or enable ${_math_prefix}_BUILD_SCALAPACK.")
      endif()
      if(CMAKE_VERSION VERSION_LESS 3.26)
        message(FATAL_ERROR "Building ScaLAPACK 2.2.3 requires CMake 3.26 or later. Supply an existing ${_math_prefix}_SCALAPACK_LIBRARY or update CMake.")
      endif()
      include(ExternalProject)
      set(_host_math_scalapack_url https://github.com/Reference-ScaLAPACK/scalapack/archive/refs/tags/v2.2.3.tar.gz)
      if(${_math_prefix}_SCALAPACK_ARCHIVE)
        set(_host_math_scalapack_url "${${_math_prefix}_SCALAPACK_ARCHIVE}")
      endif()
      set(_host_math_scalapack_build "${CMAKE_BINARY_DIR}/${_math_directory}")
      set(${_math_prefix}_SCALAPACK_LIBRARY "${_host_math_scalapack_build}/lib/libscalapack.a")
      string(REPLACE ";" "|" _host_math_blas "${${_math_prefix}_BLAS_LIBRARIES}")
      string(REPLACE ";" "|" _host_math_lapack "${${_math_prefix}_LAPACK_LIBRARIES}")
      ExternalProject_Add(${_math_target}
        LIST_SEPARATOR |
        URL "${_host_math_scalapack_url}"
        URL_HASH SHA256=5d93701eca663925e98010dd8d0f45fd79b2191d74e5afa5711d587370a8b9dd
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE UPDATE_COMMAND ""
        BINARY_DIR "${_host_math_scalapack_build}"
        CMAKE_ARGS "-DCMAKE_C_COMPILER:FILEPATH=${CMAKE_C_COMPILER}"
          "-DCMAKE_Fortran_COMPILER:FILEPATH=${CMAKE_Fortran_COMPILER}"
          "-DMPI_C_COMPILER:FILEPATH=${MPI_C_COMPILER}" "-DMPI_Fortran_COMPILER:FILEPATH=${MPI_Fortran_COMPILER}"
          "-DMPIEXEC_EXECUTABLE:FILEPATH=${MPIEXEC_EXECUTABLE}"
          "-DCMAKE_BUILD_TYPE:STRING=Release" "-DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON"
          "-DBUILD_SHARED_LIBS:BOOL=OFF" "-DSCALAPACK_BUILD_TESTS:BOOL=OFF"
          "-DBLAS_LIBRARIES:STRING=${_host_math_blas}"
          "-DLAPACK_LIBRARIES:STRING=${_host_math_lapack}"
        BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target scalapack --parallel ${OPENMX_BUILD_JOBS}
        INSTALL_COMMAND "" BUILD_BYPRODUCTS "${${_math_prefix}_SCALAPACK_LIBRARY}")
      message(STATUS "Building ScaLAPACK 2.2.3 for ${${_math_prefix}_MPI_RESOLVED}")
    endif()
    set(${_math_prefix}_LINALG_LIBRARIES "${${_math_prefix}_SCALAPACK_LIBRARY}" ${${_math_prefix}_LAPACK_LIBRARIES})
  endif()
  set(_host_math_signature "${${_math_prefix}_LINALG_RESOLVED}|${${_math_prefix}_LINALG_LIBRARIES}")
  if(DEFINED ${_math_prefix}_CONFIGURED_MATH_LIBRARIES AND
     NOT ${_math_prefix}_CONFIGURED_MATH_LIBRARIES STREQUAL _host_math_signature)
    message(FATAL_ERROR "Host numerical libraries changed. Use a new build directory so ELPA and ScaLAPACK are rebuilt for the selected libraries.")
  endif()
  set(${_math_prefix}_CONFIGURED_MATH_ROOTS "${_host_math_roots}" CACHE INTERNAL "Configured host numerical-library prefixes")
  set(${_math_prefix}_CONFIGURED_MATH_LIBRARIES "${_host_math_signature}" CACHE INTERNAL "Configured host numerical libraries")
  message(STATUS "Host numerical libraries: ${${_math_prefix}_LINALG_RESOLVED} (LP64), MPI ${${_math_prefix}_MPI_RESOLVED}")
endmacro()
