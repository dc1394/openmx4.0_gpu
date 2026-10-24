# Full CPU application. The numerical sources and calculation owners are the
# same as the GPU application; no CUDA/HIP compiler, header or library is used.
enable_language(C CXX Fortran)
find_package(MPI REQUIRED COMPONENTS C CXX Fortran)
find_package(OpenMP REQUIRED COMPONENTS C CXX)
find_path(OPENMX_CPU_FFTW_INCLUDE_DIR fftw3.h PATH_SUFFIXES include include/fftw REQUIRED)
find_library(OPENMX_CPU_FFTW_LIBRARY NAMES fftw3 REQUIRED)
set(OPENMX_CPU_LINALG_LIBRARIES "" CACHE STRING "LP64 ScaLAPACK, LAPACK and BLAS libraries (semicolon-separated)")
if(NOT OPENMX_CPU_LINALG_LIBRARIES)
  set(_cpu_scalapack_names scalapack)
  foreach(directory IN LISTS MPI_C_INCLUDE_DIRS)
    if(EXISTS "${directory}/mpi.h")
      file(STRINGS "${directory}/mpi.h" vendor REGEX "^#[ \t]*define[ \t]+(OPEN_MPI|MPICH_NAME|I_MPI_VERSION)")
      if(vendor MATCHES "OPEN_MPI")
        list(PREPEND _cpu_scalapack_names scalapack-openmpi)
        break()
      elseif(vendor MATCHES "MPICH_NAME|I_MPI_VERSION")
        list(PREPEND _cpu_scalapack_names scalapack-mpich)
        break()
      endif()
    endif()
  endforeach()
  find_library(OPENMX_CPU_SCALAPACK_LIBRARY NAMES ${_cpu_scalapack_names} REQUIRED)
  set(BLA_SIZEOF_INTEGER 4)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  set(OPENMX_CPU_LINALG_LIBRARIES "${OPENMX_CPU_SCALAPACK_LIBRARY};${LAPACK_LIBRARIES};${BLAS_LIBRARIES}" CACHE STRING "LP64 ScaLAPACK, LAPACK and BLAS libraries (semicolon-separated)" FORCE)
endif()
if(NOT OPENMX_BUILD_JOBS MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "OPENMX_BUILD_JOBS must be a positive integer")
endif()
if(OPENMX_MAKE_OPTIONS)
  message(FATAL_ERROR "CPU builds use CMake directly; use CMAKE_C_FLAGS/CMAKE_CXX_FLAGS and OPENMX_CPU_* options instead of OPENMX_MAKE_OPTIONS")
endif()

# The very same bundled modern ELPA release as CUDA/HIP, configured without
# device kernels or COSMA. An existing matching CPU installation is optional.
set(OPENMX_CPU_ELPA_ROOT "" CACHE PATH "Existing CPU ELPA 2026.02.002 prefix; empty builds the bundled release")
set(_cpu_backend "${CMAKE_SOURCE_DIR}/source/backends/cpu")
if(OPENMX_CPU_ELPA_ROOT)
  find_path(OPENMX_CPU_ELPA_INCLUDE_DIR elpa/elpa.h
    PATHS "${OPENMX_CPU_ELPA_ROOT}/include/elpa-2026.02.002" NO_DEFAULT_PATH REQUIRED)
  find_library(OPENMX_CPU_ELPA_LIBRARY NAMES elpa
    PATHS "${OPENMX_CPU_ELPA_ROOT}/lib" "${OPENMX_CPU_ELPA_ROOT}/lib64" NO_DEFAULT_PATH REQUIRED)
else()
  include(ExternalProject)
  set(_cpu_elpa_prefix "${CMAKE_BINARY_DIR}/cpu-elpa")
  set(OPENMX_CPU_ELPA_INCLUDE_DIR "${_cpu_elpa_prefix}/include/elpa-2026.02.002")
  set(OPENMX_CPU_ELPA_LIBRARY "${_cpu_elpa_prefix}/lib/libelpa.a")
  # Libtool hoists absolute library filenames before libelpa.a, which breaks
  # --as-needed and static BLAS links. Keep libraries as ordered -l
  # arguments for the configure probes and ELPA's own helper programs.
  set(_cpu_elpa_libraries)
  foreach(library IN LISTS OPENMX_CPU_LINALG_LIBRARIES)
    if(IS_ABSOLUTE "${library}")
      get_filename_component(directory "${library}" DIRECTORY)
      get_filename_component(filename "${library}" NAME)
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
        list(APPEND _cpu_elpa_libraries "-L'${directory}'" "-l:${filename}")
      else()
        string(REGEX REPLACE "^lib|\\.(a|dylib|so(\\..*)?)$" "" name "${filename}")
        list(APPEND _cpu_elpa_libraries "-L'${directory}'" "-l${name}")
      endif()
    else()
      list(APPEND _cpu_elpa_libraries "${library}")
    endif()
  endforeach()
  string(JOIN " " _cpu_linalg_flags ${_cpu_elpa_libraries})
  set(_cpu_dependency_dirs ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
  foreach(library IN LISTS OPENMX_CPU_LINALG_LIBRARIES MPI_C_LIBRARIES MPI_Fortran_LIBRARIES)
    if(IS_ABSOLUTE "${library}")
      get_filename_component(directory "${library}" DIRECTORY)
      list(APPEND _cpu_dependency_dirs "${directory}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES _cpu_dependency_dirs)
  string(JOIN ":" _cpu_dependency_path ${_cpu_dependency_dirs})
  set(_cpu_elpa_env "${CMAKE_COMMAND}" -E env
    "CC=${MPI_C_COMPILER}" "CXX=${MPI_CXX_COMPILER}" "FC=${MPI_Fortran_COMPILER}"
    "OMPI_CC=${CMAKE_C_COMPILER}" "OMPI_CXX=${CMAKE_CXX_COMPILER}" "OMPI_FC=${CMAKE_Fortran_COMPILER}"
    "MPICH_CC=${CMAKE_C_COMPILER}" "MPICH_CXX=${CMAKE_CXX_COMPILER}" "MPICH_FC=${CMAKE_Fortran_COMPILER}"
    "CFLAGS=-O3 -fPIC" "CXXFLAGS=-O3 -fPIC" "FCFLAGS=-O3 -fPIC" "CPPFLAGS=" "LDFLAGS="
    "LIBS=${_cpu_linalg_flags}" "CPATH=" "C_INCLUDE_PATH=" "CPLUS_INCLUDE_PATH="
    "LD_LIBRARY_PATH=${_cpu_dependency_path}")
  ExternalProject_Add(openmx_cpu_elpa_build
    URL "${CMAKE_SOURCE_DIR}/source/third_party/dist/elpa-new_release_2026_02_002.zip"
    URL_HASH SHA256=14c05257b2d10a80c7c825eca278c4787ef34bd7c8a6343c2fd4686ad56f22d5
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    UPDATE_COMMAND ""
    PATCH_COMMAND ${_cpu_elpa_env} sh <SOURCE_DIR>/autogen.sh
    CONFIGURE_COMMAND ${_cpu_elpa_env} <SOURCE_DIR>/configure
      --prefix=${_cpu_elpa_prefix} --disable-shared --enable-static --with-pic
      --disable-nvidia-gpu-kernels --disable-amd-gpu-kernels --disable-intel-gpu-sycl-kernels
      --enable-generic-kernels --disable-sse-assembly-kernels --disable-sse-kernels
      --disable-avx-kernels --disable-avx2-kernels --disable-avx512-kernels
      --with-test-programs=no
      --disable-c-tests --disable-cpp-tests --disable-fortran-tests --disable-Fortran-tests
    BUILD_COMMAND ${_cpu_elpa_env} "${OPENMX_MAKE_EXECUTABLE}" -j${OPENMX_BUILD_JOBS}
    INSTALL_COMMAND ${_cpu_elpa_env} "${OPENMX_MAKE_EXECUTABLE}" install
    BUILD_BYPRODUCTS "${OPENMX_CPU_ELPA_LIBRARY}")
endif()

# Keep one authoritative list of application translation units. Native GPU
# kernels have a CPU-only adapter that declines before publishing any output.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/source/Makefile")
set(_cpu_adapter "${CMAKE_BINARY_DIR}/cpu-accelerator.cpp")
set(_cpu_manifest "${CMAKE_BINARY_DIR}/cpu-sources.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/tools/prepare_cpu_adapter.py")
execute_process(COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/prepare_cpu_adapter.py"
  --output "${_cpu_adapter}" --manifest "${_cpu_manifest}" --source-root "${CMAKE_SOURCE_DIR}/source"
  COMMAND_ERROR_IS_FATAL ANY)
file(READ "${_cpu_manifest}" _cpu_manifest_json)
set(_cpu_sources)
set(_cpu_kernels)
foreach(kind sources kernels)
  string(JSON count LENGTH "${_cpu_manifest_json}" ${kind})
  math(EXPR last "${count}-1")
  foreach(index RANGE 0 ${last})
    string(JSON path GET "${_cpu_manifest_json}" ${kind} ${index})
    list(APPEND _cpu_${kind} "${path}")
  endforeach()
endforeach()
list(LENGTH _cpu_sources _cpu_source_count)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_cpu_kernels})
add_executable(openmx_cpu_application ${_cpu_sources} "${_cpu_adapter}" "${_cpu_backend}/device.c" "${_cpu_backend}/fortran_runtime.f90")
set_target_properties(openmx_cpu_application PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON
  CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON LINKER_LANGUAGE CXX)
target_compile_definitions(openmx_cpu_application PRIVATE OPENMX_GPU_BACKEND=0)
target_include_directories(openmx_cpu_application PRIVATE "${_cpu_backend}/include" "${CMAKE_SOURCE_DIR}/source"
  "${OPENMX_CPU_FFTW_INCLUDE_DIR}" "${OPENMX_CPU_ELPA_INCLUDE_DIR}")
target_compile_options(openmx_cpu_application PRIVATE
  # The original TRAN headers contain tentative definitions. Match the native
  # compiler until those legacy globals get explicit owners in a later task.
  "$<$<COMPILE_LANG_AND_ID:C,GNU,Clang,AppleClang>:-fcommon;-Wno-unknown-pragmas;-Wno-unused-parameter;-ffp-contract=off>"
  "$<$<COMPILE_LANG_AND_ID:C,IntelLLVM>:-fcommon;-Wno-unknown-pragmas;-Wno-unused-parameter;-fp-model=precise;-ffp-contract=off>"
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wno-unused-parameter;-ffp-contract=off>")
target_link_libraries(openmx_cpu_application PRIVATE OpenMX::core MPI::MPI_C MPI::MPI_CXX MPI::MPI_Fortran
  OpenMP::OpenMP_C OpenMP::OpenMP_CXX "${OPENMX_CPU_ELPA_LIBRARY}"
  "${OPENMX_CPU_FFTW_LIBRARY}" ${OPENMX_CPU_LINALG_LIBRARIES} ${CMAKE_DL_LIBS} m)
if(TARGET openmx_cpu_elpa_build)
  add_dependencies(openmx_cpu_application openmx_cpu_elpa_build)
endif()
# Compiler/MPI environments can advertise non-system library directories as
# implicit link paths. CMake then omits their RPATH, even though a clean login
# shell cannot load the executable. Record the actual linked dependency paths.
# Prefer the C/C++ OpenMP runtime over aliases shipped by another compiler.
set(_cpu_runtime_dirs)
foreach(library IN LISTS OpenMP_C_LIBRARIES OpenMP_CXX_LIBRARIES)
  if(IS_ABSOLUTE "${library}")
    get_filename_component(real_library "${library}" REALPATH)
    get_filename_component(directory "${real_library}" DIRECTORY)
    list(APPEND _cpu_runtime_dirs "${directory}")
  endif()
endforeach()
foreach(library IN LISTS OPENMX_CPU_LINALG_LIBRARIES MPI_C_LIBRARIES MPI_Fortran_LIBRARIES
    OPENMX_CPU_FFTW_LIBRARY OPENMX_CPU_ELPA_LIBRARY)
  if(IS_ABSOLUTE "${library}")
    get_filename_component(directory "${library}" DIRECTORY)
    list(APPEND _cpu_runtime_dirs "${directory}")
  endif()
endforeach()
foreach(library IN LISTS CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES)
  unset(_cpu_fortran_runtime)
  find_library(_cpu_fortran_runtime NAMES "${library}"
    PATHS ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH NO_CACHE)
  if(_cpu_fortran_runtime)
    get_filename_component(directory "${_cpu_fortran_runtime}" DIRECTORY)
    list(APPEND _cpu_runtime_dirs "${directory}")
  endif()
endforeach()
list(REMOVE_DUPLICATES _cpu_runtime_dirs)
if(APPLE)
  set(_cpu_loader "@loader_path")
else()
  set(_cpu_loader "$ORIGIN")
endif()
set_property(TARGET openmx_cpu_application APPEND PROPERTY BUILD_RPATH
  "${_cpu_loader}/lib" "${_cpu_loader}/../lib" ${_cpu_runtime_dirs})
set(_cpu_copy_command)
if(OPENMX_COPY_TO_WORK)
  # A previous GPU install may have a device-enabled Core in work/lib. Publish
  # the matching CPU libraries atomically before the executable, so a CPU copy
  # cannot accidentally require that GPU SDK. Loaded old files stay intact.
  list(APPEND _cpu_copy_command
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/copy_openmx.py"
      "$<TARGET_FILE:openmx_host_simd>" "${OPENMX_WORK_DIR}/lib/$<TARGET_FILE_NAME:openmx_host_simd>"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/copy_openmx.py"
      "$<TARGET_FILE:openmx_core>" "${OPENMX_WORK_DIR}/lib/$<TARGET_FILE_NAME:openmx_core>"
    COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_SOURCE_DIR}/tools/copy_openmx.py" "$<TARGET_FILE:openmx_cpu_application>" "${OPENMX_WORK_DIR}/openmx")
endif()
add_custom_target(openmx ALL
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tools/copy_openmx.py"
    "$<TARGET_FILE:openmx_cpu_application>" "${CMAKE_BINARY_DIR}/openmx"
  ${_cpu_copy_command} DEPENDS openmx_cpu_application
  BYPRODUCTS "${CMAKE_BINARY_DIR}/openmx" VERBATIM)
message(STATUS "Full CPU application: ${_cpu_source_count} shared source units; no GPU SDK")
