cmake_minimum_required(VERSION 3.25.2)
project(OpenMXCore VERSION 0.1 LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXBuildType.cmake")

include(CTest)
include(FetchContent)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)
set(OPENMX_CORE_BACKEND "SERIAL" CACHE STRING "Core execution backend: SERIAL, OPENMP, CUDA, HIP, SYCL")
set_property(CACHE OPENMX_CORE_BACKEND PROPERTY STRINGS SERIAL OPENMP CUDA HIP SYCL)
string(TOUPPER "${OPENMX_CORE_BACKEND}" OPENMX_CORE_BACKEND)
if(NOT OPENMX_CORE_BACKEND MATCHES "^(SERIAL|OPENMP|CUDA|HIP|SYCL)$")
  message(FATAL_ERROR "Invalid OPENMX_CORE_BACKEND=${OPENMX_CORE_BACKEND}")
endif()
if(OPENMX_CORE_BACKEND STREQUAL "CUDA")
  include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXGPUArchitecture.cmake")
  openmx_resolve_cuda_architecture()
  if(OPENMX_CUDA_ARCH)
    openmx_select_kokkos_cuda_architecture("${OPENMX_CUDA_ARCH}")
  endif()
endif()
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  if(OPENMX_CORE_BACKEND MATCHES "^(SERIAL|OPENMP)$" AND "cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    include(CheckCXXSourceCompiles)
    set(CMAKE_CXX_STANDARD 23)
    block()
      # Test header/language support without imposing a try-compile link
      # environment on installations using a separately supplied libc++.
      set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
      check_cxx_source_compiles("#include <mdspan>
        #if !defined(__cpp_lib_mdspan) || __cpp_lib_mdspan < 202207L
        #error std::mdspan unavailable
        #endif
        int main() { double x[6]{}; std::mdspan<double,std::dextents<int,2>> v(x,2,3);
          return v.extent(0) != 2; }" OPENMX_CORE_HAS_STD_MDSPAN)
    endblock()
    if(NOT OPENMX_CORE_HAS_STD_MDSPAN)
      set(CMAKE_CXX_STANDARD 20)
    endif()
  endif()
endif()
if(CMAKE_CXX_STANDARD LESS 20)
  message(FATAL_ERROR "The OpenMX C++ core requires C++20 or later")
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Kokkos_ENABLE_SERIAL ON CACHE BOOL "" FORCE)
foreach(_device CUDA HIP OPENMP SYCL)
  if(OPENMX_CORE_BACKEND STREQUAL _device)
    set(Kokkos_ENABLE_${_device} ON CACHE BOOL "" FORCE)
  else()
    set(Kokkos_ENABLE_${_device} OFF CACHE BOOL "" FORCE)
  endif()
endforeach()
if(OPENMX_CORE_BACKEND MATCHES "^(CUDA|HIP)$")
  set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE ON CACHE BOOL "" FORCE)
  # Device kernels use Kokkos::View; std::mdspan is a host-only view.
  set(CMAKE_${OPENMX_CORE_BACKEND}_STANDARD ${CMAKE_CXX_STANDARD})
  enable_language(${OPENMX_CORE_BACKEND})
else()
  set(Kokkos_ENABLE_COMPILE_AS_CMAKE_LANGUAGE OFF CACHE BOOL "" FORCE)
endif()
if(OPENMX_CORE_BACKEND STREQUAL "SYCL")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-fsycl" OPENMX_CORE_COMPILER_HAS_SYCL)
  if(NOT OPENMX_CORE_COMPILER_HAS_SYCL)
    message(FATAL_ERROR "SYCL requires a SYCL compiler, e.g. -DCMAKE_CXX_COMPILER=icpx")
  endif()
endif()
set(Kokkos_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(Kokkos_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(kokkos
  URL https://github.com/kokkos/kokkos/releases/download/5.2.2/kokkos-5.2.2.tar.gz
  URL_HASH SHA256=d6557aaef39302282a15f9c770433d1fcdf4e961dfd6d9ed726b9d0d0f546b9f
  PATCH_COMMAND "${CMAKE_COMMAND}" "-DKOKKOS_SOURCE_DIR=<SOURCE_DIR>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXKokkosGfx1200.cmake"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(kokkos)

option(OPENMX_CORE_SANITIZERS "Enable address/undefined sanitizers in host core code" OFF)
option(OPENMX_CORE_FORCE_KOKKOS_MDSPAN "Use Kokkos/mdspan even if std::mdspan is available" OFF)

# Scientific modules are independent libraries, also consumed by the legacy
# bridge. Basis deliberately stays ordinary host C++ even in a device core.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../modules/basis" "modules/basis")

set(_core_sources src/radial.cpp src/radial_table.cpp src/pao_storage.cpp src/pao_file_api.cpp src/vps_storage.cpp src/vps_file_api.cpp src/atomic_density_api.cpp src/quadrature_api.cpp
  src/product_storage.cpp src/neutral_projectors.cpp src/pair_bessel.cpp
  src/force_workspace.cpp src/force_recovery.cpp src/force_hnl_batch.cpp src/core_hole.cpp src/projector_force.cpp src/dispersion_workspace.cpp src/two_center.cpp src/overlap_workspace.cpp src/nonlocal_workspace.cpp src/vna_workspace.cpp src/vna_preparation.cpp
  src/dsvna_workspace.cpp src/hvna_layout.cpp src/hvna_workspace.cpp src/atom_pairs.cpp
  src/projector_exchange.cpp src/vna_invocation.cpp src/block_matrix.cpp src/vna_matrix_storage.cpp
  src/reciprocal_views.cpp src/reciprocal_archive.cpp src/species_storage.cpp src/species_api.cpp
  src/atomic_views.cpp src/orbital_workspace.cpp src/ragged_matrix.cpp src/matrix_scratch.cpp src/unpivoted_inverse.cpp src/pivoted_inverse.cpp src/mixing_history.cpp src/mixing_policy.cpp src/band_energy.cpp src/energy_components.cpp src/thermodynamic_energy.cpp src/dynamics.cpp src/chemical.cpp src/cwf.cpp src/cwf_workspace.cpp
  src/electronic_matrices.cpp src/electronic_matrix_api.cpp
  src/local_basis_state.cpp src/local_basis_api.cpp
  src/orbital_optimization.cpp src/orbital_optimization_api.cpp src/solver_workspace.cpp src/native_resources.cpp src/grid_assembly.cpp src/host_simd_api.cpp src/olg_api.cpp src/olg_cache_api.cpp src/energy_simd_api.cpp src/orbitals_simd_api.cpp src/force_simd_api.cpp src/vna_simd_api.cpp src/c_api.cpp)
add_library(openmx_core SHARED ${_core_sources})
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXCompilerRuntime.cmake")
add_library(OpenMX::core ALIAS openmx_core)
target_include_directories(openmx_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(openmx_core PUBLIC OpenMX::basis Kokkos::kokkos Threads::Threads)
install(FILES "${kokkos_SOURCE_DIR}/LICENSE" DESTINATION share/licenses/openmx_core RENAME Kokkos-LICENSE)
target_compile_features(openmx_core PUBLIC cxx_std_20)
if(OPENMX_CORE_FORCE_KOKKOS_MDSPAN)
  target_compile_definitions(openmx_core PUBLIC OPENMX_CORE_FORCE_KOKKOS_MDSPAN=1)
endif()
if(OPENMX_CORE_BACKEND MATCHES "^(CUDA|HIP)$")
  set_source_files_properties(${_core_sources} PROPERTIES LANGUAGE ${OPENMX_CORE_BACKEND})
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXHostSIMD.cmake")
openmx_compiler_runtime_rpath(openmx_core CXX)
# Retain the original quadrature's summation order and avoid contraction when
# comparing a refactor against the old C implementation.
target_compile_options(openmx_core PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,IntelLLVM>:-Wall;-Wextra;-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>"
  "$<$<COMPILE_LANGUAGE:HIP>:-Wall;-Wextra;-Wpedantic>")
# Inline numerical kernels and their callers share this arithmetic contract.
# In particular IntelLLVM otherwise reassociates even simple displacements.
target_compile_options(openmx_core PUBLIC
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>"
  "$<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-fp-model=precise;-ffp-contract=off>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
  "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
if(OPENMX_CORE_SANITIZERS)
  if(NOT OPENMX_CORE_BACKEND MATCHES "^(SERIAL|OPENMP)$")
    message(FATAL_ERROR "Use a host core build for address/undefined sanitizers")
  endif()
  target_compile_options(openmx_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(openmx_core PUBLIC -fsanitize=address,undefined)
endif()

option(OPENMX_CORE_BUILD_BENCHMARKS "Build portable core performance probes" OFF)
if(OPENMX_CORE_BUILD_BENCHMARKS)
  add_executable(openmx_grid_benchmark benchmarks/grid_assembly.cpp)
  target_link_libraries(openmx_grid_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_radial_benchmark benchmarks/radial_transform.cpp)
  target_link_libraries(openmx_radial_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_olg_benchmark benchmarks/olg_intersect.cpp)
  target_link_libraries(openmx_olg_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_force_benchmark benchmarks/force_simd.cpp)
  target_link_libraries(openmx_force_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_energy_benchmark benchmarks/energy_simd.cpp)
  target_link_libraries(openmx_energy_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_orbitals_benchmark benchmarks/orbitals_simd.cpp)
  target_link_libraries(openmx_orbitals_benchmark PRIVATE OpenMX::core)
  add_executable(openmx_vna_benchmark benchmarks/vna_simd.cpp)
  target_link_libraries(openmx_vna_benchmark PRIVATE OpenMX::core)
  if(OPENMX_CORE_BACKEND MATCHES "^(CUDA|HIP)$")
    set_source_files_properties(benchmarks/radial_transform.cpp benchmarks/grid_assembly.cpp
      benchmarks/olg_intersect.cpp benchmarks/force_simd.cpp benchmarks/energy_simd.cpp benchmarks/orbitals_simd.cpp benchmarks/vna_simd.cpp PROPERTIES LANGUAGE ${OPENMX_CORE_BACKEND})
  endif()
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
