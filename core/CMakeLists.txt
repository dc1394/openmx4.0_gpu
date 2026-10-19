cmake_minimum_required(VERSION 3.25.2)
project(OpenMXCore VERSION 0.1 LANGUAGES CXX)
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/OpenMXBuildType.cmake")

include(CTest)
include(FetchContent)
set(OPENMX_CORE_BACKEND "SERIAL" CACHE STRING "Core execution backend: SERIAL, OPENMP, CUDA, HIP")
set_property(CACHE OPENMX_CORE_BACKEND PROPERTY STRINGS SERIAL OPENMP CUDA HIP)
string(TOUPPER "${OPENMX_CORE_BACKEND}" OPENMX_CORE_BACKEND)
if(NOT OPENMX_CORE_BACKEND MATCHES "^(SERIAL|OPENMP|CUDA|HIP)$")
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
foreach(_device CUDA HIP OPENMP)
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

set(_core_sources src/radial.cpp src/radial_table.cpp src/pao_storage.cpp src/vps_storage.cpp
  src/product_storage.cpp src/neutral_projectors.cpp src/pair_bessel.cpp
  src/overlap_workspace.cpp src/c_api.cpp)
add_library(openmx_core SHARED ${_core_sources})
add_library(OpenMX::core ALIAS openmx_core)
target_include_directories(openmx_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../source")
target_link_libraries(openmx_core PUBLIC Kokkos::kokkos)
install(FILES "${kokkos_SOURCE_DIR}/LICENSE" DESTINATION share/licenses/openmx_core RENAME Kokkos-LICENSE)
target_compile_features(openmx_core PUBLIC cxx_std_20)
if(OPENMX_CORE_FORCE_KOKKOS_MDSPAN)
  target_compile_definitions(openmx_core PUBLIC OPENMX_CORE_FORCE_KOKKOS_MDSPAN=1)
endif()
if(OPENMX_CORE_BACKEND MATCHES "^(CUDA|HIP)$")
  set_source_files_properties(${_core_sources} PROPERTIES LANGUAGE ${OPENMX_CORE_BACKEND})
endif()
# Retain the original quadrature's summation order and avoid contraction when
# comparing a refactor against the old C implementation.
target_compile_options(openmx_core PRIVATE
  "$<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
  "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>"
  "$<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>")
if(OPENMX_CORE_SANITIZERS)
  if(NOT OPENMX_CORE_BACKEND MATCHES "^(SERIAL|OPENMP)$")
    message(FATAL_ERROR "Use a host core build for address/undefined sanitizers")
  endif()
  target_compile_options(openmx_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(openmx_core PUBLIC -fsanitize=address,undefined)
endif()

option(OPENMX_CORE_BUILD_BENCHMARKS "Build portable core performance probes" OFF)
if(OPENMX_CORE_BUILD_BENCHMARKS)
  add_executable(openmx_radial_benchmark benchmarks/radial_transform.cpp)
  target_link_libraries(openmx_radial_benchmark PRIVATE OpenMX::core)
  if(OPENMX_CORE_BACKEND MATCHES "^(CUDA|HIP)$")
    set_source_files_properties(benchmarks/radial_transform.cpp PROPERTIES LANGUAGE ${OPENMX_CORE_BACKEND})
  endif()
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
