# Kokkos 5.2.2 SIMD includes Kokkos_Core.hpp, whose CUDA/HIP configuration
# requires a device compiler even for host functions. Build a private SERIAL
# Kokkos configuration in an isolated shared library, using ordinary C++.
# Hidden C++/Kokkos symbols and a POD-only C ABI prevent configuration/ODR leaks.
option(OPENMX_CORE_HOST_SIMD "Build Kokkos host SIMD variants with safe runtime dispatch" ON)
set(OPENMX_HOST_SIMD_CXX_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH
    "Ordinary C++20 compiler for CPU SIMD (not nvcc_wrapper/hipcc)")
include(ExternalProject)
set(_host_build "${CMAKE_CURRENT_BINARY_DIR}/host-simd")
set(_host_library "${_host_build}/lib/${CMAKE_SHARED_LIBRARY_PREFIX}openmx_host_simd${CMAKE_SHARED_LIBRARY_SUFFIX}")
set(_host_cmake_args
    "-DCMAKE_CXX_COMPILER:FILEPATH=${OPENMX_HOST_SIMD_CXX_COMPILER}"
    "-DCMAKE_CXX_FLAGS:STRING="
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}"
    "-DOPENMX_KOKKOS_SOURCE_DIR:PATH=${kokkos_SOURCE_DIR}"
    "-DOPENMX_CORE_HOST_SIMD:BOOL=${OPENMX_CORE_HOST_SIMD}"
    "-DOPENMX_CORE_SANITIZERS:BOOL=${OPENMX_CORE_SANITIZERS}")
foreach(variable CMAKE_TOOLCHAIN_FILE CMAKE_SYSROOT CMAKE_CXX_COMPILER_TARGET
    CMAKE_OSX_ARCHITECTURES CMAKE_OSX_SYSROOT CMAKE_OSX_DEPLOYMENT_TARGET)
  if(DEFINED ${variable})
    list(APPEND _host_cmake_args "-D${variable}=${${variable}}")
  endif()
endforeach()
ExternalProject_Add(openmx_host_simd_build
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/host_simd"
    BINARY_DIR "${_host_build}"
    CMAKE_ARGS ${_host_cmake_args}
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG> --parallel 2
    BUILD_ALWAYS TRUE
    BUILD_BYPRODUCTS "${_host_library}"
    INSTALL_COMMAND "")
add_library(openmx_host_simd SHARED IMPORTED)
set_target_properties(openmx_host_simd PROPERTIES IMPORTED_LOCATION "${_host_library}")
add_dependencies(openmx_host_simd openmx_host_simd_build)
target_link_libraries(openmx_core PRIVATE openmx_host_simd)
if(APPLE)
  set_property(TARGET openmx_core APPEND PROPERTY INSTALL_RPATH "@loader_path")
else()
  set_property(TARGET openmx_core APPEND PROPERTY INSTALL_RPATH "$ORIGIN")
endif()
install(FILES "${_host_library}" DESTINATION lib)
