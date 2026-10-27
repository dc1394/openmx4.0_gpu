cmake_minimum_required(VERSION 3.25.2)
project(OpenMXBasis VERSION 0.1.0 LANGUAGES CXX)

# This directory can be copied and built on its own. Do not reach into core/,
# source/, or the application's SDK/dependency discovery from this package.
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(openmx_basis STATIC src/radial_table.cpp src/c_api.cpp src/pao_file.cpp src/vps_file.cpp src/density_table.cpp src/quadrature.cpp src/bessel.cpp src/harmonics.cpp)
add_library(OpenMX::basis ALIAS openmx_basis)
set_target_properties(openmx_basis PROPERTIES
  EXPORT_NAME basis POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_compile_features(openmx_basis PUBLIC cxx_std_20)
target_include_directories(openmx_basis PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# Keep the inherited OpenMX interpolation stencil and rounding contract.
# These kernels are out-of-line; consumers need no floating-point flags.
target_compile_options(openmx_basis PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:IntelLLVM>:-Wall;-Wextra;-fp-model=precise;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:MSVC>:/fp:strict>")
option(OPENMX_BASIS_SANITIZERS "Instrument the basis library and its tests" OFF)
if(OPENMX_BASIS_SANITIZERS OR OPENMX_CORE_SANITIZERS)
  target_compile_options(openmx_basis PUBLIC
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(openmx_basis PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS openmx_basis EXPORT OpenMXBasisTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/OpenMXBasis)
set(_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/OpenMXBasis")
configure_package_config_file(cmake/OpenMXBasisConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXBasisConfig.cmake"
  INSTALL_DESTINATION "${_package_dir}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/OpenMXBasisConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT OpenMXBasisTargets NAMESPACE OpenMX:: DESTINATION "${_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenMXBasisConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXBasisConfigVersion.cmake" DESTINATION "${_package_dir}")

option(OPENMX_BASIS_BUILD_TESTING "Build the independent basis tests" ${BUILD_TESTING})
if(BUILD_TESTING AND OPENMX_BASIS_BUILD_TESTING)
  add_subdirectory(tests)
endif()
