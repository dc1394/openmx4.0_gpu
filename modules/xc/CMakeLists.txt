cmake_minimum_required(VERSION 3.25.2)
project(OpenMXXc VERSION 0.1.0 LANGUAGES C CXX)

# This directory can be copied and built on its own. Do not reach into core/,
# source/, or the application's SDK/dependency discovery from this package.
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(openmx_xc STATIC src/functionals.cpp src/c_api.cpp)
add_library(OpenMX::xc ALIAS openmx_xc)
set_target_properties(openmx_xc PROPERTIES
  EXPORT_NAME xc POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_compile_features(openmx_xc PUBLIC cxx_std_20)
target_include_directories(openmx_xc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# The functionals reproduce the inherited OpenMX arithmetic bitwise: no
# contraction and no fast-math in this translation unit. Consumers need no
# floating-point flags.
target_compile_options(openmx_xc PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:IntelLLVM>:-Wall;-Wextra;-fp-model=precise;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:MSVC>:/fp:strict>")
option(OPENMX_XC_SANITIZERS "Instrument the xc library and its tests" OFF)
if(OPENMX_XC_SANITIZERS OR OPENMX_CORE_SANITIZERS)
  target_compile_options(openmx_xc PUBLIC
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(openmx_xc PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS openmx_xc EXPORT OpenMXXcTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/OpenMXXc)
set(_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/OpenMXXc")
configure_package_config_file(cmake/OpenMXXcConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXXcConfig.cmake"
  INSTALL_DESTINATION "${_package_dir}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/OpenMXXcConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT OpenMXXcTargets NAMESPACE OpenMX:: DESTINATION "${_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenMXXcConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXXcConfigVersion.cmake" DESTINATION "${_package_dir}")

option(OPENMX_XC_BUILD_TESTING "Build the independent xc tests" ${BUILD_TESTING})
if(BUILD_TESTING AND OPENMX_XC_BUILD_TESTING)
  add_subdirectory(tests)
endif()
