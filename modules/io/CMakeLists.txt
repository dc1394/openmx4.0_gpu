cmake_minimum_required(VERSION 3.25.2)
project(OpenMXIo VERSION 0.1.0 LANGUAGES C CXX)

# This directory can be copied and built on its own. Do not reach into core/,
# source/, or the application's SDK/dependency discovery from this package.
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(openmx_io STATIC src/keyword_file.cpp src/c_api.cpp src/basis_spec.cpp src/species.cpp src/atoms.cpp src/settings.cpp)
add_library(OpenMX::io ALIAS openmx_io)
set_target_properties(openmx_io PROPERTIES
  EXPORT_NAME io POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_compile_features(openmx_io PUBLIC cxx_std_20)
target_include_directories(openmx_io PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_options(openmx_io PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic>"
  "$<$<CXX_COMPILER_ID:IntelLLVM>:-Wall;-Wextra>"
  "$<$<CXX_COMPILER_ID:MSVC>:/W4>")
option(OPENMX_IO_SANITIZERS "Instrument the io library and its tests" OFF)
if(OPENMX_IO_SANITIZERS OR OPENMX_CORE_SANITIZERS)
  target_compile_options(openmx_io PUBLIC
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(openmx_io PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS openmx_io EXPORT OpenMXIoTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/OpenMXIo)
set(_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/OpenMXIo")
configure_package_config_file(cmake/OpenMXIoConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXIoConfig.cmake"
  INSTALL_DESTINATION "${_package_dir}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/OpenMXIoConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT OpenMXIoTargets NAMESPACE OpenMX:: DESTINATION "${_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenMXIoConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXIoConfigVersion.cmake" DESTINATION "${_package_dir}")

option(OPENMX_IO_BUILD_TESTING "Build the independent io tests" ${BUILD_TESTING})
if(BUILD_TESTING AND OPENMX_IO_BUILD_TESTING)
  add_subdirectory(tests)
endif()
