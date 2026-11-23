cmake_minimum_required(VERSION 3.25.2)
project(OpenMXExx VERSION 0.1.0 LANGUAGES CXX)

# This directory can be copied and built on its own. Do not reach into core/,
# source/, or the application's SDK/dependency discovery from this package.
# Its one dependency is the basis package (spherical Bessel functions, Gaunt
# coefficients, real spherical harmonics): an installed OpenMXBasis, or the
# sources of modules/basis next to this directory.
include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(OPENMX_EXX_SANITIZERS "Instrument the exx library and its tests" OFF)

if(NOT TARGET OpenMX::basis)
  find_package(OpenMXBasis 0.1 CONFIG QUIET)
endif()
if(NOT TARGET OpenMX::basis)
  set(OPENMX_EXX_BASIS_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/../basis" CACHE PATH
    "Sources of the OpenMX basis package, used when no installed OpenMXBasis is found")
  if(NOT EXISTS "${OPENMX_EXX_BASIS_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "OpenMX::exx needs OpenMX::basis. Install OpenMXBasis and add its prefix to "
      "CMAKE_PREFIX_PATH, or set OPENMX_EXX_BASIS_SOURCE to a copy of modules/basis.")
  endif()
  # The basis package keeps its own tests; this build needs the library only.
  set(OPENMX_BASIS_BUILD_TESTING OFF)
  if(OPENMX_EXX_SANITIZERS)
    set(OPENMX_BASIS_SANITIZERS ON)
  endif()
  add_subdirectory("${OPENMX_EXX_BASIS_SOURCE}" "${CMAKE_CURRENT_BINARY_DIR}/basis")
endif()

add_library(openmx_exx STATIC src/abf.cpp src/dense_backend.cpp src/kernel.cpp src/lri.cpp src/system.cpp
  src/two_center.cpp)
add_library(OpenMX::exx ALIAS openmx_exx)
set_target_properties(openmx_exx PROPERTIES
  EXPORT_NAME exx POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
target_compile_features(openmx_exx PUBLIC cxx_std_20)
target_include_directories(openmx_exx PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(openmx_exx PRIVATE OpenMX::basis)
# No contraction and no fast-math: the results must not depend on whether a
# compiler fuses a multiplication and an addition. Consumers need no
# floating-point flags.
target_compile_options(openmx_exx PRIVATE
  "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:IntelLLVM>:-Wall;-Wextra;-fp-model=precise;-ffp-contract=off>"
  "$<$<CXX_COMPILER_ID:MSVC>:/fp:strict>")
if(OPENMX_EXX_SANITIZERS OR OPENMX_CORE_SANITIZERS)
  target_compile_options(openmx_exx PUBLIC
    -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(openmx_exx PUBLIC -fsanitize=address,undefined)
endif()

install(TARGETS openmx_exx EXPORT OpenMXExxTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/OpenMXExx)
set(_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/OpenMXExx")
configure_package_config_file(cmake/OpenMXExxConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXExxConfig.cmake"
  INSTALL_DESTINATION "${_package_dir}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/OpenMXExxConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT OpenMXExxTargets NAMESPACE OpenMX:: DESTINATION "${_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/OpenMXExxConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/OpenMXExxConfigVersion.cmake" DESTINATION "${_package_dir}")

option(OPENMX_EXX_BUILD_TESTING "Build the independent exx tests" ${BUILD_TESTING})
if(BUILD_TESTING AND OPENMX_EXX_BUILD_TESTING)
  # The package-consumer test builds a copy of this directory against an
  # installed basis package: tell it where the basis package comes from.
  get_target_property(_exx_basis_imported OpenMX::basis IMPORTED)
  set(_exx_basis_source "")
  set(_exx_basis_package_dir "")
  if(_exx_basis_imported)
    set(_exx_basis_package_dir "${OpenMXBasis_DIR}")
  else()
    get_target_property(_exx_basis_target OpenMX::basis ALIASED_TARGET)
    get_target_property(_exx_basis_source ${_exx_basis_target} SOURCE_DIR)
  endif()
  add_subdirectory(tests)
endif()
