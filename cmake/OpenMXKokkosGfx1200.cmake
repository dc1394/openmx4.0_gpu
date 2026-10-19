# Kokkos 5.2.2 supports RDNA4 gfx1201 but omits gfx1200 (RX 9060 XT).
# Patch only the verified FetchContent extraction, preserving the release pin.
# Both targets use 32-lane waves and do not expose system allocations via XNACK.
if(NOT DEFINED KOKKOS_SOURCE_DIR)
  message(FATAL_ERROR "Set KOKKOS_SOURCE_DIR to the extracted Kokkos 5.2.2 source")
endif()
file(READ "${KOKKOS_SOURCE_DIR}/CMakeLists.txt" _version)
foreach(_part MAJOR MINOR PATCH)
  if(_part STREQUAL "MAJOR")
    set(_number 5)
  else()
    set(_number 2)
  endif()
  if(NOT _version MATCHES "set\\(Kokkos_VERSION_${_part} ${_number}\\)")
    message(FATAL_ERROR "The gfx1200 patch requires the pinned Kokkos 5.2.2 source")
  endif()
endforeach()

# Validate every replacement before writing any file. Repeated application is
# harmless, while an unexpected upstream layout fails without partial edits.
function(_openmx_kokkos_patch relative before after)
  set(_path "${KOKKOS_SOURCE_DIR}/${relative}")
  file(READ "${_path}" _text)
  string(FIND "${_text}" "${after}" _patched)
  if(_patched EQUAL -1)
    string(FIND "${_text}" "${before}" _original)
    if(_original EQUAL -1)
      message(FATAL_ERROR "Unsupported Kokkos source layout: ${relative}")
    endif()
    string(REPLACE "${before}" "${after}" _text "${_text}")
  endif()
  string(SHA256 _key "${relative}")
  set(_content_${_key} "${_text}" PARENT_SCOPE)
  set(_files ${_files} "${relative}" PARENT_SCOPE)
endfunction()

_openmx_kokkos_patch(cmake/kokkos_arch.cmake
  [=[list(APPEND SUPPORTED_AMD_GPUS RX9070XT RX7900XTX RX7800XT)
list(APPEND SUPPORTED_AMD_ARCHS AMD_GFX1201 AMD_GFX1100 AMD_GFX1101)
list(APPEND CORRESPONDING_AMD_FLAGS gfx1201 gfx1100 gfx1101)]=]
  [=[list(APPEND SUPPORTED_AMD_GPUS RX9060XT RX9070XT RX7900XTX RX7800XT)
list(APPEND SUPPORTED_AMD_ARCHS AMD_GFX1200 AMD_GFX1201 AMD_GFX1100 AMD_GFX1101)
list(APPEND CORRESPONDING_AMD_FLAGS gfx1200 gfx1201 gfx1100 gfx1101)]=])
_openmx_kokkos_patch(cmake/KokkosCore_config.h.in
  [=[#cmakedefine KOKKOS_ARCH_AMD_GFX1201]=]
  [=[#cmakedefine KOKKOS_ARCH_AMD_GFX1200
#cmakedefine KOKKOS_ARCH_AMD_GFX1201]=])
foreach(_file HIP/Kokkos_HIP_Instance.hpp HIP/Kokkos_HIP_IsXnack.hpp)
  _openmx_kokkos_patch("core/src/${_file}"
    [=[defined(KOKKOS_ARCH_AMD_GFX1201)]=]
    [=[defined(KOKKOS_ARCH_AMD_GFX1200) || defined(KOKKOS_ARCH_AMD_GFX1201)]=])
endforeach()
_openmx_kokkos_patch(core/src/impl/Kokkos_Core.cpp
  [=[#elif defined(KOKKOS_ARCH_AMD_GFX1201)
  declare_configuration_metadata("architecture", "GPU architecture", "AMD_GFX1201");]=]
  [=[#elif defined(KOKKOS_ARCH_AMD_GFX1200)
  declare_configuration_metadata("architecture", "GPU architecture", "AMD_GFX1200");
#elif defined(KOKKOS_ARCH_AMD_GFX1201)
  declare_configuration_metadata("architecture", "GPU architecture", "AMD_GFX1201");]=])

foreach(_file IN LISTS _files)
  string(SHA256 _key "${_file}")
  file(READ "${KOKKOS_SOURCE_DIR}/${_file}" _previous)
  if(NOT _previous STREQUAL _content_${_key})
    file(WRITE "${KOKKOS_SOURCE_DIR}/${_file}" "${_content_${_key}}")
  endif()
endforeach()
