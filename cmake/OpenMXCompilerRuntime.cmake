# Compiler drivers know their runtime directories at link time, but that does
# not make those directories available to the dynamic loader. Record them on
# each DSO: an executable's DT_RUNPATH does not cover a DSO's dependencies.
function(openmx_compiler_runtime_rpath target language)
  set(directories)
  foreach(library IN LISTS CMAKE_${language}_IMPLICIT_LINK_LIBRARIES)
    unset(runtime)
    find_library(runtime NAMES "${library}" PATHS ${CMAKE_${language}_IMPLICIT_LINK_DIRECTORIES}
      NO_DEFAULT_PATH NO_CACHE)
    if(runtime)
      get_filename_component(real_library "${runtime}" REALPATH)
      get_filename_component(directory "${real_library}" DIRECTORY)
      list(APPEND directories "${directory}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES directories)
  set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH ${directories})
  set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH ${directories})
endfunction()
