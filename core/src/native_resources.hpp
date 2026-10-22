#pragma once
#include "openmx_core_native_resources.h"
#include <map>
#include <memory>
#include <string>

struct NativeResourceGroups {
  struct Delete { void operator()(OpenMXNativeResources*) const noexcept; };
  using Group = std::unique_ptr<OpenMXNativeResources, Delete>;
  std::map<std::string,Group,std::less<>> groups;
  OpenMXNativeResources* get(const char*,OpenMXNativeResourceReset);
  OpenMXNativeResources* get_state(const char*,size_t,OpenMXNativeStateReset);
};
