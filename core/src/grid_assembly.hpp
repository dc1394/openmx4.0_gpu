#pragma once
#include "openmx_core_grid.h"
struct GridWorkspaceDelete {
  void operator()(OpenMXGridWorkspace*) const noexcept;
};
