#pragma once

#include <openmx/core/radial_table.hpp>

namespace openmx::core {

struct NeutralProjectorBasis {
  int lmax, multiplicity;
  int pao_lmax, pao_multiplicity;
  bool empty_atom = false;
};

struct NeutralProjectors {
  Tensor<double,3> values;   // [angular][multiplicity][VPS radius], includes VNA
  Tensor<double,2> energies;
};

// Borrowed, validated PAO input in (angular,multiplicity) order. No global
// state or retained references. The Gram-Schmidt dependency is evaluated on
// the host in the legacy order; all workspaces and outputs have scoped owners.
NeutralProjectors expand_neutral_projectors(const RadialTableView& pao,
    std::span<const double> radii, std::span<const double> potential,
    NeutralProjectorBasis basis, std::span<const double> nodes,
    std::span<const double> weights);

} // namespace openmx::core
