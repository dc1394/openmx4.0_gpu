#pragma once

#include <openmx/basis/radial_table.hpp>
#include <openmx/core/radial.hpp>

namespace openmx::core {

// Transitional source compatibility. The independent basis library owns
// these types and numerical functions; no second implementation lives here.
using basis::RadialKind;
using basis::ReciprocalKind;
using basis::RadialChannel;
using basis::RadialTableView;
using basis::RadialTable;
using basis::sample_radial_row;
using basis::sample_reciprocal_row;

RadialSamples sample_table(const RadialTable& table, RadialQuadrature grid);
RadialSamples sample_table(const RadialTableView& table, RadialQuadrature grid);
RadialSamples sample_product_table(const RadialTableView& table,
    const ProductLayout& layout, RadialQuadrature grid);

} // namespace openmx::core
