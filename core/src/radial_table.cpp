#include <openmx/core/radial_table.hpp>

#include <stdexcept>

namespace openmx::core {

namespace {
template<class Table>
RadialSamples sample_rows(const Table& table, RadialQuadrature grid)
{
  std::vector<Orbital> orbitals;
  orbitals.reserve(table.channel_count());
  Tensor<double,2> values({table.channel_count(),grid.radii.size()});
  auto data = values.view();
  for (std::size_t channel = 0; channel < table.channel_count(); ++channel) {
    orbitals.push_back({table.angular(channel),static_cast<int>(channel)});
    for (std::size_t i = 0; i < grid.radii.size(); ++i)
      element(data,channel,i) = table.sample(channel,grid.radii[i]);
  }
  return {std::move(grid),std::move(orbitals),std::move(values),{}};
}
} // namespace

RadialSamples sample_table(const RadialTable& table, RadialQuadrature grid)
{
  return sample_rows(table,std::move(grid));
}

RadialSamples sample_table(const RadialTableView& table, RadialQuadrature grid)
{
  return sample_rows(table,std::move(grid));
}

RadialSamples sample_product_table(const RadialTableView& table,
    const ProductLayout& layout, RadialQuadrature grid)
{
  if (table.channel_count() != layout.orbital_count())
    throw std::invalid_argument("product basis does not match input rows");
  std::size_t row = 0;
  const auto counts = layout.multiplicities();
  for (std::size_t l = 0; l < counts.size(); ++l)
    for (int m = 0; m < counts[l]; ++m)
      if (table.angular(row++) != static_cast<int>(l))
        throw std::invalid_argument("product input angular order mismatch");
  auto samples = sample_rows(table,std::move(grid));
  samples.products.assign(layout.channels().begin(),layout.channels().end());
  return samples;
}

} // namespace openmx::core
