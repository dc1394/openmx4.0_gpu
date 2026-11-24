#include <openmx/exx/forces.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace openmx::exx {
namespace {

int maximum_orbital(std::span<const SpeciesTable> species)
{
  int maximum = 0;
  for (const SpeciesTable& table : species) maximum = std::max(maximum,table.maximum_orbital_angular());
  return maximum;
}

int maximum_abf(std::span<const SpeciesTable> species)
{
  int maximum = 0;
  for (const SpeciesTable& table : species) maximum = std::max(maximum,table.maximum_abf_angular());
  return maximum;
}

} // namespace

PositionGradient::PositionGradient(const Structure& structure, std::span<const SpeciesTable> species,
    const ReciprocalGrid& reciprocal, const SystemOptions& options, const System& system)
  : structure_(&structure),
    species_(species),
    system_(&system),
    solver_(reciprocal,maximum_orbital(species),maximum_abf(species),options.lri),
    interaction_(reciprocal,options.interaction,maximum_abf(species),maximum_abf(species))
{
  const std::size_t atoms = structure.positions.size();
  if (structure.species.size() != atoms || system.orbital_count.size() != atoms || system.abf_count.size() != atoms ||
      system.c_neighbours.size() != atoms || system.v_neighbours.size() != atoms || system.c_offset.size() != atoms ||
      system.t.size() != atoms || system.v.size() != atoms)
    throw std::invalid_argument("the system was not prepared for the structure");
  for (std::size_t a = 0; a < atoms; ++a) {
    const int s = structure.species[a];
    if (s < 0 || static_cast<std::size_t>(s) >= species.size())
      throw std::invalid_argument("the species of an atom is out of range");
    const SpeciesTable& item = species[static_cast<std::size_t>(s)];
    if (item.orbital_count() != system.orbital_count[a] || item.abf_count() != system.abf_count[a])
      throw std::invalid_argument("the system was not prepared with the tables of the species");
  }
}

const SpeciesTable& PositionGradient::table(std::size_t atom) const
{
  return species_[static_cast<std::size_t>(structure_->species[atom])];
}

void PositionGradient::add(std::size_t atom, const CentreGradient& weights, std::span<double> gradient) const
{
  const System& system = *system_;
  const std::size_t atoms = system.orbital_count.size();
  if (atom >= atoms) throw std::out_of_range("the centre atom of a derivative is not in the unit cell");
  if (gradient.size() != 3*atoms)
    throw std::invalid_argument("the derivative with respect to the positions needs three entries for every atom");
  if (weights.t.size() != system.t[atom].size() || weights.v.size() != system.v[atom].size())
    throw std::invalid_argument("the weights of the blocks do not have the layout of the centre atom");
  const int a = static_cast<int>(atom);
  const SpeciesTable& table_a = table(atom);
  const std::size_t na = system.abf_count[atom], ua = system.orbital_count[atom];
  // g = dE/dd for a displacement that points from atom "from" to atom "to".
  const auto apply = [&gradient](std::size_t from, std::size_t to, const std::array<double,3>& g) {
    for (std::size_t x = 0; x < 3; ++x) {
      gradient[3*to+x] += g[x];
      gradient[3*from+x] -= g[x];
    }
  };

  // T blocks: equation (18) for every pair with another atom.
  const std::vector<Site>& c_sites = system.c_neighbours[atom];
  const std::size_t columns = system.c_offset[atom].back(), rows = na*ua;
  for (std::size_t i = 1; i < c_sites.size(); ++i) {
    const Site& site = c_sites[i];
    if (site.atom == a) continue;
    const std::size_t c = static_cast<std::size_t>(site.atom);
    const Site back = reversed_site(a,site);
    const std::size_t j = find_site(system.c_neighbours[c],back);
    if (j == system.c_neighbours[c].size()) throw std::logic_error("a C neighbour has no counterpart");
    const std::size_t first = system.c_offset[atom][i], width = system.c_offset[atom][i+1]-first;
    // The weights of the block of this neighbour, [(P*u_a + mu)*u_c + lambda].
    std::vector<double> block(rows*width);
    for (std::size_t row = 0; row < rows; ++row)
      std::copy_n(weights.t.begin()+static_cast<std::ptrdiff_t>(row*columns+first),width,
                  block.begin()+static_cast<std::ptrdiff_t>(row*width));
    PairCoefficients coefficients, pair_weights;
    if (pair_computed_from(a,site)) {
      coefficients.first = t_block(system,atom,i);
      coefficients.second = t_block(system,c,j);
      pair_weights.first = std::move(block);
      pair_weights.second.assign(coefficients.second.size(),0.0);
      apply(atom,c,solver_.pair_gradient(table_a,table(c),displacement(*structure_,a,site),coefficients,pair_weights));
    } else {
      // The pair was solved from the other atom; the block of this atom is
      // the second part of that solution.
      coefficients.first = t_block(system,c,j);
      coefficients.second = t_block(system,atom,i);
      pair_weights.first.assign(coefficients.first.size(),0.0);
      pair_weights.second = std::move(block);
      apply(c,atom,solver_.pair_gradient(table(c),table_a,displacement(*structure_,site.atom,back),coefficients,
                                         pair_weights));
    }
  }

  // V blocks: the derivative of equation (12) with the interaction of the
  // exchange.
  const std::vector<Site>& v_sites = system.v_neighbours[atom];
  for (std::size_t i = 1; i < v_sites.size(); ++i) {
    const Site& site = v_sites[i];
    if (site.atom == a) continue;
    const std::size_t b = static_cast<std::size_t>(site.atom), nb = system.abf_count[b];
    const std::vector<double>& w = weights.v[i];
    if (w.size() != na*nb || system.v[atom][i].size() != na*nb)
      throw std::invalid_argument("the weights of a V block have the wrong size");
    std::array<double,3> g{};
    if (pair_computed_from(a,site)) {
      const std::vector<double> derivative =
          two_center_gradient_matrix(interaction_.gradient_pair(displacement(*structure_,a,site)),
                                     table_a.abf_transforms(),table(b).abf_transforms());
      for (std::size_t x = 0; x < 3; ++x) {
        const double* values = derivative.data()+x*na*nb;
        double sum = 0.0;
        for (std::size_t e = 0; e < na*nb; ++e) sum += w[e]*values[e];
        g[x] = sum;
      }
      apply(atom,b,g);
    } else {
      // The stored block is the transpose of the block computed from the
      // other atom.
      const std::vector<double> derivative =
          two_center_gradient_matrix(interaction_.gradient_pair(displacement(*structure_,site.atom,reversed_site(a,site))),
                                     table(b).abf_transforms(),table_a.abf_transforms());
      for (std::size_t x = 0; x < 3; ++x) {
        const double* values = derivative.data()+x*nb*na;
        double sum = 0.0;
        for (std::size_t p = 0; p < na; ++p)
          for (std::size_t q = 0; q < nb; ++q) sum += w[p*nb+q]*values[q*na+p];
        g[x] = sum;
      }
      apply(b,atom,g);
    }
  }
}

void PositionGradient::add_gamma(std::size_t atom, double factor, std::span<const double> density,
    std::span<double> gradient, const DenseBackend& backend) const
{
  add(atom,centre_gradient_gamma(*system_,atom,factor,density,backend),gradient);
}

void PositionGradient::add_blocks(const BlockLayout& layout, std::size_t atom, double factor,
    std::span<const double> density, std::span<double> gradient, const DenseBackend& backend) const
{
  add(atom,centre_gradient_blocks(*system_,layout,atom,factor,density,backend),gradient);
}

} // namespace openmx::exx
