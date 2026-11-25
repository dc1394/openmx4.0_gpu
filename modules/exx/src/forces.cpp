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

void add_block_weights(const CentreGradient& weights, std::span<double> values)
{
  std::size_t elements = weights.t.size();
  for (const std::vector<double>& block : weights.v) elements += block.size();
  if (values.size() != elements)
    throw std::invalid_argument("the array does not have the size of the weights of the blocks of the atom");
  double* target = values.data();
  for (const double value : weights.t) *target++ += value;
  for (const std::vector<double>& block : weights.v)
    for (const double value : block) *target++ += value;
}

namespace {

// The ABFs first <= P < last of the unit cell, atom by atom: compute(atom,
// from, to) gives the derivatives of the ABFs from <= P < to of the atom,
// counted within the atom, and their rows are added to weights[atom], the
// array of add_block_weights() for the whole atom.  device(visit) offers the
// parts to the device first (device_gradients_blocks() or
// device_gradients_gamma()): it hands the parts that the device computed to
// visit and returns their number, and compute() gives the others.
template <class Compute, class Device>
void add_weight_rows(const System& system, std::size_t first, std::size_t last,
                     std::span<const std::span<double>> weights, const Compute& compute, const Device& device)
{
  const std::size_t atoms = system.orbital_count.size();
  if (system.abf_count.size() != atoms || system.c_offset.size() != atoms || system.v_neighbours.size() != atoms)
    throw std::invalid_argument("the system of the weights of the blocks is not a prepared one");
  std::vector<std::size_t> offset(1,0);
  for (const std::size_t count : system.abf_count) offset.push_back(offset.back()+count);
  if (first > last || last > offset.back()) throw std::out_of_range("the range of ABFs is not inside the unit cell");
  if (weights.size() != atoms) throw std::invalid_argument("the weights of the blocks need an array for every atom");
  // The arrays of all atoms of the range are checked before the first one is
  // changed.
  for (std::size_t a = 0; a < atoms; ++a)
    if (std::max(first,offset[a]) < std::min(last,offset[a+1]) && weights[a].size() != table_elements(system,a))
      throw std::invalid_argument("the array does not have the size of the weights of the blocks of the atom");
  // The rows of the ABFs begin <= P < begin + count of atom a.
  const auto add = [&system,weights](std::size_t a, std::size_t begin, std::size_t count, const CentreGradient& rows) {
    const std::size_t width = system.orbital_count[a]*system.c_offset[a].back();
    const std::vector<Site>& neighbours = system.v_neighbours[a];
    if (rows.t.size() != count*width || rows.v.size() != neighbours.size())
      throw std::logic_error("the derivatives of a part of the ABFs do not have the size of the part");
    for (std::size_t i = 0; i < neighbours.size(); ++i)
      if (rows.v[i].size() != count*system.abf_count[static_cast<std::size_t>(neighbours[i].atom)])
        throw std::logic_error("the derivatives of a part of the ABFs do not have the size of the part");
    double* target = weights[a].data()+begin*width;
    for (const double value : rows.t) *target++ += value;
    // The V blocks follow the T blocks of all ABFs of the atom.
    std::size_t block = system.abf_count[a]*width;
    for (std::size_t i = 0; i < neighbours.size(); ++i) {
      const std::size_t nq = system.abf_count[static_cast<std::size_t>(neighbours[i].atom)];
      target = weights[a].data()+block+begin*nq;
      for (const double value : rows.v[i]) *target++ += value;
      block += system.abf_count[a]*nq;
    }
  };
  // The parts of the device come in the order of the atoms below.
  const std::size_t done = device([&add](std::size_t a, std::size_t from, std::size_t to, CentreGradient&& rows) {
    add(a,from,to-from,rows);
  });
  std::size_t part = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t from = std::max(first,offset[a]), to = std::min(last,offset[a+1]);
    if (from >= to || part++ < done) continue;
    const std::size_t begin = from-offset[a], count = to-from;
    add(a,begin,count,compute(a,begin,begin+count));
  }
}

} // namespace

void add_block_weights_gamma_abfs(const System& system, std::size_t first, std::size_t last, double factor,
                                  std::span<const double> density, std::span<const std::span<double>> weights,
                                  const DenseBackend& backend)
{
  add_weight_rows(
      system,first,last,weights,
      [&](std::size_t atom, std::size_t from, std::size_t to) {
        return centre_gradient_gamma(system,atom,from,to,factor,density,backend);
      },
      [&](const GradientVisit& visit) {
        return device_gradients_gamma(system,first,last,factor,density,backend,visit);
      });
}

void add_block_weights_blocks_abfs(const System& system, const BlockLayout& layout, std::size_t first,
                                   std::size_t last, double factor, std::span<const double> density,
                                   std::span<const std::span<double>> weights, const DenseBackend& backend)
{
  add_weight_rows(
      system,first,last,weights,
      [&](std::size_t atom, std::size_t from, std::size_t to) {
        return centre_gradient_blocks(system,layout,atom,from,to,factor,density,backend);
      },
      [&](const GradientVisit& visit) {
        return device_gradients_blocks(system,layout,first,last,factor,density,backend,visit);
      });
}

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

void PositionGradient::add_pairs(std::span<const std::span<const double>> weights, const PrepareShare& share,
    std::span<double> gradient) const
{
  const System& system = *system_;
  const std::size_t atoms = system.orbital_count.size();
  if (gradient.size() != 3*atoms)
    throw std::invalid_argument("the derivative with respect to the positions needs three entries for every atom");
  if (share.parts < 1 || share.part >= share.parts)
    throw std::invalid_argument("the share of a derivative is not one of its parts");
  if (weights.size() != atoms) throw std::invalid_argument("the weights of the blocks are needed for every atom");
  // The first element of the weights of every V block of an atom.
  std::vector<std::vector<std::size_t>> v_offset(atoms);
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    std::size_t elements = system.t[atom].size();
    for (const std::vector<double>& block : system.v[atom]) {
      v_offset[atom].push_back(elements);
      elements += block.size();
    }
    if (weights[atom].size() != elements)
      throw std::invalid_argument("the weights of the blocks do not have the layout of their atom");
  }
  const auto mine = [&share](std::size_t pair) { return pair%share.parts == share.part; };
  // g = dE/dd for a displacement that points from atom "from" to atom "to".
  const auto apply = [&gradient](std::size_t from, std::size_t to, const std::array<double,3>& g) {
    for (std::size_t x = 0; x < 3; ++x) {
      gradient[3*to+x] += g[x];
      gradient[3*from+x] -= g[x];
    }
  };
  // The weights of the T block of an atom and its C neighbour i,
  // [(P*u + mu)*u_c + lambda].
  const auto t_weights = [&system,&weights](std::size_t atom, std::size_t i) {
    const std::size_t rows = system.abf_count[atom]*system.orbital_count[atom];
    const std::size_t first = system.c_offset[atom][i], width = system.c_offset[atom][i+1]-first;
    const std::size_t columns = system.c_offset[atom].back();
    std::vector<double> block(rows*width);
    for (std::size_t row = 0; row < rows; ++row)
      std::copy_n(weights[atom].begin()+static_cast<std::ptrdiff_t>(row*columns+first),width,
                  block.begin()+static_cast<std::ptrdiff_t>(row*width));
    return block;
  };

  // T blocks: equation (18) with the weights of the blocks of both atoms.
  std::size_t t_pair = 0;
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    const int a = static_cast<int>(atom);
    const std::vector<Site>& c_sites = system.c_neighbours[atom];
    for (std::size_t i = 1; i < c_sites.size(); ++i) {
      const Site& site = c_sites[i];
      // Two images of one atom: the displacement is a lattice vector.
      if (site.atom == a || !pair_computed_from(a,site)) continue;
      if (!mine(t_pair++)) continue;
      const std::size_t c = static_cast<std::size_t>(site.atom);
      const std::size_t j = find_site(system.c_neighbours[c],reversed_site(a,site));
      if (j == system.c_neighbours[c].size()) throw std::logic_error("a C neighbour has no counterpart");
      const PairCoefficients coefficients{t_block(system,atom,i),t_block(system,c,j)};
      const PairCoefficients pair_weights{t_weights(atom,i),t_weights(c,j)};
      apply(atom,c,solver_.pair_gradient(table(atom),table(c),displacement(*structure_,a,site),coefficients,pair_weights));
    }
  }

  // V blocks: the derivative of equation (12).  The block of the other atom
  // is the transpose, and its weights are added before the sum.
  std::size_t v_pair = 0;
  for (std::size_t atom = 0; atom < atoms; ++atom) {
    const int a = static_cast<int>(atom);
    const std::size_t na = system.abf_count[atom];
    const std::vector<Site>& v_sites = system.v_neighbours[atom];
    for (std::size_t i = 1; i < v_sites.size(); ++i) {
      const Site& site = v_sites[i];
      if (site.atom == a || !pair_computed_from(a,site)) continue;
      if (!mine(v_pair++)) continue;
      const std::size_t b = static_cast<std::size_t>(site.atom), nb = system.abf_count[b];
      const std::size_t j = find_site(system.v_neighbours[b],reversed_site(a,site));
      if (j == system.v_neighbours[b].size()) throw std::logic_error("a V neighbour has no counterpart");
      if (system.v[atom][i].size() != na*nb || system.v[b][j].size() != nb*na)
        throw std::invalid_argument("a V block has the wrong size");
      const double* w = weights[atom].data()+v_offset[atom][i];        // [p*nb + q]
      const double* back = weights[b].data()+v_offset[b][j];          // [q*na + p]
      const std::vector<double> derivative =
          two_center_gradient_matrix(interaction_.gradient_pair(displacement(*structure_,a,site)),
                                     table(atom).abf_transforms(),table(b).abf_transforms());
      std::array<double,3> g{};
      for (std::size_t x = 0; x < 3; ++x) {
        const double* values = derivative.data()+x*na*nb;
        double sum = 0.0;
        for (std::size_t p = 0; p < na; ++p)
          for (std::size_t q = 0; q < nb; ++q) sum += (w[p*nb+q]+back[q*na+p])*values[p*nb+q];
        g[x] = sum;
      }
      apply(atom,b,g);
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
