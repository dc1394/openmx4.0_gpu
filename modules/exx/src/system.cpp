#include <openmx/exx/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace openmx::exx {
namespace {

using Vec = std::array<double,3>;

Vec cross(const Vec& a, const Vec& b)
{
  return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
}

double dot(const Vec& a, const Vec& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

void check(const Structure& structure, std::size_t species_count)
{
  if (structure.positions.empty() || structure.positions.size() != structure.species.size())
    throw std::invalid_argument("a structure needs atoms, each with a position and a species");
  for (const Vec& v : structure.lattice)
    for (const double x : v)
      if (!std::isfinite(x)) throw std::invalid_argument("a lattice vector is not finite");
  for (const Vec& v : structure.positions)
    for (const double x : v)
      if (!std::isfinite(x)) throw std::invalid_argument("the position of an atom is not finite");
  for (const int s : structure.species)
    if (s < 0 || static_cast<std::size_t>(s) >= species_count)
      throw std::invalid_argument("the species of an atom is out of range");
  const double volume = dot(structure.lattice[0],cross(structure.lattice[1],structure.lattice[2]));
  if (!(std::fabs(volume) > 0.0)) throw std::invalid_argument("the lattice vectors do not span a volume");
}

Site reversed(int atom, const Site& site)
{
  return {atom,{-site.cell[0],-site.cell[1],-site.cell[2]}};
}

// Of the two descriptions of one pair, (a; b, R) and (b; a, -R), the one that
// is computed: the smaller atom first, and for one atom the cell that is
// larger than its negative.
bool computed_here(int atom, const Site& site)
{
  if (site.atom != atom) return atom < site.atom;
  return site.cell > reversed(atom,site).cell;
}

} // namespace

std::array<double,3> displacement(const Structure& structure, int atom, const Site& site)
{
  const std::size_t atoms = structure.positions.size();
  if (atom < 0 || static_cast<std::size_t>(atom) >= atoms || site.atom < 0 ||
      static_cast<std::size_t>(site.atom) >= atoms)
    throw std::out_of_range("an atom of a site is not in the structure");
  const Vec& from = structure.positions[static_cast<std::size_t>(atom)];
  const Vec& to = structure.positions[static_cast<std::size_t>(site.atom)];
  Vec d{};
  for (std::size_t x = 0; x < 3; ++x) {
    double shift = 0.0;
    for (std::size_t i = 0; i < 3; ++i) shift += static_cast<double>(site.cell[i])*structure.lattice[i][x];
    d[x] = (to[x]-from[x])+shift;
  }
  return d;
}

std::vector<std::vector<Site>> neighbour_lists(const Structure& structure, std::span<const double> cutoffs,
                                               std::size_t species_count)
{
  check(structure,species_count);
  if (cutoffs.size() != species_count*species_count)
    throw std::invalid_argument("the table of cutoff distances does not match the number of species");
  for (std::size_t s = 0; s < species_count; ++s)
    for (std::size_t t = 0; t < species_count; ++t) {
      const double cutoff = cutoffs[s*species_count+t];
      if (!(cutoff >= 0.0) || !std::isfinite(cutoff) || cutoff != cutoffs[t*species_count+s])
        throw std::invalid_argument("the cutoff distances must be finite, nonnegative and symmetric");
    }

  // Component i of a vector d in units of the lattice vectors is
  // d.(a_j x a_k)/V, and the planes of the lattice are |V|/|a_j x a_k| apart.
  const std::array<Vec,3> normal{cross(structure.lattice[1],structure.lattice[2]),
                                 cross(structure.lattice[2],structure.lattice[0]),
                                 cross(structure.lattice[0],structure.lattice[1])};
  const double volume = dot(structure.lattice[0],normal[0]);
  const std::size_t atoms = structure.positions.size();
  std::vector<std::vector<Site>> lists(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    std::vector<Site>& list = lists[a];
    list.push_back({static_cast<int>(a),{0,0,0}});
    const std::size_t sa = static_cast<std::size_t>(structure.species[a]);
    for (std::size_t b = 0; b < atoms; ++b) {
      const double cutoff = cutoffs[sa*species_count+static_cast<std::size_t>(structure.species[b])];
      Vec delta{};
      for (std::size_t x = 0; x < 3; ++x) delta[x] = structure.positions[b][x]-structure.positions[a][x];
      std::array<int,3> low{}, high{};
      for (std::size_t i = 0; i < 3; ++i) {
        const double fraction = dot(delta,normal[i])/volume;
        const double reach = cutoff*std::sqrt(dot(normal[i],normal[i]))/std::fabs(volume);
        if (!(reach < 1e6)) throw std::invalid_argument("a cutoff distance spans too many cells");
        // The margin lets rounding of the bounds add candidates, never lose
        // one; the distance decides below.
        low[i] = static_cast<int>(std::ceil(-reach-fraction-1e-8));
        high[i] = static_cast<int>(std::floor(reach-fraction+1e-8));
      }
      for (int n1 = low[0]; n1 <= high[0]; ++n1)
        for (int n2 = low[1]; n2 <= high[1]; ++n2)
          for (int n3 = low[2]; n3 <= high[2]; ++n3) {
            if (b == a && n1 == 0 && n2 == 0 && n3 == 0) continue;
            const Site site{static_cast<int>(b),{n1,n2,n3}};
            const Vec d = displacement(structure,static_cast<int>(a),site);
            if (std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]) <= cutoff) list.push_back(site);
          }
    }
  }
  return lists;
}

std::size_t find_site(std::span<const Site> list, const Site& site)
{
  if (list.empty()) return 0;
  if (list.front() == site) return 0;
  // After the first entry the list is ordered by atom and cell.
  const auto less = [](const Site& a, const Site& b) { return a.atom != b.atom ? a.atom < b.atom : a.cell < b.cell; };
  const auto found = std::lower_bound(list.begin()+1,list.end(),site,less);
  return found != list.end() && *found == site ? static_cast<std::size_t>(found-list.begin()) : list.size();
}

std::vector<double> t_block(const System& system, std::size_t atom, std::size_t neighbour)
{
  if (atom >= system.t.size() || atom >= system.c_offset.size() || neighbour+1 >= system.c_offset[atom].size())
    throw std::out_of_range("a T block of a system does not exist");
  const std::size_t rows = system.abf_count[atom]*system.orbital_count[atom];
  const std::size_t first = system.c_offset[atom][neighbour], width = system.c_offset[atom][neighbour+1]-first;
  const std::size_t columns = system.c_offset[atom].back();
  std::vector<double> block(rows*width);
  for (std::size_t row = 0; row < rows; ++row)
    std::copy_n(system.t[atom].begin()+static_cast<std::ptrdiff_t>(row*columns+first),width,
                block.begin()+static_cast<std::ptrdiff_t>(row*width));
  return block;
}

System prepare_system(const Structure& structure, std::span<const SpeciesTable> species,
                      const ReciprocalGrid& reciprocal, const SystemOptions& options)
{
  const std::size_t count = species.size(), atoms = structure.positions.size();
  check(structure,count);
  if (!(options.v_cutoff >= 0.0) || !std::isfinite(options.v_cutoff))
    throw std::invalid_argument("the cutoff of the V blocks must be finite and nonnegative");

  int maximum_orbital = 0, maximum_abf = 0;
  std::vector<double> c_cutoff(count*count), v_cutoff(count*count);
  for (std::size_t s = 0; s < count; ++s) {
    maximum_orbital = std::max(maximum_orbital,species[s].maximum_orbital_angular());
    maximum_abf = std::max(maximum_abf,species[s].maximum_abf_angular());
    for (std::size_t t = 0; t < count; ++t) {
      c_cutoff[s*count+t] = species[s].radius()+species[t].radius();
      v_cutoff[s*count+t] = std::max(species[s].radius(),species[t].radius())+options.v_cutoff;
    }
  }

  System system;
  system.orbital_count.resize(atoms);
  system.abf_count.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    const SpeciesTable& table = species[static_cast<std::size_t>(structure.species[a])];
    system.orbital_count[a] = table.orbital_count();
    system.abf_count[a] = table.abf_count();
  }
  system.c_neighbours = neighbour_lists(structure,c_cutoff,count);
  system.v_neighbours = neighbour_lists(structure,v_cutoff,count);

  // T blocks: equation (7) once for every pair of sites.  The blocks of an
  // atom lie side by side.
  system.c_offset.resize(atoms);
  system.t.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    std::vector<std::size_t>& offset = system.c_offset[a];
    offset.assign(1,0);
    for (const Site& site : system.c_neighbours[a])
      offset.push_back(offset.back()+system.orbital_count[static_cast<std::size_t>(site.atom)]);
    system.t[a].assign(system.abf_count[a]*system.orbital_count[a]*offset.back(),0.0);
  }
  // Copies block[(P*u + mu)*width + lambda] into the columns of neighbour i.
  const auto store = [&system](std::size_t a, std::size_t i, const std::vector<double>& block) {
    const std::size_t rows = system.abf_count[a]*system.orbital_count[a];
    const std::size_t first = system.c_offset[a][i], width = system.c_offset[a][i+1]-first;
    const std::size_t columns = system.c_offset[a].back();
    if (block.size() != rows*width) throw std::logic_error("a T block has the wrong size");
    for (std::size_t row = 0; row < rows; ++row)
      std::copy_n(block.begin()+static_cast<std::ptrdiff_t>(row*width),width,
                  system.t[a].begin()+static_cast<std::ptrdiff_t>(row*columns+first));
  };
  const LriSolver solver(reciprocal,maximum_orbital,maximum_abf,options.lri);
  std::vector<std::vector<double>> onsite_t(count);
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t sa = static_cast<std::size_t>(structure.species[a]);
    if (onsite_t[sa].empty()) onsite_t[sa] = solver.onsite(species[sa],&system.report);
    store(a,0,onsite_t[sa]);
    for (std::size_t i = 1; i < system.c_neighbours[a].size(); ++i) {
      const Site& site = system.c_neighbours[a][i];
      if (!computed_here(static_cast<int>(a),site)) continue;
      const std::size_t b = static_cast<std::size_t>(site.atom);
      const std::size_t j = find_site(system.c_neighbours[b],reversed(static_cast<int>(a),site));
      if (j == system.c_neighbours[b].size()) throw std::logic_error("a C neighbour has no counterpart");
      const PairCoefficients pair = solver.pair(species[sa],species[static_cast<std::size_t>(structure.species[b])],
                                                displacement(structure,static_cast<int>(a),site),&system.report);
      store(a,i,pair.first);
      store(b,j,pair.second);
    }
  }

  // V blocks: equation (12) with the interaction of the exchange, once for
  // every pair of sites; the other atom gets the transpose.
  const TwoCenterIntegrator interaction(reciprocal,options.interaction,maximum_abf,maximum_abf);
  std::vector<std::vector<double>> onsite_v(count);
  system.v.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) system.v[a].resize(system.v_neighbours[a].size());
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t sa = static_cast<std::size_t>(structure.species[a]);
    const std::size_t na = system.abf_count[a];
    if (onsite_v[sa].empty()) {
      const std::vector<double> block = two_center_matrix(interaction.pair({0.0,0.0,0.0}),
                                                          species[sa].abf_transforms(),species[sa].abf_transforms());
      onsite_v[sa].resize(na*na);
      for (std::size_t p = 0; p < na; ++p)
        for (std::size_t q = 0; q < na; ++q) onsite_v[sa][p*na+q] = 0.5*(block[p*na+q]+block[q*na+p]);
    }
    system.v[a][0] = onsite_v[sa];
    for (std::size_t i = 1; i < system.v_neighbours[a].size(); ++i) {
      const Site& site = system.v_neighbours[a][i];
      if (!computed_here(static_cast<int>(a),site)) continue;
      const std::size_t b = static_cast<std::size_t>(site.atom);
      const std::size_t j = find_site(system.v_neighbours[b],reversed(static_cast<int>(a),site));
      if (j == system.v_neighbours[b].size()) throw std::logic_error("a V neighbour has no counterpart");
      const std::size_t nb = system.abf_count[b];
      std::vector<double> block =
          two_center_matrix(interaction.pair(displacement(structure,static_cast<int>(a),site)),
                            species[sa].abf_transforms(),
                            species[static_cast<std::size_t>(structure.species[b])].abf_transforms());
      std::vector<double> transposed(nb*na);
      for (std::size_t p = 0; p < na; ++p)
        for (std::size_t q = 0; q < nb; ++q) transposed[q*na+p] = block[p*nb+q];
      system.v[a][i] = std::move(block);
      system.v[b][j] = std::move(transposed);
    }
  }
  return system;
}

} // namespace openmx::exx
