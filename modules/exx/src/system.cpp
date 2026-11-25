#include <openmx/exx/system.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
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

} // namespace

Site reversed_site(int atom, const Site& site) noexcept
{
  return {atom,{-site.cell[0],-site.cell[1],-site.cell[2]}};
}

bool pair_computed_from(int atom, const Site& site) noexcept
{
  if (site.atom != atom) return atom < site.atom;
  return site.cell > reversed_site(atom,site).cell;
}

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
                      const ReciprocalGrid& reciprocal, const SystemOptions& options, const PrepareShare& share,
                      const DenseBackend& backend)
{
  const std::size_t count = species.size(), atoms = structure.positions.size();
  check(structure,count);
  if (!(options.v_cutoff >= 0.0) || !std::isfinite(options.v_cutoff))
    throw std::invalid_argument("the cutoff of the V blocks must be finite and nonnegative");
  if (share.parts < 1 || share.part >= share.parts)
    throw std::invalid_argument("the share of a preparation is not one of its parts");
  // The pairs of this share, by their number in the order of the loops below.
  const auto mine = [&share](std::size_t pair) { return pair%share.parts == share.part; };

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
  // The pairs of the share: neighbour i of atom a and its counterpart j
  // among the neighbours of atom b.
  struct Pair {
    std::size_t a, i, b, j;
  };
  const auto used_pieces = [&backend](std::size_t pairs) {
    return std::max<std::size_t>(1,std::min(backend.pieces(),pairs));
  };
  // compute(c, pair) for every pair, pair k in piece c = k mod pieces.
  const auto in_pieces = [&backend,&used_pieces](const std::vector<Pair>& pairs, const auto& compute) {
    const std::size_t used = used_pieces(pairs.size());
    backend.parallel_for(used,[&](std::size_t c) {
      for (std::size_t k = c; k < pairs.size(); k += used) compute(c,pairs[k]);
    });
  };
  const LriSolver solver(reciprocal,maximum_orbital,maximum_abf,options.lri);
  std::vector<std::vector<double>> onsite_t(count);
  std::vector<Pair> pairs;
  std::size_t t_pair = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t sa = static_cast<std::size_t>(structure.species[a]);
    if (share.part == 0) {
      if (onsite_t[sa].empty()) onsite_t[sa] = solver.onsite(species[sa],&system.report);
      store(a,0,onsite_t[sa]);
    }
    for (std::size_t i = 1; i < system.c_neighbours[a].size(); ++i) {
      const Site& site = system.c_neighbours[a][i];
      if (!pair_computed_from(static_cast<int>(a),site)) continue;
      if (!mine(t_pair++)) continue;
      const std::size_t b = static_cast<std::size_t>(site.atom);
      const std::size_t j = find_site(system.c_neighbours[b],reversed_site(static_cast<int>(a),site));
      if (j == system.c_neighbours[b].size()) throw std::logic_error("a C neighbour has no counterpart");
      pairs.push_back({a,i,b,j});
    }
  }
  {
    std::vector<LriReport> reports(used_pieces(pairs.size()));
    in_pieces(pairs,[&](std::size_t c, const Pair& p) {
      const Site& site = system.c_neighbours[p.a][p.i];
      const PairCoefficients pair =
          solver.pair(species[static_cast<std::size_t>(structure.species[p.a])],
                      species[static_cast<std::size_t>(structure.species[p.b])],
                      displacement(structure,static_cast<int>(p.a),site),&reports[c]);
      store(p.a,p.i,pair.first);
      store(p.b,p.j,pair.second);
    });
    for (const LriReport& report : reports) system.report.merge(report);
  }

  // V blocks: equation (12) with the interaction of the exchange, once for
  // every pair of sites; the other atom gets the transpose.
  const TwoCenterIntegrator interaction(reciprocal,options.interaction,maximum_abf,maximum_abf);
  std::vector<std::vector<double>> onsite_v(count);
  system.v.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    system.v[a].resize(system.v_neighbours[a].size());
    // Every block is allocated here, on the calling thread, so that where
    // the blocks lie in memory does not depend on the threads that compute
    // them. The blocks of the other parts hold zeros.
    for (std::size_t i = 0; i < system.v[a].size(); ++i)
      system.v[a][i].assign(system.abf_count[a]*system.abf_count[static_cast<std::size_t>(system.v_neighbours[a][i].atom)],
                            0.0);
  }
  pairs.clear();
  std::size_t v_pair = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t sa = static_cast<std::size_t>(structure.species[a]);
    const std::size_t na = system.abf_count[a];
    if (share.part == 0) {
      if (onsite_v[sa].empty()) {
        const std::vector<double> block = two_center_matrix(interaction.pair({0.0,0.0,0.0}),
                                                            species[sa].abf_transforms(),species[sa].abf_transforms());
        onsite_v[sa].resize(na*na);
        for (std::size_t p = 0; p < na; ++p)
          for (std::size_t q = 0; q < na; ++q) onsite_v[sa][p*na+q] = 0.5*(block[p*na+q]+block[q*na+p]);
      }
      system.v[a][0] = onsite_v[sa];
    }
    for (std::size_t i = 1; i < system.v_neighbours[a].size(); ++i) {
      const Site& site = system.v_neighbours[a][i];
      if (!pair_computed_from(static_cast<int>(a),site)) continue;
      if (!mine(v_pair++)) continue;
      const std::size_t b = static_cast<std::size_t>(site.atom);
      const std::size_t j = find_site(system.v_neighbours[b],reversed_site(static_cast<int>(a),site));
      if (j == system.v_neighbours[b].size()) throw std::logic_error("a V neighbour has no counterpart");
      pairs.push_back({a,i,b,j});
    }
  }
  in_pieces(pairs,[&](std::size_t, const Pair& p) {
    const std::size_t na = system.abf_count[p.a], nb = system.abf_count[p.b];
    const std::vector<double> block =
        two_center_matrix(interaction.pair(displacement(structure,static_cast<int>(p.a),system.v_neighbours[p.a][p.i])),
                          species[static_cast<std::size_t>(structure.species[p.a])].abf_transforms(),
                          species[static_cast<std::size_t>(structure.species[p.b])].abf_transforms());
    if (block.size() != system.v[p.a][p.i].size()) throw std::logic_error("a V block does not have the ABFs of its atoms");
    std::copy(block.begin(),block.end(),system.v[p.a][p.i].begin());
    double* transposed = system.v[p.b][p.j].data();
    for (std::size_t r = 0; r < na; ++r)
      for (std::size_t q = 0; q < nb; ++q) transposed[q*na+r] = block[r*nb+q];
  });
  return system;
}

namespace {

// x modulo n in 0 .. n-1.
int floor_mod(int x, int n) noexcept
{
  const int r = x%n;
  return r < 0 ? r+n : r;
}

// The cells after the cell 0 in the order of half_pair() of the build: the
// first component that is not 0 is positive.
bool positive_cell(const std::array<int,3>& cell) noexcept
{
  for (const int c : cell)
    if (c != 0) return c > 0;
  return false;
}

} // namespace

void fold_v_neighbours(System& system, const Structure& structure, const std::array<int,3>& period)
{
  const std::size_t atoms = system.orbital_count.size();
  if (!system.folding.prepared.empty()) throw std::invalid_argument("the V neighbours of the system are folded already");
  for (const int n : period)
    if (n < 1) throw std::invalid_argument("the period of a folding must be at least one cell");
  if (atoms == 0 || structure.positions.size() != atoms || system.abf_count.size() != atoms ||
      system.v_neighbours.size() != atoms || system.v.size() != atoms)
    throw std::invalid_argument("the system was not prepared for the structure");
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::vector<Site>& list = system.v_neighbours[a];
    if (list.empty() || !(list.front() == Site{static_cast<int>(a),{0,0,0}}) || system.v[a].size() != list.size())
      throw std::invalid_argument("the V neighbours of an atom do not begin with the atom or lack their blocks");
    for (std::size_t j = 0; j < list.size(); ++j)
      if (list[j].atom < 0 || static_cast<std::size_t>(list[j].atom) >= atoms ||
          system.v[a][j].size() != system.abf_count[a]*system.abf_count[static_cast<std::size_t>(list[j].atom)])
        throw std::invalid_argument("a V block does not match the ABFs of its V neighbour");
  }

  // A class: the atom of the sites and their cell modulo the period.
  using Key = std::array<int,4>;
  const auto class_of = [&period](int atom, const std::array<int,3>& cell) {
    return Key{atom,floor_mod(cell[0],period[0]),floor_mod(cell[1],period[1]),floor_mod(cell[2],period[2])};
  };
  // The members of every class of every atom, in the order of the list.
  std::vector<std::map<Key,std::vector<std::size_t>>> classes(atoms);
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t j = 0; j < system.v_neighbours[a].size(); ++j) {
      const Site& site = system.v_neighbours[a][j];
      classes[a][class_of(site.atom,site.cell)].push_back(j);
    }

  // The folded V neighbours of every atom, and the folded one of every
  // prepared V neighbour (as a position in the unsorted list).
  struct Folded {
    Site site;
    std::vector<double> block;
  };
  std::vector<std::vector<Folded>> folded(atoms);
  std::vector<std::vector<std::size_t>> entry(atoms);
  for (std::size_t a = 0; a < atoms; ++a) entry[a].assign(system.v_neighbours[a].size(),system.v_neighbours[a].size());
  const auto distance = [&structure](std::size_t a, const Site& site) {
    const std::array<double,3> d = displacement(structure,static_cast<int>(a),site);
    return d[0]*d[0]+d[1]*d[1]+d[2]*d[2];
  };
  // The member of a class closest to atom a, the larger cell at the same
  // distance.
  const auto closest = [&](std::size_t a, const std::vector<std::size_t>& members) {
    std::size_t best = members.front();
    double nearest = distance(a,system.v_neighbours[a][best]);
    for (const std::size_t j : members) {
      const double d = distance(a,system.v_neighbours[a][j]);
      if (d < nearest || (d == nearest && system.v_neighbours[a][j].cell > system.v_neighbours[a][best].cell)) {
        best = j;
        nearest = d;
      }
    }
    return system.v_neighbours[a][best].cell;
  };
  const auto add = [](std::vector<double>& sum, const std::vector<double>& block) {
    if (sum.empty()) {
      sum = block;
      return;
    }
    for (std::size_t e = 0; e < sum.size(); ++e) sum[e] += block[e];
  };
  const auto transposed = [&system](const std::vector<double>& block, std::size_t a, std::size_t b) {
    const std::size_t na = system.abf_count[a], nb = system.abf_count[b];
    std::vector<double> result(block.size());
    for (std::size_t p = 0; p < na; ++p)
      for (std::size_t q = 0; q < nb; ++q) result[q*na+p] = block[p*nb+q];
    return result;
  };
  // Places a folded V neighbour of atom a for the members of a class.
  const auto place = [&](std::size_t a, const Site& site, std::vector<double> block,
                         const std::vector<std::size_t>& members) {
    for (const std::size_t j : members) entry[a][j] = folded[a].size();
    folded[a].push_back(Folded{site,std::move(block)});
  };
  const auto negative = [](const std::array<int,3>& cell) { return std::array<int,3>{-cell[0],-cell[1],-cell[2]}; };

  for (std::size_t a = 0; a < atoms; ++a) {
    for (const auto& [key, members] : classes[a]) {
      const std::size_t b = static_cast<std::size_t>(key[0]);
      if (b < a) continue;                                             // folded from the side of b
      // The class of the reversed sites among the V neighbours of b.
      const std::array<int,3> start{key[1],key[2],key[3]};
      const Key mirror = class_of(static_cast<int>(a),negative(start));
      const auto found = classes[b].find(mirror);
      if (found == classes[b].end() || found->second.size() != members.size())
        throw std::logic_error("the V neighbours of two atoms are not each other's reversed sites");
      const std::vector<std::size_t>& reversed = found->second;
      if (b > a || key < mirror) {
        // Two different classes: a site for this class and the reversed
        // site, with the transpose, for the other.
        const std::array<int,3> cell = closest(a,members);
        std::vector<double> sum;
        for (const std::size_t j : members) add(sum,system.v[a][j]);
        std::vector<double> back = transposed(sum,a,b);
        place(a,Site{static_cast<int>(b),cell},std::move(sum),members);
        place(b,Site{static_cast<int>(a),negative(cell)},std::move(back),reversed);
      } else if (key == mirror) {
        if (key == Key{static_cast<int>(a),0,0,0}) {
          // The class of the atom itself: V(0) + sum over R > 0 of V(R) + V(R)^T.
          std::vector<double> sum = system.v[a][0];
          for (const std::size_t j : members)
            if (positive_cell(system.v_neighbours[a][j].cell)) {
              const std::vector<double> back = transposed(system.v[a][j],a,a);
              for (std::size_t e = 0; e < sum.size(); ++e) sum[e] += system.v[a][j][e]+back[e];
            }
          place(a,Site{static_cast<int>(a),{0,0,0}},std::move(sum),members);
        } else {
          // A class that is its own mirror image: (a, R) with the sum of the
          // blocks of the members R > 0, and (a, -R) with the transpose.
          std::vector<std::size_t> up, down;
          for (const std::size_t j : members) (positive_cell(system.v_neighbours[a][j].cell) ? up : down).push_back(j);
          if (up.size() != down.size() || up.empty())
            throw std::logic_error("a class of images of an atom is not made of reversed pairs");
          const std::array<int,3> cell = closest(a,up);
          std::vector<double> sum;
          for (const std::size_t j : up) add(sum,system.v[a][j]);
          std::vector<double> back = transposed(sum,a,a);
          place(a,Site{static_cast<int>(a),cell},std::move(sum),up);
          place(a,Site{static_cast<int>(a),negative(cell)},std::move(back),down);
        }
      }
      // key > mirror for b = a: placed with the mirror class.
    }
  }

  // The lists in the order of neighbour_lists(): the atom itself first, then
  // by atom and cell.
  VFolding folding;
  folding.period = period;
  folding.index.resize(atoms);
  std::vector<std::vector<Site>> lists(atoms);
  std::vector<std::vector<std::vector<double>>> blocks(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    std::vector<Folded>& list = folded[a];
    std::vector<std::size_t> order(list.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    const Site self{static_cast<int>(a),{0,0,0}};
    std::sort(order.begin(),order.end(),[&list,&self](std::size_t x, std::size_t y) {
      const Site& s = list[x].site;
      const Site& t = list[y].site;
      if (s == self || t == self) return s == self && !(t == self);
      return s.atom != t.atom ? s.atom < t.atom : s.cell < t.cell;
    });
    if (order.empty() || !(list[order.front()].site == self))
      throw std::logic_error("the folded V neighbours of an atom do not begin with the atom");
    std::vector<std::size_t> position(list.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
      position[order[i]] = i;
      if (i > 0 && list[order[i]].site == list[order[i-1]].site)
        throw std::logic_error("two classes of V neighbours fold onto one site");
      lists[a].push_back(list[order[i]].site);
      blocks[a].push_back(std::move(list[order[i]].block));
    }
    folding.index[a].resize(entry[a].size());
    for (std::size_t j = 0; j < entry[a].size(); ++j) {
      if (entry[a][j] >= list.size()) throw std::logic_error("a V neighbour was left out of the folding");
      folding.index[a][j] = position[entry[a][j]];
    }
  }
  folding.prepared = std::move(system.v_neighbours);
  system.v_neighbours = std::move(lists);
  system.v = std::move(blocks);
  system.folding = std::move(folding);
}

std::size_t table_elements(const System& system, std::size_t atom)
{
  if (atom >= system.t.size() || atom >= system.v.size())
    throw std::out_of_range("the atom of the blocks of a system does not exist");
  std::size_t elements = system.t[atom].size();
  for (const std::vector<double>& block : system.v[atom]) elements += block.size();
  return elements;
}

void get_tables(const System& system, std::size_t atom, std::span<double> values)
{
  if (values.size() != table_elements(system,atom))
    throw std::invalid_argument("the array does not have the size of the blocks of the atom");
  auto target = std::copy(system.t[atom].begin(),system.t[atom].end(),values.begin());
  for (const std::vector<double>& block : system.v[atom]) target = std::copy(block.begin(),block.end(),target);
}

void set_tables(System& system, std::size_t atom, std::span<const double> values)
{
  if (values.size() != table_elements(system,atom))
    throw std::invalid_argument("the array does not have the size of the blocks of the atom");
  auto source = values.begin();
  std::copy_n(source,system.t[atom].size(),system.t[atom].begin());
  source += static_cast<std::ptrdiff_t>(system.t[atom].size());
  for (std::vector<double>& block : system.v[atom]) {
    std::copy_n(source,block.size(),block.begin());
    source += static_cast<std::ptrdiff_t>(block.size());
  }
}

} // namespace openmx::exx
