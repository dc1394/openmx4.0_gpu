// Tests of system.hpp:
//   exx-system-neighbours  the neighbour lists against a search over a block
//                          of cells, their order, and the sites of the main
//                          program's overlap matrix
//   exx-system-blocks      the T and V blocks of a small periodic structure:
//                          the two descriptions of every pair, a translated
//                          copy, and the charge of the expanded products
#include <openmx/exx/abf.hpp>
#include <openmx/exx/system.hpp>

#include "overlap_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

namespace {

using Vec = std::array<double,3>;

Structure structure_of(const OverlapFixture& fixture)
{
  Structure s;
  s.lattice = fixture.lattice;
  s.positions = fixture.positions;
  s.species = fixture.atom_species;
  return s;
}

double length(const Vec& v) { return std::sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }

// Every site within the cutoffs among the cells -range..range, in the order
// of neighbour_lists().
std::vector<std::vector<Site>> search(const Structure& s, const std::vector<double>& cutoffs, std::size_t count, int range)
{
  std::vector<std::vector<Site>> lists(s.positions.size());
  for (std::size_t a = 0; a < s.positions.size(); ++a) {
    lists[a].push_back({static_cast<int>(a),{0,0,0}});
    for (std::size_t b = 0; b < s.positions.size(); ++b)
      for (int n1 = -range; n1 <= range; ++n1)
        for (int n2 = -range; n2 <= range; ++n2)
          for (int n3 = -range; n3 <= range; ++n3) {
            if (a == b && n1 == 0 && n2 == 0 && n3 == 0) continue;
            Vec d{};
            for (std::size_t x = 0; x < 3; ++x)
              d[x] = s.positions[b][x]-s.positions[a][x]+n1*s.lattice[0][x]+n2*s.lattice[1][x]+n3*s.lattice[2][x];
            if (length(d) <= cutoffs[static_cast<std::size_t>(s.species[a])*count+static_cast<std::size_t>(s.species[b])])
              lists[a].push_back({static_cast<int>(b),{n1,n2,n3}});
          }
  }
  return lists;
}

void compare_lists(const std::vector<std::vector<Site>>& lists, const std::vector<std::vector<Site>>& reference,
                   const std::string& what)
{
  require(lists.size() == reference.size(),what+": wrong number of lists");
  for (std::size_t a = 0; a < lists.size(); ++a) {
    require(lists[a].size() == reference[a].size(),what+": wrong number of neighbours");
    for (std::size_t i = 0; i < lists[a].size(); ++i)
      require(lists[a][i] == reference[a][i],what+": a neighbour differs or is out of order");
  }
}

template <class Call>
void expect_invalid(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const std::invalid_argument&) { thrown = true; } catch (const std::out_of_range&) { thrown = true; }
  require(thrown,"no exception for "+what);
}

std::vector<SpeciesTable> tables_of(const OverlapFixture& fixture, int intervals, const ReciprocalGrid& reciprocal,
                                    double threshold)
{
  std::vector<SpeciesTable> tables;
  for (const FixtureSpecies& species : fixture.species) {
    const RadialGrid grid{0.0,species.cutoff/intervals,intervals};
    std::vector<MeshFunction> shells;
    for (const FixtureShell& shell : species.shells) shells.push_back({shell.l,shell.values});
    const std::vector<RadialFunction> orbitals = sample_orbitals(grid,species.radii,shells);
    tables.emplace_back(grid,orbitals,make_auxiliary_basis(grid,orbitals,threshold).functions,reciprocal);
  }
  return tables;
}

} // namespace

void system_neighbours(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  const Structure structure = structure_of(fixture);
  const std::size_t count = fixture.species.size();
  std::vector<double> sum(count*count), other(count*count);
  for (std::size_t s = 0; s < count; ++s)
    for (std::size_t t = 0; t < count; ++t) {
      sum[s*count+t] = fixture.species[s].cutoff+fixture.species[t].cutoff;
      other[s*count+t] = std::max(fixture.species[s].cutoff,fixture.species[t].cutoff)+8.0;
    }

  // The triclinic cell of the fixture, smaller than the cutoffs.
  const std::vector<std::vector<Site>> lists = neighbour_lists(structure,sum,count);
  compare_lists(lists,search(structure,sum,count,6),"C neighbours of the fixture");
  compare_lists(neighbour_lists(structure,other,count),search(structure,other,count,7),"V neighbours of the fixture");
  std::size_t total = 0, repeated = 0;
  for (std::size_t a = 0; a < lists.size(); ++a) {
    require(lists[a].front() == Site{static_cast<int>(a),{0,0,0}},"a list does not start with its own atom");
    total += lists[a].size();
    for (std::size_t i = 1; i < lists[a].size(); ++i) {
      const Site& site = lists[a][i];
      if (site.atom == static_cast<int>(a)) ++repeated;
      require(find_site(lists[a],site) == i,"find_site does not find a member of the list");
      // The pair seen from the other atom.
      const Site back{static_cast<int>(a),{-site.cell[0],-site.cell[1],-site.cell[2]}};
      require(find_site(lists[static_cast<std::size_t>(site.atom)],back) < lists[static_cast<std::size_t>(site.atom)].size(),
              "a neighbour has no counterpart");
      const Vec d = displacement(structure,static_cast<int>(a),site);
      const Vec e = displacement(structure,site.atom,back);
      require(d[0] == -e[0] && d[1] == -e[1] && d[2] == -e[2],"the two displacements of a pair are not opposite");
      require(length(d) <= sum[static_cast<std::size_t>(structure.species[a])*count+
                               static_cast<std::size_t>(structure.species[static_cast<std::size_t>(site.atom)])],
              "a neighbour is beyond the cutoff");
    }
    require(find_site(lists[a],Site{0,{40,0,0}}) == lists[a].size(),"find_site finds a site that is absent");
  }
  std::printf("C neighbours of the fixture: %zu sites for %zu atoms, %zu of them images of the atom itself\n",
              total,lists.size(),repeated);
  require(repeated >= 20,"the cell of the fixture is expected to be smaller than the cutoffs");

  // The blocks that the main program wrote belong to its first neighbours
  // (FNAN), which obey the same condition.
  for (const FixturePair& pair : fixture.pairs)
    require(find_site(lists[static_cast<std::size_t>(pair.a)],Site{pair.b,pair.cell}) < lists[static_cast<std::size_t>(pair.a)].size(),
            "a pair of the overlap matrix of the main program is missing from the C neighbours");

  // A simple cubic lattice: a neighbour exactly at the cutoff is included.
  {
    Structure cubic;
    cubic.lattice = {Vec{4.0,0.0,0.0},Vec{0.0,4.0,0.0},Vec{0.0,0.0,4.0}};
    cubic.positions = {Vec{0.3,-0.2,0.9}};
    cubic.species = {0};
    require(neighbour_lists(cubic,std::vector<double>{4.0},1)[0].size() == 7,"sites at the cutoff are missing");
    require(neighbour_lists(cubic,std::vector<double>{3.999999},1)[0].size() == 1,"sites beyond the cutoff are present");
    require(neighbour_lists(cubic,std::vector<double>{5.7},1)[0].size() == 19,"wrong second shell of the cubic lattice");
    require(neighbour_lists(cubic,std::vector<double>{0.0},1)[0].size() == 1,"a cutoff of zero must leave the atom alone");
  }

  // A strongly sheared cell with two species, left-handed.
  {
    Structure sheared;
    sheared.lattice = {Vec{3.0,0.0,0.0},Vec{2.9,0.5,0.0},Vec{-1.0,1.0,-4.0}};
    sheared.positions = {Vec{0.1,0.2,0.3},Vec{1.7,-0.4,2.2},Vec{-3.0,5.0,1.0}};
    sheared.species = {0,1,0};
    const std::vector<double> cutoffs{5.0,6.5,6.5,3.2};
    compare_lists(neighbour_lists(sheared,cutoffs,2),search(sheared,cutoffs,2,45),"sheared cell");
  }

  // Invalid arguments.
  {
    Structure flat = structure;
    flat.lattice[2] = {flat.lattice[0][0]+flat.lattice[1][0],flat.lattice[0][1]+flat.lattice[1][1],
                       flat.lattice[0][2]+flat.lattice[1][2]};
    expect_invalid("lattice vectors in one plane",[&] { neighbour_lists(flat,sum,count); });
    Structure wrong = structure;
    wrong.species[1] = 3;
    expect_invalid("a species out of range",[&] { neighbour_lists(wrong,sum,count); });
    Structure missing = structure;
    missing.species.pop_back();
    expect_invalid("an atom without a species",[&] { neighbour_lists(missing,sum,count); });
    expect_invalid("a table of cutoffs of the wrong size",[&] { neighbour_lists(structure,std::vector<double>(4,1.0),count); });
    std::vector<double> asymmetric = sum;
    asymmetric[1] += 0.5;
    expect_invalid("cutoffs that are not symmetric",[&] { neighbour_lists(structure,asymmetric,count); });
    std::vector<double> negative = sum;
    negative[0] = -1.0;
    expect_invalid("a negative cutoff",[&] { neighbour_lists(structure,negative,count); });
    expect_invalid("a site with an atom out of range",[&] { displacement(structure,0,Site{7,{0,0,0}}); });
  }
}

void system_blocks(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  const Structure structure = structure_of(fixture);
  const std::size_t atoms = structure.positions.size();

  // Coarser grids than the main program's keep the test short; the checks of
  // the bookkeeping do not depend on the accuracy of the quadrature.
  const int intervals = 300;
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(24.0,240);
  const std::vector<SpeciesTable> tables = tables_of(fixture,intervals,reciprocal,1e-3);
  SystemOptions options;
  options.interaction = short_range_kernel(0.11);
  options.v_cutoff = 3.0;
  const System system = prepare_system(structure,tables,reciprocal,options);

  int maximum_orbital = 0, maximum_abf = 0;
  for (const SpeciesTable& table : tables) {
    maximum_orbital = std::max(maximum_orbital,table.maximum_orbital_angular());
    maximum_abf = std::max(maximum_abf,table.maximum_abf_angular());
  }
  const LriSolver solver(reciprocal,maximum_orbital,maximum_abf,options.lri);
  const TwoCenterIntegrator interaction(reciprocal,options.interaction,maximum_abf,maximum_abf);
  const auto table_of = [&](std::size_t atom) -> const SpeciesTable& {
    return tables[static_cast<std::size_t>(structure.species[atom])];
  };

  // Lists and sizes.
  const std::size_t count = tables.size();
  std::vector<double> c_cutoff(count*count), v_cutoff(count*count);
  for (std::size_t s = 0; s < count; ++s)
    for (std::size_t t = 0; t < count; ++t) {
      c_cutoff[s*count+t] = tables[s].radius()+tables[t].radius();
      v_cutoff[s*count+t] = std::max(tables[s].radius(),tables[t].radius())+options.v_cutoff;
    }
  compare_lists(system.c_neighbours,neighbour_lists(structure,c_cutoff,count),"C neighbours of the system");
  compare_lists(system.v_neighbours,neighbour_lists(structure,v_cutoff,count),"V neighbours of the system");
  require(system.t.size() == atoms && system.v.size() == atoms && system.orbital_count.size() == atoms &&
          system.abf_count.size() == atoms,"the system has the wrong number of atoms");
  long pairs = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    require(system.orbital_count[a] == table_of(a).orbital_count() && system.abf_count[a] == table_of(a).abf_count(),
            "wrong counts of an atom");
    require(system.t[a].size() == system.c_neighbours[a].size() && system.v[a].size() == system.v_neighbours[a].size(),
            "wrong number of blocks of an atom");
    for (std::size_t i = 0; i < system.t[a].size(); ++i) {
      const std::size_t c = static_cast<std::size_t>(system.c_neighbours[a][i].atom);
      require(system.t[a][i].size() == system.abf_count[a]*system.orbital_count[a]*system.orbital_count[c],
              "a T block has the wrong size");
    }
    for (std::size_t i = 0; i < system.v[a].size(); ++i) {
      const std::size_t b = static_cast<std::size_t>(system.v_neighbours[a][i].atom);
      require(system.v[a][i].size() == system.abf_count[a]*system.abf_count[b],"a V block has the wrong size");
    }
    pairs += static_cast<long>(system.c_neighbours[a].size())-1;
  }
  require(pairs%2 == 0,"the C neighbours do not come in pairs");
  require(system.report.systems == static_cast<long>(count)+pairs/2 && system.report.eigen_decompositions == 0 &&
          system.report.dropped == 0,"unexpected report of the solutions of equation (7)");
  std::printf("T blocks: %ld pairs of sites, smallest pivot %.2e\n",pairs/2,system.report.smallest_pivot);

  // The block of the atom itself is half of the solution for the atom alone.
  for (std::size_t a = 0; a < atoms; ++a)
    require(system.t[a][0] == solver.onsite(table_of(a)),"the T block of an atom with itself differs from the solver");

  // Each pair of sites is solved once.  A sample of the T blocks: the block is
  // bit for bit a part of the solution of its pair as seen from one of the two
  // atoms, and the block of the other atom comes from the same solution.  The
  // two solutions agree to rounding only.
  Worst rounding("T blocks against the solution seen from the other atom",1e-10);
  long own = 0, taken = 0;
  std::size_t counter = 0;
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t i = 1; i < system.c_neighbours[a].size(); ++i, ++counter) {
      if (counter%11 != 0) continue;
      const Site& site = system.c_neighbours[a][i];
      const std::size_t c = static_cast<std::size_t>(site.atom);
      const Site back{static_cast<int>(a),{-site.cell[0],-site.cell[1],-site.cell[2]}};
      const std::size_t j = find_site(system.c_neighbours[c],back);
      require(j < system.c_neighbours[c].size(),"a C neighbour has no counterpart");
      const PairCoefficients forward = solver.pair(table_of(a),table_of(c),displacement(structure,static_cast<int>(a),site));
      const PairCoefficients backward = solver.pair(table_of(c),table_of(a),displacement(structure,site.atom,back));
      if (system.t[a][i] == forward.first) {
        require(system.t[c][j] == forward.second,"the two T blocks of a pair come from different solutions");
        ++own;
        for (std::size_t k = 0; k < backward.second.size(); ++k) rounding.absolute(system.t[a][i][k],backward.second[k]);
      } else {
        require(system.t[a][i] == backward.second,"a T block is not a solution of its pair");
        require(system.t[c][j] == backward.first,"the two T blocks of a pair come from different solutions");
        ++taken;
        for (std::size_t k = 0; k < forward.first.size(); ++k) rounding.absolute(system.t[a][i][k],forward.first[k]);
      }
    }
  rounding.finish();
  std::printf("T blocks checked: %ld from the solution of their own atom, %ld from the other atom's\n",own,taken);
  require(own >= 5 && taken >= 5,"too few T blocks of one of the two kinds were checked");

  // V blocks: the block of the atom itself is symmetric, the two descriptions
  // of a pair are transposes, and a sample agrees with equation (12).
  Worst transposed("V blocks of the two descriptions of a pair",0.0);
  Worst sample("V blocks against equation (12)",1e-12);
  counter = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t na = system.abf_count[a];
    for (std::size_t p = 0; p < na; ++p)
      for (std::size_t q = 0; q < p; ++q) transposed.absolute(system.v[a][0][p*na+q],system.v[a][0][q*na+p]);
    for (std::size_t i = 1; i < system.v_neighbours[a].size(); ++i, ++counter) {
      const Site& site = system.v_neighbours[a][i];
      const std::size_t b = static_cast<std::size_t>(site.atom), nb = system.abf_count[b];
      const std::size_t j = find_site(system.v_neighbours[b],Site{static_cast<int>(a),{-site.cell[0],-site.cell[1],-site.cell[2]}});
      require(j < system.v_neighbours[b].size(),"a V neighbour has no counterpart");
      for (std::size_t p = 0; p < na; ++p)
        for (std::size_t q = 0; q < nb; ++q) transposed.absolute(system.v[a][i][p*nb+q],system.v[b][j][q*na+p]);
      if (counter%7 != 0) continue;
      const std::vector<double> block = two_center_matrix(interaction.pair(displacement(structure,static_cast<int>(a),site)),
                                                          table_of(a).abf_transforms(),table_of(b).abf_transforms());
      for (std::size_t k = 0; k < block.size(); ++k) sample.absolute(system.v[a][i][k],block[k]);
    }
  }
  transposed.finish();
  sample.finish();

  // The same crystal described differently: all atoms shifted, and one atom
  // moved to another cell.  The blocks are the same; the cells of the sites
  // of that atom change.
  {
    Structure moved = structure;
    const std::array<int,3> jump{1,0,-1};
    const std::size_t jumping = 1;
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t x = 0; x < 3; ++x) {
        moved.positions[a][x] += std::array<double,3>{0.37,-1.2,2.2}[x];
        if (a == jumping)
          for (std::size_t k = 0; k < 3; ++k) moved.positions[a][x] += jump[k]*moved.lattice[k][x];
      }
    const System other = prepare_system(moved,tables,reciprocal,options);
    Worst t_moved("T blocks of a translated description",1e-10);
    Worst v_moved("V blocks of a translated description",1e-10);
    const auto translate = [&](std::size_t a, const Site& site) {
      // tau'_b + R' - tau'_a = tau_b + R - tau_a  with  tau'_x = tau_x + shift + J_x.
      Site result = site;
      for (std::size_t k = 0; k < 3; ++k) {
        if (static_cast<std::size_t>(site.atom) == jumping) result.cell[k] -= jump[k];
        if (a == jumping) result.cell[k] += jump[k];
      }
      return result;
    };
    for (std::size_t a = 0; a < atoms; ++a) {
      require(other.c_neighbours[a].size() == system.c_neighbours[a].size() &&
              other.v_neighbours[a].size() == system.v_neighbours[a].size(),"a translation changed the number of neighbours");
      for (std::size_t i = 0; i < system.c_neighbours[a].size(); ++i) {
        const std::size_t k = find_site(other.c_neighbours[a],translate(a,system.c_neighbours[a][i]));
        require(k < other.c_neighbours[a].size(),"a C neighbour is missing after a translation");
        for (std::size_t e = 0; e < system.t[a][i].size(); ++e) t_moved.absolute(other.t[a][k][e],system.t[a][i][e]);
      }
      for (std::size_t i = 0; i < system.v_neighbours[a].size(); ++i) {
        const std::size_t k = find_site(other.v_neighbours[a],translate(a,system.v_neighbours[a][i]));
        require(k < other.v_neighbours[a].size(),"a V neighbour is missing after a translation");
        for (std::size_t e = 0; e < system.v[a][i].size(); ++e) v_moved.absolute(other.v[a][k][e],system.v[a][i][e]);
      }
    }
    t_moved.finish();
    v_moved.finish();
  }

  // The charge of an expanded product, sum_P C^P int P, against the overlap
  // matrix of the main program.  Only s-type ABFs carry charge.  This checks
  // the order of the indices of the T blocks with numbers that this module
  // did not compute: a wrong order gives deviations of the size of the
  // overlaps, 0.1 to 1.  The fit in the Coulomb metric does not conserve
  // charge, so the limit is far from rounding; with these ABFs the largest
  // deviation is 4.4e-3.
  {
    Worst charge("charge of the expanded products against the overlap matrix",1e-2);
    std::vector<std::vector<double>> charges(count);
    for (std::size_t s = 0; s < count; ++s) {
      charges[s].assign(tables[s].abf_count(),0.0);
      for (std::size_t n = 0; n < tables[s].abfs().size(); ++n)
        if (tables[s].abfs()[n].angular == 0)
          charges[s][tables[s].abf_offsets()[n]] =
              std::sqrt(4.0*std::numbers::pi)*radial_moment(tables[s].grid(),tables[s].abfs()[n].values,0);
    }
    long offsite = 0;
    for (const FixturePair& pair : fixture.pairs) {
      const std::size_t a = static_cast<std::size_t>(pair.a), c = static_cast<std::size_t>(pair.b);
      const std::size_t sa = static_cast<std::size_t>(structure.species[a]), sc = static_cast<std::size_t>(structure.species[c]);
      const std::size_t ua = system.orbital_count[a], uc = system.orbital_count[c];
      const std::size_t i = find_site(system.c_neighbours[a],Site{pair.b,pair.cell});
      const std::size_t j = find_site(system.c_neighbours[c],Site{pair.a,{-pair.cell[0],-pair.cell[1],-pair.cell[2]}});
      require(i < system.c_neighbours[a].size() && j < system.c_neighbours[c].size(),"a pair of the fixture is not a C neighbour");
      for (std::size_t mu = 0; mu < ua; ++mu)
        for (std::size_t lambda = 0; lambda < uc; ++lambda) {
          double sum = 0.0;
          // Equation (8): the ABFs of the first site, then those of the second
          // one.  For the atom itself the two terms are the same block.
          for (std::size_t p = 0; p < charges[sa].size(); ++p)
            sum += charges[sa][p]*system.t[a][i][(p*ua+mu)*uc+lambda];
          for (std::size_t q = 0; q < charges[sc].size(); ++q)
            sum += charges[sc][q]*system.t[c][j][(q*uc+lambda)*ua+mu];
          charge.absolute(sum,pair.values[mu*static_cast<std::size_t>(pair.columns)+lambda]);
        }
      if (i != 0) ++offsite;
    }
    charge.finish();
    require(offsite >= 30,"too few pairs of two atoms in the charge check");
  }

  // Invalid arguments.
  {
    SystemOptions bad = options;
    bad.v_cutoff = -1.0;
    expect_invalid("a negative cutoff of the V blocks",[&] { prepare_system(structure,tables,reciprocal,bad); });
    expect_invalid("fewer species tables than species",
                   [&] { prepare_system(structure,std::span<const SpeciesTable>(tables).first(2),reciprocal,options); });
    const ReciprocalGrid other = ReciprocalGrid::uniform(24.0,120);
    expect_invalid("tables of another reciprocal grid",[&] { prepare_system(structure,tables,other,options); });
  }
}

} // namespace exx_test
