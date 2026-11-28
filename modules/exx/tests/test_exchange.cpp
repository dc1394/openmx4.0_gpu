// Tests of exchange.hpp:
//   exx-exchange-fixture   the blocks of K and E_x of the site form against
//                          the fixture of the reference implementation
//                          (tools/exx_reference.py), and equation (15)
//   exx-exchange-layout    the site form with all blocks in one array: the
//                          list of the blocks, K against the site form with a
//                          map of blocks, and the symmetry K_xy(R) = K_yx(-R)^T
//   exx-exchange-gamma     the Gamma form against the sum over R of the blocks
//                          of the site form and against the definition with
//                          dense matrices
//   exx-exchange-gradient  equations (16) and (17) against central differences
//   exx-exchange-centre-gradient  the same derivatives from the arrays of one
//                          centre atom, in the Gamma form and in the site form
//   exx-exchange-backend   the same results with a table of matrix products,
//                          and with a table whose loop runs the ABFs of a
//                          centre atom in pieces: the number of pieces, their
//                          order and their threads, sums that are shorter
//                          than the number of pieces, a product that fails
//                          inside a piece, and the derivatives, which are
//                          not divided
//   exx-exchange-system    T and V blocks made from PAOs: isolated atoms
//                          against a formula without T blocks, a periodic
//                          structure against the definition, the pieces of
//                          the ABFs on threads with and without a product of
//                          the table, and the C interface against the C++
//                          interface, also with pieces on threads and with a
//                          product that fails inside a piece; through the C
//                          interface the ABFs in ranges and a preparation
//                          that three states divide (section 6.7)
//   exx-exchange-forces    forces.hpp: the derivative of E_x with respect to
//                          the positions of the atoms against central
//                          differences of E_x of structures with moved atoms,
//                          by the route through the centre atoms and by the
//                          route through the pairs of sites, which processes
//                          divide (section 6.7)
//   exx-exchange-device    the device of exx.h (OpenMXExxDevice) in both
//                          forms, with a device in host memory that holds
//                          the library to the contract (test_device.hpp):
//                          the device against the dense backend, the centre
//                          atoms in two calls, a device that declines or
//                          lacks a function, and every call of the device
//                          failing in turn; the fixture and the system of
//                          three atoms of exx-exchange-backend.  The ABFs of
//                          the centre atoms in ranges, as several processes
//                          share them (section 6.7 of the guide), on the
//                          host and on the device: whole atoms, every cut of
//                          the ABFs in two, ranges of growing length, and
//                          the division of the ABFs among processes against
//                          all divisions
//   exx-exchange-device-system  the same for the periodic structure of
//                          exx-exchange-system, with a selection of the
//                          failures; the half build on the device is part of
//                          both
//   exx-exchange-half      the half build (section 6.4 of the guide): X +
//                          X^T against the whole build, with the ABFs in
//                          ranges and with five backends, in a system of
//                          three atoms with the symmetry of prepare_system()
//                          and in the periodic structure of
//                          exx-exchange-system
//   exx-exchange-folding   the V neighbours folded by the period of a k mesh
//                          (section 6.4 of the guide), in the periodic
//                          structure of exx-exchange-system: the folded lists
//                          and blocks, what the folding rejects, and for
//                          meshes with and without shifts K(k) at the points
//                          of the mesh, E_x and the derivative with respect
//                          to the positions by five routes against the
//                          preparation, the Gamma form, the half build of a
//                          folded system, and the period 1 against the Gamma
//                          form; the same with the columns folded as well,
//                          what that folding rejects, its bits for classes
//                          of one site, a wrong sign, and the device
#include <openmx/exx/abf.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/exx.h>
#include <openmx/exx/forces.hpp>
#include <openmx/exx/system.hpp>

#include "kbuild_fixture.hpp"
#include "overlap_fixture.hpp"
#include "test_device.hpp"
#include "test_dgemm.hpp"
#include "test_parallel.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

namespace {

template <class Call>
void expect_invalid(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const std::invalid_argument&) { thrown = true; } catch (const std::out_of_range&) { thrown = true; }
  require(thrown,"no exception for "+what);
}

template <class Call>
void expect_runtime_error(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const std::runtime_error&) { thrown = true; }
  require(thrown,"no exception for "+what);
}

// The first orbital of every atom, and the number of orbitals N.
std::vector<std::size_t> offsets(const System& system)
{
  std::vector<std::size_t> offset(1,0);
  for (const std::size_t u : system.orbital_count) offset.push_back(offset.back()+u);
  return offset;
}

// An n x n matrix of numbers without a pattern, symmetric on request.
std::vector<double> dense_matrix(std::size_t n, bool symmetric, Numbers& random)
{
  std::vector<double> d(n*n);
  for (double& value : d) value = random.next();
  if (symmetric)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < i; ++j) d[i*n+j] = d[j*n+i] = 0.5*(d[i*n+j]+d[j*n+i]);
  return d;
}

double largest(std::span<const double> values)
{
  double result = 0.0;
  for (const double value : values) result = std::max(result,std::fabs(value));
  return result;
}

// The density matrix of a cell that is sampled at the Gamma point only: every
// block D_xy(R) is the block of the atoms x and y of one dense matrix.
DensityBlocks gamma_density(const System& system, const std::vector<double>& dense)
{
  return [&system,&dense,offset = offsets(system)](const BlockKey& key, std::span<double> block) {
    const std::size_t x = static_cast<std::size_t>(key.row), y = static_cast<std::size_t>(key.column);
    const std::size_t rows = system.orbital_count[x], columns = system.orbital_count[y], n = offset.back();
    require(block.size() == rows*columns,"a block of D has the wrong size");
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < columns; ++c) block[r*columns+c] = dense[(offset[x]+r)*n+offset[y]+c];
  };
}

// sum_R M_xy(R) as a dense matrix.
std::vector<double> folded(const System& system, const BlockMatrix& matrix)
{
  const std::vector<std::size_t> offset = offsets(system);
  const std::size_t n = offset.back();
  std::vector<double> dense(n*n,0.0);
  for (const auto& [key, values] : matrix.blocks()) {
    const std::size_t x = static_cast<std::size_t>(key.row), y = static_cast<std::size_t>(key.column);
    const std::size_t rows = system.orbital_count[x], columns = system.orbital_count[y];
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < columns; ++c) dense[(offset[x]+r)*n+offset[y]+c] += values[r*columns+c];
  }
  return dense;
}

// The density matrix of the fixture.  A block that the fixture lacks is an
// error; "read" collects the blocks that were asked for.
DensityBlocks fixture_density(const KbuildFixture& fixture, std::set<BlockKey>* read = nullptr)
{
  return [&fixture,read](const BlockKey& key, std::span<double> block) {
    const auto found = fixture.density.find(key);
    require(found != fixture.density.end(),"the build reads a block of D that the fixture lacks");
    require(found->second.size() == block.size(),"a block of D has the wrong size");
    std::copy(found->second.begin(),found->second.end(),block.begin());
    if (read != nullptr) read->insert(key);
  };
}

// Two sets of blocks must have the same keys.
void compare_blocks(const BlockMatrix& a, const BlockMatrix& b, Worst& worst)
{
  require(a.blocks().size() == b.blocks().size(),"two matrices differ in the number of blocks");
  for (const auto& [key, values] : a.blocks()) {
    const std::span<const double> other = b.find(key);
    require(other.size() == values.size(),"a block is missing from one of two matrices");
    for (std::size_t e = 0; e < values.size(); ++e) worst.absolute(values[e],other[e]);
  }
}

// The same numbers bit for bit.
bool same_values(std::span<const double> a, std::span<const double> b)
{
  return a.size() == b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(double)) == 0);
}

// The same for arrays of arrays, as the derivatives with respect to the
// blocks come.
template <class Inner>
bool same_values(const std::vector<std::vector<Inner>>& a, const std::vector<std::vector<Inner>>& b)
{
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!same_values(a[i],b[i])) return false;
  return true;
}

// The same keys, and the same numbers bit for bit.  find() gives nothing for
// a block that does not exist, so a block without elements is not accepted.
bool same_blocks(const BlockMatrix& a, const BlockMatrix& b)
{
  if (a.blocks().size() != b.blocks().size()) return false;
  for (const auto& [key, values] : a.blocks())
    if (values.empty() || !same_values(values,b.find(key))) return false;
  return true;
}

// The blocks of a matrix in one array of a layout.
std::vector<double> layout_values(const BlockLayout& layout, const DensityBlocks& blocks)
{
  std::vector<double> values(layout.elements(),0.0);
  for (std::size_t i = 0; i < layout.keys.size(); ++i)
    blocks(layout.keys[i],std::span<double>(values).subspan(layout.offset[i],layout.offset[i+1]-layout.offset[i]));
  return values;
}

// K of all atoms in one array of a layout.
std::vector<double> layout_exchange(const System& system, const BlockLayout& layout, const std::vector<double>& density,
                                    double alpha, const DenseBackend& backend = DenseBackend())
{
  std::vector<double> k(layout.elements(),0.0);
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a)
    add_exchange_blocks(system,layout,a,-alpha,density,k,backend);
  return k;
}

// An array of a layout holds the blocks of a matrix, bit for bit, and the
// matrix has no other block.
bool same_as_blocks(const BlockLayout& layout, std::span<const double> array, const BlockMatrix& matrix)
{
  if (array.size() != layout.elements() || layout.keys.size() != matrix.blocks().size()) return false;
  for (std::size_t i = 0; i < layout.keys.size(); ++i) {
    const std::span<const double> block = array.subspan(layout.offset[i],layout.offset[i+1]-layout.offset[i]);
    if (block.empty() || !same_values(block,matrix.find(layout.keys[i]))) return false;
  }
  return true;
}

// The three forms of the build hand different loops to the table: the site
// form with the blocks in a map (BlockMatrix), the site form in one array of
// a layout, which the C interface uses, and the Gamma form.
enum class Form { map, layout, gamma };

// The loops that one build of K hands to the loop of a table, as the loops
// of the tests record them: the number of loops for every count of calls.
// centre_contribution() of exchange.cpp divides the ABFs of a centre atom
// into min(threads, n_P) pieces.  With more than one piece it hands over
// four kinds of loops, each with one call for every piece: one that turns G,
// two for every group of V neighbours (the four terms, and the sums of the
// second term) and one that turns the bracket.  That is 2 + 2 groups loops
// for the atom.  The site form in one array and the Gamma form also gather
// the rows of D[loc(A), .] and add the last result to K in a loop each, with
// as many calls: 4 + 2 groups loops.  The site form with a map does these
// two steps on the calling thread.  With one piece all of this runs on the
// calling thread, and the loop of the table is not called.  The groups are
// the V neighbours in the site form and the atoms among the V neighbours in
// the Gamma form.
Counts loops_of(const System& system, int threads, Form form)
{
  Counts counts;
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a) {
    const std::size_t pieces = std::min(static_cast<std::size_t>(threads),system.abf_count[a]);
    if (pieces < 2) continue;
    std::set<int> atoms;
    for (const Site& site : system.v_neighbours[a]) atoms.insert(site.atom);
    const std::size_t groups = form == Form::gamma ? atoms.size() : system.v_neighbours[a].size();
    counts[static_cast<int>(pieces)] += (form == Form::map ? 2 : 4)+2*static_cast<long>(groups);
  }
  return counts;
}

// The pairs of a site of loc(A) and a site of S(A) of the centre atoms whose
// block of K another pair of the same centre also receives, as the site form
// adds the last result.  Such pairs need two images of one atom in loc(A).
std::size_t shared_pairs(const System& system)
{
  std::size_t shared = 0;
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a) {
    std::set<std::array<int,4>> seen;
    std::vector<Site> sites;
    for (const Site& b : system.v_neighbours[a])
      for (const Site& c : system.c_neighbours[static_cast<std::size_t>(b.atom)]) {
        const Site site{c.atom,{b.cell[0]+c.cell[0],b.cell[1]+c.cell[1],b.cell[2]+c.cell[2]}};
        if (seen.insert({site.atom,site.cell[0],site.cell[1],site.cell[2]}).second) sites.push_back(site);
      }
    std::map<BlockKey,std::size_t> pairs;
    for (const Site& row : system.c_neighbours[a])
      for (const Site& column : sites)
        ++pairs[BlockKey{row.atom,column.atom,{column.cell[0]-row.cell[0],column.cell[1]-row.cell[1],
                                                column.cell[2]-row.cell[2]}}];
    for (const auto& [key, count] : pairs)
      if (count > 1) shared += count;
  }
  return shared;
}

// l_b, the number of columns of the T blocks of atom b in a build: the
// orbitals of its C neighbours in the site form, and those of the atoms among
// them in the Gamma form, where the images of an atom share their columns.
std::size_t column_count(const System& system, std::size_t b, bool gamma)
{
  if (!gamma) return system.c_offset[b].back();
  std::set<int> atoms;
  std::size_t columns = 0;
  for (const Site& site : system.c_neighbours[b])
    if (atoms.insert(site.atom).second) columns += system.orbital_count[static_cast<std::size_t>(site.atom)];
  return columns;
}

// In the small fixtures one W product covers each run of V images of the
// same atom. Both the row and the workspace limit are above the run size.
long v_product_panels(std::span<const Site> neighbours)
{
  long panels = 1;
  for (std::size_t i = 1; i < neighbours.size(); ++i)
    if (neighbours[i-1].atom != neighbours[i].atom) ++panels;
  return panels;
}

// A system of three atoms with few orbitals and more ABFs, for the sums of
// the pieces in exchange_backend().  The T and V blocks hold numbers without
// a pattern.  The neighbour lists belong to no structure, and the build
// needs none; the second atom has two images of the first atom in both
// lists.
System narrow_system(Numbers& random)
{
  System system;
  system.orbital_count = {1,1,2};
  system.abf_count = {5,4,3};
  const auto site = [](int atom, int cell = 0) { return Site{atom,{cell,0,0}}; };
  system.c_neighbours = {{site(0)},{site(1),site(0),site(0,1)},{site(2),site(0),site(1)}};
  system.v_neighbours = {{site(0),site(1),site(2)},{site(1),site(0,-1),site(0)},{site(2),site(0)}};
  const std::size_t atoms = system.orbital_count.size();
  system.c_offset.resize(atoms);
  system.t.resize(atoms);
  system.v.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    std::vector<std::size_t>& offset = system.c_offset[a];
    offset.assign(1,0);
    for (const Site& c : system.c_neighbours[a])
      offset.push_back(offset.back()+system.orbital_count[static_cast<std::size_t>(c.atom)]);
    system.t[a].resize(system.abf_count[a]*system.orbital_count[a]*offset.back());
    for (double& value : system.t[a]) value = random.next();
    for (const Site& b : system.v_neighbours[a]) {
      system.v[a].emplace_back(system.abf_count[a]*system.abf_count[static_cast<std::size_t>(b.atom)]);
      for (double& value : system.v[a].back()) value = random.next();
    }
  }
  return system;
}

// Three atoms whose V neighbours and V blocks have the symmetry of
// prepare_system() that the half build needs: (a, -R) is a V neighbour of b
// with the transpose of the V block of (b, R) of a, and the V block of an
// atom with itself is symmetric.  Atom 0 has two images of itself and an
// image of atom 1 among its V neighbours, and the C neighbours of atom 1 hold
// two images of atom 0.  The half build of atoms 1 and 2 goes through the
// centre itself only.
System mirrored_system(Numbers& random)
{
  System system;
  system.orbital_count = {1,1,2};
  system.abf_count = {5,4,3};
  const auto site = [](int atom, int cell = 0) { return Site{atom,{cell,0,0}}; };
  system.c_neighbours = {{site(0)},{site(1),site(0),site(0,1)},{site(2),site(0),site(1)}};
  system.v_neighbours = {{site(0),site(0,-1),site(0,1),site(1),site(1,1),site(2)},
                         {site(1),site(0,-1),site(0)},
                         {site(2),site(0)}};
  const std::size_t atoms = system.orbital_count.size();
  system.c_offset.resize(atoms);
  system.t.resize(atoms);
  system.v.resize(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    std::vector<std::size_t>& offset = system.c_offset[a];
    offset.assign(1,0);
    for (const Site& c : system.c_neighbours[a])
      offset.push_back(offset.back()+system.orbital_count[static_cast<std::size_t>(c.atom)]);
    system.t[a].resize(system.abf_count[a]*system.orbital_count[a]*offset.back());
    for (double& value : system.t[a]) value = random.next();
    system.v[a].resize(system.v_neighbours[a].size());
  }
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t i = 0; i < system.v_neighbours[a].size(); ++i) {
      if (!system.v[a][i].empty()) continue;           // the transpose of a pair before
      const Site& b = system.v_neighbours[a][i];
      const std::size_t atom = static_cast<std::size_t>(b.atom);
      const std::size_t na = system.abf_count[a], nb = system.abf_count[atom];
      std::vector<double> block(na*nb);
      for (double& value : block) value = random.next();
      if (i == 0) {
        for (std::size_t p = 0; p < na; ++p)
          for (std::size_t q = 0; q < p; ++q) block[p*na+q] = block[q*na+p] = 0.5*(block[p*na+q]+block[q*na+p]);
      } else {
        const Site mirror{static_cast<int>(a),{-b.cell[0],-b.cell[1],-b.cell[2]}};
        const std::vector<Site>& list = system.v_neighbours[atom];
        const auto found = std::find(list.begin(),list.end(),mirror);
        require(found != list.end(),"a V neighbour of the mirrored system has no counterpart");
        std::vector<double> transposed(nb*na);
        for (std::size_t p = 0; p < na; ++p)
          for (std::size_t q = 0; q < nb; ++q) transposed[q*na+p] = block[p*nb+q];
        system.v[atom][static_cast<std::size_t>(found-list.begin())] = std::move(transposed);
      }
      system.v[a][i] = std::move(block);
    }
  return system;
}

// The blocks of a random density matrix of the site form with D_xy(R) =
// D_yx(-R)^T.
std::vector<double> symmetric_blocks(const System& system, const BlockLayout& layout, Numbers& random)
{
  std::vector<double> blocks(layout.elements());
  for (double& value : blocks) value = random.next();
  symmetrize_blocks(system,layout,blocks);
  require(block_asymmetry(system,layout,blocks) == 0.0,"the blocks of D are not symmetric");
  return blocks;
}

// K of the Gamma form from the definition, equations (8) and (9), with dense
// N x N matrices:  C^P = E_A T_P + T_P^T E_A^T  for every ABF, and
//   K = -alpha sum_PQ V_PQ C^P D C^Q.
// All images of an atom b have the same C^Q, so the V blocks of the images of
// b are added up first.  Nothing here is shared with the four terms of
// section 5.5.
std::vector<double> definition_gamma(const System& system, const std::vector<double>& density, double alpha)
{
  const std::vector<std::size_t> offset = offsets(system);
  const std::size_t atoms = system.orbital_count.size(), n = offset.back();

  // c[a][(P*n + i)*n + j] = C^P_ij for the ABFs P of atom a.
  std::vector<std::vector<double>> c(atoms);
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t np = system.abf_count[a], ua = system.orbital_count[a], ell = system.c_offset[a].back();
    c[a].assign(np*n*n,0.0);
    for (std::size_t i = 0; i < system.c_neighbours[a].size(); ++i) {
      const std::size_t site = static_cast<std::size_t>(system.c_neighbours[a][i].atom);
      for (std::size_t lambda = 0; lambda < system.orbital_count[site]; ++lambda)
        for (std::size_t p = 0; p < np; ++p)
          for (std::size_t mu = 0; mu < ua; ++mu) {
            const double value = system.t[a][(p*ua+mu)*ell+system.c_offset[a][i]+lambda];
            c[a][(p*n+offset[a]+mu)*n+offset[site]+lambda] += value;   // E_A T_P
            c[a][(p*n+offset[site]+lambda)*n+offset[a]+mu] += value;   // T_P^T E_A^T
          }
    }
  }

  std::vector<double> k(n*n,0.0), left(n*n), right(n*n), v;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::size_t np = system.abf_count[a];
    for (std::size_t b = 0; b < atoms; ++b) {
      const std::size_t nq = system.abf_count[b];
      v.assign(np*nq,0.0);
      for (std::size_t i = 0; i < system.v_neighbours[a].size(); ++i)
        if (static_cast<std::size_t>(system.v_neighbours[a][i].atom) == b)
          for (std::size_t e = 0; e < np*nq; ++e) v[e] += system.v[a][i][e];
      for (std::size_t p = 0; p < np; ++p) {
        // left = C^P D, right = sum_Q V_PQ C^Q
        const double* cp = c[a].data()+p*n*n;
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t l = 0; l < n; ++l) sum += cp[i*n+l]*density[l*n+j];
            left[i*n+j] = sum;
          }
        std::fill(right.begin(),right.end(),0.0);
        for (std::size_t q = 0; q < nq; ++q) {
          const double factor = v[p*nq+q];
          const double* cq = c[b].data()+q*n*n;
          for (std::size_t e = 0; e < n*n; ++e) right[e] += factor*cq[e];
        }
        for (std::size_t i = 0; i < n; ++i)
          for (std::size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (std::size_t l = 0; l < n; ++l) sum += left[i*n+l]*right[l*n+j];
            k[i*n+j] -= alpha*sum;
          }
      }
    }
  }
  return k;
}

} // namespace

void exchange_fixture(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size();
  std::set<BlockKey> read;
  const DensityBlocks density = fixture_density(fixture,&read);

  // The acceptance test of milestone M3: K and E_x of the fixture.
  const BlockMatrix k = exchange_blocks(system,density,fixture.alpha);
  Worst blocks("blocks of K against the reference implementation",1e-12);
  require(k.blocks().size() == fixture.k.size(),"the number of blocks of K differs from the fixture");
  for (const auto& [key, values] : fixture.k) {
    const std::span<const double> mine = k.find(key);
    require(mine.size() == values.size(),"a block of K of the fixture is missing");
    for (std::size_t e = 0; e < values.size(); ++e) blocks.absolute(mine[e],values[e]);
  }
  blocks.finish();
  require(read.size() == fixture.density.size(),"the build does not read every block of D of the fixture");
  std::printf("%zu blocks of K from %zu blocks of D\n",k.blocks().size(),read.size());

  Worst energy("E_x against the reference implementation",1e-12);
  energy.absolute(exchange_energy(density,k),fixture.energy);
  energy.finish();

  // D_xy(R) of the fixture is the transpose of D_yx(-R), so K has the same
  // property.
  Worst symmetry("K_xy(R) against the transpose of K_yx(-R)",1e-12);
  for (const auto& [key, values] : k.blocks()) {
    const std::span<const double> mirror = k.find(BlockKey{key.column,key.row,{-key.cell[0],-key.cell[1],-key.cell[2]}});
    require(!mirror.empty(),"K lacks the mirror image of a block");
    const std::size_t rows = system.orbital_count[static_cast<std::size_t>(key.row)];
    const std::size_t columns = system.orbital_count[static_cast<std::size_t>(key.column)];
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < columns; ++c) symmetry.absolute(values[r*columns+c],mirror[c*rows+r]);
  }
  symmetry.finish();

  // The contributions of the atoms can be added in any order, and the factor
  // can be applied at the end.
  {
    BlockMatrix reversed(system.orbital_count);
    for (std::size_t a = atoms; a-- > 0;) add_exchange_blocks(system,a,1.0,density,reversed);
    reversed.scale(-fixture.alpha);
    Worst order("atoms in reverse order, factor applied at the end",1e-12);
    compare_blocks(k,reversed,order);
    order.finish();
  }

  // Equation (15).  A block that one of the two matrices lacks counts as zero.
  {
    const BlockMatrix empty(system.orbital_count);
    BlockMatrix half = k;
    half.scale(0.5);
    Worst change("equation (15)",1e-12);
    change.absolute(exchange_change(density,k,empty),2.0*fixture.energy);
    change.absolute(exchange_change(density,empty,k),-2.0*fixture.energy);
    change.absolute(exchange_change(density,k,half),fixture.energy);
    change.absolute(exchange_change(density,k,k),0.0);
    change.finish();
  }

  // Invalid arguments.
  expect_invalid("a centre atom that does not exist",[&] {
    BlockMatrix m(system.orbital_count);
    add_exchange_blocks(system,atoms,1.0,density,m);
  });
  expect_invalid("a missing density matrix",[&] {
    BlockMatrix m(system.orbital_count);
    add_exchange_blocks(system,0,1.0,DensityBlocks(),m);
  });
  expect_invalid("blocks of another system",[&] {
    BlockMatrix m(std::vector<std::size_t>{1});
    add_exchange_blocks(system,0,1.0,density,m);
  });
  expect_invalid("a block of an atom that does not exist",[&] {
    BlockMatrix m(system.orbital_count);
    (void)m.at(BlockKey{0,7,{0,0,0}});
  });
  expect_invalid("the energy without a density matrix",[&] { (void)exchange_energy(DensityBlocks(),k); });
  expect_invalid("the change between matrices of two systems",[&] {
    (void)exchange_change(density,k,BlockMatrix(std::vector<std::size_t>{1}));
  });
  {
    System broken = system;
    std::swap(broken.c_neighbours[0][0],broken.c_neighbours[0][1]);
    expect_invalid("C neighbours that do not begin with the atom itself",[&] { (void)exchange_blocks(broken,density,0.25); });
    broken = system;
    std::swap(broken.v_neighbours[1][0],broken.v_neighbours[1][1]);
    expect_invalid("V neighbours that do not begin with the atom itself",[&] { (void)exchange_blocks(broken,density,0.25); });
    broken = system;
    broken.t[1].pop_back();
    expect_invalid("T blocks of the wrong size",[&] { (void)exchange_blocks(broken,density,0.25); });
    broken = system;
    broken.v[0][2].push_back(0.0);
    expect_invalid("a V block of the wrong size",[&] { (void)exchange_blocks(broken,density,0.25); });
    broken = system;
    broken.c_offset[0].back() += 1;
    expect_invalid("column offsets that do not match the neighbours",[&] { (void)exchange_blocks(broken,density,0.25); });
    broken = system;
    broken.abf_count.pop_back();
    expect_invalid("arrays for different numbers of atoms",[&] { (void)exchange_blocks(broken,density,0.25); });
  }
}

void exchange_layout(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size();
  const double alpha = fixture.alpha;
  const DensityBlocks density = fixture_density(fixture);
  const BlockLayout layout = exchange_layout(system);
  const std::size_t blocks = layout.keys.size();

  // The layout lists the blocks of K of the fixture in the order of their
  // keys, and the build reads the same blocks of D.
  require(blocks == fixture.k.size() && layout.offset.size() == blocks+1,"the layout does not list the blocks of K of the fixture");
  {
    std::size_t position = 0, elements = 0;
    for (const auto& [key, values] : fixture.k) {
      require(layout.keys[position] == key,"a key of the layout differs from the fixture");
      require(layout.offset[position] == elements,"a block of the layout does not follow the block before");
      require(layout.find(key) == position,"the layout does not find one of its keys");
      elements += values.size();
      ++position;
    }
    require(layout.elements() == elements,"the layout has a wrong number of elements");
    require(fixture.density.size() == blocks,"the fixture is expected to hold D for the blocks of K");
    for (const auto& [key, values] : fixture.density)
      require(layout.find(key) < blocks,"the layout lacks a block of D of the fixture");
    require(layout.find(BlockKey{0,0,{99,0,0}}) == blocks && layout.find(BlockKey{-1,0,{0,0,0}}) == blocks &&
            layout.find(BlockKey{static_cast<int>(atoms),0,{0,0,0}}) == blocks,"the layout finds a key that it does not hold");
    require(BlockLayout().elements() == 0 && BlockLayout().find(BlockKey{}) == 0,"an empty layout is not empty");
  }
  std::printf("%zu blocks with %zu elements\n",blocks,layout.elements());

  // The same numbers, added in the same order, as with a map of blocks.
  const std::vector<double> d = layout_values(layout,density);
  const std::vector<double> k = layout_exchange(system,layout,d,alpha);
  {
    const BlockMatrix reference = exchange_blocks(system,density,alpha);
    Worst same("K in one array against K in a map of blocks",0.0);
    Worst fixed("K in one array against the reference implementation",1e-12);
    for (std::size_t i = 0; i < blocks; ++i) {
      const std::span<const double> mine = std::span<const double>(k).subspan(layout.offset[i],layout.offset[i+1]-layout.offset[i]);
      const std::span<const double> other = reference.find(layout.keys[i]);
      const std::vector<double>& stored = fixture.k.at(layout.keys[i]);
      require(other.size() == mine.size() && stored.size() == mine.size(),"a block has the wrong size");
      for (std::size_t e = 0; e < mine.size(); ++e) {
        same.absolute(mine[e],other[e]);
        fixed.absolute(mine[e],stored[e]);
      }
    }
    same.finish();
    fixed.finish();
  }

  // Equations (4) and (15) are sums over the elements of the arrays.
  {
    Worst energy("E_x of the arrays against the reference implementation",1e-12);
    energy.absolute(exchange_energy_gamma(d,k),fixture.energy);
    const std::vector<double> zero(k.size(),0.0);
    energy.absolute(exchange_change_gamma(d,k,zero),2.0*fixture.energy);
    energy.finish();
  }

  // K_xy(R) is the transpose of K_yx(-R) for the density matrix of the
  // fixture.  The mean of the two keeps K within the rounding errors.
  {
    const double asymmetry = block_asymmetry(system,layout,k);
    std::printf("K_xy(R) against the transpose of K_yx(-R): largest difference %.3e\n",asymmetry);
    require(asymmetry <= 1e-12,"K of a symmetric density matrix is not symmetric");
    std::vector<double> symmetric = k;
    symmetrize_blocks(system,layout,symmetric);
    require(block_asymmetry(system,layout,symmetric) == 0.0,"the mean of two blocks is not symmetric");
    Worst moved("K after taking the means against K before",1e-12);
    for (std::size_t e = 0; e < k.size(); ++e) moved.absolute(symmetric[e],k[e]);
    moved.finish();
  }

  // An array without the symmetry: every element becomes the mean of itself
  // and its mirror image.
  {
    Numbers random;
    std::vector<double> m(layout.elements());
    for (double& value : m) value = random.next();
    require(block_asymmetry(system,layout,m) > 0.5,"numbers without a pattern are expected to lack the symmetry");
    std::vector<double> symmetric = m;
    symmetrize_blocks(system,layout,symmetric);
    require(block_asymmetry(system,layout,symmetric) == 0.0,"the mean of two blocks is not symmetric");
    Worst mean("the means against (M_xy(R) + M_yx(-R)^T)/2",0.0);
    for (std::size_t i = 0; i < blocks; ++i) {
      const BlockKey& key = layout.keys[i];
      const std::size_t j = layout.find(BlockKey{key.column,key.row,{-key.cell[0],-key.cell[1],-key.cell[2]}});
      require(j < blocks,"the layout lacks the mirror image of a block");
      const std::size_t rows = system.orbital_count[static_cast<std::size_t>(key.row)];
      const std::size_t columns = system.orbital_count[static_cast<std::size_t>(key.column)];
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c)
          mean.absolute(symmetric[layout.offset[i]+r*columns+c],
                        0.5*(m[layout.offset[i]+r*columns+c]+m[layout.offset[j]+c*rows+r]));
    }
    mean.finish();
  }

  // The atoms one by one in reverse order with the factor at the end, as
  // several processes add their parts.
  {
    std::vector<double> reversed(layout.elements(),0.0);
    for (std::size_t a = atoms; a-- > 0;) add_exchange_blocks(system,layout,a,1.0,d,reversed);
    Worst order("atoms in reverse order, factor applied at the end",1e-12);
    for (std::size_t e = 0; e < k.size(); ++e) order.absolute(-alpha*reversed[e],k[e]);
    order.finish();
  }

  // Invalid arguments.
  std::vector<double> target(layout.elements(),0.0);
  expect_invalid("a centre atom that does not exist",[&] { add_exchange_blocks(system,layout,atoms,1.0,d,target); });
  expect_invalid("a density matrix of the wrong size",[&] {
    add_exchange_blocks(system,layout,0,1.0,std::span<const double>(d).first(d.size()-1),target);
  });
  expect_invalid("an exchange matrix of the wrong size",[&] {
    add_exchange_blocks(system,layout,0,1.0,d,std::span<double>(target).first(target.size()-1));
  });
  expect_invalid("an array of the wrong size for the symmetry",[&] {
    (void)block_asymmetry(system,layout,std::span<const double>(d).first(d.size()-1));
  });
  expect_invalid("an array of the wrong size for the means",[&] {
    symmetrize_blocks(system,layout,std::span<double>(target).first(target.size()-1));
  });
  {
    // A layout without its last block: the array is shorter, and the build
    // and the symmetry miss a block.
    BlockLayout shorter = layout;
    shorter.keys.pop_back();
    shorter.offset.pop_back();
    std::vector<double> few(shorter.elements(),0.0), out(shorter.elements(),0.0);
    bool missed = false;
    for (std::size_t a = 0; a < atoms && !missed; ++a) {
      try { add_exchange_blocks(system,shorter,a,1.0,few,out); } catch (const std::invalid_argument&) { missed = true; }
    }
    require(missed,"no exception for a layout that lacks a block");
    // The same when the rows of D are gathered in pieces on threads of their
    // own: the exception of a piece reaches the caller.  The atoms of the
    // fixture have two and three ABFs, so both are divided into pieces.
    const OpenMXExxDenseBackend threaded{nullptr,nullptr,&threaded_loop,3,nullptr};
    bool missed_in_pieces = false;
    for (std::size_t a = 0; a < atoms && !missed_in_pieces; ++a) {
      try {
        add_exchange_blocks(system,shorter,a,1.0,few,out,DenseBackend(&threaded));
      } catch (const std::invalid_argument&) {
        missed_in_pieces = true;
      }
    }
    require(missed_in_pieces,"no exception for a layout that lacks a block when the rows of D are gathered in pieces");
    expect_invalid("the symmetry of a layout that lacks a block",[&] { (void)block_asymmetry(system,shorter,few); });

    BlockLayout broken = layout;
    broken.offset.back() += 1;
    std::vector<double> more(broken.elements(),0.0), more_out(broken.elements(),0.0);
    expect_invalid("a block of the wrong size",[&] { add_exchange_blocks(system,broken,0,1.0,more,more_out); });
    broken = layout;
    broken.keys.front().column = static_cast<int>(atoms);
    expect_invalid("a key with an atom that does not exist",[&] { add_exchange_blocks(system,broken,0,1.0,d,target); });
    broken = layout;
    broken.offset.pop_back();
    expect_invalid("offsets that do not match the keys",[&] { add_exchange_blocks(system,broken,0,1.0,d,target); });
  }
  require(std::all_of(target.begin(),target.end(),[](double value) { return value == 0.0; }),
          "a rejected call changes the exchange matrix");
}

void exchange_gamma(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  const double alpha = fixture.alpha;
  require(n == offsets(system).back(),"wrong number of orbitals of the unit cell");
  Numbers random;

  // The cell of the fixture is smaller than its cutoffs: images of one atom
  // occur several times in a neighbour list and fold onto the same orbitals.
  std::size_t images = 0;
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t i = 1; i < system.c_neighbours[a].size(); ++i)
      for (std::size_t j = 0; j < i; ++j)
        if (system.c_neighbours[a][i].atom == system.c_neighbours[a][j].atom) { ++images; break; }
  require(images >= 6,"the fixture is expected to list images of one atom as neighbours");

  for (const bool symmetric : {true,false}) {
    const std::vector<double> d = dense_matrix(n,symmetric,random);
    const std::string tag = symmetric ? " (symmetric D)" : " (D without symmetry)";
    const DensityBlocks blocks = gamma_density(system,d);
    const BlockMatrix site = exchange_blocks(system,blocks,alpha);
    const std::vector<double> k = exchange_gamma(system,d,alpha);

    // The acceptance test of milestone M3 for the Gamma form.
    Worst sum("Gamma form against sum_R K(R) of the site form"+tag,1e-12);
    const std::vector<double> reference = folded(system,site);
    for (std::size_t e = 0; e < n*n; ++e) sum.absolute(k[e],reference[e]);
    sum.finish();

    Worst definition("Gamma form against the definition with dense matrices"+tag,1e-12);
    const std::vector<double> direct = definition_gamma(system,d,alpha);
    for (std::size_t e = 0; e < n*n; ++e) definition.absolute(k[e],direct[e]);
    definition.finish();

    Worst energy("E_x of the two forms"+tag,1e-12);
    energy.absolute(exchange_energy_gamma(d,k),exchange_energy(blocks,site));
    energy.finish();

    if (symmetric) {
      Worst transposed("K of the Gamma form against its transpose",1e-12);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) transposed.absolute(k[i*n+j],k[j*n+i]);
      transposed.finish();
    }

    // The atoms one by one, last atom first, with the factor at the end.
    std::vector<double> reversed(n*n,0.0);
    for (std::size_t a = atoms; a-- > 0;) add_exchange_gamma(system,a,1.0,d,reversed);
    Worst order("atoms in reverse order, factor applied at the end"+tag,1e-12);
    for (std::size_t e = 0; e < n*n; ++e) order.absolute(-alpha*reversed[e],k[e]);
    order.finish();

    // Equation (15).
    std::vector<double> previous = k;
    for (double& value : previous) value *= 0.25;
    const double e_x = exchange_energy_gamma(d,k);
    Worst change("equation (15), Gamma form"+tag,1e-12);
    change.absolute(exchange_change_gamma(d,k,previous),1.5*e_x);
    change.absolute(exchange_change_gamma(d,k,k),0.0);
    change.finish();
  }

  // Invalid arguments.
  const std::vector<double> d = dense_matrix(n,true,random);
  std::vector<double> k(n*n,0.0);
  expect_invalid("a density matrix of the wrong size",[&] {
    add_exchange_gamma(system,0,1.0,std::span<const double>(d).first(n*n-1),k);
  });
  expect_invalid("an exchange matrix of the wrong size",[&] {
    add_exchange_gamma(system,0,1.0,d,std::span<double>(k).first(n));
  });
  expect_invalid("a centre atom that does not exist",[&] { add_exchange_gamma(system,atoms,1.0,d,k); });
  expect_invalid("an energy of matrices of different sizes",[&] {
    (void)exchange_energy_gamma(d,std::span<const double>(k).first(n));
  });
  expect_invalid("a change of matrices of different sizes",[&] {
    (void)exchange_change_gamma(d,k,std::span<const double>(k).first(n));
  });
  require(std::all_of(k.begin(),k.end(),[](double value) { return value == 0.0; }),
          "a rejected call changes the exchange matrix");
}

void exchange_gradient(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  const double alpha = fixture.alpha;
  Numbers random;
  const std::vector<double> d = dense_matrix(n,true,random);
  const auto energy = [&](const System& s) { return exchange_energy_gamma(d,exchange_gamma(s,d,alpha)); };

  const ExchangeGradient gradient = exchange_gradient_gamma(system,d,alpha);
  require(gradient.t.size() == atoms && gradient.v.size() == atoms,"the gradient has the wrong number of atoms");
  for (std::size_t a = 0; a < atoms; ++a) {
    require(gradient.t[a].size() == system.t[a].size() && gradient.v[a].size() == system.v[a].size(),
            "the gradient of an atom has the wrong size");
    for (std::size_t i = 0; i < system.v[a].size(); ++i)
      require(gradient.v[a][i].size() == system.v[a][i].size(),"the gradient of a V block has the wrong size");
  }

  // E_x is quadratic in the T blocks and linear in the V blocks, so a central
  // difference has no truncation error and the step can be large.
  const double h = 1.0e-3;
  double largest_t = 0.0, largest_v = 0.0;
  for (std::size_t a = 0; a < atoms; ++a) {
    largest_t = std::max(largest_t,largest(gradient.t[a]));
    for (const std::vector<double>& block : gradient.v[a]) largest_v = std::max(largest_v,largest(block));
  }
  require(largest_t > 0.1 && largest_v > 0.1,"the gradient of the fixture is expected to be of order one");

  // Every stored element on its own.
  System work = system;
  Worst each_t("equation (16), every element of T, against central differences",1e-7);
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t e = 0; e < system.t[a].size(); ++e) {
      work.t[a][e] = system.t[a][e]+h;
      const double plus = energy(work);
      work.t[a][e] = system.t[a][e]-h;
      const double minus = energy(work);
      work.t[a][e] = system.t[a][e];
      each_t.add(std::fabs(gradient.t[a][e]-(plus-minus)/(2.0*h))/largest_t);
    }
  each_t.finish();
  Worst each_v("equation (17), every element of V, against central differences",1e-7);
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t i = 0; i < system.v[a].size(); ++i)
      for (std::size_t e = 0; e < system.v[a][i].size(); ++e) {
        work.v[a][i][e] = system.v[a][i][e]+h;
        const double plus = energy(work);
        work.v[a][i][e] = system.v[a][i][e]-h;
        const double minus = energy(work);
        work.v[a][i][e] = system.v[a][i][e];
        each_v.add(std::fabs(gradient.v[a][i][e]-(plus-minus)/(2.0*h))/largest_v);
      }
  each_v.finish();

  // All elements at once, as test T6 of the reference implementation does.
  {
    System plus = system, minus = system;
    double analytic = 0.0;
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t e = 0; e < system.t[a].size(); ++e) {
        const double direction = random.next();
        analytic += gradient.t[a][e]*direction;
        plus.t[a][e] += h*direction;
        minus.t[a][e] -= h*direction;
      }
    Worst all("equation (16), a change of all T blocks (relative)",1e-7);
    all.relative(analytic,(energy(plus)-energy(minus))/(2.0*h));
    all.finish();
  }
  {
    System plus = system, minus = system;
    double analytic = 0.0;
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t i = 0; i < system.v[a].size(); ++i)
        for (std::size_t e = 0; e < system.v[a][i].size(); ++e) {
          const double direction = random.next();
          analytic += gradient.v[a][i][e]*direction;
          plus.v[a][i][e] += h*direction;
          minus.v[a][i][e] -= h*direction;
        }
    Worst all("equation (17), a change of all V blocks (relative)",1e-7);
    all.relative(analytic,(energy(plus)-energy(minus))/(2.0*h));
    all.finish();
  }

  // Euler's theorem for the two degrees of homogeneity.
  {
    double t_sum = 0.0, v_sum = 0.0;
    for (std::size_t a = 0; a < atoms; ++a) {
      for (std::size_t e = 0; e < system.t[a].size(); ++e) t_sum += system.t[a][e]*gradient.t[a][e];
      for (std::size_t i = 0; i < system.v[a].size(); ++i)
        for (std::size_t e = 0; e < system.v[a][i].size(); ++e) v_sum += system.v[a][i][e]*gradient.v[a][i][e];
    }
    const double e_x = energy(system);
    Worst euler("sum T dE/dT = 2 E_x and sum V dE/dV = E_x (relative)",1e-12);
    euler.relative(t_sum,2.0*e_x);
    euler.relative(v_sum,e_x);
    euler.finish();
  }

  expect_invalid("a density matrix of the wrong size",[&] {
    (void)exchange_gradient_gamma(system,std::span<const double>(d).first(n),alpha);
  });
}

void exchange_centre_gradient(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  const double alpha = fixture.alpha;
  Numbers random;
  const std::vector<double> d = dense_matrix(n,true,random);
  const BlockLayout layout = exchange_layout(system);

  // 1. The Gamma form against exchange_gradient_gamma(), which uses the
  //    arrays of both atoms of a pair, and the site form with the same density
  //    matrix in every block against the Gamma form.
  const ExchangeGradient reference = exchange_gradient_gamma(system,d,alpha);
  double largest_t = 0.0, largest_v = 0.0;
  for (std::size_t a = 0; a < atoms; ++a) {
    largest_t = std::max(largest_t,largest(reference.t[a]));
    for (const std::vector<double>& block : reference.v[a]) largest_v = std::max(largest_v,largest(block));
  }
  require(largest_t > 0.1 && largest_v > 0.1,"the gradient of the fixture is expected to be of order one");
  {
    const std::vector<double> blocks = layout_values(layout,gamma_density(system,d));
    Worst gamma_t("dE/dT of one centre, Gamma form, against equation (16)",1e-12);
    Worst gamma_v("dE/dV of one centre, Gamma form, against equation (17)",1e-12);
    Worst site_t("dE/dT of one centre, site form against Gamma form",1e-12);
    Worst site_v("dE/dV of one centre, site form against Gamma form",1e-12);
    for (std::size_t a = 0; a < atoms; ++a) {
      const CentreGradient gamma = centre_gradient_gamma(system,a,-alpha,d);
      const CentreGradient site = centre_gradient_blocks(system,layout,a,-alpha,blocks);
      require(gamma.t.size() == system.t[a].size() && gamma.v.size() == system.v[a].size() &&
              site.t.size() == system.t[a].size() && site.v.size() == system.v[a].size(),
              "the gradient of a centre has the wrong size");
      for (std::size_t e = 0; e < gamma.t.size(); ++e) {
        gamma_t.add(std::fabs(gamma.t[e]-reference.t[a][e])/largest_t);
        site_t.add(std::fabs(site.t[e]-gamma.t[e])/largest_t);
      }
      for (std::size_t i = 0; i < gamma.v.size(); ++i) {
        require(gamma.v[i].size() == system.v[a][i].size() && site.v[i].size() == system.v[a][i].size(),
                "the gradient of a V block has the wrong size");
        for (std::size_t e = 0; e < gamma.v[i].size(); ++e) {
          gamma_v.add(std::fabs(gamma.v[i][e]-reference.v[a][i][e])/largest_v);
          site_v.add(std::fabs(site.v[i][e]-gamma.v[i][e])/largest_v);
        }
      }
    }
    gamma_t.finish();
    gamma_v.finish();
    site_t.finish();
    site_v.finish();
  }

  // 2. The site form with the blocks of D of the fixture, which differ from
  //    cell to cell: every stored element against a central difference of
  //    E_x = (1/2) sum D K over the arrays.  E_x is quadratic in the T blocks
  //    and linear in the V blocks, so the step can be large.
  {
    const std::vector<double> blocks = layout_values(layout,fixture_density(fixture));
    const auto energy = [&](const System& s) {
      return exchange_energy_gamma(blocks,layout_exchange(s,layout,blocks,alpha));
    };
    std::vector<CentreGradient> gradient;
    double scale_t = 0.0, scale_v = 0.0;
    for (std::size_t a = 0; a < atoms; ++a) {
      gradient.push_back(centre_gradient_blocks(system,layout,a,-alpha,blocks));
      scale_t = std::max(scale_t,largest(gradient.back().t));
      for (const std::vector<double>& block : gradient.back().v) scale_v = std::max(scale_v,largest(block));
    }
    require(scale_t > 0.1 && scale_v > 0.1,"the gradient of the fixture is expected to be of order one");
    const double h = 1.0e-3;
    System work = system;
    Worst each_t("dE/dT of one centre, site form, against central differences",1e-9);
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t e = 0; e < system.t[a].size(); ++e) {
        work.t[a][e] = system.t[a][e]+h;
        const double plus = energy(work);
        work.t[a][e] = system.t[a][e]-h;
        const double minus = energy(work);
        work.t[a][e] = system.t[a][e];
        each_t.add(std::fabs(gradient[a].t[e]-(plus-minus)/(2.0*h))/scale_t);
      }
    each_t.finish();
    Worst each_v("dE/dV of one centre, site form, against central differences",1e-9);
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t i = 0; i < system.v[a].size(); ++i)
        for (std::size_t e = 0; e < system.v[a][i].size(); ++e) {
          work.v[a][i][e] = system.v[a][i][e]+h;
          const double plus = energy(work);
          work.v[a][i][e] = system.v[a][i][e]-h;
          const double minus = energy(work);
          work.v[a][i][e] = system.v[a][i][e];
          each_v.add(std::fabs(gradient[a].v[i][e]-(plus-minus)/(2.0*h))/scale_v);
        }
    each_v.finish();

    // Euler's theorem for the two degrees of homogeneity.
    double t_sum = 0.0, v_sum = 0.0;
    for (std::size_t a = 0; a < atoms; ++a) {
      for (std::size_t e = 0; e < system.t[a].size(); ++e) t_sum += system.t[a][e]*gradient[a].t[e];
      for (std::size_t i = 0; i < system.v[a].size(); ++i)
        for (std::size_t e = 0; e < system.v[a][i].size(); ++e) v_sum += system.v[a][i][e]*gradient[a].v[i][e];
    }
    Worst euler("sum T dE/dT = 2 E_x and sum V dE/dV = E_x, site form (relative)",1e-12);
    euler.relative(t_sum,2.0*fixture.energy);
    euler.relative(v_sum,fixture.energy);
    euler.finish();

    // The same numbers with a table of matrix products.
    long calls = 0;
    const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0,nullptr};
    const DenseBackend backend(&table);
    Worst product("the derivatives of one centre with a table of products",1e-12);
    for (std::size_t a = 0; a < atoms; ++a) {
      const CentreGradient site = centre_gradient_blocks(system,layout,a,-alpha,blocks,backend);
      for (std::size_t e = 0; e < site.t.size(); ++e) product.add(std::fabs(site.t[e]-gradient[a].t[e])/scale_t);
      for (std::size_t i = 0; i < site.v.size(); ++i)
        for (std::size_t e = 0; e < site.v[i].size(); ++e)
          product.add(std::fabs(site.v[i][e]-gradient[a].v[i][e])/scale_v);
    }
    product.finish();
    // Two products for the centre, one per W panel, and for every V
    // neighbour one for dV and four for every ABF of the centre.
    long expected = 0;
    for (std::size_t a = 0; a < atoms; ++a)
      expected += 2+v_product_panels(system.v_neighbours[a])+
                  static_cast<long>(system.v_neighbours[a].size())*(1+4*static_cast<long>(system.abf_count[a]));
    require(calls == expected,"unexpected number of matrix products of the derivatives");

    expect_invalid("a centre atom that does not exist",[&] {
      (void)centre_gradient_blocks(system,layout,atoms,-alpha,blocks);
    });
    expect_invalid("a density array of the wrong size",[&] {
      (void)centre_gradient_blocks(system,layout,0,-alpha,std::span<const double>(blocks).first(blocks.size()-1));
    });
  }
  expect_invalid("a centre atom that does not exist",[&] { (void)centre_gradient_gamma(system,atoms,-alpha,d); });
  expect_invalid("a density matrix of the wrong size",[&] {
    (void)centre_gradient_gamma(system,0,-alpha,std::span<const double>(d).first(n));
  });
}

void exchange_backend(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  const System& system = fixture.system;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  const double alpha = fixture.alpha;
  const DensityBlocks density = fixture_density(fixture);
  Numbers random;
  const std::vector<double> d = dense_matrix(n,true,random);

  long calls = 0;
  const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0,nullptr};
  const DenseBackend products(&table);
  // One centre atom takes two products of its own. Each W panel adds one
  // product, and every V neighbour still adds four for every ABF of the
  // centre. In the Gamma form there is one group per distinct atom.
  long expected = 0, expected_gamma = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    std::set<int> distinct;
    for (const Site& site : system.v_neighbours[a]) distinct.insert(site.atom);
    const long each = 1+4*static_cast<long>(system.abf_count[a]);
    expected += 2+v_product_panels(system.v_neighbours[a])+
                (each-1)*static_cast<long>(system.v_neighbours[a].size());
    expected_gamma += 2+each*static_cast<long>(distinct.size());
  }
  require(expected_gamma < expected,"the fixture is expected to list images of one atom as V neighbours");

  // The acceptance test of milestone M3: the same results with a table.
  {
    const BlockMatrix plain = exchange_blocks(system,density,alpha);
    require(calls == 0,"the built-in loops call the table");
    const BlockMatrix through = exchange_blocks(system,density,alpha,products);
    require(calls == expected,"the site form does not take the expected number of products from the table");
    Worst worst("site form: table of the tests against the built-in loops",1e-12);
    compare_blocks(plain,through,worst);
    worst.finish();
  }
  {
    calls = 0;
    const std::vector<double> plain = exchange_gamma(system,d,alpha);
    const std::vector<double> through = exchange_gamma(system,d,alpha,products);
    require(calls == expected_gamma,"the Gamma form does not take the expected number of products from the table");
    Worst worst("Gamma form: table of the tests against the built-in loops",1e-12);
    for (std::size_t e = 0; e < n*n; ++e) worst.absolute(plain[e],through[e]);
    worst.finish();
  }
  {
    calls = 0;
    const ExchangeGradient plain = exchange_gradient_gamma(system,d,alpha);
    const ExchangeGradient through = exchange_gradient_gamma(system,d,alpha,products);
    require(calls > 0,"the gradient does not use the table");
    Worst worst("gradient: table of the tests against the built-in loops",1e-12);
    for (std::size_t a = 0; a < atoms; ++a) {
      for (std::size_t e = 0; e < plain.t[a].size(); ++e) worst.absolute(plain.t[a][e],through.t[a][e]);
      for (std::size_t i = 0; i < plain.v[a].size(); ++i)
        for (std::size_t e = 0; e < plain.v[a][i].size(); ++e) worst.absolute(plain.v[a][i][e],through.v[a][i][e]);
    }
    worst.finish();
  }
  std::printf("products per build of the fixture: %ld in the site form, %ld in the Gamma form\n",expected,expected_gamma);

  // A table that fails stops the build.
  {
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr,nullptr,0,nullptr};
    const DenseBackend broken(&failing);
    expect_runtime_error("a failing table in the site form",[&] { (void)exchange_blocks(system,density,alpha,broken); });
    expect_runtime_error("a failing table in the Gamma form",[&] { (void)exchange_gamma(system,d,alpha,broken); });
    expect_runtime_error("a failing table in the gradient",[&] { (void)exchange_gradient_gamma(system,d,alpha,broken); });
  }

  // The loop of a table: the ABFs of a centre atom in pieces.  The loops of
  // the tests make the calls in order, from the last to the first, and on
  // threads of their own.  For one number of threads the three give the same
  // numbers bit for bit; the numbers of one piece agree within rounding.
  {
    const BlockMatrix plain_site = exchange_blocks(system,density,alpha);
    const std::vector<double> plain_gamma = exchange_gamma(system,d,alpha);
    const BlockLayout layout = exchange_layout(system);
    const std::vector<double> blocks_of_d = layout_values(layout,density);
    Calls shared;
    const auto with = [&shared](bool product, Loop loop, int threads) {
      const OpenMXExxDenseBackend pieces{product ? &shared_dgemm : nullptr,&shared,loop,threads,nullptr};
      return DenseBackend(&pieces);
    };

    // The number of pieces is the smaller of the threads and the ABFs of the
    // centre atom, and every loop of an atom is asked for one call for each
    // piece.  The atoms of the fixture have 2 and 3 ABFs: two threads give
    // two pieces for both atoms, three threads and 64 threads one piece for
    // every ABF.  Each atom has 9 V neighbours, images of both atoms, so a
    // build makes 2 + 2*9 loops for an atom in the site form with a map,
    // 4 + 2*9 in the site form in one array and 4 + 2*2 in the Gamma form
    // (the rule is at loops_of()).
    //
    // The two atoms mirror each other: the first has 3 orbitals and 2 ABFs,
    // the second 2 orbitals and 3 ABFs.  The loops of a whole build are
    // therefore the same for pieces that follow the orbitals in place of the
    // ABFs, and the checks below also count the loops of every atom alone.
    //
    // The C neighbours of the atoms hold images of one atom, so pairs of
    // sites share blocks of K when the site form in one array adds the last
    // result in pieces of the blocks.  The comparisons with the map below
    // then also check the order of the additions to those blocks.
    const int abfs[] = {2,3};
    require(system.orbital_count == std::vector<std::size_t>({3,2}) &&
            system.abf_count == std::vector<std::size_t>({2,3}) &&
            loops_of(system,2,Form::map) == Counts({{2,40}}) && loops_of(system,2,Form::layout) == Counts({{2,44}}) &&
            loops_of(system,2,Form::gamma) == Counts({{2,16}}) &&
            loops_of(system,3,Form::map) == Counts({{2,20},{3,20}}) &&
            loops_of(system,3,Form::layout) == Counts({{2,22},{3,22}}) &&
            loops_of(system,3,Form::gamma) == Counts({{2,8},{3,8}}) &&
            loops_of(system,64,Form::map) == loops_of(system,3,Form::map) &&
            loops_of(system,64,Form::layout) == loops_of(system,3,Form::layout) &&
            loops_of(system,64,Form::gamma) == loops_of(system,3,Form::gamma),
            "the fixture is expected to have atoms with 3 and 2 orbitals, 2 and 3 ABFs and 9 V neighbours each, images "
            "of both atoms");
    require(shared_pairs(system) > 0,"the fixture is expected to have pairs of sites that share a block of K");

    Worst rounding("pieces of the ABFs against one piece",1e-12);
    for (const bool product : {false,true}) {
      // One piece with the same products: the limit then measures the order
      // of the sums over the ABFs and nothing else.
      const DenseBackend whole = with(product,nullptr,0);
      const BlockMatrix one_site = exchange_blocks(system,density,alpha,whole);
      const std::vector<double> one_gamma = exchange_gamma(system,d,alpha,whole);
      if (!product)
        require(same_blocks(plain_site,one_site) && same_values(plain_gamma,one_gamma),
                "a table without a product and without a loop changes the build");
      for (const int threads : {2,3,64}) {
        const DenseBackend ordered = with(product,&in_order_loop,threads);
        shared.reset();
        const BlockMatrix site = exchange_blocks(system,density,alpha,ordered);
        require(shared.counts == loops_of(system,threads,Form::map),
                "the site form does not hand the expected loops and pieces to the table");
        // Only a table with a product counts products.
        if (product) require(shared.products == expected,"the pieces change the number of products of the site form");
        shared.reset();
        const std::vector<double> in_array = layout_exchange(system,layout,blocks_of_d,alpha,ordered);
        require(shared.counts == loops_of(system,threads,Form::layout),
                "the site form in one array does not hand the expected loops and pieces to the table");
        if (product)
          require(shared.products == expected,"the pieces change the number of products of the site form in one array");
        require(same_as_blocks(layout,in_array,site),
                "the pieces of the site form in one array differ from those in a map of blocks");
        shared.reset();
        const std::vector<double> gamma = exchange_gamma(system,d,alpha,ordered);
        require(shared.counts == loops_of(system,threads,Form::gamma),
                "the Gamma form does not hand the expected loops and pieces to the table");
        if (product)
          require(shared.products == expected_gamma,"the pieces change the number of products of the Gamma form");
        // The atoms one by one: every loop of an atom is asked for as many
        // calls as the atom has ABFs, or as the table has threads.
        for (std::size_t a = 0; a < atoms; ++a) {
          const int pieces = std::min(threads,abfs[a]);
          shared.reset();
          BlockMatrix part(system.orbital_count);
          add_exchange_blocks(system,a,-alpha,density,part,ordered);
          require(shared.counts == Counts({{pieces,2+2*9}}),
                  "the site form does not divide the ABFs of a centre atom into the expected pieces");
          shared.reset();
          std::vector<double> part_array(layout.elements(),0.0);
          add_exchange_blocks(system,layout,a,-alpha,blocks_of_d,part_array,ordered);
          require(shared.counts == Counts({{pieces,4+2*9}}),
                  "the site form in one array does not divide the work of a centre atom into the expected pieces");
          shared.reset();
          std::vector<double> part_gamma(n*n,0.0);
          add_exchange_gamma(system,a,-alpha,d,part_gamma,ordered);
          require(shared.counts == Counts({{pieces,4+2*2}}),
                  "the Gamma form does not divide the work of a centre atom into the expected pieces");
        }
        for (const Loop loop : {&reversed_loop,&threaded_loop}) {
          const DenseBackend other = with(product,loop,threads);
          require(same_blocks(site,exchange_blocks(system,density,alpha,other)),
                  "the site form depends on the order in which the pieces run");
          require(same_values(gamma,exchange_gamma(system,d,alpha,other)),
                  "the Gamma form depends on the order in which the pieces run");
          // The site form in one array of a layout, as the C interface
          // builds it: the same numbers, added in the same order.
          require(same_as_blocks(layout,layout_exchange(system,layout,blocks_of_d,alpha,other),site),
                  "the pieces of the site form in one array differ from those in a map of blocks");
        }
        compare_blocks(one_site,site,rounding);
        for (std::size_t e = 0; e < n*n; ++e) rounding.absolute(one_gamma[e],gamma[e]);
      }
    }
    rounding.finish();

    // With one thread the loop of the table is not called, and the numbers
    // are those without a table.
    {
      const DenseBackend single = with(false,&reversed_loop,1);
      shared.reset();
      require(same_blocks(plain_site,exchange_blocks(system,density,alpha,single)) &&
              same_values(plain_gamma,exchange_gamma(system,d,alpha,single)) && shared.loops == 0,
              "a table with one thread changes the build");
    }
    // A loop that fails, or leaves out a piece, stops the build.
    for (const Loop loop : {&failing_loop,&skipping_loop,&repeating_loop,&stray_loop}) {
      const DenseBackend broken = with(false,loop,2);
      expect_runtime_error("a broken loop in the site form",[&] { (void)exchange_blocks(system,density,alpha,broken); });
      expect_runtime_error("a broken loop in the Gamma form",[&] { (void)exchange_gamma(system,d,alpha,broken); });
    }

    // A product that fails inside one piece while the other pieces run.  The
    // product knows the piece from the index that the loops of the tests keep
    // for every thread, so nothing here depends on timing.
    //
    // The second atom has three ABFs, one for each of three pieces, and every
    // piece is the failing one in turn: the first, whose arrays take the sums
    // of the others, the second and the last.  The build of the atom makes
    // two products before its first loop with products.  In that loop the
    // other two pieces make their four products each, and the failing piece
    // ends with its first product.  The build ends after this loop.  It is
    // the second loop that the site form with a map hands to the table, and
    // the third of the Gamma form and of the site form in one array, whose
    // first loop gathers the rows of D.  The three forms of the build go
    // through this: the blocks in a map, the Gamma form, and the site form in
    // one array of a layout, which is the one that the C interface calls.
    //
    // A whole build ends in the same way at its first atom, which has two
    // ABFs: one piece fails and one runs, and either piece is the failing one
    // in turn.
    {
      Calls failing;
      for (const Loop loop : {&in_order_loop,&reversed_loop,&threaded_loop}) {
        const OpenMXExxDenseBackend table{&piece_failing_dgemm,&failing,loop,3,nullptr};
        const DenseBackend broken(&table);
        for (const int piece : {0,1,2}) {
          failing.failing = piece;
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of the site form",[&] {
            BlockMatrix k(system.orbital_count);
            add_exchange_blocks(system,1,-alpha,density,k,broken);
          });
          require(failing.products == 2+4*2+1 && failing.counts == Counts({{3,2}}),
                  "the site form does not run the other pieces, or goes on after the loop in which a product fails");
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of the Gamma form",[&] {
            std::vector<double> k(n*n,0.0);
            add_exchange_gamma(system,1,-alpha,d,k,broken);
          });
          require(failing.products == 2+4*2+1 && failing.counts == Counts({{3,3}}),
                  "the Gamma form does not run the other pieces, or goes on after the loop in which a product fails");
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of the site form in one array",[&] {
            std::vector<double> k(layout.elements(),0.0);
            add_exchange_blocks(system,layout,1,-alpha,blocks_of_d,k,broken);
          });
          require(failing.products == 2+4*2+1 && failing.counts == Counts({{3,3}}),
                  "the site form in one array does not run the other pieces, or goes on after the loop in which a "
                  "product fails");
        }
        for (const int piece : {0,1}) {
          failing.failing = piece;
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of a build of the site form",
                               [&] { (void)exchange_blocks(system,density,alpha,broken); });
          require(failing.products == 2+4*1+1 && failing.counts == Counts({{2,2}}),
                  "a build of the site form goes on after a product has failed inside a piece");
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of a build of the site form in one array",
                               [&] { (void)layout_exchange(system,layout,blocks_of_d,alpha,broken); });
          require(failing.products == 2+4*1+1 && failing.counts == Counts({{2,3}}),
                  "a build of the site form in one array goes on after a product has failed inside a piece");
          failing.reset();
          expect_runtime_error("a product that fails inside a piece of a build of the Gamma form",
                               [&] { (void)exchange_gamma(system,d,alpha,broken); });
          require(failing.products == 2+4*1+1 && failing.counts == Counts({{2,3}}),
                  "a build of the Gamma form goes on after a product has failed inside a piece");
        }
      }
    }

    // The derivatives divide the work of the single ABFs into pieces: with a
    // table that has a loop they give the bits of the same table without the
    // loop, and the loop is called.  So does the half form, whose parts
    // through W are sums over the ABFs of the call.
    for (const bool product : {false,true}) {
      const DenseBackend whole = with(product,nullptr,0), pieces = with(product,&threaded_loop,3);
      shared.reset();
      const ExchangeGradient all = exchange_gradient_gamma(system,d,alpha,whole);
      const ExchangeGradient again = exchange_gradient_gamma(system,d,alpha,pieces);
      bool same = same_values(all.t,again.t) && same_values(all.v,again.v);
      const auto same_gradient = [](const CentreGradient& x, const CentreGradient& y) {
        return same_values(x.t,y.t) && same_values(x.v,y.v) && same_values(x.w,y.w);
      };
      for (std::size_t a = 0; a < atoms; ++a) {
        const CentreGradient gamma = centre_gradient_gamma(system,a,-alpha,d,whole);
        const CentreGradient gamma_again = centre_gradient_gamma(system,a,-alpha,d,pieces);
        const CentreGradient site = centre_gradient_blocks(system,layout,a,-alpha,blocks_of_d,whole);
        const CentreGradient site_again = centre_gradient_blocks(system,layout,a,-alpha,blocks_of_d,pieces);
        same = same && same_gradient(gamma,gamma_again) && same_gradient(site,site_again);
        const std::size_t count = system.abf_count[a];
        same = same && same_gradient(half_gradient_gamma(system,a,0,count,-alpha,d,whole),
                                     half_gradient_gamma(system,a,0,count,-alpha,d,pieces)) &&
               same_gradient(half_gradient_blocks(system,layout,a,0,count,-alpha,blocks_of_d,whole),
                             half_gradient_blocks(system,layout,a,0,count,-alpha,blocks_of_d,pieces));
      }
      require(same,"a table with a loop changes the derivatives");
      require(shared.loops > 0,"the derivatives do not call the loop of the table");
    }

    // Sums that are shorter than the number of pieces.  The sums of the
    // second term of a V neighbour have u_a l_b elements, and the build adds
    // those of the pieces in a loop that gives every call a share of the
    // elements (exchange.cpp, "The sums of the pieces").  With fewer elements
    // than pieces some calls get no element.  The fixture does not reach
    // that: its sums have 10 elements or more for at most 3 pieces.  A
    // calculation reaches it with a centre atom of few orbitals and many
    // ABFs on many threads.  In the system below every atom has such a V
    // neighbour with as many threads as the first atom has ABFs, and the
    // third atom has one whose sums are longer.
    {
      const System narrow = narrow_system(random);
      const std::size_t count = orbital_total(narrow);
      const int threads = 5;
      for (const bool gamma : {false,true}) {
        std::size_t shorter = 0, longer = 0;
        for (std::size_t a = 0; a < narrow.orbital_count.size(); ++a) {
          const std::size_t pieces = std::min(static_cast<std::size_t>(threads),narrow.abf_count[a]);
          bool found = false;
          for (const Site& b : narrow.v_neighbours[a]) {
            const std::size_t columns = column_count(narrow,static_cast<std::size_t>(b.atom),gamma);
            const bool few = narrow.orbital_count[a]*columns < pieces;
            found = found || few;
            if (!few) ++longer;
          }
          if (found) ++shorter;
        }
        require(shorter == narrow.orbital_count.size() && longer > 0,
                "the small system is expected to have sums that are shorter than the number of pieces");
      }
      const std::vector<double> dn = dense_matrix(count,true,random);
      const DensityBlocks blocks = gamma_density(narrow,dn);
      const std::vector<double> one_gamma = exchange_gamma(narrow,dn,alpha);
      const BlockMatrix one_site = exchange_blocks(narrow,blocks,alpha);
      const double scale = largest(one_gamma);
      require(scale > 1.0e-3,"the exchange matrix of the small system is expected to be of order one");
      {
        // The system is made by hand: the definition with dense matrices and
        // the sum over the cells show that the numbers of one piece are K.
        const std::vector<double> direct = definition_gamma(narrow,dn,alpha), sum = folded(narrow,one_site);
        Worst definition("small system: one piece against the definition and sum_R K(R) (relative)",1e-12);
        for (std::size_t e = 0; e < count*count; ++e) {
          definition.add(std::fabs(one_gamma[e]-direct[e])/scale);
          definition.add(std::fabs(one_gamma[e]-sum[e])/scale);
        }
        definition.finish();
      }
      // The loops in literal numbers, from the lists of narrow_system().  The
      // atoms have 5, 4 and 3 ABFs, with five threads one piece for each, and
      // 3, 3 and 2 V neighbours, which are 3, 2 and 2 atoms: an atom hands
      // the table 2 + 2*3 or 2 + 2*2 loops in the site form with a map, and
      // two loops more in the other two forms.
      //
      // The rows of D are gathered in as many pieces.  The first atom has one
      // C neighbour, one orbital and five pieces: four pieces of the gather
      // in the site form in one array, and of the gather and of the last
      // additions in the Gamma form, have no row.  The second atom holds two
      // images of the first atom among its C neighbours, so pairs of sites
      // share blocks of K.
      require(shared_pairs(narrow) > 0,"the small system is expected to have pairs of sites that share a block of K");
      const BlockLayout narrow_layout = exchange_layout(narrow);
      const std::vector<double> narrow_d = layout_values(narrow_layout,blocks);
      shared.reset();
      const BlockMatrix site = exchange_blocks(narrow,blocks,alpha,with(false,&in_order_loop,threads));
      require(shared.counts == Counts({{5,2+2*3},{4,2+2*3},{3,2+2*2}}),
              "the site form of the small system does not hand the expected loops and pieces to the table");
      shared.reset();
      const std::vector<double> in_array =
          layout_exchange(narrow,narrow_layout,narrow_d,alpha,with(false,&in_order_loop,threads));
      require(shared.counts == Counts({{5,4+2*3},{4,4+2*3},{3,4+2*2}}),
              "the site form in one array of the small system does not hand the expected loops and pieces to the "
              "table");
      require(same_as_blocks(narrow_layout,in_array,site),
              "the site form in one array of the small system differs from the site form with a map");
      shared.reset();
      const std::vector<double> gamma = exchange_gamma(narrow,dn,alpha,with(false,&in_order_loop,threads));
      require(shared.counts == Counts({{5,4+2*3},{4,4+2*2},{3,4+2*2}}),
              "the Gamma form of the small system does not hand the expected loops and pieces to the table");
      for (const Loop loop : {&reversed_loop,&threaded_loop}) {
        require(same_blocks(site,exchange_blocks(narrow,blocks,alpha,with(false,loop,threads))),
                "the site form of the small system depends on the order in which the pieces run");
        require(same_values(in_array,layout_exchange(narrow,narrow_layout,narrow_d,alpha,with(false,loop,threads))),
                "the site form in one array of the small system depends on the order in which the pieces run");
        require(same_values(gamma,exchange_gamma(narrow,dn,alpha,with(false,loop,threads))),
                "the Gamma form of the small system depends on the order in which the pieces run");
      }
      Worst sums("small system: sums shorter than the number of pieces against one piece (relative)",1e-12);
      require(site.blocks().size() == one_site.blocks().size(),"the pieces change the number of blocks of K");
      for (const auto& [key, values] : one_site.blocks()) {
        const std::span<const double> other = site.find(key);
        require(other.size() == values.size(),"the pieces change the blocks of K");
        for (std::size_t e = 0; e < values.size(); ++e) sums.add(std::fabs(values[e]-other[e])/scale);
      }
      for (std::size_t e = 0; e < count*count; ++e) sums.add(std::fabs(one_gamma[e]-gamma[e])/scale);
      sums.finish();
    }
  }
}

namespace {

// Radial functions of a species on the mesh of the fixture.
struct Shells {
  double cutoff = 0.0;
  std::vector<double> radii;
  std::vector<int> angular;
  std::vector<double> values; // [shell*radii.size() + point]
};

// Some of the shells of a species of the overlap fixture.
Shells select(const FixtureSpecies& species, const std::vector<std::size_t>& chosen)
{
  Shells shells{species.cutoff,species.radii,{},{}};
  for (const std::size_t index : chosen) {
    require(index < species.shells.size(),"the overlap fixture lacks a shell that the test selects");
    shells.angular.push_back(species.shells[index].l);
    shells.values.insert(shells.values.end(),species.shells[index].values.begin(),species.shells[index].values.end());
  }
  return shells;
}

std::vector<SpeciesTable> tables_of(const std::vector<Shells>& species, int intervals, const ReciprocalGrid& reciprocal,
                                    double threshold)
{
  std::vector<SpeciesTable> tables;
  for (const Shells& item : species) {
    const RadialGrid grid{0.0,item.cutoff/intervals,intervals};
    const std::size_t points = item.radii.size();
    std::vector<MeshFunction> functions;
    for (std::size_t shell = 0; shell < item.angular.size(); ++shell)
      functions.push_back({item.angular[shell],std::span<const double>(item.values).subspan(shell*points,points)});
    const std::vector<RadialFunction> orbitals = sample_orbitals(grid,item.radii,functions);
    tables.emplace_back(grid,orbitals,make_auxiliary_basis(grid,orbitals,threshold).functions,reciprocal);
  }
  return tables;
}

// The same preparation through the C interface.
struct CGeometry {
  std::vector<OpenMXExxSpecies> species;
  std::vector<double> positions;
  OpenMXExxGeometry geometry{};
  CGeometry(const Structure& structure, const std::vector<Shells>& shells)
  {
    for (const Shells& item : shells)
      species.push_back({item.cutoff,static_cast<int>(item.radii.size()),item.radii.data(),
                         static_cast<int>(item.angular.size()),item.angular.data(),item.values.data()});
    for (const std::array<double,3>& position : structure.positions)
      positions.insert(positions.end(),position.begin(),position.end());
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t x = 0; x < 3; ++x) geometry.lattice[i][x] = structure.lattice[i][x];
    geometry.atom_count = static_cast<int>(structure.positions.size());
    geometry.positions = positions.data();
    geometry.species = structure.species.data();
    geometry.species_count = static_cast<int>(species.size());
    geometry.species_data = species.data();
  }
  CGeometry(const CGeometry&) = delete;
  CGeometry& operator=(const CGeometry&) = delete;
};

// K of all atoms through the C interface.
std::vector<double> c_exchange(const OpenMXExxState* state, const std::vector<double>& density, double alpha)
{
  const int n = openmx_exx_orbital_count(state);
  require(n > 0 && density.size() == static_cast<std::size_t>(n)*static_cast<std::size_t>(n),
          "the C interface reports a wrong number of orbitals");
  std::vector<double> k(density.size(),0.0);
  char message[200] = "untouched";
  // The call is a statement of its own, here and wherever the text of a
  // failure holds the message of a call: a compiler may build the second
  // argument of require() before it makes a call in the first one, and the
  // text then holds what the message was before the call.
  const int status = openmx_exx_add_gamma(state,0,openmx_exx_atom_count(state),-alpha,density.data(),k.data(),nullptr,
                                          message,sizeof message);
  require(status == 0,std::string("the build through the C interface fails: ")+message);
  require(message[0] == '\0',"a successful call leaves a message");
  return k;
}

// The C interface must describe the same system as the C++ interface and give
// the same exchange matrix, bit for bit: it runs the same code on the same
// numbers.
// "listed" tells whether the state holds the list of blocks of its structure
// already: a preparation discards the list, a preparation that fails keeps it.
void compare_with_c(OpenMXExxState* state, const System& system, const std::vector<double>& density, double alpha,
                    const std::string& what, bool listed = false)
{
  const std::size_t atoms = system.orbital_count.size();
  require(openmx_exx_atom_count(state) == static_cast<int>(atoms) &&
          openmx_exx_orbital_count(state) == static_cast<int>(orbital_total(system)),what+": wrong sizes of the C state");
  for (std::size_t a = 0; a < atoms; ++a) {
    const int atom = static_cast<int>(a);
    require(openmx_exx_atom_orbitals(state,atom) == static_cast<int>(system.orbital_count[a]) &&
            openmx_exx_atom_abfs(state,atom) == static_cast<int>(system.abf_count[a]) &&
            openmx_exx_c_neighbours(state,atom) == static_cast<int>(system.c_neighbours[a].size()) &&
            openmx_exx_v_neighbours(state,atom) == static_cast<int>(system.v_neighbours[a].size()),
            what+": wrong sizes of an atom of the C state");
  }
  OpenMXExxReport report{};
  require(openmx_exx_report(state,&report) == 0 && report.systems == system.report.systems &&
          report.eigen_decompositions == system.report.eigen_decompositions && report.dropped == system.report.dropped &&
          report.smallest_pivot == system.report.smallest_pivot,what+": wrong report of the C state");

  const std::vector<double> k = exchange_gamma(system,density,alpha), through = c_exchange(state,density,alpha);
  Worst worst(what+": C interface against the C++ interface",0.0);
  for (std::size_t e = 0; e < k.size(); ++e) worst.absolute(through[e],k[e]);
  worst.finish();
  require(openmx_exx_energy_gamma(k.size(),density.data(),through.data()) == exchange_energy_gamma(density,k),
          what+": the two interfaces give different energies");

  // The site form.  A preparation discards the list of blocks of the
  // structure before; the new list is that of the C++ interface.
  const BlockLayout layout = exchange_layout(system);
  const long blocks = static_cast<long>(layout.keys.size());
  const int none[3] = {99,0,0};
  char message[200] = "untouched";
  if (!listed) {
    require(openmx_exx_block_count(state) == -1 && openmx_exx_block_elements(state) == 0 &&
            openmx_exx_block(state,0,nullptr,nullptr,nullptr,nullptr) != 0 && openmx_exx_find_block(state,0,0,none) == -1,
            what+": a list of blocks exists before it is made");
    std::vector<double> nothing(1,0.0);
    require(openmx_exx_add_blocks(state,0,1,-alpha,nothing.data(),nothing.data(),nullptr,message,sizeof message) != 0 &&
            message[0] != '\0' && nothing[0] == 0.0,what+": a build without a list of blocks is accepted");
    require(openmx_exx_symmetrize_blocks(state,nothing.data(),nullptr,message,sizeof message) != 0 && message[0] != '\0',
            what+": means without a list of blocks are accepted");
  }
  else
    require(openmx_exx_block_count(state) == blocks,what+": a preparation that fails discards the list of blocks");
  require(openmx_exx_prepare_blocks(state,message,sizeof message) == 0 && message[0] == '\0',
          what+": the list of blocks cannot be made");
  require(openmx_exx_prepare_blocks(state,nullptr,0) == 0,what+": the list of blocks cannot be asked for twice");
  require(openmx_exx_block_count(state) == blocks && openmx_exx_block_elements(state) == layout.elements(),
          what+": wrong size of the list of blocks");
  for (long i = 0; i < blocks; ++i) {
    const BlockKey& key = layout.keys[static_cast<std::size_t>(i)];
    int row = -1, column = -1, cell[3] = {0,0,0};
    std::size_t offset = 0;
    require(openmx_exx_block(state,i,&row,&column,cell,&offset) == 0 && row == key.row && column == key.column &&
            cell[0] == key.cell[0] && cell[1] == key.cell[1] && cell[2] == key.cell[2] &&
            offset == layout.offset[static_cast<std::size_t>(i)] && openmx_exx_find_block(state,row,column,cell) == i &&
            openmx_exx_block(state,i,nullptr,nullptr,nullptr,nullptr) == 0,what+": a block of the C state differs");
  }
  require(openmx_exx_block(state,blocks,nullptr,nullptr,nullptr,nullptr) != 0 &&
          openmx_exx_block(state,-1,nullptr,nullptr,nullptr,nullptr) != 0 && openmx_exx_find_block(state,0,0,none) == -1 &&
          openmx_exx_find_block(state,0,0,nullptr) == -1,what+": a block that does not exist is found");

  // D_xy(R) = D_xy for every R: the sum over R of the blocks of K is K of the
  // Gamma form.
  const std::vector<double> blocks_of_d = layout_values(layout,gamma_density(system,density));
  const std::vector<double> site = layout_exchange(system,layout,blocks_of_d,alpha);
  std::vector<double> site_through(layout.elements(),0.0);
  const int atom_count = static_cast<int>(atoms);
  for (int atom = 0; atom < atom_count; ++atom) {
    const int status = openmx_exx_add_blocks(state,atom,atom+1,-alpha,blocks_of_d.data(),site_through.data(),nullptr,
                                             message,sizeof message);
    require(status == 0 && message[0] == '\0',
            what+": the build of the site form through the C interface fails: "+message);
  }
  Worst same(what+": site form, C interface against C++",0.0);
  for (std::size_t e = 0; e < site.size(); ++e) same.absolute(site_through[e],site[e]);
  same.finish();
  {
    const std::vector<std::size_t> offset = offsets(system);
    const std::size_t n = offset.back();
    std::vector<double> sum(n*n,0.0);
    for (std::size_t i = 0; i < layout.keys.size(); ++i) {
      const std::size_t x = static_cast<std::size_t>(layout.keys[i].row), y = static_cast<std::size_t>(layout.keys[i].column);
      const std::size_t rows = system.orbital_count[x], columns = system.orbital_count[y];
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c)
          sum[(offset[x]+r)*n+offset[y]+c] += site_through[layout.offset[i]+r*columns+c];
    }
    const double scale = largest(k);
    Worst folded_sum(what+": sum_R K(R) against the Gamma form (relative)",1e-12);
    for (std::size_t e = 0; e < n*n; ++e) folded_sum.add(std::fabs(sum[e]-k[e])/scale);
    folded_sum.finish();
  }
  require(openmx_exx_energy_gamma(site.size(),blocks_of_d.data(),site_through.data()) == exchange_energy_gamma(blocks_of_d,site),
          what+": the two interfaces give different energies of the site form");

  // The means of the mirror images.
  {
    std::vector<double> mine = site, theirs = site_through;
    const double expected = block_asymmetry(system,layout,site);
    double asymmetry = -1.0;
    symmetrize_blocks(system,layout,mine);
    require(openmx_exx_symmetrize_blocks(state,theirs.data(),&asymmetry,message,sizeof message) == 0 && message[0] == '\0',
            what+": the means through the C interface fail");
    require(asymmetry == expected && asymmetry <= 1e-12*largest(site) && mine == theirs,
            what+": the means of the two interfaces differ");
    require(openmx_exx_symmetrize_blocks(state,theirs.data(),nullptr,nullptr,0) == 0 && mine == theirs,
            what+": the means change a symmetric array");
    require(openmx_exx_symmetrize_blocks(state,nullptr,&asymmetry,message,sizeof message) != 0 && message[0] != '\0',
            what+": means without an array are accepted");
  }

  // Rejected builds add nothing.
  {
    std::vector<double> untouched = site_through;
    require(openmx_exx_add_blocks(state,-1,atom_count,-alpha,blocks_of_d.data(),untouched.data(),nullptr,message,
                                  sizeof message) != 0 &&
            openmx_exx_add_blocks(state,0,atom_count+1,-alpha,blocks_of_d.data(),untouched.data(),nullptr,message,
                                  sizeof message) != 0 &&
            openmx_exx_add_blocks(state,0,atom_count,-alpha,nullptr,untouched.data(),nullptr,message,sizeof message) != 0 &&
            openmx_exx_add_blocks(state,0,atom_count,-alpha,blocks_of_d.data(),nullptr,nullptr,message,sizeof message) != 0 &&
            untouched == site_through,what+": a rejected build of the site form changes the matrix");
  }
}

// The ABFs in ranges through the C interface (section 6.7): the same bits as
// the C++ interface for the same ranges in both forms, all ABFs in one range
// against all centre atoms, and the weights of the ABFs.  The state holds the
// list of blocks of its structure.
void compare_ranges_with_c(const OpenMXExxState* state, const System& system, const std::vector<double>& density,
                           double alpha, const std::string& what)
{
  const std::size_t atoms = system.orbital_count.size();
  std::size_t total = 0;
  for (const std::size_t count : system.abf_count) total += count;
  require(openmx_exx_abf_count(state) == static_cast<long>(total) && openmx_exx_abf_count(nullptr) == -1,
          what+": wrong number of ABFs of the C state");
  // The division of the ABFs among processes, in both forms and for the
  // whole and the half build: the C interface writes the numbers of the C++
  // interface and no more, and rejects a missing state or array and fewer
  // than one process.
  for (const int sites : {0,1}) {
    for (const int half : {0,1}) {
      for (const int processes : {1,2,3,5,static_cast<int>(atoms)+4}) {
        const std::vector<std::size_t> expected =
            divide_abfs(system,sites != 0,half != 0,static_cast<std::size_t>(processes));
        std::vector<long> boundaries(expected.size()+1,-1);
        require(openmx_exx_divide_abfs(state,sites,half,processes,boundaries.data()) == 0 &&
                    boundaries.back() == -1 &&
                    std::equal(expected.begin(),expected.end(),boundaries.begin(),
                               [](std::size_t x, long y) { return y >= 0 && static_cast<std::size_t>(y) == x; }),
                what+": the divisions of the ABFs of the two interfaces differ");
      }
      std::vector<long> untouched(3,-1);
      require(openmx_exx_divide_abfs(state,sites,half,2,nullptr) != 0 &&
                  openmx_exx_divide_abfs(nullptr,sites,half,2,untouched.data()) != 0 &&
                  openmx_exx_divide_abfs(state,sites,half,0,untouched.data()) != 0 &&
                  openmx_exx_divide_abfs(state,sites,half,-3,untouched.data()) != 0 &&
                  untouched == std::vector<long>(3,-1),
              what+": a division of the ABFs without a state, an array or a process is accepted");
    }
  }

  const long all = static_cast<long>(total), cut = all/3, other = all-2;
  require(0 < cut && cut < other && other < all,what+": the test of the ranges needs more ABFs");
  const long ranges[][2] = {{0,cut},{cut,other},{other,all}};
  char message[200] = "untouched";
  const BlockLayout layout = exchange_layout(system);
  const std::vector<double> blocks_of_d = layout_values(layout,gamma_density(system,density));
  const int atom_count = static_cast<int>(atoms);
  for (const bool sites : {false,true}) {
    const std::vector<double>& d = sites ? blocks_of_d : density;
    const std::string form = what+(sites ? ", site form" : ", Gamma form");
    std::vector<double> through(d.size(),0.0), direct(d.size(),0.0), whole(d.size(),0.0), centres(d.size(),0.0);
    const auto add = [&](long first, long last, std::vector<double>& k, const OpenMXExxDenseBackend* table) {
      return sites ? openmx_exx_add_blocks_abfs(state,first,last,-alpha,d.data(),k.data(),table,message,sizeof message)
                   : openmx_exx_add_gamma_abfs(state,first,last,-alpha,d.data(),k.data(),table,message,sizeof message);
    };
    for (const auto& range : ranges) {
      const int status = add(range[0],range[1],through,nullptr);
      require(status == 0 && message[0] == '\0',form+": a range of ABFs through the C interface fails: "+message);
      const std::size_t first = static_cast<std::size_t>(range[0]), last = static_cast<std::size_t>(range[1]);
      if (sites) add_exchange_blocks_abfs(system,layout,first,last,-alpha,d,direct);
      else add_exchange_gamma_abfs(system,first,last,-alpha,d,direct);
    }
    require(same_values(through,direct),form+": the ranges of ABFs of the two interfaces give different bits");
    int status = add(0,all,whole,nullptr);
    require(status == 0,form+": all ABFs in one range through the C interface fail: "+message);
    status = sites ? openmx_exx_add_blocks(state,0,atom_count,-alpha,d.data(),centres.data(),nullptr,message,sizeof message)
                   : openmx_exx_add_gamma(state,0,atom_count,-alpha,d.data(),centres.data(),nullptr,message,sizeof message);
    require(status == 0 && same_values(whole,centres),form+": all ABFs in one range give other bits than all centre atoms");
    const double scale = largest(whole);
    Worst rounding(form+": three ranges of ABFs through C against one (relative)",1e-12);
    for (std::size_t e = 0; e < whole.size(); ++e) rounding.add(std::fabs(through[e]-whole[e])/scale);
    rounding.finish();

    // The pieces of a range on threads of their own, as a process with
    // threads runs its share.
    {
      Calls shared;
      const OpenMXExxDenseBackend table{&shared_dgemm,&shared,&threaded_loop,4,nullptr};
      std::vector<double> threaded(d.size(),0.0);
      for (const auto& range : ranges) {
        status = add(range[0],range[1],threaded,&table);
        require(status == 0 && message[0] == '\0',form+": a range of ABFs with pieces on threads fails: "+message);
      }
      require(shared.loops > 0 && shared.products > 0,form+": the ranges of ABFs do not use the loop and the product of the table");
      Worst pieces(form+": ranges of ABFs with pieces on threads (relative)",1e-12);
      for (std::size_t e = 0; e < whole.size(); ++e) pieces.add(std::fabs(threaded[e]-whole[e])/scale);
      pieces.finish();
    }

    // An empty range adds nothing; ranges that are rejected add nothing and
    // leave a message.
    std::vector<double> untouched = whole;
    status = add(cut,cut,untouched,nullptr);
    require(status == 0 && message[0] == '\0' && untouched == whole,form+": an empty range of ABFs fails or changes the matrix");
    const long rejected[][2] = {{-1,all},{0,all+1},{other,cut}};
    for (const auto& range : rejected) {
      message[0] = '\0';
      status = add(range[0],range[1],untouched,nullptr);
      require(status != 0 && message[0] != '\0' && untouched == whole,form+": a range of ABFs outside the cell is accepted");
    }
    status = sites ? openmx_exx_add_blocks_abfs(state,0,all,-alpha,nullptr,untouched.data(),nullptr,message,sizeof message)
                   : openmx_exx_add_gamma_abfs(state,0,all,-alpha,nullptr,untouched.data(),nullptr,message,sizeof message);
    require(status != 0 && untouched == whole,form+": a range of ABFs without a density matrix is accepted");
    status = sites ? openmx_exx_add_blocks_abfs(nullptr,0,all,-alpha,d.data(),untouched.data(),nullptr,message,sizeof message)
                   : openmx_exx_add_gamma_abfs(nullptr,0,all,-alpha,d.data(),untouched.data(),nullptr,message,sizeof message);
    require(status != 0 && untouched == whole,form+": a range of ABFs without a state is accepted");
  }

  // The half build (section 6.4): the same bits as the C++ interface for the
  // same ranges and for the mirror image, and X + X^T against the whole
  // build made symmetric.  The density matrix is symmetric.
  for (const bool sites : {false,true}) {
    const std::vector<double>& d = sites ? blocks_of_d : density;
    const std::string form = what+(sites ? ", site form, half build" : ", Gamma form, half build");
    const auto add = [&](long first, long last, std::vector<double>& k, const double* from) {
      return sites ? openmx_exx_add_blocks_half_abfs(state,first,last,-alpha,from,k.data(),nullptr,message,sizeof message)
                   : openmx_exx_add_gamma_half_abfs(state,first,last,-alpha,from,k.data(),nullptr,message,sizeof message);
    };
    const auto mirror = [&](double* values) {
      return sites ? openmx_exx_mirror_blocks(state,values,message,sizeof message)
                   : openmx_exx_mirror_gamma(state,values,message,sizeof message);
    };
    std::vector<double> through(d.size(),0.0), direct(d.size(),0.0);
    for (const auto& range : ranges) {
      const int status = add(range[0],range[1],through,d.data());
      require(status == 0 && message[0] == '\0',form+": a range of ABFs through the C interface fails: "+message);
      const std::size_t first = static_cast<std::size_t>(range[0]), last = static_cast<std::size_t>(range[1]);
      if (sites) add_exchange_blocks_half_abfs(system,layout,first,last,-alpha,d,direct);
      else add_exchange_gamma_half_abfs(system,first,last,-alpha,d,direct);
    }
    require(same_values(through,direct),form+": the ranges of ABFs of the two interfaces give different bits");
    require(mirror(through.data()) == 0 && message[0] == '\0',form+": the mirror image through the C interface fails");
    if (sites) mirror_blocks(system,layout,direct);
    else mirror_gamma(system,direct);
    require(same_values(through,direct),form+": the mirror images of the two interfaces differ");

    std::vector<double> whole(d.size(),0.0);
    if (sites) {
      add_exchange_blocks_abfs(system,layout,0,total,-alpha,d,whole);
      symmetrize_blocks(system,layout,whole);
      require(block_asymmetry(system,layout,through) == 0.0,form+": X + X^T is not symmetric to the last bit");
    } else {
      add_exchange_gamma_abfs(system,0,total,-alpha,d,whole);
      const std::size_t n = orbital_total(system);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i+1; j < n; ++j) {
          whole[i*n+j] = whole[j*n+i] = 0.5*(whole[i*n+j]+whole[j*n+i]);
          require(through[i*n+j] == through[j*n+i],form+": X + X^T is not symmetric to the last bit");
        }
    }
    const double scale = largest(whole);
    Worst against(form+": X + X^T through C against the whole build (relative)",1e-12);
    for (std::size_t e = 0; e < whole.size(); ++e) against.add(std::fabs(through[e]-whole[e])/scale);
    against.finish();

    // Rejected calls change nothing and leave a message.
    std::vector<double> untouched = through;
    const long rejected[][2] = {{-1,all},{0,all+1},{other,cut}};
    for (const auto& range : rejected) {
      message[0] = '\0';
      require(add(range[0],range[1],untouched,d.data()) != 0 && message[0] != '\0' && untouched == through,
              form+": a range of ABFs outside the cell is accepted");
    }
    require(add(0,all,untouched,nullptr) != 0 && untouched == through,
            form+": a range of ABFs without a density matrix is accepted");
    require((sites ? openmx_exx_add_blocks_half_abfs(nullptr,0,all,-alpha,d.data(),untouched.data(),nullptr,message,
                                                     sizeof message)
                   : openmx_exx_add_gamma_half_abfs(nullptr,0,all,-alpha,d.data(),untouched.data(),nullptr,message,
                                                    sizeof message)) != 0 &&
            untouched == through,form+": a range of ABFs without a state is accepted");
    message[0] = '\0';
    require(mirror(nullptr) != 0 && message[0] != '\0',form+": a mirror image without an array is accepted");
    require((sites ? openmx_exx_mirror_blocks(nullptr,untouched.data(),nullptr,0)
                   : openmx_exx_mirror_gamma(nullptr,untouched.data(),nullptr,0)) != 0 && untouched == through,
            form+": a mirror image without a state is accepted");
  }
}

// A preparation that "parts" states divide, as the processes of the main
// program do (section 6.7): every state computes its share of the blocks,
// the blocks of every atom are added over the states and the sums are stored
// in every state.  Until that is done for all atoms the builds and the
// derivative are refused; afterwards every state gives the exchange matrix
// of the whole preparation, bit for bit.  The reports add up.
void check_divided_preparation(const OpenMXExxOptions& options, const OpenMXExxGeometry& geometry, const System& system,
                               const std::vector<double>& density, double alpha, int parts)
{
  const std::string what = "preparation in "+std::to_string(parts)+" parts";
  const int atoms = static_cast<int>(system.orbital_count.size());
  const std::size_t n = orbital_total(system);
  struct States {
    std::vector<OpenMXExxState*> list;
    ~States()
    {
      for (OpenMXExxState* state : list) openmx_exx_destroy(state);
    }
  } states;
  char message[200] = "";
  for (int part = 0; part < parts; ++part) {
    states.list.push_back(openmx_exx_create());
    require(states.list.back() != nullptr,"no state of the C interface");
    const int status = openmx_exx_prepare_part(states.list.back(),&options,&geometry,part,parts,message,sizeof message);
    require(status == 0 && message[0] == '\0',what+": a share fails: "+message);
  }

  // The sizes are those of the whole preparation in every share, and the
  // reports add up.
  OpenMXExxReport sum{0,0,0,1.0};
  for (OpenMXExxState* state : states.list) {
    require(openmx_exx_atom_count(state) == atoms && openmx_exx_orbital_count(state) == static_cast<int>(n),
            what+": a share has other sizes");
    for (int a = 0; a < atoms; ++a)
      require(openmx_exx_atom_table_elements(state,a) == table_elements(system,static_cast<std::size_t>(a)) &&
              openmx_exx_atom_abfs(state,a) == static_cast<int>(system.abf_count[static_cast<std::size_t>(a)]),
              what+": a share has blocks of another size");
    OpenMXExxReport report{};
    require(openmx_exx_report(state,&report) == 0,what+": a share has no report");
    sum.systems += report.systems;
    sum.eigen_decompositions += report.eigen_decompositions;
    sum.dropped += report.dropped;
    sum.smallest_pivot = std::min(sum.smallest_pivot,report.smallest_pivot);
  }
  require(sum.systems == system.report.systems && sum.eigen_decompositions == system.report.eigen_decompositions &&
          sum.dropped == system.report.dropped && sum.smallest_pivot == system.report.smallest_pivot,
          what+": the reports of the shares do not add up");

  // Refused until the blocks of every atom are added up: the builds of both
  // forms, their ranges of ABFs, and the derivative.
  const auto refused = [&](OpenMXExxState* state, const std::string& when) {
    std::vector<double> k(n*n,0.0), gradient(3*static_cast<std::size_t>(atoms),0.0);
    message[0] = '\0';
    int status = openmx_exx_add_gamma(state,0,atoms,-alpha,density.data(),k.data(),nullptr,message,sizeof message);
    require(status != 0 && message[0] != '\0',what+": a build "+when+" is accepted");
    status = openmx_exx_add_gamma_abfs(state,0,1,-alpha,density.data(),k.data(),nullptr,message,sizeof message);
    require(status != 0,what+": a build of a range of ABFs "+when+" is accepted");
    status = openmx_exx_add_position_gradient_gamma(state,0,atoms,-alpha,density.data(),gradient.data(),nullptr,message,
                                                    sizeof message);
    require(status != 0 && message[0] != '\0',what+": a derivative "+when+" is accepted");
    require(openmx_exx_prepare_blocks(state,message,sizeof message) == 0,what+": the list of blocks cannot be made "+when);
    std::vector<double> blocks(openmx_exx_block_elements(state),0.0), site_k(blocks.size(),0.0);
    status = openmx_exx_add_blocks(state,0,atoms,-alpha,blocks.data(),site_k.data(),nullptr,message,sizeof message);
    require(status != 0,what+": a build of the site form "+when+" is accepted");
    status = openmx_exx_add_blocks_abfs(state,0,1,-alpha,blocks.data(),site_k.data(),nullptr,message,sizeof message);
    require(status != 0,what+": a build of a range of ABFs of the site form "+when+" is accepted");
    status = openmx_exx_add_position_gradient_blocks(state,0,atoms,-alpha,blocks.data(),gradient.data(),nullptr,message,
                                                     sizeof message);
    require(status != 0,what+": a derivative of the site form "+when+" is accepted");
    require(largest(k) == 0.0 && largest(site_k) == 0.0 && largest(gradient) == 0.0,
            what+": a refused call "+when+" changes its output");
  };
  if (parts > 1)
    for (OpenMXExxState* state : states.list) refused(state,"before the blocks are added up");

  for (int a = 0; a < atoms; ++a) {
    const std::size_t elements = table_elements(system,static_cast<std::size_t>(a));
    std::vector<double> total(elements,0.0), mine(elements);
    for (OpenMXExxState* state : states.list) {
      require(openmx_exx_get_atom_tables(state,a,mine.data()) == 0,what+": the blocks of an atom cannot be read");
      for (std::size_t e = 0; e < elements; ++e) total[e] += mine[e];
    }
    std::vector<double> whole(elements);
    get_tables(system,static_cast<std::size_t>(a),whole);
    require(total == whole,what+": the blocks of the shares do not add up to the whole preparation");
    for (OpenMXExxState* state : states.list)
      require(openmx_exx_set_atom_tables(state,a,total.data()) == 0,what+": the sums of the blocks of an atom cannot be stored");
    if (parts > 1 && a+1 < atoms) refused(states.list.front(),"while the blocks of an atom are missing");
  }

  const std::vector<double> k = exchange_gamma(system,density,alpha);
  for (OpenMXExxState* state : states.list) {
    require(same_values(c_exchange(state,density,alpha),k),
            what+": a state with the sums gives other bits than the whole preparation");
    // The sums stored once more change nothing.
    std::vector<double> mine(table_elements(system,0));
    require(openmx_exx_get_atom_tables(state,0,mine.data()) == 0 && openmx_exx_set_atom_tables(state,0,mine.data()) == 0 &&
            same_values(c_exchange(state,density,alpha),k),what+": the blocks stored a second time change the matrix");
  }
  std::printf("%s: every state gives the exchange matrix of the whole preparation\n",what.c_str());

  // What the entries reject.
  OpenMXExxState* state = states.list.front();
  std::vector<double> values(table_elements(system,0),0.0);
  require(openmx_exx_atom_table_elements(nullptr,0) == 0 && openmx_exx_atom_table_elements(state,-1) == 0 &&
          openmx_exx_atom_table_elements(state,atoms) == 0,what+": blocks of an atom that does not exist have a size");
  require(openmx_exx_get_atom_tables(nullptr,0,values.data()) != 0 && openmx_exx_get_atom_tables(state,atoms,values.data()) != 0 &&
          openmx_exx_get_atom_tables(state,0,nullptr) != 0 && openmx_exx_set_atom_tables(nullptr,0,values.data()) != 0 &&
          openmx_exx_set_atom_tables(state,-1,values.data()) != 0 && openmx_exx_set_atom_tables(state,0,nullptr) != 0,
          what+": blocks of an atom that does not exist, or without an array, are read or stored");
  for (const auto& wrong : {std::array<int,2>{-1,parts},std::array<int,2>{parts,parts},std::array<int,2>{0,0}}) {
    message[0] = '\0';
    const int status = openmx_exx_prepare_part(state,&options,&geometry,wrong[0],wrong[1],message,sizeof message);
    require(status != 0 && message[0] != '\0',what+": a part that does not exist is accepted");
  }
  // The failures left the state as it was.
  require(same_values(c_exchange(state,density,alpha),k),what+": a preparation that fails changes the state");
}

} // namespace

void exchange_system(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  require(fixture.species.size() == 3 && fixture.positions.size() == 3,"the test is written for the three species of the fixture");
  const double alpha = 0.25;
  Numbers random;

  // Coarser grids than the main program's keep the test short; the checks do
  // not depend on the accuracy of the quadrature.
  const int intervals = 300;
  const double maximum = 24.0;
  const int reciprocal_intervals = 240;
  const double threshold = 1.0e-3;
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(maximum,reciprocal_intervals);
  SystemOptions options;
  options.interaction = short_range_kernel(0.11);
  options.v_cutoff = 3.0;

  // 1. One atom far from its images, with all shells of the species.  With
  // the coefficients c^P of the products of the atom (the whole solution of
  // equation (7), without the T blocks and their factor 1/2),
  //   K = -alpha sum_PQ V_PQ c^P D c^Q.
  // A T block that held the whole solution would give four times this K.
  {
    std::vector<Shells> all;
    for (const FixtureSpecies& species : fixture.species) {
      std::vector<std::size_t> every(species.shells.size());
      for (std::size_t i = 0; i < every.size(); ++i) every[i] = i;
      all.push_back(select(species,every));
    }
    const std::vector<SpeciesTable> tables = tables_of(all,intervals,reciprocal,threshold);
    Worst worst("one atom: K against the products without T blocks (relative)",1e-12);
    for (std::size_t s = 0; s < tables.size(); ++s) {
      const SpeciesTable& table = tables[s];
      Structure alone;
      alone.lattice = {std::array<double,3>{40.0,0.0,0.0},std::array<double,3>{0.0,40.0,0.0},std::array<double,3>{0.0,0.0,40.0}};
      alone.positions = {std::array<double,3>{1.0,-2.0,0.5}};
      alone.species = {static_cast<int>(s)};
      const System system = prepare_system(alone,tables,reciprocal,options);
      require(system.c_neighbours[0].size() == 1 && system.v_neighbours[0].size() == 1,
              "the atom is expected to have no neighbours");
      const std::size_t u = table.orbital_count(), np = table.abf_count();
      require(system.orbital_count[0] == u && system.abf_count[0] == np,"wrong counts of the atom");

      // c[(P*u + mu)*u + lambda]
      std::vector<double> c(table.onsite_right_hand_side().begin(),table.onsite_right_hand_side().end());
      solve_metric(table.onsite_metric(),np,c,options.lri);
      const TwoCenterIntegrator interaction(reciprocal,options.interaction,table.maximum_abf_angular(),
                                            table.maximum_abf_angular());
      const std::vector<double> v = two_center_matrix(interaction.pair({0.0,0.0,0.0}),table.abf_transforms(),
                                                      table.abf_transforms());
      require(v.size() == np*np,"wrong size of the interaction between the ABFs of an atom");

      const std::vector<double> d = dense_matrix(u,true,random);
      // left[P] = c^P D, right[P] = sum_Q V_PQ c^Q
      std::vector<double> left(np*u*u,0.0), right(np*u*u,0.0), reference(u*u,0.0);
      for (std::size_t p = 0; p < np; ++p) {
        for (std::size_t i = 0; i < u; ++i)
          for (std::size_t l = 0; l < u; ++l)
            for (std::size_t j = 0; j < u; ++j) left[(p*u+i)*u+j] += c[(p*u+i)*u+l]*d[l*u+j];
        for (std::size_t q = 0; q < np; ++q)
          for (std::size_t e = 0; e < u*u; ++e) right[p*u*u+e] += v[p*np+q]*c[q*u*u+e];
        for (std::size_t i = 0; i < u; ++i)
          for (std::size_t l = 0; l < u; ++l)
            for (std::size_t j = 0; j < u; ++j) reference[i*u+j] -= alpha*left[(p*u+i)*u+l]*right[(p*u+l)*u+j];
      }
      const std::vector<double> k = exchange_gamma(system,d,alpha);
      const double scale = largest(reference);
      require(scale > 1.0e-3,"the exchange matrix of an atom is expected to be of order one");
      for (std::size_t e = 0; e < u*u; ++e) worst.add(std::fabs(k[e]-reference[e])/scale);
      std::printf("%-6s %2zu orbitals, %3zu ABFs, largest element of K %.3e\n",fixture.species[s].name.c_str(),u,np,scale);
    }
    worst.finish();
  }

  // 2. The periodic structure of the fixture with a few shells per species:
  // the cell is smaller than the cutoffs, and the three atoms differ in the
  // numbers of orbitals and ABFs.
  const std::vector<Shells> few{select(fixture.species[0],{0,2}),select(fixture.species[1],{0,1,2}),
                                select(fixture.species[2],{0})};
  Structure structure;
  structure.lattice = fixture.lattice;
  structure.positions = fixture.positions;
  structure.species = fixture.atom_species;
  const std::vector<SpeciesTable> tables = tables_of(few,intervals,reciprocal,threshold);
  const System system = prepare_system(structure,tables,reciprocal,options);
  const std::size_t n = orbital_total(system);
  const std::vector<double> d = dense_matrix(n,true,random);
  // The Gamma form against the definition and against the site form.
  const auto compare_forms = [&](const System& prepared, const std::string& name) {
    const std::vector<double> k = exchange_gamma(prepared,d,alpha);
    const std::vector<double> direct = definition_gamma(prepared,d,alpha);
    const double scale = largest(direct);
    require(scale > 1.0e-3,"the exchange matrix of the "+name+" is expected to be of order one");
    Worst definition(name+": Gamma form against the definition (relative)",1e-12);
    for (std::size_t e = 0; e < n*n; ++e) definition.add(std::fabs(k[e]-direct[e])/scale);
    definition.finish();

    const DensityBlocks blocks = gamma_density(prepared,d);
    const BlockMatrix site = exchange_blocks(prepared,blocks,alpha);
    const std::vector<double> sum = folded(prepared,site);
    Worst form(name+": Gamma form against sum_R K(R) (relative)",1e-12);
    for (std::size_t e = 0; e < n*n; ++e) form.add(std::fabs(k[e]-sum[e])/scale);
    form.finish();
    std::printf("%s, site form: %zu blocks of K, largest element of sum_R K(R) %.3e\n",name.c_str(),
                site.blocks().size(),scale);

    Worst energy(name+": E_x of the two forms (relative)",1e-12);
    energy.relative(exchange_energy(blocks,site),exchange_energy_gamma(d,k));
    energy.finish();
  };
  {
    std::size_t c_sites = 0, v_sites = 0;
    for (std::size_t a = 0; a < system.orbital_count.size(); ++a) {
      c_sites += system.c_neighbours[a].size();
      v_sites += system.v_neighbours[a].size();
    }
    std::printf("periodic structure: %zu orbitals, %zu + %zu + %zu ABFs, %zu C sites, %zu V sites\n",n,
                system.abf_count[0],system.abf_count[1],system.abf_count[2],c_sites,v_sites);
    require(c_sites > 50 && v_sites > 50,"the structure is expected to have many images among its neighbours");
    compare_forms(system,"periodic structure");
  }

  // The ABFs of the centre atoms in pieces.  The pieces one after the other,
  // from the last to the first and on threads of their own give the same
  // bits, and the numbers of one piece agree within rounding.
  //
  // The atoms have 10, 15 and 1 ABFs.  With 2, 4 and 7 threads every loop of
  // the first two atoms is asked for as many calls as there are threads, and
  // the pieces are uneven (10 ABFs in 4 and in 7 pieces, 15 in 2, 4 and 7).
  // With 64 threads the ABFs are the limit: the loops are asked for 10 and
  // for 15 calls.  The third atom is one piece and does not reach the loop of
  // the table.  The number of loops follows the rule at loops_of().  The
  // site form in one array gives the bits of the site form with a map: many
  // pairs of sites share a block of K here.
  {
    require(system.abf_count == std::vector<std::size_t>({10,15,1}),
            "the test is written for atoms with 10, 15 and 1 ABFs");
    require(shared_pairs(system) > 0,"the structure is expected to have pairs of sites that share a block of K");
    const DensityBlocks blocks = gamma_density(system,d);
    const BlockLayout layout = exchange_layout(system);
    const std::vector<double> blocks_of_d = layout_values(layout,blocks);
    Calls shared;
    const auto pieces = [&shared] {
      std::set<int> counts;
      for (const auto& entry : shared.counts) counts.insert(entry.first);
      return counts;
    };
    Worst rounding("periodic structure: pieces of the ABFs against one piece",1e-12);
    for (const bool product : {false,true}) {
      const auto with = [&shared,product](Loop loop, int threads) {
        const OpenMXExxDenseBackend table{product ? &shared_dgemm : nullptr,&shared,loop,threads,nullptr};
        return DenseBackend(&table);
      };
      // One piece with the same products: the limit then measures the order
      // of the sums over the ABFs and nothing else.
      shared.reset();
      const std::vector<double> one = exchange_gamma(system,d,alpha,with(nullptr,0));
      const long products_gamma = shared.products;
      shared.reset();
      const BlockMatrix one_site = exchange_blocks(system,blocks,alpha,with(nullptr,0));
      const long products_site = shared.products;
      const double scale = largest(one);
      // The table with a product of its own runs with seven pieces only.
      for (const int threads : product ? std::vector<int>{7} : std::vector<int>{2,4,7,64}) {
        // The counts in literal numbers.  The comparisons with loops_of()
        // below contain them: they are written out because loops_of()
        // restates the rule of the library, and a mistake that the two share
        // would pass without them.
        const std::set<int> expected = threads < 64 ? std::set<int>{threads} : std::set<int>{10,15};
        shared.reset();
        const std::vector<double> k = exchange_gamma(system,d,alpha,with(&in_order_loop,threads));
        require(pieces() == expected && shared.counts == loops_of(system,threads,Form::gamma),
                "the Gamma form does not hand the expected loops and pieces to the table");
        // Only a table with a product counts products.
        if (product)
          require(products_gamma > 0 && shared.products == products_gamma,
                  "the pieces change the number of products of the Gamma form");
        shared.reset();
        const BlockMatrix site = exchange_blocks(system,blocks,alpha,with(&in_order_loop,threads));
        require(pieces() == expected && shared.counts == loops_of(system,threads,Form::map),
                "the site form does not hand the expected loops and pieces to the table");
        if (product)
          require(products_site > products_gamma && shared.products == products_site,
                  "the pieces change the number of products of the site form");
        shared.reset();
        const std::vector<double> in_array = layout_exchange(system,layout,blocks_of_d,alpha,with(&in_order_loop,threads));
        require(pieces() == expected && shared.counts == loops_of(system,threads,Form::layout),
                "the site form in one array does not hand the expected loops and pieces to the table");
        if (product)
          require(shared.products == products_site,"the pieces change the number of products of the site form in one array");
        require(same_as_blocks(layout,in_array,site),
                "the pieces of the site form in one array differ from those in a map of blocks");
        for (const Loop loop : {&reversed_loop,&threaded_loop}) {
          require(same_values(k,exchange_gamma(system,d,alpha,with(loop,threads))),
                  "the Gamma form depends on the order or the threads of the pieces");
          require(same_blocks(site,exchange_blocks(system,blocks,alpha,with(loop,threads))),
                  "the site form depends on the order or the threads of the pieces");
          require(same_values(in_array,layout_exchange(system,layout,blocks_of_d,alpha,with(loop,threads))),
                  "the site form in one array depends on the order or the threads of the pieces");
        }
        for (std::size_t e = 0; e < n*n; ++e) rounding.add(std::fabs(k[e]-one[e])/scale);
        require(site.blocks().size() == one_site.blocks().size(),"the pieces change the number of blocks of K");
        for (const auto& [key, values] : one_site.blocks()) {
          const std::span<const double> other = site.find(key);
          require(other.size() == values.size(),"the pieces change the blocks of K");
          for (std::size_t e = 0; e < values.size(); ++e) rounding.add(std::fabs(values[e]-other[e])/scale);
        }
      }
    }
    rounding.finish();
  }

  // The same atoms in a large cell.  No neighbour list holds two images of
  // one atom, so the Gamma form has nothing to fold and works with the
  // blocks as they are stored.
  {
    Structure apart = structure;
    apart.lattice = {std::array<double,3>{40.0,0.0,0.0},std::array<double,3>{0.0,40.0,0.0},std::array<double,3>{0.0,0.0,40.0}};
    const System molecule = prepare_system(apart,tables,reciprocal,options);
    std::size_t c_sites = 0, v_sites = 0;
    for (std::size_t a = 0; a < molecule.orbital_count.size(); ++a) {
      std::set<int> c, v;
      for (const Site& site : molecule.c_neighbours[a]) c.insert(site.atom);
      for (const Site& site : molecule.v_neighbours[a]) v.insert(site.atom);
      require(c.size() == molecule.c_neighbours[a].size() && v.size() == molecule.v_neighbours[a].size(),
              "the large cell is expected to have no images among the neighbours");
      c_sites += c.size();
      v_sites += v.size();
    }
    std::printf("atoms in a large cell: %zu C sites, %zu V sites\n",c_sites,v_sites);
    require(c_sites > molecule.orbital_count.size() && v_sites > molecule.orbital_count.size(),
            "the atoms in the large cell are expected to be neighbours of each other");
    compare_forms(molecule,"atoms in a large cell");
  }

  // 3. The C interface, with the same species, grids and structure.
  {
    OpenMXExxOptions c_options;
    openmx_exx_default_options(&c_options);
    require(c_options.kernel == OPENMX_EXX_KERNEL_SHORT_RANGE && c_options.kernel_parameter == 0.11 &&
            c_options.abf_threshold == 1.0e-4 && c_options.v_cutoff == 8.0 && c_options.radial_intervals == 900 &&
            c_options.reciprocal_maximum == 60.0 && c_options.reciprocal_intervals == 900,
            "the default options are not those of the keywords");
    c_options.abf_threshold = threshold;
    c_options.v_cutoff = options.v_cutoff;
    c_options.radial_intervals = intervals;
    c_options.reciprocal_maximum = maximum;
    c_options.reciprocal_intervals = reciprocal_intervals;

    OpenMXExxState* state = openmx_exx_create();
    require(state != nullptr,"no state of the C interface");
    struct Guard {
      OpenMXExxState* state;
      ~Guard() { openmx_exx_destroy(state); }
    } guard{state};
    char message[200] = "";
    const CGeometry geometry(structure,few);
    int prepared = openmx_exx_prepare(state,&c_options,&geometry.geometry,message,sizeof message);
    require(prepared == 0,std::string("the preparation through the C interface fails: ")+message);
    compare_with_c(state,system,d,alpha,"HSE06 kernel");
    compare_ranges_with_c(state,system,d,alpha,"HSE06 kernel");

    // The preparation divided among states (section 6.7).  One part is the
    // whole preparation; three and seven parts leave blocks to add up, and
    // with seven parts the three atoms have fewer blocks of their own than
    // there are parts.
    {
      prepared = openmx_exx_prepare_part(state,&c_options,&geometry.geometry,0,1,message,sizeof message);
      require(prepared == 0,std::string("the preparation in one part fails: ")+message);
      compare_with_c(state,system,d,alpha,"preparation in one part");
      for (const int parts : {1,3,7}) check_divided_preparation(c_options,geometry.geometry,system,d,alpha,parts);
    }

    // A table with a product and a loop, the pieces on threads of their own:
    // the C interface runs the same code on the same numbers as the C++
    // interface in both forms, and the site form agrees within rounding with
    // the numbers of the same products in one piece.
    {
      Calls shared;
      const OpenMXExxDenseBackend table{&shared_dgemm,&shared,&threaded_loop,4,nullptr};
      const OpenMXExxDenseBackend whole{&shared_dgemm,&shared,nullptr,0,nullptr};
      const int atom_count = openmx_exx_atom_count(state);
      std::vector<double> through(n*n,0.0);
      // The calls are statements of their own: the text of a failure holds
      // the message that the call writes, and a compiler may build that text
      // before it makes a call in the same list of arguments.
      int status = openmx_exx_add_gamma(state,0,atom_count,-alpha,d.data(),through.data(),&table,message,
                                        sizeof message);
      require(status == 0 && message[0] == '\0',
              std::string("the Gamma form with pieces through the C interface fails: ")+message);
      require(shared.counts == loops_of(system,4,Form::gamma),
              "the Gamma form through the C interface does not hand the expected loops and pieces to the table");
      require(same_values(through,exchange_gamma(system,d,alpha,DenseBackend(&table))),
              "the pieces of the Gamma form through the C interface differ from those of the C++ interface");

      const BlockLayout layout = exchange_layout(system);
      const std::vector<double> blocks_of_d = layout_values(layout,gamma_density(system,d));
      std::vector<double> site_through(layout.elements(),0.0);
      shared.reset();
      status = openmx_exx_add_blocks(state,0,atom_count,-alpha,blocks_of_d.data(),site_through.data(),&table,message,
                                     sizeof message);
      require(status == 0 && message[0] == '\0',
              std::string("the site form with pieces through the C interface fails: ")+message);
      require(shared.counts == loops_of(system,4,Form::layout),
              "the site form through the C interface does not hand the expected loops and pieces to the table");
      require(same_values(site_through,layout_exchange(system,layout,blocks_of_d,alpha,DenseBackend(&table))),
              "the pieces of the site form through the C interface differ from those of the C++ interface");
      const std::vector<double> one = layout_exchange(system,layout,blocks_of_d,alpha,DenseBackend(&whole));
      const double scale = largest(one);
      Worst rounding("C interface, site form: pieces on threads against one piece (relative)",1e-12);
      for (std::size_t e = 0; e < one.size(); ++e) rounding.add(std::fabs(site_through[e]-one[e])/scale);
      rounding.finish();

      // A product that fails inside a piece, on a thread of its own: the C
      // interface ends the build with a status and a message in both forms.
      // The first atom has 10 ABFs, in four pieces of 2, 3, 2 and 3.  Every
      // piece is the failing one in turn; it ends with its first product
      // while the others make four products for each of their ABFs.  Two
      // products come before this loop, which is the third loop of the build
      // after the rows of D and the turn of G, and the build ends after it.
      Calls failing;
      const OpenMXExxDenseBackend broken{&piece_failing_dgemm,&failing,&threaded_loop,4,nullptr};
      const long sizes[] = {2,3,2,3};
      // What a build that fails has added to its output is not looked at.
      std::vector<double> output(std::max(n*n,layout.elements()),0.0);
      for (int piece = 0; piece < 4; ++piece) {
        const long products = 2+4*(10-sizes[piece])+1;
        failing.failing = piece;
        failing.reset();
        message[0] = '\0';
        status = openmx_exx_add_gamma(state,0,atom_count,-alpha,d.data(),output.data(),&broken,message,sizeof message);
        require(status != 0 && message[0] != '\0',
                "the Gamma form of the C interface accepts a product that fails inside a piece");
        require(failing.products == products && failing.counts == Counts({{4,3}}),
                "the Gamma form of the C interface does not run the other pieces, or goes on after a failed loop");
        failing.reset();
        message[0] = '\0';
        status = openmx_exx_add_blocks(state,0,atom_count,-alpha,blocks_of_d.data(),output.data(),&broken,message,
                                       sizeof message);
        require(status != 0 && message[0] != '\0',
                "the site form of the C interface accepts a product that fails inside a piece");
        require(failing.products == products && failing.counts == Counts({{4,3}}),
                "the site form of the C interface does not run the other pieces, or goes on after a failed loop");
      }
    }

    // Another kernel with the same species: the tables of the species are kept
    // and the blocks are made again.
    c_options.kernel = OPENMX_EXX_KERNEL_TRUNCATED_COULOMB;
    c_options.kernel_parameter = 12.0;
    prepared = openmx_exx_prepare(state,&c_options,&geometry.geometry,message,sizeof message);
    require(prepared == 0,std::string("the second preparation fails: ")+message);
    SystemOptions truncated = options;
    truncated.interaction = truncated_coulomb_kernel(12.0);
    compare_with_c(state,prepare_system(structure,tables,reciprocal,truncated),d,alpha,"truncated Coulomb kernel");

    // Another structure.
    Structure moved = structure;
    moved.positions[1][0] += 0.4;
    moved.positions[2][2] -= 0.3;
    const CGeometry moved_geometry(moved,few);
    prepared = openmx_exx_prepare(state,&c_options,&moved_geometry.geometry,message,sizeof message);
    require(prepared == 0,std::string("the preparation of a moved structure fails: ")+message);
    compare_with_c(state,prepare_system(moved,tables,reciprocal,truncated),d,alpha,"moved atoms");

    // Another threshold: the tables of the species are made again.
    c_options.abf_threshold = 1.0e-2;
    prepared = openmx_exx_prepare(state,&c_options,&moved_geometry.geometry,message,sizeof message);
    require(prepared == 0,std::string("the preparation with another threshold fails: ")+message);
    const std::vector<SpeciesTable> coarse = tables_of(few,intervals,reciprocal,1.0e-2);
    {
      // The call above is told apart from one that kept the old tables only
      // if the threshold changes the number of ABFs.
      std::size_t fine_count = 0, coarse_count = 0;
      for (std::size_t s = 0; s < tables.size(); ++s) {
        fine_count += tables[s].abf_count();
        coarse_count += coarse[s].abf_count();
      }
      std::printf("ABFs of the three species: %zu with the threshold %.0e, %zu with 1e-02\n",fine_count,threshold,coarse_count);
      require(coarse_count < fine_count,"the coarser threshold is expected to drop ABFs");
    }
    compare_with_c(state,prepare_system(moved,coarse,reciprocal,truncated),d,alpha,"another threshold");

    // A failure leaves the state as it was.
    OpenMXExxOptions wrong = c_options;
    wrong.kernel = 99;
    require(openmx_exx_prepare(state,&wrong,&moved_geometry.geometry,message,sizeof message) != 0 && message[0] != '\0',
            "an unknown kernel is accepted");
    wrong = c_options;
    wrong.reciprocal_intervals = 0;
    require(openmx_exx_prepare(state,&wrong,&moved_geometry.geometry,message,sizeof message) != 0 && message[0] != '\0',
            "a reciprocal grid without intervals is accepted");
    compare_with_c(state,prepare_system(moved,coarse,reciprocal,truncated),d,alpha,"after two failures",true);
  }
}

namespace {

// The derivative of E_x with respect to the positions of one structure, in
// both forms, and what the comparisons of exchange_forces() need again.
struct Forces {
  System system;
  BlockLayout layout;
  std::vector<double> density;        // N x N, symmetric
  std::vector<double> blocks;         // in the layout, D_xy(R) = D_yx(-R)^T, different from cell to cell
  std::vector<double> gamma, site;    // dE_x/dtau, [3*atom + axis]
};

// The weights of the blocks of all atoms, one array for every atom in the
// order of get_tables(), the view of them that add_pairs() takes, and the
// arrays as add_block_weights_*_abfs() take them.
struct BlockWeights {
  std::vector<std::vector<double>> values;
  std::vector<std::span<const double>> view;
  std::vector<std::span<double>> arrays;
  explicit BlockWeights(const System& system)
      : values(system.orbital_count.size()), view(values.size()), arrays(values.size())
  {
    for (std::size_t a = 0; a < values.size(); ++a) {
      values[a].assign(table_elements(system,a),0.0);
      view[a] = values[a];
      arrays[a] = values[a];
    }
  }
  BlockWeights(const BlockWeights&) = delete;
  BlockWeights& operator=(const BlockWeights&) = delete;
};

// The weight of the V block of the V neighbour i of centre a in the half form
// of the weights, in units of its weight in the form of the centre atoms: 2
// for a V neighbour that the half build goes through, 1 for the centre
// itself, which enters with half its V block, and 0 for the others.  The
// Gamma form goes through the atoms above the centre and all images of the
// centre, the site form through the cells R > 0 (in the order of the
// components) and through R = 0 with an atom above the centre.
double half_weight_ratio(const System& system, std::size_t a, std::size_t i, bool sites)
{
  const Site& b = system.v_neighbours[a][i];
  const std::size_t atom = static_cast<std::size_t>(b.atom);
  if (!sites) return atom > a ? 2.0 : atom == a ? 1.0 : 0.0;
  if (i == 0) return 1.0;
  for (const int c : b.cell)
    if (c != 0) return c > 0 ? 2.0 : 0.0;
  return atom > a ? 2.0 : 0.0;
}

// Compares the derivatives of both forms with central differences of fourth
// order of E_x: every atom is moved along every axis, and the system is
// prepared again for each of the four structures.  The density matrices stay
// as they are.  The derivative holds for fixed neighbour lists, so the moved
// structures must have the lists of the structure itself.
Forces check_forces(const std::string& name, const Structure& structure, const std::vector<SpeciesTable>& tables,
                    const ReciprocalGrid& reciprocal, const SystemOptions& options, double alpha, Numbers& random,
                    double limit)
{
  Forces result{prepare_system(structure,tables,reciprocal,options),{},{},{},{},{}};
  const System& system = result.system;
  const std::size_t atoms = structure.positions.size(), n = orbital_total(system);
  result.layout = exchange_layout(system);
  const BlockLayout& layout = result.layout;
  result.density = dense_matrix(n,true,random);
  result.blocks.resize(layout.elements());
  for (double& value : result.blocks) value = random.next();
  symmetrize_blocks(system,layout,result.blocks);
  require(block_asymmetry(system,layout,result.blocks) == 0.0,name+": the blocks of D are not symmetric");
  const std::vector<double>& d = result.density;
  const std::vector<double>& blocks = result.blocks;

  const PositionGradient derivative(structure,tables,reciprocal,options,system);
  result.gamma.assign(3*atoms,0.0);
  result.site.assign(3*atoms,0.0);
  for (std::size_t a = 0; a < atoms; ++a) {
    derivative.add_gamma(a,-alpha,d,result.gamma);
    derivative.add_blocks(layout,a,-alpha,blocks,result.site);
  }

  // E_x of both forms for a moved structure.
  const auto energies = [&](std::size_t atom, std::size_t axis, double shift) {
    Structure moved = structure;
    moved.positions[atom][axis] += shift;
    const System other = prepare_system(moved,tables,reciprocal,options);
    require(other.c_neighbours == system.c_neighbours && other.v_neighbours == system.v_neighbours,
            name+": a moved atom changes the neighbour lists");
    require(other.report.systems == system.report.systems &&
            other.report.eigen_decompositions == system.report.eigen_decompositions &&
            other.report.dropped == system.report.dropped,name+": a moved atom changes the way equation (7) is solved");
    return std::array<double,2>{exchange_energy_gamma(d,exchange_gamma(other,d,alpha)),
                                exchange_energy_gamma(blocks,layout_exchange(other,layout,blocks,alpha))};
  };
  const double step = 2.0e-3;
  std::vector<double> numeric_gamma(3*atoms), numeric_site(3*atoms);
  for (std::size_t a = 0; a < atoms; ++a)
    for (std::size_t x = 0; x < 3; ++x) {
      const std::array<double,2> p1 = energies(a,x,step), m1 = energies(a,x,-step);
      const std::array<double,2> p2 = energies(a,x,2.0*step), m2 = energies(a,x,-2.0*step);
      numeric_gamma[3*a+x] = (8.0*(p1[0]-m1[0])-(p2[0]-m2[0]))/(12.0*step);
      numeric_site[3*a+x] = (8.0*(p1[1]-m1[1])-(p2[1]-m2[1]))/(12.0*step);
    }
  const double scale_gamma = largest(numeric_gamma), scale_site = largest(numeric_site);
  require(scale_gamma > 1.0e-5 && scale_site > 1.0e-5,name+": the derivatives are expected to be far from zero");
  Worst gamma(name+": Gamma form against central differences",limit);
  Worst site(name+": site form against central differences",limit);
  for (std::size_t e = 0; e < 3*atoms; ++e) {
    gamma.add(std::fabs(result.gamma[e]-numeric_gamma[e])/scale_gamma);
    site.add(std::fabs(result.site[e]-numeric_site[e])/scale_site);
  }
  gamma.finish();
  site.finish();

  // The route through the pairs of sites (section 6.7): the weights of the
  // blocks of all atoms first, then every pair once.  In one part and in the
  // parts of a division, taken from the last to the first, it gives the
  // derivative of the route through the centre atoms within rounding; it is
  // compared with the central differences as well.
  {
    BlockWeights gamma_weights(system), site_weights(system);
    for (std::size_t a = 0; a < atoms; ++a) {
      add_block_weights(centre_gradient_gamma(system,a,-alpha,d),gamma_weights.values[a]);
      add_block_weights(centre_gradient_blocks(system,layout,a,-alpha,blocks),site_weights.values[a]);
    }
    Worst routes(name+": pairs of sites against centre atoms (relative)",1e-12);
    Worst numeric(name+": pairs of sites against central differences",limit);
    for (const std::size_t parts : {std::size_t{1},std::size_t{2},std::size_t{3},std::size_t{7}}) {
      std::vector<double> through_gamma(3*atoms,0.0), through_site(3*atoms,0.0);
      for (std::size_t part = parts; part-- > 0;) {
        derivative.add_pairs(gamma_weights.view,PrepareShare{part,parts},through_gamma);
        derivative.add_pairs(site_weights.view,PrepareShare{part,parts},through_site);
      }
      for (std::size_t e = 0; e < 3*atoms; ++e) {
        routes.add(std::fabs(through_gamma[e]-result.gamma[e])/scale_gamma);
        routes.add(std::fabs(through_site[e]-result.site[e])/scale_site);
        numeric.add(std::fabs(through_gamma[e]-numeric_gamma[e])/scale_gamma);
        numeric.add(std::fabs(through_site[e]-numeric_site[e])/scale_site);
      }
    }
    routes.finish();
    numeric.finish();

    // The pairs in the pieces of a backend: the vectors of the pairs are
    // added in the order of the pairs, so every number of pieces, every
    // order of the pieces and pieces on threads give the bits of no pieces.
    // For each kind of block the loop is asked for min(threads, pairs) calls;
    // a share with fewer than two pairs of a kind computes them on the
    // calling thread.  The pairs are those of two different atoms.
    {
      std::size_t t_count = 0, v_count = 0;
      for (std::size_t a = 0; a < atoms; ++a) {
        const int atom = static_cast<int>(a);
        for (std::size_t i = 1; i < system.c_neighbours[a].size(); ++i)
          t_count += system.c_neighbours[a][i].atom != atom && pair_computed_from(atom,system.c_neighbours[a][i]);
        for (std::size_t i = 1; i < system.v_neighbours[a].size(); ++i)
          v_count += system.v_neighbours[a][i].atom != atom && pair_computed_from(atom,system.v_neighbours[a][i]);
      }
      long asked = 0;
      for (const std::size_t parts : {std::size_t{1},std::size_t{3}}) {
        std::vector<double> plain(3*atoms,0.0);
        for (std::size_t part = 0; part < parts; ++part)
          derivative.add_pairs(site_weights.view,PrepareShare{part,parts},plain);
        for (const Loop loop : {&in_order_loop,&reversed_loop,&threaded_loop})
          for (const int threads : {2,3,64}) {
            Calls calls;
            const OpenMXExxDenseBackend table{nullptr,&calls,loop,threads,nullptr};
            std::vector<double> pieces(3*atoms,0.0);
            Counts expected;
            for (std::size_t part = 0; part < parts; ++part) {
              derivative.add_pairs(site_weights.view,PrepareShare{part,parts},pieces,DenseBackend(&table));
              for (const std::size_t count : {t_count,v_count}) {
                const std::size_t mine = count > part ? (count-part-1)/parts+1 : 0;
                const std::size_t used = std::min<std::size_t>(static_cast<std::size_t>(threads),mine);
                if (used > 1) ++expected[static_cast<int>(used)];
              }
            }
            require(same_values(pieces,plain),name+": the pairs in pieces give other bits");
            require(calls.counts == expected && calls.products == 0,
                    name+": the pairs in pieces do not ask the loop for the expected calls");
            asked += calls.loops;
          }
      }
      require(t_count < 2 || v_count < 2 || asked > 0,name+": the pairs in pieces never reach the loop");
      std::printf("%s: %zu and %zu pairs of the T and V blocks in pieces, the same bits as without pieces\n",
                  name.c_str(),t_count,v_count);
    }

    // The weights of ranges of ABFs (section 6.7).  The ABFs of the unit cell
    // are counted atom after atom, and a range adds the rows of its ABFs to
    // the arrays of their atoms.  All ABFs in one range give the bits of the
    // centre atoms.  Two ranges that meet anywhere, inside an atom as well,
    // fill different elements and make up the weights of the atoms within
    // rounding; so do ranges of 1, 2, 3, ... ABFs.
    std::vector<std::size_t> first_abf(1,0);
    for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
    const std::size_t total = first_abf.back();
    require(total >= 3,name+": the test of the ranges needs three ABFs or more");
    for (const bool sites : {false,true}) {
      const std::string form = name+(sites ? ", site form" : ", Gamma form");
      const BlockWeights& whole = sites ? site_weights : gamma_weights;
      const auto add = [&](std::size_t first, std::size_t last, BlockWeights& target) {
        if (sites) add_block_weights_blocks_abfs(system,layout,first,last,-alpha,blocks,target.arrays);
        else add_block_weights_gamma_abfs(system,first,last,-alpha,d,target.arrays);
      };
      double scale = 0.0;
      for (std::size_t a = 0; a < atoms; ++a) scale = std::max(scale,largest(whole.values[a]));
      require(scale > 0.0,form+": the weights of the blocks are expected to differ from zero");
      {
        BlockWeights all(system);
        add(0,total,all);
        require(same_values(all.values,whole.values),form+": all ABFs in one range do not give the bits of the atoms");
        // An empty range adds nothing, wherever it is.
        add(0,0,all);
        add(first_abf[1],first_abf[1],all);
        add(total,total,all);
        require(same_values(all.values,whole.values),form+": an empty range of ABFs changes the weights");
      }
      Worst cut(form+": the weights of two ranges of ABFs (relative)",1e-12);
      long cuts = 0, inside = 0;
      const std::size_t stride = std::max<std::size_t>(1,total/7);
      for (std::size_t c = 1; c < total; c += stride) {
        BlockWeights lower(system), upper(system);
        add(0,c,lower);
        add(c,total,upper);
        for (std::size_t a = 0; a < atoms; ++a)
          for (std::size_t e = 0; e < whole.values[a].size(); ++e) {
            require(lower.values[a][e] == 0.0 || upper.values[a][e] == 0.0,
                    form+": two ranges of ABFs give a weight of the same element");
            cut.add(std::fabs(lower.values[a][e]+upper.values[a][e]-whole.values[a][e])/scale);
          }
        ++cuts;
        if (std::find(first_abf.begin(),first_abf.end(),c) == first_abf.end()) ++inside;
      }
      cut.finish();
      require(inside > 0,form+": no range of the test ends inside an atom");
      BlockWeights growing(system);
      for (std::size_t from = 0, width = 1; from < total; from += width, ++width)
        add(from,std::min(total,from+width),growing);
      Worst grown(form+": the weights of ranges of 1, 2, 3, ... ABFs (relative)",1e-12);
      for (std::size_t a = 0; a < atoms; ++a)
        for (std::size_t e = 0; e < whole.values[a].size(); ++e)
          grown.add(std::fabs(growing.values[a][e]-whole.values[a][e])/scale);
      grown.finish();
      // The arrays of the atoms that a range does not touch are not looked
      // at: the ABFs of the last atom with empty arrays for the others.
      {
        BlockWeights last(system);
        std::vector<std::span<double>> only(atoms);
        only[atoms-1] = last.arrays[atoms-1];
        if (sites) add_block_weights_blocks_abfs(system,layout,first_abf[atoms-1],total,-alpha,blocks,only);
        else add_block_weights_gamma_abfs(system,first_abf[atoms-1],total,-alpha,d,only);
        require(same_values(last.values[atoms-1],whole.values[atoms-1]),
                form+": the ABFs of one atom do not give the weights of that atom");
        // What is rejected changes nothing.
        const std::vector<std::vector<double>> before = last.values;
        expect_invalid("a range of ABFs for the weights that ends before it starts",[&] { add(2,1,last); });
        expect_invalid("a range of ABFs for the weights beyond the unit cell",[&] { add(0,total+1,last); });
        expect_invalid("weights of a range of ABFs for fewer atoms",[&] {
          if (sites) add_block_weights_blocks_abfs(system,layout,0,total,-alpha,blocks,
                                                   std::span<const std::span<double>>(last.arrays).first(atoms-1));
          else add_block_weights_gamma_abfs(system,0,total,-alpha,d,
                                            std::span<const std::span<double>>(last.arrays).first(atoms-1));
        });
        expect_invalid("a range of ABFs that touches an atom without an array",[&] {
          if (sites) add_block_weights_blocks_abfs(system,layout,0,total,-alpha,blocks,only);
          else add_block_weights_gamma_abfs(system,0,total,-alpha,d,only);
        });
        expect_invalid("a part of the ABFs of an atom that ends before it starts",[&] {
          (void)(sites ? centre_gradient_blocks(system,layout,0,2,1,-alpha,blocks) : centre_gradient_gamma(system,0,2,1,-alpha,d));
        });
        expect_invalid("a part of the ABFs beyond those of the atom",[&] {
          (void)(sites ? centre_gradient_blocks(system,layout,0,0,system.abf_count[0]+1,-alpha,blocks)
                       : centre_gradient_gamma(system,0,0,system.abf_count[0]+1,-alpha,d));
        });
        require(last.values == before,form+": a rejected range of ABFs changes the weights");
        const CentreGradient none = sites ? centre_gradient_blocks(system,layout,0,1,1,-alpha,blocks)
                                          : centre_gradient_gamma(system,0,1,1,-alpha,d);
        require(none.t.empty() && none.v.size() == system.v_neighbours[0].size() &&
                std::all_of(none.v.begin(),none.v.end(),[](const std::vector<double>& block) { return block.empty(); }),
                form+": an empty part of the ABFs of an atom does not give empty arrays");
      }
      std::printf("%s: the weights of the blocks in ranges of the %zu ABFs, %ld cuts in two, %ld of them inside an atom\n",
                  form.c_str(),total,cuts,inside);
    }

    // The half form of the weights (section 6.7), the E_x of the half build.
    // Its arrays differ from those of the centre atoms: over all ABFs the T
    // blocks of every atom are the same within rounding, every V block that
    // the half build goes through has twice the weight (the centre itself
    // the same), and the other V blocks have none.  Through the pairs it
    // gives the derivative of the centre atoms within rounding, in one range,
    // in two ranges that meet anywhere and in ranges of 1, 2, 3, ... ABFs.
    for (const bool sites : {false,true}) {
      const std::string form = name+(sites ? ", site form, half weights" : ", Gamma form, half weights");
      const BlockWeights& whole = sites ? site_weights : gamma_weights;
      const std::vector<double>& exact = sites ? result.site : result.gamma;
      const double scale = sites ? scale_site : scale_gamma;
      const auto add = [&](std::size_t first, std::size_t last, std::span<const std::span<double>> arrays) {
        if (sites) add_block_weights_blocks_half_abfs(system,layout,first,last,-alpha,blocks,arrays);
        else add_block_weights_gamma_half_abfs(system,first,last,-alpha,d,arrays);
      };
      Worst paired(form+": through the pairs of sites against the centre atoms (relative)",1e-12);
      const auto through_pairs = [&](const BlockWeights& w) {
        for (const std::size_t parts : {std::size_t{1},std::size_t{3}}) {
          std::vector<double> through(3*atoms,0.0);
          for (std::size_t part = 0; part < parts; ++part)
            derivative.add_pairs(w.view,PrepareShare{part,parts},through);
          for (std::size_t e = 0; e < 3*atoms; ++e) paired.add(std::fabs(through[e]-exact[e])/scale);
        }
      };
      BlockWeights all(system);
      add(0,total,all.arrays);
      through_pairs(all);
      double weight_scale = 0.0;
      for (std::size_t a = 0; a < atoms; ++a) weight_scale = std::max(weight_scale,largest(whole.values[a]));
      Worst blocks_worst(form+": T and V blocks against those of the centre atoms (relative)",1e-12);
      long visited = 0, skipped = 0;
      for (std::size_t a = 0; a < atoms; ++a) {
        const std::vector<double>& got = all.values[a];
        const std::vector<double>& want = whole.values[a];
        std::size_t e = 0;
        for (; e < system.t[a].size(); ++e) blocks_worst.add(std::fabs(got[e]-want[e])/weight_scale);
        for (std::size_t i = 0; i < system.v_neighbours[a].size(); ++i) {
          const double ratio = half_weight_ratio(system,a,i,sites);
          (ratio == 0.0 ? skipped : visited) += 1;
          for (const std::size_t end = e+system.v[a][i].size(); e < end; ++e)
            blocks_worst.add(std::fabs(got[e]-ratio*want[e])/weight_scale);
        }
        require(e == got.size(),form+": the arrays of the weights have another layout");
      }
      blocks_worst.finish();
      require(visited > 0,form+": the half form is expected to keep some V blocks");
      for (std::size_t c = 1; c < total; c += std::max<std::size_t>(1,total/7)) {
        BlockWeights cut(system);
        add(0,c,cut.arrays);
        add(c,total,cut.arrays);
        through_pairs(cut);
      }
      BlockWeights growing(system);
      for (std::size_t from = 0, width = 1; from < total; from += width, ++width)
        add(from,std::min(total,from+width),growing.arrays);
      through_pairs(growing);
      paired.finish();
      // A range that is not empty needs the arrays of all atoms; an empty one
      // adds nothing and needs none.
      {
        BlockWeights last(system);
        std::vector<std::span<double>> only(atoms);
        only[atoms-1] = last.arrays[atoms-1];
        expect_invalid("a half range of ABFs without the arrays of the other atoms",
                       [&] { add(first_abf[atoms-1],total,only); });
        add(0,0,only);
        add(total,total,std::vector<std::span<double>>(atoms));
        require(largest(last.values[atoms-1]) == 0.0,form+": a rejected or empty half range changes the weights");
      }
      std::printf("%s: %ld V blocks with weights, %ld without\n",form.c_str(),visited,skipped);
    }
  }

  // Every pair adds a vector to one atom and subtracts it from the other.
  Worst total(name+": sum of the derivatives over the atoms",1e-12);
  for (std::size_t x = 0; x < 3; ++x) {
    double sum_gamma = 0.0, sum_site = 0.0;
    for (std::size_t a = 0; a < atoms; ++a) {
      sum_gamma += result.gamma[3*a+x];
      sum_site += result.site[3*a+x];
    }
    total.add(std::fabs(sum_gamma)/scale_gamma);
    total.add(std::fabs(sum_site)/scale_site);
  }
  total.finish();

  // D_xy(R) = D_xy for every R: the site form gives the derivative of the
  // Gamma form.
  {
    const std::vector<double> same = layout_values(layout,gamma_density(system,d));
    std::vector<double> through(3*atoms,0.0);
    for (std::size_t a = 0; a < atoms; ++a) derivative.add_blocks(layout,a,-alpha,same,through);
    Worst forms(name+": site form with D(R) = D against the Gamma form",1e-12);
    for (std::size_t e = 0; e < 3*atoms; ++e) forms.add(std::fabs(through[e]-result.gamma[e])/scale_gamma);
    forms.finish();
  }
  std::printf("%s: %ld systems of equation (7), %ld by eigen-decomposition, %ld eigenvectors left out; "
              "largest derivative %.3e (Gamma form), %.3e (site form)\n",name.c_str(),system.report.systems,
              system.report.eigen_decompositions,system.report.dropped,scale_gamma,scale_site);
  return result;
}

} // namespace

void exchange_strain(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(24.0,240);
  const std::vector<Shells> one{select(fixture.species.front(),{0})};
  const std::vector<SpeciesTable> tables = tables_of(one,300,reciprocal,1.0e-3);
  Structure structure;
  structure.lattice = {{{4.25,0.9,0.3},{0.15,8.3,0.4},{0.3,0.7,10.9}}};
  structure.positions = {{0.45,0.1,0.2}};
  structure.species = {0};
  const double alpha=0.25, step=2.0e-5;
  const std::array<double,1> density{0.6};
  Worst numerical("EXX strain with periodic images and a moving Coulomb cutoff",2.0e-8);
  for (int variant=0; variant<3; ++variant) {
    const bool scaled=variant==2;
    SystemOptions options;
    options.interaction=variant==0 ? short_range_kernel(0.11) : truncated_coulomb_kernel(5.1);
    options.v_cutoff=8.0;
    const System system=prepare_system(structure,tables,reciprocal,options);
    require(system.c_neighbours.front().size()>1,"the test needs PAO images in its cell derivative");
    BlockWeights weights(system);
    add_block_weights_gamma_half_abfs(system,0,system.abf_count[0],-alpha,density,weights.arrays);
    const PositionGradient derivative(structure,tables,reciprocal,options,system);
    std::array<double,3> force{};
    std::array<double,9> strain{};
    for (std::size_t part=0; part<3; ++part)
      derivative.add_pairs(weights.view,{part,3},force,DenseBackend{},strain,scaled);
    require(force==std::array<double,3>{},"periodic images must exert no force on their sole atom");

    const auto energy = [&](std::size_t i, std::size_t j, double delta) {
      Structure moved=structure;
      for (auto& row : moved.lattice) row[i]+=delta*row[j];
      for (auto& row : moved.positions) row[i]+=delta*row[j];
      SystemOptions displaced_options=options;
      if (scaled && i==j)
        displaced_options.interaction=truncated_coulomb_kernel(options.interaction.parameter*std::cbrt(1.0+delta));
      const System displaced=prepare_system(moved,tables,reciprocal,displaced_options);
      require(displaced.c_neighbours==system.c_neighbours && displaced.v_neighbours==system.v_neighbours,
              "the strain difference changed the lists of neighbours");
      return exchange_energy_gamma(density,exchange_gamma(displaced,density,alpha));
    };
    for (const auto [i,j] : {std::array<std::size_t,2>{0,0},std::array<std::size_t,2>{1,0}}) {
      const double numerical_value=(energy(i,j,step)-energy(i,j,-step))/(2.0*step);
      require(std::fabs(numerical_value)>1.0e-6,"the strain test must have a measurable derivative");
      numerical.absolute(strain[3*i+j],numerical_value);
    }
  }
  numerical.finish();
}

void exchange_forces(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  require(fixture.species.size() == 3 && fixture.positions.size() == 3,"the test is written for the three species of the fixture");
  const double alpha = 0.25;
  Numbers random;

  // The grids and the shells of exx-exchange-system: the derivative is that
  // of the discretized integrals, so the checks do not depend on the accuracy
  // of the quadrature.  With the cutoff 3.1 no pair of sites of the fixture
  // is closer than 0.019 bohr to a cutoff of the two lists, so the moves of
  // the central differences (up to 0.004 bohr) leave the lists as they are;
  // with 3.0 a pair of sites lies 0.0005 bohr outside the V list.
  const int intervals = 300;
  const double maximum = 24.0;
  const int reciprocal_intervals = 240;
  const double threshold = 1.0e-3;
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(maximum,reciprocal_intervals);
  SystemOptions options;
  options.interaction = short_range_kernel(0.11);
  options.v_cutoff = 3.1;
  const std::vector<Shells> few{select(fixture.species[0],{0,2}),select(fixture.species[1],{0,1,2}),
                                select(fixture.species[2],{0})};
  const std::vector<SpeciesTable> tables = tables_of(few,intervals,reciprocal,threshold);
  Structure structure;
  structure.lattice = fixture.lattice;
  structure.positions = fixture.positions;
  structure.species = fixture.atom_species;
  const std::size_t atoms = structure.positions.size();

  // 1. The periodic structure: the cell is smaller than the cutoffs, so the
  // lists hold many images of every atom, the atom itself among them.
  const Forces periodic = check_forces("periodic structure",structure,tables,reciprocal,options,alpha,random,1e-6);
  require(periodic.system.report.eigen_decompositions == 0,"the periodic structure is expected to need no eigen-decomposition");

  // 2. The same atoms in a large cell, with the truncated Coulomb kernel.
  Structure apart = structure;
  apart.lattice = {std::array<double,3>{40.0,0.0,0.0},std::array<double,3>{0.0,40.0,0.0},std::array<double,3>{0.0,0.0,40.0}};
  SystemOptions truncated = options;
  truncated.interaction = truncated_coulomb_kernel(12.0);
  check_forces("atoms in a large cell",apart,tables,reciprocal,truncated,alpha,random,1e-6);

  // 3. Equation (7) by eigen-decomposition: with all eigenvectors, and with
  // the eigenvectors below a threshold left out.
  {
    SystemOptions decomposed = options;
    decomposed.lri = LriOptions{0.999999,0.0};
    const Forces all = check_forces("eigen-decomposition",apart,tables,reciprocal,decomposed,alpha,random,1e-6);
    // The species with a single ABF has a system of one equation, whose pivot
    // is one.
    require(tables[2].abf_count() == 1 && all.system.report.eigen_decompositions == all.system.report.systems-1 &&
            all.system.report.dropped == 0,
            "all systems but one are expected to be solved by eigen-decomposition with all eigenvectors");
    // The scaled metrics of this structure have four eigenvalues between
    // 0.037 and 0.045, one of an atom and three of two pairs, and none
    // between 0.045 and 0.082.
    decomposed.lri = LriOptions{0.999999,6.0e-2};
    const Forces some = check_forces("eigenvectors left out",apart,tables,reciprocal,decomposed,alpha,random,1e-6);
    require(some.system.report.dropped == 4,"the threshold is expected to leave four eigenvectors out");
  }

  // 4. A table of matrix products gives the same derivative up to rounding.
  {
    const PositionGradient derivative(structure,tables,reciprocal,options,periodic.system);
    long calls = 0;
    const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0,nullptr};
    const DenseBackend products(&table);
    std::vector<double> gamma(3*atoms,0.0), site(3*atoms,0.0);
    for (std::size_t a = 0; a < atoms; ++a) {
      derivative.add_gamma(a,-alpha,periodic.density,gamma,products);
      derivative.add_blocks(periodic.layout,a,-alpha,periodic.blocks,site,products);
    }
    require(calls > 0,"the table of matrix products is not used");
    Worst worst("periodic structure: a table of matrix products (relative)",1e-12);
    for (std::size_t e = 0; e < 3*atoms; ++e) {
      worst.add(std::fabs(gamma[e]-periodic.gamma[e])/largest(periodic.gamma));
      worst.add(std::fabs(site[e]-periodic.site[e])/largest(periodic.site));
    }
    worst.finish();

    // Arguments that do not fit the system.
    std::vector<double> scratch(3*atoms,0.0);
    const CentreGradient weights = centre_gradient_gamma(periodic.system,0,-alpha,periodic.density);
    expect_invalid("a centre atom outside the cell",[&] { derivative.add(atoms,weights,scratch); });
    expect_invalid("a derivative of the wrong size",
                   [&] { derivative.add(0,weights,std::span<double>(scratch).first(3*atoms-1)); });
    expect_invalid("weights of another atom",[&] { derivative.add(1,weights,scratch); });
    require(largest(scratch) == 0.0,"a rejected call changes the derivative");

    // The route through the pairs.  The weights of two calls add up.
    BlockWeights all(periodic.system);
    for (std::size_t a = 0; a < atoms; ++a) {
      const CentreGradient half = centre_gradient_gamma(periodic.system,a,-0.5*alpha,periodic.density);
      add_block_weights(half,all.values[a]);
      add_block_weights(half,all.values[a]);
    }
    {
      std::vector<double> twice(3*atoms,0.0);
      derivative.add_pairs(all.view,PrepareShare{},twice);
      Worst sum("periodic structure: the weights of two calls add up (relative)",1e-12);
      for (std::size_t e = 0; e < 3*atoms; ++e) sum.add(std::fabs(twice[e]-periodic.gamma[e])/largest(periodic.gamma));
      sum.finish();
    }
    std::vector<double> short_array(all.values[0].size()-1,0.0);
    expect_invalid("an array of the wrong size for the weights of an atom",[&] { add_block_weights(weights,short_array); });
    expect_invalid("weights of the pairs for fewer atoms",
                   [&] { derivative.add_pairs(std::span<const std::span<const double>>(all.view).first(atoms-1),PrepareShare{},scratch); });
    expect_invalid("a share of the pairs that is not one of the parts",
                   [&] { derivative.add_pairs(all.view,PrepareShare{3,3},scratch); });
    expect_invalid("a derivative of the wrong size for the pairs",
                   [&] { derivative.add_pairs(all.view,PrepareShare{},std::span<double>(scratch).first(3*atoms-1)); });
    {
      std::vector<std::span<const double>> wrong = all.view;
      wrong[1] = wrong[1].first(wrong[1].size()-1);
      expect_invalid("weights that do not have the layout of their atom",
                     [&] { derivative.add_pairs(wrong,PrepareShare{},scratch); });
    }
    require(largest(scratch) == 0.0,"a rejected call of the pairs changes the derivative");
    Structure fewer = structure;
    fewer.positions.pop_back();
    fewer.species.pop_back();
    expect_invalid("a system of another structure",
                   [&] { const PositionGradient wrong(fewer,tables,reciprocal,options,periodic.system); });
    Structure other = structure;
    std::swap(other.species[0],other.species[1]);
    expect_invalid("a system of other species",
                   [&] { const PositionGradient wrong(other,tables,reciprocal,options,periodic.system); });
  }

  // 5. The C interface: the same code on the same numbers.
  {
    OpenMXExxOptions c_options;
    openmx_exx_default_options(&c_options);
    c_options.abf_threshold = threshold;
    c_options.v_cutoff = options.v_cutoff;
    c_options.radial_intervals = intervals;
    c_options.reciprocal_maximum = maximum;
    c_options.reciprocal_intervals = reciprocal_intervals;
    OpenMXExxState* state = openmx_exx_create();
    require(state != nullptr,"no state of the C interface");
    struct Guard {
      OpenMXExxState* state;
      ~Guard() { openmx_exx_destroy(state); }
    } guard{state};
    char message[200] = "untouched";
    std::vector<double> gamma(3*atoms,0.0), site(3*atoms,0.0);
    require(openmx_exx_add_position_gradient_gamma(state,0,1,-alpha,periodic.density.data(),gamma.data(),nullptr,message,
                                                   sizeof message) != 0 && message[0] != '\0',
            "a derivative without a prepared structure is accepted");
    const CGeometry geometry(structure,few);
    // The calls are statements of their own where the text of a failure
    // holds their message: a compiler may build that text before it makes a
    // call in the same list of arguments.
    const int prepared = openmx_exx_prepare(state,&c_options,&geometry.geometry,message,sizeof message);
    require(prepared == 0,std::string("the preparation through the C interface fails: ")+message);
    require(openmx_exx_add_position_gradient_blocks(state,0,1,-alpha,periodic.blocks.data(),site.data(),nullptr,message,
                                                    sizeof message) != 0 && message[0] != '\0',
            "a derivative of the site form without a list of blocks is accepted");
    require(openmx_exx_prepare_blocks(state,message,sizeof message) == 0 &&
            openmx_exx_block_elements(state) == periodic.layout.elements(),"the list of blocks cannot be made");

    // The centre atoms in two calls, as two processes would share them.
    const int count = static_cast<int>(atoms);
    bool done =
        openmx_exx_add_position_gradient_gamma(state,0,1,-alpha,periodic.density.data(),gamma.data(),nullptr,message,
                                               sizeof message) == 0 && message[0] == '\0' &&
        openmx_exx_add_position_gradient_gamma(state,1,count,-alpha,periodic.density.data(),gamma.data(),nullptr,
                                               message,sizeof message) == 0;
    require(done,std::string("the derivative of the Gamma form through the C interface fails: ")+message);
    done =
        openmx_exx_add_position_gradient_blocks(state,0,2,-alpha,periodic.blocks.data(),site.data(),nullptr,message,
                                                sizeof message) == 0 && message[0] == '\0' &&
        openmx_exx_add_position_gradient_blocks(state,2,count,-alpha,periodic.blocks.data(),site.data(),nullptr,
                                                message,sizeof message) == 0;
    require(done,std::string("the derivative of the site form through the C interface fails: ")+message);
    Worst same("C interface against the C++ interface (relative)",1e-13);
    for (std::size_t e = 0; e < 3*atoms; ++e) {
      same.add(std::fabs(gamma[e]-periodic.gamma[e])/largest(periodic.gamma));
      same.add(std::fabs(site[e]-periodic.site[e])/largest(periodic.site));
    }
    same.finish();

    // An empty range adds nothing; rejected calls and a product that fails
    // leave the output as it was.
    const std::vector<double> before_gamma = gamma, before_site = site;
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr,nullptr,0,nullptr};
    require(openmx_exx_add_position_gradient_gamma(state,1,1,-alpha,periodic.density.data(),gamma.data(),nullptr,message,
                                                   sizeof message) == 0 && gamma == before_gamma,
            "an empty range of centre atoms changes the derivative");
    require(openmx_exx_add_position_gradient_gamma(state,-1,count,-alpha,periodic.density.data(),gamma.data(),nullptr,
                                                   message,sizeof message) != 0 &&
            openmx_exx_add_position_gradient_gamma(state,0,count+1,-alpha,periodic.density.data(),gamma.data(),nullptr,
                                                   message,sizeof message) != 0 &&
            openmx_exx_add_position_gradient_gamma(state,0,count,-alpha,nullptr,gamma.data(),nullptr,message,
                                                   sizeof message) != 0 &&
            openmx_exx_add_position_gradient_gamma(state,0,count,-alpha,periodic.density.data(),nullptr,nullptr,message,
                                                   sizeof message) != 0 &&
            openmx_exx_add_position_gradient_gamma(state,0,count,-alpha,periodic.density.data(),gamma.data(),&failing,
                                                   message,sizeof message) != 0 && message[0] != '\0' &&
            gamma == before_gamma,"a rejected derivative of the Gamma form changes the output");
    require(openmx_exx_add_position_gradient_blocks(state,-1,count,-alpha,periodic.blocks.data(),site.data(),nullptr,
                                                    message,sizeof message) != 0 &&
            openmx_exx_add_position_gradient_blocks(state,0,count+1,-alpha,periodic.blocks.data(),site.data(),nullptr,
                                                    message,sizeof message) != 0 &&
            openmx_exx_add_position_gradient_blocks(state,0,count,-alpha,nullptr,site.data(),nullptr,message,
                                                    sizeof message) != 0 &&
            openmx_exx_add_position_gradient_blocks(state,0,count,-alpha,periodic.blocks.data(),nullptr,nullptr,message,
                                                    sizeof message) != 0 &&
            openmx_exx_add_position_gradient_blocks(state,0,count,-alpha,periodic.blocks.data(),site.data(),&failing,
                                                    message,sizeof message) != 0 && message[0] != '\0' &&
            site == before_site,"a rejected derivative of the site form changes the output");

    // The route through the pairs (section 6.7): the weights of the ABFs in
    // two calls whose ranges meet inside an atom, as two processes would
    // compute them, and the pairs in three parts.
    std::vector<long> first_abf(1,0);
    for (const std::size_t abfs : periodic.system.abf_count) first_abf.push_back(first_abf.back()+static_cast<long>(abfs));
    const long total = first_abf.back();
    std::size_t widest = 0;
    for (std::size_t a = 0; a < atoms; ++a)
      if (periodic.system.abf_count[a] > periodic.system.abf_count[widest]) widest = a;
    require(periodic.system.abf_count[widest] >= 2,"the test needs an atom with two ABFs or more");
    const long cut = first_abf[widest]+static_cast<long>(periodic.system.abf_count[widest]/2);
    for (const bool sites : {false,true}) {
      const std::string form = sites ? "site form" : "Gamma form";
      const std::vector<double>& d = sites ? periodic.blocks : periodic.density;
      const std::vector<double>& reference = sites ? periodic.site : periodic.gamma;
      std::vector<std::vector<double>> values(atoms);
      std::vector<double*> arrays(atoms);
      for (std::size_t a = 0; a < atoms; ++a) {
        values[a].assign(openmx_exx_atom_table_elements(state,static_cast<int>(a)),0.0);
        require(values[a].size() == table_elements(periodic.system,a),"wrong size of the weights of an atom");
        arrays[a] = values[a].data();
      }
      const auto add_weights = [&](long first, long last, const double* density, double* const* target,
                                   const OpenMXExxDenseBackend* table) {
        return sites ? openmx_exx_add_block_weights_blocks_abfs(state,first,last,-alpha,density,target,table,message,
                                                                sizeof message)
                     : openmx_exx_add_block_weights_gamma_abfs(state,first,last,-alpha,density,target,table,message,
                                                               sizeof message);
      };
      int status = add_weights(0,cut,d.data(),arrays.data(),nullptr);
      require(status == 0 && message[0] == '\0',form+": the weights through the C interface fail: "+message);
      status = add_weights(cut,total,d.data(),arrays.data(),nullptr);
      require(status == 0,form+": the weights of the other ABFs through the C interface fail: "+message);
      {
        // The C++ interface gives the same bits for the same ranges, and
        // the weights of the centre atoms within rounding.
        BlockWeights direct(periodic.system), whole(periodic.system);
        for (const std::array<long,2>& range : {std::array<long,2>{0,cut},std::array<long,2>{cut,total}}) {
          const std::size_t first = static_cast<std::size_t>(range[0]), last = static_cast<std::size_t>(range[1]);
          if (sites) add_block_weights_blocks_abfs(periodic.system,periodic.layout,first,last,-alpha,d,direct.arrays);
          else add_block_weights_gamma_abfs(periodic.system,first,last,-alpha,d,direct.arrays);
        }
        require(same_values(values,direct.values),form+": the weights of the two interfaces give different bits");
        double scale = 0.0;
        for (std::size_t a = 0; a < atoms; ++a) {
          add_block_weights(sites ? centre_gradient_blocks(periodic.system,periodic.layout,a,-alpha,d)
                                  : centre_gradient_gamma(periodic.system,a,-alpha,d),whole.values[a]);
          scale = std::max(scale,largest(whole.values[a]));
        }
        Worst atoms_against_ranges("C interface, "+form+": weights of two ranges of ABFs against centre atoms (relative)",
                                   1e-12);
        for (std::size_t a = 0; a < atoms; ++a)
          for (std::size_t e = 0; e < values[a].size(); ++e)
            atoms_against_ranges.add(std::fabs(values[a][e]-whole.values[a][e])/scale);
        atoms_against_ranges.finish();
      }
      const std::vector<const double*> weights(arrays.begin(),arrays.end());
      std::vector<double> through(3*atoms,0.0);
      for (int part = 0; part < 3; ++part) {
        status = openmx_exx_add_position_gradient_pairs(state,weights.data(),part,3,through.data(),message,sizeof message);
        require(status == 0 && message[0] == '\0',form+": a share of the pairs through the C interface fails: "+message);
      }
      Worst pairs("C interface, "+form+": pairs of sites against centre atoms (relative)",1e-12);
      for (std::size_t e = 0; e < 3*atoms; ++e) pairs.add(std::fabs(through[e]-reference[e])/largest(reference));
      pairs.finish();

      // What the entries reject; the outputs stay as they are.
      const std::vector<double> before = through;
      const std::vector<std::vector<double>> before_values = values;
      std::vector<const double*> missing = weights;
      missing[atoms-1] = nullptr;
      require(openmx_exx_add_position_gradient_pairs(state,weights.data(),3,3,through.data(),message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(state,weights.data(),-1,3,through.data(),message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(state,weights.data(),0,0,through.data(),message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(state,nullptr,0,1,through.data(),message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(state,weights.data(),0,1,nullptr,message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(state,missing.data(),0,1,through.data(),message,sizeof message) != 0 &&
              openmx_exx_add_position_gradient_pairs(nullptr,weights.data(),0,1,through.data(),message,sizeof message) != 0 &&
              message[0] != '\0' && through == before,form+": a rejected call of the pairs changes the derivative");
      std::vector<double*> lacking = arrays;
      lacking[0] = nullptr;
      require(add_weights(-1,total,d.data(),arrays.data(),nullptr) != 0 &&
              add_weights(0,total+1,d.data(),arrays.data(),nullptr) != 0 &&
              add_weights(2,1,d.data(),arrays.data(),nullptr) != 0 &&
              add_weights(0,total,nullptr,arrays.data(),nullptr) != 0 &&
              add_weights(0,total,d.data(),nullptr,nullptr) != 0 &&
              add_weights(0,total,d.data(),lacking.data(),nullptr) != 0 && message[0] != '\0' &&
              values == before_values,form+": rejected weights change the arrays");
      {
        // A product that fails; the call may have added the weights of some
        // of its atoms, so it gets arrays of its own.
        std::vector<std::vector<double>> scratch = values;
        std::vector<double*> targets(atoms);
        for (std::size_t a = 0; a < atoms; ++a) targets[a] = scratch[a].data();
        require(add_weights(0,total,d.data(),targets.data(),&failing) != 0 && message[0] != '\0',
                form+": the weights with a product that fails are accepted");
      }
      // An empty range adds nothing, and the arrays of the atoms that a range
      // does not touch are not read.
      std::vector<double*> only(atoms,nullptr);
      only[1] = arrays[1];
      status = add_weights(cut,cut,d.data(),arrays.data(),nullptr);
      require(status == 0 && values == before_values,form+": an empty range of ABFs changes the weights");
      status = add_weights(first_abf[1],first_abf[2],d.data(),only.data(),nullptr);
      require(status == 0 && values[0] == before_values[0] && values[1] != before_values[1],
              form+": the weights of the ABFs of one atom need the arrays of the other atoms");
    }
  }
}

namespace {

// ---- the device (section 11 of the guide) ------------------------------------

// One form of the build for the device tests: build(first, last, k, backend)
// adds the contributions of the centre atoms first to last-1 to k, an array
// of "size" elements, and returns whether the device computed them.
struct DeviceForm {
  std::string name;
  std::size_t size = 0;
  std::function<bool(std::size_t, std::size_t, std::span<double>, const DenseBackend&)> build;
};

// The calls of the device that are made to fail in turn: all, or for a large
// system the first and the last call of every kind and about forty more,
// evenly spaced.
std::vector<long> failures(const std::vector<FakeDevice::Kind>& sequence, bool every)
{
  std::set<long> chosen;
  const long total = static_cast<long>(sequence.size());
  if (every || total <= 64) {
    for (long i = 0; i < total; ++i) chosen.insert(i);
  } else {
    std::map<FakeDevice::Kind,std::pair<long,long>> range;
    for (long i = 0; i < total; ++i) {
      const auto found = range.find(sequence[static_cast<std::size_t>(i)]);
      if (found == range.end()) range.emplace(sequence[static_cast<std::size_t>(i)],std::pair<long,long>(i,i));
      else found->second.second = i;
    }
    for (const auto& [kind, ends] : range) {
      chosen.insert(ends.first);
      chosen.insert(ends.second);
    }
    for (long i = 0; i < total; i += std::max(1L,total/40)) chosen.insert(i);
  }
  return std::vector<long>(chosen.begin(),chosen.end());
}

// The device of one form against the dense backend on the host.  The table
// has a product and a loop of four pieces, which the dense backend uses when
// the device does not compute the call.  k holds numbers of the size of the
// contributions on entry: the build adds to it.
void check_device(const System& system, const DeviceForm& form, Numbers& random, bool every)
{
  const std::size_t atoms = system.orbital_count.size();
  FakeDevice device;
  Calls calls;
  const OpenMXExxDenseBackend host_table{&shared_dgemm,&calls,&in_order_loop,4,nullptr};
  const OpenMXExxDenseBackend device_table{&shared_dgemm,&calls,&in_order_loop,4,&device.table()};
  const DenseBackend host(&host_table), with_device(&device_table);
  const std::string& name = form.name;

  std::vector<double> alone(form.size,0.0);
  require(!form.build(0,atoms,alone,host),name+": the dense backend reports a device");
  const double scale = largest(alone);
  require(scale > 1.0e-6,name+": the contributions are expected to be of order one");
  std::vector<double> start(form.size);
  for (double& value : start) value = scale*random.next();
  std::vector<double> reference = start;
  form.build(0,atoms,reference,host);

  // A call with the device; the record of the device must end clean.
  const auto run = [&](const DenseBackend& backend, std::size_t first, std::size_t last, std::vector<double>& k) {
    device.reset();
    calls.reset();
    const bool computed = form.build(first,last,k,backend);
    const std::string ending = device.ending();
    require(ending.empty(),name+": "+ending);
    return computed;
  };

  // The device computes the call: the product and the loop of the table are
  // not used, accept is told the bytes that the allocations take and the
  // multiplications of all products, and k agrees with the host within
  // rounding.
  std::vector<double> k = start;
  require(run(with_device,0,atoms,k),name+": the device does not compute the call");
  require(calls.products == 0 && calls.loops == 0,name+": the product or the loop of the table is used with a device");
  require(device.accepted == 1 && device.count[FakeDevice::allocate_call] > 0 &&
          device.count[FakeDevice::allocate_call] == device.count[FakeDevice::release_call] &&
          device.count[FakeDevice::download_call] == 1,name+": wrong calls of the device");
  require(device.bytes == device.allocated,name+": accept is told other bytes than the allocations take");
  require(device.multiplications == device.all_products,
          name+": accept is told other multiplications than the products of the device have in all");
  Worst worst(name+": device against the dense backend (relative)",1e-12);
  for (std::size_t e = 0; e < k.size(); ++e) worst.add(std::fabs(k[e]-reference[e])/scale);
  worst.finish();
  const std::vector<FakeDevice::Kind> sequence = device.sequence;
  std::printf("%s: %ld calls of the device, %.0f bytes, %.3e multiplications in all, the largest product %.3e\n",
              name.c_str(),static_cast<long>(sequence.size()),static_cast<double>(device.bytes),device.multiplications,
              device.largest_product);

  {
    const double largest_work = device.largest_product;
    require(device.all_products > largest_work,name+": expected several products");
    std::vector<double> exact = start;
    require(run(DenseBackend(&device_table,nullptr,largest_work),0,atoms,exact) && same_values(exact,k),
            name+": a product at the GPU threshold is rejected");
    const double higher = std::nextafter(largest_work,device.all_products);
    std::vector<double> small = start;
    require(!run(DenseBackend(&device_table,nullptr,higher),0,atoms,small) && device.accepted == 0 &&
            same_values(small,reference),name+": small products do not stay on the host");
  }

  // The same bits again.  The centre atoms in two calls give the bits of one
  // call: every centre makes its arrays anew, and only k carries over.  With
  // the products of this device the same arguments give the same bits.
  {
    std::vector<double> again = start;
    require(run(with_device,0,atoms,again) && same_values(again,k),name+": the device gives other bits the second time");
    for (std::size_t middle = 1; middle < atoms; ++middle) {
      std::vector<double> split = start;
      require(run(with_device,0,middle,split) && run(with_device,middle,atoms,split) && same_values(split,k),
              name+": the centre atoms in two calls give other bits than in one call");
    }
  }

  // A call without centre atoms does not ask the device and adds nothing.
  {
    std::vector<double> none = start;
    require(!run(with_device,atoms,atoms,none) && device.accepted == 0 && same_values(none,start),
            name+": a call without centre atoms uses the device or changes k");
  }

  // The device declines: the dense backend computes the call with the
  // product and the loop of the table.
  {
    device.answer = 0;
    std::vector<double> declined = start;
    require(!run(with_device,0,atoms,declined) && device.accepted == 1 && device.calls == 0 &&
            device.count[FakeDevice::release_call] == 0 && calls.products > 0 && calls.loops > 0 &&
            same_values(declined,reference),name+": a device that declines is used, or the host gives other bits");
    device.answer = 1;
  }

  // A table that lacks a function is not asked.
  for (int member = 0; member < 8; ++member) {
    OpenMXExxDevice partial = device.table();
    switch (member) {
      case 0: partial.accept = nullptr; break;
      case 1: partial.allocate = nullptr; break;
      case 2: partial.release = nullptr; break;
      case 3: partial.upload = nullptr; break;
      case 4: partial.download = nullptr; break;
      case 5: partial.dgemm = nullptr; break;
      case 6: partial.move = nullptr; break;
      default: partial.finish = nullptr; break;
    }
    const OpenMXExxDenseBackend table{&shared_dgemm,&calls,&in_order_loop,4,&partial};
    std::vector<double> without = start;
    require(!run(DenseBackend(&table),0,atoms,without) && device.accepted == 0 && same_values(without,reference),
            name+": a device without one of its functions is used");
  }

  // Every call of the device fails in turn: the library stops using it,
  // finishes the work it gave, releases every block, and the dense backend
  // computes the call from the unchanged k.  run() checks the first three.
  const std::vector<long> chosen = failures(sequence,every);
  for (const long index : chosen) {
    device.failing = index;
    std::vector<double> failed = start;
    require(!run(with_device,0,atoms,failed) && device.failed && device.calls > index,
            name+": a failure of the device is not noticed");
    require(calls.products > 0 && same_values(failed,reference),
            name+": after a failure of the device the host does not give its own bits");
  }
  device.failing = -1;
  std::printf("%s: %zu failures of the device, each one followed by the bits of the host\n",name.c_str(),chosen.size());
}

// The estimate of the ABFs first <= P < last by which divide_abfs() divides
// them: n per_abf + per_part of build_costs() over the atoms whose ABFs the
// range holds, n of them.
double range_estimate(const System& system, const BuildCosts& cost, std::size_t first, std::size_t last)
{
  double estimate = 0.0;
  std::size_t begin = 0;
  for (std::size_t a = 0; a < system.abf_count.size(); ++a) {
    const std::size_t end = begin+system.abf_count[a], from = std::max(begin,first), to = std::min(end,last);
    if (from < to) estimate += static_cast<double>(to-from)*cost.per_abf[a]+cost.per_part[a];
    begin = end;
  }
  return estimate;
}

// The estimates of build_costs() and the division of the ABFs among
// processes (divide_abfs()), in both forms and for the whole and the half
// build.  The estimates are positive and the same on every call, and the
// time of an ABF is no less than its multiplications.  For one to six
// processes and for more processes than ABFs the ranges are contiguous,
// cover the ABFs in order and are the same on every call; their largest
// estimate is the smallest of all divisions into as many contiguous ranges,
// which a search of all of them by dynamic programming finds.
void check_division(const System& system, const std::string& name)
{
  const std::size_t atoms = system.abf_count.size(), most = 6;
  std::vector<std::size_t> atom_of;
  for (std::size_t a = 0; a < atoms; ++a) atom_of.insert(atom_of.end(),system.abf_count[a],a);
  const std::size_t total = atom_of.size();
  for (const bool sites : {false,true}) {
    for (const bool half : {false,true}) {
      const std::string form = name+(sites ? ", site form" : ", Gamma form")+(half ? ", half build" : ", whole build");
      const BuildCosts cost = build_costs(system,sites,half), again = build_costs(system,sites,half);
      require(cost.multiplications.size() == atoms && cost.per_abf.size() == atoms && cost.per_part.size() == atoms &&
                  cost.multiplications == again.multiplications && cost.per_abf == again.per_abf &&
                  cost.per_part == again.per_part,
              form+": the estimates of the ABFs are not one per atom or change between calls");
      for (std::size_t a = 0; a < atoms; ++a)
        require(cost.multiplications[a] >= 1.0 && std::isfinite(cost.per_abf[a]) &&
                    cost.per_abf[a] >= cost.multiplications[a] && cost.per_part[a] > 0.0 &&
                    std::isfinite(cost.per_part[a]),
                form+": an estimate of the ABFs of an atom is not a positive count");

      // best[k][j]: the smallest largest estimate of k contiguous ranges,
      // some of them empty, of the ABFs 0 <= P < j.  The estimate of the ABFs
      // i <= P < j comes from sums over the ABFs and over the atoms here, and
      // from range_estimate() for the ranges of divide_abfs().
      std::vector<double> abfs(1,0.0), parts(1,0.0);
      for (std::size_t p = 0; p < total; ++p) abfs.push_back(abfs.back()+cost.per_abf[atom_of[p]]);
      for (std::size_t a = 0; a < atoms; ++a)
        parts.push_back(parts.back()+(system.abf_count[a] != 0 ? cost.per_part[a] : 0.0));
      const auto estimate = [&](std::size_t i, std::size_t j) {
        return i < j ? abfs[j]-abfs[i]+parts[atom_of[j-1]+1]-parts[atom_of[i]] : 0.0;
      };
      const double unreached = 1e300;
      std::vector<std::vector<double>> best(most+1,std::vector<double>(total+1,unreached));
      best[0][0] = 0.0;
      for (std::size_t k = 1; k <= most; ++k)
        for (std::size_t j = 0; j <= total; ++j)
          for (std::size_t i = 0; i <= j; ++i) best[k][j] = std::min(best[k][j],std::max(best[k-1][i],estimate(i,j)));

      double single = 0.0;
      for (std::size_t p = 0; p < total; ++p) single = std::max(single,estimate(p,p+1));
      for (std::size_t processes = 1; processes <= most+1; ++processes) {
        const std::size_t count = processes <= most ? processes : total+3;
        const std::vector<std::size_t> boundaries = divide_abfs(system,sites,half,count);
        require(boundaries.size() == count+1 && boundaries.front() == 0 && boundaries.back() == total &&
                    std::is_sorted(boundaries.begin(),boundaries.end()) &&
                    boundaries == divide_abfs(system,sites,half,count),
                form+": the ranges of the processes do not cover the ABFs in order");
        double largest = 0.0;
        for (std::size_t r = 0; r < count; ++r)
          largest = std::max(largest,range_estimate(system,cost,boundaries[r],boundaries[r+1]));
        const double smallest = count <= most ? best[count][total] : single;
        require(std::fabs(largest-smallest) <= 1e-9*smallest,
                form+": the division of the ABFs among "+std::to_string(count)+" processes is not the best one");
      }
      require(divide_abfs(system,sites,half,1) == std::vector<std::size_t>({0,total}),
              form+": one process does not take all ABFs");
      std::printf("%s: four processes, the largest estimate %.4f times a quarter of that of one process\n",
                  form.c_str(),best[4][total]/(0.25*estimate(0,total)));
    }
  }
  expect_invalid("a division of the ABFs among no processes",[&] { divide_abfs(system,false,false,0); });
}

// The ABFs of the centre atoms in ranges (section 6.7 of the guide), in both
// forms and with four backends: the built-in loops, a table with a product,
// a table whose loop runs four pieces, and the device.  A range of the ABFs
// of whole atoms gives the bits of these atoms as a range of centre atoms.
// Ranges that divide the ABFs add up to K within rounding wherever they are
// cut: in two at every ABF, and in ranges of 1, 2, 3, ... ABFs.  The device
// is told the bytes and the multiplications of the products of the range
// that it computes.
// An empty range adds nothing.  With "every" the cut is made at every ABF;
// otherwise at about a dozen of them, and at the first and the last one.
// The division of the ABFs among processes is check_division().
void check_abf_ranges(const System& system, const std::string& name, Numbers& random, bool every)
{
  const double alpha = 0.25;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  std::vector<std::size_t> first_abf(1,0);
  for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
  const std::size_t total = first_abf.back();
  require(total >= 3,name+": the test of the ranges needs three ABFs or more");
  const std::vector<double> dense = dense_matrix(n,false,random);
  const BlockLayout layout = exchange_layout(system);
  std::vector<double> blocks(layout.elements());
  for (double& value : blocks) value = random.next();

  using Build = std::function<bool(std::size_t, std::size_t, std::span<double>, const DenseBackend&)>;
  struct RangeForm {
    std::string name;
    std::size_t size = 0;
    Build abfs, centres;
  };
  const RangeForm forms[] = {
      {name+", Gamma form",n*n,
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_gamma_abfs(system,first,last,-alpha,dense,k,backend);
       },
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_gamma_range(system,first,last,-alpha,dense,k,backend);
       }},
      {name+", site form",layout.elements(),
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_blocks_abfs(system,layout,first,last,-alpha,blocks,k,backend);
       },
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_blocks_range(system,layout,first,last,-alpha,blocks,k,backend);
       }}};

  FakeDevice device;
  Calls calls;
  const OpenMXExxDenseBackend product_table{&shared_dgemm,&calls,nullptr,0,nullptr};
  const OpenMXExxDenseBackend pieces_table{&shared_dgemm,&calls,&reversed_loop,4,nullptr};
  const OpenMXExxDenseBackend device_table{&shared_dgemm,&calls,&in_order_loop,4,&device.table()};
  const OpenMXExxDenseBackend* const tables[] = {nullptr,&product_table,&pieces_table,&device_table};
  const char* const backend_names[] = {"built-in loops","product of the table","four pieces","device"};
  const std::size_t step = every ? 1 : std::max<std::size_t>(1,total/12);

  for (const RangeForm& form : forms) {
    std::vector<double> alone(form.size,0.0);
    form.centres(0,atoms,alone,DenseBackend());
    const double scale = largest(alone);
    require(scale > 1.0e-6,form.name+": the contributions are expected to be of order one");
    // K holds numbers of the size of the contributions on entry: a range adds
    // to it.
    std::vector<double> start(form.size);
    for (double& value : start) value = scale*random.next();

    long builds = 0;
    for (std::size_t b = 0; b < 4; ++b) {
      const DenseBackend backend(tables[b]);
      const bool on_device = b == 3;
      const std::string what = form.name+", "+backend_names[b];
      // A call of one range; the record of the device must end clean, and
      // the device computes every range that is not empty.
      const auto run = [&](const Build& build, std::size_t first, std::size_t last, std::vector<double>& k) {
        device.reset();
        const bool computed = build(first,last,k,backend);
        const std::string ending = device.ending();
        require(ending.empty(),what+": "+ending);
        require(computed == (on_device && first < last),what+": the wrong side computes a range");
        if (computed)
          require(device.accepted == 1 && device.bytes == device.allocated &&
                  device.multiplications == device.all_products && device.count[FakeDevice::download_call] == 1,
                  what+": accept is told other sizes than the range takes, or K is not fetched once");
        ++builds;
      };

      std::vector<double> reference = start;
      run(form.centres,0,atoms,reference);
      {
        std::vector<double> all = start;
        run(form.abfs,0,total,all);
        require(same_values(all,reference),what+": all ABFs in one range give other bits than all centre atoms");
      }
      for (std::size_t i = 0; i < atoms; ++i)
        for (std::size_t j = i+1; j <= atoms; ++j) {
          if (i == 0 && j == atoms) continue;
          std::vector<double> by_atoms = start, by_abfs = start;
          run(form.centres,i,j,by_atoms);
          run(form.abfs,first_abf[i],first_abf[j],by_abfs);
          require(same_values(by_abfs,by_atoms),what+": the ABFs of whole atoms give other bits than the atoms");
        }

      Worst cut(what+": two ranges of ABFs (relative)",1e-12);
      std::vector<std::size_t> middles;
      for (std::size_t middle = 1; middle < total; middle += step) middles.push_back(middle);
      if (middles.back() != total-1) middles.push_back(total-1);
      for (const std::size_t middle : middles) {
        std::vector<double> k = start;
        // The second range first for every other cut: the order of the calls
        // is that of the processes, which is not fixed.
        if (middle%2 == 0) {
          run(form.abfs,0,middle,k);
          run(form.abfs,middle,total,k);
        } else {
          run(form.abfs,middle,total,k);
          run(form.abfs,0,middle,k);
        }
        for (std::size_t e = 0; e < k.size(); ++e) cut.add(std::fabs(k[e]-reference[e])/scale);
      }
      cut.finish();

      Worst growing(what+": ranges of 1, 2, 3, ... ABFs (relative)",1e-12);
      {
        std::vector<double> k = start;
        for (std::size_t first = 0, length = 1; first < total; ++length) {
          const std::size_t last = std::min(total,first+length);
          run(form.abfs,first,last,k);
          first = last;
        }
        for (std::size_t e = 0; e < k.size(); ++e) growing.add(std::fabs(k[e]-reference[e])/scale);
      }
      growing.finish();

      for (const std::size_t at : {std::size_t{0},first_abf[1],total-1,total}) {
        std::vector<double> none = start;
        run(form.abfs,at,at,none);
        require(device.accepted == 0 && same_values(none,start),what+": an empty range of ABFs asks the device or changes K");
      }
    }
    std::printf("%s: %ld builds of ranges of the %zu ABFs\n",form.name.c_str(),builds,total);

    // Ranges that the library rejects add nothing, with a device too.
    std::vector<double> k(form.size,0.0);
    device.reset();
    for (const OpenMXExxDenseBackend* const table : {tables[0],tables[3]}) {
      const DenseBackend backend(table);
      expect_invalid("a range of ABFs beyond the cell",[&] { form.abfs(0,total+1,k,backend); });
      expect_invalid("a range of ABFs that ends before it begins",[&] { form.abfs(2,1,k,backend); });
      expect_invalid("an exchange matrix of the wrong size for a range of ABFs",
                     [&] { form.abfs(0,total,std::span<double>(k).first(k.size()-1),backend); });
    }
    require(largest(k) == 0.0 && device.accepted == 0 && device.calls == 0,
            form.name+": a rejected range of ABFs changes K or asks the device");
  }

  check_division(system,name);
}

// The half build (section 6.4 of the guide) against the whole build.  With a
// symmetric density matrix and a system with the symmetry of
// prepare_system(), the sum X of the half build gives K = X + X^T, in both
// forms and with five backends: the built-in loops, a table with a product,
// a table whose loop runs four pieces in reverse order, the same pieces on
// threads of their own, and the device.  The ABFs in one range, in two
// ranges cut at several ABFs in both orders, and in ranges of growing
// length; X + X^T is symmetric to the last bit.  An empty range adds nothing,
// and rejected arguments change nothing.  The estimates of the ABFs of the
// half build are smaller than those of the whole build.
void check_half_build(const System& system, const std::string& name, Numbers& random, bool every)
{
  const double alpha = 0.25;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  std::vector<std::size_t> first_abf(1,0);
  for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
  const std::size_t total = first_abf.back();
  require(total >= 3,name+": the test of the half build needs three ABFs or more");
  const std::vector<double> dense = dense_matrix(n,true,random);
  const BlockLayout layout = exchange_layout(system);
  const std::vector<double> blocks = symmetric_blocks(system,layout,random);

  using Build = std::function<bool(std::size_t, std::size_t, std::span<double>, const DenseBackend&)>;
  using Change = std::function<void(std::span<double>)>;
  struct HalfForm {
    std::string name;
    std::size_t size = 0;
    Build whole, half;
    Change mirror, symmetrize;
    std::function<bool(std::span<const double>)> symmetric;
  };
  const HalfForm forms[] = {
      {name+", Gamma form",n*n,
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_gamma_abfs(system,first,last,-alpha,dense,k,backend);
       },
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_gamma_half_abfs(system,first,last,-alpha,dense,k,backend);
       },
       [&](std::span<double> k) { mirror_gamma(system,k); },
       [&](std::span<double> k) {
         for (std::size_t i = 0; i < n; ++i)
           for (std::size_t j = i+1; j < n; ++j) k[i*n+j] = k[j*n+i] = 0.5*(k[i*n+j]+k[j*n+i]);
       },
       [&](std::span<const double> k) {
         for (std::size_t i = 0; i < n; ++i)
           for (std::size_t j = i+1; j < n; ++j)
             if (k[i*n+j] != k[j*n+i]) return false;
         return true;
       }},
      {name+", site form",layout.elements(),
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_blocks_abfs(system,layout,first,last,-alpha,blocks,k,backend);
       },
       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
         return add_exchange_blocks_half_abfs(system,layout,first,last,-alpha,blocks,k,backend);
       },
       [&](std::span<double> k) { mirror_blocks(system,layout,k); },
       [&](std::span<double> k) { symmetrize_blocks(system,layout,k); },
       [&](std::span<const double> k) { return block_asymmetry(system,layout,k) == 0.0; }}};

  FakeDevice device;
  Calls calls;
  const OpenMXExxDenseBackend product_table{&shared_dgemm,&calls,nullptr,0,nullptr};
  const OpenMXExxDenseBackend pieces_table{&shared_dgemm,&calls,&reversed_loop,4,nullptr};
  const OpenMXExxDenseBackend threads_table{&shared_dgemm,&calls,&threaded_loop,4,nullptr};
  const OpenMXExxDenseBackend device_table{&shared_dgemm,&calls,&in_order_loop,4,&device.table()};
  const OpenMXExxDenseBackend* const tables[] = {nullptr,&product_table,&pieces_table,&threads_table,&device_table};
  const char* const backend_names[] = {"built-in loops","product of the table","four pieces",
                                       "four pieces on threads","device"};
  const std::size_t step = every ? 1 : std::max<std::size_t>(1,total/12);

  for (const HalfForm& form : forms) {
    // K of the whole build, made symmetric.
    std::vector<double> reference(form.size,0.0);
    form.whole(0,total,reference,DenseBackend());
    form.symmetrize(reference);
    const double scale = largest(reference);
    require(scale > 1.0e-6,form.name+": the contributions are expected to be of order one");

    long builds = 0;
    for (std::size_t b = 0; b < 5; ++b) {
      const DenseBackend backend(tables[b]);
      const bool on_device = b == 4;
      const std::string what = form.name+", "+backend_names[b];
      // A call of one range; the record of the device must end clean, and
      // the device computes every range that is not empty.
      const auto run = [&](std::size_t first, std::size_t last, std::vector<double>& x) {
        device.reset();
        const bool computed = form.half(first,last,x,backend);
        const std::string ending = device.ending();
        require(ending.empty(),what+": "+ending);
        require(computed == (on_device && first < last),what+": the wrong side computes a range of the half build");
        if (computed)
          require(device.accepted == 1 && device.bytes == device.allocated &&
                  device.multiplications == device.all_products && device.count[FakeDevice::download_call] == 1,
                  what+": accept is told other sizes than the half build takes, or K is not fetched once");
        ++builds;
      };
      // X + X^T against the reference.
      const auto compare = [&](std::vector<double>& x, Worst& worst) {
        form.mirror(x);
        require(form.symmetric(x),what+": X + X^T is not symmetric to the last bit");
        for (std::size_t e = 0; e < x.size(); ++e) worst.add(std::fabs(x[e]-reference[e])/scale);
      };

      Worst one(what+": half build, all ABFs (relative)",1e-12);
      {
        std::vector<double> x(form.size,0.0);
        run(0,total,x);
        compare(x,one);
      }
      one.finish();

      Worst cut(what+": half build in two ranges (relative)",1e-12);
      std::vector<std::size_t> middles;
      for (std::size_t middle = 1; middle < total; middle += step) middles.push_back(middle);
      if (middles.back() != total-1) middles.push_back(total-1);
      for (const std::size_t middle : middles) {
        std::vector<double> x(form.size,0.0);
        if (middle%2 == 0) {
          run(0,middle,x);
          run(middle,total,x);
        } else {
          run(middle,total,x);
          run(0,middle,x);
        }
        compare(x,cut);
      }
      cut.finish();

      Worst growing(what+": half build in ranges of 1, 2, 3, ... ABFs (relative)",1e-12);
      {
        std::vector<double> x(form.size,0.0);
        for (std::size_t first = 0, length = 1; first < total; ++length) {
          const std::size_t last = std::min(total,first+length);
          run(first,last,x);
          first = last;
        }
        compare(x,growing);
      }
      growing.finish();

      for (const std::size_t at : {std::size_t{0},first_abf[1],total-1,total}) {
        std::vector<double> none(form.size,0.0);
        run(at,at,none);
        require(device.accepted == 0 && largest(none) == 0.0,
                what+": an empty range of the half build asks the device or changes X");
      }
    }
    std::printf("%s: %ld half builds of ranges of the %zu ABFs\n",form.name.c_str(),builds,total);

    // Arguments that the library rejects change nothing, with a device too.
    std::vector<double> k(form.size,0.0);
    device.reset();
    for (const OpenMXExxDenseBackend* const table : {tables[0],tables[4]}) {
      const DenseBackend backend(table);
      expect_invalid("a range of ABFs of the half build beyond the cell",[&] { form.half(0,total+1,k,backend); });
      expect_invalid("a range of ABFs of the half build that ends before it begins",
                     [&] { form.half(2,1,k,backend); });
      expect_invalid("an exchange matrix of the half build of the wrong size",
                     [&] { form.half(0,total,std::span<double>(k).first(k.size()-1),backend); });
    }
    expect_invalid("a mirror image of an array of the wrong size",
                   [&] { form.mirror(std::span<double>(k).first(k.size()-1)); });
    require(largest(k) == 0.0 && device.accepted == 0 && device.calls == 0,
            form.name+": a rejected half build changes X or asks the device");
  }

  // The estimates of the half build: no more than those of the whole build
  // for any atom, and fewer multiplications and less time in all.
  for (const bool sites : {false,true}) {
    const BuildCosts half = build_costs(system,sites,true), whole = build_costs(system,sites);
    require(half.multiplications.size() == atoms && half.per_abf.size() == atoms && half.per_part.size() == atoms,
            name+": wrong number of estimates of the ABFs of the half build");
    double half_total = 0.0, whole_total = 0.0, half_time = 0.0, whole_time = 0.0;
    for (std::size_t a = 0; a < atoms; ++a) {
      require(half.multiplications[a] >= 1.0 && half.multiplications[a] <= whole.multiplications[a] &&
                  half.per_abf[a] <= whole.per_abf[a] && half.per_part[a] > 0.0 &&
                  half.per_part[a] <= whole.per_part[a],
              name+": an estimate of the ABFs of the half build is not a count below that of the whole build");
      const double count = static_cast<double>(system.abf_count[a]);
      half_total += half.multiplications[a]*count;
      whole_total += whole.multiplications[a]*count;
      half_time += half.per_abf[a]*count+half.per_part[a];
      whole_time += whole.per_abf[a]*count+whole.per_part[a];
    }
    require(half_total < whole_total && half_time < whole_time,
            name+": the half build is not estimated to take fewer multiplications and less time");
    std::printf("%s, %s form: the half build has %.3f of the estimated multiplications and %.3f of the estimated "
                "time of the whole build\n",
                name.c_str(),sites ? "site" : "Gamma",half_total/whole_total,half_time/whole_time);
  }
}

// The weights of the blocks on the device (device_gradients_gamma() and
// device_gradients_blocks()), in both forms, against the table of the host
// with a product and a loop of four pieces.  The density matrices need not
// be symmetric here: both sides compute the same sums.  With "every" every
// call of the device fails in turn; otherwise about forty of them do.
void check_device_gradients(const System& system, const std::string& name, Numbers& random, bool every)
{
  const double alpha = 0.25;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  std::vector<std::size_t> first_abf(1,0);
  for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
  const std::size_t total = first_abf.back();
  const std::vector<double> dense = dense_matrix(n,false,random);
  const BlockLayout layout = exchange_layout(system);
  std::vector<double> blocks(layout.elements());
  for (double& value : blocks) value = random.next();

  using Weights = std::function<void(std::size_t, std::size_t, std::span<const std::span<double>>, const DenseBackend&)>;
  using Direct = std::function<std::size_t(std::size_t, std::size_t, const DenseBackend&, const GradientVisit&)>;
  using Host = std::function<CentreGradient(std::size_t, std::size_t, std::size_t, const DenseBackend&)>;
  struct GradientForm {
    std::string name;
    Weights weights;
    Direct direct;
    Host host;
  };
  const GradientForm forms[] = {
      {name+", Gamma form",
       [&](std::size_t first, std::size_t last, std::span<const std::span<double>> arrays, const DenseBackend& backend) {
         add_block_weights_gamma_abfs(system,first,last,-alpha,dense,arrays,backend);
       },
       [&](std::size_t first, std::size_t last, const DenseBackend& backend, const GradientVisit& visit) {
         return device_gradients_gamma(system,first,last,-alpha,dense,backend,visit);
       },
       [&](std::size_t atom, std::size_t from, std::size_t to, const DenseBackend& backend) {
         return centre_gradient_gamma(system,atom,from,to,-alpha,dense,backend);
       }},
      {name+", site form",
       [&](std::size_t first, std::size_t last, std::span<const std::span<double>> arrays, const DenseBackend& backend) {
         add_block_weights_blocks_abfs(system,layout,first,last,-alpha,blocks,arrays,backend);
       },
       [&](std::size_t first, std::size_t last, const DenseBackend& backend, const GradientVisit& visit) {
         return device_gradients_blocks(system,layout,first,last,-alpha,blocks,backend,visit);
       },
       [&](std::size_t atom, std::size_t from, std::size_t to, const DenseBackend& backend) {
         return centre_gradient_blocks(system,layout,atom,from,to,-alpha,blocks,backend);
       }}};

  FakeDevice device;
  Calls calls;
  const OpenMXExxDenseBackend host_table{&shared_dgemm,&calls,&in_order_loop,4,nullptr};
  const OpenMXExxDenseBackend device_table{&shared_dgemm,&calls,&in_order_loop,4,&device.table()};
  const DenseBackend host(&host_table), with_device(&device_table);
  const auto flat = [](const BlockWeights& w) {
    std::vector<double> all;
    for (const std::vector<double>& values : w.values) all.insert(all.end(),values.begin(),values.end());
    return all;
  };

  for (const GradientForm& form : forms) {
    const std::string& what = form.name;
    // The weights of the range on the host and with the device.
    BlockWeights reference(system);
    form.weights(0,total,reference.arrays,host);
    const std::vector<double> expected = flat(reference);
    const double scale = largest(expected);
    require(scale > 1.0e-6,what+": the weights are expected to be of order one");
    const auto run = [&](std::size_t first, std::size_t last, BlockWeights& w) {
      device.reset();
      calls.reset();
      form.weights(first,last,w.arrays,with_device);
      const std::string ending = device.ending();
      require(ending.empty(),what+": "+ending);
    };
    std::size_t parts = 0;                          // the atoms with ABFs
    for (std::size_t a = 0; a < atoms; ++a) parts += first_abf[a] < first_abf[a+1] ? 1 : 0;

    // The device computes every part: the table of the host is not used,
    // accept is told the bytes and the multiplications of all products, and every part
    // brings back two arrays.
    BlockWeights device_weights(system);
    run(0,total,device_weights);
    require(calls.products == 0 && calls.loops == 0,what+": the product or the loop of the table is used with a device");
    require(device.accepted == 1 && device.bytes == device.allocated &&
            device.multiplications == device.all_products &&
            device.count[FakeDevice::download_call] == static_cast<long>(2*parts) &&
            device.count[FakeDevice::allocate_call] == device.count[FakeDevice::release_call],
            what+": accept is told other sizes than the call takes, or wrong calls of the device");
    const std::vector<double> computed = flat(device_weights);
    Worst worst(what+": weights on the device against the host (relative)",1e-12);
    for (std::size_t e = 0; e < computed.size(); ++e) worst.add(std::fabs(computed[e]-expected[e])/scale);
    worst.finish();
    const std::vector<FakeDevice::Kind> sequence = device.sequence;
    std::printf("%s: %ld calls of the device for the weights, %.0f bytes\n",what.c_str(),
                static_cast<long>(sequence.size()),static_cast<double>(device.bytes));

    // The same bits again, and the parts directly: in the order of the
    // atoms, each one the derivatives of the host for that part within
    // rounding.
    {
      BlockWeights again(system);
      run(0,total,again);
      require(same_values(flat(again),computed),what+": the device gives other bits the second time");
      Worst part_worst(what+": parts on the device against the host (relative)",1e-12);
      for (const std::size_t middle : {first_abf[1]/2,first_abf[1],total-1}) {
        device.reset();
        std::vector<std::array<std::size_t,3>> seen;
        const std::size_t visited = form.direct(middle,total,with_device,
            [&](std::size_t atom, std::size_t from, std::size_t to, CentreGradient&& gradient) {
              seen.push_back({atom,from,to});
              const CentreGradient other = form.host(atom,from,to,host);
              require(gradient.t.size() == other.t.size() && gradient.v.size() == other.v.size(),
                      what+": a part on the device has other sizes than on the host");
              for (std::size_t e = 0; e < other.t.size(); ++e)
                part_worst.add(std::fabs(gradient.t[e]-other.t[e])/scale);
              for (std::size_t i = 0; i < other.v.size(); ++i) {
                require(gradient.v[i].size() == other.v[i].size(),what+": a V block of a part has another size");
                for (std::size_t e = 0; e < other.v[i].size(); ++e)
                  part_worst.add(std::fabs(gradient.v[i][e]-other.v[i][e])/scale);
              }
            });
        require(device.ending().empty(),what+": "+device.ending());
        std::vector<std::array<std::size_t,3>> wanted;
        for (std::size_t a = 0; a < atoms; ++a) {
          const std::size_t from = std::max(middle,first_abf[a]), to = std::min(total,first_abf[a+1]);
          if (from < to) wanted.push_back({a,from-first_abf[a],to-first_abf[a]});
        }
        require(visited == wanted.size() && seen == wanted,what+": the device visits other parts than the range holds");
      }
      part_worst.finish();
    }

    // Ranges that divide the ABFs fill the arrays within rounding.
    {
      Worst cut_worst(what+": two ranges of ABFs on the device (relative)",1e-12);
      for (const std::size_t middle : {std::size_t{1},first_abf[1],total-1}) {
        BlockWeights cut(system);
        run(0,middle,cut);
        run(middle,total,cut);
        const std::vector<double> sum = flat(cut);
        for (std::size_t e = 0; e < sum.size(); ++e) cut_worst.add(std::fabs(sum[e]-expected[e])/scale);
      }
      cut_worst.finish();
    }

    // An empty range and a backend without a device do not ask the device.
    {
      BlockWeights none(system);
      for (const std::size_t at : {std::size_t{0},first_abf[1],total}) {
        run(at,at,none);
        require(device.accepted == 0,what+": an empty range of ABFs asks the device");
      }
      require(largest(flat(none)) == 0.0,what+": an empty range of ABFs changes the weights");
      device.reset();
      require(form.direct(0,total,host,[](std::size_t, std::size_t, std::size_t, CentreGradient&&) {}) == 0 &&
              device.accepted == 0,what+": a backend without a device visits parts");
    }

    // The device declines, or lacks a function: the host computes the
    // weights with its own bits.
    {
      device.answer = 0;
      BlockWeights declined(system);
      run(0,total,declined);
      require(device.accepted == 1 && device.calls == 0 && calls.products > 0 && same_values(flat(declined),expected),
              what+": a device that declines is used, or the host gives other bits");
      device.answer = 1;
      OpenMXExxDevice partial = device.table();
      partial.move = nullptr;
      const OpenMXExxDenseBackend table{&shared_dgemm,&calls,&in_order_loop,4,&partial};
      BlockWeights without(system);
      device.reset();
      form.weights(0,total,without.arrays,DenseBackend(&table));
      require(device.accepted == 0 && same_values(flat(without),expected),what+": a device without one of its functions is used");
    }

    // Every call of the device fails in turn: the library stops using the
    // device, releases every block, and the host computes the parts that
    // the device did not bring back.  Those parts have the bits of the host.
    const std::vector<long> chosen = failures(sequence,every);
    Worst failed_worst(what+": weights after a failure of the device (relative)",1e-12);
    for (const long index : chosen) {
      device.failing = index;
      BlockWeights failed(system);
      run(0,total,failed);
      require(device.failed && device.calls > index && calls.products > 0,what+": a failure of the device is not noticed");
      const std::vector<double> mixed = flat(failed);
      for (std::size_t e = 0; e < mixed.size(); ++e) failed_worst.add(std::fabs(mixed[e]-expected[e])/scale);
      // The parts of the host: those after the last part that came back.
      long downloads = 0;
      for (long i = 0; i < index; ++i) downloads += sequence[static_cast<std::size_t>(i)] == FakeDevice::download_call;
      const std::size_t back = static_cast<std::size_t>(downloads/2);
      std::size_t part = 0;
      for (std::size_t a = 0; a < atoms; ++a) {
        if (first_abf[a] == first_abf[a+1] || part++ < back) continue;
        require(same_values(failed.values[a],reference.values[a]),
                what+": a part that the device did not bring back has other bits than on the host");
      }
    }
    device.failing = -1;
    failed_worst.finish();
    std::printf("%s: %zu failures of the device for the weights, each one followed by the host\n",what.c_str(),
                chosen.size());

    // Arguments that the library rejects are rejected before the device is
    // asked.
    device.reset();
    BlockWeights rejected(system);
    expect_invalid("a range of ABFs beyond the cell, with a device",
                   [&] { form.weights(0,total+1,rejected.arrays,with_device); });
    expect_invalid("a range of ABFs beyond the cell, on the device",
                   [&] { form.direct(0,total+1,with_device,[](std::size_t, std::size_t, std::size_t, CentreGradient&&) {}); });
    require(device.accepted == 0 && largest(flat(rejected)) == 0.0,
            what+": a rejected range asks the device or changes the weights");
  }
}

// The half form of the weights on the device (device_gradients_gamma() and
// device_gradients_blocks() with half), in both forms, against the host with
// the same table.  The density matrices need not be symmetric: both sides
// compute the same sums.  Every part brings back three arrays: dt, dv and
// the T blocks through W.  After a failure of the device the host computes
// the parts that did not come back, and these add to the T blocks of other
// atoms too, so the arrays are compared within rounding only.
void check_device_half_gradients(const System& system, const std::string& name, Numbers& random, bool every)
{
  const double alpha = 0.25;
  const std::size_t atoms = system.orbital_count.size(), n = orbital_total(system);
  std::vector<std::size_t> first_abf(1,0);
  for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
  const std::size_t total = first_abf.back();
  const std::vector<double> dense = dense_matrix(n,false,random);
  const BlockLayout layout = exchange_layout(system);
  std::vector<double> blocks(layout.elements());
  for (double& value : blocks) value = random.next();

  using Weights = std::function<void(std::size_t, std::size_t, std::span<const std::span<double>>, const DenseBackend&)>;
  struct HalfForm {
    std::string name;
    Weights weights;
  };
  const HalfForm forms[] = {
      {name+", Gamma form, half weights",
       [&](std::size_t first, std::size_t last, std::span<const std::span<double>> arrays, const DenseBackend& backend) {
         add_block_weights_gamma_half_abfs(system,first,last,-alpha,dense,arrays,backend);
       }},
      {name+", site form, half weights",
       [&](std::size_t first, std::size_t last, std::span<const std::span<double>> arrays, const DenseBackend& backend) {
         add_block_weights_blocks_half_abfs(system,layout,first,last,-alpha,blocks,arrays,backend);
       }}};

  FakeDevice device;
  Calls calls;
  const OpenMXExxDenseBackend host_table{&shared_dgemm,&calls,&in_order_loop,4,nullptr};
  const OpenMXExxDenseBackend device_table{&shared_dgemm,&calls,&in_order_loop,4,&device.table()};
  const DenseBackend host(&host_table), with_device(&device_table);
  const auto flat = [](const BlockWeights& w) {
    std::vector<double> all;
    for (const std::vector<double>& values : w.values) all.insert(all.end(),values.begin(),values.end());
    return all;
  };

  for (const HalfForm& form : forms) {
    const std::string& what = form.name;
    BlockWeights reference(system);
    form.weights(0,total,reference.arrays,host);
    const std::vector<double> expected = flat(reference);
    const double scale = largest(expected);
    require(scale > 1.0e-6,what+": the weights are expected to be of order one");
    const auto run = [&](std::size_t first, std::size_t last, BlockWeights& w) {
      device.reset();
      calls.reset();
      form.weights(first,last,w.arrays,with_device);
      const std::string ending = device.ending();
      require(ending.empty(),what+": "+ending);
    };
    const auto compare = [&](const BlockWeights& w, Worst& worst) {
      const std::vector<double> values = flat(w);
      for (std::size_t e = 0; e < values.size(); ++e) worst.add(std::fabs(values[e]-expected[e])/scale);
    };
    std::size_t parts = 0;                          // the atoms with ABFs
    for (std::size_t a = 0; a < atoms; ++a) parts += first_abf[a] < first_abf[a+1] ? 1 : 0;

    // The device computes every part: the table of the host is not used,
    // accept is told the bytes and the multiplications of all products, and every part
    // brings back three arrays.
    BlockWeights device_weights(system);
    run(0,total,device_weights);
    const std::vector<FakeDevice::Kind> sequence = device.sequence;
    require(calls.products == 0 && calls.loops == 0,what+": the product or the loop of the table is used with a device");
    require(device.accepted == 1 && device.bytes == device.allocated &&
            device.multiplications == device.all_products &&
            device.count[FakeDevice::download_call] == static_cast<long>(3*parts) &&
            device.count[FakeDevice::allocate_call] == device.count[FakeDevice::release_call],
            what+": accept is told other sizes than the call takes, or wrong calls of the device");
    Worst worst(what+": weights on the device against the host (relative)",1e-12);
    compare(device_weights,worst);
    worst.finish();
    {
      BlockWeights again(system);
      run(0,total,again);
      require(same_values(flat(again),flat(device_weights)),what+": the device gives other bits the second time");
    }

    // Ranges that divide the ABFs, inside an atom as well.
    {
      Worst cut_worst(what+": two ranges of ABFs on the device (relative)",1e-12);
      for (const std::size_t middle : {std::size_t{1},first_abf[1]/2,first_abf[1],total-1}) {
        BlockWeights cut(system);
        run(0,middle,cut);
        run(middle,total,cut);
        compare(cut,cut_worst);
      }
      cut_worst.finish();
    }

    // The device declines: the host computes the weights with its own bits.
    {
      device.answer = 0;
      BlockWeights declined(system);
      run(0,total,declined);
      require(device.accepted == 1 && device.calls == 0 && calls.products > 0 && same_values(flat(declined),expected),
              what+": a device that declines is used, or the host gives other bits");
      device.answer = 1;
    }

    // Every call of the device fails in turn: the host computes the parts
    // that the device did not bring back.
    const std::vector<long> chosen = failures(sequence,every);
    Worst failed_worst(what+": weights after a failure of the device (relative)",1e-12);
    for (const long index : chosen) {
      device.failing = index;
      BlockWeights failed(system);
      run(0,total,failed);
      require(device.failed && device.calls > index && calls.products > 0,what+": a failure of the device is not noticed");
      compare(failed,failed_worst);
    }
    device.failing = -1;
    failed_worst.finish();
    std::printf("%s: %ld calls of the device, %zu failures, each one followed by the host\n",what.c_str(),
                static_cast<long>(sequence.size()),chosen.size());
  }
}

// Both forms of a system with a density matrix of numbers without a
// pattern, not symmetric: a transposed block does not pass unnoticed.  The
// blocks of the site form are numbers of their own, not those of one dense
// matrix: blocks of different cells differ.
void check_device_forms(const System& system, const std::string& name, Numbers& random, bool every)
{
  const double alpha = 0.25;
  const std::size_t n = orbital_total(system);
  const std::vector<double> dense = dense_matrix(n,false,random);
  const BlockLayout layout = exchange_layout(system);
  std::vector<double> blocks(layout.elements());
  for (double& value : blocks) value = random.next();

  DeviceForm gamma{name+", Gamma form",n*n,
                   [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
                     return add_exchange_gamma_range(system,first,last,-alpha,dense,k,backend);
                   }};
  DeviceForm site{name+", site form",layout.elements(),
                  [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
                    return add_exchange_blocks_range(system,layout,first,last,-alpha,blocks,k,backend);
                  }};
  // The half build of the centre atoms first to last-1: the device and the
  // host compute the same sums, which need no symmetry here.
  std::vector<std::size_t> first_abf(1,0);
  for (const std::size_t count : system.abf_count) first_abf.push_back(first_abf.back()+count);
  DeviceForm half_gamma{name+", Gamma form, half build",n*n,
                        [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
                          return add_exchange_gamma_half_abfs(system,first_abf[first],first_abf[last],-alpha,dense,k,
                                                              backend);
                        }};
  DeviceForm half_site{name+", site form, half build",layout.elements(),
                       [&](std::size_t first, std::size_t last, std::span<double> k, const DenseBackend& backend) {
                         return add_exchange_blocks_half_abfs(system,layout,first_abf[first],first_abf[last],-alpha,
                                                              blocks,k,backend);
                       }};
  check_device(system,gamma,random,every);
  check_device(system,site,random,every);
  check_device(system,half_gamma,random,every);
  check_device(system,half_site,random,every);
  check_abf_ranges(system,name,random,every);
  check_device_gradients(system,name,random,every);
  check_device_half_gradients(system,name,random,every);

  // Arguments that the library rejects are rejected before the device is
  // asked; the derivatives of one centre atom do not use the device.
  FakeDevice device;
  const OpenMXExxDenseBackend table{nullptr,nullptr,nullptr,0,&device.table()};
  const DenseBackend backend(&table);
  const std::size_t atoms = system.orbital_count.size();
  std::vector<double> k(n*n,0.0), wrong(n*n+1,0.0), site_k(layout.elements(),0.0);
  device.reset();
  expect_invalid("a range of centre atoms beyond the cell, with a device",
                 [&] { add_exchange_gamma_range(system,0,atoms+1,-alpha,dense,k,backend); });
  expect_invalid("a range of centre atoms that ends before it begins, with a device",
                 [&] { add_exchange_blocks_range(system,layout,1,0,-alpha,blocks,site_k,backend); });
  expect_invalid("an exchange matrix of the wrong size, with a device",
                 [&] { add_exchange_gamma_range(system,0,atoms,-alpha,dense,wrong,backend); });
  expect_invalid("a density matrix of the wrong size, with a device",
                 [&] { add_exchange_blocks_range(system,layout,0,atoms,-alpha,wrong,site_k,backend); });
  (void)centre_gradient_gamma(system,0,-alpha,dense,backend);
  (void)centre_gradient_blocks(system,layout,0,-alpha,blocks,backend);
  require(device.accepted == 0 && device.calls == 0 && device.ending().empty(),
          name+": the device is asked for a call that is rejected, or for a derivative");
}

} // namespace

void exchange_device(const std::string& path)
{
  const KbuildFixture fixture = read_kbuild_fixture(path);
  Numbers random;
  check_device_forms(fixture.system,"fixture",random,true);
  check_device_forms(narrow_system(random),"narrow system",random,true);
}

namespace {

// The periodic structure of exx-exchange-system: the atoms have 10, 15 and 1
// ABFs, the cell is smaller than the cutoffs, and pairs of sites share blocks
// of K.
System periodic_system(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  require(fixture.species.size() == 3 && fixture.positions.size() == 3,"the test is written for the three species of the fixture");
  const int intervals = 300;
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(24.0,240);
  SystemOptions options;
  options.interaction = short_range_kernel(0.11);
  options.v_cutoff = 3.0;
  const std::vector<Shells> few{select(fixture.species[0],{0,2}),select(fixture.species[1],{0,1,2}),
                                select(fixture.species[2],{0})};
  Structure structure;
  structure.lattice = fixture.lattice;
  structure.positions = fixture.positions;
  structure.species = fixture.atom_species;
  System system = prepare_system(structure,tables_of(few,intervals,reciprocal,1.0e-3),reciprocal,options);
  require(system.abf_count == std::vector<std::size_t>({10,15,1}),"the test is written for atoms with 10, 15 and 1 ABFs");
  require(shared_pairs(system) > 0,"the structure is expected to have pairs of sites that share a block of K");
  return system;
}

void check_projected_blocks(const System& system, Numbers& random)
{
  const BlockLayout full = exchange_layout(system);
  std::vector<BlockKey> keys;
  std::vector<double> full_density(full.elements(),0.0), density;
  std::vector<std::size_t> indices;
  for (std::size_t b = 0; b < full.keys.size(); ++b) {
    const auto& cell = full.keys[b].cell;
    if (std::abs(cell[0]) > 1 || cell[1] != 0 || cell[2] != 0) continue;
    keys.push_back(full.keys[b]);
    for (std::size_t i = full.offset[b]; i < full.offset[b+1]; ++i) {
      indices.push_back(i);
      density.push_back(random.next());
    }
  }
  const BlockLayout local = projected_exchange_layout(system,std::move(keys));
  require(!density.empty() && density.size() < full.elements(),"the projected layout must omit blocks");
  symmetrize_blocks(system,local,density);
  for (std::size_t i = 0; i < indices.size(); ++i) full_density[indices[i]] = density[i];

  FakeDevice device;
  const OpenMXExxDenseBackend table{nullptr,nullptr,nullptr,0,&device.table()};
  const DenseBackend host, gpu(&table);
  const std::size_t abfs = std::accumulate(system.abf_count.begin(),system.abf_count.end(),std::size_t{0});
  constexpr double factor = -0.25;
  std::vector<double> full_k(full.elements()), local_k(local.elements()), device_k(local.elements());
  add_exchange_blocks_half_abfs(system,full,0,abfs,factor,full_density,full_k,host);
  add_exchange_blocks_half_abfs(system,local,0,abfs,factor,density,local_k,host);
  device.reset();
  require(add_exchange_blocks_half_abfs(system,local,0,abfs,factor,density,device_k,gpu) &&
          device.ending().empty(),"the projected exchange build did not run on the device");
  Worst k("projected K against the full layout with zero-padded D",1e-12);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    k.absolute(local_k[i],full_k[indices[i]]);
    k.absolute(device_k[i],full_k[indices[i]]);
  }
  k.finish();

  BlockWeights full_weights(system), local_weights(system), device_weights(system);
  add_block_weights_blocks_half_abfs(system,full,0,abfs,factor,full_density,full_weights.arrays,host);
  add_block_weights_blocks_half_abfs(system,local,0,abfs,factor,density,local_weights.arrays,host);
  device.reset();
  add_block_weights_blocks_half_abfs(system,local,0,abfs,factor,density,device_weights.arrays,gpu);
  require(device.accepted == 1 && device.ending().empty(),"the projected force weights did not run on the device");
  Worst weights("projected force weights against the full layout with zero-padded D",1e-12);
  for (std::size_t atom = 0; atom < full_weights.values.size(); ++atom)
    for (std::size_t i = 0; i < full_weights.values[atom].size(); ++i) {
      weights.absolute(local_weights.values[atom][i],full_weights.values[atom][i]);
      weights.absolute(device_weights.values[atom][i],full_weights.values[atom][i]);
    }
  weights.finish();
}

} // namespace

void exchange_device_system(const std::string& path)
{
  Numbers random;
  const System system = periodic_system(path);
  check_device_forms(system,"periodic structure",random,false);
  check_projected_blocks(system,random);
}

void exchange_half(const std::string& path)
{
  Numbers random;
  check_half_build(mirrored_system(random),"mirrored system",random,true);
  check_half_build(periodic_system(path),"periodic structure",random,false);
}

namespace {

constexpr double two_pi = 6.283185307179586476925286766559;

// The points of a k mesh of period[0] x period[1] x period[2] points in the
// coordinates of the reciprocal lattice, k_i = (m + shift_i/2)/n_i: a shift
// of 1 moves the mesh by half a point along b_i, as REGULAR does for an even
// number of points.
std::vector<std::array<double,3>> mesh_points(const std::array<int,3>& period, const std::array<int,3>& shift)
{
  std::vector<std::array<double,3>> points;
  for (int m0 = 0; m0 < period[0]; ++m0)
    for (int m1 = 0; m1 < period[1]; ++m1)
      for (int m2 = 0; m2 < period[2]; ++m2)
        points.push_back({(m0+0.5*shift[0])/period[0],(m1+0.5*shift[1])/period[1],(m2+0.5*shift[2])/period[2]});
  return points;
}

// Hermitian N x N matrices D(k) without a pattern at the points of a mesh:
// the real parts are symmetric, the imaginary parts antisymmetric.
struct MeshMatrices {
  std::vector<std::array<double,3>> points;
  std::vector<std::vector<double>> real, imaginary;
};

MeshMatrices mesh_matrices(std::size_t n, const std::vector<std::array<double,3>>& points, Numbers& random)
{
  MeshMatrices mesh{points,{},{}};
  for (std::size_t p = 0; p < points.size(); ++p) {
    mesh.real.push_back(dense_matrix(n,true,random));
    std::vector<double> imaginary(n*n,0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < i; ++j) {
        imaginary[i*n+j] = random.next();
        imaginary[j*n+i] = -imaginary[i*n+j];
      }
    mesh.imaginary.push_back(std::move(imaginary));
  }
  return mesh;
}

// The density matrix of the mesh in the blocks of a layout, equation (5):
// D_xy(R) = Re (1/N_k) sum_k e^{2 pi i k.R} D_xy(k).  It repeats with the
// period of the mesh, up to the sign of a shifted direction.
std::vector<double> mesh_blocks(const System& system, const BlockLayout& layout, const MeshMatrices& mesh)
{
  const std::vector<std::size_t> offset = offsets(system);
  const std::size_t n = offset.back();
  const double weight = 1.0/static_cast<double>(mesh.points.size());
  std::vector<double> blocks(layout.elements(),0.0);
  for (std::size_t b = 0; b < layout.keys.size(); ++b) {
    const BlockKey& key = layout.keys[b];
    const std::size_t x = static_cast<std::size_t>(key.row), y = static_cast<std::size_t>(key.column);
    const std::size_t rows = system.orbital_count[x], columns = system.orbital_count[y];
    double* target = blocks.data()+layout.offset[b];
    for (std::size_t p = 0; p < mesh.points.size(); ++p) {
      const std::array<double,3>& k = mesh.points[p];
      const double phase = two_pi*(k[0]*key.cell[0]+k[1]*key.cell[1]+k[2]*key.cell[2]);
      const double co = weight*std::cos(phase), si = weight*std::sin(phase);
      for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < columns; ++c) {
          const std::size_t e = (offset[x]+r)*n+offset[y]+c;
          target[r*columns+c] += co*mesh.real[p][e]-si*mesh.imaginary[p][e];
        }
    }
  }
  symmetrize_blocks(system,layout,blocks);
  require(block_asymmetry(system,layout,blocks) == 0.0,"the blocks of the density matrix of a mesh are not symmetric");
  return blocks;
}

// M(k) = sum_R e^{2 pi i k.R} M(R) of an array in a layout: the N x N real
// part, then the N x N imaginary part.
std::vector<double> bloch_sum(const System& system, const BlockLayout& layout, std::span<const double> values,
                              const std::array<double,3>& k)
{
  const std::vector<std::size_t> offset = offsets(system);
  const std::size_t n = offset.back();
  std::vector<double> sum(2*n*n,0.0);
  for (std::size_t b = 0; b < layout.keys.size(); ++b) {
    const BlockKey& key = layout.keys[b];
    const std::size_t x = static_cast<std::size_t>(key.row), y = static_cast<std::size_t>(key.column);
    const std::size_t rows = system.orbital_count[x], columns = system.orbital_count[y];
    const double phase = two_pi*(k[0]*key.cell[0]+k[1]*key.cell[1]+k[2]*key.cell[2]);
    const double co = std::cos(phase), si = std::sin(phase);
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < columns; ++c) {
        const double value = values[layout.offset[b]+r*columns+c];
        const std::size_t e = (offset[x]+r)*n+offset[y]+c;
        sum[e] += co*value;
        sum[n*n+e] += si*value;
      }
  }
  return sum;
}

std::array<int,3> negative_cell(const std::array<int,3>& cell)
{
  return {-cell[0],-cell[1],-cell[2]};
}

// The transpose of an na x nb block.
std::vector<double> transposed_block(const std::vector<double>& block, std::size_t na, std::size_t nb)
{
  std::vector<double> result(block.size());
  for (std::size_t p = 0; p < na; ++p)
    for (std::size_t q = 0; q < nb; ++q) result[q*na+p] = block[p*nb+q];
  return result;
}

// The V neighbours and V blocks of a folded system against those of the
// preparation.  Returns the numbers of V neighbours before and after.
std::array<std::size_t,2> check_folded_lists(const System& prepared, const System& folded,
                                             const std::array<int,3>& period, const std::string& name)
{
  const std::size_t atoms = prepared.orbital_count.size();
  require(folded.folding.period == period && folded.folding.prepared == prepared.v_neighbours &&
          folded.folding.index.size() == atoms,name+": the folding does not record the preparation");
  require(folded.c_neighbours == prepared.c_neighbours && folded.c_offset == prepared.c_offset &&
          same_values(folded.t,prepared.t) && folded.abf_count == prepared.abf_count,
          name+": the folding changes the C neighbours or the T blocks");
  double scale = 0.0;
  for (std::size_t a = 0; a < atoms; ++a)
    for (const std::vector<double>& block : prepared.v[a]) scale = std::max(scale,largest(block));
  const auto congruent = [&period](const std::array<int,3>& x, const std::array<int,3>& y) {
    for (std::size_t i = 0; i < 3; ++i)
      if ((x[i]-y[i])%period[i] != 0) return false;
    return true;
  };
  Worst sums(name+": folded V blocks against the sums of their members (relative)",1e-14);
  std::array<std::size_t,2> counts{};
  long single = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    const std::vector<Site>& list = folded.v_neighbours[a];
    const Site self{static_cast<int>(a),{0,0,0}};
    require(!list.empty() && list.front() == self && folded.v[a].size() == list.size(),
            name+": a folded list does not begin with the atom or lacks its blocks");
    for (std::size_t i = 2; i < list.size(); ++i)
      require(list[i-1].atom < list[i].atom || (list[i-1].atom == list[i].atom && list[i-1].cell < list[i].cell),
              name+": a folded list is not in the order of neighbour_lists()");
    const std::vector<std::size_t>& index = folded.folding.index[a];
    require(index.size() == prepared.v_neighbours[a].size(),name+": the folding lacks a V neighbour");
    std::vector<std::vector<std::size_t>> members(list.size());
    for (std::size_t j = 0; j < index.size(); ++j) {
      require(index[j] < list.size(),name+": a V neighbour folds onto a site that does not exist");
      const Site& from = prepared.v_neighbours[a][j];
      const Site& to = list[index[j]];
      require(from.atom == to.atom && congruent(from.cell,to.cell),
              name+": a V neighbour folds onto a site of another class");
      members[index[j]].push_back(j);
    }
    for (std::size_t i = 0; i < list.size(); ++i) {
      require(!members[i].empty(),name+": a folded V neighbour holds no block");
      require(std::any_of(members[i].begin(),members[i].end(),[&](std::size_t j) { return prepared.v_neighbours[a][j] == list[i]; }),
              name+": a folded V neighbour is none of the sites that it holds");
      std::vector<double> sum(folded.v[a][i].size(),0.0);
      for (const std::size_t j : members[i]) {
        require(prepared.v[a][j].size() == sum.size(),name+": a folded V block has another size than its members");
        for (std::size_t e = 0; e < sum.size(); ++e) sum[e] += prepared.v[a][j][e];
      }
      for (std::size_t e = 0; e < sum.size(); ++e) sums.add(std::fabs(folded.v[a][i][e]-sum[e])/scale);
      // A class of one member keeps its block, bit for bit.
      if (members[i].size() == 1) {
        require(same_values(folded.v[a][i],prepared.v[a][members[i].front()]),
                name+": a folded V neighbour of one member changes its block");
        ++single;
      }
      // The reversed site with the transpose, to the last bit.
      const std::size_t b = static_cast<std::size_t>(list[i].atom);
      const Site back{static_cast<int>(a),negative_cell(list[i].cell)};
      const std::size_t j = find_site(folded.v_neighbours[b],back);
      require(j < folded.v_neighbours[b].size(),name+": a folded V neighbour has no reversed site");
      require(same_values(folded.v[b][j],transposed_block(folded.v[a][i],folded.abf_count[a],folded.abf_count[b])),
              name+": the block of a reversed folded site is not the transpose");
    }
    counts[0] += prepared.v_neighbours[a].size();
    counts[1] += list.size();
  }
  sums.finish();
  std::printf("%s: %zu V neighbours fold onto %zu, %ld of them with one member\n",name.c_str(),counts[0],counts[1],single);
  return counts;
}

// What the folded columns of a system meet: two C neighbours of one atom
// whose cells agree modulo the period, which read one column of D (shared),
// and two sites of S(A) of one class that differ by an odd number of
// periods along the antiperiodic directions, whose columns differ in sign
// (negative).
struct ColumnCases {
  bool shared = false, negative = false;
};

ColumnCases column_cases(const System& system)
{
  const VFolding& folding = system.folding;
  const std::array<int,3>& n = folding.period;
  const auto congruent = [&n](const std::array<int,3>& x, const std::array<int,3>& y) {
    for (std::size_t i = 0; i < 3; ++i)
      if ((x[i]-y[i])%n[i] != 0) return false;
    return true;
  };
  const auto odd = [&](const std::array<int,3>& x, const std::array<int,3>& y) {
    int turns = 0;
    for (std::size_t i = 0; i < 3; ++i)
      if (folding.antiperiodic[i]) turns += (x[i]-y[i])/n[i];
    return turns%2 != 0;
  };
  ColumnCases cases;
  for (std::size_t a = 0; a < system.orbital_count.size(); ++a) {
    const std::vector<Site>& loc = system.c_neighbours[a];
    for (std::size_t i = 0; i < loc.size(); ++i)
      for (std::size_t j = i+1; j < loc.size(); ++j)
        if (loc[i].atom == loc[j].atom && congruent(loc[i].cell,loc[j].cell)) cases.shared = true;
    std::vector<Site> sites;                       // S(A)
    for (const Site& b : system.v_neighbours[a])
      for (const Site& c : system.c_neighbours[static_cast<std::size_t>(b.atom)])
        sites.push_back(Site{c.atom,{b.cell[0]+c.cell[0],b.cell[1]+c.cell[1],b.cell[2]+c.cell[2]}});
    for (std::size_t i = 0; i < sites.size() && !cases.negative; ++i)
      for (std::size_t j = i+1; j < sites.size(); ++j)
        if (sites[i].atom == sites[j].atom && congruent(sites[i].cell,sites[j].cell) &&
            odd(sites[i].cell,sites[j].cell)) {
          cases.negative = true;
          break;
        }
  }
  return cases;
}

// The sum over the atoms of the estimated time of a build.
double estimated_time(const System& system, const BuildCosts& cost)
{
  double time = 0.0;
  for (std::size_t a = 0; a < system.abf_count.size(); ++a)
    time += cost.per_abf[a]*static_cast<double>(system.abf_count[a])+cost.per_part[a];
  return time;
}

// The derivative of E_x with respect to the positions by the three routes:
// through the centre atoms, through the pairs with the weights of the
// centre atoms, and through the pairs with the half weights.  The pairs are
// taken in one part and in three.
std::vector<std::vector<double>> folding_gradients(const PositionGradient& derivative, const System& system,
                                                   const BlockLayout& layout, const std::vector<double>& density,
                                                   double alpha)
{
  const std::size_t atoms = system.orbital_count.size();
  std::size_t total = 0;
  for (const std::size_t count : system.abf_count) total += count;
  std::vector<std::vector<double>> result;
  result.emplace_back(3*atoms,0.0);
  for (std::size_t a = 0; a < atoms; ++a) derivative.add_blocks(layout,a,-alpha,density,result.back());
  BlockWeights whole(system), half(system);
  for (std::size_t a = 0; a < atoms; ++a)
    add_block_weights(centre_gradient_blocks(system,layout,a,-alpha,density),whole.values[a]);
  add_block_weights_blocks_half_abfs(system,layout,0,total,-alpha,density,half.arrays);
  for (const BlockWeights* weights : {&whole,&half})
    for (const std::size_t parts : {std::size_t{1},std::size_t{3}}) {
      result.emplace_back(3*atoms,0.0);
      for (std::size_t part = 0; part < parts; ++part)
        derivative.add_pairs(weights->view,PrepareShare{part,parts},result.back());
    }
  return result;
}

} // namespace

void exchange_folding(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  require(fixture.species.size() == 3 && fixture.positions.size() == 3,"the test is written for the three species of the fixture");
  const double alpha = 0.25;
  Numbers random;

  // The periodic structure of exx-exchange-system: the cell is smaller than
  // the cutoffs, so the V neighbours hold many images of every atom.
  const int intervals = 300;
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(24.0,240);
  SystemOptions options;
  options.interaction = short_range_kernel(0.11);
  options.v_cutoff = 3.0;
  const std::vector<Shells> few{select(fixture.species[0],{0,2}),select(fixture.species[1],{0,1,2}),
                                select(fixture.species[2],{0})};
  const std::vector<SpeciesTable> tables = tables_of(few,intervals,reciprocal,1.0e-3);
  Structure structure;
  structure.lattice = fixture.lattice;
  structure.positions = fixture.positions;
  structure.species = fixture.atom_species;
  const System prepared = prepare_system(structure,tables,reciprocal,options);
  const std::size_t atoms = structure.positions.size(), n = orbital_total(prepared);
  std::size_t total = 0;
  for (const std::size_t count : prepared.abf_count) total += count;
  const PositionGradient prepared_derivative(structure,tables,reciprocal,options,prepared);
  const BlockLayout prepared_layout = exchange_layout(prepared);
  int reach = 0;
  for (const std::vector<Site>& list : prepared.v_neighbours)
    for (const Site& site : list)
      for (const int c : site.cell) reach = std::max(reach,std::abs(c));
  require(reach >= 1,"the V neighbours are expected to reach the neighbouring cells");

  // 1. What the folding rejects leaves the system as it was.
  {
    const auto unchanged = [&prepared](const System& system) {
      return system.v_neighbours == prepared.v_neighbours && same_values(system.v,prepared.v) &&
             system.folding.prepared.empty() && system.folding.index.empty() &&
             system.folding.period == std::array<int,3>{};
    };
    System system = prepared;
    expect_invalid("a period of zero cells",[&] { fold_v_neighbours(system,structure,{2,0,2}); });
    expect_invalid("a negative period",[&] { fold_v_neighbours(system,structure,{1,1,-2}); });
    Structure fewer = structure;
    fewer.positions.pop_back();
    fewer.species.pop_back();
    expect_invalid("a structure with fewer atoms",[&] { fold_v_neighbours(system,fewer,{2,2,2}); });
    require(unchanged(system),"a rejected folding changes the system");
    System shorter = prepared;
    shorter.v[0][1].pop_back();
    expect_invalid("a V block of the wrong size",[&] { fold_v_neighbours(shorter,structure,{2,2,2}); });
    // Lists that are not each other's reversed sites.
    System lopsided = prepared;
    lopsided.v_neighbours[0].pop_back();
    lopsided.v[0].pop_back();
    const System before = lopsided;
    bool thrown = false;
    try { fold_v_neighbours(lopsided,structure,{2,2,2}); } catch (const std::logic_error&) { thrown = true; }
    require(thrown && lopsided.v_neighbours == before.v_neighbours && same_values(lopsided.v,before.v) &&
            lopsided.folding.prepared.empty(),"lists without their reversed sites are folded, or the failure changes them");
    // A folded system is not folded again.
    fold_v_neighbours(system,structure,{2,2,2});
    const System once = system;
    expect_invalid("a folding of a folded system",[&] { fold_v_neighbours(system,structure,{2,2,2}); });
    require(system.v_neighbours == once.v_neighbours && same_values(system.v,once.v) &&
            system.folding.prepared == once.folding.prepared && system.folding.index == once.folding.index,
            "a rejected second folding changes the system");
    // The columns are folded by the period of folded V neighbours, once.
    System unfolded = prepared;
    expect_invalid("a folding of the columns without folded V neighbours",
                   [&] { fold_columns(unfolded,{false,false,false}); });
    require(unfolded.folding.prepared.empty() && !unfolded.folding.columns,
            "a rejected folding of the columns changes the system");
    fold_columns(system,{true,false,false});
    expect_invalid("a second folding of the columns",[&] { fold_columns(system,{false,true,false}); });
    require(system.folding.columns && system.folding.antiperiodic == std::array<bool,3>{true,false,false} &&
            system.folding.period == once.folding.period,"a rejected second folding of the columns changes the system");
  }

  // 2. A period longer than twice the reach of the lists leaves every class
  // with one member: the system keeps its lists and blocks, bit for bit.
  {
    System system = prepared;
    const int period = 2*reach+1;
    fold_v_neighbours(system,structure,{period,period,period});
    require(system.v_neighbours == prepared.v_neighbours && same_values(system.v,prepared.v),
            "a folding with classes of one member changes the system");
    for (std::size_t a = 0; a < atoms; ++a)
      for (std::size_t j = 0; j < system.folding.index[a].size(); ++j)
        require(system.folding.index[a][j] == j,"a folding with classes of one member moves a block");
    check_folded_lists(prepared,system,{period,period,period},"period "+std::to_string(period));
  }

  // 2b. A period longer than twice the reach of S(A) leaves every class of
  // columns with one site: the builds and the derivatives of a system with
  // folded columns have the bits of the preparation.
  {
    int columns_reach = 0;
    for (std::size_t a = 0; a < atoms; ++a)
      for (const Site& b : prepared.v_neighbours[a])
        for (const Site& c : prepared.c_neighbours[static_cast<std::size_t>(b.atom)])
          for (std::size_t i = 0; i < 3; ++i) columns_reach = std::max(columns_reach,std::abs(b.cell[i]+c.cell[i]));
    const int period = 2*columns_reach+1;
    System system = prepared;
    fold_v_neighbours(system,structure,{period,period,period});
    fold_columns(system,{true,true,true});
    const BlockLayout layout = exchange_layout(system);
    require(layout.keys == prepared_layout.keys && layout.offset == prepared_layout.offset,
            "a folding with classes of one member changes the layout");
    std::vector<double> d(layout.elements());
    for (double& value : d) value = random.next();
    symmetrize_blocks(system,layout,d);
    std::vector<double> half(layout.elements(),0.0), prepared_half(layout.elements(),0.0);
    add_exchange_blocks_half_abfs(system,layout,0,total,-alpha,d,half);
    add_exchange_blocks_half_abfs(prepared,prepared_layout,0,total,-alpha,d,prepared_half);
    require(same_values(layout_exchange(system,layout,d,alpha),layout_exchange(prepared,prepared_layout,d,alpha)) &&
            same_values(half,prepared_half),"columns folded into classes of one site change the bits of K");
    for (std::size_t a = 0; a < atoms; ++a) {
      const CentreGradient folded = half_gradient_blocks(system,layout,a,0,system.abf_count[a],-alpha,d);
      const CentreGradient plain = half_gradient_blocks(prepared,prepared_layout,a,0,prepared.abf_count[a],-alpha,d);
      require(same_values(folded.t,plain.t) && same_values(folded.v,plain.v) && same_values(folded.w,plain.w) &&
              same_values(centre_gradient_blocks(system,layout,a,-alpha,d).t,
                          centre_gradient_blocks(prepared,prepared_layout,a,-alpha,d).t),
              "columns folded into classes of one site change the bits of the derivatives");
    }
    const BuildCosts folded_cost = build_costs(system,true,true), plain_cost = build_costs(prepared,true,true);
    require(folded_cost.per_abf == plain_cost.per_abf && folded_cost.per_part == plain_cost.per_part,
            "columns folded into classes of one site change the estimates of the build");
  }

  // 3. Meshes with and without shifts.  For a density matrix of the points
  // of the mesh the folded system gives K(k) at these points, E_x and its
  // derivative with respect to the positions of the preparation, within
  // rounding, in the whole and the half build; K(k) between the points of
  // the mesh differs.  The Gamma form folds all images anyway and keeps its
  // matrix.
  struct Mesh {
    std::array<int,3> period, shift;
  };
  const Mesh meshes[] = {{{2,2,2},{0,0,0}},{{2,2,2},{1,1,1}},{{3,2,1},{0,1,0}},{{1,2,3},{1,0,0}},{{1,1,1},{0,0,0}}};
  const std::vector<double> dense = dense_matrix(n,true,random);
  const std::vector<double> prepared_gamma = exchange_gamma(prepared,dense,alpha);
  bool halves_checked = false;
  std::size_t devices_checked = 0;
  double v_only = 0.0;                     // the estimated time with the V neighbours of the mesh folded only
  for (const Mesh& mesh : meshes) for (const bool columns : {false,true}) {
    const std::array<bool,3> antiperiodic{mesh.shift[0] == 1,mesh.shift[1] == 1,mesh.shift[2] == 1};
    const std::string name = "mesh "+std::to_string(mesh.period[0])+"x"+std::to_string(mesh.period[1])+"x"+
                             std::to_string(mesh.period[2])+" shift "+std::to_string(mesh.shift[0])+
                             std::to_string(mesh.shift[1])+std::to_string(mesh.shift[2])+
                             (columns ? ", columns folded" : "");
    System folded = prepared;
    fold_v_neighbours(folded,structure,mesh.period);
    const BlockLayout layout = exchange_layout(folded);
    if (columns) {
      fold_columns(folded,antiperiodic);
      const BlockLayout same = exchange_layout(folded);
      require(same.keys == layout.keys && same.offset == layout.offset,name+": the folding of the columns changes the layout");
    }
    const std::array<std::size_t,2> counts = check_folded_lists(prepared,folded,mesh.period,name);
    require(counts[1] < counts[0],name+": the folding is expected to reduce the V neighbours");

    const MeshMatrices matrices = mesh_matrices(n,mesh_points(mesh.period,mesh.shift),random);
    const std::vector<double> prepared_d = mesh_blocks(prepared,prepared_layout,matrices);
    const std::vector<double> folded_d = mesh_blocks(folded,layout,matrices);
    const std::vector<double> prepared_k = layout_exchange(prepared,prepared_layout,prepared_d,alpha);
    const std::vector<double> folded_k = layout_exchange(folded,layout,folded_d,alpha);
    std::vector<double> folded_half(layout.elements(),0.0);
    add_exchange_blocks_half_abfs(folded,layout,0,total,-alpha,folded_d,folded_half);
    mirror_blocks(folded,layout,folded_half);

    std::vector<std::vector<double>> reference;
    double scale = 0.0;
    for (const std::array<double,3>& k : matrices.points) {
      reference.push_back(bloch_sum(prepared,prepared_layout,prepared_k,k));
      scale = std::max(scale,largest(reference.back()));
    }
    require(scale > 1.0e-6,name+": K(k) is expected to be of order one");
    Worst whole(name+": K(k) of the whole build at the points (relative)",1e-12);
    Worst half(name+": K(k) of the half build at the points (relative)",1e-12);
    for (std::size_t p = 0; p < matrices.points.size(); ++p) {
      const std::vector<double> a = bloch_sum(folded,layout,folded_k,matrices.points[p]);
      const std::vector<double> b = bloch_sum(folded,layout,folded_half,matrices.points[p]);
      for (std::size_t e = 0; e < a.size(); ++e) {
        whole.add(std::fabs(a[e]-reference[p][e])/scale);
        half.add(std::fabs(b[e]-reference[p][e])/scale);
      }
    }
    whole.finish();
    half.finish();
    // Between the points of the mesh K(k) changes.
    {
      const std::array<double,3> between{0.5/mesh.period[0]+0.11,0.5/mesh.period[1]+0.07,0.5/mesh.period[2]+0.13};
      const std::vector<double> a = bloch_sum(folded,layout,folded_k,between);
      const std::vector<double> b = bloch_sum(prepared,prepared_layout,prepared_k,between);
      double difference = 0.0;
      for (std::size_t e = 0; e < a.size(); ++e) difference = std::max(difference,std::fabs(a[e]-b[e]));
      require(difference > 1.0e-6*scale,name+": K(k) between the points of the mesh is expected to change");
    }
    const double energy = exchange_energy_gamma(prepared_d,prepared_k);
    require(std::fabs(energy) > 1.0e-6,name+": E_x is expected to differ from zero");
    Worst energies(name+": E_x of the whole and the half build (relative)",1e-12);
    energies.relative(exchange_energy_gamma(folded_d,folded_k),energy);
    energies.relative(exchange_energy_gamma(folded_d,folded_half),energy);
    energies.finish();

    // The derivative with respect to the positions.
    const PositionGradient derivative(structure,tables,reciprocal,options,folded);
    const std::vector<std::vector<double>> exact =
        folding_gradients(prepared_derivative,prepared,prepared_layout,prepared_d,alpha);
    const std::vector<std::vector<double>> routes = folding_gradients(derivative,folded,layout,folded_d,alpha);
    const double steepest = largest(exact.front());
    require(steepest > 1.0e-6,name+": the derivative is expected to differ from zero");
    Worst gradients(name+": dE_x/dtau of five routes (relative)",1e-12);
    for (const std::vector<double>& route : routes)
      for (std::size_t e = 0; e < 3*atoms; ++e) gradients.add(std::fabs(route[e]-exact.front()[e])/steepest);
    gradients.finish();

    // The Gamma form of the folded system.
    Worst gamma(name+": K of the Gamma form (relative)",1e-12);
    const std::vector<double> folded_gamma = exchange_gamma(folded,dense,alpha);
    const double gamma_scale = largest(prepared_gamma);
    for (std::size_t e = 0; e < n*n; ++e) gamma.add(std::fabs(folded_gamma[e]-prepared_gamma[e])/gamma_scale);
    gamma.finish();

    // The estimated time of the half build of the site form.
    const double before = estimated_time(prepared,build_costs(prepared,true,true));
    const double after = estimated_time(folded,build_costs(folded,true,true));
    require(after < before,name+": the folded system is expected to take less time");
    require(!columns || after < v_only,name+": the folded columns are expected to take less time");
    if (!columns) v_only = after;
    std::printf("%s: the half build of the site form takes %.3f of the estimated time of the preparation\n",
                name.c_str(),after/before);

    // The half build of a system with folded columns gives X with other
    // blocks than the whole build: K(k) at the points above holds.
    if (!columns && mesh.period == std::array<int,3>{2,2,2} && !halves_checked) {
      check_half_build(folded,name+", folded",random,false);
      halves_checked = true;
    }

    // The sign of the antiperiodic directions: without it K(k) is wrong.
    if (columns && mesh.shift != std::array<int,3>{}) {
      System unsigned_columns = prepared;
      fold_v_neighbours(unsigned_columns,structure,mesh.period);
      fold_columns(unsigned_columns,{false,false,false});
      const std::vector<double> wrong = layout_exchange(unsigned_columns,layout,folded_d,alpha);
      double difference = 0.0;
      for (std::size_t p = 0; p < matrices.points.size(); ++p) {
        const std::vector<double> a = bloch_sum(folded,layout,wrong,matrices.points[p]);
        for (std::size_t e = 0; e < a.size(); ++e) difference = std::max(difference,std::fabs(a[e]-reference[p][e]));
      }
      require(difference > 1.0e-6*scale,name+": the columns folded without their signs are expected to give another K(k)");
    }

    // The device against the host, with signs and with columns of D that
    // two columns of the T blocks share (the period 1 along a_1).
    if (columns && (mesh.period == std::array<int,3>{2,2,2} || mesh.period == std::array<int,3>{1,2,3}) &&
        mesh.shift != std::array<int,3>{}) {
      const ColumnCases cases = column_cases(folded);
      require(cases.negative && (mesh.period[0] != 1 || cases.shared),
              name+": the device test is expected to meet negative columns, and shared ones with the period 1");
      check_device_forms(folded,name,random,false);
      ++devices_checked;
    }

    // 4. With the period 1 the site form of the Gamma point, D_xy(R) = D_xy,
    // gives the Gamma form of the preparation: the matrix, E_x and the
    // derivative.
    if (mesh.period == std::array<int,3>{1,1,1}) {
      for (std::size_t a = 0; a < atoms; ++a)
        require(folded.v_neighbours[a].size() <= atoms,name+": the period 1 leaves more than one site of an atom");
      const std::vector<double> d = layout_values(layout,gamma_density(folded,dense));
      const std::vector<double> k = layout_exchange(folded,layout,d,alpha);
      const std::vector<double> at_gamma = bloch_sum(folded,layout,k,{0.0,0.0,0.0});
      Worst site(name+": site form at Gamma against the Gamma form (relative)",1e-12);
      for (std::size_t e = 0; e < n*n; ++e) site.add(std::fabs(at_gamma[e]-prepared_gamma[e])/gamma_scale);
      site.relative(exchange_energy_gamma(d,k),exchange_energy_gamma(dense,prepared_gamma));
      site.finish();
      std::vector<double> gamma_gradient(3*atoms,0.0), site_gradient(3*atoms,0.0);
      for (std::size_t a = 0; a < atoms; ++a) {
        prepared_derivative.add_gamma(a,-alpha,dense,gamma_gradient);
        derivative.add_blocks(layout,a,-alpha,d,site_gradient);
      }
      const double gamma_steepest = largest(gamma_gradient);
      Worst forms(name+": derivative of the site form at Gamma against the Gamma form (relative)",1e-12);
      for (std::size_t e = 0; e < 3*atoms; ++e) forms.add(std::fabs(site_gradient[e]-gamma_gradient[e])/gamma_steepest);
      forms.finish();
    }
  }
  require(halves_checked,"the half build of a folded system was not checked");
  require(devices_checked == 2,"the device was not checked with folded columns");
}

} // namespace exx_test
