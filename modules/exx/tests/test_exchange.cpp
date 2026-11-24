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
//                          product that fails inside a piece
//   exx-exchange-forces    forces.hpp: the derivative of E_x with respect to
//                          the positions of the atoms against central
//                          differences of E_x of structures with moved atoms
#include <openmx/exx/abf.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/exx.h>
#include <openmx/exx/forces.hpp>
#include <openmx/exx/system.hpp>

#include "kbuild_fixture.hpp"
#include "overlap_fixture.hpp"
#include "test_dgemm.hpp"
#include "test_parallel.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
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
    const OpenMXExxDenseBackend threaded{nullptr,nullptr,&threaded_loop,3};
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
    const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0};
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
    // Two products for the centre, and for every V neighbour two and four for
    // every ABF of the centre.
    long expected = 0;
    for (std::size_t a = 0; a < atoms; ++a)
      expected += 2+static_cast<long>(system.v_neighbours[a].size())*(2+4*static_cast<long>(system.abf_count[a]));
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
  const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0};
  const DenseBackend products(&table);
  // One centre atom takes two products of its own.  Every V neighbour adds
  // one product for W and four for every ABF of the centre: the four terms
  // are computed for one ABF at a time.  The Gamma form adds up the V blocks
  // of the images of an atom first, so it counts the atoms among the V
  // neighbours.
  long expected = 0, expected_gamma = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    std::set<int> distinct;
    for (const Site& site : system.v_neighbours[a]) distinct.insert(site.atom);
    const long each = 1+4*static_cast<long>(system.abf_count[a]);
    expected += 2+each*static_cast<long>(system.v_neighbours[a].size());
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
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr,nullptr,0};
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
      const OpenMXExxDenseBackend pieces{product ? &shared_dgemm : nullptr,&shared,loop,threads};
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
        const OpenMXExxDenseBackend table{&piece_failing_dgemm,&failing,loop,3};
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

    // The derivatives do not divide their work: with a table that has a loop
    // they give the bits of the same table without the loop, and the loop is
    // not called.
    for (const bool product : {false,true}) {
      const DenseBackend whole = with(product,nullptr,0), pieces = with(product,&threaded_loop,3);
      shared.reset();
      const ExchangeGradient all = exchange_gradient_gamma(system,d,alpha,whole);
      const ExchangeGradient again = exchange_gradient_gamma(system,d,alpha,pieces);
      bool same = same_values(all.t,again.t) && same_values(all.v,again.v);
      for (std::size_t a = 0; a < atoms; ++a) {
        const CentreGradient gamma = centre_gradient_gamma(system,a,-alpha,d,whole);
        const CentreGradient gamma_again = centre_gradient_gamma(system,a,-alpha,d,pieces);
        const CentreGradient site = centre_gradient_blocks(system,layout,a,-alpha,blocks_of_d,whole);
        const CentreGradient site_again = centre_gradient_blocks(system,layout,a,-alpha,blocks_of_d,pieces);
        same = same && same_values(gamma.t,gamma_again.t) && same_values(gamma.v,gamma_again.v) &&
               same_values(site.t,site_again.t) && same_values(site.v,site_again.v);
      }
      require(same,"a table with a loop changes the derivatives");
      require(shared.loops == 0,"the derivatives call the loop of the table");
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
        const OpenMXExxDenseBackend table{product ? &shared_dgemm : nullptr,&shared,loop,threads};
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

    // A table with a product and a loop, the pieces on threads of their own:
    // the C interface runs the same code on the same numbers as the C++
    // interface in both forms, and the site form agrees within rounding with
    // the numbers of the same products in one piece.
    {
      Calls shared;
      const OpenMXExxDenseBackend table{&shared_dgemm,&shared,&threaded_loop,4};
      const OpenMXExxDenseBackend whole{&shared_dgemm,&shared,nullptr,0};
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
      const OpenMXExxDenseBackend broken{&piece_failing_dgemm,&failing,&threaded_loop,4};
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
    const OpenMXExxDenseBackend table{&counting_dgemm,&calls,nullptr,0};
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
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr,nullptr,0};
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
  }
}

} // namespace exx_test
