// Tests of exchange.hpp:
//   exx-exchange-fixture   the blocks of K and E_x of the site form against
//                          the fixture of the reference implementation
//                          (tools/exx_reference.py), and equation (15)
//   exx-exchange-gamma     the Gamma form against the sum over R of the blocks
//                          of the site form and against the definition with
//                          dense matrices
//   exx-exchange-gradient  equations (16) and (17) against central differences
//   exx-exchange-backend   the same results with a table of matrix products
//   exx-exchange-system    T and V blocks made from PAOs: isolated atoms
//                          against a formula without T blocks, a periodic
//                          structure against the definition, and the C
//                          interface against the C++ interface
#include <openmx/exx/abf.hpp>
#include <openmx/exx/exchange.hpp>
#include <openmx/exx/exx.h>
#include <openmx/exx/system.hpp>

#include "kbuild_fixture.hpp"
#include "overlap_fixture.hpp"
#include "test_dgemm.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
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
  const OpenMXExxDenseBackend table{&counting_dgemm,&calls};
  const DenseBackend products(&table);
  // One centre atom takes two products of its own and five for every V
  // neighbour: every sum over the ABFs is one product.  The Gamma form adds
  // up the V blocks of the images of an atom first, so it counts the atoms
  // among the V neighbours.
  long expected = 0, expected_gamma = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    std::set<int> distinct;
    for (const Site& site : system.v_neighbours[a]) distinct.insert(site.atom);
    expected += 2+5*static_cast<long>(system.v_neighbours[a].size());
    expected_gamma += 2+5*static_cast<long>(distinct.size());
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
    const OpenMXExxDenseBackend failing{&failing_dgemm,nullptr};
    const DenseBackend broken(&failing);
    expect_runtime_error("a failing table in the site form",[&] { (void)exchange_blocks(system,density,alpha,broken); });
    expect_runtime_error("a failing table in the Gamma form",[&] { (void)exchange_gamma(system,d,alpha,broken); });
    expect_runtime_error("a failing table in the gradient",[&] { (void)exchange_gradient_gamma(system,d,alpha,broken); });
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
  require(openmx_exx_add_gamma(state,0,openmx_exx_atom_count(state),-alpha,density.data(),k.data(),nullptr,message,
                               sizeof message) == 0,std::string("the build through the C interface fails: ")+message);
  require(message[0] == '\0',"a successful call leaves a message");
  return k;
}

// The C interface must describe the same system as the C++ interface and give
// the same exchange matrix, bit for bit: it runs the same code on the same
// numbers.
void compare_with_c(const OpenMXExxState* state, const System& system, const std::vector<double>& density, double alpha,
                    const std::string& what)
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
    require(openmx_exx_prepare(state,&c_options,&geometry.geometry,message,sizeof message) == 0,
            std::string("the preparation through the C interface fails: ")+message);
    compare_with_c(state,system,d,alpha,"HSE06 kernel");

    // Another kernel with the same species: the tables of the species are kept
    // and the blocks are made again.
    c_options.kernel = OPENMX_EXX_KERNEL_TRUNCATED_COULOMB;
    c_options.kernel_parameter = 12.0;
    require(openmx_exx_prepare(state,&c_options,&geometry.geometry,message,sizeof message) == 0,
            std::string("the second preparation fails: ")+message);
    SystemOptions truncated = options;
    truncated.interaction = truncated_coulomb_kernel(12.0);
    compare_with_c(state,prepare_system(structure,tables,reciprocal,truncated),d,alpha,"truncated Coulomb kernel");

    // Another structure.
    Structure moved = structure;
    moved.positions[1][0] += 0.4;
    moved.positions[2][2] -= 0.3;
    const CGeometry moved_geometry(moved,few);
    require(openmx_exx_prepare(state,&c_options,&moved_geometry.geometry,message,sizeof message) == 0,
            std::string("the preparation of a moved structure fails: ")+message);
    compare_with_c(state,prepare_system(moved,tables,reciprocal,truncated),d,alpha,"moved atoms");

    // Another threshold: the tables of the species are made again.
    c_options.abf_threshold = 1.0e-2;
    require(openmx_exx_prepare(state,&c_options,&moved_geometry.geometry,message,sizeof message) == 0,
            std::string("the preparation with another threshold fails: ")+message);
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
    compare_with_c(state,prepare_system(moved,coarse,reciprocal,truncated),d,alpha,"after two failures");
  }
}

} // namespace exx_test
