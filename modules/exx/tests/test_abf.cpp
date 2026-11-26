// exx-abf-pca: the auxiliary basis functions made from the PAOs of the overlap
// fixture (Si s2p2d1f1, C s2p2d1, H s2p1).
#include <openmx/exx/abf.hpp>

#include "overlap_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

namespace {

bool couples(int l1, int l2, int angular)
{
  return angular >= std::abs(l1-l2) && angular <= l1+l2 && (l1+l2+angular)%2 == 0;
}

// The products of the orbitals that have a component with one angular momentum.
std::vector<std::vector<double>> products(const std::vector<RadialFunction>& orbitals, int angular)
{
  std::vector<std::vector<double>> list;
  for (std::size_t i = 0; i < orbitals.size(); ++i)
    for (std::size_t j = i; j < orbitals.size(); ++j) {
      if (!couples(orbitals[i].angular,orbitals[j].angular,angular)) continue;
      std::vector<double> g(orbitals[i].values.size());
      for (std::size_t k = 0; k < g.size(); ++k) g[k] = orbitals[i].values[k]*orbitals[j].values[k];
      list.push_back(std::move(g));
    }
  return list;
}

template <class Call>
void expect_invalid(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const std::invalid_argument&) { thrown = true; }
  require(thrown,"no std::invalid_argument for "+what);
}

} // namespace

void abf_pca(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);

  Worst orthonormal("orthonormality of the ABFs",1e-12);
  Worst trace("sum of the eigenvalues / norms of the products",1e-12);
  Worst residual("projection residual / discarded eigenvalues",1e-8);
  Worst complete("projection residual when nothing is discarded",1e-20);
  Worst same_values("eigenvalues from reordered, sign-flipped orbitals",1e-12);
  Worst same_functions("overlap of the ABFs of the two orders",1e-12);
  Worst nested("ABFs of a larger threshold inside a smaller one",0.0);
  long discarded_cases = 0;

  for (const FixtureSpecies& species : fixture.species) {
    const RadialGrid grid{0.0,species.cutoff/fixture.radial_intervals,fixture.radial_intervals};
    const std::vector<RadialFunction> orbitals = sample_shells(species,grid);
    int maximum = 0;
    for (const RadialFunction& f : orbitals) maximum = std::max(maximum,f.angular);

    // The same orbitals in the opposite order, every second one with the other sign.
    std::vector<RadialFunction> changed(orbitals.rbegin(),orbitals.rend());
    for (std::size_t i = 0; i < changed.size(); i += 2)
      for (double& value : changed[i].values) value = -value;

    AuxiliaryBasis previous;
    for (const double threshold : {1e-2,1e-4,1e-6,1e-8}) {
      const AuxiliaryBasis basis = make_auxiliary_basis(grid,orbitals,threshold);
      const AuxiliaryBasis other = make_auxiliary_basis(grid,changed,threshold);
      require(other.functions.size() == basis.functions.size() && other.channels.size() == basis.channels.size(),
              "the number of ABFs depends on the order of the orbitals");

      std::printf("%-6s threshold %.0e: %3zu radial functions, %3zu ABFs;  kept/products by L:",
                  species.name.c_str(),threshold,basis.functions.size(),basis.components());
      std::size_t first = 0, counted = 0, components = 0;
      int expected_angular = 0;
      for (std::size_t c = 0; c < basis.channels.size(); ++c) {
        const AbfChannel& channel = basis.channels[c];
        const int angular = channel.angular;
        std::printf(" %zu/%zu",channel.kept,channel.eigenvalues.size());
        require(angular >= expected_angular && angular <= 2*maximum,"the channels are not in ascending order");
        expected_angular = angular+1;

        const std::vector<std::vector<double>> g = products(orbitals,angular);
        require(g.size() == channel.eigenvalues.size() && !g.empty(),"wrong number of products in a channel");
        require(channel.kept >= 1 && channel.kept <= g.size(),"wrong number of kept components");
        double norms = 0.0, all = 0.0, discarded = 0.0;
        for (const auto& product : g) norms += radial_integral(grid,product,product);
        for (std::size_t n = 0; n < channel.eigenvalues.size(); ++n) {
          const double value = channel.eigenvalues[n];
          require(value >= 0.0 && (n == 0 || value <= channel.eigenvalues[n-1]),"the eigenvalues are not in decreasing order");
          require((value > threshold*channel.eigenvalues[0]) == (n < channel.kept),"the threshold was not applied");
          all += value;
          if (n >= channel.kept) discarded += value;
          same_values.add(std::fabs(value-other.channels[c].eigenvalues[n])/channel.eigenvalues[0]);
        }
        trace.relative(all,norms);
        require(other.channels[c].kept == channel.kept,"the number of kept components depends on the order");

        // Orthonormality, the sign, and the agreement of the two orders.
        for (std::size_t n = 0; n < channel.kept; ++n) {
          const RadialFunction& f = basis.functions[first+n];
          require(f.angular == angular && f.values.size() == grid.size(),"an ABF is in the wrong channel");
          require(radial_moment(grid,f.values,angular) > 0.0,"the moment of an ABF is not positive");
          for (std::size_t m = 0; m <= n; ++m)
            orthonormal.absolute(radial_integral(grid,f.values,basis.functions[first+m].values),n == m ? 1.0 : 0.0);
          same_functions.absolute(radial_integral(grid,f.values,other.functions[first+n].values),1.0);
        }

        // sum_k |g_k - sum_n f_n <f_n|g_k>|^2 = sum of the discarded eigenvalues.
        double left_over = 0.0;
        for (const auto& product : g) {
          std::vector<double> rest = product;
          for (std::size_t n = 0; n < channel.kept; ++n) {
            const std::vector<double>& f = basis.functions[first+n].values;
            const double coefficient = radial_integral(grid,f,product);
            for (std::size_t k = 0; k < rest.size(); ++k) rest[k] -= coefficient*f[k];
          }
          left_over += radial_integral(grid,rest,rest);
        }
        if (channel.kept == g.size()) {
          complete.add(left_over/norms);
        } else if (discarded > 1e-12*norms) {
          // Below this ratio the subtraction above loses the digits that the
          // relative limit asks for.
          residual.relative(left_over,discarded);
          ++discarded_cases;
        } else {
          require(left_over <= 2e-12*norms,"the projection residual exceeds the discarded eigenvalues");
        }

        // The ABFs of the previous (larger) threshold are the first ones here.
        if (!previous.channels.empty()) {
          std::size_t before = 0;
          for (std::size_t p = 0; p < c; ++p) before += previous.channels[p].kept;
          require(previous.channels[c].kept <= channel.kept,"a smaller threshold kept fewer components");
          for (std::size_t n = 0; n < previous.channels[c].kept; ++n)
            for (std::size_t k = 0; k < grid.size(); ++k)
              nested.absolute(basis.functions[first+n].values[k],previous.functions[before+n].values[k]);
        }

        first += channel.kept;
        counted += channel.kept;
        components += channel.kept*static_cast<std::size_t>(2*angular+1);
      }
      std::printf("\n");
      require(counted == basis.functions.size() && components == basis.components(),"the counts of the ABFs disagree");
      previous = basis;
    }
  }
  orthonormal.finish();
  trace.finish();
  residual.finish();
  complete.finish();
  same_values.finish();
  same_functions.finish();
  nested.finish();
  require(discarded_cases >= 10,"too few channels with discarded components");

  // The count of section 6.2 of the guide for s2p2d1: 22 radial products, 78 ABFs.
  {
    const FixtureSpecies* carbon = nullptr;
    for (const FixtureSpecies& species : fixture.species)
      if (species.shells.size() == 5) carbon = &species;
    require(carbon != nullptr,"the fixture has no species with five shells");
    const RadialGrid grid{0.0,carbon->cutoff/fixture.radial_intervals,fixture.radial_intervals};
    const AuxiliaryBasis basis = make_auxiliary_basis(grid,sample_shells(*carbon,grid),1e-4);
    const std::vector<std::size_t> expected{7,6,6,2,1};
    require(basis.channels.size() == expected.size(),"s2p2d1 must have products with L = 0..4");
    std::size_t radial = 0, all = 0;
    for (std::size_t c = 0; c < expected.size(); ++c) {
      require(basis.channels[c].angular == static_cast<int>(c) && basis.channels[c].eigenvalues.size() == expected[c],
              "wrong number of products of s2p2d1");
      radial += expected[c];
      all += expected[c]*(2*c+1);
    }
    require(radial == 22 && all == 78,"the count of the products of s2p2d1 changed");
  }

  // sample_orbitals() against the interpolation of the main program, which
  // the other tests take from the basis package directly.
  {
    const FixtureSpecies& species = fixture.species.front();
    const RadialGrid grid{0.0,species.cutoff/200,200};
    std::vector<MeshFunction> shells;
    for (const FixtureShell& shell : species.shells) shells.push_back({shell.l,shell.values});
    const std::vector<RadialFunction> sampled = sample_orbitals(grid,species.radii,shells);
    const std::vector<RadialFunction> reference = sample_shells(species,grid);
    require(sampled.size() == reference.size() && !sampled.empty(),"sample_orbitals returns the wrong number of functions");
    for (std::size_t k = 0; k < sampled.size(); ++k)
      require(sampled[k].angular == reference[k].angular && sampled[k].values == reference[k].values,
              "sample_orbitals differs from the interpolation of the main program");
    // Below the first mesh point and beyond the last one.
    require(species.radii.front() > 0.0 && sampled[0].angular == 0 && sampled[0].values[0] == species.shells[0].values[0],
            "an s function must keep its first value at the origin");
    const RadialGrid wide{0.0,(species.radii.back()+1.0)/50,50};
    for (const RadialFunction& f : sample_orbitals(wide,species.radii,shells)) {
      require(f.values.back() == 0.0,"a function must vanish beyond its mesh");
      require(f.angular == 0 || f.values[0] == 0.0,"a function with l > 0 must vanish at the origin");
    }
    std::vector<double> unordered(species.radii);
    std::swap(unordered[10],unordered[11]);
    expect_invalid("a mesh that does not increase",[&] { sample_orbitals(grid,unordered,shells); });
    std::vector<MeshFunction> shorter = shells;
    shorter[1].values = shorter[1].values.first(shorter[1].values.size()-1);
    expect_invalid("a function that does not match the mesh",[&] { sample_orbitals(grid,species.radii,shorter); });
    expect_invalid("a grid without a step",[&] { sample_orbitals(RadialGrid{0.0,0.0,200},species.radii,shells); });
    require(sample_orbitals(grid,species.radii,std::vector<MeshFunction>{}).empty(),"no functions must give no functions");
  }

  // Invalid arguments.
  {
    const RadialGrid grid{0.0,0.01,400};
    std::vector<RadialFunction> one{RadialFunction{0,std::vector<double>(grid.size(),1.0)}};
    expect_invalid("a threshold of zero",[&] { make_auxiliary_basis(grid,one,0.0); });
    expect_invalid("a threshold of one",[&] { make_auxiliary_basis(grid,one,1.0); });
    expect_invalid("no orbitals",[&] { make_auxiliary_basis(grid,std::vector<RadialFunction>{},1e-4); });
    std::vector<RadialFunction> wrong_size{RadialFunction{0,std::vector<double>(grid.size()-1,1.0)}};
    expect_invalid("a function of the wrong size",[&] { make_auxiliary_basis(grid,wrong_size,1e-4); });
    std::vector<RadialFunction> wrong_angular{RadialFunction{15,std::vector<double>(grid.size(),1.0)}};
    expect_invalid("an angular momentum of 15",[&] { make_auxiliary_basis(grid,wrong_angular,1e-4); });
    std::vector<RadialFunction> zero{RadialFunction{0,std::vector<double>(grid.size(),0.0)}};
    expect_invalid("orbitals that vanish",[&] { make_auxiliary_basis(grid,zero,1e-4); });
    // A single constant function: one product, one ABF, normalized.
    const AuxiliaryBasis basis = make_auxiliary_basis(grid,one,1e-4);
    require(basis.functions.size() == 1 && basis.components() == 1,"a single orbital must give one ABF");
    require(std::fabs(radial_integral(grid,basis.functions[0].values,basis.functions[0].values)-1.0) <= 1e-14,
            "the ABF of a single orbital is not normalized");
  }
}

} // namespace exx_test
