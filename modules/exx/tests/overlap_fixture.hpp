// The fixture tests/fixtures/overlap_reference.txt, written by
// tools/exx_overlap_fixture.py from a run of the main program: the radial
// functions of the PAOs of a few species, a geometry, and blocks of the
// overlap matrix OLP[0].  The tests of the PAO products reuse its species.
#pragma once

#include <openmx/exx/two_center.hpp>

#include <array>
#include <string>
#include <vector>

namespace exx_test {

struct FixtureShell {
  int l = 0;
  std::vector<double> values; // on the radial mesh of the species
};
struct FixtureSpecies {
  std::string name;
  double cutoff = 0.0;        // cutoff radius of the PAOs, bohr
  std::vector<double> radii;  // radial mesh of the .pao file
  std::vector<FixtureShell> shells; // in the order of the orbitals
};
// One block of the overlap matrix: rows on atom a in the cell at the origin,
// columns on atom b in the cell "cell".
struct FixturePair {
  int a = 0, b = 0;
  std::array<int,3> cell{};
  int rows = 0, columns = 0;
  std::vector<double> values; // row-major
};
struct OverlapFixture {
  int radial_intervals = 0;   // 1DFFT.NumGridR
  int reciprocal_points = 0;  // 1DFFT.NumGridK
  double kmax = 0.0, kmin = 0.0;
  std::array<std::array<double,3>,3> lattice{};
  std::vector<FixtureSpecies> species;
  std::vector<int> atom_species;
  std::vector<std::array<double,3>> positions;
  std::vector<FixturePair> pairs;
};

OverlapFixture read_overlap_fixture(const std::string& path);

// The PAOs of a species at the nodes of a uniform grid, by the interpolation
// of the main program.
std::vector<openmx::exx::RadialFunction> sample_shells(const FixtureSpecies& species,
                                                       const openmx::exx::RadialGrid& grid);

} // namespace exx_test
