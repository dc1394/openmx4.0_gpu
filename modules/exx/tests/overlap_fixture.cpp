#include "overlap_fixture.hpp"

#include <openmx/basis/radial_table.hpp>

#include "test_support.hpp"

#include <cstddef>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <utility>

namespace exx_test {
namespace {

std::vector<double> numbers(std::istream& in, std::size_t count, const std::string& what)
{
  std::vector<double> values(count);
  for (double& value : values) require(static_cast<bool>(in >> value),"the overlap fixture ends inside "+what);
  return values;
}

} // namespace

OverlapFixture read_overlap_fixture(const std::string& path)
{
  std::ifstream in(path);
  require(in.good(),"cannot open the overlap fixture "+path);
  OverlapFixture f;
  std::string word, line;
  int version = 0;
  require(static_cast<bool>(in >> word >> version) && word == "exx_overlap_fixture" && version == 1,
          "unsupported overlap fixture");
  std::getline(in,line);
  while (in >> word) {
    if (word[0] == '#') { std::getline(in,line); continue; }
    if (word == "grid") {
      require(static_cast<bool>(in >> f.radial_intervals >> f.reciprocal_points >> f.kmax >> f.kmin),"malformed grid record");
    } else if (word == "lattice") {
      for (auto& v : f.lattice) for (double& x : v) require(static_cast<bool>(in >> x),"malformed lattice record");
    } else if (word == "nspecies") {
      std::size_t count = 0;
      require(static_cast<bool>(in >> count),"malformed nspecies record");
      f.species.resize(count);
    } else if (word == "species") {
      std::size_t index = 0, mesh = 0, shells = 0;
      std::string name;
      double cutoff = 0.0;
      require(static_cast<bool>(in >> index >> name >> cutoff >> mesh >> shells) && index < f.species.size(),
              "malformed species record");
      f.species[index].name = name;
      f.species[index].cutoff = cutoff;
      f.species[index].radii.resize(mesh);
    } else if (word == "radii") {
      std::size_t index = 0;
      require(static_cast<bool>(in >> index) && index < f.species.size(),"malformed radii record");
      f.species[index].radii = numbers(in,f.species[index].radii.size(),"a radial mesh");
    } else if (word == "shell") {
      std::size_t index = 0;
      int l = 0, multiplicity = 0;
      require(static_cast<bool>(in >> index >> l >> multiplicity) && index < f.species.size(),"malformed shell record");
      f.species[index].shells.push_back({l,numbers(in,f.species[index].radii.size(),"a radial function")});
    } else if (word == "natoms") {
      std::size_t count = 0;
      require(static_cast<bool>(in >> count),"malformed natoms record");
      f.atom_species.resize(count);
      f.positions.resize(count);
    } else if (word == "atom") {
      std::size_t index = 0;
      require(static_cast<bool>(in >> index) && index < f.positions.size(),"malformed atom record");
      require(static_cast<bool>(in >> f.atom_species[index] >> f.positions[index][0] >> f.positions[index][1] >>
                                f.positions[index][2]),"malformed atom record");
    } else if (word == "npairs") {
      std::size_t count = 0;
      require(static_cast<bool>(in >> count),"malformed npairs record");
      f.pairs.reserve(count);
    } else if (word == "pair") {
      FixturePair p;
      double distance = 0.0;
      require(static_cast<bool>(in >> p.a >> p.b >> p.cell[0] >> p.cell[1] >> p.cell[2] >> p.rows >> p.columns >> distance),
              "malformed pair record");
      p.values = numbers(in,static_cast<std::size_t>(p.rows)*static_cast<std::size_t>(p.columns),"an overlap block");
      f.pairs.push_back(std::move(p));
    } else {
      throw std::runtime_error("unknown record in the overlap fixture: "+word);
    }
  }
  require(!f.species.empty() && !f.positions.empty() && f.pairs.size() >= 10,"the overlap fixture is incomplete");
  return f;
}

std::vector<openmx::exx::RadialFunction> sample_shells(const FixtureSpecies& species,
                                                       const openmx::exx::RadialGrid& grid)
{
  std::vector<openmx::exx::RadialFunction> functions;
  for (const FixtureShell& shell : species.shells) {
    openmx::exx::RadialFunction f{shell.l,std::vector<double>(grid.size())};
    for (std::size_t i = 0; i < grid.size(); ++i)
      f.values[i] = openmx::basis::sample_radial_row(openmx::basis::RadialKind::pao,shell.l,species.radii,
                                                     shell.values,0.0,grid.radius(i));
    functions.push_back(std::move(f));
  }
  return functions;
}

} // namespace exx_test
