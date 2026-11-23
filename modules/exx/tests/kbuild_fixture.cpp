#include "kbuild_fixture.hpp"

#include "test_support.hpp"

#include <cstddef>
#include <fstream>
#include <istream>
#include <utility>

namespace exx_test {
namespace {

using namespace openmx::exx;

std::vector<double> numbers(std::istream& in, std::size_t count, const std::string& what)
{
  std::vector<double> values(count);
  for (double& value : values) require(static_cast<bool>(in >> value),"the K fixture ends inside "+what);
  return values;
}

std::vector<Site> sites(std::istream& in, std::size_t count, std::size_t atoms)
{
  std::vector<Site> list(count);
  for (Site& site : list) {
    require(static_cast<bool>(in >> site.atom >> site.cell[0] >> site.cell[1] >> site.cell[2]) && site.atom >= 0 &&
            static_cast<std::size_t>(site.atom) < atoms,"malformed neighbour of the K fixture");
  }
  return list;
}

// "count" records "x y R1 R2 R3 rows columns" with their values.
std::map<BlockKey,std::vector<double>> blocks(std::istream& in, std::size_t count, const System& system,
                                              const std::string& what)
{
  std::map<BlockKey,std::vector<double>> result;
  for (std::size_t n = 0; n < count; ++n) {
    BlockKey key;
    std::size_t rows = 0, columns = 0;
    require(static_cast<bool>(in >> key.row >> key.column >> key.cell[0] >> key.cell[1] >> key.cell[2] >> rows >> columns),
            "malformed header of "+what);
    require(key.row >= 0 && key.column >= 0 && static_cast<std::size_t>(key.row) < system.orbital_count.size() &&
            static_cast<std::size_t>(key.column) < system.orbital_count.size() &&
            rows == system.orbital_count[static_cast<std::size_t>(key.row)] &&
            columns == system.orbital_count[static_cast<std::size_t>(key.column)],what+" has the wrong shape");
    require(result.emplace(key,numbers(in,rows*columns,what)).second,what+" occurs twice");
  }
  return result;
}

} // namespace

KbuildFixture read_kbuild_fixture(const std::string& path)
{
  std::ifstream in(path);
  require(in.good(),"cannot open the K fixture "+path);
  KbuildFixture f;
  System& system = f.system;
  std::string word;
  int version = 0;
  require(static_cast<bool>(in >> word >> version) && word == "exx_fixture" && version == 1,"unsupported K fixture");

  std::size_t atoms = 0, t_blocks = 0, v_blocks = 0;
  bool has_density = false, has_k = false, has_energy = false;
  // The T blocks of an atom lie side by side in the system; the offsets of
  // the blocks follow from the C neighbours, which the file lists first.
  const auto lay_out = [&system,&atoms] {
    if (!system.t.empty()) return;
    system.c_offset.resize(atoms);
    system.t.resize(atoms);
    system.v.resize(atoms);
    for (std::size_t a = 0; a < atoms; ++a) {
      require(!system.c_neighbours[a].empty() && !system.v_neighbours[a].empty(),
              "the K fixture lists a block before the neighbours of its atom");
      std::vector<std::size_t>& offset = system.c_offset[a];
      offset.assign(1,0);
      for (const Site& site : system.c_neighbours[a])
        offset.push_back(offset.back()+system.orbital_count[static_cast<std::size_t>(site.atom)]);
      system.t[a].assign(system.abf_count[a]*system.orbital_count[a]*offset.back(),0.0);
      system.v[a].resize(system.v_neighbours[a].size());
    }
  };

  while (in >> word) {
    if (word == "alpha") {
      require(static_cast<bool>(in >> f.alpha),"malformed alpha record");
    } else if (word == "lattice") {
      (void)numbers(in,9,"the lattice");          // the exchange build needs no geometry
    } else if (word == "natoms") {
      require(static_cast<bool>(in >> atoms) && atoms > 0,"malformed natoms record");
      system.orbital_count.assign(atoms,0);
      system.abf_count.assign(atoms,0);
      system.c_neighbours.resize(atoms);
      system.v_neighbours.resize(atoms);
    } else if (word == "atom") {
      std::size_t a = 0;
      require(static_cast<bool>(in >> a) && a < atoms,"malformed atom record");
      require(static_cast<bool>(in >> system.orbital_count[a] >> system.abf_count[a]),"malformed atom record");
      (void)numbers(in,3,"the position of an atom");
    } else if (word == "cnb" || word == "vnb") {
      std::size_t a = 0, count = 0;
      require(static_cast<bool>(in >> a >> count) && a < atoms,"malformed "+word+" record");
      (word == "cnb" ? system.c_neighbours : system.v_neighbours)[a] = sites(in,count,atoms);
    } else if (word == "T") {
      lay_out();
      std::size_t a = 0, i = 0, np = 0, ua = 0, uc = 0;
      require(static_cast<bool>(in >> a >> i >> np >> ua >> uc) && a < atoms && i < system.c_neighbours[a].size(),
              "malformed T record");
      const std::size_t first = system.c_offset[a][i], columns = system.c_offset[a].back();
      require(np == system.abf_count[a] && ua == system.orbital_count[a] && uc == system.c_offset[a][i+1]-first,
              "a T block of the K fixture has the wrong shape");
      const std::vector<double> block = numbers(in,np*ua*uc,"a T block");
      for (std::size_t row = 0; row < np*ua; ++row)
        for (std::size_t lambda = 0; lambda < uc; ++lambda) system.t[a][row*columns+first+lambda] = block[row*uc+lambda];
      ++t_blocks;
    } else if (word == "V") {
      lay_out();
      std::size_t a = 0, i = 0, np = 0, nq = 0;
      require(static_cast<bool>(in >> a >> i >> np >> nq) && a < atoms && i < system.v_neighbours[a].size(),
              "malformed V record");
      require(np == system.abf_count[a] && nq == system.abf_count[static_cast<std::size_t>(system.v_neighbours[a][i].atom)] &&
              system.v[a][i].empty(),"a V block of the K fixture has the wrong shape or occurs twice");
      system.v[a][i] = numbers(in,np*nq,"a V block");
      ++v_blocks;
    } else if (word == "D" || word == "K") {
      std::size_t count = 0;
      require(static_cast<bool>(in >> count),"malformed "+word+" record");
      (word == "D" ? f.density : f.k) = blocks(in,count,system,"a block of "+word);
      (word == "D" ? has_density : has_k) = true;
    } else if (word == "energy") {
      require(static_cast<bool>(in >> f.energy),"malformed energy record");
      has_energy = true;
    } else {
      throw std::runtime_error("unknown record of the K fixture: "+word);
    }
  }

  std::size_t c_sites = 0, v_sites = 0;
  for (std::size_t a = 0; a < atoms; ++a) {
    c_sites += system.c_neighbours[a].size();
    v_sites += system.v_neighbours[a].size();
  }
  require(atoms > 0 && t_blocks == c_sites && v_blocks == v_sites && has_density && has_k && has_energy,
          "the K fixture is incomplete");
  return f;
}

} // namespace exx_test
