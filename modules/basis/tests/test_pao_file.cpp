#include <openmx/basis/pao_file.hpp>
#include "allocation_gate.hpp"
#include "pao_fixture.hpp"

#include <future>
#include <iostream>
#include <stdexcept>

namespace {
using namespace openmx::basis;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void near(double value, double reference) {
  require(std::isfinite(value) && std::abs(value-reference) < 2e-12*(1+std::abs(reference)),"PAO value differs from analytic fixture");
}
PaoFile parse(const std::string& text, PaoReadMode mode = PaoReadMode::full) {
  std::istringstream input(text);
  return PaoFile::read(input,mode);
}
void replace(std::string& text, const std::string& old, const std::string& value) {
  const auto p = text.find(old); require(p != std::string::npos,"fixture replacement not found");
  text.replace(p,old.size(),value);
}
template<class F> void rejects(F&& function) {
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid PAO input was accepted");
}

void contents()
{
  const auto file = parse(pao_fixture());
  const auto& m = file.metadata();
  require(m.atom_species == 6.25 && m.atomic_number() == 6 && m.mesh_points == 8 &&
          m.lmax == 4 && m.multiplicity == 2 && m.cutoff == 6,"wrong PAO metadata");
  for (std::size_t i = 0; i < file.radii().size(); ++i) {
    near(file.log_radii()[i],std::log(file.radii()[i]));
    near(file.density()[i],.125*(i+1));
    for (int l = 0; l <= 4; ++l)
      for (int mu = 0; mu < 2; ++mu) {
        const double r = file.radii()[i];
        near(file.orbital(l,mu)[i],(l+1)*(mu+1)*(2+3*r+4*r*r));
        require(file.orbital_energy(l,mu).has_value(),"missing generator eigenvalue");
        near(*file.orbital_energy(l,mu),-.5*(l+1)/(mu+1));
      }
  }
  for (int l = 0; l <= 4; ++l)
    for (int mu = 0; mu < 2; ++mu) {
      for (double r : {.1,.15,.3,.6,1.,1.5,2.})
        near(file.sample(l,mu,r),(l+1)*(mu+1)*(2+3*r+4*r*r));
      near(file.sample(l,mu,5),0);
      if (l) near(file.sample(l,mu,0),0);
    }
  auto variant = pao_fixture(false,.75);
  replace(variant,"AtomSpecies","atomspecies");
  replace(variant,"radial.cutoff.pao 6\n","");
  replace(variant,"grid.num.output 8","GRID.NUM.OUTPUT +8");
  std::string crlf;
  for (char c : variant) { if (c == '\n') crlf += '\r'; crlf += c; }
  const auto virtual_atom = parse(crlf);
  require(virtual_atom.metadata().atomic_number() == 1 && virtual_atom.metadata().cutoff == 5,
          "virtual species/default cutoff semantics changed");
  require(!virtual_atom.orbital_energy(4,1),"invented absent eigenvalues");
}

void metadata()
{
  const std::string header = "AtomSpecies 6\ngrid.num.output 8\nPAO.Lmax 4\nPAO.Mul 2\n";
  const auto file = parse(header,PaoReadMode::metadata);
  require(!file.has_radial_data() && file.radii().empty() && file.density().empty(),"metadata read allocated numerical tables");
  rejects([&] { (void)file.orbital(0,0); });
  rejects([&] { (void)file.sample(0,0,1); });
  rejects([&] { (void)parse(header); });
  const auto with_energy = parse(pao_fixture(),PaoReadMode::metadata);
  near(*with_energy.orbital_energy(4,0),-2.5);
}

void invalid()
{
  const auto original = pao_fixture();
  for (auto [from,to] : std::initializer_list<std::pair<std::string,std::string>>{
      {"AtomSpecies 6.25","AtomSpecies nan"},{"AtomSpecies 6.25","AtomSpecies +-6"},
      {"grid.num.output 8","grid.num.output -1"},{"grid.num.output 8","grid.num.output 2147483647"},
      {"PAO.Lmax 4","PAO.Lmax 30"},{"PAO.Mul 2","PAO.Mul 0"},{"PAO.Mul 2","PAO.Mul 2147483647"},
      {"radial.cutoff.pao 6","radial.cutoff.pao inf"},
      {"valence.charge.density>","missing.end"},{"pseudo.atomic.orbitals.L=4>","missing.end"},
      {"l mu 4 1 -1.25","l mu 4 0 -1.25"},
      {"l mu 0 0 -0.5","l mu 30 0 -0.5"},
      {"valence.charge.density>","7\nvalence.charge.density>"}}) {
    auto bad = original; replace(bad,from,to); rejects([&] { (void)parse(bad); });
  }
  auto inconsistent = original;
  const auto pos = inconsistent.find("<pseudo.atomic.orbitals.L=0");
  auto tail = inconsistent.substr(pos);
  replace(tail," 0.10000000000000001 "," 0.11 ");
  inconsistent.replace(pos,std::string::npos,tail);
  rejects([&] { (void)parse(inconsistent); });
  auto nul = original; nul.push_back('\0'); rejects([&] { (void)parse(nul); });
  std::istringstream broken(original); broken.setstate(std::ios::badbit);
  rejects([&] { (void)PaoFile::read(broken); });
  const auto file = parse(original);
  rejects([&] { (void)file.orbital(5,0); }); rejects([&] { (void)file.orbital(0,2); });
  rejects([&] { (void)file.orbital_energy(-1,0); }); rejects([&] { (void)file.sample(0,0,-1); });
}

void lifetime()
{
  auto input = pao_fixture();
  auto owner = parse(input);
  auto copy = owner;
  auto moved = std::move(copy);
  input.assign("input was destroyed");
  copy = moved;
  moved = std::move(copy);
  const auto expected = owner.sample(2,1,.6);
  owner = parse(pao_fixture(false,1));
  near(moved.sample(2,1,.6),expected);
  require(moved.metadata().atom_species == 6.25,"owner retained input/another owner's address");
  const auto row = moved.orbital(4,1);
  allocations_before_failure = 0;
  const auto value = moved.sample(4,1,.6);
  allocations_before_failure = -1;
  require(row.size() == 8,"lost row extent"); near(value,10*(2+3*.6+4*.6*.6));
}

void concurrent()
{
  std::array<std::future<void>,4> workers;
  for (std::size_t i = 0; i < workers.size(); ++i)
    workers[i] = std::async(std::launch::async,[i] {
      for (int repeat = 0; repeat < 12; ++repeat) {
        const auto file = parse(pao_fixture(i%2 == 0,i+1));
        require(file.metadata().atomic_number() == static_cast<int>(i+1),"cross-read metadata contamination");
        near(file.sample(4,1,.6),10*(2+3*.6+4*.6*.6));
      }
    });
  for (auto& worker : workers) worker.get();
}

void allocation()
{
  const auto text = pao_fixture();
  bool completed = false;
  int failures = 0;
  for (int index = 0; index < 256; ++index) {
    std::istringstream input(text);
    allocations_before_failure = index;
    try { const auto file = PaoFile::read(input); completed = file.has_radial_data(); }
    catch (const std::bad_alloc&) { ++failures; }
    allocations_before_failure = -1;
    if (completed) break;
  }
  require(completed && failures > 10,"reader allocation fault sweep incomplete");
  auto target = parse(pao_fixture(false,1));
  const auto source = parse(text);
  completed = false; failures = 0;
  for (int index = 0; index < 32; ++index) {
    allocations_before_failure = index;
    try { target = source; completed = true; } catch (const std::bad_alloc&) { ++failures; }
    allocations_before_failure = -1;
    if (completed) break;
    require(target.metadata().atom_species == 1 && !target.orbital_energy(0,0),"failed assignment changed the target");
    near(target.sample(4,1,.6),10*(2+3*.6+4*.6*.6));
  }
  require(completed && failures >= 5 && target.metadata().atom_species == 6.25,"copy fault sweep incomplete");
}
} // namespace

int main(int argc, char** argv)
{
  try {
    if (argc != 2) throw std::invalid_argument("expected PAO test case");
    const std::string name=argv[1];
    if (name == "contents") contents(); else if (name == "metadata") metadata();
    else if (name == "invalid") invalid(); else if (name == "lifetime") lifetime();
    else if (name == "concurrent") concurrent(); else if (name == "allocation") allocation();
    else throw std::invalid_argument("unknown PAO test case");
    std::cout << "PASS PAO " << name << '\n'; return 0;
  } catch (const std::exception& error) {
    allocations_before_failure = -1; std::cerr << error.what() << '\n'; return 1;
  }
}
