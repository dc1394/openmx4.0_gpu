#include <openmx/basis/vps_file.hpp>
#include "allocation_gate.hpp"
#include "vps_fixture.hpp"

#include <future>
#include <iostream>
#include <stdexcept>

namespace {
using namespace openmx::basis;
void require(bool value, const char* reason) { if (!value) throw std::runtime_error(reason); }
void near(double value, double reference) {
  require(std::isfinite(value) && std::abs(value-reference) < 2e-12*(1+std::abs(reference)),"VPS value differs from analytic fixture");
}
VpsFile parse(const std::string& text, VpsReadMode mode = VpsReadMode::full) {
  std::istringstream input(text);return VpsFile::read(input,mode);
}
void replace(std::string& text, const std::string& old, const std::string& value) {
  const auto position = text.find(old);require(position != std::string::npos,"fixture replacement absent");
  text.replace(position,old.size(),value);
}
template<class F> void rejects(F&& function) {
  bool caught = false;try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid VPS input was accepted");
}

void contents()
{
  for (bool legacy : {false,true}) for (bool jdep : {false,true}) for (bool pcc : {false,true}) {
    const auto file = parse(vps_fixture(jdep,pcc,legacy));
    const auto& m = file.metadata();const int spins = 1+(jdep && !legacy);
    require(m.atom_species == 6.25 && m.atomic_number() == 6 && m.core_charge() == 4.25 &&
            m.mesh_points == 8 && m.projector_components == 13 && m.max_angular == 4 &&
            m.spin_channels() == spins && m.partial_core == pcc,"wrong VPS metadata");
    require(file.angular().size() == 3 && file.angular()[2] == 4,"wrong projector angular ordering");
    require((m.format == VpsFormat::legacy) == legacy,"wrong legacy detection");
    for (std::size_t i = 0; i < file.radii().size(); ++i) {
      const auto r = file.radii()[i];near(file.log_radii()[i],std::log(r));near(file.local_potential()[i],-4/(1+r));
      if (pcc) near(file.partial_core_density()[i],.125*(i+1));
      for (int channel = 0; channel < 3; ++channel) for (int spin = 0; spin < spins; ++spin) {
        near(file.projector(channel,spin)[i],(channel+1)*(spin+1)*(2+3*r+4*r*r));
        near(file.projector_energy(channel,spin),-.5*(channel+1)/(spin+1));
        const double x=.9;
        near(file.sample_projector(channel,spin,x),(channel+1)*(spin+1)*(2+3*x+4*x*x));
      }
    }
    require(pcc || file.partial_core_density().empty(),"disabled PCC has numerical storage");
  }
  for (bool legacy : {false,true}) {
    const auto empty = parse(vps_fixture(false,false,legacy,true,0));
    require(empty.metadata().atomic_number() == 0 && empty.angular().empty() &&
            empty.projector_values().empty() && empty.has_radial_data(),"zero-projector file failed");
  }
  auto variant = vps_fixture(false,false,false,false,.5);
  replace(variant,"j.dependent.pseudo.potentials off","J.DEPENDENT.PSEUDO.POTENTIALS .FALSE.");
  replace(variant,"charge.pcc.calc off","");replace(variant,"total.electron 6","");
  std::string crlf;for (char c : variant) { if (c=='\n') crlf+='\r';crlf+=c; }
  const auto virtual_atom = parse(crlf);
  require(virtual_atom.metadata().atomic_number()==1 && virtual_atom.metadata().core_charge()==4,"virtual atom/default total charge failed");
}

void metadata()
{
  for (bool legacy : {false,true}) {
    auto text = vps_fixture(true,true,legacy);text.resize(text.find("<Pseudo.Potentials"));
    const auto file = parse(text,VpsReadMode::metadata);
    require(!file.has_radial_data() && file.projector_values().empty() && file.partial_core_density().empty(),"metadata read retained radial tables");
    require(file.angular().size()==3,"metadata read lost projector count");
    rejects([&] { (void)file.projector(0,0); });rejects([&] { (void)parse(text); });
    if (legacy) {
      rejects([&] { (void)file.projector_energy(0,0); });
      replace(text,"local.part.vps 0","local.part.vps 99");
      require(parse(text,VpsReadMode::metadata).angular().size()==4,"historical local-row selection changed");
    } else near(file.projector_energy(2,1),-.75);
  }
}

void invalid()
{
  const auto original = vps_fixture();
  for (auto [from,to] : std::initializer_list<std::pair<std::string,std::string>>{
      {"AtomSpecies 6.25","AtomSpecies nan"},{"AtomSpecies 6.25","AtomSpecies -1"},
      {"grid.num.output 8","grid.num.output 5"},{"grid.num.output 8","grid.num.output 2147483647"},
      {"j.dependent.pseudo.potentials on","j.dependent.pseudo.potentials maybe"},
      {"charge.pcc.calc on","charge.pcc.calc invalid"},
      {"<project.energies\n3","<project.energies\n2147483647"},
      {"4 -1.5 -0.75","30 -1.5 -0.75"},{"4 -1.5 -0.75","4 inf -0.75"},
      {"project.energies>","missing.end"},{"Pseudo.Potentials>","missing.end"},
      {"density.PCC>","missing.end"},{"density.PCC>","1\ndensity.PCC>"}}) {
    auto bad = original;replace(bad,from,to);rejects([&] { (void)parse(bad); });
  }
  auto bad=original;const auto pos=bad.find("<density.PCC");auto tail=bad.substr(pos);
  replace(tail," 0.10000000000000001 "," 0.11 ");bad.replace(pos,std::string::npos,tail);
  rejects([&] { (void)parse(bad); });
  bad=original;bad.push_back('\0');rejects([&] { (void)parse(bad); });
  std::istringstream broken(original);broken.setstate(std::ios::badbit);rejects([&] { (void)VpsFile::read(broken); });
  auto old=vps_fixture(false,true,true);replace(old,"projection.energies>","missing.end");rejects([&] { (void)parse(old); });
  const auto file=parse(original);
  rejects([&] { (void)file.projector(3,0); });rejects([&] { (void)file.projector(0,2); });
  rejects([&] { (void)file.projector_energy(0,-1); });rejects([&] { (void)file.sample_projector(0,0,-1); });
}

void lifetime()
{
  auto text=vps_fixture();auto source=parse(text);auto copy=source;auto moved=std::move(copy);
  text.clear();copy=moved;moved=std::move(copy);source=parse(vps_fixture(false,false));
  require(moved.metadata().j_dependent && moved.projector(2,1).size()==8,"copy/move borrowed another owner");
  allocations_before_failure=0;const auto value=moved.sample_projector(2,1,.9);allocations_before_failure=-1;
  near(value,6*(2+3*.9+4*.9*.9));
}
void concurrent()
{
  std::array<std::future<void>,4> workers;
  for (std::size_t i=0;i<workers.size();++i) workers[i]=std::async(std::launch::async,[i] {
    for (int repeat=0;repeat<12;++repeat) {
      const auto file=parse(vps_fixture(i%2==0,true,false,false,i+1));
      require(file.metadata().atomic_number()==static_cast<int>(i+1),"cross-read VPS metadata contamination");
      near(file.sample_projector(2,0,.9),3*(2+3*.9+4*.9*.9));
    }
  });
  for (auto& worker:workers) worker.get();
}
void allocation()
{
  const auto text=vps_fixture();bool completed=false;int failures=0;
  for (int index=0;index<256;++index) {
    std::istringstream input(text);allocations_before_failure=index;
    try { const auto file=VpsFile::read(input);completed=file.has_radial_data(); }catch(const std::bad_alloc&){++failures;}
    allocations_before_failure=-1;if(completed)break;
  }
  require(completed && failures>10,"VPS read allocation sweep incomplete");
  auto target=parse(vps_fixture(false,false,false,false,1));const auto source=parse(text);completed=false;failures=0;
  for (int index=0;index<32;++index) {
    allocations_before_failure=index;
    try {target=source;completed=true;}catch(const std::bad_alloc&){++failures;}
    allocations_before_failure=-1;if(completed)break;
    require(target.metadata().atom_species==1 && !target.metadata().j_dependent && target.partial_core_density().empty(),"failed assignment changed VPS target");
    near(target.sample_projector(2,0,.9),3*(2+3*.9+4*.9*.9));
  }
  require(completed && failures>=6 && target.metadata().j_dependent,"VPS assignment sweep incomplete");
}
} // namespace

int main(int argc,char** argv)
{
  try {
    if(argc!=2)throw std::invalid_argument("expected VPS test case");const std::string name=argv[1];
    if(name=="contents")contents();else if(name=="metadata")metadata();else if(name=="invalid")invalid();
    else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();else if(name=="allocation")allocation();
    else throw std::invalid_argument("unknown VPS test case");
    std::cout<<"PASS VPS "<<name<<'\n';return 0;
  }catch(const std::exception& error){allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;}
}
