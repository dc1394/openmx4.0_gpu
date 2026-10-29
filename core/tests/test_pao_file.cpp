#include "openmx_core_pao_file.h"
#include "allocation_faults.hpp"
#include "pao_fixture.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message) { if (!condition) throw std::runtime_error(message); }
struct FixtureFile {
  std::string path;
  explicit FixtureFile(std::string name) : path(std::move(name)) { write(pao_fixture()); }
  void write(const std::string& text) const {
    std::ofstream output(path,std::ios::binary);
    output << text; require(static_cast<bool>(output),"cannot write PAO fixture");
  }
  ~FixtureFile() { std::error_code error; std::filesystem::remove(path,error); }
};
struct Target {
  std::array<int,3> number{-7,-7,-7}, mesh{-7,-7,-7}, lmax{-7,-7,-7}, mul{-7,-7,-7};
  std::array<double,3> cutoff{-7,-7,-7};
  std::array<int,3> selected_lmax{4,4,4};
  std::array<int,18> basis{};
  std::vector<double> log_radii=std::vector<double>(3*10,-99),radii=log_radii;
  std::vector<double> density=std::vector<double>(3*12,-99),backup=density;
  std::vector<double> orbitals=std::vector<double>(3*6*3*10,-99);
  Target() { basis.fill(2); }
  OpenMXAtomicViews view() {
    OpenMXAtomicViews v{};
    auto& s=v.species;s.species=3;s.angular_channels=6;
    s.atomic_number=number.data();s.pao_mesh=mesh.data();s.pao_lmax=lmax.data();s.pao_multiplicity=mul.data();s.cutoff=cutoff.data();
    s.basis_lmax=selected_lmax.data();s.basis=basis.data();
    v.pao_log_radii={log_radii.data(),{3,10}};v.pao_radii={radii.data(),{3,10}};
    v.density={density.data(),{3,12}};v.density_backup={backup.data(),{3,12}};
    v.orbitals={orbitals.data(),{3,6,3,10}};return v;
  }
  void pristine() const {
    for(auto v:number)require(v==-7,"failed read changed atomic metadata");
    for(auto v:mesh)require(v==-7,"failed read changed mesh metadata");
    for(auto v:lmax)require(v==-7,"failed read changed angular metadata");
    for(auto v:mul)require(v==-7,"failed read changed multiplicity metadata");
    for(auto v:cutoff)require(v==-7,"failed read changed cutoff metadata");
    for(const auto* row:{&log_radii,&radii,&density,&backup,&orbitals})
      for(auto value:*row)require(value==-99,"failed read changed numerical output");
  }
};

void contents(const FixtureFile& file)
{
  Target target;auto view=target.view();OpenMXPaoFileInfo info{};char error[256];
  require(openmx_core_pao_read_file(file.path.c_str(),1,4,&view,1,&info,error,sizeof error),error);
  require(info.atom_species==6.25 && info.eigenvalue_found && info.eigenvalue==-2.5,"metadata or energy mismatch");
  require(target.number[1]==6 && target.mesh[1]==8 && target.lmax[1]==4 && target.mul[1]==2 && target.cutoff[1]==6,"species metadata mismatch");
  for(auto v:target.orbitals)require(v==-99,"metadata read touched orbital storage");
  require(openmx_core_pao_read_file(file.path.c_str(),0,4,&view,1,&info,error,sizeof error),error);
  require(error[0]=='\0',"successful read retained an error");
  for(int species=0;species<3;++species) {
    if(species!=1)require(target.number[species]==-7 && target.mesh[species]==-7,"read changed another species");
    for(int point=0;point<10;++point) {
      const auto offset=species*10+point;
      if(species!=1 || point>=8)require(target.radii[offset]==-99 && target.log_radii[offset]==-99,"read changed mesh padding/another species");
    }
    for(int point=0;point<12;++point) {
      const auto offset=species*12+point;
      const double expected=species==1 && point>=1 && point<=8 ? .125*point : -99;
      require(target.density[offset]==expected && target.backup[offset]==expected,"density +1 offset/padding mismatch");
    }
    for(int l=0;l<6;++l)for(int mu=0;mu<3;++mu)for(int point=0;point<10;++point) {
      const auto offset=((species*6+l)*3+mu)*10+point;
      if(species==1 && l<=4 && mu<2 && point<8) {
        const double r=target.radii[10+point],expected=(l+1)*(mu+1)*(2+3*r+4*r*r);
        require(std::abs(target.orbitals[offset]-expected)<1e-12,"orbital row/stride mismatch");
      } else require(target.orbitals[offset]==-99,"read changed orbital padding/another species");
    }
  }
  file.write(pao_fixture(false,.5));
  require(openmx_core_pao_read_file(file.path.c_str(),1,4,&view,1,&info,error,sizeof error),error);
  require(info.atomic_number==1 && !info.eigenvalue_found,"virtual-atom/absent eigenvalue handling");
}

void invalid(const FixtureFile& file)
{
  for(int mode=0;mode<15;++mode) {
    Target target;auto view=target.view();OpenMXPaoFileInfo info{};info.mesh_points=-88;char error[256];
    const char* path=file.path.c_str();std::size_t species=1;int metadata=0,eigen=-1;
    if(mode==0)path=nullptr;
    if(mode==1)species=3;
    if(mode==2)view.species.pao_mesh=nullptr;
    if(mode==3)view.species.angular_channels=0;
    if(mode==4)target.selected_lmax[1]=5;
    if(mode==5)target.basis[6]=3;
    if(mode==6)view.pao_radii.extent[1]=7;
    if(mode==7)view.density.extent[1]=9;
    if(mode==8)view.orbitals.extent[1]=4;
    if(mode==9)view.orbitals.extent[2]=1;
    if(mode==10)view.orbitals.values=nullptr;
    if(mode==11)view.orbitals.extent[3]=std::numeric_limits<std::size_t>::max();
    if(mode==12)metadata=2;
    if(mode==13)eigen=30;
    if(mode==14)path="/does-not-exist/openmx-pao-file-fixture";
    require(!openmx_core_pao_read_file(path,metadata,eigen,&view,species,&info,error,sizeof error),"invalid read accepted");
    require(error[0] && info.mesh_points==-88,"failed read overwrote info/lost diagnostic");target.pristine();
  }
  Target target;auto view=target.view();OpenMXPaoFileInfo info{};info.mesh_points=-88;
  file.write("AtomSpecies 6\ngrid.num.output 8\nPAO.Lmax 4\nPAO.Mul 2\n<valence.charge.density\ninvalid\nvalence.charge.density>\n");
  char error[2]={'x','x'};
  require(!openmx_core_pao_read_file(file.path.c_str(),0,-1,&view,1,&info,error,2),"malformed table accepted");
  require(error[0] && !error[1] && info.mesh_points==-88,"short error buffer/transaction failed");target.pristine();
}

void allocation(const FixtureFile& file)
{
  // Warm up standard-library IO before tracking live allocations.
  {Target warm;auto view=warm.view();OpenMXPaoFileInfo info;char error[256];
   require(openmx_core_pao_read_file(file.path.c_str(),0,-1,&view,1,&info,error,sizeof error),error);}
  bool completed=false;int failures=0;
  for(int index=0;index<256;++index) {
    Target target;auto view=target.view();OpenMXPaoFileInfo info{};info.mesh_points=-88;char error[256];
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=openmx_core_pao_read_file(file.path.c_str(),0,-1,&view,1,&info,error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    require(openmx::test::live_allocations()==live,"PAO read leaked an intermediate allocation");
    if(ok){completed=true;break;}
    ++failures;require(info.mesh_points==-88 && error[0],"failed read lost transaction/error");target.pristine();
  }
  require(completed && failures>10,"PAO allocation failure sweep incomplete");
}
} // namespace

int main(int argc,char** argv)
{
  try {
    if(argc!=3)throw std::invalid_argument("expected case and temporary fixture path");
    FixtureFile file(argv[2]);const std::string name=argv[1];
    if(name=="contents")contents(file);else if(name=="invalid")invalid(file);else if(name=="allocation")allocation(file);
    else throw std::invalid_argument("unknown PAO adapter case");
    std::cout<<"PASS PAO adapter "<<name<<'\n';return 0;
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
