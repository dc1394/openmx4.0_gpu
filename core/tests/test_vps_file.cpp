#include "openmx_core_vps_file.h"
#include "allocation_faults.hpp"
#include "vps_fixture.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value,const char* reason) { if (!value) throw std::runtime_error(reason); }
struct FixtureFile {
  std::string path;
  explicit FixtureFile(std::string name):path(std::move(name)){write(vps_fixture());}
  void write(const std::string& text) const {
    std::ofstream output(path,std::ios::binary);output<<text;require(static_cast<bool>(output),"cannot write VPS fixture");
  }
  ~FixtureFile(){std::error_code error;std::filesystem::remove(path,error);}
};
struct Consumer {
  int calls=0;
  int behavior=0; // 0=validate, 1=fail, 2=diagnostic, 3=throw, 4=nonstandard exception
  bool legacy=false,jdep=true,pcc=true,empty=false;
  const char* nested=nullptr;
};
int consume(void* user,const OpenMXVpsFileData* data,char* error,std::size_t capacity)
{
  auto& state=*static_cast<Consumer*>(user);++state.calls;
  if(state.behavior==1)return 0;
  if(state.behavior==2){if(error && capacity)std::snprintf(error,capacity,"consumer diagnostic");return 0;}
  if(state.behavior==3)throw std::runtime_error("consumer exception");
  if(state.behavior==4)throw 9;
  const int spins=1+(state.jdep && !state.legacy),count=state.empty ? 0 : 3;
  require(data->atom_species==6.25 && data->atomic_number==6 && data->mesh_points==8 &&
          data->projector_count==static_cast<std::size_t>(count) && data->j_dependent==spins-1 &&
          data->legacy_format==state.legacy && data->partial_core==state.pcc,"VPS bridge metadata changed");
  if(count)require(data->angular[2]==4,"VPS bridge angular order changed");
  if(data->metadata_only) {
    require(!data->local_potential_capacity,"VPS metadata allocated local potential");
    if(!state.legacy && count)require(data->energies[2*spins]==-1.5,"VPS metadata energy lost");
  }else{
    require(data->local_potential_capacity==10 && data->local_potential[8]==0 && data->local_potential[9]==0,
            "VPS local-potential padding is not initialized");
    for(int i=0;i<8;++i){
      const double r=data->radii[i];
      require(std::abs(data->local_potential[i]+4/(1+r))<1e-12,"local-potential value changed");
      if(state.pcc)require(data->partial_core_density[i]==.125*(i+1),"PCC value changed");
      for(int channel=0;channel<count;++channel)for(int spin=0;spin<spins;++spin){
        require(data->energies[channel*spins+spin]==-.5*(channel+1)/(spin+1),"projector energy layout changed");
        const double expected=(channel+1)*(spin+1)*(2+3*r+4*r*r);
        require(std::abs(data->projectors[(channel*spins+spin)*8+i]-expected)<1e-12,"projector row layout changed");
      }
    }
    // The legacy charge correction can mutate scratch without changing the
    // immutable library file or requiring a manually freed C allocation.
    data->local_potential[0]=999;
  }
  if(state.nested){
    Consumer inner=state;inner.calls=0;inner.nested=nullptr;
    char diagnostic[256];
    require(openmx_core_with_vps_file(state.nested,data->metadata_only,consume,&inner,diagnostic,sizeof diagnostic),diagnostic);
    require(inner.calls==1,"nested VPS consumer count");
    if(!data->metadata_only)require(data->local_potential[0]==999,"nested read invalidated outer scratch");
  }
  return 1;
}

void contents(const FixtureFile& file)
{
  for(bool legacy:{false,true})for(bool jdep:{false,true})for(bool pcc:{false,true})for(bool empty:{false,true}){
    file.write(vps_fixture(jdep,pcc,legacy,empty));
    for(int metadata:{0,1}){
      Consumer state;state.legacy=legacy;state.jdep=jdep;state.pcc=pcc;state.empty=empty;state.nested=file.path.c_str();
      char error[256];require(openmx_core_with_vps_file(file.path.c_str(),metadata,consume,&state,error,sizeof error),error);
      require(state.calls==1 && !*error,"successful consumer count/diagnostic");
    }
  }
}
void invalid(const FixtureFile& file)
{
  for(int kind=0;kind<4;++kind){
    Consumer state;char error[256];
    const char* path=kind==0 ? nullptr : kind==3 ? "/does-not-exist/openmx-vps-fixture" : file.path.c_str();
    require(!openmx_core_with_vps_file(path,kind==1 ? 2 : 0,kind==2 ? nullptr : consume,&state,error,sizeof error),"invalid VPS request accepted");
    require(!state.calls && *error,"invalid request called consumer/lost diagnostic");
  }
  for(int behavior=1;behavior<=4;++behavior){
    Consumer state;state.behavior=behavior;char error[256];
    require(!openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,error,sizeof error),"consumer failure escaped status");
    require(state.calls==1 && *error,"consumer failure missing diagnostic");
  }
  Consumer state;state.behavior=1;
  require(!openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,nullptr,0),"null diagnostic failure");
  file.write("AtomSpecies 6\ngrid.num.output 8\n<project.energies\n1\n0 invalid\nproject.energies>\n");
  state.calls=0;char short_error[2]={'x','x'};
  require(!openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,short_error,2),"malformed VPS accepted");
  require(!state.calls && short_error[0] && !short_error[1],"malformed read called consumer/truncated diagnostic unterminated");
}
void allocation(const FixtureFile& file)
{
  {Consumer state;char error[256];require(openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,error,sizeof error),error);}
  bool completed=false;int failures=0;
  for(int index=0;index<256;++index){
    Consumer state;char error[256];const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(index);
    const int ok=openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,error,sizeof error);
    openmx::test::fail_allocation_after(-1);
    require(openmx::test::live_allocations()==live,"VPS read leaked an allocation");
    if(ok){require(state.calls==1,"successful read skipped consumer");completed=true;break;}
    ++failures;require(!state.calls && *error,"allocation failure invoked consumer/lost diagnostic");
  }
  require(completed && failures>10,"VPS allocation failure sweep incomplete");
  for(int behavior=1;behavior<=4;++behavior){
    Consumer state;state.behavior=behavior;char error[256];const auto live=openmx::test::live_allocations();
    require(!openmx_core_with_vps_file(file.path.c_str(),0,consume,&state,error,sizeof error),"failure callback accepted");
    require(openmx::test::live_allocations()==live,"consumer failure leaked VPS owner");
  }
}
} // namespace

int main(int argc,char** argv)
{
  try{
    if(argc!=3)throw std::invalid_argument("expected case and temporary path");
    FixtureFile file(argv[2]);const std::string name=argv[1];
    if(name=="contents")contents(file);else if(name=="invalid")invalid(file);else if(name=="allocation")allocation(file);
    else throw std::invalid_argument("unknown VPS adapter case");
    std::cout<<"PASS VPS adapter "<<name<<'\n';return 0;
  }catch(const std::exception& error){openmx::test::fail_allocation_after(-1);std::cerr<<error.what()<<'\n';return 1;}
}
