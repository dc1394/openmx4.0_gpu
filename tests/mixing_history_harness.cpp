#include "openmx_core_mixing.h"
#include "openmx_core_native_resources.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int mpi_comm_level1=0,Matomnum=1,SpinP_switch=0;
int M2G[]={0,1},WhatSpecies[]={0,0},Spe_Total_CNO[]={1},FNAN[]={0,0};
int neighbor[]={1};int* natn[]={neighbor,neighbor};
int SO_switch=0,Hub_U_switch=0,Constraint_NCS_switch=0,Zeeman_NCS_switch=0,Zeeman_NCO_switch=0;
double Min_Mixing_weight=.1,Max_Mixing_weight=.4,Uele=2;
constexpr int YOUSO10=1024,fp_bsize=65536;
int Host_ID=0,Cnt_switch=0,Solver=2,Hub_Type=0,dc_Type=0,Yukawa_on=0,Nmul=0;
double Ukin=0,UH0=0,UH1=0,Una=0,Unl=0,Uxc0=0,Uxc1=0,Ucore=0,Uhub=0,Ucs=0,Uzs=0,Uzo=0,Uef=0,UvdW=0,Uch=0,Utot=0,UpV=0;
double UH_dc=0,Uxc0_dc=0,Uxc1_dc=0,ChemP=0,orbitalOpt_criterion=0,eV2Hartree=27.2114;
int B_spe[30]={},B_l[30]={},B_mul[30]={};char* SpeName[1]={};
double lambda[30]={},U[30]={},J[30]={},Slater_F0[30]={},Slater_F2[30]={},Slater_F4[30]={},Slater_F6[30]={};
double parser_value=0;bool parser_default=false;
void input_double(const char* key,double* out,double fallback){
  if(std::strcmp(key,"scf.Init.Mixing.Weight"))throw std::runtime_error("input binding used the wrong keyword");
  *out=parser_default?fallback:parser_value;
}
void MPI_Abort(int,int){throw std::runtime_error("unexpected fatal input binding");}
constexpr int MPI_DOUBLE=0,MPI_SUM=0;
void MPI_Comm_rank(int,int* rank){*rank=0;}
void MPI_Comm_size(int,int* size){*size=1;}
void MPI_Allreduce(const double* from,double* to,int count,int,int,int){std::copy_n(from,count,to);}
#include "mixing_history.inc"

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void equal(double a,double b){require(std::isfinite(a) && std::abs(a-b)<1e-12*(1+std::abs(b)),"simple mixing differs from independent expected value");}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);require(bool(c),error);return c;}
OpenMXMixingHistory* history(OpenMXCoreContext* c){
  char error[256];OpenMXMixingHistory* h=nullptr;require(openmx_core_mixing_history(c,&h,error,sizeof(error)),error);return h;
}
// Fixed test topology: 4 spin slots, 2 atom slots, one neighbor/row and 2
// columns. Only atom=1/column=0 is active; the other values are canaries.
struct Field {
  std::array<double,16> values;
  std::array<double*,8> rows;
  std::array<double**,8> neighbors;
  std::array<double***,8> atoms;
  std::array<double****,4> spins;
  explicit Field(double canary=-91){
    values.fill(canary);
    for(int n=0;n<8;++n){rows[n]=values.data()+2*n;neighbors[n]=&rows[n];atoms[n]=&neighbors[n];}
    for(int s=0;s<4;++s)spins[s]=atoms.data()+2*s;
  }
  double***** data(){return spins.data();}
  double& active(int spin){return values[4*spin+2];}
  void canaries()const{for(int i=0;i<16;++i)if(i%4!=2)equal(values[i],-91);}
};
void step(OpenMXMixingHistory* h,int change,double supplied,double difference,double final_weight,int final_counter,double final_max){
  Field c,p,pp,ic,ip,ipp,r,ir;c.active(0)=1+difference;p.active(0)=1;
  Simple_Mixing_DM(h,change,supplied,c.data(),p.data(),pp.data(),ic.data(),ip.data(),ipp.data(),r.data(),ir.data());
  const double applied=change?final_weight:supplied;
  equal(c.active(0),1+applied*difference);equal(p.active(0),c.active(0));equal(pp.active(0),1);
  equal(r.active(0),difference);equal(h->norm[0],difference*difference);equal(h->weight,final_weight);
  equal(h->adaptive_max_weight,final_max);require(h->consecutive==final_counter,"adaptive counter differs");
  for(auto* f:{&c,&p,&pp,&ic,&ip,&ipp,&r,&ir})f->canaries();
  for(int spin=0;spin<4;++spin){equal(ic.active(spin),-91);equal(ip.active(spin),-91);equal(ipp.active(spin),-91);equal(ir.active(spin),-91);}
}
void golden(){
  struct Case{int change,counter;double cap,norm,difference,weight,energy,old_energy,older_energy,supplied,expected_weight,expected_cap;int expected_counter;};
  // Independent hand-calculated trajectories cover four trend quadrants,
  // both clamps, equal residuals, first-step and the 1000-step ceiling.
  const Case cases[]={
    {1,-1,0,4,1,.2,2,1,0,.8,4.0/15,.4,-1},
    {1,-1,0,1,2,.2,2,1,0,.8,.1,.4,-1},
    {1,-1,0,4,1,.2,0,1,0,.8,4.0/15,.4,-1},
    {1,-1,0,1,2,.2,0,1,0,.8,.1,.4,-1},
    {1,-1,0,2,1,.3,2,1,0,.8,.4,.4,0},
    {1,-1,0,.0625,1,.2,2,1,0,.8,.1,.4,-1},
    {1,-1,0,1,1,.2,2,1,0,.8,.2,.4,-1},
    {0,-1,0,4,1,.2,2,1,0,.25,.2,.4,-1},
    {1,1000,.5,1,1,.2,2,1,0,.8,.2,.7,0},
    {1,1000,.2,1,1,.2,2,1,0,.8,.2,.4,0},
    {1,0,.25,2,1,.3,2,1,0,.8,.25,.25,1},
    {1,-1,0,4,1,.2,1,1,1,.8,4.0/15,.4,-1}};
  auto c=calculation();auto* h=history(c.get());
  for(const auto& x:cases){
    h->consecutive=x.counter;h->adaptive_max_weight=x.cap;for(int i=0;i<5;++i)h->energy[i]=20+i;
    h->energy[0]=x.old_energy;h->energy[1]=x.older_energy;
    h->norm[0]=x.norm;for(int i=1;i<5;++i)h->norm[i]=30+i;
    h->weight=x.weight;Uele=x.energy;step(h,x.change,x.supplied,x.difference,x.expected_weight,x.expected_counter,x.expected_cap);
    const double expected[]={x.energy,x.old_energy,x.older_energy,22,23};
    for(int i=0;i<5;++i)equal(h->energy[i],expected[i]);
    equal(h->norm[1],x.norm);equal(h->norm[2],31);equal(h->norm[3],32);equal(h->norm[4],33);
  }
}
void imaginary(){
  auto c=calculation();auto* h=history(c.get());SpinP_switch=3;
  std::array<int*,5> flags{&SO_switch,&Hub_U_switch,&Constraint_NCS_switch,&Zeeman_NCS_switch,&Zeeman_NCO_switch};
  for(int selected=-1;selected<5;++selected){
    for(auto* flag:flags)*flag=0;if(selected>=0)*flags[selected]=1;
    Field real,previous,older,imag,iprevious,iolder,residual,iresidual;
    double norm=0;for(int s=0;s<4;++s){real.active(s)=1+.25*(s+1);previous.active(s)=1;norm+=.0625*(s+1)*(s+1);}
    for(int s=0;s<2;++s){imag.active(s)=2+.125*(s+1);iprevious.active(s)=2;if(selected>=0)norm+=.015625*(s+1)*(s+1);}
    h->consecutive=-1;Simple_Mixing_DM(h,0,.25,real.data(),previous.data(),older.data(),imag.data(),iprevious.data(),iolder.data(),residual.data(),iresidual.data());
    equal(h->norm[0],norm);
    for(int s=0;s<4;++s){equal(real.active(s),1+.0625*(s+1));equal(previous.active(s),real.active(s));equal(older.active(s),1);equal(residual.active(s),.25*(s+1));}
    for(int s=0;s<2;++s){equal(imag.active(s),2+(selected>=0?.03125:.125)*(s+1));equal(iprevious.active(s),selected>=0?imag.active(s):2);equal(iolder.active(s),selected>=0?2:-91);equal(iresidual.active(s),selected>=0?.125*(s+1):-91);}
    for(int s=2;s<4;++s){equal(imag.active(s),-91);equal(iprevious.active(s),-91);equal(iolder.active(s),-91);equal(iresidual.active(s),-91);}
    for(auto* f:{&real,&previous,&older,&imag,&iprevious,&iolder,&residual,&iresidual})f->canaries();
  }
  SpinP_switch=0;for(auto* flag:flags)*flag=0;
}
void interleaved(){
  for(int repeat=0;repeat<100;++repeat){
    auto a=calculation(),b=calculation();auto* x=history(a.get());auto* y=history(b.get());require(x!=y,"independent calculations share history");
    for(int n=0;n<5;++n){
      x->consecutive=-1;x->weight=.2;x->norm[0]=4;Uele=100+10*repeat+n;step(x,0,.25,1,.2,-1,.4);
      y->consecutive=-1;y->weight=.2;y->norm[0]=4;Uele=-100-10*repeat-n;step(y,0,.5,1,.2,-1,.4);
      for(int k=0;k<=n;++k){equal(x->energy[k],100+10*repeat+n-k);equal(y->energy[k],-100-10*repeat-n+k);}
    }
    openmx_core_native_resources_clear(a.get());x=history(a.get());for(double e:x->energy)equal(e,0);equal(x->adaptive_max_weight,0);require(x->consecutive==0,"clear retained counter");
    equal(y->energy[0],-104-10*repeat);openmx_core_finish(b.get());y=history(b.get());for(double e:y->energy)equal(e,0);
  }
}
void input_and_output(char* path){
  auto first=calculation(),second=calculation();auto* a=history(first.get());auto* b=history(second.get());
  parser_value=.125;parser_default=false;parse_weight(first.get());
  parser_default=true;parse_weight(second.get());equal(a->weight,.125);equal(b->weight,.3);
  a->norm[0]=4;b->norm[0]=-9;Uele=7.25;double chem[2]={};std::remove(path);
  Cnt_switch=0;outputfile1(a,1,1,0,0,2,path,chem,0);
  Cnt_switch=1;outputfile1(b,1,2,3,1,4,path,chem,0);
  FILE* file=std::fopen(path,"r");require(file,"missing production SCF output");
  char buffer[4096];const auto size=std::fread(buffer,1,sizeof(buffer),file);std::fclose(file);
  const std::string text(buffer,size);
  require(text.find("SCF=   2  NormRD=  2.000000000000  Uele=  7.250000000000")!=std::string::npos,"ordinary SCF output read another calculation's residual");
  require(text.find("OrbOpt=  3  SCF=   4  NormRD=  3.000000000000  Uele=  7.250000000000")!=std::string::npos,"orbital SCF output read another calculation's residual");
  equal(a->norm[0],4);equal(b->norm[0],-9);equal(a->weight,.125);equal(b->weight,.3);Cnt_switch=0;
  openmx_core_native_resources_clear(first.get());a=history(first.get());equal(a->weight,0);for(double n:a->norm)equal(n,0);
  equal(b->weight,.3);equal(b->norm[0],-9);parser_default=false;parser_value=.0625;parse_weight(first.get());equal(a->weight,.0625);
  std::remove(path);
}
}
int main(int argc,char** argv){try{require(argc==2,"expected an output path");golden();imaginary();interleaved();input_and_output(argv[1]);std::cout<<"PASS: actual simple mixing, input binding and SCF output, 12 golden trajectories, 6 imaginary-spin modes, 100 interleaved calculations\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
