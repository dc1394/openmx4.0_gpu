#include "openmx_core_energy.h"
#include "openmx_core_dynamics.h"
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
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <string>

int mpi_comm_level1=0,Matomnum=1,SpinP_switch=0;
int M2G[]={0,1},WhatSpecies[]={0,0},Spe_Total_CNO[]={1},FNAN[]={0,0};
int neighbor[]={1};int* natn[]={neighbor,neighbor};
int SO_switch=0,Hub_U_switch=0,Constraint_NCS_switch=0,Zeeman_NCS_switch=0,Zeeman_NCO_switch=0;
double Min_Mixing_weight=.1,Max_Mixing_weight=.4,incoming_energy=2;
constexpr int YOUSO10=1024,fp_bsize=65536;
int Host_ID=0,Cnt_switch=0,Solver=2,Hub_Type=0,dc_Type=0,Yukawa_on=0,Nmul=0;
double ChemP=0,orbitalOpt_criterion=0,eV2Hartree=27.2114;
int B_spe[30]={},B_l[30]={},B_mul[30]={};char* SpeName[1]={};
double lambda[30]={},U[30]={},J[30]={},Slater_F0[30]={},Slater_F2[30]={},Slater_F4[30]={},Slater_F6[30]={};
double parser_value=0;bool parser_default=false;
int MD_switch=0,atomnum=1;
double Udc=0,Ucoh=0;
double Weight=0,Cell_Volume=1,GP=0,GT=0,Max_Force=0,Total_SpinS=0,BohrR=.529177;
double tv[4][4]={},coordinates[2][32]={};double* Gxyz[]={coordinates[0],coordinates[1]};
char filepath[YOUSO10]="./",filename[YOUSO10]="fixture";
int MD_Init_Velocity=1,TempNum=1;
int fixed_axes[2][4]={};int* atom_Fixed_XYZ[]={fixed_axes[0],fixed_axes[1]};
double unified_atomic_mass_unit=2,electron_mass=1,kB=.5,MD_TimeStep=.25,TempPara[30][3]={};
double rnd(double width){return width*(std::rand()/(double)RAND_MAX-.5);}

std::map<std::string,double> nose_input;
void input_double(const char* key,double* out,double fallback){
  if(!std::strcmp(key,"scf.Init.Mixing.Weight")){
    *out=parser_default?fallback:parser_value;return;
  }
  if(std::strcmp(key,"NH.R") && std::strcmp(key,"NH.nzeta") && std::strcmp(key,"NH.czeta"))
    throw std::runtime_error("input binding used the wrong keyword");
  const auto it=nose_input.find(key);*out=it==nose_input.end()?fallback:it->second;
}
void MPI_Abort(int,int){throw std::runtime_error("unexpected fatal input binding");}
constexpr int MPI_DOUBLE=0,MPI_SUM=0;
void MPI_Comm_rank(int,int* rank){*rank=0;}
void MPI_Comm_size(int,int* size){*size=1;}
void MPI_Allreduce(const double* from,double* to,int count,int,int,int){std::copy_n(from,count,to);}
void MPI_Allreduce(const int* from,int* to,int count,int,int,int){std::copy_n(from,count,to);}
void MPI_Bcast(double*,int,int,int,int){}
constexpr int MPI_INT=1;
#include "mixing_history.inc"

namespace {
void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void equal(double a,double b){require(std::isfinite(a) && std::abs(a-b)<1e-12*(1+std::abs(b)),"simple mixing differs from independent expected value");}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context calculation(){char error[256];Context c(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);require(bool(c),error);return c;}
OpenMXMixingHistory* history(OpenMXCoreContext* c){
  char error[256];OpenMXMixingHistory* h=nullptr;require(openmx_core_mixing_history(c,&h,error,sizeof(error)),error);return h;
}
OpenMXBandEnergies* energies(OpenMXCoreContext* c){
  char error[256];OpenMXBandEnergies* e=nullptr;require(openmx_core_band_energies(c,&e,error,sizeof(error)),error);return e;
}
OpenMXThermodynamicEnergies* totals(OpenMXCoreContext* c){
  char error[256];OpenMXThermodynamicEnergies* e=nullptr;
  require(openmx_core_thermodynamic_energies(c,&e,error,sizeof(error)),error);return e;
}
OpenMXDynamicsState* dynamics(OpenMXCoreContext* c){
  char error[256];OpenMXDynamicsState* out=nullptr;
  require(openmx_core_dynamics(c,&out,error,sizeof(error)),error);return out;
}
OpenMXEnergyComponents* components(OpenMXCoreContext* c){
  char error[256];OpenMXEnergyComponents* e=nullptr;
  require(openmx_core_energy_components(c,&e,error,sizeof(error)),error);return e;
}
void seed_components(OpenMXEnergyComponents* e,double base){
  e->kinetic=base+1;e->hartree[0]=base+2;e->hartree[1]=base+3;
  e->neutral_atom=base+4;e->nonlocal=base+5;e->xc[0]=base+6;e->xc[1]=base+7;
  e->core=base+8;e->hubbard=base+9;e->spin_constraint=base+10;
  e->zeeman_spin=base+11;e->zeeman_orbital=base+12;e->electric_field=base+13;
  e->vdw=base+14;e->core_hole=base+15;e->hartree_double_counting=base+16;
  e->xc_double_counting[0]=base+17;e->xc_double_counting[1]=base+18;e->atomic_reference=base+19;
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
void step(OpenMXMixingHistory* h,OpenMXBandEnergies* energy,int change,double supplied,double difference,double final_weight,int final_counter,double final_max){
  Field c,p,pp,ic,ip,ipp,r,ir;c.active(0)=1+difference;p.active(0)=1;
  energy->band=incoming_energy;Simple_Mixing_DM(h,energy,change,supplied,c.data(),p.data(),pp.data(),ic.data(),ip.data(),ipp.data(),r.data(),ir.data());
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
  auto c=calculation();auto* h=history(c.get());auto* e=energies(c.get());
  for(const auto& x:cases){
    h->consecutive=x.counter;h->adaptive_max_weight=x.cap;for(int i=0;i<5;++i)h->energy[i]=20+i;
    h->energy[0]=x.old_energy;h->energy[1]=x.older_energy;
    h->norm[0]=x.norm;for(int i=1;i<5;++i)h->norm[i]=30+i;
    h->weight=x.weight;incoming_energy=x.energy;step(h,e,x.change,x.supplied,x.difference,x.expected_weight,x.expected_counter,x.expected_cap);
    const double expected[]={x.energy,x.old_energy,x.older_energy,22,23};
    for(int i=0;i<5;++i)equal(h->energy[i],expected[i]);
    equal(h->norm[1],x.norm);equal(h->norm[2],31);equal(h->norm[3],32);equal(h->norm[4],33);
  }
}
void imaginary(){
  auto c=calculation();auto* h=history(c.get());auto* e=energies(c.get());SpinP_switch=3;
  std::array<int*,5> flags{&SO_switch,&Hub_U_switch,&Constraint_NCS_switch,&Zeeman_NCS_switch,&Zeeman_NCO_switch};
  for(int selected=-1;selected<5;++selected){
    for(auto* flag:flags)*flag=0;if(selected>=0)*flags[selected]=1;
    Field real,previous,older,imag,iprevious,iolder,residual,iresidual;
    double norm=0;for(int s=0;s<4;++s){real.active(s)=1+.25*(s+1);previous.active(s)=1;norm+=.0625*(s+1)*(s+1);}
    for(int s=0;s<2;++s){imag.active(s)=2+.125*(s+1);iprevious.active(s)=2;if(selected>=0)norm+=.015625*(s+1)*(s+1);}
    h->consecutive=-1;e->band=incoming_energy;Simple_Mixing_DM(h,e,0,.25,real.data(),previous.data(),older.data(),imag.data(),iprevious.data(),iolder.data(),residual.data(),iresidual.data());
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
    auto a=calculation(),b=calculation();auto* x=history(a.get());auto* y=history(b.get());auto* xe=energies(a.get());auto* ye=energies(b.get());require(x!=y,"independent calculations share history");
    for(int n=0;n<5;++n){
      x->consecutive=-1;x->weight=.2;x->norm[0]=4;incoming_energy=100+10*repeat+n;step(x,xe,0,.25,1,.2,-1,.4);
      y->consecutive=-1;y->weight=.2;y->norm[0]=4;incoming_energy=-100-10*repeat-n;step(y,ye,0,.5,1,.2,-1,.4);
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
  a->norm[0]=4;b->norm[0]=-9;incoming_energy=7.25;double chem[2]={};std::remove(path);
  auto* ea=energies(first.get());auto* eb=energies(second.get());ea->band=7.25;eb->band=-17.5;
  Cnt_switch=0;outputfile1(totals(first.get()),a,ea,components(first.get()),1,1,0,0,2,path,chem,0);
  Cnt_switch=1;outputfile1(totals(second.get()),b,eb,components(second.get()),1,2,3,1,4,path,chem,0);
  FILE* file=std::fopen(path,"r");require(file,"missing production SCF output");
  char buffer[4096];const auto size=std::fread(buffer,1,sizeof(buffer),file);std::fclose(file);
  const std::string text(buffer,size);
  require(text.find("SCF=   2  NormRD=  2.000000000000  Uele=  7.250000000000")!=std::string::npos,"ordinary SCF output read another calculation's residual");
  require(text.find("OrbOpt=  3  SCF=   4  NormRD=  3.000000000000  Uele= -17.500000000000")!=std::string::npos,"orbital SCF output read another calculation's residual");
  equal(a->norm[0],4);equal(b->norm[0],-9);equal(a->weight,.125);equal(b->weight,.3);Cnt_switch=0;
  openmx_core_native_resources_clear(first.get());a=history(first.get());equal(a->weight,0);for(double n:a->norm)equal(n,0);
  equal(b->weight,.3);equal(b->norm[0],-9);parser_default=false;parser_value=.0625;parse_weight(first.get());equal(a->weight,.0625);
  std::remove(path);
}
void md_output(char* path){
  auto first=calculation(),second=calculation();auto* a=energies(first.get());auto* b=energies(second.get());
  *a={7.25,{1.25,3.75},{2.5,4.5}};*b={-17.5,{-11.5,-13.25},{-12.5,-14.5}};
  auto* ac=components(first.get());auto* bc=components(second.get());
  seed_components(ac,100);seed_components(bc,-200);
  auto* ta=totals(first.get());auto* tb=totals(second.get());
  *ta={44.25,2.5,1.25};*tb={-77.5,1.5,-2};
  auto* da=dynamics(first.get());auto* db=dynamics(second.get());
  *da={25.5,30.25,9.5,.125,-.25,110.25};*db={225.5,230.25,-9.5,-.125,.25,-110.25};
  std::remove(path);iterout_md(da,ta,a,ac,1,.25,path);iterout_md(db,tb,b,bc,2,.5,path);
  openmx_core_native_resources_clear(first.get());a=energies(first.get());
  ac=components(first.get());ta=totals(first.get());da=dynamics(first.get());iterout_md(da,ta,a,ac,3,.75,path);
  FILE* file=std::fopen(path,"r");require(file,"missing production MD energy output");
  const double expected[3][6]={{1,.25,1.25,2.5,3.75,7.25},{2,.5,-11.5,-12.5,-13.25,-17.5},{3,.75,0,0,0,0}};
  for(int row=0;row<3;++row)for(int column=0;column<34;++column){
    double value=0;require(std::fscanf(file,"%lf",&value)==1,"MD output column layout changed");
    if(column<6)equal(value,expected[row][column]);
    const double expected_components[3][5]={{106,107,102,103,108},{-194,-193,-198,-197,-192},{0,0,0,0,0}};
    if(column>=6 && column<11)equal(value,expected_components[row][column-6]);
    const double expected_totals[3][3]={{2.5,44.25,46.75},{1.5,-77.5,-76},{0,0,0}};
    if(column>=13 && column<16)equal(value,expected_totals[row][column-13]);
    const double expected_dynamics[3][5]={{30.25,25.5,-.25,.125,110.25},{230.25,225.5,.25,-.125,-110.25},{0,0,0,0,0}};
    if(column>=17 && column<22)equal(value,expected_dynamics[row][column-17]);
  }
  double extra=0;require(std::fscanf(file,"%lf",&extra)==EOF,"MD output added unexpected columns");std::fclose(file);
  equal(b->band,-17.5);equal(b->os[0],-11.5);equal(b->os[1],-13.25);equal(b->is[0],-12.5);equal(b->is[1],-14.5);
  equal(a->band,0);equal(a->os[0],0);equal(a->os[1],0);equal(a->is[0],0);equal(a->is[1],0);
  std::remove(path);
}
void component_output(char* path){
  auto first=calculation(),second=calculation();double chem[2]={};
  auto* a=components(first.get());auto* b=components(second.get());
  seed_components(a,100);seed_components(b,-200);
  *totals(first.get())={42.5,.125,3.25};*totals(second.get())={-42.5,.25,-3.25};
  for(int mode:{2,3})for(int owner=0;owner<3;++owner){
    const auto c=owner==1?second.get():first.get();
    if(owner==2)openmx_core_native_resources_clear(c);
    auto* values=components(c);auto* band=energies(c);band->band=owner==1?-7.5:9.25;
    std::remove(path);outputfile1(totals(c),history(c),band,values,mode,owner+1,0,0,1,path,chem,0);
    std::ifstream file(path);require(bool(file),"missing component energy report");
    std::map<std::string,double> actual;std::string line;
    while(std::getline(file,line)){
      std::istringstream in(line);std::string label,separator;double value;
      if(!(in>>label))continue;
      if(mode==2 && ((!label.empty() && label.back()=='.') || label=="Enpy")){
        if(label.back()=='.')label.pop_back();if(in>>value)actual[label]=value;
      }else if(mode==3 && (in>>separator) && separator=="=" && (in>>value))actual[label]=value;
    }
    const double base=owner==1?-200:100;
    const std::pair<const char*,int> standard[]={{"Ukin",1},{"UH0",2},{"UH1",3},{"Una",4},{"Unl",5},
      {"Uxc0",6},{"Uxc1",7},{"Ucore",8},{"Uhub",9},{"Ucs",10},{"Uzs",11},{"Uzo",12},
      {"Uef",13},{"UvdW",14},{"Uch",15}};
    const std::pair<const char*,int> harris[]={{"UH0",2},{"UH1",3},{"Uxc0",6},{"Uxc1",7},
      {"UH_dc",16},{"Uxc0_dc",17},{"Uxc1_dc",18},{"Ucore",8},{"UvdW",14}};
    auto check=[&](const auto& expected){for(const auto& term:expected){
      require(actual.count(term.first),"energy component output label missing");
      equal(actual.at(term.first),owner==2?0:base+term.second);
    }};
    if(mode==2)check(standard);else check(harris);
    equal(actual.at("Utot"),owner==2?0:owner==1?-42.5:42.5);
    equal(actual.at("UpV"),owner==2?0:owner==1?-3.25:3.25);
    equal(actual.at("Enpy"),owner==2?0:owner==1?-45.75:45.75);equal(actual.at("Uele"),band->band);
    equal(b->kinetic,-199);equal(b->xc_double_counting[1],-182);
    if(owner==2){a=components(first.get());seed_components(a,100);*totals(first.get())={42.5,.125,3.25};}
  }
  std::remove(path);
}

void thermostat_input(){
  auto a=calculation(),b=calculation();auto* x=dynamics(a.get());auto* y=dynamics(b.get());
  nose_input={{"NH.R",1.25},{"NH.nzeta",-.125},{"NH.czeta",.75}};parse_nose(a.get());
  equal(x->nose_coordinate,1.25);equal(x->nose_next_friction,-.125);equal(x->nose_friction,.75);
  nose_input.clear();parse_nose(b.get());equal(y->nose_coordinate,0);equal(y->nose_next_friction,0);equal(y->nose_friction,0);
  y->nose_coordinate=-3.5;y->nose_energy=12.25;
  openmx_core_finish(a.get());x=dynamics(a.get());nose_input={{"NH.nzeta",.375}};parse_nose(a.get());
  equal(x->nose_coordinate,0);equal(x->nose_next_friction,.375);equal(x->nose_friction,0);
  equal(x->temperature,0);equal(x->target_temperature,0);equal(x->nose_energy,0);
  equal(y->nose_coordinate,-3.5);equal(y->nose_energy,12.25);nose_input.clear();
}
void kinetic_updates(){
  auto first=calculation(),second=calculation();auto* a=totals(first.get());auto* b=totals(second.get());
  *a={12.25,-1,2.5};*b={-17.25,-2,-3.5};
  auto* components_a=components(first.get());components_a->kinetic=99;
  auto* da=dynamics(first.get());auto* db=dynamics(second.get());
  MD_switch=1;MD_Init_Velocity=1;fixed_axes[1][2]=1;coordinates[1][20]=2;
  TempNum=1;TempPara[0][1]=0;TempPara[1][1]=10;TempPara[0][2]=300;TempPara[1][2]=400;
  coordinates[1][24]=.2;coordinates[1][25]=.3;coordinates[1][26]=.4;
  InitV(da,a);equal(a->ionic_kinetic,.4);equal(da->temperature,.4/(1.5*.5)*eV2Hartree);
  equal(b->ionic_kinetic,-2);equal(a->total,12.25);equal(a->pressure_volume,2.5);
  Calc_Temp_Atoms(da,a,5);equal(a->ionic_kinetic,.4);equal(da->temperature,.4/.5*eV2Hartree);equal(da->target_temperature,350);
  coordinates[1][24]=.4;coordinates[1][25]=.6;coordinates[1][26]=.8;
  InitV(db,b);Calc_Temp_Atoms(db,b,10);equal(b->ionic_kinetic,1.6);equal(db->temperature,1.6/.5*eV2Hartree);equal(db->target_temperature,400);
  equal(da->target_temperature,350);equal(da->temperature,.4/.5*eV2Hartree);
  equal(a->ionic_kinetic,.4);equal(b->total,-17.25);equal(b->pressure_volume,-3.5);
  equal(components_a->kinetic,99);
  openmx_core_finish(first.get());a=totals(first.get());da=dynamics(first.get());MD_switch=0;MD_Init_Velocity=0;
  InitV(da,a);equal(a->ionic_kinetic,0);equal(a->total,0);equal(a->pressure_volume,0);equal(b->ionic_kinetic,1.6);
  equal(da->temperature,0);equal(da->target_temperature,0);equal(db->target_temperature,400);
  fixed_axes[1][2]=0;
}

}
int main(int argc,char** argv){try{require(argc==2,"expected an output path");golden();imaginary();interleaved();input_and_output(argv[1]);md_output(argv[1]);component_output(argv[1]);thermostat_input();kinetic_updates();std::cout<<"PASS: actual simple mixing, input binding, independent SCF/MD thermodynamic and component output, owned thermostat input and output, actual velocity/temperature reductions, 12 golden trajectories, 6 imaginary-spin modes, 100 interleaved calculations\n";}
catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
