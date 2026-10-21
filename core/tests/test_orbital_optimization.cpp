#include "openmx_core_orbital_optimization.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok,const char* s) {if(!ok)throw std::runtime_error(s);}
void equal(double a,double b,double tolerance=2e-10)
{ require(std::isfinite(a) && std::isfinite(b) && std::abs(a-b)<=tolerance*(1+std::abs(b)),"orbital optimization numerical mismatch"); }
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {char e[256];Context c(openmx_core_create(e,sizeof(e)),openmx_core_destroy);require(bool(c),e);return c;}

// Independent partial-pivot Gauss-Jordan and Jacobi services isolate the
// optimizer from native LAPACK. Native integration exercises the real callbacks.
void inverse_reference(int n,const double* src,size_t stride,double* out,size_t os)
{
  std::vector<long double> a(size_t(n)*2*n);
  auto v=[&](int i,int j)->long double&{return a[size_t(i)*2*n+j];};
  for(int i=0;i<n;i++)for(int j=0;j<n;j++){v(i,j)=src[size_t(i)*stride+j];v(i,n+j)=i==j;}
  for(int k=0;k<n;k++) {
    int pivot=k;for(int i=k+1;i<n;i++)if(fabsl(v(i,k))>fabsl(v(pivot,k)))pivot=i;
    require(fabsl(v(pivot,k))>1e-24L,"singular reference matrix");
    for(int j=0;j<2*n;j++)std::swap(v(k,j),v(pivot,j));
    long double scale=v(k,k);for(int j=0;j<2*n;j++)v(k,j)/=scale;
    for(int i=0;i<n;i++)if(i!=k){scale=v(i,k);for(int j=0;j<2*n;j++)v(i,j)-=scale*v(k,j);}
  }
  for(int i=0;i<n;i++)for(int j=0;j<n;j++)out[size_t(i)*os+j]=double(v(i,n+j));
}
void eigen_reference(double** rows,double* values,int n)
{
  std::vector<double> a(size_t(n)*n),q(size_t(n)*n);
  for(int i=0;i<n;i++)for(int j=0;j<n;j++){a[size_t(i)*n+j]=rows[i+1][j+1];q[size_t(i)*n+j]=i==j;}
  for(int iteration=0;iteration<100*n*n;iteration++) {
    int p=0,r=0;double largest=0;
    for(int i=0;i<n;i++)for(int j=i+1;j<n;j++)if(std::abs(a[size_t(i)*n+j])>largest){largest=std::abs(a[size_t(i)*n+j]);p=i;r=j;}
    if(largest<1e-13)break;
    const double theta=.5*std::atan2(2*a[size_t(p)*n+r],a[size_t(r)*n+r]-a[size_t(p)*n+p]);
    const double c=std::cos(theta),s=std::sin(theta);
    for(int i=0;i<n;i++){double x=a[size_t(i)*n+p],y=a[size_t(i)*n+r];a[size_t(i)*n+p]=c*x-s*y;a[size_t(i)*n+r]=s*x+c*y;}
    for(int j=0;j<n;j++){double x=a[size_t(p)*n+j],y=a[size_t(r)*n+j];a[size_t(p)*n+j]=c*x-s*y;a[size_t(r)*n+j]=s*x+c*y;}
    for(int i=0;i<n;i++){double x=q[size_t(i)*n+p],y=q[size_t(i)*n+r];q[size_t(i)*n+p]=c*x-s*y;q[size_t(i)*n+r]=s*x+c*y;}
  }
  std::vector<int> order(n);std::iota(order.begin(),order.end(),0);
  std::sort(order.begin(),order.end(),[&](int i,int j){return a[size_t(i)*n+i]<a[size_t(j)*n+j];});
  for(int j=0;j<n;j++){values[j+1]=a[size_t(order[j])*n+order[j]];for(int i=0;i<n;i++)rows[i+1][j+1]=q[size_t(i)*n+order[j]];}
}
int inverse_service(void*,int n,const double* a,size_t stride,double* b,size_t bs,double*,double*,int*)
{ inverse_reference(n,a,stride,b,bs);return 1; }
int eigen_service(void*,double** a,double* values,int n) {eigen_reference(a,values,n);return 1;}
OpenMXOrbOptCallbacks services() {OpenMXOrbOptCallbacks c{};c.inverse=inverse_service;c.eigen=eigen_service;return c;}

struct Rows3 {
  std::vector<double> values;
  std::vector<double*> rows;
  std::vector<double**> planes;
  Rows3(int n,int m,int k):values(size_t(n)*m*k),rows(size_t(n)*m),planes(n)
  {for(int i=0;i<n;i++){planes[i]=rows.data()+size_t(i)*m;for(int j=0;j<m;j++)rows[size_t(i)*m+j]=values.data()+(size_t(i)*m+j)*k;}}
};
}

namespace legacy {
using INTEGER=int;
struct LegacyLocalBasisViews {double*** CntCoes;double*** CntCoes_Species;};
double* openmx_local_contraction_row(const LegacyLocalBasisViews* v,size_t a,size_t o){return v->CntCoes[a][o];}
double* openmx_local_species_contraction_row(const LegacyLocalBasisViews* v,size_t a,size_t o){return v->CntCoes_Species[a][o];}
int Matomnum,MatomnumF,SpeciesNum,SpinP_switch,ACnt_switch,SCnt_switch;
int orbitalOpt_History=2,orbitalOpt_StartPulay=1,OrbOpt_OptMethod=2,Host_ID=0,level_stdout=0,memoryusage_fileout=0;
int *M2G,*F_M2G,*F_G2M,*WhatSpecies,*FNAN,**natn,*Spe_Total_NO,*Spe_Total_CNO,*Spe_MaxL_Basis;
int **Spe_Num_CBasis,**Spe_Specified_Num,***Spe_Trans_Orbital,*F_Snd_Num,*F_Rcv_Num,*F_TopMAN,**Snd_MAN,**Snd_GAN,**Rcv_GAN;
int List_YOUSO[60]={},mpi_comm_level1=0;
double orbitalOpt_SD_step=.01,Oopt_NormD[10]={};
double ****His_CntCoes,****His_D_CntCoes,****His_CntCoes_Species,****His_D_CntCoes_Species,**OrbOpt_Hessian,*His_OrbOpt_Etot;
using MPI_Status=int;using MPI_Request=int;
enum {MPI_DOUBLE,MPI_INT,MPI_SUM,MPI_MAX};
void MPI_Comm_size(int,int* n){*n=1;}void MPI_Comm_rank(int,int* n){*n=0;}void MPI_Barrier(int){}
void MPI_Allreduce(const void* a,void* b,int n,int type,int,int){std::memcpy(b,a,n*(type==MPI_DOUBLE?sizeof(double):sizeof(int)));}
void MPI_Bcast(void*,int,int,int,int){}void MPI_Isend(const void*,int,int,int,int,int,int*){}
void MPI_Recv(void*,int,int,int,int,int,int*){}void MPI_Wait(int*,int*){}
void PrintMemory(const char*,size_t,const char*){}
void Eigen_lapack(double** a,double* k,int n,int){eigen_reference(a,k,n);}
void dgetrf(int*,int*,double*,int*,int*,int* info){*info=0;}
void dgetri(int* n,double* a,int*,int*,double*,int*,int* info)
{std::vector<double> copy(a,a+size_t(*n)*(*n));inverse_reference(*n,copy.data(),*n,a,*n);*info=0;}
#define F77_NAME(lower,upper) lower
#include "fixtures/orbital_optimization_legacy.inc"
#undef F77_NAME
}

namespace {
struct Fixture {
  int species[3]{-1,0,1},orbs[2]{2,3},cno[2]{1,1},lmax[2]{0,0},cb[2]{1,1},projectors[2]{1,1};
  int map[3]{-1,2,1},inverse[3]{-1,2,1},fnan[3]{-1,1,1},n1[2]{1,2},n2[2]{2,1};
  int* neighbors[3]{nullptr,n1,n2};
  int counts[2][1]{{2},{3}},*count_rows[2]{counts[0],counts[1]};
  int indices[2][1][3]{{{0,1,0}},{{2,0,1}}};
  int* index_rows[2][1]{{indices[0][0]},{indices[1][0]}},**index_species[2]{index_rows[0],index_rows[1]};
  int sends[2]{0,1},receives[2]{0,1},receive_first[2]{0,2},send_local[1]{1},send_global[1]{2},receive_global[1]{1};
  int *send_locals[2]{nullptr,send_local},*send_globals[2]{nullptr,send_global},*receive_globals[2]{nullptr,receive_global};
  OpenMXSpeciesViews sp{};OpenMXOrbOptInput input{};OpenMXOrbOptViews views{};
  OpenMXElectronicViews e{};OpenMXLocalBasisViews l{};char error[256]{};
  Fixture(int mode=0,int spin=0) {
    sp.species=2;sp.angular_channels=1;sp.orbitals=orbs;sp.contracted_orbitals=cno;sp.basis_lmax=lmax;sp.contracted_basis=cb;
    input.atoms=2;input.local_atoms=2;input.max_orbitals=3;input.max_coefficients=3;
    input.history=2;input.spin=spin;input.species_mode=mode;input.method=2;input.start_pulay=1;input.ranks=1;input.descent_step=.01;
    input.species=&sp;input.atom_species=species;input.local_to_global=input.first_to_global=map;input.global_to_first=inverse;
    input.neighbors=fnan;input.neighbor_atoms=neighbors;input.coefficient_counts=count_rows;input.orbital_indices=index_species;
    input.send_counts=sends;input.receive_counts=receives;input.receive_first=receive_first;
    input.send_local_atoms=send_locals;input.send_global_atoms=send_globals;input.receive_global_atoms=receive_globals;
  }
  bool begin(OpenMXCoreContext* c){return openmx_core_orbopt_begin(c,&input,&views,error,sizeof(error));}
  void initialize(OpenMXCoreContext* c) {
    OpenMXElectronicMatrixInput i{};
    i.atoms=2;i.local_atoms=input.local_atoms;i.first_halo_atoms=input.first_halo_atoms;i.species=2;i.spin=input.spin;
    i.mixing=1;i.nonlocal_components=3;i.density_history=2;i.hamiltonian_history=2;
    i.scratch_orbitals=3;i.scratch_neighbors=2;i.scratch_projectors=3;i.contracted=1;
    i.atom_species=species;i.orbitals=orbs;i.contracted_orbitals=cno;i.projector_components=projectors;
    i.local_to_global=i.first_to_global=i.second_to_global=map;i.neighbors=fnan;i.neighbor_atoms=neighbors;
    require(openmx_core_electronic_allocate(c,&i,&e,error,sizeof(error)),error);
    OpenMXLocalBasisInput b{};b.atoms=2;b.local_atoms=input.local_atoms;b.first_halo_atoms=input.first_halo_atoms;b.species=2;
    b.spin=input.spin;b.max_orbitals=b.max_coefficients=3;b.contraction=1;
    b.atom_species=species;b.orbitals=orbs;b.contracted_orbitals=cno;b.local_to_global=b.first_to_global=map;
    require(openmx_core_local_basis_allocate(c,&b,&l,error,sizeof(error)),error);
    for(int a=1;a<3;a++) {
      int w=species[map[a]];
      for(int p=0;p<counts[w][0];p++) {
        openmx_local_contraction_row(&l,a,0)[p]=.7-.17*p+.03*w;
        openmx_local_species_contraction_row(&l,w,0)[p]=.7-.17*p+.03*w;
      }
    }
    fill(0);require(begin(c),error);
  }
  void fill(int iteration) {
    for(int a=1;a<=input.local_atoms;a++)for(int h=0;h<=1;h++) {
      const int g=map[a],n=neighbors[g][h],r=orbs[species[g]],cols=orbs[species[n]];
      for(int i=0;i<r;i++)for(int j=0;j<cols;j++) {
        e.OLP[0][a][h][i][j]=g==n ? (i==j?1.3+.1*i:.02):.03/(1+std::abs(i-j));
        for(int s=0;s<=input.spin;s++)e.H[s][a][h][i][j]=(.7+.15*s)*(std::cos((g+n)*.4+(i+j)*.17+iteration*.13)+(g==n && i==j?2.:0.));
      }
      for(int s=0;s<=input.spin;s++){e.DM[0][s][a][h][0][0]=.25+.02*(g+n)+.03*s;e.EDM[s][a][h][0][0]=.04+.01*s;}
    }
  }
  void bind_legacy() {
    using namespace legacy;
    Matomnum=input.local_atoms;MatomnumF=input.first_halo_atoms;SpeciesNum=2;SpinP_switch=input.spin;ACnt_switch=input.species_mode;
    SCnt_switch=input.last_radial_only;OrbOpt_OptMethod=input.method;orbitalOpt_History=input.history;orbitalOpt_StartPulay=input.start_pulay;
    M2G=F_M2G=map;F_G2M=inverse;WhatSpecies=species;FNAN=fnan;natn=neighbors;Spe_Total_NO=orbs;Spe_Total_CNO=cno;Spe_MaxL_Basis=lmax;
    static int* cbrows[2];cbrows[0]=cb;cbrows[1]=cb+1;Spe_Num_CBasis=cbrows;
    Spe_Specified_Num=count_rows;Spe_Trans_Orbital=index_species;List_YOUSO[7]=List_YOUSO[24]=3;
  }
};

void reference_test() {
  for(int mode:{0,1})for(int spin:{0,1})for(int method:{1,2}) {
    auto c=context();Fixture f(mode,spin);f.input.method=method;f.initialize(c.get());f.bind_legacy();
    Rows3 atoms(3,3,3),species(3,3,3);
    std::copy_n(f.l.tensors[OPENMX_LOCAL_CONTRACTION].values,27,atoms.values.data());
    std::copy_n(f.l.tensors[OPENMX_LOCAL_SPECIES_CONTRACTION].values,27,species.values.data());
    legacy::LegacyLocalBasisViews old{atoms.planes.data(),species.planes.data()};
    legacy::Allocate_Free_OrbOpt(1);auto callbacks=services();
    for(int iteration=1;iteration<=4;iteration++) {
      f.fill(iteration);double norm=-1;
      double expected=legacy::Opt_Contraction(iteration,0,f.e.H,f.e.OLP,f.e.DM[0],f.e.EDM,
        legacy::His_CntCoes,legacy::His_D_CntCoes,legacy::His_CntCoes_Species,legacy::His_D_CntCoes_Species,
        legacy::His_OrbOpt_Etot,legacy::OrbOpt_Hessian,&old);
      require(openmx_core_orbopt_step(c.get(),iteration,0,&f.e,&f.l,&callbacks,&norm,f.error,sizeof(f.error)),f.error);equal(norm,expected);
      for(int i=0;i<27;i++){equal(f.l.tensors[2].values[i],atoms.values[i]);equal(f.l.tensors[3].values[i],species.values[i]);}
      const auto& hc=mode ? f.views.species_coefficients:f.views.atomic_coefficients;
      const auto& hd=mode ? f.views.species_gradients:f.views.atomic_gradients;
      auto oc=mode ? legacy::His_CntCoes_Species:legacy::His_CntCoes;
      auto od=mode ? legacy::His_D_CntCoes_Species:legacy::His_D_CntCoes;
      const int history=std::min(2,iteration);
      for(int h=0;h<history;h++)for(int a=mode?0:1;a<(mode?2:3);a++) {
        const int w=mode?a:f.species[f.map[a]];
        for(int p=0;p<f.counts[w][0];p++) {
          size_t offset=((size_t(h)*3+a)*3)*3+p;equal(hc.values[offset],oc[h][a][0][p]);equal(hd.values[offset],od[h][a][0][p]);
        }
      }
      if(mode)for(int i=0;i<5;i++)for(int j=0;j<5;j++)equal(f.views.hessian.values[size_t(i)*f.views.hessian.columns+j],legacy::OrbOpt_Hessian[i][j]);
    }
    legacy::Allocate_Free_OrbOpt(2);
  }
}

long double energy(Fixture& f) {
  long double value=0;
  for(int a=1;a<=2;a++)for(int h=0;h<2;h++) {
    const int g=f.map[a],ng=f.neighbors[g][h],b=f.inverse[ng],wa=f.species[g],wb=f.species[ng];
    for(int p=0;p<f.counts[wa][0];p++)for(int q=0;q<f.counts[wb][0];q++) {
      const int i=f.indices[wa][0][p],j=f.indices[wb][0][q];
      const double ca=openmx_local_contraction_row(&f.l,a,0)[p],cb=openmx_local_contraction_row(&f.l,b,0)[q];
      for(int s=0;s<=f.input.spin;s++)value+=(f.input.spin?1:2)*(long double)ca*cb*
        (f.e.DM[0][s][a][h][0][0]*f.e.H[s][a][h][i][j]-f.e.EDM[s][a][h][0][0]*f.e.OLP[0][a][h][i][j]);
    }
  }
  return value;
}
// Two radial p functions exercise tied magnetic coefficients, species sums
// across atoms, and the last-radial-only constraint independently of legacy code.
void angular_gradient_test() {
  for(int mode:{0,1})for(int spin:{0,1})for(int last:{0,1}) {
    auto c=context();char error[256];
    int orbitals[1]={8},contracted[1]={7},lmax[1]={1},basis[2]={1,2};
    int species[3]={0,0,0},map[3]={0,2,1},inverse[3]={0,2,1},neighbors[3]={0,1,1};
    int row1[2]={1,2},row2[2]={2,1},*rows[3]={nullptr,row1,row2};
    int counts[7]={2,2,2,2,2,2,2},*count_rows[1]={counts},indices[7][2];
    int *index_rows[7],**index_species[1]={index_rows};
    for(int i=0;i<7;i++){index_rows[i]=indices[i];for(int p=0;p<2;p++)indices[i][p]=i==0?p:2+(i-1)%3+3*p;}
    OpenMXSpeciesViews sp{};sp.species=1;sp.angular_channels=2;sp.orbitals=orbitals;
    sp.contracted_orbitals=contracted;sp.basis_lmax=lmax;sp.contracted_basis=basis;
    OpenMXOrbOptInput in{};in.atoms=in.local_atoms=2;in.max_orbitals=8;in.max_coefficients=2;
    in.history=2;in.spin=spin;in.species_mode=mode;in.last_radial_only=last;in.method=2;in.start_pulay=1;
    in.ranks=1;in.descent_step=.01;in.species=&sp;in.atom_species=species;
    in.local_to_global=in.first_to_global=map;in.global_to_first=inverse;in.neighbors=neighbors;
    in.neighbor_atoms=rows;in.coefficient_counts=count_rows;in.orbital_indices=index_species;
    OpenMXOrbOptViews views{};require(openmx_core_orbopt_begin(c.get(),&in,&views,error,sizeof(error)),error);
    counts[2]=1;
    require(!openmx_core_orbopt_begin(c.get(),&in,&views,error,sizeof(error)),"unequal magnetic channel counts accepted");
    counts[2]=2;
    OpenMXLocalBasisInput b{};b.atoms=b.local_atoms=2;b.species=1;b.spin=spin;
    b.max_orbitals=8;b.max_coefficients=2;b.contraction=1;b.atom_species=species;b.orbitals=orbitals;
    b.contracted_orbitals=contracted;b.local_to_global=b.first_to_global=map;
    OpenMXLocalBasisViews local{};require(openmx_core_local_basis_allocate(c.get(),&b,&local,error,sizeof(error)),error);
    OpenMXElectronicMatrixInput e{};e.atoms=e.local_atoms=2;e.species=1;e.spin=spin;e.mixing=1;
    e.nonlocal_components=3;e.density_history=e.hamiltonian_history=2;e.scratch_orbitals=8;
    e.scratch_neighbors=2;e.scratch_projectors=3;e.contracted=1;e.atom_species=species;e.orbitals=orbitals;
    e.contracted_orbitals=contracted;int projectors[1]={1};e.projector_components=projectors;
    e.local_to_global=e.first_to_global=e.second_to_global=map;e.neighbors=neighbors;e.neighbor_atoms=rows;
    OpenMXElectronicViews electronic{};require(openmx_core_electronic_allocate(c.get(),&e,&electronic,error,sizeof(error)),error);
    for(int a=1;a<=2;a++)for(int i=0;i<7;i++)for(int p=0;p<2;p++)
      openmx_local_contraction_row(&local,a,i)[p]=.6+.1*p+.02*(i==0?0:1+(i-1)/3)+(mode?0:.03*a);
    for(int a=1;a<=2;a++)for(int h=0;h<2;h++) {
      const int g=map[a],n=rows[g][h];
      for(int i=0;i<8;i++)for(int j=0;j<8;j++) {
        electronic.OLP[0][a][h][i][j]=g==n && i==j?1.2+.01*i:.03;
        for(int s=0;s<=spin;s++)electronic.H[s][a][h][i][j]=.2*std::cos(.2*(g+n)+.13*(i+j))+.03*s;
      }
      for(int s=0;s<=spin;s++)for(int i=0;i<7;i++)for(int j=0;j<7;j++) {
        electronic.DM[0][s][a][h][i][j]=.2+.01*(i+j+g+n)+.04*s;
        electronic.EDM[s][a][h][i][j]=.03+.001*(i+j+g+n)+.01*s;
      }
    }
    auto energy=[&] {
      long double result=0;
      for(int a=1;a<=2;a++)for(int h=0;h<2;h++)for(int i=0;i<7;i++)for(int j=0;j<7;j++)
        for(int p=0;p<2;p++)for(int q=0;q<2;q++)for(int s=0;s<=spin;s++) {
          const int other=inverse[rows[map[a]][h]],x=indices[i][p],y=indices[j][q];
          result+=(spin?1:2)*(long double)openmx_local_contraction_row(&local,a,i)[p]*
            openmx_local_contraction_row(&local,other,j)[q]*(electronic.DM[0][s][a][h][i][j]*
            electronic.H[s][a][h][x][y]-electronic.EDM[s][a][h][i][j]*electronic.OLP[0][a][h][x][y]);
        }
      return result;
    };
    auto callbacks=services();double norm;
    require(openmx_core_orbopt_gradient(c.get(),&electronic,&local,&callbacks,&norm,error,sizeof(error)),error);
    double squared=0;const int first[3]={0,1,4},magnetic[3]={1,3,3};
    for(int a=1;a<=(mode?1:2);a++)for(int group=0;group<3;group++)for(int p=0;p<2;p++) {
      auto shift=[&](double delta) {
        for(int atom=mode?1:a;atom<=(mode?2:a);atom++)for(int m=0;m<magnetic[group];m++)
          openmx_local_contraction_row(&local,atom,first[group]+m)[p]+=delta;
      };
      const double h=1e-5;shift(h);const auto plus=energy();shift(-2*h);const auto minus=energy();shift(h);
      const double expected=last && group==1?0:double((plus-minus)/(2*h));
      for(int m=0;m<magnetic[group];m++) {
        double actual=openmx_local_tensor_row(mode?&views.species_gradient:&views.atomic_gradient,mode?0:a,first[group]+m)[p];
        equal(actual,expected,2e-9);squared+=actual*actual;
      }
    }
    equal(norm,squared);
  }
}
void gradient_test() {
  for(int mode:{0,1})for(int spin:{0,1}) {
    auto c=context();Fixture f(mode,spin);f.initialize(c.get());auto callbacks=services();double norm;
    require(openmx_core_orbopt_gradient(c.get(),&f.e,&f.l,&callbacks,&norm,f.error,sizeof(f.error)),f.error);
    double squared=0;
    for(int a=1;a<=2;a++)for(int p=0;p<f.counts[f.species[f.map[a]]][0];p++) {
      double& x=openmx_local_contraction_row(&f.l,a,0)[p];const double old=x,h=1e-5;
      x=old+h;const auto plus=energy(f);x=old-h;const auto minus=energy(f);x=old;
      const double expected=double((plus-minus)/(2*h));const double actual=openmx_local_tensor_row(&f.views.atomic_gradient,a,0)[p];
      equal(actual,expected,2e-9);squared+=actual*actual;
      if(mode)equal(openmx_local_tensor_row(&f.views.species_gradient,f.species[f.map[a]],0)[p],actual);
    }
    equal(norm,squared);
  }
  angular_gradient_test();
}
void layouts_test() {
  for(int mode:{0,1}) {
    auto c=context();Fixture f(mode);require(f.begin(c.get()),f.error);
    auto h=mode ? f.views.species_coefficients:f.views.atomic_coefficients;
    require(h.extent[0]==3 && h.extent[1]==3 && h.extent[2]==3 && h.extent[3]==3,"history legacy dimensions");
    for(size_t i=0;i<81;i++)equal(h.values[i],0);
    require(f.views.hessian.rows==size_t(mode ? 7:0) && f.views.energy_count==size_t(mode ? 4:0),"unused Hessian/history allocated");
    require(f.views.atomic_gradient.n0==3 && f.views.atomic_gradient.n1==3 && f.views.atomic_gradient.n2==3,"gradient dimensions");
  }
}
void ownership_test() {
  auto c=context();OpenMXElectronicViews e{};OpenMXLocalBasisViews l{};OpenMXOrbOptViews kept{};
  {
    Fixture f;f.initialize(c.get());e=f.e;l=f.l;kept=f.views;
    f.map[1]=-1;f.species[1]=-1;f.counts[0][0]=-1;f.indices[1][0][0]=-1;
  }
  auto callbacks=services();char error[256];double norm;
  require(openmx_core_orbopt_gradient(c.get(),&e,&l,&callbacks,&norm,error,sizeof(error)),error);require(norm>0,"metadata lifetime leaked into optimizer");
  OpenMXOrbOptViews current{};require(openmx_core_orbopt_get(c.get(),&current,error,sizeof(error)),error);
  require(current.atomic_coefficients.values==kept.atomic_coefficients.values,"lookup reallocates history");
}
void invalid_test() {
  auto c=context();
  for(int bad=0;bad<18;bad++) {
    Fixture f;require(f.begin(c.get()),f.error);auto saved=f.views;
    saved.atomic_coefficients.values[0]=17;
    switch(bad) {
      case 0:f.input.atoms=-1;break;case 1:f.input.local_atoms=3;break;case 2:f.input.first_halo_atoms=INT_MAX;break;
      case 3:f.input.history=0;break;case 4:f.input.spin=3;break;case 5:f.input.species_mode=2;break;
      case 6:f.input.method=0;break;case 7:f.input.start_pulay=0;break;case 8:f.input.ranks=0;break;
      case 9:f.input.rank=1;break;case 10:f.input.species=nullptr;break;case 11:f.counts[0][0]=4;break;
      case 12:f.indices[0][0][1]=2;break;case 13:f.map[1]=3;break;case 14:f.inverse[1]=0;break;
      case 15:f.input.atom_species=nullptr;break;case 16:f.cb[0]=2;break;
      case 17:f.input.atoms=INT_MAX;break;
    }
    auto live=openmx::test::live_allocations();require(!f.begin(c.get()) && f.error[0],"invalid optimization input accepted");
    require(live==openmx::test::live_allocations() && !std::memcmp(&f.views,&saved,sizeof(saved)),"failed begin leaks/changes output");
    OpenMXOrbOptViews got{};require(openmx_core_orbopt_get(c.get(),&got,f.error,sizeof(f.error)),f.error);equal(got.atomic_coefficients.values[0],17);
  }
}
void allocation_test() {
  auto c=context();Fixture f(1);require(f.begin(c.get()),f.error);
  openmx::test::fail_allocation_after(10000);int ok=f.begin(c.get());auto count=10000-openmx::test::remaining_allocations();openmx::test::fail_allocation_after(-1);
  require(ok && count>20,"allocation interception unavailable");auto saved=f.views;saved.species_coefficients.values[0]=19;
  for(long fault=0;fault<count;fault++) {
    auto live=openmx::test::live_allocations();openmx::test::fail_allocation_after(fault);ok=f.begin(c.get());openmx::test::fail_allocation_after(-1);
    require(!ok && live==openmx::test::live_allocations() && !std::memcmp(&saved,&f.views,sizeof(saved)),"partial begin leaks/changes output");equal(saved.species_coefficients.values[0],19);
  }
}
void lifecycle_test() {
  auto c=context();Fixture f;require(openmx_core_orbopt_get(c.get(),&f.views,f.error,sizeof(f.error)) && !f.views.atomic_coefficients.values,"initial optimizer not empty");
  for(int i=0;i<6;i++){f.input.species_mode=i%2;require(f.begin(c.get()),f.error);openmx_core_orbopt_end(c.get());openmx_core_orbopt_end(c.get());}
  require(f.begin(c.get()),f.error);openmx_core_finish(c.get());require(!f.begin(c.get()) && !openmx_core_orbopt_get(c.get(),&f.views,f.error,sizeof(f.error)),"finished context accepted");
}
void concurrent_test() {
  int results[4]{};std::thread jobs[4];
  for(int i=0;i<4;i++)jobs[i]=std::thread([&,i]{try{auto c=context();Fixture f;f.initialize(c.get());auto cb=services();double norm;
    for(int j=0;j<4;j++)require(openmx_core_orbopt_gradient(c.get(),&f.e,&f.l,&cb,&norm,f.error,sizeof(f.error)),f.error);results[i]=1;
  }catch(...){results[i]=-1;}});
  for(auto& t:jobs)t.join();for(auto ok:results)require(ok==1,"independent optimizer contexts failed");
}
void stationary_test() {
  auto c=context();Fixture f(1);f.input.method=1;f.initialize(c.get());
  for(int a=1;a<=2;a++)for(int h=0;h<2;h++){f.e.DM[0][0][a][h][0][0]=0;f.e.EDM[0][a][h][0][0]=0;}
  auto callbacks=services();auto before=std::vector<double>(f.l.tensors[2].values,f.l.tensors[2].values+27);
  double norm=99;require(openmx_core_orbopt_step(c.get(),2,0,&f.e,&f.l,&callbacks,&norm,f.error,sizeof(f.error)),f.error);
  equal(norm,0);require(std::equal(before.begin(),before.end(),f.l.tensors[2].values),"stationary gradient changes coefficients");

  // The old species DIIS branch left its descent coefficient uninitialized
  // below squared norm 1e-6. Check the explicit 0.4 continuation at that limit.
  Fixture tiny(1);tiny.input.method=1;tiny.initialize(c.get());
  auto small_operators=[&](int iteration) {
    tiny.fill(iteration);
    for(int a=1;a<=2;a++)for(int h=0;h<2;h++) {
      const int r=tiny.orbs[tiny.species[tiny.map[a]]],n=tiny.orbs[tiny.species[tiny.neighbors[tiny.map[a]][h]]];
      for(int i=0;i<r;i++)for(int j=0;j<n;j++)tiny.e.H[0][a][h][i][j]*=1e-6;
      tiny.e.EDM[0][a][h][0][0]=0;
    }
  };
  small_operators(1);
  require(openmx_core_orbopt_step(c.get(),1,0,&tiny.e,&tiny.l,&callbacks,&norm,tiny.error,sizeof(tiny.error)),tiny.error);
  small_operators(2);
  require(openmx_core_orbopt_gradient(c.get(),&tiny.e,&tiny.l,&callbacks,&norm,tiny.error,sizeof(tiny.error)),tiny.error);
  require(norm>0 && norm<1e-6,"small-gradient fixture missed the DIIS limit");
  double matrix[9]={0,0,1,0,0,1,1,1,0},inverse[9];
  for(int w=0;w<2;w++)for(int p=0;p<tiny.counts[w][0];p++) {
    const auto offset=size_t(w)*9+p;
    const double current=tiny.views.species_gradient.values[offset],previous=tiny.views.species_gradients.values[offset];
    matrix[0]+=current*current;matrix[1]+=current*previous;matrix[3]+=previous*current;matrix[4]+=previous*previous;
  }
  inverse_reference(3,matrix,3,inverse,3);
  double expected[2][3]{};
  for(int w=0;w<2;w++) {
    for(int p=0;p<tiny.counts[w][0];p++) {
      const auto offset=size_t(w)*9+p;
      expected[w][p]=inverse[2]*(tiny.l.tensors[3].values[offset]-.4*tiny.views.species_gradient.values[offset])+
        inverse[5]*(tiny.views.species_coefficients.values[offset]-.4*tiny.views.species_gradients.values[offset]);
    }
    const int atom=tiny.inverse[w+1];double length=0;
    for(int p=0;p<tiny.counts[w][0];p++)for(int q=0;q<tiny.counts[w][0];q++)
      length+=expected[w][p]*expected[w][q]*tiny.e.OLP[0][atom][0][tiny.indices[w][0][p]][tiny.indices[w][0][q]];
    for(int p=0;p<tiny.counts[w][0];p++)expected[w][p]/=std::sqrt(length);
  }
  require(openmx_core_orbopt_step(c.get(),2,0,&tiny.e,&tiny.l,&callbacks,&norm,tiny.error,sizeof(tiny.error)),tiny.error);
  for(int w=0;w<2;w++)for(int p=0;p<tiny.counts[w][0];p++)
    equal(openmx_local_species_contraction_row(&tiny.l,w,0)[p],expected[w][p],2e-12);
}
struct Transport {int reductions=0,exchanges=0;};
int fake_sum(void* u,const double* a,double* b,int n){((Transport*)u)->reductions++;std::copy_n(a,n,b);return 1;}
int fake_exchange(void* u,int to,const double* send,int ns,int from,double* receive,int nr)
{require(to==1 && from==1 && ns==3 && nr==2 && std::isfinite(send[0]),"bad halo message plan");receive[0]=1.2;receive[1]=-.7;((Transport*)u)->exchanges++;return 1;}
void exchange_test() {
  auto c=context();Fixture f;f.input.local_atoms=1;f.input.first_halo_atoms=1;f.input.ranks=2;f.initialize(c.get());
  Transport t;auto cb=services();cb.user=&t;cb.sum=fake_sum;cb.exchange=fake_exchange;double norm;
  require(openmx_core_orbopt_step(c.get(),1,0,&f.e,&f.l,&cb,&norm,f.error,sizeof(f.error)),f.error);
  require(t.reductions && t.exchanges==1,"halo exchange missing");equal(openmx_local_contraction_row(&f.l,2,0)[0],1.2);equal(openmx_local_contraction_row(&f.l,2,0)[1],-.7);
}
void services_test() {
  auto c=context();Fixture f;f.input.local_atoms=1;f.input.first_halo_atoms=1;f.input.ranks=2;f.initialize(c.get());
  OpenMXOrbOptCallbacks cb{};double norm=37;
  require(!openmx_core_orbopt_step(c.get(),1,0,&f.e,&f.l,&cb,&norm,f.error,sizeof(f.error)) && f.error[0],"missing reduction callback accepted");equal(norm,37);
  f.e.fields[OPENMX_ELECTRONIC_HAMILTONIAN].components=0;
  require(!openmx_core_orbopt_gradient(c.get(),&f.e,&f.l,&cb,&norm,f.error,sizeof(f.error)),"short operator accepted");equal(norm,37);
}
void scratch_test() {
  const size_t shape[]{9,4,4};OpenMXLocalTensor3 view{};char error[256];
  auto* owner=openmx_core_local_tensor_create(shape,&view,error,sizeof(error));require(owner,error);
  for(int i=0;i<144;i++)equal(view.values[i],0);openmx_local_tensor_row(&view,8,3)[3]=17;equal(view.values[143],17);
  auto saved=view;const size_t bad[]{SIZE_MAX,2,2};require(!openmx_core_local_tensor_create(bad,&view,error,sizeof(error)) && !std::memcmp(&saved,&view,sizeof(view)),"scratch overflow changes view");
  openmx_core_local_tensor_destroy(owner);openmx_core_local_tensor_destroy(nullptr);
}
}

int main(int argc,char** argv) {
  try {
    std::string test=argc==2?argv[1]:"";
    if(test=="reference")reference_test();else if(test=="gradient")gradient_test();else if(test=="layouts")layouts_test();
    else if(test=="ownership")ownership_test();else if(test=="invalid")invalid_test();else if(test=="allocation")allocation_test();
    else if(test=="lifecycle")lifecycle_test();else if(test=="concurrent")concurrent_test();else if(test=="stationary")stationary_test();
    else if(test=="exchange")exchange_test();else if(test=="services")services_test();else if(test=="scratch")scratch_test();else throw std::runtime_error("unknown optimizer test");
    std::cout<<"PASS orbital optimization "<<test<<'\n';return 0;
  }catch(const std::exception& e){openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
