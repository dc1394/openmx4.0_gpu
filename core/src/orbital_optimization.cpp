#include <openmx/core/orbital_optimization.hpp>
#include <openmx/core/tensor.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace openmx::core {
namespace {
template<class T,std::size_t N,class... I> decltype(auto) at(Tensor<T,N>& a,I... i)
{ return element(a.view(),i...); }
template<class T,std::size_t N,class... I> decltype(auto) at(const Tensor<T,N>& a,I... i)
{ return element(a.view(),i...); }
double& at(const OpenMXLocalTensor3& a,std::size_t i,std::size_t j,std::size_t k)
{ return a.values[(i*a.n1+j)*a.n2+k]; }
OpenMXLocalTensor3 view3(Tensor<double,3>& a)
{ return {a.extent(0),a.extent(1),a.extent(2),a.data()}; }
OpenMXOrbOptHistory history_view(Tensor<double,4>& a)
{ return {{a.extent(0),a.extent(1),a.extent(2),a.extent(3)},a.data()}; }
struct Operator {
  const OpenMXElectronicMatrixView& view;
  double operator()(std::size_t s,std::size_t a,std::size_t n,std::size_t i,std::size_t j) const
  { return openmx_electronic_row(&view,s,a,n,i)[j]; }
};
struct Session {
  const OpenMXLocalBasisViews *local_basis;
  const OpenMXOrbOptCallbacks& callbacks;
  Operator H,OLP,CDM,EDM;
  Session(const OpenMXElectronicViews& e,const OpenMXLocalBasisViews& l,const OpenMXOrbOptCallbacks& c)
    : local_basis(&l),callbacks(c),H{e.fields[OPENMX_ELECTRONIC_HAMILTONIAN]},
      OLP{e.fields[OPENMX_ELECTRONIC_OVERLAP]},CDM{e.fields[OPENMX_ELECTRONIC_DENSITY]},
      EDM{e.fields[OPENMX_ELECTRONIC_ENERGY_DENSITY]} {}
};
void require(bool ok,const char* what) { if(!ok) throw std::invalid_argument(what); }
void service(bool ok) { if(!ok) throw std::runtime_error("orbital optimization service failed"); }
std::vector<int> copy(const int* p,std::size_t n)
{ require(!n || p,"missing orbital optimization metadata");return n ? std::vector<int>(p,p+n):std::vector<int>{}; }
int dimension(const OpenMXOrbOptInput& in)
{
  require(in.atoms>=0 && in.atoms<INT_MAX && in.local_atoms>=0 && in.local_atoms<=in.atoms && in.first_halo_atoms>=0 &&
      in.local_atoms<=INT_MAX-2-in.first_halo_atoms && in.max_orbitals>=0 && in.max_coefficients>=0 &&
      in.history>0 && in.history<INT_MAX-3 && (in.spin==0 || in.spin==1) &&
      (in.species_mode==0 || in.species_mode==1) && (in.last_radial_only==0 || in.last_radial_only==1) &&
      (in.method==1 || in.method==2) && in.start_pulay>=1 && in.ranks>0 && in.rank>=0 && in.rank<in.ranks &&
      in.host_rank>=0 && in.host_rank<in.ranks && std::isfinite(in.descent_step) && in.descent_step>=0 && in.species,
      "invalid orbital optimization dimensions or options");
  const auto& s=*in.species;
  require(s.species<=INT_MAX-1 && s.angular_channels<=INT_MAX &&
      (!s.species || (s.orbitals && s.contracted_orbitals && s.basis_lmax && s.contracted_basis && in.coefficient_counts && in.orbital_indices)),
      "missing orbital optimization species metadata");
  int total=0;
  for(std::size_t w=0;w<s.species;++w) {
    require(s.orbitals[w]>=0 && s.orbitals[w]<=in.max_orbitals && s.contracted_orbitals[w]>=0 &&
        s.contracted_orbitals[w]<=s.orbitals[w] && s.basis_lmax[w]>=0 && s.basis_lmax[w]<=14 && std::size_t(s.basis_lmax[w])<s.angular_channels &&
        (!s.contracted_orbitals[w] || (in.coefficient_counts[w] && in.orbital_indices[w])),"invalid orbital optimization species shape");
    int channels=0;
    for(int l=0;l<=s.basis_lmax[w];++l) {
      const int n=s.contracted_basis[w*s.angular_channels+l];
      require(n>=0 && n<=INT_MAX/(2*l+1) && channels<=INT_MAX-n*(2*l+1),"invalid contracted angular count");
      channels+=n*(2*l+1);
    }
    require(channels==s.contracted_orbitals[w],"contracted angular/orbital counts differ");
    int orbital=0;
    for(int l=0;l<=s.basis_lmax[w];++l)
      for(int radial=0;radial<s.contracted_basis[w*s.angular_channels+l];++radial) {
        for(int m=1;m<=2*l;++m)
          require(in.coefficient_counts[w][orbital+m]==in.coefficient_counts[w][orbital],
              "magnetic channels have different contraction coefficient counts");
        orbital+=2*l+1;
      }
    for(int a=0;a<s.contracted_orbitals[w];++a) {
      const int n=in.coefficient_counts[w][a];
      require(n>0 && n<=in.max_coefficients && in.orbital_indices[w][a] && total<=INT_MAX-5-n,"invalid contraction coefficient count");
      for(int p=0;p<n;++p) require(in.orbital_indices[w][a][p]>=0 && in.orbital_indices[w][a][p]<s.orbitals[w],"invalid contraction orbital index");
      total+=n;
    }
  }
  return total;
}
}

struct OrbitalOptimization::Implementation {
  int Matomnum,MatomnumF,SpeciesNum,SpinP_switch,ACnt_switch,SCnt_switch;
  int orbitalOpt_History,orbitalOpt_StartPulay,OrbOpt_OptMethod,rank,ranks,Host_ID,level_stdout;
  int max_orbitals,max_coefficients,dimension_;
  double orbitalOpt_SD_step;
  std::vector<int> M2G,F_M2G,F_G2M,WhatSpecies,FNAN,Spe_Total_NO,Spe_Total_CNO,Spe_MaxL_Basis;
  std::vector<std::size_t> neighbor_offsets;
  std::vector<int> neighbor_atoms;
  Tensor<int,2> Spe_Num_CBasis,Spe_Specified_Num;
  Tensor<int,3> Spe_Trans_Orbital;
  Tensor<double,4> His_CntCoes,His_D_CntCoes,His_CntCoes_Species,His_D_CntCoes_Species;
  Tensor<double,2> OrbOpt_Hessian,My_A,A,IA,U,da;
  Tensor<double,3> D_CntCoes,D0_CntCoes,DCntCoes_Spe,My_DCntCoes_Spe,OLPtmp;
  std::vector<double> His_OrbOpt_Etot,DIIS_coes,B,TmpVec1,TmpVec2,ko,TmpD,x,y,packed,lapack_work;
  std::vector<int> pivots;
  std::vector<double*> eigen_rows;
  struct Packet { std::vector<std::size_t> send,receive;std::vector<double> outgoing,incoming; };
  std::vector<Packet> packets;

  explicit Implementation(const OpenMXOrbOptInput& in) : Implementation(in,dimension(in)) {}
  Implementation(const OpenMXOrbOptInput&,int);
  int neighbor(int atom,int h) const { return neighbor_atoms[neighbor_offsets[atom]+h]; }
  void reduce_sum(Session&,const double*,double*,int=1) const;
  void maximum(Session&,int,int*) const;
  void broadcast(Session&,double*,int,int) const;
  void exchange(Session&);
  void eigen(Session&,int);
  void inverse(Session&,int);
  void unpivoted_inverse(int);
  void validate(const OpenMXElectronicViews&,const OpenMXLocalBasisViews&) const;
  double atoms(int,Session&);
  double species(int,double,Session&);
  void derivative(Session&,const OpenMXLocalTensor3&,Tensor<double,3>&);
  void species_derivative(Session&);
  double atom_norm(Session&,const Tensor<double,3>&);
  double species_norm(const Tensor<double,3>&) const;
};

OrbitalOptimization::Implementation::Implementation(const OpenMXOrbOptInput& in,int dim)
 : Matomnum(in.local_atoms),MatomnumF(in.first_halo_atoms),SpeciesNum(int(in.species->species)),SpinP_switch(in.spin),
   ACnt_switch(in.species_mode),SCnt_switch(in.last_radial_only),orbitalOpt_History(in.history),
   orbitalOpt_StartPulay(in.start_pulay),OrbOpt_OptMethod(in.method),rank(in.rank),ranks(in.ranks),Host_ID(in.host_rank),level_stdout(in.verbosity),
   max_orbitals(in.max_orbitals),max_coefficients(in.max_coefficients),dimension_(dim),orbitalOpt_SD_step(in.descent_step),
   M2G(Matomnum+1),F_M2G(Matomnum+MatomnumF+1),F_G2M(in.atoms+1),WhatSpecies(in.atoms+1),FNAN(in.atoms+1),
   Spe_Total_NO(copy(in.species->orbitals,SpeciesNum)),Spe_Total_CNO(copy(in.species->contracted_orbitals,SpeciesNum)),
   Spe_MaxL_Basis(copy(in.species->basis_lmax,SpeciesNum)),neighbor_offsets(std::size_t(in.atoms)+2),
   Spe_Num_CBasis({std::size_t(SpeciesNum),in.species->angular_channels}),
   Spe_Specified_Num({std::size_t(SpeciesNum),std::size_t(max_orbitals)}),
   Spe_Trans_Orbital({std::size_t(SpeciesNum),std::size_t(max_orbitals),std::size_t(max_coefficients)}),
   His_CntCoes({ACnt_switch ? 0U:std::size_t(in.history+1),std::size_t(Matomnum+1),std::size_t(max_orbitals),std::size_t(max_coefficients)}),
   His_D_CntCoes(His_CntCoes),
   His_CntCoes_Species({ACnt_switch ? std::size_t(in.history+1):0U,std::size_t(SpeciesNum+1),std::size_t(max_orbitals),std::size_t(max_coefficients)}),
   His_D_CntCoes_Species(His_CntCoes_Species),
   OrbOpt_Hessian({ACnt_switch ? std::size_t(dim+2):0U,ACnt_switch ? std::size_t(dim+2):0U}),
   My_A({std::size_t(in.history+2),std::size_t(in.history+2)}),A(My_A),IA(My_A),
   U({ACnt_switch ? std::size_t(dim+5):0U,ACnt_switch ? std::size_t(dim+5):0U}),da(My_A),
   D_CntCoes({std::size_t(Matomnum+1),std::size_t(max_orbitals),std::size_t(max_coefficients)}),D0_CntCoes(D_CntCoes),
   DCntCoes_Spe({ACnt_switch ? std::size_t(SpeciesNum+1):0U,std::size_t(max_orbitals),std::size_t(max_coefficients)}),My_DCntCoes_Spe(DCntCoes_Spe),
   OLPtmp({ACnt_switch ? std::size_t(SpeciesNum+1):0U,std::size_t(max_orbitals),std::size_t(max_orbitals)}),
   His_OrbOpt_Etot(ACnt_switch ? in.history+2:0),DIIS_coes(in.history+2),B(ACnt_switch ? in.history+2:0),
   TmpVec1(ACnt_switch ? dim+1:0),TmpVec2(ACnt_switch ? dim+1:0),ko(ACnt_switch ? dim+5:0),TmpD(max_coefficients),
   x(in.history+2),y(in.history+2),packed(ACnt_switch ? std::size_t(in.history+2)*(in.history+2):0),
   lapack_work(ACnt_switch ? in.history+2:0),pivots(ACnt_switch ? in.history+2:0),eigen_rows(ACnt_switch ? dim+5:0)
{
  require(!in.atoms || in.atom_species,"missing atom species");
  for(int g=1;g<=in.atoms;++g) {
    const int s=in.atom_species[g];require(s>=0 && s<SpeciesNum,"invalid atom species");WhatSpecies[g]=s;
  }
  auto map=[&](std::vector<int>& out,const int* src) {
    require(out.size()==1 || src,"missing atom map");
    for(std::size_t a=1;a<out.size();++a) {require(src[a]>0 && src[a]<=in.atoms,"invalid atom map");out[a]=src[a];}
  };
  map(M2G,in.local_to_global);map(F_M2G,in.first_to_global);
  for(std::size_t a=1;a<F_M2G.size();++a) {
    const int g=F_M2G[a];require(in.global_to_first && in.global_to_first[g]>0 &&
        std::size_t(in.global_to_first[g])<F_M2G.size(),"invalid inverse atom map");F_G2M[g]=in.global_to_first[g];
  }
  for(int w=0;w<SpeciesNum;++w) {
    for(std::size_t l=0;l<in.species->angular_channels;++l)
      at(Spe_Num_CBasis,w,l)=in.species->contracted_basis[w*in.species->angular_channels+l];
    for(int a=0;a<Spe_Total_CNO[w];++a) {
      at(Spe_Specified_Num,w,a)=in.coefficient_counts[w][a];
      for(int p=0;p<at(Spe_Specified_Num,w,a);++p)at(Spe_Trans_Orbital,w,a,p)=in.orbital_indices[w][a][p];
    }
  }
  std::vector<unsigned char> needed(std::size_t(in.atoms)+1);
  for(int a=1;a<=Matomnum;++a)needed[M2G[a]]=1;
  for(int g=0;g<=in.atoms;++g) {
    if(needed[g]) {
      require(in.neighbors && in.neighbors[g]>=0 && in.neighbors[g]<INT_MAX && in.neighbor_atoms && in.neighbor_atoms[g],"invalid neighbor row");
      FNAN[g]=in.neighbors[g];
      for(int h=0;h<=FNAN[g];++h) {
        const int n=in.neighbor_atoms[g][h];require(n>0 && n<=in.atoms && F_G2M[n]>0,"neighbor missing from halo map");neighbor_atoms.push_back(n);
      }
    }
    neighbor_offsets[g+1]=neighbor_atoms.size();
  }
  for(std::size_t i=0;i<eigen_rows.size();++i)eigen_rows[i]=U.data()+i*U.extent(1);
  if(!ACnt_switch && ranks>1) {
    require(in.send_counts && in.receive_counts && in.receive_first && in.send_local_atoms && in.send_global_atoms && in.receive_global_atoms,"missing contraction exchange maps");
    packets.resize(ranks);
    for(int peer=0;peer<ranks;++peer) if(peer!=rank) {
      auto& p=packets[peer];const int ns=in.send_counts[peer],nr=in.receive_counts[peer];
      require(ns>=0 && nr>=0 && (!ns || (in.send_local_atoms[peer] && in.send_global_atoms[peer])) &&
          (!nr || in.receive_global_atoms[peer]),"invalid contraction exchange row");
      auto append=[&](std::vector<std::size_t>& indices,int a,int g) {
        require(a>0 && std::size_t(a)<F_M2G.size() && g>0 && g<=in.atoms,"invalid contraction exchange atom");
        const int w=WhatSpecies[g];
        for(int o=0;o<Spe_Total_CNO[w];++o) for(int c=0;c<at(Spe_Specified_Num,w,o);++c) {
          require(indices.size()<INT_MAX,"contraction message exceeds MPI count");
          const auto offset=(std::size_t(a)*max_orbitals+o)*max_coefficients+c;
          indices.push_back(offset);
        }
      };
      for(int a=0;a<ns;++a)append(p.send,in.send_local_atoms[peer][a],in.send_global_atoms[peer][a]);
      for(int a=0;a<nr;++a)append(p.receive,in.receive_first[peer]+a,in.receive_global_atoms[peer][a]);
      p.outgoing.resize(p.send.size());p.incoming.resize(p.receive.size());
    }
  }
}

void OrbitalOptimization::Implementation::reduce_sum(Session& s,const double* a,double* b,int n) const
{
  if(ranks==1) {std::copy_n(a,n,b);return;}
  require(s.callbacks.sum,"missing sum callback");service(s.callbacks.sum(s.callbacks.user,a,b,n));
}
void OrbitalOptimization::Implementation::maximum(Session& s,int a,int* b) const
{
  if(ranks==1){*b=a;return;}
  require(s.callbacks.maximum,"missing maximum callback");service(s.callbacks.maximum(s.callbacks.user,a,b));
}
void OrbitalOptimization::Implementation::broadcast(Session& s,double* a,int n,int root) const
{
  require(root>=0 && root<ranks,"orbital species absent from calculation");
  if(ranks==1)return;
  require(s.callbacks.broadcast,"missing broadcast callback");service(s.callbacks.broadcast(s.callbacks.user,a,n,root));
}
void OrbitalOptimization::Implementation::exchange(Session& s)
{
  if(ranks==1)return;
  require(s.callbacks.exchange,"missing exchange callback");
  double* values=s.local_basis->tensors[OPENMX_LOCAL_CONTRACTION].values;
  for(int distance=1;distance<ranks;++distance) {
    const int to=(rank+distance)%ranks,from=(rank-distance+ranks)%ranks;
    auto& send=packets[to];auto& recv=packets[from];
    for(std::size_t i=0;i<send.send.size();++i)send.outgoing[i]=values[send.send[i]];
    service(s.callbacks.exchange(s.callbacks.user,send.send.empty() ? -1:to,send.outgoing.data(),int(send.outgoing.size()),
        recv.receive.empty() ? -1:from,recv.incoming.data(),int(recv.incoming.size())));
    for(std::size_t i=0;i<recv.receive.size();++i)values[recv.receive[i]]=recv.incoming[i];
  }
}
void OrbitalOptimization::Implementation::eigen(Session& s,int n)
{
  require(s.callbacks.eigen,"missing eigensolver callback");service(s.callbacks.eigen(s.callbacks.user,eigen_rows.data(),ko.data(),n));
}
void OrbitalOptimization::Implementation::inverse(Session& s,int n)
{
  require(s.callbacks.inverse,"missing inverse callback");service(s.callbacks.inverse(s.callbacks.user,n+1,A.data(),A.extent(1),
      IA.data(),IA.extent(1),packed.data(),lapack_work.data(),pivots.data()));
}
void OrbitalOptimization::Implementation::validate(const OpenMXElectronicViews& e,const OpenMXLocalBasisViews& l) const
{
  const auto& c=l.tensors[OPENMX_LOCAL_CONTRACTION];
  require(c.n0==F_M2G.size() && c.n1==std::size_t(max_orbitals) && c.n2==std::size_t(max_coefficients) &&
      (!c.n0 || !c.n1 || !c.n2 || c.values),"contraction view does not match optimization owner");
  const auto& sc=l.tensors[OPENMX_LOCAL_SPECIES_CONTRACTION];
  require(!ACnt_switch || (sc.n0==std::size_t(SpeciesNum+1) && sc.n1==c.n1 && sc.n2==c.n2 &&
      (!sc.n1 || !sc.n2 || sc.values)),"species contraction view does not match optimization owner");
  if(!Matomnum)return;
  for(auto field:{OPENMX_ELECTRONIC_HAMILTONIAN,OPENMX_ELECTRONIC_OVERLAP,OPENMX_ELECTRONIC_DENSITY,OPENMX_ELECTRONIC_ENERGY_DENSITY}) {
    const auto& v=e.fields[field];const std::size_t components=field==OPENMX_ELECTRONIC_OVERLAP ? 1:SpinP_switch+1;
    require(v.values && v.component_offsets && v.atom_offsets && v.block_offsets && v.shapes && v.components>=components,"missing optimization operator");
    for(std::size_t s=0;s<components;++s) {
      require(v.component_offsets[s+1]<=v.atoms && v.component_offsets[s+1]>=v.component_offsets[s]+Matomnum+1,"short optimization atom layout");
      for(int a=1;a<=Matomnum;++a) {
        const int g=M2G[a];const auto index=v.component_offsets[s]+a;
        require(v.atom_offsets[index+1]<=v.blocks && v.atom_offsets[index+1]>=v.atom_offsets[index]+FNAN[g]+1,"short optimization neighbor layout");
        for(int h=0;h<=FNAN[g];++h) {
          const auto b=v.atom_offsets[index]+h;const bool contracted=field==OPENMX_ELECTRONIC_DENSITY || field==OPENMX_ELECTRONIC_ENERGY_DENSITY;
          const auto& sizes=contracted ? Spe_Total_CNO:Spe_Total_NO;
          const int r=sizes[WhatSpecies[g]],c0=sizes[WhatSpecies[neighbor(g,h)]];
          require(v.shapes[2*b]>=r && v.shapes[2*b+1]>=c0 && v.block_offsets[b+1]>=v.block_offsets[b] &&
              v.block_offsets[b+1]<=v.elements && std::size_t(v.shapes[2*b])*v.shapes[2*b+1]<=v.block_offsets[b+1]-v.block_offsets[b],"invalid optimization operator block");
        }
      }
    }
  }
}

// The numerical routines below preserve the legacy summation and update order.
// History, scratch, species metadata and topology are explicit members of this
// invocation; operators and services are borrowed only by the current Session.

// Derived from OpenMX Opt_Contraction.c, released by T. Ozaki (2001).
// Scalar operations and loop order are preserved for numerical regression.
double OrbitalOptimization::Implementation::atoms(int orbitalOpt_iter,Session& session)
{
  const auto* local_basis=session.local_basis;
  const auto& OLP=session.OLP;

  const int myid=rank;

  int Mc_AN,Gc_AN,Cwan,be,his;
  int his1,his2,diis_dim,al,p,q,p0,q0;

  double norm_deri,sum,sum1,sum2;
  double coef_OptG,tmp0,mixing_coef;

  derivative(session, local_basis->tensors[OPENMX_LOCAL_CONTRACTION], D_CntCoes);
  norm_deri = atom_norm(session,D_CntCoes);
  if(norm_deri==0.0) return norm_deri;

  if (orbitalOpt_History<(orbitalOpt_iter-orbitalOpt_StartPulay+1))
    diis_dim = orbitalOpt_History;
  else
    diis_dim = orbitalOpt_iter-orbitalOpt_StartPulay+1;

  if (diis_dim<1) diis_dim = 1;

  mixing_coef = norm_deri>0 ? 4.0*orbitalOpt_SD_step/norm_deri:0.0;
  if (0.05<mixing_coef) mixing_coef = 0.05;

  for (his=(diis_dim-1); 0<his; his--){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){
	  at(His_CntCoes, his, Mc_AN, be, p) = at(His_CntCoes, his-1, Mc_AN, be, p);
	  at(His_D_CntCoes, his, Mc_AN, be, p) = at(His_D_CntCoes, his-1, Mc_AN, be, p);
	}
      }
    }
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (be=0; be<Spe_Total_CNO[Cwan]; be++){
      for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){
	at(His_CntCoes, 0, Mc_AN, be, p) = openmx_local_contraction_row(local_basis, Mc_AN, be)[p];
	at(His_D_CntCoes, 0, Mc_AN, be, p) = at(D_CntCoes, Mc_AN, be, p);
      }
    }
  }

  if (diis_dim==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){
	  openmx_local_contraction_row(local_basis, Mc_AN, be)[p] -= mixing_coef*at(D_CntCoes, Mc_AN, be, p);
	}
      }
    }
  }

  else {

    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){

	sum = 0.0;
	for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
	  Gc_AN = M2G[Mc_AN];
	  Cwan = WhatSpecies[Gc_AN];
	  for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	    for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){

	      sum += at(His_D_CntCoes, his1, Mc_AN, be, p)*at(His_D_CntCoes, his2, Mc_AN, be, p);
	    }
	  }
	}

	at(My_A, his1, his2) = sum;
      }
    }

    for (his1=0; his1<diis_dim; his1++){
      for (his2=his1; his2<diis_dim; his2++){

	reduce_sum(session,&at(My_A, his1, his2),&at(A, his1, his2));

	at(A, his2, his1) = at(A, his1, his2);
      }
    }

    for (his1=1; his1<=diis_dim; his1++){
      at(A, his1-1, diis_dim) = 1.0;
      at(A, diis_dim, his1-1) = 1.0;
    }
    at(A, diis_dim, diis_dim) = 0.0;

    unpivoted_inverse(diis_dim);

    for (his1=0; his1<diis_dim; his1++){
      DIIS_coes[his1] = at(IA, his1, diis_dim);
    }

    if      (1.0e-2<norm_deri)    coef_OptG = 0.1;
    else if (1.0e-3<norm_deri)    coef_OptG = 0.2;
    else if (1.0e-4<norm_deri)    coef_OptG = 0.4;
    else if (1.0e-5<norm_deri)    coef_OptG = 0.5;
    else                          coef_OptG = 0.7;

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
	for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){

	  sum1 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum1 += DIIS_coes[his1]*at(His_CntCoes, his1, Mc_AN, be, p);
	  }

	  sum2 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum2 += DIIS_coes[his1]*at(His_D_CntCoes, his1, Mc_AN, be, p);
	  }

	  openmx_local_contraction_row(local_basis, Mc_AN, be)[p] = sum1 - coef_OptG*sum2;
	}
      }
    }
  }

  if (myid==Host_ID) printf("<Opt_Contraction> Norm of derivatives = %15.12f \n",norm_deri);

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];

    for (al=0; al<Spe_Total_CNO[Cwan]; al++){

      sum = 0.0;
      for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	p0 = at(Spe_Trans_Orbital, Cwan, al, p);

	for (q=0; q<at(Spe_Specified_Num, Cwan, al); q++){
          q0 = at(Spe_Trans_Orbital, Cwan, al, q);

          tmp0 = openmx_local_contraction_row(local_basis, Mc_AN, al)[p]*openmx_local_contraction_row(local_basis, Mc_AN, al)[q];
          sum += tmp0*OLP(0, Mc_AN, 0, p0, q0);
        }
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
        openmx_local_contraction_row(local_basis, Mc_AN, al)[p] = openmx_local_contraction_row(local_basis, Mc_AN, al)[p]*tmp0;
      }

    }
  }

  if (2<=level_stdout){
    printf("<Opt_Contraction> Contraction coefficients\n");
    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];
      for (be=0; be<Spe_Total_CNO[Cwan]; be++){
        for (p=0; p<at(Spe_Specified_Num, Cwan, be); p++){
          printf("  Mc_AN=%2d Gc_AN=%2d be=%2d p=%2d  %15.12f\n",
                    Mc_AN,Gc_AN,be,p,openmx_local_contraction_row(local_basis, Mc_AN, be)[p]);
        }
      }
    }
  }

  exchange(session);
  return norm_deri;
}

double OrbitalOptimization::Implementation::species(int orbitalOpt_iter,double TotalE,Session& session)
{
  const auto* local_basis=session.local_basis;
  const auto& OLP=session.OLP;

  const int myid=rank;

  int i,j,Mc_AN,Mc_AN1,Cwan,Gc_AN,be,his,po,my_po;
  int his1,his2,diis_dim,al,p,q,p0,q0;
  int wan,dim_H;
  int wan1,al1,p1,wan2,al2,p2;

  double c0,c1,MinKo,MaxStep;
  double tmp1,tmp2,d1,d2,d;
  double norm_deri,sum,sum1,sum2;
  double coef_OptG,tmp0,mixing_coef;

  int ID;

 dim_H=dimension_;

  if (orbitalOpt_iter==1){

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        at(OrbOpt_Hessian, i, j) = 0.0;
      }
      at(OrbOpt_Hessian, i, i) = 1.0;
    }
  }

  {
    int p,Mc_AN,Gc_AN,wan,al;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];

    if (3<=level_stdout){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
          printf("E Init_CntCoes Mc_AN=%2d Gc_AN=%2d al=%2d p=%2d  %15.12f\n",
                  Mc_AN,Gc_AN,al,p,openmx_local_contraction_row(local_basis, Mc_AN, al)[p]);
        }
      }
    }
  }
  }

  species_derivative(session);
  norm_deri = species_norm(DCntCoes_Spe);
  if(norm_deri==0.0) return norm_deri;

  if (orbitalOpt_History<(orbitalOpt_iter-orbitalOpt_StartPulay+1))
    diis_dim = orbitalOpt_History;
  else
    diis_dim = orbitalOpt_iter-orbitalOpt_StartPulay+1;

  if (diis_dim<1) diis_dim = 1;

  for (his=(diis_dim-1); 0<his; his--){
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	  at(His_CntCoes_Species, his, wan, al, p) = at(His_CntCoes_Species, his-1, wan, al, p);
	  at(His_D_CntCoes_Species, his, wan, al, p) = at(His_D_CntCoes_Species, his-1, wan, al, p);
	}
      }
    }

    His_OrbOpt_Etot[his] = His_OrbOpt_Etot[his-1];
  }

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	at(His_CntCoes_Species, 0, wan, al, p)   = openmx_local_species_contraction_row(local_basis, wan, al)[p];
	at(His_D_CntCoes_Species, 0, wan, al, p) = at(DCntCoes_Spe, wan, al, p);
      }
    }
  }

  His_OrbOpt_Etot[0] = TotalE;

  if (diis_dim==1){

    MaxStep = 0.0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	  if (MaxStep<fabs(at(DCntCoes_Spe, wan, al, p))) MaxStep = fabs(at(DCntCoes_Spe, wan, al, p));
	}
      }
    }

    mixing_coef = orbitalOpt_SD_step/MaxStep;

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	  openmx_local_species_contraction_row(local_basis, wan, al)[p] -= mixing_coef*at(DCntCoes_Spe, wan, al, p);
	}
      }
    }
  }

  else {

    for (his1=0; his1<diis_dim; his1++){
      for (his2=0; his2<diis_dim; his2++){

	sum = 0.0;

        for (wan=0; wan<SpeciesNum; wan++){
          for (al=0; al<Spe_Total_CNO[wan]; al++){
            for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	      sum += at(His_D_CntCoes_Species, his1, wan, al, p)*at(His_D_CntCoes_Species, his2, wan, al, p);
	    }
	  }
	}

	at(A, his1, his2) = sum;

	if (his1==his2) B[his1] = 0.0;

      }
    }
    B[diis_dim] = 1.0;

    for (his1=1; his1<=diis_dim; his1++){
      at(A, his1-1, diis_dim) = 1.0;
      at(A, diis_dim, his1-1) = 1.0;
    }
    at(A, diis_dim, diis_dim) = 0.0;

    inverse(session,diis_dim);

    for (his1=0; his1<diis_dim; his1++){

      sum = 0.0;
      for (his2=0; his2<=diis_dim; his2++){
        sum += at(IA, his1, his2)*B[his2];
      }
      DIIS_coes[his1] = sum;
    }

    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

	  sum1 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum1 += DIIS_coes[his1]*at(His_CntCoes_Species, his1, wan, al, p);
	  }

	  sum2 = 0.0;
	  for (his1=0; his1<diis_dim; his1++){
	    sum2 += DIIS_coes[his1]*at(His_D_CntCoes_Species, his1, wan, al, p);
	  }

	  openmx_local_species_contraction_row(local_basis, wan, al)[p] = sum1;
          at(DCntCoes_Spe, wan, al, p)    = sum2;
	}
      }
    }

    for (i=0; i<dim_H; i++){

      j = 0;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
          for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

            d = at(His_CntCoes_Species, 0, wan, al, p) - at(His_CntCoes_Species, 1, wan, al, p);
	    sum += at(OrbOpt_Hessian, i, j)*d;
            j++;
	  }
	}
      }

      TmpVec1[i] = sum;
    }

    tmp1 = 0.0;
    tmp2 = 0.0;

    i = 0;
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
	for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

          d1 = at(His_D_CntCoes_Species, 0, wan, al, p)-at(His_D_CntCoes_Species, 1, wan, al, p);
          d2 = at(His_CntCoes_Species, 0, wan, al, p) - at(His_CntCoes_Species, 1, wan, al, p);

          tmp1 += d1*d2;
          tmp2 += d2*TmpVec1[i];

          i++;
	}
      }
    }

    c0 = 1.0/tmp1;
    c1 = 1.0/tmp2;

    if (0.0<c0){

      i = 0;
      for (wan1=0; wan1<SpeciesNum; wan1++){
	for (al1=0; al1<Spe_Total_CNO[wan1]; al1++){
	  for (p1=0; p1<at(Spe_Specified_Num, wan1, al1); p1++){

	    j = 0;
	    for (wan2=0; wan2<SpeciesNum; wan2++){
	      for (al2=0; al2<Spe_Total_CNO[wan2]; al2++){
		for (p2=0; p2<at(Spe_Specified_Num, wan2, al2); p2++){

		  d1 = at(His_D_CntCoes_Species, 0, wan1, al1, p1)-at(His_D_CntCoes_Species, 1, wan1, al1, p1);
		  d2 = at(His_D_CntCoes_Species, 0, wan2, al2, p2)-at(His_D_CntCoes_Species, 1, wan2, al2, p2);
		  at(OrbOpt_Hessian, i, j) += c0*d1*d2 -c1*TmpVec1[i]*TmpVec1[j];

		  j++;
		}
	      }
	    }

	    i++;
	  }
	}
      }
    }

    for (i=0; i<dim_H; i++){
      for (j=0; j<dim_H; j++){
        at(U, i+1, j+1) = at(OrbOpt_Hessian, i, j);
      }
    }

    eigen(session,dim_H);

    MinKo = 0.1;
    for (i=0; i<dim_H; i++){
      if (ko[i+1]<MinKo) ko[i+1] = MinKo;
    }

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

	    sum += at(U, j, i+1)*at(DCntCoes_Spe, wan, al, p);
	    j++;
	  }
	}
      }

      at(U, i+1, 0) = sum;
    }

    for (i=0; i<dim_H; i++){

      j = 1;
      sum = 0.0;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

	    sum += at(U, i+1, j)*at(U, j, 0)/ko[j];
	    j++;
	  }
	}
      }

      at(U, 0, i+1) = sum;
    }

    MaxStep = 0.0;
    for (i=1; i<=dim_H; i++){
      if (MaxStep<fabs(at(U, 0, i))) MaxStep = fabs(at(U, 0, i));
    }

    if (0.001<MaxStep){
      for (i=1; i<=dim_H; i++){
	at(U, 0, i) = 0.001*at(U, 0, i)/MaxStep;
      }
    }

    if (OrbOpt_OptMethod==2){

      i = 1;
      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<at(Spe_Specified_Num, wan, al); p++){

	    openmx_local_species_contraction_row(local_basis, wan, al)[p] -= at(U, 0, i);
	    i++;
	  }
	}
      }

    }

    else if (OrbOpt_OptMethod==1) {

      if      (1.0e-2<norm_deri)  coef_OptG = 0.05;
      else if (1.0e-3<norm_deri)  coef_OptG = 0.10;
      else if (1.0e-4<norm_deri)  coef_OptG = 0.15;
      else if (1.0e-5<norm_deri)  coef_OptG = 0.20;
      else if (1.0e-6<norm_deri)  coef_OptG = 0.40;
      else coef_OptG = 0.40;

      for (wan=0; wan<SpeciesNum; wan++){
	for (al=0; al<Spe_Total_CNO[wan]; al++){
	  for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	    openmx_local_species_contraction_row(local_basis, wan, al)[p] -= coef_OptG*at(DCntCoes_Spe, wan, al, p);
	  }
	}
      }

    }

  }

  if (myid==Host_ID) printf("<Opt_Contraction> Norm of derivatives = %15.12f \n",norm_deri);

  for (wan=0; wan<SpeciesNum; wan++){

    Mc_AN = 1;
    po = 0;

    while (po==0 && Mc_AN<=Matomnum){

      Gc_AN = M2G[Mc_AN];
      Cwan = WhatSpecies[Gc_AN];

      if (wan==Cwan){
        po = 1;
        Mc_AN1 = Mc_AN;
      }

      Mc_AN++;
    }

    my_po = po*(myid+1);
    maximum(session,my_po,&po);
    ID = po - 1;

    if (myid==ID){
      for (al=0; al<Spe_Total_NO[wan]; al++){
	for (be=0; be<Spe_Total_NO[wan]; be++){
          at(OLPtmp, wan, al, be) = OLP(0, Mc_AN1, 0, al, be);
	}
      }
    }

    for (al=0; al<Spe_Total_NO[wan]; al++){
      broadcast(session,&at(OLPtmp, wan, al, 0),Spe_Total_NO[wan],ID);
    }

  }

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){

      sum = 0.0;
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	p0 = at(Spe_Trans_Orbital, wan, al, p);

	for (q=0; q<at(Spe_Specified_Num, wan, al); q++){
	  q0 = at(Spe_Trans_Orbital, wan, al, q);

	  tmp0 = openmx_local_species_contraction_row(local_basis, wan, al)[p]*openmx_local_species_contraction_row(local_basis, wan, al)[q];
	  sum += tmp0*at(OLPtmp, wan, p0, q0);
	}
      }

      tmp0 = 1.0/sqrt(sum);
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
	openmx_local_species_contraction_row(local_basis, wan, al)[p] = openmx_local_species_contraction_row(local_basis, wan, al)[p]*tmp0;
      }
    }
  }

  if (2<=level_stdout){
    printf("<Opt_Contraction> Contraction coefficients\n");fflush(stdout);
    for (wan=0; wan<SpeciesNum; wan++){
      for (al=0; al<Spe_Total_CNO[wan]; al++){
        for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
          printf("  wan=%2d al=%2d p=%2d  %15.12f\n",
                    wan,al,p,openmx_local_species_contraction_row(local_basis, wan, al)[p]);fflush(stdout);
        }
      }
    }
  }

  for (Mc_AN=1; Mc_AN<=(Matomnum+MatomnumF); Mc_AN++){
    Gc_AN = F_M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
        openmx_local_contraction_row(local_basis, Mc_AN, al)[p] = openmx_local_species_contraction_row(local_basis, wan, al)[p];
      }
    }
  }

  return norm_deri;
}

void OrbitalOptimization::Implementation::derivative(Session& session,const OpenMXLocalTensor3& CntCoes0,Tensor<double,3>& D_CntCoes)
{
  const auto& H=session.H; const auto& OLP=session.OLP;
  const auto& CDM=session.CDM; const auto& EDM=session.EDM;

  int Cwan,Mc_AN,Gc_AN,Mh_AN;
  int p,q,al,al0,be,p0,q0,wan;
  int h_AN,Gh_AN,Hwan,spin;
  int L0,Mul0,M0;
  double sum,sum0;

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[Cwan]; al++){
      for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	p0 = at(Spe_Trans_Orbital, Cwan, al, p);

	sum = 0.0;
	for (h_AN=0; h_AN<=FNAN[Gc_AN]; h_AN++){
	  Gh_AN = neighbor(Gc_AN, h_AN);
	  Hwan = WhatSpecies[Gh_AN];
          Mh_AN = F_G2M[Gh_AN];
	  for (be=0; be<Spe_Total_CNO[Hwan]; be++){
	    for (q=0; q<at(Spe_Specified_Num, Hwan, be); q++){
	      q0 = at(Spe_Trans_Orbital, Hwan, be, q);

	      sum0 = 0.0;
	      for (spin=0; spin<=SpinP_switch; spin++){
		sum0 = sum0 +
		   CDM(spin, Mc_AN, h_AN, al, be)*H(spin, Mc_AN, h_AN, p0, q0)
		  -EDM(spin, Mc_AN, h_AN, al, be)*OLP(0, Mc_AN, h_AN, p0, q0);
	      }
	      if (SpinP_switch==0) sum0 = 2.0*sum0;

	      sum = sum + 2.0*sum0*at(CntCoes0, Mh_AN, be, q);

	    }
	  }
	}

	at(D0_CntCoes, Mc_AN, al, p) = sum;

      }
    }

    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<at(Spe_Num_CBasis, Cwan, L0); Mul0++){

	for (p=0; p<at(Spe_Specified_Num, Cwan, al+1); p++){
	  TmpD[p] = 0.0;
	}

	al0 = al;
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	    TmpD[p] = TmpD[p] + at(D0_CntCoes, Mc_AN, al, p);
	  }
	}
	al = al0;

	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	    at(D_CntCoes, Mc_AN, al, p) = TmpD[p];
	  }
	}
      }
    }
  }

  if (SCnt_switch==1){

    for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){

      Gc_AN = M2G[Mc_AN];
      wan = WhatSpecies[Gc_AN];

      al = -1;
      for (L0=0; L0<=Spe_MaxL_Basis[wan]; L0++){
	for (Mul0=0; Mul0<at(Spe_Num_CBasis, wan, L0); Mul0++){
	  for (M0=0; M0<=2*L0; M0++){

	    al++;

            if ( Mul0!=(at(Spe_Num_CBasis, wan, L0)-1) ){

	      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
		at(D_CntCoes, Mc_AN, al, p) = 0.0;
	      }
            }

	  }
	}
      }
    }

  }

}

void OrbitalOptimization::Implementation::species_derivative(Session& session)
{
  const auto* local_basis=session.local_basis;

  int i,j,Mc_AN,Gc_AN;
  int p,al;

  int wan;

 derivative(session,local_basis->tensors[OPENMX_LOCAL_CONTRACTION],D_CntCoes);
  for (wan=0; wan<SpeciesNum; wan++){
    for (i=0; i<max_orbitals; i++){
      for (j=0; j<max_coefficients; j++){
        at(My_DCntCoes_Spe, wan, i, j) = 0.0;
      }
    }
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
        at(My_DCntCoes_Spe, wan, al, p) += at(D_CntCoes, Mc_AN, al, p);
      }
    }
  }

  for (wan=0; wan<SpeciesNum; wan++){
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
        reduce_sum(session,&at(My_DCntCoes_Spe, wan, al, p),&at(DCntCoes_Spe, wan, al, p));

      }
    }
  }

  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    wan = WhatSpecies[Gc_AN];
    for (al=0; al<Spe_Total_CNO[wan]; al++){
      for (p=0; p<at(Spe_Specified_Num, wan, al); p++){
        at(D_CntCoes, Mc_AN, al, p) = at(DCntCoes_Spe, wan, al, p);
      }
    }
  }

}

double OrbitalOptimization::Implementation::atom_norm(Session& session,const Tensor<double,3>& D_CntCoes)
{
  int Cwan,Mc_AN,Gc_AN,p,al;
  int L0,Mul0,M0;
  double My_NormD,NormD;

  My_NormD = 0.0;
  for (Mc_AN=1; Mc_AN<=Matomnum; Mc_AN++){
    Gc_AN = M2G[Mc_AN];
    Cwan = WhatSpecies[Gc_AN];
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<at(Spe_Num_CBasis, Cwan, L0); Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	    My_NormD += at(D_CntCoes, Mc_AN, al, p)*at(D_CntCoes, Mc_AN, al, p);
	  }
	}
      }
    }
  }

  reduce_sum(session,&My_NormD,&NormD);

  return NormD;
}

double OrbitalOptimization::Implementation::species_norm(const Tensor<double,3>& DCntCoes_Spe) const
{
  int Cwan,p,al;
  int L0,Mul0,M0;
  double My_NormD,NormD;

  My_NormD = 0.0;

  for (Cwan=0; Cwan<SpeciesNum; Cwan++){
    al = -1;
    for (L0=0; L0<=Spe_MaxL_Basis[Cwan]; L0++){
      for (Mul0=0; Mul0<at(Spe_Num_CBasis, Cwan, L0); Mul0++){
	for (M0=0; M0<=2*L0; M0++){
	  al++;
	  for (p=0; p<at(Spe_Specified_Num, Cwan, al); p++){
	    My_NormD += at(DCntCoes_Spe, Cwan, al, p)*at(DCntCoes_Spe, Cwan, al, p);
	  }
	}
      }
    }
  }

  NormD = My_NormD;

  return NormD;
}

void OrbitalOptimization::Implementation::unpivoted_inverse(int n)
{
  int i,j,k;double w;

  if (n==-1){
    for (i=0; i<(orbitalOpt_History+2); i++){
      for (j=0; j<(orbitalOpt_History+2); j++){
	at(A, i, j) = 0.0;
      }
    }
  }
  else{
    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	at(da, i, j) = at(A, i, j);
      }
    }

    for (k=0; k<=n-1; k++){
      w = 1.0/at(A, k, k);
      for (i=k+1; i<=n; i++){
	at(A, i, k) = w*at(A, i, k);
	for (j=k+1; j<=n; j++){
	  at(A, i, j) = at(A, i, j) - at(A, i, k)*at(A, k, j);
	}
      }
    }

    for (k=0; k<=n; k++){

      for (i=0; i<=n; i++){
	if (i==k)
	  y[i] = 1.0;
	else
	  y[i] = 0.0;
	for (j=0; j<=i-1; j++){
	  y[i] = y[i] - at(A, i, j)*y[j];
	}
      }

      for (i=n; 0<=i; i--){
	x[i] = y[i];
	for (j=n; (i+1)<=j; j--){
	  x[i] = x[i] - at(A, i, j)*x[j];
	}
	x[i] = x[i]/at(A, i, i);
	at(IA, i, k) = x[i];
      }
    }

    for (i=0; i<=n; i++){
      for (j=0; j<=n; j++){
	at(A, i, j) = at(da, i, j);
      }
    }
  }

}

OrbitalOptimization::OrbitalOptimization(const OpenMXOrbOptInput& in)
  : implementation_(std::make_unique<Implementation>(in)) {}
OrbitalOptimization::~OrbitalOptimization()=default;
OpenMXOrbOptViews OrbitalOptimization::views() noexcept
{
  auto& s=*implementation_;
  return {history_view(s.His_CntCoes),history_view(s.His_D_CntCoes),history_view(s.His_CntCoes_Species),
      history_view(s.His_D_CntCoes_Species),{s.OrbOpt_Hessian.extent(0),s.OrbOpt_Hessian.extent(1),s.OrbOpt_Hessian.data()},
      s.His_OrbOpt_Etot.data(),s.His_OrbOpt_Etot.size(),view3(s.D_CntCoes),view3(s.DCntCoes_Spe)};
}
double OrbitalOptimization::step(int iteration,double energy,const OpenMXElectronicViews& e,
    const OpenMXLocalBasisViews& l,const OpenMXOrbOptCallbacks& c)
{
  require(iteration>0 && std::isfinite(energy),"invalid optimization iteration or energy");
  auto& state=*implementation_;state.validate(e,l);Session session(e,l,c);
  return state.ACnt_switch ? state.species(iteration,energy,session):state.atoms(iteration,session);
}
double OrbitalOptimization::gradient(const OpenMXElectronicViews& e,const OpenMXLocalBasisViews& l,const OpenMXOrbOptCallbacks& c)
{
  auto& state=*implementation_;state.validate(e,l);Session session(e,l,c);
  if(state.ACnt_switch){state.species_derivative(session);return state.species_norm(state.DCntCoes_Spe);}
  state.derivative(session,l.tensors[OPENMX_LOCAL_CONTRACTION],state.D_CntCoes);return state.atom_norm(session,state.D_CntCoes);
}
} // namespace openmx::core
