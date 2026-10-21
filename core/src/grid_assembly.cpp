#include "grid_assembly.hpp"
#include "calculation_context.hpp"
#include <openmx/core/tensor.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && \
    !defined(__SYCL_DEVICE_ONLY__) && !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
#define GRID_X86 1
#define GRID_AVX2 __attribute__((target("avx2"),noinline))
#define GRID_AVX512 __attribute__((target("avx512f"),noinline))
#define GRID_INLINE inline __attribute__((always_inline))
#else
#define GRID_X86 0
#define GRID_AVX2
#define GRID_AVX512
#define GRID_INLINE inline
#endif

namespace {
using openmx::core::Tensor;
using openmx::core::element;
void require(bool ok,const char* message) { if(!ok)throw std::invalid_argument(message); }
size_t add(size_t a,size_t b) {
  if(b>size_t(PTRDIFF_MAX) || a>size_t(PTRDIFF_MAX)-b)throw std::length_error("grid offset overflow");
  return a+b;
}
size_t mul(size_t a,size_t b) {
  if(b && a>size_t(PTRDIFF_MAX)/b)throw std::length_error("grid extent overflow");
  return a*b;
}
template<class F> int checked(char* error,size_t capacity,F&& operation) noexcept {
  if(error && capacity)*error='\0';
  try { operation();return 1; }
  catch(const std::exception& e) { if(error && capacity)std::snprintf(error,capacity,"%s",e.what()); }
  catch(...) { if(error && capacity)std::snprintf(error,capacity,"grid assembly failed"); }
  return 0;
}
bool valid_spins(int n) { return n==1 || n==2 || n==4; }

struct Density {
  int spins;
  std::vector<size_t> points,offsets,send_offsets,receive_offsets;
  std::vector<int> send_counts,receive_counts;
  Tensor<double,2> values;
  Tensor<double,1> send_values,receive_values;
  std::vector<double*> rows,send_rows,receive_rows;
  std::vector<double**> components;
  static std::vector<size_t> prefix(size_t n,const size_t* counts) {
    std::vector<size_t> out(add(n,1),0);
    for(size_t i=0;i<n;i++)out[i+1]=add(out[i],counts[i]);
    return out;
  }
  static std::vector<size_t> packets(size_t n,const int* counts,int spins) {
    std::vector<size_t> out(add(n,1),0);
    for(size_t i=0;i<n;i++) {
      require(counts[i]>=0,"negative grid exchange count");
      out[i+1]=add(out[i],mul(size_t(counts[i]),size_t(spins)));
    }
    return out;
  }
  Density(int s,size_t n,const size_t* p,size_t ranks,const int* snd,const int* rcv)
    :spins(s),points(p,p+n),offsets(prefix(n,p)),send_offsets(packets(ranks,snd,s)),
     receive_offsets(packets(ranks,rcv,s)),send_counts(ranks),receive_counts(ranks),
     values({size_t(s),offsets.back()}),send_values({send_offsets.back()}),receive_values({receive_offsets.back()}),
     rows(mul(size_t(s),n)),send_rows(ranks),receive_rows(ranks),components(s) {
    if(ranks) {std::copy_n(snd,ranks,send_counts.data());std::copy_n(rcv,ranks,receive_counts.data());}
    for(int c=0;c<s;c++) {
      components[c]=n?rows.data()+size_t(c)*n:rows.data();
      for(size_t a=0;a<n;a++)rows[size_t(c)*n+a]=values.size()?values.data()+size_t(c)*offsets.back()+offsets[a]:nullptr;
    }
    for(size_t r=0;r<ranks;r++) {
      send_rows[r]=send_values.size()?send_values.data()+send_offsets[r]:nullptr;
      receive_rows[r]=receive_values.size()?receive_values.data()+receive_offsets[r]:nullptr;
    }
  }
  bool matches(int s,size_t n,const size_t* p,size_t ranks,const int* snd,const int* rcv) const {
    return spins==s && points.size()==n && send_counts.size()==ranks &&
      std::equal(points.begin(),points.end(),p) && std::equal(send_counts.begin(),send_counts.end(),snd) &&
      std::equal(receive_counts.begin(),receive_counts.end(),rcv);
  }
  OpenMXGridDensity view() {
    return {components.data(),send_rows.data(),receive_rows.data(),values.size(),send_values.size(),receive_values.size()};
  }
};
struct Scratch {
  Tensor<double,1> storage;
  size_t offset;
  explicit Scratch(size_t n):storage({add(n,7)}) {
    // Align a subspan of an owned array, retaining the normal C++ allocator
    // and its failure guarantees. No vendor allocator or runtime is needed.
    void* pointer=storage.data();size_t space=mul(storage.size(),sizeof(double));
    require(std::align(64,mul(n,sizeof(double)),pointer,space)!=nullptr,"grid scratch alignment failed");
    offset=static_cast<double*>(pointer)-storage.data();
  }
  double* data() noexcept {return storage.data()+offset;}
};
struct Worker {
  Scratch dm,orb0,orb1,h;
  Worker(size_t n,int spins):dm(mul(mul(n,n),size_t(spins))),orb0(mul(n,16)),orb1(mul(n,16)),
    h(mul(mul(n,n),size_t(spins))) {}
};
struct Plan {
  int spins,simd;
  size_t orbitals;
  std::vector<OpenMXGridAtom> atoms;
  std::vector<OpenMXGridPair> pairs;
  std::vector<Worker> workers;
  Plan(int s,size_t a,size_t p,size_t n,size_t w,int mode):spins(s),simd(mode),orbitals(n),atoms(a),pairs(p) {
    workers.reserve(w);for(size_t i=0;i<w;i++)workers.emplace_back(n,s);
  }
};
struct Exchange {
  Tensor<double,1> send,receive;
  std::vector<std::max_align_t> send_requests,receive_requests,send_status,receive_status;
  size_t sends,receives,request_bytes,status_bytes;
  static size_t opaque(size_t n,size_t bytes) {
    const auto b=mul(n,bytes);return b/sizeof(std::max_align_t)+(b%sizeof(std::max_align_t)!=0);
  }
  Exchange(size_t s,size_t r,size_t ns,size_t nr,size_t rb,size_t sb):send({s}),receive({r}),
    send_requests(opaque(ns,rb)),receive_requests(opaque(nr,rb)),send_status(opaque(ns,sb)),receive_status(opaque(nr,sb)),
    sends(ns),receives(nr),request_bytes(rb),status_bytes(sb) {}
  bool fits(size_t s,size_t r,size_t ns,size_t nr,size_t rb,size_t sb) const {
    return s<=send.size() && r<=receive.size() && ns<=sends && nr<=receives && rb==request_bytes && sb==status_bytes;
  }
  OpenMXGridExchange view() {return {send.data(),receive.data(),send_requests.data(),receive_requests.data(),send_status.data(),receive_status.data()};}
};
template<class T> GRID_INLINE double orbital(const OpenMXGridOrbitals& orbs,int point,int orbital_index) {
  const auto rows=static_cast<const T* const*>(orbs.rows);
  return orbs.transposed?double(rows[orbital_index][point]):double(rows[point][orbital_index]);
}
GRID_INLINE double orbital(const OpenMXGridOrbitals& orbs,int point,int orbital_index) {
  return orbs.element_bytes==int(sizeof(float))?orbital<float>(orbs,point,orbital_index):orbital<double>(orbs,point,orbital_index);
}
void validate(const OpenMXGridPair& p,const Plan& plan,bool hamiltonian) {
  require(p.rows>=0 && p.columns>=0 && p.points>=0 && size_t(p.rows)<=plan.orbitals && size_t(p.columns)<=plan.orbitals,
    "invalid grid pair dimensions");
  if(!p.rows || !p.columns || !p.points)return;
  require(p.center_points && p.neighbor_points && p.center.rows && p.neighbor.rows,"missing grid pair orbitals/indices");
  require((p.center.element_bytes==4 || p.center.element_bytes==8) &&
    (p.neighbor.element_bytes==4 || p.neighbor.element_bytes==8),"unsupported grid orbital precision");
  for(int s=0;s<plan.spins;s++) {
    require(p.matrix[s],"missing grid pair matrix");
    for(int i=0;i<p.rows;i++)require(p.matrix[s][i],"missing grid matrix row");
    if(hamiltonian)require(p.potential_index && p.potential[s],"missing grid potential");
  }
}

// A lane is a grid point. Every component retains the original ascending j,
// i, neighbor and grid-point summation order, including repeated center points.
template<int Spins,int Lanes> GRID_INLINE void density_pair(const OpenMXGridPair& p,Worker& w,Density& density) {
  typename Tensor<double,3>::view_type dm(w.dm.data(),typename Tensor<double,3>::extents_type(p.rows,p.columns,Spins));
  for(int i=0;i<p.rows;i++)for(int j=0;j<p.columns;j++)for(int s=0;s<Spins;s++)element(dm,i,j,s)=p.matrix[s][i][j];
  for(int first=0;first<p.points;first+=Lanes) {
    const int count=std::min(Lanes,p.points-first);
    for(int i=0;i<p.rows;i++)for(int g=0;g<Lanes;g++)
      w.orb0.data()[i*Lanes+g]=g<count?orbital(p.center,p.center_points[first+g],i):0.0;
    for(int j=0;j<p.columns;j++)for(int g=0;g<Lanes;g++)
      w.orb1.data()[j*Lanes+g]=g<count?orbital(p.neighbor,p.neighbor.overlap_rows?first+g:p.neighbor_points[first+g],j):0.0;
    double sum[Spins][Lanes]{};
    for(int i=0;i<p.rows;i++) {
      double tmp[Spins][Lanes]{};
      for(int j=0;j<p.columns;j++)for(int s=0;s<Spins;s++) {
        const double c=element(dm,i,j,s);
#pragma omp simd
        for(int g=0;g<Lanes;g++)tmp[s][g]+=c*w.orb1.data()[j*Lanes+g];
      }
      for(int s=0;s<Spins;s++) {
#pragma omp simd
        for(int g=0;g<Lanes;g++)sum[s][g]+=w.orb0.data()[i*Lanes+g]*tmp[s][g];
      }
    }
    for(int s=0;s<Spins;s++)for(int g=0;g<count;g++)density.components[s][p.atom][p.center_points[first+g]]+=sum[s][g];
  }
}
template<int Lanes> GRID_INLINE void density_dispatch(const OpenMXGridPair& p,Worker& w,Density& d,int spins) {
  if(spins==1)density_pair<1,Lanes>(p,w,d);else if(spins==2)density_pair<2,Lanes>(p,w,d);else density_pair<4,Lanes>(p,w,d);
}
GRID_AVX2 void density_avx2(const OpenMXGridPair& p,Worker& w,Density& d,int s) {density_dispatch<4>(p,w,d,s);}
GRID_AVX512 void density_avx512(const OpenMXGridPair& p,Worker& w,Density& d,int s) {density_dispatch<8>(p,w,d,s);}

// Small spin-unpolarized pairs keep the original 16-point blocked accumulator.
// Other shapes seed their accumulator with H before processing grid points.
template<int Spins> GRID_INLINE void hamiltonian_pair(const OpenMXGridPair& p,Worker& w,size_t max_orbitals) {
  if(!p.points || !p.rows || !p.columns)return;
  const bool blocked=Spins==1 && p.rows<=16 && p.columns<=16 && !p.center.transposed;
  const size_t stride=blocked?16:((max_orbitals+7)/8*8);
  typename Tensor<double,3>::view_type h(w.h.data(),typename Tensor<double,3>::extents_type(Spins,p.rows,stride));
  // For a small maximum orbital count, the blocked row stride is still 16.
  for(int s=0;s<Spins;s++)for(int i=0;i<p.rows;i++)for(size_t j=0;j<stride;j++)
    element(h,s,i,j)=blocked || j>=size_t(p.columns)?0.0:p.matrix[s][i][j];
  if(blocked)for(size_t g=0;g<16;g++)
    std::fill(w.orb1.data()+g*max_orbitals+p.columns,w.orb1.data()+g*max_orbitals+16,0.0);
  for(int first=0;first<p.points;first+=16) {
    const int count=std::min(16,p.points-first);
    for(int g=0;g<count;g++) {
      const int nc=p.center_points[first+g];
      const int nh=p.neighbor.overlap_rows?first+g:p.neighbor_points[first+g];
      for(int i=0;i<p.rows;i++)w.orb0.data()[g*max_orbitals+i]=orbital(p.center,nc,i);
      for(int j=0;j<p.columns;j++)w.orb1.data()[g*max_orbitals+j]=orbital(p.neighbor,nh,j);
    }
    for(int g=0;g<count;g++) {
      const int nc=p.center_points[first+g],mn=p.potential_index[nc];
      for(int s=0;s<Spins;s++) {
        const double v=p.volume*p.potential[s][mn];
        for(int i=0;i<p.rows;i++) {
          const double a=v*w.orb0.data()[g*max_orbitals+i];
          double* row=&element(h,s,i,0);const double* b=w.orb1.data()+g*max_orbitals;
          if(blocked) {
            // Preserve the native block's padded width: a 13-orbital basis
            // still uses complete vector updates instead of a scalar tail.
#pragma omp simd aligned(row,b:64)
            for(int j=0;j<16;j++)row[j]+=a*b[j];
          } else {
#pragma omp simd aligned(row,b:64)
            for(int j=0;j<p.columns;j++)row[j]+=a*b[j];
          }
        }
      }
    }
  }
  for(int s=0;s<Spins;s++)for(int i=0;i<p.rows;i++)for(int j=0;j<p.columns;j++) {
    if(blocked)p.matrix[s][i][j]+=element(h,s,i,j);else p.matrix[s][i][j]=element(h,s,i,j);
  }
}
GRID_INLINE void hamiltonian_dispatch(const OpenMXGridPair& p,Worker& w,const Plan& plan) {
  if(plan.spins==1)hamiltonian_pair<1>(p,w,plan.orbitals);
  else if(plan.spins==2)hamiltonian_pair<2>(p,w,plan.orbitals);
  else hamiltonian_pair<4>(p,w,plan.orbitals);
}
GRID_AVX2 void hamiltonian_avx2(const OpenMXGridPair& p,Worker& w,const Plan& plan) {hamiltonian_dispatch(p,w,plan);}
GRID_AVX512 void hamiltonian_avx512(const OpenMXGridPair& p,Worker& w,const Plan& plan) {hamiltonian_dispatch(p,w,plan);}
}

struct OpenMXGridWorkspace {
  std::unique_ptr<Density> density;
  std::array<std::unique_ptr<Plan>,2> plans;
  std::array<std::unique_ptr<Exchange>,4> exchanges;
};
void GridWorkspaceDelete::operator()(OpenMXGridWorkspace* p) const noexcept {delete p;}
extern "C" int openmx_core_grid_workspace(OpenMXCoreContext* context,OpenMXGridWorkspace** out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(context && out,"missing grid calculation/output");
    if(!context->grid)context->grid.reset(new OpenMXGridWorkspace);
    *out=context->grid.get();
  });
}
extern "C" void openmx_core_grid_clear(OpenMXCoreContext* context) {if(context)context->grid.reset();}
extern "C" int openmx_core_grid_density_prepare(OpenMXGridWorkspace* w,int spins,size_t atoms,const size_t* points,
    size_t ranks,const int* send,const int* receive,OpenMXGridDensity* out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(w && out && valid_spins(spins) && atoms && points && (!ranks || (send && receive)),"invalid density grid storage request");
    (void)mul(add(atoms,1),sizeof(size_t));(void)mul(add(ranks,1),sizeof(size_t));
    if(!w->density || !w->density->matches(spins,atoms,points,ranks,send,receive))
      w->density=std::make_unique<Density>(spins,atoms,points,ranks,send,receive);
    auto& d=*w->density;std::fill_n(d.values.data(),d.values.size(),0.0);*out=d.view();
  });
}
extern "C" int openmx_core_grid_plan(OpenMXGridWorkspace* w,int stage,int spins,size_t atoms,size_t pairs,
    size_t max_orbitals,size_t workers,int simd,OpenMXGridAtom** atom_out,OpenMXGridPair** pair_out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(w && stage>=0 && stage<2 && valid_spins(spins) && workers && max_orbitals && atom_out && pair_out,
      "invalid grid assembly plan");
    require(simd>=OPENMX_GRID_SCALAR && simd<=openmx_core_grid_simd(nullptr),"unsupported grid SIMD mode");
    // Round up the minimum allocation as the blocked Hamiltonian uses 16 columns.
    max_orbitals=mul(add(std::max(size_t(16),max_orbitals),7)/8,8);
    auto& plan=w->plans[stage];
    if(!plan || plan->spins!=spins || plan->atoms.size()!=atoms || plan->pairs.size()!=pairs ||
        plan->orbitals!=max_orbitals || plan->workers.size()!=workers)
      plan=std::make_unique<Plan>(spins,atoms,pairs,max_orbitals,workers,simd);
    plan->simd=simd;*atom_out=plan->atoms.data();*pair_out=plan->pairs.data();
  });
}
extern "C" int openmx_core_grid_density_worker(OpenMXGridWorkspace* w,size_t worker,size_t team,
    double* seconds,void (*clock)(double*),char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(w && w->density && w->plans[OPENMX_GRID_DENSITY],"density grid not prepared");
    auto& p=*w->plans[OPENMX_GRID_DENSITY];auto& d=*w->density;
    require(team && worker<team && team<=p.workers.size() && p.spins==d.spins,"invalid density worker team");
    const size_t n=p.atoms.size(),base=n/team,tail=n%team;
    const size_t first=worker*base+std::min(worker,tail),count=base+(worker<tail);
    auto& scratch=p.workers[worker];
    for(size_t a=first;a<first+count;a++) {
      const auto& atom=p.atoms[a];double begin=0,end=0;if(seconds && clock)clock(&begin);
      require(atom.first<=p.pairs.size() && atom.count<=p.pairs.size()-atom.first,"invalid grid atom pair range");
      for(size_t k=atom.first;k<atom.first+atom.count;k++) {
        const auto& pair=p.pairs[k];validate(pair,p,false);
        require(pair.atom>=0 && size_t(pair.atom)<d.points.size(),"invalid density output atom");
        if(!pair.points || !pair.rows || !pair.columns)continue;
        if(p.simd==OPENMX_GRID_AVX512)density_avx512(pair,scratch,d,p.spins);
        else if(p.simd==OPENMX_GRID_AVX2)density_avx2(pair,scratch,d,p.spins);
        else density_dispatch<1>(pair,scratch,d,p.spins);
      }
      if(seconds && clock) {clock(&end);seconds[atom.global_atom]+=end-begin;}
    }
  });
}
extern "C" int openmx_core_grid_hamiltonian_worker(OpenMXGridWorkspace* w,size_t worker,size_t team,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(w && w->plans[OPENMX_GRID_HAMILTONIAN],"Hamiltonian grid not prepared");
    auto& plan=*w->plans[OPENMX_GRID_HAMILTONIAN];
    require(team && worker<team && team<=plan.workers.size(),"invalid Hamiltonian worker team");
    for(size_t k=worker;k<plan.pairs.size();k+=team) {
      const auto& pair=plan.pairs[k];validate(pair,plan,true);
      if(plan.simd==OPENMX_GRID_AVX512)hamiltonian_avx512(pair,plan.workers[worker],plan);
      else if(plan.simd==OPENMX_GRID_AVX2)hamiltonian_avx2(pair,plan.workers[worker],plan);
      else hamiltonian_dispatch(pair,plan.workers[worker],plan);
    }
  });
}
extern "C" int openmx_core_grid_exchange(OpenMXGridWorkspace* w,int kind,size_t send_values,size_t receive_values,
    size_t sends,size_t receives,size_t request_bytes,size_t status_bytes,OpenMXGridExchange* out,char* error,size_t capacity) {
  return checked(error,capacity,[&] {
    require(w && out && kind>=0 && kind<4 && request_bytes && status_bytes,"invalid grid exchange request");
    auto& e=w->exchanges[kind];
    if(!e || !e->fits(send_values,receive_values,sends,receives,request_bytes,status_bytes))
      e=std::make_unique<Exchange>(send_values,receive_values,sends,receives,request_bytes,status_bytes);
    *out=e->view();
  });
}
extern "C" int openmx_core_grid_simd(const char* value) {
  if(value && (value[0]=='0' || value[0]=='n' || value[0]=='N' || value[0]=='f' || value[0]=='F' ||
      std::strcmp(value,"scalar")==0 || std::strcmp(value,"SCALAR")==0))return OPENMX_GRID_SCALAR;
#if GRID_X86
  __builtin_cpu_init();
  if((!value || (std::strcmp(value,"avx2") && std::strcmp(value,"AVX2"))) && __builtin_cpu_supports("avx512f"))return OPENMX_GRID_AVX512;
  if(__builtin_cpu_supports("avx2"))return OPENMX_GRID_AVX2;
#endif
  return OPENMX_GRID_SCALAR;
}
extern "C" const char* openmx_core_grid_simd_name(int mode) {
  return mode==OPENMX_GRID_AVX512?"AVX-512":mode==OPENMX_GRID_AVX2?"AVX2":"scalar";
}
