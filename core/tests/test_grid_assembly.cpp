#include "openmx_core_grid.h"
#include "allocation_faults.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void require(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
using Context=std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)>;
Context context() {
  char error[256];Context out(openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
  require(bool(out),error);return out;
}
OpenMXGridWorkspace* workspace(OpenMXCoreContext* c) {
  char error[256];OpenMXGridWorkspace* out=nullptr;
  require(openmx_core_grid_workspace(c,&out,error,sizeof(error)),error);return out;
}
OpenMXGridDensity density(OpenMXGridWorkspace* w,int spins,const std::vector<size_t>& points) {
  char error[256];OpenMXGridDensity out;const int send[3]={0,7,2},receive[3]={9,0,4};
  require(openmx_core_grid_density_prepare(w,spins,points.size(),points.data(),3,send,receive,&out,error,sizeof(error)),error);
  return out;
}
template<class T> struct Orbitals {
  int points,orbitals,pad;bool transposed;
  std::vector<T> values;
  std::vector<T*> rows;
  // pad extra elements per ordinary row: a block stride wider than the row;
  // shift moves the block off the allocator's alignment.
  Orbitals(int n,int m,bool transpose,double phase,int padding=0,int shift=0):points(n),orbitals(m),pad(padding),
    transposed(transpose),values(size_t(n)*(m+padding)+size_t(shift)),rows(transpose?m:n) {
    for(size_t i=0;i<rows.size();i++)rows[i]=values.data()+shift+i*(transpose?size_t(n):size_t(m+padding));
    for(int g=0;g<n;g++)for(int i=0;i<m;i++) {
      const T v=T(std::sin(phase+g*0.073+i*0.33)*0.2);
      if(transpose)rows[i][g]=v;else rows[g][i]=v;
    }
    for(int g=0;g<n && !transpose;g++)for(int i=m;i<m+padding;i++)rows[g][i]=T(1e30);  // never read
  }
  double at(int g,int i) const {return transposed?double(rows[i][g]):double(rows[g][i]);}
  OpenMXGridOrbitals view(bool halo) const {return {rows.data(),transposed,halo,int(sizeof(T))};}
};
template<class T> struct Pair {
  static constexpr int grid_points=37;
  Orbitals<T> center,neighbor;
  std::vector<int> nc,nh,potential_index;
  std::array<std::vector<double>,4> matrix,potential;
  std::array<std::vector<double*>,4> rows;
  int r,c,points,spins;bool halo;
  Pair(int nr,int cols,int n,int s,bool transpose,bool overlap,int id,int padding=0,int shift=0)
    :center(grid_points,nr,transpose,0.3+id,padding,shift),neighbor(std::max(grid_points,n),cols,transpose,0.7+id,padding,shift),
     nc(n),nh(n),potential_index(grid_points),r(nr),c(cols),points(n),spins(s),halo(overlap) {
    for(int g=0;g<n;g++){nc[g]=(g*3+1)%grid_points;nh[g]=(g*7+3)%grid_points;}
    for(int g=0;g<grid_points;g++)potential_index[g]=(g*11+3)%grid_points;
    for(int spin=0;spin<s;spin++) {
      matrix[spin].resize(size_t(r)*c);rows[spin].resize(r);potential[spin].resize(grid_points);
      for(int i=0;i<r;i++) {
        rows[spin][i]=matrix[spin].data()+size_t(i)*c;
        for(int j=0;j<c;j++)rows[spin][i][j]=std::cos(0.71+spin*0.2+i*0.11+j*0.17)*0.13;
      }
      for(int g=0;g<grid_points;g++)potential[spin][g]=std::cos(g*0.13+spin*0.3);
    }
  }
  OpenMXGridPair view(int atom) {
    OpenMXGridPair p{};p.atom=atom;p.rows=r;p.columns=c;p.points=points;p.center_points=nc.data();p.neighbor_points=nh.data();
    p.potential_index=potential_index.data();p.center=center.view(false);p.neighbor=neighbor.view(halo);p.volume=0.137;
    for(int s=0;s<spins;s++){p.matrix[s]=rows[s].data();p.potential[s]=potential[s].data();}return p;
  }
};
template<class F> void team(size_t threads,F&& f) {
  std::array<int,8> success{};std::array<std::string,8> errors;
  std::vector<std::thread> workers;
  for(size_t t=0;t<threads;t++)workers.emplace_back([&,t] {
    char error[256];success[t]=f(t,threads,error,sizeof(error));errors[t]=error;
  });
  for(auto& worker:workers)worker.join();
  for(size_t t=0;t<threads;t++)require(success[t],errors[t].c_str());
}
template<class T> void quadrature(bool hamiltonian) {
  constexpr int atoms=3;
  for(int spins:{1,2,4})for(auto dims:{std::array<int,2>{3,5},std::array<int,2>{16,16},std::array<int,2>{19,17}})
    for(int points:{0,1,3,7,8,9,17,33,73})for(bool transpose:{false,true})for(int mode=0;mode<OPENMX_GRID_MODE_COUNT;mode++) {
      if(!openmx_core_grid_simd_width(mode))continue;
      auto ctx=context();auto* w=workspace(ctx.get());const auto storage=density(w,spins,{1,37,37,37});
      std::vector<std::unique_ptr<Pair<T>>> input;
      for(int i=0;i<atoms*2;i++)input.emplace_back(new Pair<T>(dims[0],dims[1],points,spins,transpose,i%2,i));
      OpenMXGridAtom* atom_plan;OpenMXGridPair* pairs;char error[256];
      const int stage=hamiltonian?OPENMX_GRID_HAMILTONIAN:OPENMX_GRID_DENSITY;
      require(openmx_core_grid_plan(w,stage,spins,atoms,input.size(),std::max(dims[0],dims[1]),4,mode,
          &atom_plan,&pairs,error,sizeof(error)),error);
      std::vector<long double> reference(size_t(spins)*atoms*37,0);
      std::vector<std::array<std::vector<long double>,4>> h_reference(input.size());
      for(int a=0;a<atoms;a++)atom_plan[a]={size_t(2*a),2,a+1};
      for(size_t k=0;k<input.size();k++) {
        auto& f=*input[k];pairs[k]=f.view(int(k/2)+1);
        for(int s=0;s<spins;s++) {
          h_reference[k][s].assign(f.matrix[s].begin(),f.matrix[s].end());
          for(int g=0;g<points;g++) {
            const int nc=f.nc[g],nh=f.halo?g:f.nh[g];long double sum=0;
            for(int i=0;i<f.r;i++) {
              long double tmp=0;
              for(int j=0;j<f.c;j++) {
                tmp+=(long double)f.rows[s][i][j]*f.neighbor.at(nh,j);
                h_reference[k][s][size_t(i)*f.c+j]+=(long double)0.137*f.potential[s][f.potential_index[nc]]*
                    f.center.at(nc,i)*f.neighbor.at(nh,j);
              }
              sum+=(long double)f.center.at(nc,i)*tmp;
            }
            reference[(size_t(s)*atoms+k/2)*37+nc]+=sum;
          }
        }
      }
      // A smaller actual team than the reserved team exercises dynamic OpenMP
      // team sizing without mixing the compiler's OpenMP runtime into this test.
      team(points%2?4:2,[&](size_t t,size_t n,char* e,size_t cap) {
        return hamiltonian?openmx_core_grid_hamiltonian_worker(w,t,n,e,cap):
          openmx_core_grid_density_worker(w,t,n,nullptr,nullptr,e,cap);
      });
      if(hamiltonian) {
        for(size_t k=0;k<input.size();k++)for(int s=0;s<spins;s++)for(size_t ij=0;ij<input[k]->matrix[s].size();ij++)
          require(std::abs((long double)input[k]->matrix[s][ij]-h_reference[k][s][ij])<2e-12L,"Hamiltonian differs from long double reference");
      } else {
        for(int s=0;s<spins;s++)for(int a=0;a<atoms;a++)for(int g=0;g<37;g++)
          require(std::abs((long double)storage.atoms[s][a+1][g]-reference[(size_t(s)*atoms+a)*37+g])<2e-12L,
            "density differs from long double reference");
      }
    }
}
void reuse() {
  auto ctx=context();auto* w=workspace(ctx.get());auto first=density(w,4,{1,9,0,7});
  first.atoms[3][3][6]=91;first.send[1][7]=17;first.receive[0][2]=29;
  const std::vector<size_t> same_points={1,9,0,7};
  openmx::test::fail_allocation_after(0);
  auto again=density(w,4,same_points);
  openmx::test::fail_allocation_after(-1);
  require(first.atoms==again.atoms && first.send==again.send && first.receive==again.receive,"density allocated during reuse");
  require(again.atoms[3][3][6]==0 && again.send[1][7]==17 && again.receive[0][2]==29,"density reset or packet lifetime changed");
  require(again.atom_values==68 && again.send_values==36 && again.receive_values==52,"density sizes differ");
  auto smaller=density(w,1,{1});require(smaller.atoms[0][0][0]==0,"empty rank storage invalid");
  OpenMXGridAtom* a;OpenMXGridPair* p;char error[256];
  require(openmx_core_grid_plan(w,0,1,0,0,1,4,0,&a,&p,error,sizeof(error)),error);
  team(4,[&](size_t t,size_t n,char* e,size_t cap){return openmx_core_grid_density_worker(w,t,n,nullptr,nullptr,e,cap);});
  openmx_core_grid_clear(ctx.get());w=workspace(ctx.get());
  auto fresh=density(w,4,{1,9,0,7});require(fresh.atoms[3][3][6]==0,"calculation retained previous grid");
  openmx_core_finish(ctx.get());w=workspace(ctx.get());require(w,"grid cannot be recreated after finish");
}
void exchange() {
  auto ctx=context();auto* w=workspace(ctx.get());char error[256];OpenMXGridExchange e;
  for(int kind=0;kind<4;kind++) {
    require(openmx_core_grid_exchange(w,kind,91,73,4,7,9,17,&e,error,sizeof(error)),error);
    require(uintptr_t(e.send_requests)%alignof(std::max_align_t)==0,"MPI request alignment");
    require(uintptr_t(e.receive_status)%alignof(std::max_align_t)==0,"MPI status alignment");
    e.send[90]=19;e.receive[72]=7;auto old=e;
    openmx::test::fail_allocation_after(0);
    const int ok=openmx_core_grid_exchange(w,kind,81,63,3,6,9,17,&e,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    require(ok && e.send==old.send && e.send_requests==old.send_requests && e.send[90]==19,"exchange reuse allocated/lost values");
    require(!openmx_core_grid_exchange(w,kind,SIZE_MAX,73,4,7,9,17,&e,error,sizeof(error)) && e.send==old.send,
      "exchange overflow damaged previous buffer");
  }
}
void invalid() {
  auto ctx=context();auto* w=workspace(ctx.get());char error[256];OpenMXGridAtom* a=nullptr;OpenMXGridPair* p=nullptr;
  require(!openmx_core_grid_plan(w,2,1,0,0,1,1,0,&a,&p,error,sizeof(error)),"invalid stage accepted");
  require(!openmx_core_grid_plan(w,0,3,0,0,1,1,0,&a,&p,error,sizeof(error)),"invalid spins accepted");
  require(!openmx_core_grid_plan(w,0,1,0,0,SIZE_MAX,1,0,&a,&p,error,sizeof(error)),"orbital overflow accepted");
  require(!openmx_core_grid_hamiltonian_worker(w,0,1,error,sizeof(error)),"missing plan accepted");
  const size_t bad[2]={1,SIZE_MAX};const int counts[1]={0};OpenMXGridDensity d{};
  require(!openmx_core_grid_density_prepare(w,1,2,bad,1,counts,counts,&d,error,sizeof(error)),"grid point overflow accepted");
  require(!d.atoms,"failed density changed output");
  require(openmx_core_grid_simd("scalar")==0 && openmx_core_grid_simd("0")==0 &&
    openmx_core_grid_simd("false")==0 && openmx_core_grid_simd("FALSE")==0 &&
    openmx_core_grid_simd("no")==0 && openmx_core_grid_simd("NO")==0 &&
    openmx_core_grid_simd_width(openmx_core_grid_simd("avx2"))<=4,"SIMD cap ignored");
  require(std::string(openmx_core_grid_simd_name(-1))=="scalar","SIMD name fallback");
  require(openmx_core_grid_simd_width(-1)==0 && openmx_core_grid_simd_width(99)==0 &&
      openmx_core_grid_simd_width(0)==1,"invalid SIMD width");
  for(int mode=1;mode<OPENMX_GRID_MODE_COUNT;++mode) {
    const int width=openmx_core_grid_simd_width(mode);
    require(!width || width==(mode==OPENMX_GRID_AVX2 || mode==OPENMX_GRID_SVE256?4:
        mode==OPENMX_GRID_AVX512 || mode==OPENMX_GRID_SVE512?8:2),
        "host SIMD ABI silently became scalar");
    if(!width)require(!openmx_core_grid_plan(w,0,1,0,0,1,1,mode,&a,&p,error,sizeof(error)),
        "unsupported host ISA accepted");
  }
}
template<class T> void equivalence() {
  // Cancellation, negative zero, unaligned orbital rows, every lane tail,
  // repeated center points, transposed/halo input, and nonzero H seeds.
  for(bool hamiltonian:{false,true})for(int spins:{1,2,4})for(bool transpose:{false,true})
    for(int columns:{1,4,5,8,12,13,19})for(int points=1;points<=17;++points) {
      std::vector<double> reference;
      for(int mode=0;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_grid_simd_width(mode))continue;
        Pair<T> f(13,columns,points+74,spins,transpose,true,1);
        for(auto& values:f.matrix)for(size_t i=0;i<values.size();++i)
          values[i]=i%5==0?-0.0:i%5==1?1e8:i%5==2?-1e8:values[i];
        auto ctx=context();auto* w=workspace(ctx.get());auto out=density(w,spins,{37});
        char error[256];OpenMXGridAtom* atoms;OpenMXGridPair* pairs;
        require(openmx_core_grid_plan(w,hamiltonian?1:0,spins,1,1,19,1,mode,
            &atoms,&pairs,error,sizeof(error)),error);
        atoms[0]={0,1,0};pairs[0]=f.view(0);
        openmx::test::fail_allocation_after(0);
        const int ok=hamiltonian?openmx_core_grid_hamiltonian_worker(w,0,1,error,sizeof(error)):
            openmx_core_grid_density_worker(w,0,1,nullptr,nullptr,error,sizeof(error));
        openmx::test::fail_allocation_after(-1);require(ok,error);
        std::vector<double> values;
        for(int s=0;s<spins;++s) {
          if(hamiltonian)values.insert(values.end(),f.matrix[s].begin(),f.matrix[s].end());
          else values.insert(values.end(),out.atoms[s][0],out.atoms[s][0]+37);
        }
        if(mode==0)reference=values;
        else require(values.size()==reference.size() &&
            !std::memcmp(values.data(),reference.data(),values.size()*sizeof(double)),
            "host SIMD changed the scalar accumulation order");
      }
    }
}
// Row layouts and every kernel the per-ISA choice can take (committed, rows
// or points across lanes) must give bitwise the scalar result: padded
// and misaligned row-major tables, halo (positional) neighbours, every lane
// tail, and transposed tables for the density (the Hamiltonian accumulates
// small transposed blocks unblocked, by design, so only its row-major
// layouts are compared).
template<class T> void layouts() {
  auto run=[](Pair<T>& f,bool hamiltonian,int spins,int mode) {
    for(auto& values:f.matrix)for(size_t i=0;i<values.size();++i)
      values[i]=i%5==0?-0.0:i%5==1?1e8:i%5==2?-1e8:values[i];
    auto ctx=context();auto* w=workspace(ctx.get());auto out=density(w,spins,{37});
    char error[256];OpenMXGridAtom* atoms;OpenMXGridPair* pairs;
    require(openmx_core_grid_plan(w,hamiltonian?1:0,spins,1,1,19,1,mode,&atoms,&pairs,error,sizeof(error)),error);
    atoms[0]={0,1,0};pairs[0]=f.view(0);
    openmx::test::fail_allocation_after(0);
    const int ok=hamiltonian?openmx_core_grid_hamiltonian_worker(w,0,1,error,sizeof(error)):
        openmx_core_grid_density_worker(w,0,1,nullptr,nullptr,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);require(ok,error);
    std::vector<double> values;
    for(int s=0;s<spins;++s) {
      if(hamiltonian)values.insert(values.end(),f.matrix[s].begin(),f.matrix[s].end());
      else values.insert(values.end(),out.atoms[s][0],out.atoms[s][0]+37);
    }
    return values;
  };
  auto same=[](const std::vector<double>& a,const std::vector<double>& b,const char* what) {
    require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(double)),what);
  };
  // rows and columns straddle every crossover of the per-ISA kernel choice
  for(bool hamiltonian:{false,true})for(int spins:{1,2,4})for(bool halo:{false,true})for(int rows:{1,2,3,4,7,8,13,16,19})
    for(int columns:{1,2,3,4,5,7,8,12,13,16,19})for(int points:{1,3,8,9,17,33}) {
    std::vector<double> scalar,scalar_transposed;
    for(int mode=0;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_grid_simd_width(mode))continue;
      Pair<T> plain(rows,columns,points,spins,false,halo,2);
      const auto reference=run(plain,hamiltonian,spins,mode);
      if(mode==OPENMX_GRID_SCALAR)scalar=reference;
      else same(reference,scalar,"host SIMD changed the scalar accumulation order");
      for(int padding:{1,3})for(int shift:{0,1}) {
        Pair<T> spaced(rows,columns,points,spins,false,halo,2,padding,shift);
        same(run(spaced,hamiltonian,spins,mode),reference,"padded or misaligned rows changed the result");
      }
      Pair<T> transposed(rows,columns,points,spins,true,halo,2);
      const auto columnwise=run(transposed,hamiltonian,spins,mode);
      if(mode==OPENMX_GRID_SCALAR)scalar_transposed=columnwise;
      else same(columnwise,scalar_transposed,"host SIMD changed the transposed-table result");
      if(!hamiltonian)same(columnwise,reference,"a transposed table changed the density");
    }
    }
}
void packed() {
  for(int spins:{1,2,4})for(int rows:{0,1,13})for(int columns:{0,1,3,5,7,8,9,13,17,19})
    for(int points:{0,1,3,17,33})for(int mode=0;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_grid_simd_width(mode))continue;
      const size_t stride=std::max(rows,columns)+3,count=size_t(spins)*stride*stride;
      std::vector<double> a(size_t(points)*rows),b(size_t(points)*columns),w(size_t(points)*spins);
      std::vector<double> result(count+2,719),expected(count,0.0);
      for(size_t i=0;i<a.size();++i)a[i]=i%5==0?-0.0:double(int(i%19)-9)*.13;
      for(size_t i=0;i<b.size();++i)b[i]=double(int(i%17)-8)*.07;
      for(size_t i=0;i<w.size();++i)w[i]=double(int(i%13)-6)*.19;
      for(int s=0;s<spins;++s)for(int i=0;i<rows;++i)for(int j=0;j<columns;++j) {
        double sum=0;
        for(int k=0;k<points;++k)sum+=((.237*w[size_t(k)*spins+s])*a[size_t(k)*rows+i])*b[size_t(k)*columns+j];
        expected[(size_t(s)*stride+i)*stride+j]=sum;
      }
      OpenMXGridPackedHamiltonian input{rows,columns,points,spins,stride,.237,a.data(),b.data(),w.data(),result.data()+1};
      openmx::test::fail_allocation_after(0);
      const int ok=openmx_core_grid_hamiltonian_packed(&input,mode);
      openmx::test::fail_allocation_after(-1);
      require(ok && result.front()==719 && result.back()==719,"packed SIMD guard or allocation failure");
      require(!std::memcmp(result.data()+1,expected.data(),count*sizeof(double)),"packed SIMD differs from original ascending quadrature");
    }
  double sentinel=19;OpenMXGridPackedHamiltonian input{0,0,0,1,0,1,nullptr,nullptr,nullptr,&sentinel};
  require(openmx_core_grid_hamiltonian_packed(&input,0) && sentinel==19,"empty packed output changed");
  for(int kind=0;kind<6;++kind) {
    auto bad=input;
    if(kind==0)bad.rows=-1;if(kind==1)bad.columns=1;if(kind==2)bad.points=-1;
    if(kind==3)bad.spins=3;if(kind==4)bad.output_stride=SIZE_MAX;
    require(!openmx_core_grid_hamiltonian_packed(&bad,kind==5?99:0) && sentinel==19,
        "invalid packed request changed output");
  }
  require(!openmx_core_grid_hamiltonian_packed(nullptr,0),"null packed request accepted");
}
void allocation() {
  // Sweep every allocation in a new topology. Failed replacement must preserve
  // both the old row tables and the values used by a live native consumer.
  int successes=0,failures=0;
  for(int fail=0;fail<35;fail++) {
    auto ctx=context();auto* w=workspace(ctx.get());auto old=density(w,1,{1,9});old.atoms[0][1][8]=31;
    const size_t points[3]={1,23,17};const int counts[3]={0,7,2};OpenMXGridDensity out=old;char error[256];
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    int ok=openmx_core_grid_density_prepare(w,4,3,points,3,counts,counts,&out,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(ok)successes++;else {
      failures++;require(out.atoms==old.atoms && old.atoms[0][1][8]==31 && openmx::test::live_allocations()==live,
        "failed density replacement leaked/invalidated old views");
    }
  }
  require(successes && failures,"allocation sweep did not cover success and failure");
}
thread_local double clock_tick=0;
void clock(double* t) {*t=clock_tick++;}
void plans() {
  for(int fail=0;fail<35;fail++) {
    auto ctx=context();auto* w=workspace(ctx.get());(void)density(w,1,{1,0});
    char error[256];OpenMXGridAtom* a=nullptr;OpenMXGridPair* p=nullptr;
    require(openmx_core_grid_plan(w,0,1,1,0,1,4,0,&a,&p,error,sizeof(error)),error);
    a[0]={0,0,1};auto* saved=a;auto* saved_pairs=p;
    const auto live=openmx::test::live_allocations();
    openmx::test::fail_allocation_after(fail);
    int ok=openmx_core_grid_plan(w,0,4,7,11,19,3,0,&a,&p,error,sizeof(error));
    openmx::test::fail_allocation_after(-1);
    if(!ok) {
      require(a==saved && p==saved_pairs && a[0].global_atom==1 && openmx::test::live_allocations()==live,
        "failed plan replacement invalidated old plan/leaked storage");
      double seconds[2]={0,0};clock_tick=0;
      require(openmx_core_grid_density_worker(w,0,1,seconds,clock,error,sizeof(error)),error);
      require(seconds[1]==1 && seconds[0]==0,"atom timing changed");
      require(!openmx_core_grid_density_worker(w,0,5,nullptr,nullptr,error,sizeof(error)),"oversized worker team accepted");
    }
  }
  auto ctx=context();auto* w=workspace(ctx.get());char error[256];OpenMXGridAtom* a;OpenMXGridPair* p;
  require(openmx_core_grid_plan(w,1,2,0,1,3,2,0,&a,&p,error,sizeof(error)),error);
  p[0]={};p[0].rows=999;
  require(!openmx_core_grid_hamiltonian_worker(w,0,1,error,sizeof(error)),"oversized matrix accepted");
  require(openmx_core_grid_plan(w,1,2,0,0,3,2,0,&a,&p,error,sizeof(error)),error);
  openmx::test::fail_allocation_after(0);
  int ok=openmx_core_grid_plan(w,1,2,0,0,3,2,0,&a,&p,error,sizeof(error));
  if(ok)ok=openmx_core_grid_hamiltonian_worker(w,0,1,error,sizeof(error));
  openmx::test::fail_allocation_after(-1);
  require(ok,"reused plan/worker allocated");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc==2?argv[1]:"";
    if(name=="density-float")quadrature<float>(false);else if(name=="density-double")quadrature<double>(false);
    else if(name=="hamiltonian-float")quadrature<float>(true);else if(name=="hamiltonian-double")quadrature<double>(true);
    else if(name=="reuse")reuse();else if(name=="exchange")exchange();else if(name=="invalid")invalid();
    else if(name=="simd-equivalence"){equivalence<float>();equivalence<double>();}
    else if(name=="packed")packed();
    else if(name=="layouts"){layouts<float>();layouts<double>();}
    else if(name=="allocation")allocation();else if(name=="plans")plans();else throw std::runtime_error("unknown grid assembly case");
    std::cout<<"PASS grid assembly "<<name<<'\n';return 0;
  }catch(const std::exception& e) {openmx::test::fail_allocation_after(-1);std::cerr<<e.what()<<'\n';return 1;}
}
