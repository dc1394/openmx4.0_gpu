// Benchmark existing exact-order kernel alternatives, before changing dispatch.
// Compiled only as a local probe; no benchmark switch enters the application.
#include "host_grid_simd.cpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace openmx::core::host;

template<int S> void choose_density(int method,const OpenMXGridPair& p,GridScratch w,double** out) {
  if(method==0)committed().density(p,w,out,S);
  if(method==1)density<S>(p,w,out);
  if(method==2)density_rowwise<S,2>(p,w,out);
  if(method==3)density_rowwise<S,4>(p,w,out);
  if(method==4)density_tiled<S,2>(p,w,out);
  if(method==5)density_tiled<S,4>(p,w,out);
  if constexpr(S==4) {
    if(method==6)density_spins<1>(p,w,out);
    if(method==7)density_spins<2>(p,w,out);
    if(method==8)density_spins<4>(p,w,out);
  }
}
template<int S> void choose_hamiltonian(int method,const OpenMXGridPair& p,GridScratch w) {
  if(method==0)committed().hamiltonian(p,w,S);
  if(method==1)hamiltonian<S>(p,w);
  if(method==2)hamiltonian<S,2>(p,w);
  if(method==3)hamiltonian<S,4>(p,w);
  if constexpr(S==4) {
    if(method==4)hamiltonian_spins<1,4>(p,w);
    if(method==5)hamiltonian_spins<2,4>(p,w);
    if(method==6)hamiltonian_spins<4,4>(p,w);
    if(method==7)hamiltonian_spins<2,2>(p,w);
  }
}
int main() {
  constexpr int points=2003;
  std::puts("stage,rows,columns,spins,method,seconds,bitwise_equal");
  for(int rows:{1,4,8,12,13,19})for(int columns:{1,4,8,12,13,19})for(int spins:{1,2,4}) {
    const int o=(std::max(rows,columns)+7)/8*8;
    std::vector<double> dm(4*o*o),a(16*o),b(16*o),h(4*o*o);
    GridScratch scratch{dm.data(),a.data(),b.data(),h.data(),size_t(o)};
    std::vector<float> left(points*rows),right(points*columns);
    std::vector<const void*> lp(points),rp(points);
    std::vector<int> nc(points),nh(points),indices(points);
    for(int g=0;g<points;++g) {
      lp[g]=left.data()+g*rows;rp[g]=right.data()+g*columns;
      nc[g]=(g*7)%points;nh[g]=(g*11)%points;indices[g]=g;
      for(int i=0;i<rows;++i)left[g*rows+i]=std::sin(g*.031+i*.13);
      for(int j=0;j<columns;++j)right[g*columns+j]=std::cos(g*.071+j*.19);
    }
    OpenMXGridPair p{};p.rows=rows;p.columns=columns;p.points=points;p.volume=.0137;
    p.center_points=nc.data();p.neighbor_points=nh.data();p.potential_index=indices.data();
    p.center={lp.data(),0,0,4};p.neighbor={rp.data(),0,0,4};
    std::array<std::vector<double>,4> matrix,potential,out;
    std::array<std::vector<double*>,4> mr;
    double* output[4];
    for(int s=0;s<spins;++s) {
      matrix[s].resize(rows*columns);mr[s].resize(rows);potential[s].resize(points);out[s].resize(points);
      for(int i=0;i<rows;++i)mr[s][i]=matrix[s].data()+i*columns;
      for(int g=0;g<points;++g)potential[s][g]=std::sin(.01*g+.4*s);
      p.matrix[s]=mr[s].data();p.potential[s]=potential[s].data();output[s]=out[s].data();
    }
    for(int stage:{0,1}) {
      std::vector<double> reference;
      for(int method=0;method<(stage?(spins==4?8:4):(spins==4?9:6));++method) {
        for(int s=0;s<spins;++s) {
          std::fill(out[s].begin(),out[s].end(),0.);
          for(int i=0;i<rows;++i)for(int j=0;j<columns;++j)matrix[s][i*columns+j]=std::cos(.1*i+.2*j+.3*s);
        }
        auto run=[&] {
          if(stage) {
            if(spins==1)choose_hamiltonian<1>(method,p,scratch);
            else if(spins==2)choose_hamiltonian<2>(method,p,scratch);
            else choose_hamiltonian<4>(method,p,scratch);
          }else {
            if(spins==1)choose_density<1>(method,p,scratch,output);
            else if(spins==2)choose_density<2>(method,p,scratch,output);
            else choose_density<4>(method,p,scratch,output);
          }
        };
        run();std::vector<double> actual;
        for(int s=0;s<spins;++s) {
          auto& v=stage?matrix[s]:out[s];actual.insert(actual.end(),v.begin(),v.end());
        }
        if(method==0)reference=actual;
        bool same=std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(double))==0;
        if(!same)return 1;
        double times[5];
        for(int trial=0;trial<5;++trial) {
          auto start=std::chrono::steady_clock::now();
          for(int rep=0;rep<10;++rep)run();
          times[trial]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/10;
        }
        std::sort(times,times+5);
        std::printf("%s,%d,%d,%d,%d,%.12g,%d\n",stage?"hamiltonian":"density",rows,columns,spins,method,times[2],same);
      }
    }
  }
}
