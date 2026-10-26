#include "openmx_core_grid.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Uses only the established C ABI, so the same executable can benchmark an
// older shared Core via LD_PRELOAD. Include packing/worker overhead in timing.
int main(int argc,char** argv) {
  try {
    const int repeats=argc>1?std::atoi(argv[1]):100;
    if(repeats<1)throw std::invalid_argument("positive repetition count required");
    const int mode=openmx_core_grid_simd(argc>2?argv[2]:nullptr);
    // argv[3]: "double" (default) or "float", the application's Orbs_Grid
    // element type.
    const bool single=argc>3 && std::string(argv[3])=="float";
    constexpr int points=4093;
    char error[256];
    auto check=[&](int ok){if(!ok)throw std::runtime_error(error);};
    for(int spins:{1,2,4})for(int orbitals:{1,4,8,12,13,19})for(int stage:{0,1}) {
      std::unique_ptr<OpenMXCoreContext,decltype(&openmx_core_destroy)> context(
          openmx_core_create_host(error,sizeof(error)),openmx_core_destroy);
      check(bool(context));OpenMXGridWorkspace* workspace;
      check(openmx_core_grid_workspace(context.get(),&workspace,error,sizeof(error)));
      std::vector<double> a(points*orbitals),b(points*orbitals);
      std::vector<float> af(points*orbitals),bf(points*orbitals);
      std::vector<const void*> ar(points),br(points);
      std::vector<int> nc(points),nh(points),potential_index(points);
      for(int g=0;g<points;++g) {
        ar[g]=single?(const void*)(af.data()+g*orbitals):(const void*)(a.data()+g*orbitals);
        br[g]=single?(const void*)(bf.data()+g*orbitals):(const void*)(b.data()+g*orbitals);
        nc[g]=(g*7)%points;nh[g]=(g*11)%points;potential_index[g]=g;
        for(int i=0;i<orbitals;++i) {
          a[size_t(g)*orbitals+i]=std::sin(g*.031+i*.13);b[size_t(g)*orbitals+i]=std::cos(g*.071+i*.19);
          af[size_t(g)*orbitals+i]=float(a[size_t(g)*orbitals+i]);bf[size_t(g)*orbitals+i]=float(b[size_t(g)*orbitals+i]);
        }
      }
      std::array<std::vector<double>,4> matrix,potential;
      std::array<std::vector<double*>,4> rows;
      OpenMXGridAtom* atoms;OpenMXGridPair* pairs;
      check(openmx_core_grid_plan(workspace,stage,spins,1,1,orbitals,1,mode,&atoms,&pairs,error,sizeof(error)));
      atoms[0]={0,1,0};auto& p=pairs[0];p={};p.rows=p.columns=orbitals;p.points=points;
      p.center_points=nc.data();p.neighbor_points=nh.data();p.potential_index=potential_index.data();
      const int bytes=single?int(sizeof(float)):int(sizeof(double));
      p.center={ar.data(),0,0,bytes};p.neighbor={br.data(),0,0,bytes};p.volume=.0137;
      for(int s=0;s<spins;++s) {
        matrix[s].resize(orbitals*orbitals);rows[s].resize(orbitals);potential[s].resize(points);
        for(int i=0;i<orbitals;++i){rows[s][i]=matrix[s].data()+i*orbitals;
          for(int j=0;j<orbitals;++j)rows[s][i][j]=std::cos(.1*i+.2*j+.3*s);}
        for(int g=0;g<points;++g)potential[s][g]=std::sin(.01*g+.4*s);
        p.matrix[s]=rows[s].data();p.potential[s]=potential[s].data();
      }
      OpenMXGridDensity density;const size_t counts[]={points};
      check(openmx_core_grid_density_prepare(workspace,spins,1,counts,0,nullptr,nullptr,&density,error,sizeof(error)));
      auto run=[&](){check(stage?openmx_core_grid_hamiltonian_worker(workspace,0,1,error,sizeof(error)):
          openmx_core_grid_density_worker(workspace,0,1,nullptr,nullptr,error,sizeof(error)));};
      run();std::vector<double> times;
      for(int trial=0;trial<5;++trial) {
        const auto start=std::chrono::steady_clock::now();
        for(int i=0;i<repeats;++i)run();
        times.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats);
      }
      std::sort(times.begin(),times.end());double checksum=0;
      for(int s=0;s<spins;++s) {
        if(stage)for(double v:matrix[s])checksum+=v;
        else for(int g=0;g<points;++g)checksum+=density.atoms[s][0][g];
      }
      std::cout<<std::setprecision(17)<<"{\"stage\":\""<<(stage?"hamiltonian":"density")
          <<"\",\"spins\":"<<spins<<",\"orbitals\":"<<orbitals<<",\"points\":"<<points
          <<",\"mode\":\""<<openmx_core_grid_simd_name(mode)
          <<"\",\"element\":\""<<(single?"float":"double")<<"\",\"seconds\":"<<times[2]
          <<",\"checksum\":"<<checksum<<"}\n";
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
