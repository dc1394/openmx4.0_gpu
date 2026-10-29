#include "openmx_core_olg.h"
#include "openmx_core_host_simd.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// Overlap-grid intersection at the scale of one atom pair of a few hundred
// atoms (about 180k grid points per atom sphere), against the per-query
// leftmost binary search of the legacy device kernel run on the host.
namespace {
int legacy(const int* grid,const int* cell,int nc,const int* atv,const int* ratv,int cpy,const OpenMXOlgQuery& q) {
  const int rdim=2*cpy+1;
  const int l1=q.l1+atv[3*q.cell],l2=q.l2+atv[3*q.cell+1],l3=q.l3+atv[3*q.cell+2];
  if(!(nc && l1>=-cpy && l1<=cpy && l2>=-cpy && l2<=cpy && l3>=-cpy && l3<=cpy))return -1;
  const int target=ratv[((size_t)(l1+cpy)*rdim+(l2+cpy))*rdim+(l3+cpy)];
  int lo=0,hi=nc;
  while(lo<hi){int mid=lo+(hi-lo)/2;if(grid[mid]<q.grid)lo=mid+1;else hi=mid;}
  while(lo<nc && grid[lo]==q.grid){if(cell[lo]==target)return lo;lo++;}
  return -1;
}
}
int main(int argc,char** argv) {
  try {
    const int points=argc>1?std::atoi(argv[1]):180000,pairs=argc>2?std::atoi(argv[2]):20;
    if(points<1 || pairs<1)throw std::invalid_argument("positive point and pair counts required");
    std::mt19937 rng(42u);
    std::uniform_int_distribution<int> step(1,2),cell(0,26),shift(-1,1);
    // Two spheres of equal density that overlap over three quarters of their
    // points; half of the overlapping points share the periodic cell (a hit).
    std::vector<int> grid,cells;int value=0;
    for(int i=0;i<points;++i){grid.push_back(value);cells.push_back(cell(rng));value+=step(rng);}
    std::vector<OpenMXOlgQuery> queries;
    for(int i=0;i<points;++i) {
      const int j=points/4+i;OpenMXOlgQuery q;
      q.grid=j<points?grid[j]:grid.back()+1+(j-points)*2;
      q.cell=j<points && i%2?cells[j]:cell(rng);q.l1=0;q.l2=0;q.l3=0;
      queries.push_back(q);
    }
    std::vector<int> shifts(3*27),periodic(27);
    // cell c <-> lattice offset (a,b,c) in [-1,1]^3, so a zero shift maps a cell to itself
    for(int c=0;c<27;++c){shifts[3*c]=c/9-1;shifts[3*c+1]=c/3%3-1;shifts[3*c+2]=c%3-1;periodic[c]=c;}
    (void)shift;
    const OpenMXOlgLists lists={grid.data(),cells.data(),points,shifts.data(),27,periodic.data(),1};
    std::vector<int> out(points),reference(points);
    auto seconds=[&](auto&& body){const auto t0=std::chrono::steady_clock::now();for(int p=0;p<pairs;++p)body();
      return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();};
    const double legacy_seconds=seconds([&]{for(int i=0;i<points;++i)reference[i]=legacy(grid.data(),cells.data(),points,shifts.data(),periodic.data(),1,queries[i]);});
    size_t hits=0;for(int v:reference)hits+=v>=0;
    std::cout<<std::fixed<<std::setprecision(2)<<"points="<<points<<" pairs="<<pairs<<" hits="<<hits<<'\n'
        <<std::setw(24)<<"legacy binary search"<<std::setw(10)<<1e9*legacy_seconds/(double(points)*pairs)<<" ns/query\n";
    for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      const double s=seconds([&]{if(!openmx_core_olg_intersect(&lists,queries.data(),points,out.data(),mode))throw std::runtime_error("intersect failed");});
      if(out!=reference)throw std::runtime_error("host kernel differs from the legacy search");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*s/(double(points)*pairs)<<" ns/query  ("<<legacy_seconds/s<<"x)\n";
      std::vector<int> qgrid,qcell;for(const auto& q:queries){qgrid.push_back(q.grid);qcell.push_back(q.cell);}
      const OpenMXOlgNeighbours arrays={qgrid.data(),qcell.data(),0,0,0};
      const double a=seconds([&]{if(!openmx_core_olg_intersect_lists(&lists,&arrays,points,out.data(),mode))throw std::runtime_error("intersect_lists failed");});
      if(out!=reference)throw std::runtime_error("list kernel differs from the legacy search");
      std::cout<<std::setw(24)<<(std::string(openmx_core_host_simd_name(mode))+" (arrays)")<<std::setw(10)<<1e9*a/(double(points)*pairs)<<" ns/query  ("<<legacy_seconds/a<<"x)\n";
      const size_t extent=size_t(std::max(grid.back(),queries.back().grid))+1;
      std::vector<int> map(extent,-1);
      // In UCell_Box the map is filled once per centre atom and read by all
      // its neighbours: fill and reset once around the repeated pairs.
      const auto t0=std::chrono::steady_clock::now();
      if(!openmx_core_olg_map_fill(&lists,map.data(),extent,mode))throw std::runtime_error("map_fill failed");
      const double m=seconds([&]{if(!openmx_core_olg_intersect_map(&lists,&arrays,map.data(),extent,points,out.data(),mode))throw std::runtime_error("intersect_map failed");});
      if(!openmx_core_olg_map_reset(&lists,map.data(),extent,mode))throw std::runtime_error("map_reset failed");
      const double fill=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()-m;
      if(out!=reference)throw std::runtime_error("map kernel differs from the legacy search");
      {
        std::vector<int> pc(points),pn(points);size_t found=0;
        if(!openmx_core_olg_map_fill(&lists,map.data(),extent,mode))throw std::runtime_error("map_fill failed");
        const double f=seconds([&]{if(!openmx_core_olg_pairs(&lists,&arrays,map.data(),extent,points,pc.data(),pn.data(),&found,mode))throw std::runtime_error("pairs failed");});
        if(!openmx_core_olg_map_reset(&lists,map.data(),extent,mode))throw std::runtime_error("map_reset failed");
        size_t expect=0;for(int v:reference)expect+=v>=0;
        if(found!=expect)throw std::runtime_error("pairs count differs");
        std::cout<<std::setw(24)<<(std::string(openmx_core_host_simd_name(mode))+" (pairs)")<<std::setw(10)<<1e9*f/(double(points)*pairs)<<" ns/query  (search + compaction)\n";
      }
      std::cout<<std::setw(24)<<(std::string(openmx_core_host_simd_name(mode))+" (map)")<<std::setw(10)<<1e9*m/(double(points)*pairs)<<" ns/query  ("<<legacy_seconds/m<<"x; fill+reset "<<1e3*fill<<" ms per centre)\n";
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
