#include "openmx_core_olg.h"
#include "openmx_core_host_simd.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The overlap-grid family must reproduce the device kernel of UCell_Box
// bit for bit: an independent brute-force reference, every usable ISA,
// sorted and shuffled queries, repeated grid indices, every lane tail.
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
struct Case {
  std::vector<int> grid,cell,shifts,periodic;
  std::vector<OpenMXOlgQuery> queries;
  int cpy=0,cells=0;
  OpenMXOlgLists lists() const {
    return {grid.data(),cell.data(),int(grid.size()),shifts.data(),cells,periodic.data(),cpy};
  }
};
Case make(std::mt19937& rng,int nc,int nq,int cpy,int cells,bool sorted,bool duplicates) {
  Case c;c.cpy=cpy;c.cells=cells;
  std::uniform_int_distribution<int> cell(0,cells-1),shift(-cpy,cpy),step(0,3),coin(0,9);
  int value=step(rng);
  for(int i=0;i<nc;++i) {
    c.grid.push_back(value);c.cell.push_back(cell(rng));
    if(!(duplicates && coin(rng)<3))value+=1+step(rng);  // 30%: the same grid index again
  }
  for(int i=0;i<3*cells;++i)c.shifts.push_back(shift(rng));
  const int rdim=2*cpy+1;
  for(int i=0;i<rdim*rdim*rdim;++i)c.periodic.push_back(cell(rng));
  for(int i=0;i<nq;++i) {
    OpenMXOlgQuery q;
    q.grid=nc && coin(rng)<7?c.grid[std::uniform_int_distribution<int>(0,nc-1)(rng)]:std::uniform_int_distribution<int>(0,value+3)(rng);
    q.cell=cell(rng);q.l1=shift(rng);q.l2=shift(rng);q.l3=shift(rng);
    c.queries.push_back(q);
  }
  if(sorted)std::sort(c.queries.begin(),c.queries.end(),[](auto& a,auto& b){return a.grid<b.grid;});
  return c;
}
// The reference answer, written without any search structure.
int reference(const Case& c,const OpenMXOlgQuery& q) {
  const int cpy=c.cpy,rdim=2*cpy+1;
  const int l1=q.l1+c.shifts[3*q.cell],l2=q.l2+c.shifts[3*q.cell+1],l3=q.l3+c.shifts[3*q.cell+2];
  if(c.grid.empty() || l1<-cpy || l1>cpy || l2<-cpy || l2>cpy || l3<-cpy || l3>cpy)return -1;
  const int target=c.periodic[((l1+cpy)*rdim+(l2+cpy))*rdim+(l3+cpy)];
  for(size_t i=0;i<c.grid.size();++i)if(c.grid[i]==q.grid && c.cell[i]==target)return int(i);
  return -1;
}
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
void compare(const Case& c,const char* what) {
  std::vector<int> expected;
  for(const auto& q:c.queries)expected.push_back(reference(c,q));
  const auto lists=c.lists();
  // The array form carries one shift: the queries' shifts are made uniform.
  std::vector<int> qgrid,qcell;
  for(const auto& q:c.queries){qgrid.push_back(q.grid);qcell.push_back(q.cell);}
  Case u=c;for(auto& q:u.queries){q.l1=c.queries[0].l1;q.l2=c.queries[0].l2;q.l3=c.queries[0].l3;}
  std::vector<int> uniform;
  for(const auto& q:u.queries)uniform.push_back(reference(u,q));
  const OpenMXOlgNeighbours arrays={qgrid.data(),qcell.data(),u.queries.empty()?0:u.queries[0].l1,
      u.queries.empty()?0:u.queries[0].l2,u.queries.empty()?0:u.queries[0].l3};
  for(int mode:modes()) {
    std::vector<int> out(c.queries.size()+2,-7);
    require(openmx_core_olg_intersect(&lists,c.queries.data(),c.queries.size(),out.data()+1,mode),"intersect refused a valid request");
    require(out.front()==-7 && out.back()==-7,"intersect wrote outside the output");
    for(size_t i=0;i<expected.size();++i)
      if(out[i+1]!=expected[i]) {
        std::cerr<<what<<": mode "<<openmx_core_host_simd_name(mode)<<" query "<<i<<" got "<<out[i+1]<<" expected "<<expected[i]<<'\n';
        throw std::runtime_error("host overlap-grid kernel differs from the reference");
      }
    // Map form: one map per centre list, unsorted queries allowed.
    {
      int extent=1;
      for(int g:c.grid)extent=std::max(extent,g+1);
      for(int g:qgrid)extent=std::max(extent,g+1);
      std::vector<int> map(size_t(extent)+3,-1);
      require(openmx_core_olg_map_fill(&lists,map.data(),size_t(extent),mode),"map_fill refused a valid list");
      std::vector<int> map_out(c.queries.size()+2,-7);
      require(openmx_core_olg_intersect_map(&lists,&arrays,map.data(),size_t(extent),c.queries.size(),map_out.data()+1,mode),"intersect_map refused a valid request");
      require(map_out.front()==-7 && map_out.back()==-7,"intersect_map wrote outside the output");
      for(size_t i=0;i<uniform.size();++i)
        if(map_out[i+1]!=uniform[i]) {
          std::cerr<<what<<": map mode "<<openmx_core_host_simd_name(mode)<<" query "<<i<<" got "<<map_out[i+1]<<" expected "<<uniform[i]<<'\n';
          throw std::runtime_error("host overlap-grid map kernel differs from the reference");
        }
      // Fused search + compaction, map and merge forms, against the reference.
      std::vector<int> ref_center,ref_neighbour;
      for(size_t i=0;i<uniform.size();++i)if(uniform[i]>=0){ref_center.push_back(uniform[i]);ref_neighbour.push_back(int(i));}
      for(const int* use:{(const int*)map.data(),(const int*)nullptr}) {
        std::vector<int> pc(c.queries.size()+1,-9),pn(c.queries.size()+1,-9);size_t found=777;
        require(openmx_core_olg_pairs(&lists,&arrays,use,size_t(extent),c.queries.size(),pc.data(),pn.data(),&found,mode),"pairs refused a valid request");
        require(found==ref_center.size(),"pairs found a different number of matches");
        for(size_t k=0;k<found;++k)require(pc[k]==ref_center[k] && pn[k]==ref_neighbour[k],"pairs differ from the compacted reference");
        require(pc.back()==-9 && pn.back()==-9,"pairs wrote beyond count");
      }
      require(openmx_core_olg_map_reset(&lists,map.data(),size_t(extent),mode),"map_reset refused a valid list");
      for(int v:map)require(v==-1,"map_reset left an entry");
    }
    std::vector<int> lists_out(c.queries.size()+2,-7);
    require(openmx_core_olg_intersect_lists(&lists,&arrays,c.queries.size(),lists_out.data()+1,mode),"intersect_lists refused a valid request");
    require(lists_out.front()==-7 && lists_out.back()==-7,"intersect_lists wrote outside the output");
    for(size_t i=0;i<uniform.size();++i)
      if(lists_out[i+1]!=uniform[i]) {
        std::cerr<<what<<": lists mode "<<openmx_core_host_simd_name(mode)<<" query "<<i<<" got "<<lists_out[i+1]<<" expected "<<uniform[i]<<'\n';
        throw std::runtime_error("host overlap-grid list kernel differs from the reference");
      }
  }
}
void sweep(bool sorted) {
  std::mt19937 rng(sorted?12345u:54321u);
  for(int nc:{0,1,2,3,7,15,16,17,31,33,100,1000,4097})for(int nq:{0,1,2,15,16,17,33,257,1000,5000})
    for(int cpy:{0,1,2})for(bool duplicates:{false,true}) {
      if(nq>1000 && nc<1000)continue;
      compare(make(rng,nc,nq,cpy,cpy==0?1:27,sorted,duplicates),sorted?"sorted":"shuffled");
    }
}
void edges() {
  std::mt19937 rng(7u);
  // An empty centre list answers -1 everywhere; an empty query set writes nothing.
  Case c=make(rng,0,40,1,27,true,false);
  compare(c,"empty centre");
  Case d=make(rng,50,0,1,27,true,true);
  const auto lists=d.lists();
  int sentinel=11;
  for(int mode:modes())require(openmx_core_olg_intersect(&lists,nullptr,0,&sentinel,mode) && sentinel==11,"empty query set");
  // Queries beyond either end of the centre list, every one at the same index,
  // and lists where every query repeats one grid value.
  Case e=make(rng,200,300,1,27,true,true);
  for(size_t i=0;i<e.queries.size();++i)e.queries[i].grid=i<100?0:i<200?e.grid[37]:e.grid.back()+9;
  compare(e,"ends and repeats");
  // A single cell image table, no periodic copies.
  compare(make(rng,500,700,0,1,true,true),"no copies");
}
void invalid() {
  std::mt19937 rng(3u);
  Case c=make(rng,20,10,1,27,true,false);
  auto lists=c.lists();std::vector<int> out(10,5);
  require(!openmx_core_olg_intersect(&lists,c.queries.data(),10,out.data(),99),"unknown mode accepted");
  require(!openmx_core_olg_intersect(&lists,c.queries.data(),10,out.data(),-1),"legacy mode accepted as a kernel");
  require(!openmx_core_olg_intersect(nullptr,c.queries.data(),10,out.data(),OPENMX_GRID_SCALAR),"missing lists accepted");
  require(!openmx_core_olg_intersect(&lists,nullptr,10,out.data(),OPENMX_GRID_SCALAR),"missing queries accepted");
  require(!openmx_core_olg_intersect(&lists,c.queries.data(),10,nullptr,OPENMX_GRID_SCALAR),"missing output accepted");
  auto broken=lists;broken.grid=nullptr;
  require(!openmx_core_olg_intersect(&broken,c.queries.data(),10,out.data(),OPENMX_GRID_SCALAR),"missing centre list accepted");
  broken=lists;broken.nc=-1;
  require(!openmx_core_olg_intersect(&broken,c.queries.data(),10,out.data(),OPENMX_GRID_SCALAR),"negative count accepted");
  broken=lists;broken.periodic=nullptr;
  require(!openmx_core_olg_intersect(&broken,c.queries.data(),10,out.data(),OPENMX_GRID_SCALAR),"missing periodic table accepted");
  std::vector<int> g(10,0),cl(10,0);const OpenMXOlgNeighbours arrays={g.data(),cl.data(),0,0,0};
  require(!openmx_core_olg_intersect_lists(&lists,&arrays,10,out.data(),99),"lists: unknown mode accepted");
  require(!openmx_core_olg_intersect_lists(&lists,nullptr,10,out.data(),OPENMX_GRID_SCALAR),"lists: missing neighbours accepted");
  OpenMXOlgNeighbours missing=arrays;missing.cell=nullptr;
  require(!openmx_core_olg_intersect_lists(&lists,&missing,10,out.data(),OPENMX_GRID_SCALAR),"lists: missing cells accepted");
  require(!openmx_core_olg_intersect_lists(&lists,&arrays,10,nullptr,OPENMX_GRID_SCALAR),"lists: missing output accepted");
  std::vector<int> map(64,-1);
  require(!openmx_core_olg_map_fill(&lists,map.data(),64,99),"map: unknown mode accepted");
  require(!openmx_core_olg_map_fill(&lists,map.data(),size_t(c.grid.back()),OPENMX_GRID_SCALAR),"map: short extent accepted");
  for(int v:map)require(v==-1,"a refused map_fill wrote the map");
  require(!openmx_core_olg_intersect_map(&lists,&arrays,nullptr,64,10,out.data(),OPENMX_GRID_SCALAR),"map: missing map accepted");
  std::vector<int> big(10,1000);const OpenMXOlgNeighbours beyond={big.data(),cl.data(),0,0,0};
  require(!openmx_core_olg_intersect_map(&lists,&beyond,map.data(),64,10,out.data(),OPENMX_GRID_SCALAR),"map: query beyond the extent accepted");
  {size_t found=5;std::vector<int> a(10,3),b(10,3);
   require(!openmx_core_olg_pairs(&lists,&arrays,map.data(),64,10,a.data(),b.data(),nullptr,OPENMX_GRID_SCALAR),"pairs: missing count accepted");
   require(!openmx_core_olg_pairs(&lists,&arrays,map.data(),64,10,nullptr,b.data(),&found,OPENMX_GRID_SCALAR),"pairs: missing output accepted");
   require(!openmx_core_olg_pairs(&lists,&arrays,map.data(),64,10,a.data(),b.data(),&found,99),"pairs: unknown mode accepted");
   require(found==5,"a refused pairs call wrote the count");for(int v:a)require(v==3,"a refused pairs call wrote output");}
  std::vector<int> negative(10,-1);const OpenMXOlgNeighbours below={negative.data(),cl.data(),0,0,0};
  require(!openmx_core_olg_intersect_map(&lists,&below,map.data(),64,10,out.data(),OPENMX_GRID_SCALAR),"map: negative query accepted");
  // The merge forms accept any query value (a negative one simply never matches).
  std::vector<int> any_out(10,5);
  require(openmx_core_olg_intersect_lists(&lists,&below,10,any_out.data(),OPENMX_GRID_SCALAR),"lists: negative query refused");
  for(int v:any_out)require(v==-1,"lists: negative query matched");
  for(int v:out)require(v==5,"a refused call wrote output");
}
void policy() {
  const int widest=openmx_core_host_simd_mode(nullptr);
  require(openmx_core_host_simd_usable(widest),"auto policy selected an unusable ISA");
  for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode)
    require(!openmx_core_host_simd_usable(mode) || mode<=widest || (mode==OPENMX_GRID_NEON && widest==OPENMX_GRID_SCALAR),"auto policy is not the widest usable ISA");
  require(openmx_core_host_simd_mode("")==widest && openmx_core_host_simd_mode("auto")==widest && openmx_core_host_simd_mode("AUTO")==widest,"auto spellings");
  for(const char* legacy:{"legacy","off","0","no","false","none","Legacy"})
    require(openmx_core_host_simd_mode(legacy)==OPENMX_HOST_SIMD_LEGACY,"legacy spellings");
  require(openmx_core_host_simd_mode("scalar")==OPENMX_GRID_SCALAR && openmx_core_host_simd_mode("SCALAR")==OPENMX_GRID_SCALAR,"scalar spelling");
  const int avx512=openmx_core_host_simd_mode("avx512");
  require(avx512==(openmx_core_host_simd_usable(OPENMX_GRID_AVX512)?OPENMX_GRID_AVX512:widest),"avx512 request");
  const int avx2=openmx_core_host_simd_mode("avx2");
  require(avx2==(openmx_core_host_simd_usable(OPENMX_GRID_AVX2)?OPENMX_GRID_AVX2:widest),"avx2 request");
  require(openmx_core_host_simd_mode("sse9")==widest,"unknown spelling falls back to the default policy");
  for(int mode:{OPENMX_GRID_NEON,OPENMX_GRID_SVE128,OPENMX_GRID_SVE256,OPENMX_GRID_SVE512}) {
    const char* name=openmx_core_host_simd_name(mode);
    const int expected=openmx_core_host_simd_usable(mode)?mode:widest;
    require(openmx_core_host_simd_mode(name)==expected,"ARM explicit selection");
    require(openmx_core_grid_simd(name)==expected,"grid ARM explicit selection");
    const int width=openmx_core_grid_simd_width(mode);
    require(bool(width)==bool(openmx_core_host_simd_usable(mode)),"grid and host ISA availability differ");
  }
  require(openmx_core_host_simd_mode("sve")==widest && openmx_core_grid_simd("SVE")==widest,
      "SVE auto spelling");
  if(openmx_core_host_simd_usable(OPENMX_GRID_NEON))
    require(openmx_core_grid_simd("neon")==OPENMX_GRID_NEON,"NEON was parsed as no");
  require(!std::strcmp(openmx_core_host_simd_name(OPENMX_HOST_SIMD_LEGACY),"legacy") &&
      !std::strcmp(openmx_core_host_simd_name(OPENMX_GRID_SCALAR),"scalar") &&
      !std::strcmp(openmx_core_host_simd_name(OPENMX_GRID_AVX2),"AVX2") &&
      !std::strcmp(openmx_core_host_simd_name(OPENMX_GRID_AVX512),"AVX-512") &&
      !std::strcmp(openmx_core_host_simd_name(OPENMX_GRID_NEON),"NEON"),"mode names");
  require(!openmx_core_host_simd_usable(99) && !openmx_core_host_simd_usable(-1),"unknown modes usable");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"equivalence";
    if(name=="equivalence")sweep(true);
    else if(name=="unsorted")sweep(false);
    else if(name=="edges")edges();
    else if(name=="invalid")invalid();
    else if(name=="modes")policy();
    else throw std::runtime_error("unknown host olg case");
    std::cout<<"PASS host olg "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
