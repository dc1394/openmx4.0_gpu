#include "openmx_core_olg_cache.h"
#include "olg_cache.hpp"
#include "allocation_faults.hpp"
#include <atomic>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
struct Context {
  OpenMXCoreContext* value;
  Context() {char e[256];value=openmx_core_create_host(e,sizeof(e));require(value,e);}
  ~Context() {openmx_core_destroy(value);}
  Context(const Context&)=delete;Context& operator=(const Context&)=delete;
};
struct Input {
  int map[4]{0,3,1,4},neighbors[5]{0,2,0,0,4};
  OpenMXOlgCacheInput view() const {return {1,2,3,4,map,neighbors};}
};
void prepare(Context& c,const OpenMXOlgCacheInput& input) {
  char e[256];require(openmx_core_olg_cache_prepare(c.value,&input,e,sizeof(e)),e);
}
bool matches(Context& c,const OpenMXOlgCacheInput& input) {
  char e[256];int result=-1;require(openmx_core_olg_cache_matches(c.value,&input,&result,e,sizeof(e)),e);return result;
}
void store(Context& c,size_t atom,size_t pair,size_t count,int seed) {
  std::vector<int> left(count),right(count);
  for(size_t i=0;i<count;++i){left[i]=seed+int(i)*3;right[i]=-seed-int(i)*7;}
  char e[256];require(openmx_core_olg_cache_store(c.value,atom,pair,count,left.data(),right.data(),e,sizeof(e)),e);
}
void fill(Context& c,const OpenMXOlgCacheInput& input,int seed) {
  for(size_t a=0;a<input.local_atoms;++a)
    for(int p=0;p<=input.neighbor_counts[input.local_to_global[a+1]];++p)
      store(c,a,size_t(p),size_t(p)%3==0?0:37,seed+int(a)*100+p);
}
// Exactly the native consumer's free() contract, including empty pair rows.
struct NativeAtom {
  OpenMXOlgCacheAtom value{};
  ~NativeAtom() {
    for(size_t p=0;p<value.pairs;++p){std::free(value.center[p]);std::free(value.neighbor[p]);}
    std::free(value.center);std::free(value.neighbor);
  }
  NativeAtom()=default;NativeAtom(const NativeAtom&)=delete;NativeAtom& operator=(const NativeAtom&)=delete;
};
void take(Context& c,size_t a,NativeAtom& atom) {
  char e[256];require(openmx_core_olg_cache_take_atom(c.value,a,&atom.value,e,sizeof(e)),e);
}
void check(const NativeAtom& atom,int seed) {
  for(size_t p=0;p<atom.value.pairs;++p) {
    const int count=p%3==0?0:37;
    require(atom.value.counts[p]==count && atom.value.center[p] && atom.value.neighbor[p],"wrong cache pair shape");
    for(int i=0;i<count;++i)
      require(atom.value.center[p][i]==seed+int(p)+i*3 && atom.value.neighbor[p][i]==-seed-int(p)-i*7,"cached pair changed");
  }
}
void layout() {
  Context c;Input tables;const auto input=tables.view();prepare(c,input);
  require(!matches(c,input),"incomplete cache matched");fill(c,input,11);require(matches(c,input),"complete cache missed");
  for(size_t a=0;a<input.local_atoms;++a){NativeAtom atom;take(c,a,atom);check(atom,11+int(a)*100);}
  require(!matches(c,input),"transferred cache matched");
}
void identity() {
  Context c,other;Input tables;auto input=tables.view();prepare(c,input);fill(c,input,13);
  require(!matches(other,input),"two contexts share a cache");
  auto key=input;key.md_iter++;require(!matches(c,key),"MD identity ignored");
  key=input;key.copied_cells++;require(!matches(c,key),"cell extent ignored");
  key=input;key.global_atoms++;require(!matches(c,key),"global extent ignored");
  key=input;key.local_atoms--;require(!matches(c,key),"local extent ignored");
  tables.map[2]=2;require(!matches(c,input),"atom identity ignored");tables.map[2]=1;
  tables.neighbors[1]++;require(!matches(c,input),"neighbour extent ignored");tables.neighbors[1]--;
  require(matches(c,input),"input mutation altered the cache snapshot");
  NativeAtom atom;take(c,1,atom);check(atom,113);
}
void invalid() {
  Context c;Input tables;auto input=tables.view();prepare(c,input);fill(c,input,17);char e[256];
  auto rejected=[&](OpenMXOlgCacheInput bad) {
    require(!openmx_core_olg_cache_prepare(c.value,&bad,e,sizeof(e)) && e[0],"bad cache identity accepted");
    int answer=83;require(!openmx_core_olg_cache_matches(c.value,&bad,&answer,e,sizeof(e)) && answer==83,"bad match changed output");
  };
  auto bad=input;bad.md_iter=-1;rejected(bad);bad=input;bad.copied_cells=-1;rejected(bad);
  bad=input;bad.local_atoms=5;rejected(bad);bad=input;bad.global_atoms=std::numeric_limits<size_t>::max();rejected(bad);
  bad=input;bad.local_to_global=nullptr;rejected(bad);bad=input;bad.neighbor_counts=nullptr;rejected(bad);
  for(int id:{-1,0,5}){tables.map[1]=id;rejected(input);}tables.map[1]=3;
  for(int n:{-1,INT_MAX}){tables.neighbors[3]=n;rejected(input);}tables.neighbors[3]=0;
  require(matches(c,input),"invalid prepare damaged old cache");
  int value=3;
  for(auto count:{size_t(INT_MAX)+1,std::numeric_limits<size_t>::max()})
    require(!openmx_core_olg_cache_store(c.value,0,0,count,&value,&value,e,sizeof(e)),"oversized list accepted");
  require(!openmx_core_olg_cache_store(c.value,3,0,0,nullptr,nullptr,e,sizeof(e)),"bad atom accepted");
  require(!openmx_core_olg_cache_store(c.value,0,1,0,nullptr,nullptr,e,sizeof(e)),"bad pair accepted");
  require(!openmx_core_olg_cache_store(c.value,0,0,1,nullptr,&value,e,sizeof(e)),"missing pair buffer accepted");
  OpenMXOlgCacheAtom out{};out.pairs=97;
  require(!openmx_core_olg_cache_take_atom(c.value,3,&out,e,sizeof(e)) && out.pairs==97 && !out.center,"failed take changed output");
  require(!openmx_core_olg_cache_take_atom(c.value,0,nullptr,e,sizeof(e)),"missing take output accepted");
  require(!openmx_core_olg_cache_prepare(nullptr,&input,e,sizeof(e)),"missing calculation accepted");
  require(!openmx_core_olg_cache_prepare(c.value,nullptr,e,sizeof(e)),"missing input accepted");
  require(!openmx_core_olg_cache_matches(c.value,&input,nullptr,e,sizeof(e)),"missing match output accepted");
  char tiny='x';bad=input;bad.md_iter=-1;
  require(!openmx_core_olg_cache_prepare(c.value,&bad,&tiny,1) && tiny=='\0',"short error not terminated");
  require(!openmx_core_olg_cache_prepare(c.value,&bad,nullptr,0),"error buffer required");
  NativeAtom atom;take(c,1,atom);check(atom,117);
}
void transfer() {
  Context c;Input tables;const auto input=tables.view();prepare(c,input);char e[256];
  OpenMXOlgCacheAtom refused{};refused.pairs=91;
  require(!openmx_core_olg_cache_take_atom(c.value,1,&refused,e,sizeof(e)) && refused.pairs==91,"partial atom transferred");
  fill(c,input,19);NativeAtom atom;take(c,1,atom);check(atom,119);
  const int* address=atom.value.center[1];
  require(!openmx_core_olg_cache_take_atom(c.value,1,&refused,e,sizeof(e)) && refused.pairs==91,"atom transferred twice");
  int value=1;require(!openmx_core_olg_cache_store(c.value,1,1,1,&value,&value,e,sizeof(e)),"transferred atom overwritten");
  // Cached counts expire here, while the transferred malloc buffers survive.
  openmx_core_olg_cache_release(c.value);atom.value.counts=nullptr;
  require(atom.value.center[1]==address && address[36]==119+1+36*3,"transfer copied or released the native row");
  prepare(c,input);fill(c,input,29);openmx_core_finish(c.value);
  require(address[0]==120,"finish freed a transferred row");
}
void failures() {
  Context c;Input tables;auto input=tables.view();prepare(c,input);fill(c,input,23);
  auto replacement=input;replacement.md_iter=7;char e[256];const auto live=openmx::test::live_allocations();
  int failures=0;bool success=false;
  for(int n=0;n<48;++n) {
    openmx::test::fail_allocation_after(n);
    const int ok=openmx_core_olg_cache_prepare(c.value,&replacement,e,sizeof(e));
    openmx::test::fail_allocation_after(-1);
    if(ok){success=true;break;}++failures;
    require(e[0] && matches(c,input),"failed prepare replaced old cache");
    require(openmx::test::live_allocations()==live,"failed prepare leaked metadata");
  }
  require(success && failures>=8,"cache metadata failure sweep incomplete");
  fill(c,replacement,31);NativeAtom atom;take(c,1,atom);check(atom,131);
}
long malloc_countdown=-1;
void* failing_malloc(size_t bytes) {
  if(malloc_countdown==0)return nullptr;
  if(malloc_countdown>0)--malloc_countdown;
  return std::malloc(bytes);
}
void allocator() {
  Input tables;const auto input=tables.view();
  for(int n=0;n<6;++n) {
    malloc_countdown=n;bool failed=false;
    try {openmx::core::OlgCache owner(input,failing_malloc);}catch(const std::bad_alloc&){failed=true;}
    malloc_countdown=-1;require(failed,"pointer-table allocation failure missed");
  }
  for(int n:{0,1}) {
    openmx::core::OlgCache owner(input,failing_malloc);int left=7,right=9;
    owner.store(0,0,1,&left,&right);malloc_countdown=n;bool failed=false;
    try {owner.store(0,0,1,&right,&left);}catch(const std::bad_alloc&){failed=true;}
    malloc_countdown=-1;require(failed,"pair allocation failure missed");NativeAtom atom;atom.value=owner.take_atom(0);
    require(atom.value.counts[0]==1 && atom.value.center[0][0]==7 && atom.value.neighbor[0][0]==9,"failed store replaced old pair");
  }
}
void empty() {
  Context c;const OpenMXOlgCacheInput input{1,0,0,0,nullptr,nullptr};prepare(c,input);
  require(matches(c,input),"empty partition cache missed");openmx_core_olg_cache_release(c.value);
  require(!matches(c,input),"released empty cache matched");openmx_core_olg_cache_release(c.value);openmx_core_olg_cache_release(nullptr);
}
void lifetime() {
  const auto live=openmx::test::live_allocations();
  for(int i=0;i<12;++i) {
    Context c;Input tables;auto input=tables.view();prepare(c,input);store(c,1,1,37,i);
    openmx_core_finish(c.value);require(!matches(c,input),"finish retained partial cache");
    prepare(c,input);fill(c,input,i);prepare(c,input);store(c,2,0,0,i);
  }
  require(openmx::test::live_allocations()==live,"context destruction leaked cache metadata");
}
void concurrent() {
  Context a,b;Input tables;const auto input=tables.view();prepare(a,input);prepare(b,input);
  std::atomic<bool> failed=false;
  std::vector<std::thread> threads;
  for(Context* c:{&a,&b})for(size_t atom=0;atom<input.local_atoms;++atom)
    for(int pair=0;pair<=tables.neighbors[tables.map[atom+1]];++pair)
      threads.emplace_back([&,c,atom,pair] {try {store(*c,atom,size_t(pair),pair%3==0?0:37,(c==&a?41:53)+int(atom)*100+pair);}catch(...){failed=true;}});
  for(auto& worker:threads)worker.join();
  require(!failed && matches(a,input) && matches(b,input),"concurrent store failed");
  NativeAtom left,right;take(a,1,left);take(b,1,right);check(left,141);check(right,153);
}
}
int main(int argc,char** argv) {
  try {
    require(argc==2,"expected one cache test case");const std::string_view name=argv[1];
    if(name=="layout")layout();else if(name=="identity")identity();else if(name=="invalid")invalid();
    else if(name=="transfer")transfer();else if(name=="failures")failures();else if(name=="allocator")allocator();
    else if(name=="empty")empty();else if(name=="lifetime")lifetime();else if(name=="concurrent")concurrent();
    else throw std::runtime_error("unknown cache test case");
    std::cout<<"PASS: "<<name<<'\n';return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
