#include "host_olg_kernels.hpp"
#include "host_simd_abi.hpp"
#include <algorithm>
#include <climits>
#include <span>

// Overlap-grid intersection of UCell_Box on the host. The legacy device kernel
// answers every neighbour point with a leftmost binary search over the centre
// list; here a block of wide_lanes queries is answered at once. Because both
// lists are sorted by grid index, the leftmost index of a query is the number
// of centre entries below it, counted over a window that starts at the
// previous block's last answer (a vector merge: O(nc + nh) instead of
// O(nh log nc)). Out-of-order input falls back to the per-query search, so the
// result is identical to the device kernel for every input.
namespace openmx::core::host {
namespace {
static_assert(sizeof(OpenMXOlgQuery)==5*sizeof(int),"queries are read as five ints");
using Int=IntVector;
using Mask=typename Int::mask_type;
constexpr int V=wide_lanes;
constexpr int W=2*V;             // centre entries examined per window
constexpr int table_cells=1024;  // per-cell image table on the stack

Int gather(const int* p,size_t n,Int index) {
  return simd::unchecked_gather_from<Int>(std::span<const int>(p,n),index);
}
Int gather(const int* p,size_t n,Mask mask,Int index) {
  return simd::unchecked_gather_from<Int>(std::span<const int>(p,n),mask,index);
}
Int select(Mask mask,Int a,Int b) { return simd::condition(mask,a,b); }
#if defined(OPENMX_HOST_SIMD_AVX512)
bool any(Mask m) { return static_cast<__mmask16>(m)!=0; }
#elif defined(OPENMX_HOST_SIMD_AVX2)
bool any(Mask m) { return _mm256_movemask_ps(_mm256_castsi256_ps(static_cast<__m256i>(m)))!=0; }
#else
bool any(Mask m) { return simd::any_of(m); }
#endif

int lower_bound(const int* grid,int nc,int value) {
  int lo=0,hi=nc;
  while(lo<hi){const int mid=lo+(hi-lo)/2;if(grid[mid]<value)lo=mid+1;else hi=mid;}
  return lo;
}
// The device kernel's walk over repeated grid indices from a leftmost index.
int walk(const OpenMXOlgLists& l,int lo,int value,int target) {
  while(lo<l.nc && l.grid[lo]==value){if(l.cell[lo]==target)return lo;++lo;}
  return -1;
}
// The periodic image of a cell under a shift, or -1 outside the copied cells.
int image(const OpenMXOlgLists& l,int cell,int l1,int l2,int l3) {
  const int cpy=l.cpy,rdim=2*cpy+1;
  const int s1=l1+l.shifts[3*cell],s2=l2+l.shifts[3*cell+1],s3=l3+l.shifts[3*cell+2];
  if(s1<-cpy || s1>cpy || s2<-cpy || s2>cpy || s3<-cpy || s3>cpy)return -1;
  return l.periodic[(size_t(s1+cpy)*rdim+size_t(s2+cpy))*rdim+size_t(s3+cpy)];
}
// The two ways UCell_Box presents a neighbour list: query records (the device
// kernel's layout, one cell shift per record) or the list's own arrays with the
// pair's single shift.
struct Records {
  const OpenMXOlgQuery* q;
  int grid(size_t i) const { return q[i].grid; }
  int cell(size_t i) const { return q[i].cell; }
  int l1(size_t i) const { return q[i].l1; }
  int l2(size_t i) const { return q[i].l2; }
  int l3(size_t i) const { return q[i].l3; }
  static constexpr bool uniform=false;
};
struct Arrays {
  const OpenMXOlgNeighbours* n;
  int grid(size_t i) const { return n->grid[i]; }
  int cell(size_t i) const { return n->cell[i]; }
  int l1(size_t) const { return n->l1; }
  int l2(size_t) const { return n->l2; }
  int l3(size_t) const { return n->l3; }
  static constexpr bool uniform=true;
};
template<class Source> int scalar_query(const OpenMXOlgLists& l,const Source& s,size_t i) {
  const int target=image(l,s.cell(i),s.l1(i),s.l2(i),s.l3(i));
  return target<0?-1:walk(l,lower_bound(l.grid,l.nc,s.grid(i)),s.grid(i),target);
}

template<class Source> void intersect_impl(const OpenMXOlgLists& l,const Source& src,size_t count,int* out) {
  const int nc=l.nc;
  if(nc<=0){std::fill_n(out,count,-1);return;}
  // Every query of a UCell_Box chunk carries the pair's cell shift, so the
  // image of a cell is a table lookup; a block with other shifts, or more
  // cells than the table, uses the per-query image.
  int table[table_cells];
  const bool tabled=count && l.cells<=table_cells;
  const int L1=count?src.l1(0):0,L2=count?src.l2(0):0,L3=count?src.l3(0):0;
  if(tabled)for(int c=0;c<l.cells;++c)table[c]=image(l,c,L1,L2,L3);
  const Int zero(0),one(1),lane=Int([](auto k){return std::int32_t(k);});
  int pos=0;         // leftmost index of the previous block's last query
  int prev=INT_MIN;  // that query's grid index
  alignas(64) std::int32_t grids[V],cells[V];
  for(size_t first=0;first<count;first+=V) {
    const int n=int(std::min<size_t>(V,count-first));
    // One pass over the block: ascending from the previous block (else the
    // merge assumption fails), a uniform shift, and the vector operands.
    // Tail lanes replay the last query, so they never widen the window.
    bool ascending=true,uniform=tabled;
    for(int k=0;k<n;++k) {
      const int g=src.grid(first+k);
      ascending=ascending && g>=prev;prev=g;
      if(!Source::uniform)uniform=uniform && src.l1(first+k)==L1 && src.l2(first+k)==L2 && src.l3(first+k)==L3;
      grids[k]=g;cells[k]=src.cell(first+k);
    }
    for(int k=n;k<V;++k){grids[k]=grids[n-1];cells[k]=cells[n-1];}
    if(!ascending) {
      for(int k=0;k<n;++k)out[first+k]=scalar_query(l,src,first+k);
      pos=lower_bound(l.grid,nc,prev);
      continue;
    }
    const Int grid=simd::simd_unchecked_load<Int>(grids),cell=simd::simd_unchecked_load<Int>(cells);
    Int target;
    if(uniform)target=gather(table,size_t(l.cells),cell);
    else target=Int([&](auto k){const size_t i=first+size_t(std::min<int>(int(k),n-1));
        return std::int32_t(image(l,cells[k],src.l1(i),src.l2(i),src.l3(i)));});
    const Mask inside=target>=zero;
    // Leftmost index = pos + the number of centre entries in [pos, nc) below
    // the query. Lanes that met an entry >= their query stop counting by
    // themselves (the list ascends), so the window only moves on while a lane
    // has counted everything seen so far.
    Int below=zero;
    for(int base=pos;;) {
      const int end=std::min(base+W,nc);
      for(int w=base;w<end;++w)below+=select(grid>Int(l.grid[w]),one,zero);
      if(end==nc || !any(below==Int(end-pos)))break;
      base=end;
    }
    const Int lo=Int(pos)+below;
    const Mask within=lo<Int(nc);
    const Mask equal=within&&(gather(l.grid,size_t(nc),within,lo)==grid);
    const Mask hit=equal&&inside&&(gather(l.cell,size_t(nc),equal,lo)==target);
    simd::simd_partial_store(select(hit,lo,Int(-1)),out+first,lane<Int(n));
    // A repeated grid index whose first entry has another cell: walk on, but
    // only where the next centre entry repeats the index (rare).
    const Mask missed=equal&&inside&&!hit;
    if(any(missed)) {
      const Int next=lo+Int(1);
      const Mask repeated=missed&&(next<Int(nc));
      const Mask pending=repeated&&(gather(l.grid,size_t(nc),repeated,next)==grid);
      if(any(pending))for(int k=0;k<n;++k)if(pending[k])out[first+k]=walk(l,lo[k]+1,grids[k],target[k]);
    }
    pos=lo[n-1];
  }
}
void intersect(const OpenMXOlgLists& l,const OpenMXOlgQuery* q,size_t count,int* out) {
  intersect_impl(l,Records{q},count,out);
}
void intersect_lists(const OpenMXOlgLists& l,const OpenMXOlgNeighbours& n,size_t count,int* out) {
  intersect_impl(l,Arrays{&n},count,out);
}
// Map form: no merge state, so the queries need not be sorted and a block is
// three gathers (map, image table, centre cell) plus the duplicate walk.
void intersect_map(const OpenMXOlgLists& l,const OpenMXOlgNeighbours& nb,const int* map,size_t extent,
    size_t count,int* out) {
  const int nc=l.nc;
  if(nc<=0){std::fill_n(out,count,-1);return;}
  int table[table_cells];
  const bool tabled=l.cells<=table_cells;
  if(tabled)for(int c=0;c<l.cells;++c)table[c]=image(l,c,nb.l1,nb.l2,nb.l3);
  const Int zero(0),lane=Int([](auto k){return std::int32_t(k);});
  for(size_t first=0;first<count;first+=V) {
    const int n=int(std::min<size_t>(V,count-first));
    const Mask live=lane<Int(n);
    const Int grid=simd::simd_partial_load<Int>(nb.grid+first,live),cell=simd::simd_partial_load<Int>(nb.cell+first,live);
    const Int target=tabled?gather(table,size_t(l.cells),live,cell)
        :Int([&](auto k){return int(k)<n?std::int32_t(image(l,nb.cell[first+size_t(k)],nb.l1,nb.l2,nb.l3)):-1;});
    const Int index=gather(map,extent,live,grid);   // leftmost centre entry or -1
    const Mask valid=live&&(index>=zero)&&(target>=zero);
    const Mask hit=valid&&(gather(l.cell,size_t(nc),valid,index)==target);
    simd::simd_partial_store(select(hit,index,Int(-1)),out+first,live);
    // A matched grid index with another cell: walk on only where the next
    // centre entry repeats the index (rare: periodic images of one point).
    const Mask missed=valid&&!hit;
    if(any(missed)) {
      const Int next=index+Int(1);
      const Mask repeated=missed&&(next<Int(nc));
      const Mask pending=repeated&&(gather(l.grid,size_t(nc),repeated,next)==grid);
      if(any(pending))for(int k=0;k<n;++k)if(pending[k])out[first+k]=walk(l,index[k]+1,grid[k],target[k]);
    }
  }
}
// Branch-free compaction of one block of answers: every entry is written at
// the running position, which only advances on a match (at <= index always).
inline size_t compact(const int* answers,int n,size_t first,size_t at,int* center_out,int* neighbour_out) {
  for(int k=0;k<n;++k) {
    const int r=answers[k];
    center_out[at]=r;neighbour_out[at]=int(first)+k;at+=size_t(r>=0);
  }
  return at;
}
// Map form fused with the compaction: the block's answers never leave L1.
size_t pairs_map(const OpenMXOlgLists& l,const OpenMXOlgNeighbours& nb,const int* map,size_t extent,
    size_t count,int* center_out,int* neighbour_out) {
  const int nc=l.nc;
  if(nc<=0 || !count)return 0;
  int table[table_cells];
  const bool tabled=l.cells<=table_cells;
  if(tabled)for(int c=0;c<l.cells;++c)table[c]=image(l,c,nb.l1,nb.l2,nb.l3);
  const Int zero(0),lane=Int([](auto k){return std::int32_t(k);});
  alignas(64) std::int32_t answers[V];
  size_t at=0;
  for(size_t first=0;first<count;first+=V) {
    const int n=int(std::min<size_t>(V,count-first));
    const Mask live=lane<Int(n);
    const Int grid=simd::simd_partial_load<Int>(nb.grid+first,live),cell=simd::simd_partial_load<Int>(nb.cell+first,live);
    const Int target=tabled?gather(table,size_t(l.cells),live,cell)
        :Int([&](auto k){return int(k)<n?std::int32_t(image(l,nb.cell[first+size_t(k)],nb.l1,nb.l2,nb.l3)):-1;});
    const Int index=gather(map,extent,live,grid);
    const Mask valid=live&&(index>=zero)&&(target>=zero);
    const Mask hit=valid&&(gather(l.cell,size_t(nc),valid,index)==target);
    simd::simd_unchecked_store(select(hit,index,Int(-1)),answers);
    const Mask missed=valid&&!hit;
    if(any(missed)) {
      const Int next=index+Int(1);
      const Mask repeated=missed&&(next<Int(nc));
      const Mask pending=repeated&&(gather(l.grid,size_t(nc),repeated,next)==grid);
      if(any(pending))for(int k=0;k<n;++k)if(pending[k])answers[k]=walk(l,index[k]+1,grid[k],target[k]);
    }
    if(any(hit) || any(missed))at=compact(answers,n,first,at,center_out,neighbour_out);
  }
  return at;
}
// Merge form: answers first (into center_out), then compaction in place.
size_t pairs_lists(const OpenMXOlgLists& l,const OpenMXOlgNeighbours& nb,size_t count,int* center_out,int* neighbour_out) {
  intersect_lists(l,nb,count,center_out);
  size_t at=0;
  for(size_t i=0;i<count;++i){const int r=center_out[i];center_out[at]=r;neighbour_out[at]=int(i);at+=size_t(r>=0);}
  return at;
}
size_t pairs(const OpenMXOlgLists& l,const OpenMXOlgNeighbours& nb,const int* map,size_t extent,size_t count,
    int* center_out,int* neighbour_out) {
  return map?pairs_map(l,nb,map,extent,count,center_out,neighbour_out):pairs_lists(l,nb,count,center_out,neighbour_out);
}
constexpr OlgKernels kernels{V,intersect,intersect_lists,intersect_map,pairs};
}
const OlgKernels& OPENMX_HOST_SIMD_FACTORY(olg)() noexcept {return kernels;}
}
