#include "openmx_core_force_simd.h"
#include "openmx_core_host_simd.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The force family must reproduce the legacy C loops of Force.c and the scalar
// Core projector block bit for bit: transcriptions of those loops are the
// reference, every usable ISA is compared with memcmp. The tiled drivers
// mirror the block orderings of the Force.c glue (k-outer per-q accumulators,
// kl buckets with slots, h tiles with an in-order pass).
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
template<class T> T awkward(std::mt19937& rng,size_t i) {
  // negative zero, cancellation pairs and ordinary values
  std::uniform_real_distribution<double> u(-1.0,1.0);
  const double v=u(rng);
  switch(i%7){case 0:return T(-0.0);case 1:return T(1e4*v);case 2:return T(-1e4*v);case 3:return T(1e-3*v);default:return T(v);}
}
struct Rows {
  std::vector<std::vector<float>> storage;std::vector<const float*> pointers;
  Rows()=default;
  Rows(std::mt19937& rng,int count,int length,size_t salt) {
    storage.resize(count);
    for(int n=0;n<count;++n){storage[n].resize(length+3);for(size_t l=0;l<storage[n].size();++l)storage[n][l]=awkward<float>(rng,l+n+salt);pointers.push_back(storage[n].data());}
  }
  const float* const* get() const { return pointers.data(); }
};
struct Lanes {
  std::vector<unsigned char> memory;OpenMXForceVnaLanes view{};
  bool prepare(const float* const* const* rows,int components,int count,int projectors,int mode) {
    const size_t bytes=openmx_core_force_vna_lanes_bytes(count,projectors,components);
    memory.assign(bytes+1,0);
    return openmx_core_force_vna_lanes(rows,components,count,projectors,memory.data(),bytes,&view,mode)!=0;
  }
};
// Force4B_case1_trace_fused_rows / Force4B_case2_trace_fused inner loops.
void legacy_vna(const OpenMXForceVnaBlock& b,std::vector<double> out[3]) {
  for(int m=0;m<b.rows;++m)for(int n=0;n<b.columns;++n) {
    double sum[3]={0.0,0.0,0.0};
    for(int l=0;l<b.projectors;++l)for(int c=0;c<3;++c) {
      const float* h=b.derivative[c][m];const float* q0=b.value[n];
      if(b.terms==1)sum[c]+=(b.fp32?(double)(h[l]*q0[l]):(double)h[l]*(double)q0[l]);
      else {
        const float* h0=b.row_value[m];const float* qc=b.column_derivative[c][n];
        sum[c]-=(b.fp32?(double)(h[l]*q0[l]):(double)h[l]*(double)q0[l])+(b.fp32?(double)(h0[l]*qc[l]):(double)h0[l]*(double)qc[l]);
      }
    }
    const size_t at=b.lanes_rows?size_t(n)*b.output_stride+m:size_t(m)*b.output_stride+n;
    for(int c=0;c<3;++c)out[c][at]=sum[c];
  }
}
void vna() {
  std::mt19937 rng(11u);
  const auto ms=modes();
  for(int terms:{1,2})for(int fp32:{0,1})for(int lanes_rows:{0,1})for(int prepared:{0,1})
    for(int rows:{1,3,8,13,17,70})for(int columns:{1,5,8,13,16,19,70})for(int P:{0,1,7,64,131}) {
    if((rows==3 || rows==17) && columns!=13 && prepared)continue;  // keep the sweep short
    // more than 64 lanes: the element-wise transposition; with one other extent only
    if((rows==70 || columns==70) && (rows+columns<70+13 || P==64 || P==1))continue;
    Rows dx(rng,rows,P,1),dy(rng,rows,P,2),dz(rng,rows,P,3),h0(rng,rows,P,4),q0(rng,columns,P,5),qx(rng,columns,P,6),qy(rng,columns,P,7),qz(rng,columns,P,8);
    OpenMXForceVnaBlock b{};
    b.rows=rows;b.columns=columns;b.projectors=P;b.terms=terms;b.fp32=fp32;b.lanes_rows=lanes_rows;
    b.derivative[0]=dx.get();b.derivative[1]=dy.get();b.derivative[2]=dz.get();b.row_value=h0.get();
    b.value=q0.get();b.column_derivative[0]=qx.get();b.column_derivative[1]=qy.get();b.column_derivative[2]=qz.get();
    const int lane_count=lanes_rows?rows:columns;
    b.output_stride=size_t(lane_count)+3;
    const size_t extent=size_t(lanes_rows?columns:rows)*b.output_stride+5;
    std::vector<double> expected[3],out[3];
    for(int c=0;c<3;++c){expected[c].assign(extent,-9.0);out[c]=expected[c];b.output[c]=expected[c].data();}
    legacy_vna(b,expected);
    const size_t bytes=openmx_core_force_vna_scratch(lane_count,P,terms,lanes_rows);
    std::vector<unsigned char> scratch(bytes+1);
    for(int mode:ms) {
      Lanes lanes;
      if(prepared) {
        const float* const* rows4[4];
        if(!lanes_rows){rows4[0]=q0.get();rows4[1]=qx.get();rows4[2]=qy.get();rows4[3]=qz.get();}
        else{rows4[0]=dx.get();rows4[1]=dy.get();rows4[2]=dz.get();rows4[3]=h0.get();}
        require(lanes.prepare(rows4,terms==2?4:(lanes_rows?3:1),lane_count,P,mode),"lane preparation refused");
        b.lanes=&lanes.view;
      }else b.lanes=nullptr;
      for(int c=0;c<3;++c){std::fill(out[c].begin(),out[c].end(),-9.0);b.output[c]=out[c].data();}
      require(openmx_core_force_vna_block(&b,prepared?nullptr:scratch.data(),prepared?0:bytes,mode),"vna block refused a valid request");
      for(int c=0;c<3;++c)if(std::memcmp(out[c].data(),expected[c].data(),out[c].size()*sizeof(double))) {
        std::cerr<<"vna terms="<<terms<<" fp32="<<fp32<<" lanes_rows="<<lanes_rows<<" prepared="<<prepared<<" rows="<<rows<<" columns="<<columns<<" P="<<P<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';
        throw std::runtime_error("vna block differs from the legacy loop");
      }
    }
  }
}
// Trace(CDM0*H) of one block in the legacy order; H is [m][n] or [n][m].
void trace(const double* const h[3],size_t stride,bool transposed,int rows,int columns,double pref,
    const std::vector<double>& cdm0,const std::vector<double>* cdm1,double f[3]) {
  for(int m=0;m<rows;++m)for(int n=0;n<columns;++n) {
    const size_t at=transposed?size_t(n)*stride+m:size_t(m)*stride+n;
    const double cdm=cdm1?cdm0[size_t(m)*columns+n]+(*cdm1)[size_t(m)*columns+n]:cdm0[size_t(m)*columns+n];
    const double scale=pref*cdm;
    f[0]+=scale*h[0][at];f[1]+=scale*h[1][at];f[2]+=scale*h[2][at];
  }
}
// The legacy per-block sum of case 1 (terms 1) or case 2 (terms 2), fp32.
void legacy_block(int terms,int rows,int columns,int P,const Rows* d,const Rows* rv,const Rows* value,const Rows* cd,
    std::vector<double> h[3]) {
  for(int c=0;c<3;++c)h[c].assign(size_t(rows)*columns,0.0);
  for(int m=0;m<rows;++m)for(int n=0;n<columns;++n)for(int l=0;l<P;++l)for(int c=0;c<3;++c) {
    const float* hx=d[c].pointers[m];const float* q0=value->pointers[n];
    if(terms==1)h[c][size_t(m)*columns+n]+=(double)(hx[l]*q0[l]);
    else h[c][size_t(m)*columns+n]-=(double)(hx[l]*q0[l])+(double)(rv->pointers[m][l]*cd[c].pointers[n][l]);
  }
}
void tiled() {
  std::mt19937 rng(41u);
  const auto ms=modes();
  const int fan=23,P=37,ian=7;
  std::uniform_int_distribution<int> coin(0,9),size_pick(0,3);
  const int sizes[4]={5,7,8,13};
  std::vector<int> jan(fan+1);for(auto& j:jan)j=sizes[size_pick(rng)];
  // ----- case 1, local q's: left blocks per k, right blocks per (q,k) -----
  std::vector<std::vector<Rows>> left(fan+1);  // left[k][c]
  for(int k=0;k<=fan;++k)for(int c=0;c<3;++c)left[k].emplace_back(rng,ian,P,size_t(k*3+c));
  std::vector<int> qs={1,2,5,9,14,20,23};
  std::vector<std::vector<int>> valid(qs.size(),std::vector<int>(fan+1));
  std::vector<std::vector<Rows>> right(qs.size());
  std::vector<std::vector<double>> cdm0(qs.size()),cdm1(qs.size());
  for(size_t qi=0;qi<qs.size();++qi) {
    right[qi].resize(fan+1);
    for(int k=0;k<=fan;++k){valid[qi][k]=coin(rng)<5;if(valid[qi][k])right[qi][k]=Rows(rng,jan[qs[qi]],P,size_t(100+qi*31+k));}
    cdm0[qi].resize(size_t(ian)*jan[qs[qi]]);cdm1[qi].resize(cdm0[qi].size());
    for(size_t t=0;t<cdm0[qi].size();++t){cdm0[qi][t]=awkward<double>(rng,t);cdm1[qi][t]=awkward<double>(rng,t+3);}
  }
  const double pref=4.0;
  for(int spin:{0,1}) {
    std::vector<double> expected(qs.size()*3,0.0);
    for(size_t qi=0;qi<qs.size();++qi) {
      double f[3]={0.0,0.0,0.0};
      for(int k=0;k<=fan;++k) {
        if(!valid[qi][k])continue;
        std::vector<double> h[3];legacy_block(1,ian,jan[qs[qi]],P,left[k].data(),nullptr,&right[qi][k],nullptr,h);
        const double* hp[3]={h[0].data(),h[1].data(),h[2].data()};
        trace(hp,size_t(jan[qs[qi]]),false,ian,jan[qs[qi]],pref,cdm0[qi],spin?&cdm1[qi]:nullptr,f);
      }
      for(int a=0;a<3;++a)expected[qi*3+a]=f[a];
    }
    for(int mode:ms) {
      // k-outer, q-inner: the left block of k prepared once, per-q accumulators
      std::vector<double> force(qs.size()*3,0.0);
      std::vector<double> block[3];for(auto& b:block)b.assign(size_t(ian)*(ian+1)*4,0.0);
      for(int k=0;k<=fan;++k) {
        bool any=false;for(size_t qi=0;qi<qs.size();++qi)any=any || valid[qi][k];
        if(!any)continue;
        const float* const* rows3[4]={left[k][0].get(),left[k][1].get(),left[k][2].get(),nullptr};
        Lanes lanes;require(lanes.prepare(rows3,3,ian,P,mode),"left lane preparation refused");
        for(size_t qi=0;qi<qs.size();++qi) {
          if(!valid[qi][k])continue;
          OpenMXForceVnaBlock b{};
          b.rows=ian;b.columns=jan[qs[qi]];b.projectors=P;b.terms=1;b.fp32=1;b.lanes_rows=1;b.lanes=&lanes.view;
          b.value=right[qi][k].get();b.output_stride=size_t(ian)+1;
          for(int c=0;c<3;++c)b.output[c]=block[c].data();
          require(openmx_core_force_vna_block(&b,nullptr,0,mode),"tiled case-1 block refused");
          const double* hp[3]={block[0].data(),block[1].data(),block[2].data()};
          trace(hp,b.output_stride,true,ian,jan[qs[qi]],pref,cdm0[qi],spin?&cdm1[qi]:nullptr,force.data()+qi*3);
        }
      }
      require(!std::memcmp(force.data(),expected.data(),force.size()*sizeof(double)),"k-outer case-1 tiling changed the forces");
      // kl buckets with slots (case 1, received atom): several centre atoms
      // share each right block; slots are summed per atom in ascending k.
      // Here the "atoms" are the q's with the roles swapped: left blocks of
      // q's are the per-atom derivatives, the right block of k is shared.
      std::vector<std::vector<double>> slots(qs.size()*(fan+1));
      for(int k=0;k<=fan;++k) {
        bool any=false;for(size_t qi=0;qi<qs.size();++qi)any=any || valid[qi][k];
        if(!any)continue;
        for(size_t qi=0;qi<qs.size();++qi) {  // every q has its own right block here, prepared per (q,k) as the bucket's lanes
          if(!valid[qi][k])continue;
          const float* const* rows1[4]={right[qi][k].get(),nullptr,nullptr,nullptr};
          Lanes lanes;require(lanes.prepare(rows1,1,jan[qs[qi]],P,mode),"right lane preparation refused");
          auto& slot=slots[qi*(fan+1)+k];slot.assign(3*size_t(ian)*(jan[qs[qi]]+2),0.0);
          OpenMXForceVnaBlock b{};
          b.rows=ian;b.columns=jan[qs[qi]];b.projectors=P;b.terms=1;b.fp32=1;b.lanes_rows=0;b.lanes=&lanes.view;
          for(int c=0;c<3;++c){b.derivative[c]=left[k][c].get();b.output[c]=slot.data()+size_t(c)*ian*(jan[qs[qi]]+2);}
          b.output_stride=size_t(jan[qs[qi]])+2;
          require(openmx_core_force_vna_block(&b,nullptr,0,mode),"bucketed case-1 block refused");
        }
      }
      std::fill(force.begin(),force.end(),0.0);
      for(size_t qi=0;qi<qs.size();++qi)for(int k=0;k<=fan;++k) {
        if(!valid[qi][k])continue;
        const auto& slot=slots[qi*(fan+1)+k];const size_t plane=size_t(ian)*(jan[qs[qi]]+2);
        const double* hp[3]={slot.data(),slot.data()+plane,slot.data()+2*plane};
        trace(hp,size_t(jan[qs[qi]])+2,false,ian,jan[qs[qi]],pref,cdm0[qi],spin?&cdm1[qi]:nullptr,force.data()+qi*3);
      }
      require(!std::memcmp(force.data(),expected.data(),force.size()*sizeof(double)),"bucketed case-1 slots changed the forces");
    }
  }
  // ----- case 2: neighbour blocks (value + derivatives), h-major pair list -----
  std::vector<std::vector<Rows>> blocks(fan+1);  // blocks[h][c], c=0 value, 1..3 derivatives
  for(int h=0;h<=fan;++h)for(int c=0;c<4;++c)blocks[h].emplace_back(rng,jan[h],P,size_t(500+h*4+c));
  struct Pair { int h,q,fused;double scale0;std::vector<double> cdm;double standin[3]; };
  std::vector<Pair> pairs;
  for(int h=0;h<=fan;++h)for(int q=h;q<=fan;++q) {
    if(coin(rng)>=6)continue;  // kl < 0
    Pair p;p.h=h;p.q=q;p.fused=h!=0 && q!=0 && h!=q;p.scale0=awkward<double>(rng,pairs.size());
    p.cdm.resize(size_t(jan[h])*jan[q]);for(auto& c:p.cdm)c=awkward<double>(rng,pairs.size()+1);
    for(int a=0;a<3;++a)p.standin[a]=awkward<double>(rng,a+pairs.size());
    pairs.push_back(p);
  }
  double expected2[3]={0.0,0.0,0.0};
  for(const auto& p:pairs) {
    if(!p.fused){for(int a=0;a<3;++a)expected2[a]+=p.standin[a];continue;}
    std::vector<double> h[3];legacy_block(2,jan[p.h],jan[p.q],P,&blocks[p.h][1],&blocks[p.h][0],&blocks[p.q][0],&blocks[p.q][1],h);
    const double* hp[3]={h[0].data(),h[1].data(),h[2].data()};
    trace(hp,size_t(jan[p.q]),false,jan[p.h],jan[p.q],p.scale0,p.cdm,nullptr,expected2);
  }
  for(int mode:ms)for(int tile_h:{1,2,3,5}) {
    double de[3]={0.0,0.0,0.0};
    size_t begin=0;
    while(begin<pairs.size()) {
      // the tile: tile_h distinct h values from begin on
      size_t end=begin;int distinct=0,last=-1;
      while(end<pairs.size() && (pairs[end].h==last || distinct<tile_h)){if(pairs[end].h!=last){++distinct;last=pairs[end].h;}++end;}
      std::vector<int> hs;for(size_t i=begin;i<end;++i)if(hs.empty() || hs.back()!=pairs[i].h)hs.push_back(pairs[i].h);
      std::vector<Lanes> lanes(hs.size());
      for(size_t i=0;i<hs.size();++i) {
        const float* const* rows4[4]={blocks[hs[i]][1].get(),blocks[hs[i]][2].get(),blocks[hs[i]][3].get(),blocks[hs[i]][0].get()};
        require(lanes[i].prepare(rows4,4,jan[hs[i]],P,mode),"h lane preparation refused");
      }
      // q-outer evaluation into slots
      std::vector<std::vector<double>> slot(end-begin);
      for(int q=0;q<=fan;++q)for(size_t i=begin;i<end;++i) {
        const auto& p=pairs[i];
        if(p.q!=q || !p.fused)continue;
        const size_t hi=size_t(std::find(hs.begin(),hs.end(),p.h)-hs.begin());
        auto& s=slot[i-begin];const size_t stride=size_t(jan[p.h])+1;s.assign(3*stride*jan[p.q],0.0);
        OpenMXForceVnaBlock b{};
        b.rows=jan[p.h];b.columns=jan[p.q];b.projectors=P;b.terms=2;b.fp32=1;b.lanes_rows=1;b.lanes=&lanes[hi].view;
        b.value=blocks[p.q][0].get();for(int c=0;c<3;++c){b.column_derivative[c]=blocks[p.q][c+1].get();b.output[c]=s.data()+size_t(c)*stride*jan[p.q];}
        b.output_stride=stride;
        require(openmx_core_force_vna_block(&b,nullptr,0,mode),"tiled case-2 block refused");
      }
      // in-order pass over the tile's pairs
      for(size_t i=begin;i<end;++i) {
        const auto& p=pairs[i];
        if(!p.fused){for(int a=0;a<3;++a)de[a]+=p.standin[a];continue;}
        const auto& s=slot[i-begin];const size_t stride=size_t(jan[p.h])+1;
        const double* hp[3]={s.data(),s.data()+stride*jan[p.q],s.data()+2*stride*jan[p.q]};
        trace(hp,stride,true,jan[p.h],jan[p.q],p.scale0,p.cdm,nullptr,de);
      }
      begin=end;
    }
    if(std::memcmp(de,expected2,sizeof(de))){std::cerr<<"case 2 tile_h="<<tile_h<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';throw std::runtime_error("h-tiled case-2 pass changed the force");}
  }
}
// Force3 part-2 loop (Force.c) for one pair.
void legacy_grid(const OpenMXForceGridPair& g,double sums[3]) {
  std::vector<double> orbs1(g.columns);
  for(int p=0;p<g.points;++p) {
    const int nc=g.center[p];const float* src=g.neighbor_rows[g.neighbor?g.neighbor[p]:p];
    for(int j=0;j<g.columns;++j)orbs1[j]=(double)src[j];
    for(int s=0;s<g.spins;++s) {
      double tmpx=0.0,tmpy=0.0,tmpz=0.0;
      for(int i=0;i<g.rows;++i) {
        double tmp0=0.0;const double* row=g.density[s][i];
        for(int j=0;j<g.columns;++j)tmp0+=orbs1[j]*row[j];
        tmpx+=g.derivative[nc][i][0]*tmp0;tmpy+=g.derivative[nc][i][1]*tmp0;tmpz+=g.derivative[nc][i][2]*tmp0;
      }
      const double v=g.potential[s][nc];
      sums[0]+=tmpx*v;sums[1]+=tmpy*v;sums[2]+=tmpz*v;
    }
  }
}
void grid() {
  std::mt19937 rng(23u);
  const auto ms=modes();
  for(int spins:{1,2,4})for(int rows:{1,5,8,13,19})for(int columns:{1,4,13,16})for(int points:{0,1,17,300})for(bool indexed:{false,true}) {
    const int grid_points=points+7;
    std::vector<std::vector<double>> dm(size_t(spins)*rows);std::vector<std::vector<const double*>> dmrows(spins);
    for(int s=0;s<spins;++s)for(int i=0;i<rows;++i){auto& r=dm[size_t(s)*rows+i];r.resize(columns);for(int j=0;j<columns;++j)r[j]=awkward<double>(rng,j+i);dmrows[s].push_back(r.data());}
    Rows orbitals(rng,grid_points,columns,9);
    std::vector<double> dstore(size_t(grid_points)*rows*3);std::vector<const double*> drows(size_t(grid_points)*rows);std::vector<const double* const*> dpoints(grid_points);
    for(size_t k=0;k<dstore.size();++k)dstore[k]=awkward<double>(rng,k);
    for(int nc=0;nc<grid_points;++nc){for(int i=0;i<rows;++i)drows[size_t(nc)*rows+i]=dstore.data()+(size_t(nc)*rows+i)*3;dpoints[nc]=drows.data()+size_t(nc)*rows;}
    std::vector<std::vector<double>> potential(spins);std::vector<const double*> prow;
    for(int s=0;s<spins;++s){potential[s].resize(grid_points);for(auto& v:potential[s])v=awkward<double>(rng,s);prow.push_back(potential[s].data());}
    std::vector<int> center(points),neighbor(points);
    std::uniform_int_distribution<int> pick(0,grid_points-1);
    for(int p=0;p<points;++p){center[p]=pick(rng);neighbor[p]=pick(rng);}
    OpenMXForceGridPair g{};
    g.rows=rows;g.columns=columns;g.spins=spins;g.points=points;
    for(int s=0;s<spins;++s)g.density[s]=dmrows[s].data();
    g.center=center.data();g.neighbor=indexed?neighbor.data():nullptr;g.neighbor_rows=orbitals.get();
    g.derivative=dpoints.data();g.potential=prow.data();
    double expected[3]={0.5,-0.25,3.0};legacy_grid(g,expected);
    const size_t bytes=openmx_core_force_grid_scratch(rows,columns,spins);
    std::vector<unsigned char> scratch(bytes+1);
    for(int mode:ms) {
      double sums[3]={0.5,-0.25,3.0};
      require(openmx_core_force_grid_pair(&g,sums,scratch.data(),bytes,mode),"grid pair refused a valid request");
      if(std::memcmp(sums,expected,sizeof(sums))) {
        std::cerr<<"grid spins="<<spins<<" rows="<<rows<<" columns="<<columns<<" points="<<points<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';
        throw std::runtime_error("grid pair differs from the legacy loop");
      }
    }
  }
}
// Point-major Force3 part 2 of one atom against the pair-major legacy loops.
void points() {
  std::mt19937 rng(61u);
  const auto ms=modes();
  for(int spins:{1,2})for(int rows:{1,9,13})for(size_t row_stride:{size_t(3),size_t(4)})for(int extra:{0,5}) {
    const int npoints=120,npairs=5;const int columns_of[5]={5,7,8,13,16};
    const int columns_max=16;
    const size_t point_stride=size_t(rows)*row_stride+extra;
    std::vector<double> derivative(size_t(npoints)*point_stride+1);for(size_t k=0;k<derivative.size();++k)derivative[k]=awkward<double>(rng,k);
    std::vector<std::vector<double>> potential(spins);std::vector<const double*> prow;
    for(int s=0;s<spins;++s){potential[s].resize(npoints);for(auto& v:potential[s])v=awkward<double>(rng,s+2);prow.push_back(potential[s].data());}
    struct PairData { int columns;std::vector<std::vector<double>> dm;std::vector<std::vector<const double*>> dmrows;Rows rows;std::vector<int> center,neighbor; };
    std::vector<PairData> pd(npairs);
    std::uniform_int_distribution<int> row_pick(0,49),coin(0,9);
    for(int pr=0;pr<npairs;++pr) {
      auto& d=pd[pr];d.columns=columns_of[pr];d.dm.resize(size_t(spins)*rows);d.dmrows.resize(spins);
      for(int s=0;s<spins;++s)for(int i=0;i<rows;++i){auto& r=d.dm[size_t(s)*rows+i];r.resize(d.columns);for(auto& v:r)v=awkward<double>(rng,i+pr);d.dmrows[s].push_back(r.data());}
      d.rows=Rows(rng,50,d.columns,size_t(pr*7));
      for(int p=0;p<npoints;++p)if(coin(rng)<4){d.center.push_back(p);d.neighbor.push_back(row_pick(rng));}
    }
    // legacy: pair by pair, point by point, spin by spin
    double expected[3]={1.0,2.0,-3.0};
    for(int pr=0;pr<npairs;++pr) {
      const auto& d=pd[pr];std::vector<double> orbs1(d.columns);
      for(size_t v=0;v<d.center.size();++v) {
        const int nc=d.center[v];const float* src=d.rows.pointers[d.neighbor[v]];
        for(int j=0;j<d.columns;++j)orbs1[j]=(double)src[j];
        for(int s=0;s<spins;++s) {
          double tx=0,ty=0,tz=0;
          for(int i=0;i<rows;++i){double t0=0;for(int j=0;j<d.columns;++j)t0+=orbs1[j]*d.dmrows[s][i][j];
            const double* di=derivative.data()+size_t(nc)*point_stride+size_t(i)*row_stride;tx+=di[0]*t0;ty+=di[1]*t0;tz+=di[2]*t0;}
          const double pot=potential[s][nc];expected[0]+=tx*pot;expected[1]+=ty*pot;expected[2]+=tz*pot;
        }
      }
    }
    // CSR over points, terms in pair/visit order
    std::vector<size_t> offsets(npoints+1,0);size_t total=0;
    for(const auto& d:pd)for(int nc:d.center)offsets[nc+1]++;
    for(int p=0;p<npoints;++p)offsets[p+1]+=offsets[p];
    std::vector<size_t> cursor(offsets.begin(),offsets.end()-1);
    std::vector<int> visit_pair(offsets[npoints]),visit_row(offsets[npoints]),visit_term(offsets[npoints]),pair_columns(npairs);
    std::vector<const float* const*> pair_rows(npairs);
    std::vector<size_t> pair_offset(npairs+1,0);
    for(int pr=0;pr<npairs;++pr){pair_offset[pr+1]=pair_offset[pr]+pd[pr].center.size();pair_columns[pr]=pd[pr].columns;pair_rows[pr]=pd[pr].rows.get();
      for(size_t v=0;v<pd[pr].center.size();++v){const size_t at=cursor[pd[pr].center[v]]++;visit_pair[at]=pr;visit_row[at]=pd[pr].neighbor[v];visit_term[at]=int(pair_offset[pr]+v);total++;}}
    for(int mode:ms) {
      const size_t dm_bytes=openmx_core_force_grid_density_bytes(rows,columns_max,spins,npairs);
      std::vector<double> dm(dm_bytes/sizeof(double)+1,7.0);
      for(int pr=0;pr<npairs;++pr) {
        const double* const* density[4]={nullptr,nullptr,nullptr,nullptr};
        for(int s=0;s<spins;++s)density[s]=pd[pr].dmrows[s].data();
        require(openmx_core_force_grid_density(density,rows,pd[pr].columns,columns_max,spins,pr,dm.data(),dm_bytes,mode),"density planes refused");
      }
      std::vector<double> terms(total*spins*3+1,-5.0);
      OpenMXForceGridPoints g{};
      g.rows=rows;g.spins=spins;g.points=npoints;g.columns_max=columns_max;g.pad_rows=(size_t(rows)+15)/16*16;
      g.dm=dm.data();g.pair_columns=pair_columns.data();g.point_offsets=offsets.data();g.visit_pair=visit_pair.data();g.visit_row=visit_row.data();
      g.visit_term=visit_term.data();g.pair_rows=pair_rows.data();g.derivative=derivative.data();g.point_stride=point_stride;g.row_stride=row_stride;
      g.potential=prow.data();g.terms=terms.data();
      const size_t bytes=openmx_core_force_grid_points_scratch(rows);std::vector<unsigned char> scratch(bytes+1);
      require(openmx_core_force_grid_points(&g,0,70,scratch.data(),bytes,mode) && openmx_core_force_grid_points(&g,70,npoints-70,scratch.data(),bytes,mode),"grid points refused");
      require(terms.back()==-5.0,"grid points wrote beyond the terms");
      double sums[3]={1.0,2.0,-3.0};
      for(int pr=0;pr<npairs;++pr)for(size_t v=0;v<pd[pr].center.size();++v)for(int s=0;s<spins;++s)
        for(int a=0;a<3;++a)sums[a]+=terms[((pair_offset[pr]+v)*spins+s)*3+a];
      if(std::memcmp(sums,expected,sizeof(sums))){std::cerr<<"points spins="<<spins<<" rows="<<rows<<" row_stride="<<row_stride<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';throw std::runtime_error("point-major pass changed the force");}
    }
  }
}
// projector_scalar_block of core/src/projector_force.cpp.
void legacy_nl(const OpenMXForceProjectorSpecies& sp,const OpenMXProjectorForceBlock& b) {
  for(size_t m=0;m<b.rows;++m)for(size_t n=0;n<b.columns;++n) {
    double sum[3]={0.0,0.0,0.0};
    for(size_t t=0;t<b.term_count;++t) {
      const auto& term=b.terms[t];
      const size_t ai=(term.transpose?n:m)*term.derivative_stride,bi=(term.transpose?m:n)*term.value_stride;
      for(size_t j=0;j<sp.count;++j) {
        const auto& channel=sp.channels[j];
        const double energy=term.reverse_energy?-channel.energy[0]:channel.energy[0];
        for(int magnetic=0;magnetic<=channel.twice_angular;++magnetic) {
          const size_t p=channel.offset+magnetic;double x[3];
          for(int axis=0;axis<3;++axis) {
            const double a=term.derivative[0][axis][ai+p],v=term.value[0][bi+p];
            x[axis]=term.value_first?energy*v*a:energy*a*v;
          }
          for(int axis=0;axis<3;++axis){if(term.sign==1)sum[axis]+=x[axis];else sum[axis]-=x[axis];}
        }
      }
    }
    for(int axis=0;axis<3;++axis) {
      dcomplex* out=b.output[axis];
      if(b.accumulate)for(size_t spin=0;spin<2;++spin)out[(spin*b.output_rows+m)*b.output_stride+n].r+=sum[axis];
      else {for(size_t spin=0;spin<3;++spin)out[(spin*b.output_rows+m)*b.output_stride+n]=dcomplex{0.0,0.0};
        out[(0*b.output_rows+m)*b.output_stride+n].r=sum[axis];out[(1*b.output_rows+m)*b.output_stride+n].r=sum[axis];}
    }
  }
}
void nl() {
  std::mt19937 rng(37u);
  const auto ms=modes();
  const int twice[]={0,2,4,0,2,6};
  std::vector<OpenMXForceProjectorChannel> channels;size_t components=0;
  for(int t:twice){OpenMXForceProjectorChannel c{};c.twice_angular=t;c.offset=components;c.energy[0]=awkward<double>(rng,components+4);c.energy[1]=0.0;c.weight[0]=c.weight[1]=1.0;channels.push_back(c);components+=size_t(t)+1;}
  OpenMXForceProjectorSpecies sp{};sp.relativistic=0;sp.count=channels.size();sp.components=components;sp.channels=channels.data();
  for(size_t term_count:{size_t(1),size_t(2)})for(int rows:{1,4,13})for(int columns:{1,7,13})for(int flags=0;flags<16;++flags)for(int accumulate:{0,1}) {
    const size_t dstride=components+2,vstride=components+1;
    const size_t extent=size_t(std::max(rows,columns))+1;
    std::vector<double> d[2][3],v[2];
    OpenMXProjectorForceTerm terms[2]{};
    for(size_t t=0;t<term_count;++t) {
      for(int axis=0;axis<3;++axis){d[t][axis].resize(extent*dstride);for(size_t k=0;k<d[t][axis].size();++k)d[t][axis][k]=awkward<double>(rng,k+axis);terms[t].derivative[0][axis]=d[t][axis].data();}
      v[t].resize(extent*vstride);for(size_t k=0;k<v[t].size();++k)v[t][k]=awkward<double>(rng,k+t);terms[t].value[0]=v[t].data();
      terms[t].derivative_rows=extent;terms[t].derivative_stride=dstride;terms[t].value_rows=extent;terms[t].value_stride=vstride;
      const int f=t?flags^0xA:flags;
      terms[t].sign=(f&1)?-1.0:1.0;terms[t].reverse_energy=(f>>1)&1;terms[t].transpose=(f>>2)&1;terms[t].value_first=(f>>3)&1;
    }
    OpenMXProjectorForceBlock b{};
    b.rows=size_t(rows);b.columns=size_t(columns);b.output_rows=size_t(rows)+1;b.output_stride=size_t(columns)+2;b.term_count=term_count;b.terms=terms;b.accumulate=accumulate;
    std::vector<dcomplex> seed(3*b.output_rows*b.output_stride);
    for(size_t k=0;k<seed.size();++k)seed[k]=dcomplex{awkward<double>(rng,k),awkward<double>(rng,k+1)};
    std::vector<dcomplex> expected[3],out[3];
    for(int axis=0;axis<3;++axis){expected[axis]=seed;b.output[axis]=expected[axis].data();}
    legacy_nl(sp,b);
    const size_t bytes=openmx_core_force_nl_scratch(columns,components,term_count);
    std::vector<unsigned char> scratch(bytes+1);
    for(int mode:ms) {
      for(int axis=0;axis<3;++axis){out[axis]=seed;b.output[axis]=out[axis].data();}
      require(openmx_core_force_nl_block(&sp,&b,scratch.data(),bytes,mode),"nl block refused a valid request");
      for(int axis=0;axis<3;++axis)if(std::memcmp(out[axis].data(),expected[axis].data(),out[axis].size()*sizeof(dcomplex))) {
        std::cerr<<"nl terms="<<term_count<<" rows="<<rows<<" columns="<<columns<<" flags="<<flags<<" accumulate="<<accumulate<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';
        throw std::runtime_error("nl block differs from the scalar Core block");
      }
    }
  }
  OpenMXForceProjectorSpecies rel=sp;rel.relativistic=1;
  OpenMXProjectorForceTerm term{};OpenMXProjectorForceBlock b{};b.rows=b.columns=0;b.term_count=1;b.terms=&term;
  require(!openmx_core_force_nl_block(&rel,&b,nullptr,0,OPENMX_GRID_SCALAR),"relativistic species accepted");
}
void invalid() {
  std::mt19937 rng(5u);
  Rows r(rng,2,3,1);
  OpenMXForceVnaBlock b{};b.rows=2;b.columns=2;b.projectors=3;b.terms=1;
  for(int c=0;c<3;++c)b.derivative[c]=r.get();b.value=r.get();
  std::vector<double> out[3];for(int c=0;c<3;++c){out[c].assign(10,7.0);b.output[c]=out[c].data();}b.output_stride=2;
  const size_t bytes=openmx_core_force_vna_scratch(2,3,1,0);std::vector<unsigned char> scratch(bytes);
  require(openmx_core_force_vna_block(&b,scratch.data(),bytes,OPENMX_GRID_SCALAR),"valid vna request refused");
  for(int c=0;c<3;++c)std::fill(out[c].begin(),out[c].end(),7.0);
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes,99),"unknown mode accepted");
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes,OPENMX_HOST_SIMD_LEGACY),"legacy mode accepted as a kernel");
  require(!openmx_core_force_vna_block(nullptr,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing block accepted");
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes-1,OPENMX_GRID_SCALAR),"short scratch accepted");
  require(!openmx_core_force_vna_block(&b,nullptr,bytes,OPENMX_GRID_SCALAR),"missing scratch accepted");
  auto broken=b;broken.terms=3;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"bad term count accepted");
  broken=b;broken.terms=2;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"term 2 without rows accepted");
  broken=b;broken.lanes_rows=2;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"bad lane side accepted");
  broken=b;broken.output_stride=1;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"short output stride accepted");
  broken=b;broken.value=nullptr;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing value rows accepted");
  broken=b;broken.output[1]=nullptr;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing output accepted");
  broken=b;broken.rows=-1;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"negative rows accepted");
  Lanes lanes;const float* const* rows1[4]={r.get(),nullptr,nullptr,nullptr};
  require(lanes.prepare(rows1,1,2,3,OPENMX_GRID_SCALAR),"lane preparation refused");
  broken=b;broken.lanes=&lanes.view;broken.rows=2;broken.columns=1;
  require(!openmx_core_force_vna_block(&broken,nullptr,0,OPENMX_GRID_SCALAR),"lanes of another extent accepted");
  OpenMXForceVnaLanes wrong=lanes.view;wrong.projectors=4;broken=b;broken.lanes=&wrong;
  require(!openmx_core_force_vna_block(&broken,nullptr,0,OPENMX_GRID_SCALAR),"lanes of another projector count accepted");
  require(!openmx_core_force_vna_lanes(rows1,5,2,3,lanes.memory.data(),lanes.memory.size(),&lanes.view,OPENMX_GRID_SCALAR),"five components accepted");
  require(!openmx_core_force_vna_lanes(rows1,1,2,3,lanes.memory.data(),3,&lanes.view,OPENMX_GRID_SCALAR),"short lane memory accepted");
  for(int c=0;c<3;++c)for(double v:out[c])require(v==7.0,"a refused vna call wrote output");
  require(openmx_core_force_vna_scratch(-1,3,1,0)==0 && openmx_core_force_vna_scratch(2,3,3,0)==0 && openmx_core_force_vna_scratch(2,3,1,2)==0 &&
      openmx_core_force_vna_scratch(2,3,2,0)==4*openmx_core_force_vna_scratch(2,3,1,0) && openmx_core_force_vna_scratch(2,3,1,1)==3*openmx_core_force_vna_scratch(2,3,1,0),"vna scratch sizes");
  OpenMXForceGridPair g{};g.rows=1;g.columns=1;g.spins=1;g.points=1;
  double sums[3]={1,2,3};const double dmrow[1]={1.0};const double* dmrows[1]={dmrow};g.density[0]=dmrows;
  const int center[1]={0};g.center=center;g.neighbor_rows=r.get();const double drow[3]={1,2,3};const double* drows[1]={drow};const double* const* dpoints[1]={drows};g.derivative=dpoints;
  const double pot[1]={2.0};const double* prow[1]={pot};g.potential=prow;
  const size_t gbytes=openmx_core_force_grid_scratch(1,1,1);std::vector<unsigned char> gscratch(gbytes);
  require(openmx_core_force_grid_pair(&g,sums,gscratch.data(),gbytes,OPENMX_GRID_SCALAR),"valid grid request refused");
  double keep[3]={1,2,3};
  require(!openmx_core_force_grid_pair(&g,keep,gscratch.data(),gbytes,99),"grid unknown mode accepted");
  require(!openmx_core_force_grid_pair(&g,nullptr,gscratch.data(),gbytes,OPENMX_GRID_SCALAR),"grid missing sums accepted");
  require(!openmx_core_force_grid_pair(&g,keep,gscratch.data(),gbytes-1,OPENMX_GRID_SCALAR),"grid short scratch accepted");
  auto gb=g;gb.spins=5;require(!openmx_core_force_grid_pair(&gb,keep,gscratch.data(),gbytes,OPENMX_GRID_SCALAR),"grid bad spins accepted");
  gb=g;gb.center=nullptr;require(!openmx_core_force_grid_pair(&gb,keep,gscratch.data(),gbytes,OPENMX_GRID_SCALAR),"grid missing centers accepted");
  gb=g;gb.potential=nullptr;require(!openmx_core_force_grid_pair(&gb,keep,gscratch.data(),gbytes,OPENMX_GRID_SCALAR),"grid missing potential accepted");
  require(keep[0]==1 && keep[1]==2 && keep[2]==3,"a refused grid call changed the sums");
  require(openmx_core_force_grid_scratch(1,1,0)==0 && openmx_core_force_grid_scratch(-1,1,1)==0,"grid scratch sizes");
  // point-major entries
  OpenMXForceGridPoints gp{};gp.rows=1;gp.spins=1;gp.points=1;gp.columns_max=1;gp.pad_rows=16;
  const size_t offsets[2]={0,0};gp.point_offsets=offsets;gp.potential=prow;double terms[3];gp.terms=terms;gp.derivative=drow;gp.point_stride=3;gp.row_stride=3;
  const size_t pbytes=openmx_core_force_grid_points_scratch(1);std::vector<unsigned char> pscratch(pbytes);
  require(openmx_core_force_grid_points(&gp,0,1,pscratch.data(),pbytes,OPENMX_GRID_SCALAR),"valid point request refused");
  require(!openmx_core_force_grid_points(&gp,0,2,pscratch.data(),pbytes,OPENMX_GRID_SCALAR),"point range beyond the atom accepted");
  auto pb=gp;pb.pad_rows=8;require(!openmx_core_force_grid_points(&pb,0,1,pscratch.data(),pbytes,OPENMX_GRID_SCALAR),"wrong row padding accepted");
  pb=gp;pb.point_stride=2;require(!openmx_core_force_grid_points(&pb,0,1,pscratch.data(),pbytes,OPENMX_GRID_SCALAR),"short point stride accepted");
  pb=gp;pb.derivative=nullptr;require(!openmx_core_force_grid_points(&pb,0,1,pscratch.data(),pbytes,OPENMX_GRID_SCALAR),"missing derivative accepted");
  require(openmx_core_force_grid_density_bytes(1,1,1,0)==0 && openmx_core_force_grid_density_bytes(13,13,1,2)==2*13*16*sizeof(double),"density plane sizes");
  require(openmx_core_force_nl_scratch(1,1,3)==0 && openmx_core_force_nl_scratch(-1,1,1)==0 && openmx_core_force_nl_scratch(3,2,1)==3*2*16*sizeof(double),"nl scratch sizes");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"vna";
    if(name=="vna")vna();else if(name=="tiled")tiled();else if(name=="grid")grid();else if(name=="points")points();
    else if(name=="nl")nl();else if(name=="invalid")invalid();
    else throw std::runtime_error("unknown host force case");
    std::cout<<"PASS host force "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
