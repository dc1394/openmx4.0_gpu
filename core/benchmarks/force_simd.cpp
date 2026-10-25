#include "openmx_core_force_simd.h"
#include "openmx_core_host_simd.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// Force-family kernels at the sizes of a Si s2p2d1 run (13 orbitals, 972 VNA
// projector components = (2+6+1)^2 * 12, 85 neighbours, thousands of
// overlap-grid points per pair). The atom-scale sections hold the blocks of a
// whole atom (working set far beyond L2) and compare the per-block order of
// the legacy loops with the cache-reusing orders of the Force.c glue.
namespace {
struct Rows {
  std::vector<float> storage;std::vector<const float*> pointers;
  Rows()=default;
  Rows(int count,int length,double seed) {
    storage.resize(size_t(count)*length);
    for(size_t k=0;k<storage.size();++k)storage[k]=float(std::sin(seed+0.37*double(k)));
    for(int n=0;n<count;++n)pointers.push_back(storage.data()+size_t(n)*length);
  }
};
double seconds(auto&& body,int repeats) {
  const auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<repeats;++r)body();
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}
void report(const char* what,double legacy,double s,double work) {
  std::cout<<std::setw(28)<<what<<std::setw(10)<<std::fixed<<std::setprecision(3)<<1e9*s/work<<" ns/elem"
      <<std::setw(8)<<std::setprecision(2)<<legacy/s<<"x\n";
}
void trace(const double* const h[3],size_t stride,bool transposed,int rows,int columns,double pref,const double* cdm,double f[3]) {
  for(int m=0;m<rows;++m)for(int n=0;n<columns;++n) {
    const size_t at=transposed?size_t(n)*stride+m:size_t(m)*stride+n;
    const double scale=pref*cdm[size_t(m)*columns+n];
    f[0]+=scale*h[0][at];f[1]+=scale*h[1][at];f[2]+=scale*h[2][at];
  }
}
void check(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
}
int main(int argc,char** argv) {
  try {
    const int repeats=argc>1?std::atoi(argv[1]):20;
    if(repeats<1)throw std::invalid_argument("positive repetition count required");
    const int orbitals=13,P=972,points=20000;
    std::vector<int> isas;
    for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode)if(openmx_core_host_simd_usable(mode))isas.push_back(mode);
    // ---------------- single blocks ----------------
    Rows dx(orbitals,P,1),dy(orbitals,P,2),dz(orbitals,P,3),h0(orbitals,P,4),q0(orbitals,P,5),qx(orbitals,P,6),qy(orbitals,P,7),qz(orbitals,P,8);
    std::vector<double> out[3];for(auto& o:out)o.assign(size_t(orbitals)*orbitals,0.0);
    for(int terms:{1,2})for(int lanes_rows:{0,1}) {
      OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=terms;b.fp32=1;b.lanes_rows=lanes_rows;
      b.derivative[0]=dx.pointers.data();b.derivative[1]=dy.pointers.data();b.derivative[2]=dz.pointers.data();b.row_value=h0.pointers.data();
      b.value=q0.pointers.data();b.column_derivative[0]=qx.pointers.data();b.column_derivative[1]=qy.pointers.data();b.column_derivative[2]=qz.pointers.data();
      for(int c=0;c<3;++c)b.output[c]=out[c].data();b.output_stride=orbitals;
      const double work=double(repeats)*3*orbitals*orbitals*P*terms;
      const double legacy=seconds([&]{
        for(int m=0;m<orbitals;++m)for(int n=0;n<orbitals;++n) {
          double sx=0,sy=0,sz=0;const float* hx=dx.pointers[m];const float* hy=dy.pointers[m];const float* hz=dz.pointers[m];const float* q=q0.pointers[n];
          if(terms==1)for(int l=0;l<P;++l){sx+=(double)(hx[l]*q[l]);sy+=(double)(hy[l]*q[l]);sz+=(double)(hz[l]*q[l]);}
          else{const float* h=h0.pointers[m];const float* vx=qx.pointers[n];const float* vy=qy.pointers[n];const float* vz=qz.pointers[n];
            for(int l=0;l<P;++l){sx-=(double)(hx[l]*q[l])+(double)(h[l]*vx[l]);sy-=(double)(hy[l]*q[l])+(double)(h[l]*vy[l]);sz-=(double)(hz[l]*q[l])+(double)(h[l]*vz[l]);}}
          out[0][size_t(m)*orbitals+n]=sx;out[1][size_t(m)*orbitals+n]=sy;out[2][size_t(m)*orbitals+n]=sz;
        }},repeats);
      std::cout<<"VNA block terms="<<terms<<" lanes_rows="<<lanes_rows<<" ("<<orbitals<<"x"<<orbitals<<", P="<<P<<", transposing per call)\n";
      report("legacy loop",legacy,legacy,work);
      const size_t bytes=openmx_core_force_vna_scratch(orbitals,P,terms,lanes_rows);std::vector<unsigned char> scratch(bytes);
      for(int mode:isas) {
        const double s=seconds([&]{if(!openmx_core_force_vna_block(&b,scratch.data(),bytes,mode))throw std::runtime_error("vna block failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
    }
    // ---------------- Force4B case 1, one atom ----------------
    {
      const int fan=84;const double density=0.4;
      std::mt19937 rng(3u);std::uniform_real_distribution<double> u(0.0,1.0);
      std::vector<std::vector<Rows>> left(fan+1);
      for(int k=0;k<=fan;++k)for(int c=0;c<3;++c)left[k].emplace_back(orbitals,P,double(k*3+c));
      std::vector<std::vector<int>> valid(fan+1,std::vector<int>(fan+1,0));
      std::vector<std::vector<Rows>> right(fan+1);size_t blocks=0;
      for(int q=1;q<=fan;++q){right[q].resize(fan+1);for(int k=0;k<=fan;++k)if(u(rng)<density){valid[q][k]=1;right[q][k]=Rows(orbitals,P,double(1000+q*97+k));++blocks;}}
      std::vector<double> cdm(size_t(fan+1)*orbitals*orbitals);for(size_t t=0;t<cdm.size();++t)cdm[t]=std::cos(0.01*double(t));
      const double block_kb=double(orbitals)*P*4/1024.0;
      std::cout<<"Force4B case 1, one atom: "<<fan<<" neighbours, "<<blocks<<" (q,k) blocks, block "<<std::setprecision(1)<<block_kb<<" KB; left blocks "
          <<std::setprecision(1)<<3*block_kb*(fan+1)/1024<<" MB, right blocks "<<blocks*block_kb/1024<<" MB\n";
      const int atom_repeats=std::max(1,repeats/10);
      std::vector<double> force((fan+1)*3),reference((fan+1)*3);
      std::vector<double> h[3];for(auto& v:h)v.assign(size_t(orbitals)*(orbitals+1),0.0);
      const double* hp[3]={h[0].data(),h[1].data(),h[2].data()};
      for(int mode:isas) {
        // (a) q-outer, k-inner, right block transposed per (q,k): the round-1 order
        const size_t bytes=openmx_core_force_vna_scratch(orbitals,P,1,0);std::vector<unsigned char> scratch(bytes);
        const double untiled=seconds([&]{
          std::fill(force.begin(),force.end(),0.0);
          for(int q=1;q<=fan;++q)for(int k=0;k<=fan;++k) {
            if(!valid[q][k])continue;
            OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=1;b.fp32=1;b.lanes_rows=0;
            for(int c=0;c<3;++c){b.derivative[c]=left[k][c].pointers.data();b.output[c]=h[c].data();}
            b.value=right[q][k].pointers.data();b.output_stride=orbitals;
            check(openmx_core_force_vna_block(&b,scratch.data(),bytes,mode),"untiled block failed");
            trace(hp,orbitals,false,orbitals,orbitals,4.0,cdm.data()+size_t(q)*orbitals*orbitals,force.data()+q*3);
          }},atom_repeats);
        if(mode==isas.front())reference=force;
        check(force==reference,"untiled case-1 result differs between ISAs");
        // (b) k-outer, q-inner with the left block prepared once per k
        const size_t lane_bytes=openmx_core_force_vna_lanes_bytes(orbitals,P,3);std::vector<unsigned char> lane_memory(lane_bytes);
        const double tiled=seconds([&]{
          std::fill(force.begin(),force.end(),0.0);
          for(int k=0;k<=fan;++k) {
            bool any=false;for(int q=1;q<=fan && !any;++q)any=valid[q][k];
            if(!any)continue;
            OpenMXForceVnaLanes lanes;const float* const* rows3[4]={left[k][0].pointers.data(),left[k][1].pointers.data(),left[k][2].pointers.data(),nullptr};
            check(openmx_core_force_vna_lanes(rows3,3,orbitals,P,lane_memory.data(),lane_bytes,&lanes,mode),"lanes failed");
            for(int q=1;q<=fan;++q) {
              if(!valid[q][k])continue;
              OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=1;b.fp32=1;b.lanes_rows=1;b.lanes=&lanes;
              for(int c=0;c<3;++c){b.output[c]=h[c].data();b.derivative[c]=left[k][c].pointers.data();}
              b.value=right[q][k].pointers.data();b.output_stride=orbitals+1;
              check(openmx_core_force_vna_block(&b,nullptr,0,mode),"tiled block failed");
              trace(hp,orbitals+1,true,orbitals,orbitals,4.0,cdm.data()+size_t(q)*orbitals*orbitals,force.data()+q*3);
            }
          }},atom_repeats);
        check(force==reference,"k-outer case-1 result differs from the untiled order");
        const double trace_only=seconds([&]{std::fill(force.begin(),force.end(),0.0);
          for(int q=1;q<=fan;++q)for(int k=0;k<=fan;++k)if(valid[q][k])trace(hp,orbitals,false,orbitals,orbitals,4.0,cdm.data()+size_t(q)*orbitals*orbitals,force.data()+q*3);},atom_repeats);
        const double untiled_mb=blocks*(4*block_kb)/1024,tiled_mb=(3*block_kb*(fan+1)+blocks*block_kb)/1024;
        std::cout<<std::setw(10)<<openmx_core_host_simd_name(mode)<<std::setprecision(1)<<"  q-outer: "<<1e3*untiled/atom_repeats<<" ms/atom ("<<untiled_mb<<" MB read)"
            <<"  k-outer: "<<1e3*tiled/atom_repeats<<" ms/atom ("<<tiled_mb<<" MB read)  "<<std::setprecision(2)<<untiled/tiled<<"x; trace alone "
            <<std::setprecision(1)<<1e3*trace_only/atom_repeats<<" ms/atom\n";
      }
    }
    // ---------------- Force4B case 2, one atom ----------------
    {
      const int fan=84;const double density=0.4;
      std::mt19937 rng(5u);std::uniform_real_distribution<double> u(0.0,1.0);
      std::vector<std::vector<Rows>> blocks(fan+1);
      for(int hh=0;hh<=fan;++hh)for(int c=0;c<4;++c)blocks[hh].emplace_back(orbitals,P,double(2000+hh*4+c));
      struct Pair { int h,q; };std::vector<Pair> pairs;
      for(int hh=1;hh<=fan;++hh)for(int q=hh+1;q<=fan;++q)if(u(rng)<density)pairs.push_back({hh,q});
      std::vector<double> cdm(pairs.size()*orbitals*orbitals);for(size_t t=0;t<cdm.size();++t)cdm[t]=std::cos(0.013*double(t));
      const double block_kb=double(orbitals)*P*4/1024.0;
      std::cout<<"Force4B case 2, one atom: "<<fan<<" neighbours, "<<pairs.size()<<" pairs, 4 blocks per neighbour ("<<std::setprecision(1)<<4*block_kb*(fan+1)/1024<<" MB)\n";
      const int atom_repeats=std::max(1,repeats/10);
      std::vector<double> h[3];for(auto& v:h)v.assign(size_t(orbitals)*(orbitals+1),0.0);
      const double* hp[3]={h[0].data(),h[1].data(),h[2].data()};
      double reference[3]={0,0,0};
      for(int mode:isas) {
        double de[3];
        const size_t bytes=openmx_core_force_vna_scratch(orbitals,P,2,0);std::vector<unsigned char> scratch(bytes);
        const double untiled=seconds([&]{
          de[0]=de[1]=de[2]=0.0;
          for(size_t i=0;i<pairs.size();++i) {
            const auto& p=pairs[i];
            OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=2;b.fp32=1;b.lanes_rows=0;
            for(int c=0;c<3;++c){b.derivative[c]=blocks[p.h][c+1].pointers.data();b.column_derivative[c]=blocks[p.q][c+1].pointers.data();b.output[c]=h[c].data();}
            b.row_value=blocks[p.h][0].pointers.data();b.value=blocks[p.q][0].pointers.data();b.output_stride=orbitals;
            check(openmx_core_force_vna_block(&b,scratch.data(),bytes,mode),"untiled case-2 block failed");
            trace(hp,orbitals,false,orbitals,orbitals,2.0,cdm.data()+i*orbitals*orbitals,de);
          }},atom_repeats);
        if(mode==isas.front()){reference[0]=de[0];reference[1]=de[1];reference[2]=de[2];}
        check(de[0]==reference[0] && de[1]==reference[1] && de[2]==reference[2],"untiled case-2 result differs between ISAs");
        for(int tile:{2,3,4}) {
          const size_t lane_bytes=openmx_core_force_vna_lanes_bytes(orbitals,P,4);std::vector<unsigned char> lane_memory(lane_bytes*tile);
          std::vector<double> slots(size_t(tile)*(fan+1)*3*orbitals*(orbitals+1));
          const double tiled=seconds([&]{
            de[0]=de[1]=de[2]=0.0;
            size_t begin=0;
            while(begin<pairs.size()) {
              size_t end=begin;int distinct=0,last=-1;
              while(end<pairs.size() && (pairs[end].h==last || distinct<tile)){if(pairs[end].h!=last){++distinct;last=pairs[end].h;}++end;}
              std::vector<int> hs;std::vector<OpenMXForceVnaLanes> lanes;
              for(size_t i=begin;i<end;++i)if(hs.empty() || hs.back()!=pairs[i].h) {
                hs.push_back(pairs[i].h);OpenMXForceVnaLanes l;
                const float* const* rows4[4]={blocks[pairs[i].h][1].pointers.data(),blocks[pairs[i].h][2].pointers.data(),blocks[pairs[i].h][3].pointers.data(),blocks[pairs[i].h][0].pointers.data()};
                check(openmx_core_force_vna_lanes(rows4,4,orbitals,P,lane_memory.data()+(hs.size()-1)*lane_bytes,lane_bytes,&l,mode),"h lanes failed");
                lanes.push_back(l);
              }
              const size_t plane=size_t(orbitals)*(orbitals+1);
              for(int q=0;q<=fan;++q)for(size_t i=begin;i<end;++i) {
                const auto& p=pairs[i];if(p.q!=q)continue;
                int hi=0;while(hs[hi]!=p.h)++hi;
                double* slot=slots.data()+(i-begin)*3*plane;
                OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=2;b.fp32=1;b.lanes_rows=1;b.lanes=&lanes[hi];
                b.value=blocks[p.q][0].pointers.data();b.row_value=blocks[p.h][0].pointers.data();
                for(int c=0;c<3;++c){b.column_derivative[c]=blocks[p.q][c+1].pointers.data();b.derivative[c]=blocks[p.h][c+1].pointers.data();b.output[c]=slot+size_t(c)*plane;}
                b.output_stride=orbitals+1;
                check(openmx_core_force_vna_block(&b,nullptr,0,mode),"tiled case-2 block failed");
              }
              for(size_t i=begin;i<end;++i) {
                const double* slot=slots.data()+(i-begin)*3*plane;const double* sp[3]={slot,slot+plane,slot+2*plane};
                trace(sp,orbitals+1,true,orbitals,orbitals,2.0,cdm.data()+i*orbitals*orbitals,de);
              }
              begin=end;
            }},atom_repeats);
          check(de[0]==reference[0] && de[1]==reference[1] && de[2]==reference[2],"h-tiled case-2 result differs from the pair order");
          const double untiled_mb=pairs.size()*8*block_kb/1024;
          double tiled_mb=0;{size_t begin=0;while(begin<pairs.size()){size_t end=begin;int distinct=0,last=-1;while(end<pairs.size() && (pairs[end].h==last || distinct<tile)){if(pairs[end].h!=last){++distinct;last=pairs[end].h;}++end;}
              tiled_mb+=distinct*4*block_kb/1024;std::vector<int> qs;for(size_t i=begin;i<end;++i)qs.push_back(pairs[i].q);std::sort(qs.begin(),qs.end());tiled_mb+=double(std::unique(qs.begin(),qs.end())-qs.begin())*4*block_kb/1024;begin=end;}}
          std::cout<<std::setw(10)<<openmx_core_host_simd_name(mode)<<std::setprecision(1)<<"  pair order: "<<1e3*untiled/atom_repeats<<" ms/atom ("<<untiled_mb<<" MB read)"
              <<"  h-tile "<<tile<<": "<<1e3*tiled/atom_repeats<<" ms/atom ("<<tiled_mb<<" MB read)  "<<std::setprecision(2)<<untiled/tiled<<"x\n";
        }
      }
    }
    // ---------------- Force3 pair contraction ----------------
    {
      std::vector<double> dm(size_t(orbitals)*orbitals);for(size_t k=0;k<dm.size();++k)dm[k]=std::cos(0.11*double(k));
      std::vector<const double*> dmrows;for(int i=0;i<orbitals;++i)dmrows.push_back(dm.data()+size_t(i)*orbitals);
      Rows orbs(points,orbitals,9);
      std::vector<double> dstore(size_t(points)*orbitals*3);for(size_t k=0;k<dstore.size();++k)dstore[k]=std::sin(0.013*double(k));
      std::vector<const double*> drows(size_t(points)*orbitals);std::vector<const double* const*> dpoints(points);
      for(int nc=0;nc<points;++nc){for(int i=0;i<orbitals;++i)drows[size_t(nc)*orbitals+i]=dstore.data()+(size_t(nc)*orbitals+i)*3;dpoints[nc]=drows.data()+size_t(nc)*orbitals;}
      std::vector<double> potential(points);for(int p=0;p<points;++p)potential[p]=std::cos(0.021*p);
      const double* prow[1]={potential.data()};
      std::vector<int> center(points);for(int p=0;p<points;++p)center[p]=p;
      OpenMXForceGridPair g{};g.rows=g.columns=orbitals;g.spins=1;g.points=points;g.density[0]=dmrows.data();g.center=center.data();g.neighbor_rows=orbs.pointers.data();g.derivative=dpoints.data();g.potential=prow;
      const double work=double(repeats)*points*orbitals*orbitals;
      double sums[3];
      const double legacy=seconds([&]{sums[0]=sums[1]=sums[2]=0.0;std::vector<double> orbs1(orbitals);
        for(int p=0;p<points;++p){const float* src=orbs.pointers[p];for(int j=0;j<orbitals;++j)orbs1[j]=(double)src[j];
          double tx=0,ty=0,tz=0;for(int i=0;i<orbitals;++i){double t0=0;const double* row=dmrows[i];for(int j=0;j<orbitals;++j)t0+=orbs1[j]*row[j];
            tx+=dpoints[p][i][0]*t0;ty+=dpoints[p][i][1]*t0;tz+=dpoints[p][i][2]*t0;}
          sums[0]+=tx*potential[p];sums[1]+=ty*potential[p];sums[2]+=tz*potential[p];}},repeats);
      std::cout<<"Force3 pair ("<<points<<" points, "<<orbitals<<"x"<<orbitals<<")\n";
      report("legacy loop",legacy,legacy,work);
      const size_t bytes=openmx_core_force_grid_scratch(orbitals,orbitals,1);std::vector<unsigned char> scratch(bytes);
      for(int mode:isas) {
        const double s=seconds([&]{sums[0]=sums[1]=sums[2]=0.0;if(!openmx_core_force_grid_pair(&g,sums,scratch.data(),bytes,mode))throw std::runtime_error("grid pair failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
      // point-major: one visit per point here, flat derivatives
      std::vector<size_t> offsets(points+1);for(int p=0;p<=points;++p)offsets[p]=size_t(p);
      std::vector<int> vpair(points,0),vrow(points),vterm(points);for(int p=0;p<points;++p){vrow[p]=p;vterm[p]=p;}
      const int pair_columns[1]={orbitals};const float* const* pair_rows[1]={orbs.pointers.data()};
      std::vector<double> terms(size_t(points)*3);
      const size_t dm_bytes=openmx_core_force_grid_density_bytes(orbitals,orbitals,1,1);std::vector<double> planes(dm_bytes/sizeof(double));
      const double* const* density[4]={dmrows.data(),nullptr,nullptr,nullptr};
      OpenMXForceGridPoints gp{};gp.rows=orbitals;gp.spins=1;gp.points=points;gp.columns_max=orbitals;gp.pad_rows=16;gp.dm=planes.data();gp.pair_columns=pair_columns;
      gp.point_offsets=offsets.data();gp.visit_pair=vpair.data();gp.visit_row=vrow.data();gp.visit_term=vterm.data();gp.pair_rows=pair_rows;gp.derivative=dstore.data();
      gp.point_stride=size_t(orbitals)*3;gp.row_stride=3;gp.potential=prow;gp.terms=terms.data();
      const size_t pbytes=openmx_core_force_grid_points_scratch(orbitals);std::vector<unsigned char> pscratch(pbytes);
      for(int mode:isas) {
        const double s=seconds([&]{check(openmx_core_force_grid_density(density,orbitals,orbitals,orbitals,1,0,planes.data(),dm_bytes,mode),"density planes failed");
          check(openmx_core_force_grid_points(&gp,0,points,pscratch.data(),pbytes,mode),"grid points failed");
          double acc[3]={0,0,0};for(int p=0;p<points;++p)for(int a=0;a<3;++a)acc[a]+=terms[size_t(p)*3+a];sums[0]=acc[0];sums[1]=acc[1];sums[2]=acc[2];},repeats);
        std::string name=std::string(openmx_core_host_simd_name(mode))+" point-major";
        report(name.c_str(),legacy,s,work);
      }
    }
    // ---------------- Nonlocal projector block ----------------
    {
      const int twice[]={0,0,2,2,4,4};std::vector<OpenMXForceProjectorChannel> channels;size_t components=0;
      for(int t:twice){OpenMXForceProjectorChannel c{};c.twice_angular=t;c.offset=components;c.energy[0]=1.0+0.1*double(components);channels.push_back(c);components+=size_t(t)+1;}
      OpenMXForceProjectorSpecies sp{};sp.count=channels.size();sp.components=components;sp.channels=channels.data();
      const size_t stride=components;std::vector<double> d[3],v;
      for(auto& a:d){a.resize(size_t(orbitals)*stride);for(size_t k=0;k<a.size();++k)a[k]=std::sin(0.3*double(k));}
      v.resize(size_t(orbitals)*stride);for(size_t k=0;k<v.size();++k)v[k]=std::cos(0.2*double(k));
      OpenMXProjectorForceTerm term{};for(int axis=0;axis<3;++axis)term.derivative[0][axis]=d[axis].data();term.value[0]=v.data();
      term.derivative_rows=term.value_rows=orbitals;term.derivative_stride=term.value_stride=stride;term.sign=1.0;
      std::vector<dcomplex> outc[3];for(auto& o:outc)o.assign(3*size_t(orbitals)*orbitals,dcomplex{0,0});
      OpenMXProjectorForceBlock b{};b.rows=b.columns=b.output_rows=b.output_stride=orbitals;b.term_count=1;b.terms=&term;b.accumulate=1;
      for(int axis=0;axis<3;++axis)b.output[axis]=outc[axis].data();
      const double work=double(repeats)*3*orbitals*orbitals*components;
      const double legacy=seconds([&]{for(int m=0;m<orbitals;++m)for(int n=0;n<orbitals;++n){double sum[3]={0,0,0};
          for(size_t j=0;j<sp.count;++j){const auto& c=sp.channels[j];for(int mg=0;mg<=c.twice_angular;++mg){const size_t p=c.offset+mg;
            for(int axis=0;axis<3;++axis)sum[axis]+=c.energy[0]*d[axis][size_t(m)*stride+p]*v[size_t(n)*stride+p];}}
          for(int axis=0;axis<3;++axis)for(int spin=0;spin<2;++spin)outc[axis][(size_t(spin)*orbitals+m)*orbitals+n].r+=sum[axis];}},repeats);
      std::cout<<"Nonlocal block ("<<orbitals<<"x"<<orbitals<<", "<<components<<" components)\n";
      report("legacy loop",legacy,legacy,work);
      const size_t bytes=openmx_core_force_nl_scratch(orbitals,components,1);std::vector<unsigned char> scratch(bytes);
      for(int mode:isas) {
        const double s=seconds([&]{if(!openmx_core_force_nl_block(&sp,&b,scratch.data(),bytes,mode))throw std::runtime_error("nl block failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
