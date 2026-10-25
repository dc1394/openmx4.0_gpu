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
// reference, every usable ISA is compared with memcmp.
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
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
  Rows(std::mt19937& rng,int count,int length,size_t salt) {
    storage.resize(count);
    for(int n=0;n<count;++n){storage[n].resize(length+3);for(size_t l=0;l<storage[n].size();++l)storage[n][l]=awkward<float>(rng,l+n+salt);pointers.push_back(storage[n].data());}
  }
  const float* const* get() const { return pointers.data(); }
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
    for(int c=0;c<3;++c)out[c][size_t(m)*b.output_stride+n]=sum[c];
  }
}
void vna() {
  std::mt19937 rng(11u);
  const auto ms=modes();
  for(int terms:{1,2})for(int fp32:{0,1})for(int rows:{1,3,8,13,17})for(int columns:{1,5,8,13,16,19})for(int P:{0,1,7,64,131}) {
    Rows dx(rng,rows,P,1),dy(rng,rows,P,2),dz(rng,rows,P,3),h0(rng,rows,P,4),q0(rng,columns,P,5),qx(rng,columns,P,6),qy(rng,columns,P,7),qz(rng,columns,P,8);
    OpenMXForceVnaBlock b{};
    b.rows=rows;b.columns=columns;b.projectors=P;b.terms=terms;b.fp32=fp32;
    b.derivative[0]=dx.get();b.derivative[1]=dy.get();b.derivative[2]=dz.get();b.row_value=h0.get();
    b.value=q0.get();b.column_derivative[0]=qx.get();b.column_derivative[1]=qy.get();b.column_derivative[2]=qz.get();
    b.output_stride=size_t(columns)+3;
    std::vector<double> expected[3],out[3];
    for(int c=0;c<3;++c){expected[c].assign(size_t(rows)*b.output_stride+5,-9.0);out[c]=expected[c];b.output[c]=expected[c].data();}
    legacy_vna(b,expected);
    const size_t bytes=openmx_core_force_vna_scratch(columns,P,terms);
    std::vector<unsigned char> scratch(bytes+1);
    for(int mode:ms) {
      for(int c=0;c<3;++c){std::fill(out[c].begin(),out[c].end(),-9.0);b.output[c]=out[c].data();}
      require(openmx_core_force_vna_block(&b,scratch.data(),bytes,mode),"vna block refused a valid request");
      for(int c=0;c<3;++c)if(std::memcmp(out[c].data(),expected[c].data(),out[c].size()*sizeof(double))) {
        std::cerr<<"vna terms="<<terms<<" fp32="<<fp32<<" rows="<<rows<<" columns="<<columns<<" P="<<P<<" mode="<<openmx_core_host_simd_name(mode)<<'\n';
        throw std::runtime_error("vna block differs from the legacy loop");
      }
    }
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
    // Operand matrices large enough for either role (rows or columns).
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
  // A relativistic species keeps the Core's general block.
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
  const size_t bytes=openmx_core_force_vna_scratch(2,3,1);std::vector<unsigned char> scratch(bytes);
  require(openmx_core_force_vna_block(&b,scratch.data(),bytes,OPENMX_GRID_SCALAR),"valid vna request refused");
  for(int c=0;c<3;++c)std::fill(out[c].begin(),out[c].end(),7.0);
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes,99),"unknown mode accepted");
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes,OPENMX_HOST_SIMD_LEGACY),"legacy mode accepted as a kernel");
  require(!openmx_core_force_vna_block(nullptr,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing block accepted");
  require(!openmx_core_force_vna_block(&b,scratch.data(),bytes-1,OPENMX_GRID_SCALAR),"short scratch accepted");
  require(!openmx_core_force_vna_block(&b,nullptr,bytes,OPENMX_GRID_SCALAR),"missing scratch accepted");
  auto broken=b;broken.terms=3;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"bad term count accepted");
  broken=b;broken.terms=2;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"term 2 without rows accepted");
  broken=b;broken.output_stride=1;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"short output stride accepted");
  broken=b;broken.value=nullptr;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing value rows accepted");
  broken=b;broken.output[1]=nullptr;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"missing output accepted");
  broken=b;broken.rows=-1;require(!openmx_core_force_vna_block(&broken,scratch.data(),bytes,OPENMX_GRID_SCALAR),"negative rows accepted");
  for(int c=0;c<3;++c)for(double v:out[c])require(v==7.0,"a refused vna call wrote output");
  require(openmx_core_force_vna_scratch(-1,3,1)==0 && openmx_core_force_vna_scratch(2,3,3)==0 && openmx_core_force_vna_scratch(2,3,2)==4*openmx_core_force_vna_scratch(2,3,1),"vna scratch sizes");
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
  require(openmx_core_force_nl_scratch(1,1,3)==0 && openmx_core_force_nl_scratch(-1,1,1)==0 && openmx_core_force_nl_scratch(3,2,1)==3*2*16*sizeof(double),"nl scratch sizes");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"vna";
    if(name=="vna")vna();else if(name=="grid")grid();else if(name=="nl")nl();else if(name=="invalid")invalid();
    else throw std::runtime_error("unknown host force case");
    std::cout<<"PASS host force "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
