#include "openmx_core_force_simd.h"
#include "openmx_core_host_simd.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

// Force-family kernels at the sizes of a Si s2p2d1 run (13 orbitals, 972 VNA
// projector components = (2+6+1)^2 * 12, thousands of overlap-grid points per
// pair) against transcriptions of the legacy loops.
namespace {
struct Rows {
  std::vector<float> storage;std::vector<const float*> pointers;
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
}
int main(int argc,char** argv) {
  try {
    const int repeats=argc>1?std::atoi(argv[1]):20;
    if(repeats<1)throw std::invalid_argument("positive repetition count required");
    const int orbitals=13,P=972,points=20000;
    // Force4B contraction, terms 1 (case 1) and 2 (case 2), fp32 on.
    Rows dx(orbitals,P,1),dy(orbitals,P,2),dz(orbitals,P,3),h0(orbitals,P,4),q0(orbitals,P,5),qx(orbitals,P,6),qy(orbitals,P,7),qz(orbitals,P,8);
    std::vector<double> out[3];for(auto& o:out)o.assign(size_t(orbitals)*orbitals,0.0);
    for(int terms:{1,2}) {
      OpenMXForceVnaBlock b{};b.rows=b.columns=orbitals;b.projectors=P;b.terms=terms;b.fp32=1;
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
      std::cout<<"VNA block terms="<<terms<<" ("<<orbitals<<"x"<<orbitals<<", P="<<P<<")\n";
      report("legacy loop",legacy,legacy,work);
      const size_t bytes=openmx_core_force_vna_scratch(orbitals,P,terms);std::vector<unsigned char> scratch(bytes);
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds([&]{if(!openmx_core_force_vna_block(&b,scratch.data(),bytes,mode))throw std::runtime_error("vna block failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
    }
    // Force3 pair contraction.
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
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds([&]{sums[0]=sums[1]=sums[2]=0.0;if(!openmx_core_force_grid_pair(&g,sums,scratch.data(),bytes,mode))throw std::runtime_error("grid pair failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
    }
    // Nonlocal projector block, one term, 18 components (s,p,d x 2 radial).
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
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds([&]{if(!openmx_core_force_nl_block(&sp,&b,scratch.data(),bytes,mode))throw std::runtime_error("nl block failed");},repeats);
        report(openmx_core_host_simd_name(mode),legacy,s,work);
      }
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
