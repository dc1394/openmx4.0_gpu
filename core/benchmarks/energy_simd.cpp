#include "openmx_core_energy_simd.h"
#include "openmx_core_host_simd.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// Energy family at application sizes: a neutral-atom pair integral of a few
// thousand grid points on a VPS mesh of a few hundred radii (with the force
// terms), and a 13 x 13 band density-matrix block over a few hundred bands.
namespace {
double VH_AtomF(double core_charge,int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  const double xmin=xv[0],xmax=xv[N-1];
  if(xmax<=x)return core_charge/r;
  if(r<rv[0]) {
    const int m=4;const double rm=rv[m],h1=rv[m-1]-rv[m-2],h2=rv[m]-rv[m-1],h3=rv[m+1]-rv[m];
    const double f1=yv[m-2],f2=yv[m-1],f3=yv[m],f4=yv[m+1];
    const double g1=((f3-f2)*h1/h2+(f2-f1)*h2/h1)/(h1+h2),g2=((f4-f3)*h2/h3+(f3-f2)*h3/h2)/(h2+h3);
    const double y1=(rm-rv[m-1])/h2,y2=(rm-rv[m])/h2,y12=y1*y1,y22=y2*y2;
    const double f=y22*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y12*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1);
    const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y22*(2.0*f2+h2*g1)/h2+2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)-y12*(2.0*f3-h2*g2)/h2;
    const double a=0.5*df/rm;return a*r*r+(f-a*rm*rm);
  }
  x=x>xmin?x:xmin;const double t=((double)N-1.0)*(x-xmin)/(xmax-xmin);const int i=std::floor(t);const double dt=t-(double)i;
  return 0.5*(((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt+(yv[i+2]-yv[i]))*dt+yv[i+1];
}
double Dr_VH_AtomF(double core_charge,int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  const double xmin=xv[0],xmax=xv[N-1];
  if(xmax<=x)return -core_charge/r/r;
  if(r<rv[0]) {
    const int m=4;const double rm=rv[m],h1=rv[m-1]-rv[m-2],h2=rv[m]-rv[m-1],h3=rv[m+1]-rv[m];
    const double f1=yv[m-2],f2=yv[m-1],f3=yv[m],f4=yv[m+1];
    const double g1=((f3-f2)*h1/h2+(f2-f1)*h2/h1)/(h1+h2),g2=((f4-f3)*h2/h3+(f3-f2)*h3/h2)/(h2+h3);
    const double y1=(rm-rv[m-1])/h2,y2=(rm-rv[m])/h2,y12=y1*y1,y22=y2*y2;
    const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y22*(2.0*f2+h2*g1)/h2+2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)-y12*(2.0*f3-h2*g2)/h2;
    return 2.0*(0.5*df/rm)*r;
  }
  x=x>xmin?x:xmin;const double tmp=((double)N-1.0)/(xmax-xmin),t=(x-xmin)*tmp;const int i=std::floor(t);const double dt=t-(double)i;
  return 0.5*((3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt+2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt+(yv[i+2]-yv[i]))*tmp/r;
}
template<class F> double seconds(int repeats,F&& body) {
  const auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<repeats;++r)body();
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}
}
int main(int argc,char** argv) {
  try {
    const int points=argc>1?std::atoi(argv[1]):4000,repeats=argc>2?std::atoi(argv[2]):200;
    const int bands=argc>3?std::atoi(argv[3]):440,blocks=argc>4?std::atoi(argv[4]):2000;
    if(points<1 || repeats<1 || bands<1 || blocks<1)throw std::invalid_argument("positive sizes required");
    std::mt19937 rng(9u);std::uniform_real_distribution<double> u(0.0,1.0);
    // neutral-atom pair integral
    const int N=240;const double charge=4.0,xmin=std::log(1e-5),xmax=std::log(10.0);
    std::vector<double> xv(N),rv(N),yv(N+2);
    for(int i=0;i<N;++i){xv[i]=xmin+(xmax-xmin)*i/(N-1.0);rv[i]=std::exp(xv[i]);}
    for(int i=0;i<N+2;++i){const double r=i<N?rv[i]:rv[N-1]*(1.0+0.1*(i-N+1));yv[i]=charge/r*(1.0-std::exp(-r));}
    std::vector<double> x(points),y(points),z(points),rho(points),wt(points),values(points),derivatives(points);
    for(int n=0;n<points;++n){x[n]=7.0*(u(rng)-0.5);y[n]=0.0;z[n]=7.0*(u(rng)-0.5);rho[n]=std::exp(-std::sqrt(x[n]*x[n]+z[n]*z[n]));wt[n]=u(rng);}
    const double separation=2.35;
    double legacy_sum=0,legacy_sumr=0;
    const double legacy_seconds=seconds(repeats,[&]{
      double sum=0.0,sumr=0.0;
      for(int n=0;n<points;++n){const double z2=z[n]-separation,r2=x[n]*x[n]+y[n]*y[n]+z2*z2;double r=std::sqrt(r2);const double xx=0.5*std::log(r2);
        if(r<1.0e-10)r=1.0e-10;const double va0=VH_AtomF(charge,N,xx,r,xv.data(),rv.data(),yv.data());sum+=wt[n]*va0*rho[n];
        if(1.0e-14<r){const double dr=Dr_VH_AtomF(charge,N,xx,r,xv.data(),rv.data(),yv.data());sumr-=wt[n]*rho[n]*dr*z2/r;}}
      legacy_sum=sum;legacy_sumr=sumr;});
    std::cout<<std::fixed<<std::setprecision(2)<<"neutral pair: points="<<points<<" mesh="<<N<<" repeats="<<repeats<<'\n'
        <<std::setw(24)<<"legacy loop"<<std::setw(10)<<1e9*legacy_seconds/(double(points)*repeats)<<" ns/point\n";
    for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      double sum=0,sumr=0;
      OpenMXEnergyNeutralPoints s{points,1,1,x.data(),y.data(),z.data(),rho.data(),wt.data(),separation,charge,N,xv.data(),rv.data(),yv.data(),values.data(),derivatives.data()};
      const double t=seconds(repeats,[&]{if(!openmx_core_energy_neutral_points(&s,mode))throw std::runtime_error("neutral points failed");
        sum=0.0;sumr=0.0;for(int n=0;n<points;++n)sum+=values[n];for(int n=0;n<points;++n)sumr-=derivatives[n];});
      if(sum!=legacy_sum || sumr!=legacy_sumr)throw std::runtime_error("host kernel differs from the legacy loop");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*t/(double(points)*repeats)<<" ns/point  ("<<legacy_seconds/t<<"x)\n";
    }
    // band density-matrix block
    const int rows=13,columns=13;const size_t stride=16;
    std::vector<double> re0(size_t(rows)*bands),im0(re0.size()),re1(size_t(columns)*bands),im1(re1.size()),eigen(bands);
    for(auto* v:{&re0,&im0,&re1,&im1,&eigen})for(auto& e:*v)e=u(rng)-0.5;
    std::vector<double> re1T(size_t(bands)*stride,0.0),im1T(re1T.size(),0.0);
    for(int k=0;k<bands;++k)for(int j=0;j<columns;++j){re1T[k*stride+j]=re1[size_t(j)*bands+k];im1T[k*stride+j]=im1[size_t(j)*bands+k];}
    std::vector<const double*> re0_rows,im0_rows;
    for(int i=0;i<rows;++i){re0_rows.push_back(re0.data()+size_t(i)*bands);im0_rows.push_back(im0.data()+size_t(i)*bands);}
    const double co=0.6,si=0.8;
    std::vector<double> cdm_ref(size_t(rows)*columns,0.0),edm_ref(cdm_ref.size(),0.0);
    const double band_legacy=seconds(blocks,[&]{
      size_t p=0;
      for(int i=0;i<rows;++i){const double* r0=re0.data()+size_t(i)*bands;const double* i0=im0.data()+size_t(i)*bands;
        for(int j=0;j<columns;++j,++p){const double* r1=re1.data()+size_t(j)*bands;const double* j1=im1.data()+size_t(j)*bands;
          double d1=0,d2=0,d3=0,d4=0;
          for(int k=0;k<bands;++k){const double reA=r0[k]*r1[k]+i0[k]*j1[k],imA=r0[k]*j1[k]-i0[k]*r1[k];d1+=reA;d2+=imA;d3+=reA*eigen[k];d4+=imA*eigen[k];}
          cdm_ref[p]+=co*d1-si*d2;edm_ref[p]+=co*d3-si*d4;}}});
    const double macs=double(rows)*columns*bands*blocks;
    std::cout<<"band block: "<<rows<<"x"<<columns<<" bands="<<bands<<" blocks="<<blocks<<'\n'
        <<std::setw(24)<<"legacy loops"<<std::setw(10)<<1e9*band_legacy/macs<<" ns/(i,j,k)\n";
    for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      std::vector<double> cdm(cdm_ref.size(),0.0),edm(cdm_ref.size(),0.0);
      OpenMXEnergyBandBlock b{rows,columns,bands,stride,re0_rows.data(),im0_rows.data(),re1T.data(),im1T.data(),eigen.data(),co,si,cdm.data(),edm.data()};
      const double t=seconds(blocks,[&]{if(!openmx_core_energy_band_block(&b,mode))throw std::runtime_error("band block failed");});
      if(cdm!=cdm_ref || edm!=edm_ref)throw std::runtime_error("band block differs from the legacy loops");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*t/macs<<" ns/(i,j,k)  ("<<band_legacy/t<<"x)\n";
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
