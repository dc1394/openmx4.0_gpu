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
double KumoF(int N,double x,const double* xv,const double* rv,const double* yv) {
  if(x<xv[0]) {
    const double r=std::exp(x);const int m=4;const double rm=rv[m],h1=rv[m-1]-rv[m-2],h2=rv[m]-rv[m-1],h3=rv[m+1]-rv[m];
    const double f1=yv[m-2],f2=yv[m-1],f3=yv[m],f4=yv[m+1];
    const double g1=((f3-f2)*h1/h2+(f2-f1)*h2/h1)/(h1+h2),g2=((f4-f3)*h2/h3+(f3-f2)*h3/h2)/(h2+h3);
    const double y1=(rm-rv[m-1])/h2,y2=(rm-rv[m])/h2,y12=y1*y1,y22=y2*y2;
    const double f=y22*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y12*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1);
    const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y22*(2.0*f2+h2*g1)/h2+2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)-y12*(2.0*f3-h2*g2)/h2;
    const double a=0.5*df/rm;return a*r*r+(f-a*rm*rm);
  }
  const double xmin=xv[0],xmax=xv[N-1];x=x<xmax?x:xmax;x=x>xmin?x:xmin;
  const double t=((double)N-1.0)*(x-xmin)/(xmax-xmin);const int i=std::floor(t);const double dt=t-(double)i;
  return 0.5*(((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt+(yv[i+2]-yv[i]))*dt+yv[i+1];
}
double Dr_KumoF(int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  if(x<xv[0]) {
    r=std::exp(x);const int m=4;const double rm=rv[m],h1=rv[m-1]-rv[m-2],h2=rv[m]-rv[m-1],h3=rv[m+1]-rv[m];
    const double f1=yv[m-2],f2=yv[m-1],f3=yv[m],f4=yv[m+1];
    const double g1=((f3-f2)*h1/h2+(f2-f1)*h2/h1)/(h1+h2),g2=((f4-f3)*h2/h3+(f3-f2)*h3/h2)/(h2+h3);
    const double y1=(rm-rv[m-1])/h2,y2=(rm-rv[m])/h2,y12=y1*y1,y22=y2*y2;
    const double df=2.0*y2/h2*(3.0*f2+h2*g1+(2.0*f2+h2*g1)*y2)+y22*(2.0*f2+h2*g1)/h2+2.0*y1/h2*(3.0*f3-h2*g2-(2.0*f3-h2*g2)*y1)-y12*(2.0*f3-h2*g2)/h2;
    return 2.0*(0.5*df/rm)*r;
  }
  const double xmin=xv[0],xmax=xv[N-1];x=x<xmax?x:xmax;x=x>xmin?x:xmin;
  const double tmp=((double)N-1.0)/(xmax-xmin),t=(x-xmin)*tmp;const int i=std::floor(t);const double dt=t-(double)i;
  return 0.5*((3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt+2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt+(yv[i+2]-yv[i]))*tmp/r;
}
double XC_Ceperly_Alder(double den,int P_switch) {
  if(den<=1.0e-15)return 0.0;
  const double coe=0.6203504908994,rs=coe*std::pow(den,-0.3333333333333333333);
  double tmp0=0.458165293632163/rs;const double Ex=-tmp0,dEx=tmp0/rs;double Ec,dEc;
  if(1.0<=rs){tmp0=std::sqrt(rs);const double dum=(1.0+1.0529*tmp0+0.3334*rs),tmp1=0.1423/dum;Ec=-tmp1;dEc=tmp1/dum*(0.52645/tmp0+0.3334);}
  else{tmp0=std::log(rs);Ec=-0.0480+0.0311*tmp0+rs*(0.0020*tmp0-0.0116);dEc=0.0311/rs+0.0020*tmp0-0.0096;}
  return P_switch==0?Ex+Ec:-0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx+dEc);
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
    for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      double sum=0,sumr=0;
      OpenMXEnergyNeutralPoints s{points,1,1,x.data(),y.data(),z.data(),rho.data(),wt.data(),separation,charge,N,xv.data(),rv.data(),yv.data(),values.data(),derivatives.data()};
      const double t=seconds(repeats,[&]{if(!openmx_core_energy_neutral_points(&s,mode))throw std::runtime_error("neutral points failed");
        sum=0.0;sumr=0.0;for(int n=0;n<points;++n)sum+=values[n];for(int n=0;n<points;++n)sumr-=derivatives[n];});
      if(sum!=legacy_sum || sumr!=legacy_sumr)throw std::runtime_error("host kernel differs from the legacy loop");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*t/(double(points)*repeats)<<" ns/point  ("<<legacy_seconds/t<<"x)\n";
    }
    // the legacy glue pattern (256-point chunks, sums through pointers) against one accumulating call
    for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      double sum=0,sumr=0;
      double chunk_values[256],chunk_derivatives[256];
      const double chunked=seconds(repeats,[&]{sum=0.0;sumr=0.0;
        for(int first=0;first<points;first+=256){const int n=std::min(256,points-first);
          OpenMXEnergyNeutralPoints s{n,1,1,x.data()+first,y.data()+first,z.data()+first,rho.data()+first,wt.data()+first,separation,charge,N,xv.data(),rv.data(),yv.data(),chunk_values,chunk_derivatives};
          if(!openmx_core_energy_neutral_points(&s,mode))throw std::runtime_error("neutral points failed");
          for(int k=0;k<n;++k)*(volatile double*)&sum+=chunk_values[k];for(int k=0;k<n;++k)*(volatile double*)&sumr-=chunk_derivatives[k];}});
      double total=0,total_r=0;
      OpenMXEnergyNeutralPoints s{points,1,1,x.data(),y.data(),z.data(),rho.data(),wt.data(),separation,charge,N,xv.data(),rv.data(),yv.data(),nullptr,nullptr};
      const double once=seconds(repeats,[&]{total=0.0;total_r=0.0;if(!openmx_core_energy_neutral_sum(&s,&total,&total_r,mode))throw std::runtime_error("neutral sum failed");});
      if(total!=legacy_sum || total_r!=legacy_sumr)throw std::runtime_error("neutral sum differs from the legacy loop");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<"  chunked glue "<<std::setw(8)<<1e9*chunked/(double(points)*repeats)<<" ns/point, one call "<<std::setw(8)<<1e9*once/(double(points)*repeats)<<" ns/point\n";
    }
    // atomic exchange-correlation shell: 194 Lebedev points, 86 neighbours, PAO mesh 480
    {
      const int lpoints=194,nb=86,N=480;
      std::vector<double> leb(size_t(lpoints)*4),nx(nb),ny(nb),nz(nb),cut2(nb);
      std::vector<int> mesh(nb,N);
      std::vector<std::vector<double>> xvs(nb),rvs(nb),yvs(nb);
      std::vector<const double*> xvp(nb),rvp(nb),yvp(nb);
      for(int ia=0;ia<lpoints;++ia){const double th=std::acos(2.0*u(rng)-1.0),ph=6.283185307179586*u(rng);
        leb[ia*4]=std::sin(th)*std::cos(ph);leb[ia*4+1]=std::sin(th)*std::sin(ph);leb[ia*4+2]=std::cos(th);leb[ia*4+3]=1.0/lpoints;}
      const double rcut=7.0;
      for(int h=0;h<nb;++h) {
        const double d=h?2.0*rcut*std::cbrt(u(rng)):0.0,th=std::acos(2.0*u(rng)-1.0),ph=6.283185307179586*u(rng);
        nx[h]=d*std::sin(th)*std::cos(ph);ny[h]=d*std::sin(th)*std::sin(ph);nz[h]=d*std::cos(th);cut2[h]=rcut*rcut;
        const double xmn=std::log(1.0e-4),xmx=std::log(rcut*1.3);
        for(int i=0;i<N;++i){xvs[h].push_back(xmn+(xmx-xmn)*i/(N-1.0));rvs[h].push_back(std::exp(xvs[h].back()));}
        for(int i=0;i<N+2;++i){const double r=i<N?rvs[h][i]:rvs[h][N-1]*(1.0+0.1*(i-N+1));yvs[h].push_back(0.5*std::exp(-1.3*r)*(1.0+0.1*r));}
        xvp[h]=xvs[h].data();rvp[h]=rvs[h].data();yvp[h]=yvs[h].data();
      }
      const int shells=argc>5?std::atoi(argv[5]):16;
      std::vector<double> radii(shells);for(auto& r:radii)r=rcut*(0.05+0.9*u(rng));
      const double vxc=1.0;
      std::vector<double> ref_sumt(shells),ref_gx(size_t(shells)*nb),ref_gy(ref_gx.size()),ref_gz(ref_gx.size());
      double den0=0.0;
      const double xc_legacy=seconds(1,[&]{
        std::vector<double> gx(nb),gy(nb),gz(nb);
        for(int sidx=0;sidx<shells;++sidx){const double r=radii[sidx];double sumt=0.0;
          std::vector<double> sgx(nb,0.0),sgy(nb,0.0),sgz(nb,0.0);
          for(int ia=0;ia<lpoints;++ia){const double x0=r*leb[ia*4],y0=r*leb[ia*4+1],z0=r*leb[ia*4+2];double den=0.0;
            for(int h=0;h<nb;++h){const double dx=nx[h]-x0,dy=ny[h]-y0,dz=nz[h]-z0,r2=dx*dx+dy*dy+dz*dz;gx[h]=gy[h]=gz[h]=0.0;
              if(r2<cut2[h]){const double x=0.5*std::log(r2);den+=KumoF(N,x,xvp[h],rvp[h],yvp[h])*vxc;if(h==0)den0=den;
                if(h){const double r1=std::sqrt(dx*dx+dy*dy+dz*dz);const double g=Dr_KumoF(N,x,r1,xvp[h],rvp[h],yvp[h])*vxc;gx[h]=g/r1*dx;gy[h]=g/r1*dy;gz[h]=g/r1*dz;}}}
            const double exc0=XC_Ceperly_Alder(den,0),dexc0=XC_Ceperly_Alder(den,3),w=leb[ia*4+3];
            sumt+=w*den0*exc0;for(int h=1;h<nb;++h){sgx[h]+=w*den0*dexc0*gx[h];sgy[h]+=w*den0*dexc0*gy[h];sgz[h]+=w*den0*dexc0*gz[h];}}
          ref_sumt[sidx]=sumt;std::copy(sgx.begin(),sgx.end(),ref_gx.begin()+size_t(sidx)*nb);std::copy(sgy.begin(),sgy.end(),ref_gy.begin()+size_t(sidx)*nb);std::copy(sgz.begin(),sgz.end(),ref_gz.begin()+size_t(sidx)*nb);}});
      const double pairs=double(shells)*lpoints*nb;
      std::cout<<"xc shell: points="<<lpoints<<" neighbours="<<nb<<" mesh="<<N<<" shells="<<shells<<'\n'
          <<std::setw(24)<<"legacy loop"<<std::setw(10)<<1e9*xc_legacy/pairs<<" ns/(point,neighbour)  "<<1e6*xc_legacy/shells<<" us/shell\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        std::vector<double> scratch(openmx_core_energy_xc_shell_scratch(nb));
        std::vector<double> got_sumt(shells),gxs(ref_gx.size()),gys(ref_gx.size()),gzs(ref_gx.size());
        double carry=0.0;
        const double t=seconds(1,[&]{for(int sidx=0;sidx<shells;++sidx){double sumt=0.0;std::vector<double> sgx(nb,0.0),sgy(nb,0.0),sgz(nb,0.0);
          OpenMXEnergyXcShell shell{lpoints,leb.data(),4,radii[sidx],{0.0,0.0,0.0},vxc,nb,nx.data(),ny.data(),nz.data(),cut2.data(),mesh.data(),xvp.data(),rvp.data(),yvp.data(),scratch.data(),scratch.size(),&carry,&sumt,sgx.data(),sgy.data(),sgz.data()};
          if(!openmx_core_energy_xc_shell(&shell,mode))throw std::runtime_error("xc shell failed");
          got_sumt[sidx]=sumt;std::copy(sgx.begin(),sgx.end(),gxs.begin()+size_t(sidx)*nb);std::copy(sgy.begin(),sgy.end(),gys.begin()+size_t(sidx)*nb);std::copy(sgz.begin(),sgz.end(),gzs.begin()+size_t(sidx)*nb);}});
        if(got_sumt!=ref_sumt || gxs!=ref_gx || gys!=ref_gy || gzs!=ref_gz)throw std::runtime_error("xc shell differs from the legacy loop");
        std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*t/pairs<<" ns/(point,neighbour)  "<<1e6*t/shells<<" us/shell  ("<<xc_legacy/t<<"x)\n";
      }
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
    for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
      if(!openmx_core_host_simd_usable(mode))continue;
      std::vector<double> cdm(cdm_ref.size(),0.0),edm(cdm_ref.size(),0.0);
      OpenMXEnergyBandBlock b{rows,columns,bands,stride,re0_rows.data(),im0_rows.data(),re1T.data(),im1T.data(),eigen.data(),co,si,cdm.data(),edm.data()};
      const double t=seconds(blocks,[&]{if(!openmx_core_energy_band_block(&b,mode))throw std::runtime_error("band block failed");});
      if(cdm!=cdm_ref || edm!=edm_ref)throw std::runtime_error("band block differs from the legacy loops");
      std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<std::setw(10)<<1e9*t/macs<<" ns/(i,j,k)  ("<<band_legacy/t<<"x)\n";
    }
    // The band DM glue of one k point at sidia333 sizes (216 atoms x 13
    // orbitals, 87 neighbours each, nk bands): the round-2 glue transposed the
    // neighbour's rows into a panel and zeroed its padding for every (atom,
    // neighbour) block; the round-3 glue builds every atom's panel once per k
    // point. Both call the Core kernel; the outputs must agree bitwise.
    {
      const int atoms=argc>6?std::atoi(argv[6]):216,tno=13,neighbours=87,nk=bands;
      const int n=atoms*tno;const size_t pstride=16;
      std::vector<double> evr(size_t(n)*nk),evi(size_t(n)*nk);   // [orbital][band], re/im of the dcomplex rows
      for(auto& v:evr)v=u(rng)-0.5;for(auto& v:evi)v=u(rng)-0.5;
      std::vector<std::vector<int>> natn(atoms);
      std::uniform_int_distribution<int> pick(0,atoms-1);
      for(int a=0;a<atoms;++a){natn[a].push_back(a);for(int h=1;h<neighbours;++h)natn[a].push_back(pick(rng));}
      std::vector<double> phase_co(size_t(atoms)*neighbours),phase_si(phase_co.size());
      for(size_t q=0;q<phase_co.size();++q){phase_co[q]=std::cos(0.37*double(q));phase_si[q]=std::sin(0.37*double(q));}
      std::vector<double> re0(size_t(tno)*nk),im0(re0.size());
      std::vector<const double*> re0_rows(tno),im0_rows(tno);
      for(int i=0;i<tno;++i){re0_rows[i]=re0.data()+size_t(i)*nk;im0_rows[i]=im0.data()+size_t(i)*nk;}
      const size_t outputs=size_t(atoms)*neighbours*tno*tno;
      auto rows_of=[&](int a){for(int i=0;i<tno;++i)for(int k=0;k<nk;++k){re0[size_t(i)*nk+k]=evr[size_t(a*tno+i)*nk+k];im0[size_t(i)*nk+k]=evi[size_t(a*tno+i)*nk+k];}};
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        std::vector<double> c_old(outputs,0.0),e_old(outputs,0.0),c_new(outputs,0.0),e_new(outputs,0.0);
        std::vector<double> pr(size_t(nk)*pstride),pi(pr.size());
        const double old_glue=seconds(1,[&]{
          size_t p=0;
          for(int a=0;a<atoms;++a) {
            rows_of(a);
            for(int h=0;h<neighbours;++h) {
              const int b=natn[a][h];
              for(int j=0;j<tno;++j)for(int k=0;k<nk;++k){pr[size_t(k)*pstride+j]=evr[size_t(b*tno+j)*nk+k];pi[size_t(k)*pstride+j]=evi[size_t(b*tno+j)*nk+k];}
              for(int k=0;k<nk;++k)for(size_t j=tno;j<pstride;++j){pr[size_t(k)*pstride+j]=0.0;pi[size_t(k)*pstride+j]=0.0;}
              OpenMXEnergyBandBlock blk{tno,tno,nk,pstride,re0_rows.data(),im0_rows.data(),pr.data(),pi.data(),eigen.data(),
                  phase_co[size_t(a)*neighbours+h],phase_si[size_t(a)*neighbours+h],c_old.data()+p,e_old.data()+p};
              if(!openmx_core_energy_band_block(&blk,mode))throw std::runtime_error("band block failed");
              p+=size_t(tno)*tno;
            }
          }});
        std::vector<double> panels_r(size_t(atoms)*nk*pstride),panels_i(panels_r.size());
        double build=0.0;
        const double new_glue=seconds(1,[&]{
          const auto t0=std::chrono::steady_clock::now();
          for(int g=0;g<atoms;++g) {
            double* dr=panels_r.data()+size_t(g)*nk*pstride;double* di=panels_i.data()+size_t(g)*nk*pstride;
            for(int j=0;j<tno;++j)for(int k=0;k<nk;++k){dr[size_t(k)*pstride+j]=evr[size_t(g*tno+j)*nk+k];di[size_t(k)*pstride+j]=evi[size_t(g*tno+j)*nk+k];}
            for(int k=0;k<nk;++k)for(size_t j=tno;j<pstride;++j){dr[size_t(k)*pstride+j]=0.0;di[size_t(k)*pstride+j]=0.0;}
          }
          build=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
          size_t p=0;
          for(int a=0;a<atoms;++a) {
            rows_of(a);
            for(int h=0;h<neighbours;++h) {
              const int b=natn[a][h];
              OpenMXEnergyBandBlock blk{tno,tno,nk,pstride,re0_rows.data(),im0_rows.data(),
                  panels_r.data()+size_t(b)*nk*pstride,panels_i.data()+size_t(b)*nk*pstride,eigen.data(),
                  phase_co[size_t(a)*neighbours+h],phase_si[size_t(a)*neighbours+h],c_new.data()+p,e_new.data()+p};
              if(!openmx_core_energy_band_block(&blk,mode))throw std::runtime_error("band block failed");
              p+=size_t(tno)*tno;
            }
          }});
        if(c_old!=c_new || e_old!=e_new)throw std::runtime_error("per-k-point panels changed the density matrix");
        std::cout<<std::setw(24)<<openmx_core_host_simd_name(mode)<<"  band glue per k point (atoms="<<atoms<<", nk="<<nk<<"): per-block panels "
            <<std::setprecision(1)<<1e3*old_glue<<" ms, per-k-point panels "<<1e3*new_glue<<" ms (build "<<std::setprecision(2)<<1e3*build<<" ms)\n"<<std::setprecision(2);
      }
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
