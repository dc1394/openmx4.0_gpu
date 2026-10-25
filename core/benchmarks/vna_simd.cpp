#include "openmx_core_vna_simd.h"
#include "openmx_core_host_simd.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// The Set_ProExpn_VNA host kernels at the sizes of a Si s2p2d1 run
// (1DFFT.NumGridK 900, scf.RadialF.VNA 12, scf.BufferL.VNA 6, 13 orbitals),
// against transcriptions of the legacy loops.
namespace {
void legacy_bessel(double x,int lmax,double* sb,double* dsb) {
  int m,n,nmax;double tsb[40],invx,vsb0,vsb1,vsb2,vsbi=0.0,j0,j1,sf,tmp,si,co,ix;
  nmax=lmax+3*x+20;if(nmax<100)nmax=100;
  if(0.0<x) {
    invx=1.0/x;vsb0=0.0;vsb1=1.0e-14;
    for(n=nmax-1;(lmax+2)<n;n--){vsb2=(2.0*n+1.0)*invx*vsb1-vsb0;if(1.0e+250<vsb2){tmp=1.0/vsb2;vsb2*=tmp;vsb1*=tmp;}vsbi=vsb0;vsb0=vsb1;vsb1=vsb2;}
    n=lmax+3;tsb[n-1]=vsb1;tsb[n]=vsb0;tsb[n+1]=vsbi;tmp=tsb[n-1];tsb[n-1]/=tmp;tsb[n]/=tmp;
    for(n=lmax+2;0<n;n--){tsb[n-1]=(2.0*n+1.0)*invx*tsb[n]-tsb[n+1];if(1.0e+250<tsb[n-1]){tmp=tsb[n-1];for(m=n-1;m<=lmax+1;m++)tsb[m]/=tmp;}}
    si=std::sin(x);co=std::cos(x);ix=1.0/x;j0=si*ix;j1=si*ix*ix-co*ix;
    if(std::fabs(tsb[1])<std::fabs(tsb[0]))sf=j0/tsb[0];else sf=j1/tsb[1];
    for(n=0;n<=lmax+1;n++)sb[n]=tsb[n]*sf;
    dsb[0]=co*ix-si*ix*ix;
    for(n=1;n<=lmax;n++)dsb[n]=((double)n*sb[n-1]-(double)(n+1.0)*sb[n+1])/(2.0*(double)n+1.0);
  } else {for(n=0;n<=lmax+1;n++)sb[n]=0.0;sb[0]=1.0;dsb[0]=0.0;for(n=1;n<=lmax;n++)dsb[n]=((double)n*sb[n-1]-(double)(n+1.0)*sb[n+1])/(2.0*(double)n+1.0);}
}
template<class F> double seconds(int repeats,F&& body) {
  const auto t0=std::chrono::steady_clock::now();
  for(int r=0;r<repeats;++r)body();
  return std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}
void line(const char* what,double legacy,double s,const char* unit) {
  std::cout<<std::setw(28)<<what<<std::setw(10)<<s<<' '<<unit<<"  ("<<legacy/s<<"x)\n";
}
}
int main(int argc,char** argv) {
  try {
    const int repeats=argc>1?std::atoi(argv[1]):20;
    if(repeats<1)throw std::invalid_argument("positive repetition count required");
    std::mt19937 rng(42u);std::uniform_real_distribution<double> u(-1.0,1.0);
    const int points=900,proj_lmax=8,mult=12,rows=(proj_lmax+1)*mult,radials=5,orbitals=13;
    std::vector<double> k(points),w(points);
    for(int i=0;i<points;++i){k[i]=60.0*(i+0.5)/points;w[i]=0.1+u(rng)*0.01;}
    std::cout<<std::fixed<<std::setprecision(3);
    // Spherical Bessel functions: lmax 16 (DS_VNA) and 4 (HVNA23), r = 10 bohr
    for(int lmax:{16,4}) {
      std::vector<double> x(points),sb(size_t(lmax+1)*points),dsb(sb.size()),rsb(sb.size()),rdsb(sb.size());
      for(int i=0;i<points;++i)x[i]=k[i]*10.0;
      const double legacy=seconds(repeats,[&]{double a[40],b[40];for(int i=0;i<points;++i){legacy_bessel(x[i],lmax,a,b);for(int l=0;l<=lmax;++l){rsb[size_t(l)*points+i]=a[l];rdsb[size_t(l)*points+i]=b[l];}}});
      std::cout<<"Spherical_Bessel2 lmax="<<lmax<<", "<<points<<" arguments: legacy "<<1e3*legacy/repeats<<" ms per pair\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        OpenMXVnaBessel b={points,lmax,x.data(),sb.data(),dsb.data(),size_t(points),nullptr,nullptr,0};
        const double s=seconds(repeats,[&]{if(!openmx_core_vna_bessel(&b,mode))throw std::runtime_error("bessel failed");});
        if(sb!=rsb || dsb!=rdsb)throw std::runtime_error("Bessel kernel differs from the legacy routine");
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
      }
    }
    // DS_VNA contraction: 17 Bessel orders x 5 radial functions x 108 projector rows per pair
    {
      const int orders=17;
      std::vector<double> table(size_t(points)*rows),pro0(size_t(radials)*points),pro1(pro0.size()),sphb(size_t(orders)*points),sphbp(sphb.size());
      for(auto& v:table)v=u(rng);for(auto& v:pro0)v=u(rng);for(auto& v:pro1)v=u(rng);for(auto& v:sphb)v=u(rng);for(auto& v:sphbp)v=u(rng);
      std::vector<double> out0(size_t(radials)*rows),out1(out0.size()),ref0(out0.size()),ref1(out0.size()),bes0(points),bes1(points);
      std::vector<const double*> a0,a1;for(int m=0;m<radials;++m){a0.push_back(pro0.data()+size_t(m)*points);a1.push_back(pro1.data()+size_t(m)*points);}
      std::vector<double> w0(size_t(radials)*points),w1(w0.size());std::vector<const double*> pw0,pw1;
      for(int m=0;m<radials;++m){pw0.push_back(w0.data()+size_t(m)*points);pw1.push_back(w1.data()+size_t(m)*points);}
      const double legacy=seconds(repeats,[&]{
        for(int ll=0;ll<orders;++ll)for(int m=0;m<radials;++m) {
          for(int i=0;i<points;++i){bes0[i]=a0[m][i]*sphb[size_t(ll)*points+i];bes1[i]=a1[m][i]*sphbp[size_t(ll)*points+i];}
          for(int r=0;r<rows;++r){double t1=0,t2=0;for(int i=0;i<points;++i){t1+=bes0[i]*table[size_t(i)*rows+r];t2+=bes1[i]*table[size_t(i)*rows+r];}ref0[size_t(m)*rows+r]=t1;ref1[size_t(m)*rows+r]=t2;}
        }});
      std::cout<<"DS_VNA contraction: legacy "<<1e3*legacy/repeats<<" ms per pair\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds(repeats,[&]{for(int ll=0;ll<orders;++ll){
          for(int m=0;m<radials;++m)for(int i=0;i<points;++i){w0[size_t(m)*points+i]=a0[m][i]*sphb[size_t(ll)*points+i];w1[size_t(m)*points+i]=a1[m][i]*sphbp[size_t(ll)*points+i];}
          OpenMXVnaContract c={points,rows,radials,size_t(rows),size_t(rows),table.data(),pw0.data(),pw1.data(),out0.data(),out1.data()};
          if(!openmx_core_vna_contract(&c,mode))throw std::runtime_error("contract failed");}});
        if(out0!=ref0 || out1!=ref1)throw std::runtime_error("contraction differs from the legacy loop");
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
      }
    }
    // HVNA23 quadrature: 15 radial pairs (L0 <= L1) with up to 5 orders
    {
      const int pairs=15,orders=5;
      std::vector<double> product(size_t(points)*orders),bessel(product.size()),bd(product.size()),crude(points);
      for(auto& v:product)v=u(rng);for(auto& v:bessel)v=u(rng);for(auto& v:bd)v=u(rng);for(auto& v:crude)v=u(rng);
      std::vector<double> sum(orders),sumr(orders),rsum(orders),rsumr(orders);
      const double dk=60.0;
      const double legacy=seconds(repeats,[&]{for(int p=0;p<pairs;++p)for(int ll=0;ll<orders;++ll){double s=0,sr=0;
        for(int i=0;i<points;++i){const double t=0.50*dk*crude[i]*product[size_t(i)*orders+ll]*w[i]*k[i]*k[i];s+=t*bessel[size_t(i)*orders+ll];sr+=t*bd[size_t(i)*orders+ll]*k[i];}
        rsum[ll]=s;rsumr[ll]=sr;}});
      std::cout<<"HVNA23 quadrature: legacy "<<1e3*legacy/repeats<<" ms per pair\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        OpenMXVnaQuadrature q={points,orders,size_t(orders),product.data(),bessel.data(),bd.data(),crude.data(),w.data(),k.data(),0.50*dk,sum.data(),sumr.data()};
        const double s=seconds(repeats,[&]{for(int p=0;p<pairs;++p)if(!openmx_core_vna_quadrature(&q,mode))throw std::runtime_error("quadrature failed");});
        if(sum!=rsum || sumr!=rsumr)throw std::runtime_error("quadrature differs from the legacy loop");
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
      }
    }
    // HVNA projector contraction: 13x13 orbitals, 108 groups (972 components), 43 overlapping neighbours
    {
      std::vector<int> lengths;std::vector<double> energies;
      for(int l=0;l<=proj_lmax;++l)for(int m=0;m<mult;++m){lengths.push_back(2*l+1);energies.push_back(u(rng));}
      const int components=(proj_lmax+1)*(proj_lmax+1)*mult,neighbours=43;
      std::vector<float> left(size_t(orbitals)*components),right(size_t(orbitals)*components),leftT(size_t(components)*16),rightT(size_t(components)*16);
      for(auto& v:left)v=float(u(rng));for(auto& v:right)v=float(u(rng));
      for(int m=0;m<orbitals;++m)for(int l=0;l<components;++l){leftT[size_t(l)*16+m]=left[size_t(m)*components+l];rightT[size_t(l)*16+m]=right[size_t(m)*components+l];}
      std::vector<double> matrix(size_t(orbitals)*orbitals),ref(matrix.size());
      for(int fp32:{1,0}) {
        const double legacy=seconds(repeats,[&]{for(int kk=0;kk<neighbours;++kk)for(int m=0;m<orbitals;++m)for(int n=0;n<orbitals;++n){
          double sum=0;int L=0;for(size_t g=0;g<lengths.size();++g){double tmp0=0;for(int member=0;member<lengths[g];++member,++L)
            tmp0+=fp32?(double)(left[size_t(m)*components+L]*right[size_t(n)*components+L]):(double)left[size_t(m)*components+L]*(double)right[size_t(n)*components+L];
            sum+=energies[g]*tmp0;}
          if(kk==0)ref[size_t(m)*orbitals+n]=sum;else ref[size_t(m)*orbitals+n]+=sum;}});
        std::cout<<"HVNA projector contraction (fp32="<<fp32<<"): legacy "<<1e3*legacy/repeats<<" ms per pair\n";
        for(int transposed:{0,1})for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode) {
          if(!openmx_core_host_simd_usable(mode))continue;
          const double s=seconds(repeats,[&]{for(int kk=0;kk<neighbours;++kk){
            OpenMXVnaHvnaBlock b={orbitals,orbitals,int(lengths.size()),fp32,kk==0,transposed,lengths.data(),energies.data(),
                transposed?left.data():leftT.data(),transposed?size_t(components):16,transposed?rightT.data():right.data(),transposed?16:size_t(components),matrix.data(),size_t(orbitals)};
            if(!openmx_core_vna_hvna_block(&b,mode))throw std::runtime_error("hvna failed");}});
          if(matrix!=ref)throw std::runtime_error("HVNA block differs from the legacy rows");
          std::string what=std::string(openmx_core_host_simd_name(mode))+(transposed?" (lanes over n)":" (lanes over m)");
          line(what.c_str(),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
        }
      }
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
