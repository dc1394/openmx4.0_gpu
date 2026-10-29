#include "openmx_core_vna_simd.h"
#include "openmx_core_nonlocal.h"
#include "openmx_core_host_simd.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
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
      std::vector<double> w0(size_t(radials)*points),w1(w0.size());
      const double legacy=seconds(repeats,[&]{
        for(int ll=0;ll<orders;++ll)for(int m=0;m<radials;++m) {
          for(int i=0;i<points;++i){bes0[i]=a0[m][i]*sphb[size_t(ll)*points+i];bes1[i]=a1[m][i]*sphbp[size_t(ll)*points+i];}
          for(int r=0;r<rows;++r){double t1=0,t2=0;for(int i=0;i<points;++i){t1+=bes0[i]*table[size_t(i)*rows+r];t2+=bes1[i]*table[size_t(i)*rows+r];}ref0[size_t(m)*rows+r]=t1;ref1[size_t(m)*rows+r]=t2;}
        }});
      std::cout<<"DS_VNA contraction: legacy "<<1e3*legacy/repeats<<" ms per pair\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds(repeats,[&]{for(int ll=0;ll<orders;++ll){
          for(int m=0;m<radials;++m)for(int i=0;i<points;++i){w0[size_t(m)*points+i]=a0[m][i]*sphb[size_t(ll)*points+i];w1[size_t(m)*points+i]=a1[m][i]*sphbp[size_t(ll)*points+i];}
          OpenMXVnaContract c={points,rows,radials,size_t(rows),size_t(rows),table.data(),w0.data(),w1.data(),out0.data(),out1.data(),size_t(points)};
          if(!openmx_core_vna_contract_flat(&c,mode))throw std::runtime_error("contract failed");}});
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
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
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
        for(int transposed:{0,1})for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
          if(!openmx_core_host_simd_usable(mode))continue;
          const double s=seconds(repeats,[&]{for(int kk=0;kk<neighbours;++kk){
            OpenMXVnaHvnaBlock b={orbitals,orbitals,int(lengths.size()),fp32,kk==0,transposed,lengths.data(),energies.data(),
                transposed?left.data():leftT.data(),transposed?size_t(components):16,transposed?rightT.data():right.data(),transposed?16:size_t(components),matrix.data(),size_t(orbitals)};
            if(!openmx_core_vna_hvna_block(&b,mode))throw std::runtime_error("hvna failed");}});
          if(matrix!=ref)throw std::runtime_error("HVNA block differs from the legacy rows");
          std::string what=std::string(openmx_core_host_simd_name(mode))+(transposed?" (lanes over n)":" (lanes over m)");
          line(what.c_str(),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
        }
        if(!fp32) {
          // operands widened once (the host cache stores double lanes under fp64)
          std::vector<double> dleft(left.begin(),left.end()),dright(right.begin(),right.end()),dleftT(leftT.begin(),leftT.end()),drightT(rightT.begin(),rightT.end());
          for(int transposed:{0,1})for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
            if(!openmx_core_host_simd_usable(mode))continue;
            const double s=seconds(repeats,[&]{for(int kk=0;kk<neighbours;++kk){
              OpenMXVnaHvnaBlock b={orbitals,orbitals,int(lengths.size()),0,kk==0,transposed,lengths.data(),energies.data(),
                  nullptr,transposed?size_t(components):16,nullptr,transposed?16:size_t(components),matrix.data(),size_t(orbitals)};
              b.left_double=transposed?dleft.data():dleftT.data();b.right_double=transposed?drightT.data():dright.data();
              if(!openmx_core_vna_hvna_block(&b,mode))throw std::runtime_error("hvna failed");}});
            if(matrix!=ref)throw std::runtime_error("HVNA block from double operands differs from the legacy rows");
            std::string what=std::string(openmx_core_host_simd_name(mode))+(transposed?" (lanes over n, f64 ops)":" (lanes over m, f64 ops)");
            line(what.c_str(),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
          }
        }
      }
    }
    // DS_VNA m sum and conversions: Si s2p2d1 (L0<=2, radials 2,2,1) against 108 projectors (L1<=8), LL<=16
    {
      struct Cx{double r,i;};
      auto cmul=[](Cx a,Cx b){Cx c;c.r=a.r*b.r-a.i*b.i;c.i=a.i*b.r+a.r*b.i;return c;};
      auto cadd=[](Cx a,Cx b){Cx c;c.r=a.r+b.r;c.i=a.i+b.i;return c;};
      auto im_pow=[](int Ls){if(Ls%2==0)return Ls%4==0?Cx{1,0}:Cx{-1,0};return (Ls+1)%4==0?Cx{0,1}:Cx{0,-1};};
      const int L0max=2,nb[3]={2,2,1},l1max=proj_lmax,llmax=16,width=2*l1max+1,cols=(l1max+1)*(l1max+1)*mult,num_rvna=rows;
      int orb_rows=0,orb_off[3];for(int l=0;l<=L0max;++l){orb_off[l]=orb_rows;orb_rows+=nb[l]*(2*l+1);}
      std::vector<double> sum0(size_t(radials)*num_rvna),sum1(sum0.size());for(auto& v:sum0)v=u(rng);for(auto& v:sum1)v=u(rng);
      std::vector<Cx> Y((llmax+1)*(llmax+1)),Yt(Y.size()),Yp(Y.size());for(auto& v:Y)v={u(rng),u(rng)};for(auto& v:Yt)v={u(rng),u(rng)};for(auto& v:Yp)v={u(rng),u(rng)};
      const size_t gstride=size_t(2*l1max+1);
      std::vector<double> gaunt(size_t(9)*(llmax+1)*(l1max+1)*gstride);for(auto& v:gaunt)v=u(rng);
      auto grow=[&](int L0,int M0,int LL,int L1){return ((((size_t)(L0*(L0+1)+M0))*(llmax+1)+LL)*(l1max+1)+L1)*gstride;};
      std::vector<double> conv(2*size_t(l1max+1)*width*width),convt_r(size_t(l1max+1)*width*width),convt_i(convt_r.size());
      for(auto& v:conv)v=u(rng);
      for(int l=0;l<=l1max;++l)for(int row=0;row<=2*l;++row)for(int col=0;col<=2*l;++col){convt_r[((size_t)l*width+col)*width+row]=conv[2*(((size_t)l*width+row)*width+col)];convt_i[((size_t)l*width+col)*width+row]=conv[2*(((size_t)l*width+row)*width+col)+1];}
      auto Conv=[&](int l,int row,int col){return Cx{conv[2*(((size_t)l*width+row)*width+col)],conv[2*(((size_t)l*width+row)*width+col)+1]};};
      auto pairi=[&](int L0,int Mul0,int m0,int L,int m1){const int L1=L/mult,Mul1=L%mult;return (size_t)(orb_off[L0]+Mul0*(2*L0+1)+m0)*cols+(size_t)mult*L1*L1+(size_t)Mul1*(2*L1+1)+m1;};
      auto sidx=[&](int L0,int Mul0,int L){return (size_t)(L0==0?Mul0:L0==1?2+Mul0:4)*num_rvna+L;};
      const double siT=0.6,coT=0.8,siP=0.3,coP=0.95,r=5.0;
      std::vector<Cx> T0((size_t)orb_rows*cols),Tr(T0.size()),Tt(T0.size()),Tp(T0.size());
      std::vector<Cx> C0((size_t)5*width),Cr(C0.size()),Ct(C0.size()),Cp(C0.size());
      std::vector<float> ref(4*(size_t)orb_rows*cols),out(ref.size());
      const double legacy=seconds(repeats,[&]{
        for(auto& v:T0)v={0,0};for(auto& v:Tr)v={0,0};for(auto& v:Tt)v={0,0};for(auto& v:Tp)v={0,0};
        for(int LL=0;LL<=llmax;++LL)for(int m=-LL;m<=LL;++m){const Cx cy=Y[LL*LL+LL+m],cyt=Yt[LL*LL+LL+m],cyp=Yp[LL*LL+LL+m];
          for(int L0=0;L0<=L0max;++L0)for(int Mul0=0;Mul0<nb[L0];++Mul0)for(int L=0;L<num_rvna;++L){const int L1=L/mult,Ls=-L0+L1+LL;
            if(std::abs(L1-LL)<=L0 && L0<=L1+LL){const Cx cp=im_pow(Ls),y1=cmul(cp,cy),t1=cmul(cp,cyt),p1=cmul(cp,cyp);
              for(int M0=-L0;M0<=L0;++M0){const int M1=M0-m;if(std::abs(M1)<=L1){const size_t pi=pairi(L0,Mul0,L0+M0,L,L1+M1);const double g=gaunt[grow(L0,M0,LL,L1)+(L1+M1)];
                const double tmp2=g*sum0[sidx(L0,Mul0,L)],tmp0=g*sum1[sidx(L0,Mul0,L)];
                T0[pi].r+=y1.r*tmp2;T0[pi].i+=y1.i*tmp2;Tr[pi].r+=y1.r*tmp0;Tr[pi].i+=y1.i*tmp0;Tt[pi].r+=t1.r*tmp2;Tt[pi].i+=t1.i*tmp2;Tp[pi].r+=p1.r*tmp2;Tp[pi].i+=p1.i*tmp2;}}}}}
        int num0=0;
        for(int L0=0;L0<=L0max;++L0)for(int Mul0=0;Mul0<nb[L0];++Mul0){int num1=0;
          for(int L=0;L<num_rvna;++L){const int L1=L/mult;
            for(int M0=-L0;M0<=L0;++M0)for(int M1=-L1;M1<=L1;++M1){Cx a={0,0},b={0,0},c={0,0},d={0,0};
              for(int k=-L0;k<=L0;++k){const size_t pi=pairi(L0,Mul0,L0+k,L,L1+M1);Cx c1=Conv(L0,L0+M0,L0+k);c1.i=-c1.i;
                a=cadd(a,cmul(c1,T0[pi]));b=cadd(b,cmul(c1,Tr[pi]));c=cadd(c,cmul(c1,Tt[pi]));d=cadd(d,cmul(c1,Tp[pi]));}
              C0[(size_t)(L0+M0)*width+(L1+M1)]=a;Cr[(size_t)(L0+M0)*width+(L1+M1)]=b;Ct[(size_t)(L0+M0)*width+(L1+M1)]=c;Cp[(size_t)(L0+M0)*width+(L1+M1)]=d;}
            for(int M0=-L0;M0<=L0;++M0)for(int M1=-L1;M1<=L1;++M1){Cx a={0,0},b={0,0},c={0,0},d={0,0};
              for(int k=-L1;k<=L1;++k){const size_t ci=(size_t)(L0+M0)*width+(L1+k);const Cx cv=Conv(L1,L1+M1,L1+k);
                a.r+=(C0[ci].r*cv.r-C0[ci].i*cv.i);a.i+=(C0[ci].r*cv.i+C0[ci].i*cv.r);b.r+=(Cr[ci].r*cv.r-Cr[ci].i*cv.i);b.i+=(Cr[ci].r*cv.i+Cr[ci].i*cv.r);
                c.r+=(Ct[ci].r*cv.r-Ct[ci].i*cv.i);c.i+=(Ct[ci].r*cv.i+Ct[ci].i*cv.r);d.r+=(Cp[ci].r*cv.r-Cp[ci].i*cv.i);d.i+=(Cp[ci].r*cv.i+Cp[ci].i*cv.r);}
              float* o=ref.data()+(size_t)(num0+L0+M0)*cols+num1+L1+M1;const size_t plane=(size_t)orb_rows*cols;
              o[0]=8.0*(float)a.r;o[plane]=-8.0*(float)(siT*coP*b.r+coT*coP/r*c.r-siP/siT/r*d.r);o[2*plane]=-8.0*(float)(siT*siP*b.r+coT*siP/r*c.r+coP/siT/r*d.r);o[3*plane]=-8.0*(float)(coT*b.r-siT/r*c.r);}
            num1+=2*L1+1;}
          num0+=2*L0+1;}});
      std::cout<<"DS_VNA m sum + conversions: legacy "<<1e3*legacy/repeats<<" ms per pair\n";
      const int pad=L0max+l1max;const size_t cy_stride=size_t(pad+std::max(2*llmax+1,llmax+L0max+l1max+8)+8),cy_length=size_t(llmax+1)*cy_stride,plane=(size_t)orb_rows*cols;
      std::vector<double> cy(6*cy_length,0.0),acc(8*plane),cpr(llmax+1),cpi(llmax+1),s0v(llmax+1),s1v(llmax+1);
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        const double s=seconds(repeats,[&]{
          for(int LL=0;LL<=llmax;++LL)for(int m=-LL;m<=LL;++m){const size_t at=size_t(LL)*cy_stride+size_t(pad+LL-m);cy[at]=Y[LL*LL+LL+m].r;cy[cy_length+at]=Y[LL*LL+LL+m].i;cy[2*cy_length+at]=Yt[LL*LL+LL+m].r;cy[3*cy_length+at]=Yt[LL*LL+LL+m].i;cy[4*cy_length+at]=Yp[LL*LL+LL+m].r;cy[5*cy_length+at]=Yp[LL*LL+LL+m].i;}
          for(int L0=0;L0<=L0max;++L0)for(int Mul0=0;Mul0<nb[L0];++Mul0)for(int L=0;L<num_rvna;++L){const int L1=L/mult;
            const int first=std::abs(L1-L0),last=std::min(L1+L0,llmax);int n=0;
            for(int LL=first;LL<=last;++LL){const Cx cp=im_pow(-L0+L1+LL);cpr[n]=cp.r;cpi[n]=cp.i;s0v[n]=sum0[sidx(L0,Mul0,L)];s1v[n]=sum1[sidx(L0,Mul0,L)];++n;}
            OpenMXVnaAngular a;a.rows=2*L0+1;a.columns=2*L1+1;a.l0=L0;a.l1=L1;a.pad=pad;a.ll_first=first;a.ll_count=n;a.cy_length=cy_length;a.cy_stride=cy_stride;
            for(int c=0;c<6;++c)a.cy[c]=cy.data()+c*cy_length;
            a.cpow_r=cpr.data();a.cpow_i=cpi.data();a.s0=s0v.data();a.s1=s1v.data();a.gaunt=gaunt.data()+grow(L0,-L0,first,L1);a.gaunt_ll_stride=size_t(l1max+1)*(2*l1max+1);a.gaunt_stride=grow(L0,-L0+1,0,L1)-grow(L0,-L0,0,L1);
            const size_t origin=pairi(L0,Mul0,0,L,0);for(int c=0;c<8;++c)a.acc[c]=acc.data()+c*plane+origin;a.acc_stride=cols;
            if(!openmx_core_vna_angular_flat(&a,mode))throw std::runtime_error("angular failed");}
          OpenMXVnaConvert v;v.onsite=0;v.polar=0;v.acc_stride=cols;v.width=width;v.ld=width;v.out_stride=cols;
          v.c1=siT*coP;v.c2=coT*coP/r;v.c3=siP/siT/r;v.c4=siT*siP;v.c5=coT*siP/r;v.c6=coP/siT/r;v.c7=coT;v.c8=siT/r;
          int num0=0;
          for(int L0=0;L0<=L0max;++L0)for(int Mul0=0;Mul0<nb[L0];++Mul0){int num1=0;
            for(int L=0;L<num_rvna;++L){const int L1=L/mult;const size_t origin=pairi(L0,Mul0,0,L,0);
              v.rows0=2*L0+1;v.columns=2*L1+1;for(int c=0;c<8;++c)v.acc[c]=acc.data()+c*plane+origin;
              v.conv0=conv.data()+2*((size_t)L0*width*width);v.convt_r=convt_r.data()+(size_t)L1*width*width;v.convt_i=convt_i.data()+(size_t)L1*width*width;
              for(int c=0;c<4;++c)v.out[c]=out.data()+c*plane+(size_t)num0*cols+num1;
              if(!openmx_core_vna_convert(&v,mode))throw std::runtime_error("convert failed");
              num1+=2*L1+1;}
            num0+=2*L0+1;}});
        for(size_t i=0;i<out.size();++i)if(std::memcmp(&out[i],&ref[i],sizeof(float))) {
          std::cerr<<"component "<<i/plane<<" row "<<(i%plane)/cols<<" column "<<(i%plane)%cols<<": kernel "<<out[i]<<" legacy "<<ref[i]<<'\n';
          throw std::runtime_error("m sum/conversions differ from the legacy loops");
        }
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per pair");
      }
    }
    // Set_OLP_Kin radial quadrature: 901 points, orders l <= 4, 25 radial pairs (Si s2p2d1 x Si), 1044 pairs
    {
      const int pts=901,ords=5,combos=25,npairs=1044;const size_t ts=32;
      std::vector<double> k(pts),r0(size_t(pts)*ts),r1(r0.size()),sb(size_t(ords)*pts),sbp(sb.size());
      for(int i=0;i<pts;++i){k[i]=0.01*i;for(int c=0;c<combos;++c){r0[size_t(i)*ts+c]=1.0/(1+c+i);r1[size_t(i)*ts+c]=1.0/(2+c+0.5*i);}
        for(int l=0;l<ords;++l){sb[size_t(l)*pts+i]=1.0/(1+l+i);sbp[size_t(l)*pts+i]=1.0/(3+l+i);}}
      std::vector<double> ref[4],out[4];for(int q=0;q<4;++q){ref[q].assign(size_t(ords)*combos,0);out[q].assign(ref[q].size(),0);}
      const double h=0.01;
      const double legacy=seconds(repeats,[&]{for(int pp=0;pp<npairs;++pp)for(int l=0;l<ords;++l)for(int c=0;c<combos;++c){
        double s0=0,k0=0,sr=0,kr=0;
        for(int i=0;i<pts;++i){const double coe0=(i==0||i==pts-1)?0.5:1.0,N=k[i],N2=N*N,t0=coe0*h*N2*r0[size_t(i)*ts+c],t1=t0*sb[size_t(l)*pts+i],t2=t0*N*sbp[size_t(l)*pts+i],t3=t1*r1[size_t(i)*ts+c],t4=t2*r1[size_t(i)*ts+c];
          s0+=t3;k0+=t3*N2;sr+=t4;kr+=t4*N2;}
        ref[0][size_t(l)*combos+c]=s0;ref[1][size_t(l)*combos+c]=k0;ref[2][size_t(l)*combos+c]=sr;ref[3][size_t(l)*combos+c]=kr;}});
      std::cout<<"Set_OLP_Kin radial quadrature: legacy "<<1e3*legacy/repeats<<" ms per call ("<<npairs<<" pairs)\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        OpenMXVnaOverlapRadial q={};q.points=pts;q.orders=ords;q.combos=combos;q.onsite=0;q.h=h;q.k=k.data();
        q.rf0t=r0.data();q.rf1t=r1.data();q.table_stride=ts;q.sb=sb.data();q.sbp=sbp.data();q.sb_stride=pts;
        for(int c=0;c<4;++c)q.out[c]=out[c].data();
        q.out_stride=combos;
        const double s=seconds(repeats,[&]{for(int pp=0;pp<npairs;++pp)if(!openmx_core_vna_overlap_radial(&q,mode))throw std::runtime_error("overlap radial failed");});
        for(int c=0;c<4;++c)if(out[c]!=ref[c])throw std::runtime_error("overlap radial differs from the legacy loop");
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per call");
      }
    }
    // Set_Nonlocal radial quadrature: 901 points, orders l <= 4, 30 combos (Si s2p2d1 x 6 projectors), 1044 pairs
    {
      const int pts=901,ords=5,nb=5,np=6,combos=nb*np,npairs=1044;const size_t ts=32;
      std::vector<double> k(pts),rf(size_t(nb)*pts),nl(size_t(np)*pts),rft(size_t(pts)*ts),nlt(rft.size()),sb(size_t(ords)*pts),sbp(sb.size());
      for(int i=0;i<pts;++i){k[i]=0.01*i;for(int b=0;b<nb;++b)rf[size_t(b)*pts+i]=1.0/(1+b+i);for(int L=0;L<np;++L)nl[size_t(L)*pts+i]=1.0/(2+L+0.5*i);
        for(int l=0;l<ords;++l){sb[size_t(l)*pts+i]=1.0/(1+l+i);sbp[size_t(l)*pts+i]=1.0/(3+l+i);}
        for(int c=0;c<combos;++c){rft[size_t(i)*ts+c]=rf[size_t(c/np)*pts+i];nlt[size_t(i)*ts+c]=nl[size_t(c%np)*pts+i];}}
      std::vector<double> ref[2],out[2];for(int q=0;q<2;++q){ref[q].assign(size_t(ords)*combos,0);out[q].assign(ref[q].size(),0);}
      const double h=0.01;
      // the legacy loop as openmx_nonlocal_cpu.h runs it: per order, points
      // outermost, basis rows, projectors innermost, the sums addressed through
      // the workspace buffers (openmx_core_nonlocal_sum) and species tables
      const int lmax=2,basis[3]={2,2,1};
      std::vector<double> sum_storage[2];
      for(auto& v:sum_storage)v.assign(size_t(lmax+1)*2*(np+1),0.0);
      OpenMXNonlocalBuffers buffers{};
      buffers.multiplicity=2;buffers.projector_slots=np+1;
      buffers.sums[0]=sum_storage[0].data();buffers.sums[1]=sum_storage[1].data();
      std::vector<double> rf_cache(size_t(lmax+1)*2*pts),nl_cache(size_t(np+1)*pts);
      for(int L0=0,b=0;L0<=lmax;++L0)for(int m=0;m<basis[L0];++m,++b)for(int i=0;i<pts;++i)rf_cache[size_t(L0*2+m)*pts+i]=rf[size_t(b)*pts+i];
      for(int L=1;L<=np;++L)for(int i=0;i<pts;++i)nl_cache[size_t(L)*pts+i]=nl[size_t(L-1)*pts+i];
      buffers.radial=rf_cache.data();buffers.nonlocal_radial[0]=nl_cache.data();
      const double legacy=seconds(repeats,[&]{for(int pp=0;pp<npairs;++pp)for(int LL=0;LL<ords;++LL){
        double *SumNL0=buffers.sums[0],*SumNLr0=buffers.sums[1];
        const double *RF_BesselCache=buffers.radial,*NLRF_BesselCache=buffers.nonlocal_radial[0];
        for(int L0=0;L0<=lmax;++L0)for(int Mul0=0;Mul0<basis[L0];++Mul0)for(int L=1;L<=np;++L){
          SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)]=0.0;SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)]=0.0;}
        for(int i=0;i<pts;++i){
          const double coe0=(i==0 || i==(pts-1))?0.50:1.00,Normk=k[i],sj=sb[size_t(LL)*pts+i],sjp=sbp[size_t(LL)*pts+i];
          for(int L0=0;L0<=lmax;++L0)for(int Mul0=0;Mul0<basis[L0];++Mul0){
            const double Bessel_Pro0=RF_BesselCache[size_t(L0*2+Mul0)*pts+i];
            const double tmp0=coe0*h*Normk*Normk*Bessel_Pro0,tmp1=tmp0*sj,tmp2=tmp0*Normk*sjp;
            for(int L=1;L<=np;++L){
              const double Bessel_Pro1=NLRF_BesselCache[size_t(L)*pts+i];
              SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)]+=tmp1*Bessel_Pro1;
              SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)]+=tmp2*Bessel_Pro1;
            }
          }
        }
        for(int L0=0,c=0;L0<=lmax;++L0)for(int Mul0=0;Mul0<basis[L0];++Mul0)for(int L=1;L<=np;++L,++c){
          ref[0][size_t(LL)*combos+c]=SumNL0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];
          ref[1][size_t(LL)*combos+c]=SumNLr0[openmx_core_nonlocal_sum(&buffers,L0,Mul0,L)];}}});
      std::cout<<"Set_Nonlocal radial quadrature: legacy "<<1e3*legacy/repeats<<" ms per call ("<<npairs<<" pairs)\n";
      for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode) {
        if(!openmx_core_host_simd_usable(mode))continue;
        OpenMXVnaNonlocalRadial q={};q.points=pts;q.orders=ords;q.combos=combos;q.onsite=0;q.h=h;q.k=k.data();
        q.rft=rft.data();q.nlrft=nlt.data();q.table_stride=ts;q.sb=sb.data();q.sbp=sbp.data();q.sb_stride=pts;
        q.out[0]=out[0].data();q.out[1]=out[1].data();
        q.out_stride=combos;
        const double s=seconds(repeats,[&]{for(int pp=0;pp<npairs;++pp)if(!openmx_core_vna_nonlocal_radial(&q,mode))throw std::runtime_error("nonlocal radial failed");});
        for(int c=0;c<2;++c)if(out[c]!=ref[c])throw std::runtime_error("nonlocal radial differs from the legacy loop");
        line(openmx_core_host_simd_name(mode),legacy/repeats*1e3,s/repeats*1e3,"ms per call");
      }
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
