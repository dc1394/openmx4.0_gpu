#include "openmx_core_vna_simd.h"
#include "openmx_core_host_simd.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The Set_ProExpn_VNA host family must reproduce the legacy C loops bit for
// bit: each case transcribes the legacy loop (compiled here without
// contraction), compares the scalar Core kernel with it, and every usable ISA
// with the scalar kernel, over shapes that cover every lane tail.
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<OPENMX_GRID_MODE_COUNT;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
bool same(const std::vector<double>& a,const std::vector<double>& b) {
  return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(double)));
}
void report(const char* what,int mode,const std::vector<double>& got,const std::vector<double>& expected) {
  for(size_t i=0;i<got.size();++i)if(std::memcmp(&got[i],&expected[i],sizeof(double))) {
    std::cerr<<what<<": mode "<<openmx_core_host_simd_name(mode)<<" element "<<i<<" got "<<got[i]<<" expected "<<expected[i]<<'\n';break;
  }
  throw std::runtime_error(what);
}
double noisy(std::mt19937& rng) {  // signs, magnitudes 1e-8..1e8, exact zeros and negative zeros
  std::uniform_real_distribution<double> u(-1.0,1.0);
  const int kind=int(rng()%16);
  if(kind==0)return 0.0;if(kind==1)return -0.0;
  return u(rng)*std::pow(10.0,double(int(rng()%17)-8));
}

// ---- contraction -----------------------------------------------------------
void contract_case() {
  std::mt19937 rng(11u);
  for(int points:{0,1,2,7,64,900})for(int rows:{0,1,3,8,9,16,17,108})for(int count:{0,1,2,3,4,5,9}) {
    const size_t stride=size_t(rows)+(rows%3),ostride=size_t(rows)+(rows%2);
    std::vector<double> table(size_t(points)*std::max<size_t>(stride,1)),b0(points),b1(points);
    std::vector<std::vector<double>> a0(count,std::vector<double>(points)),a1(count,std::vector<double>(points));
    for(auto& v:table)v=noisy(rng);for(auto& v:b0)v=noisy(rng);for(auto& v:b1)v=noisy(rng);
    for(auto& r:a0)for(auto& v:r)v=noisy(rng);for(auto& r:a1)for(auto& v:r)v=noisy(rng);
    // legacy: Bes00[i]=a0[i]*b0[i]; tmp1 += Bes00[i]*row[i] over ascending i
    std::vector<double> expected(2*size_t(count)*std::max<size_t>(ostride,1),7.0);
    for(int m=0;m<count;++m)for(int r=0;r<rows;++r) {
      double tmp1=0.0,tmp2=0.0;
      for(int i=0;i<points;++i) {
        const double bes00=a0[m][i]*b0[i],bes01=a1[m][i]*b1[i];
        tmp1+=bes00*table[size_t(i)*stride+r];tmp2+=bes01*table[size_t(i)*stride+r];
      }
      expected[size_t(m)*ostride+r]=tmp1;expected[size_t(count)*ostride+size_t(m)*ostride+r]=tmp2;
    }
    // the glue forms the legacy Bes00/Bes01 rows (radial function times Bessel value)
    std::vector<std::vector<double>> w0(count,std::vector<double>(points)),w1(count,std::vector<double>(points));
    for(int m=0;m<count;++m)for(int i=0;i<points;++i){w0[m][i]=a0[m][i]*b0[i];w1[m][i]=a1[m][i]*b1[i];}
    std::vector<const double*> pw0,pw1;
    for(int m=0;m<count;++m){pw0.push_back(w0[m].data());pw1.push_back(w1[m].data());}
    for(int mode:modes()) {
      std::vector<double> out(expected.size(),7.0);
      OpenMXVnaContract c={points,rows,count,stride,ostride,table.data(),pw0.data(),pw1.data(),
          out.data(),out.data()+size_t(count)*ostride};
      require(openmx_core_vna_contract(&c,mode),"contract refused a valid request");
      if(!same(out,expected))report("contraction differs from the legacy loop",mode,out,expected);
    }
  }
}

// ---- Spherical_Bessel2 transcription ---------------------------------------
void legacy_bessel(double x,int lmax,double* sb,double* dsb) {
  int m,n,nmax;
  double tsb[40];
  double invx,vsb0,vsb1,vsb2,vsbi=0.0;
  double j0,j1,sf,tmp,si,co,ix;
  nmax=lmax+3*x+20;
  if(nmax<100)nmax=100;
  if(0.0<x) {
    invx=1.0/x;
    vsb0=0.0;vsb1=1.0e-14;
    for(n=nmax-1;(lmax+2)<n;n--) {
      vsb2=(2.0*n+1.0)*invx*vsb1-vsb0;
      if(1.0e+250<vsb2){tmp=1.0/vsb2;vsb2*=tmp;vsb1*=tmp;}
      vsbi=vsb0;vsb0=vsb1;vsb1=vsb2;
    }
    n=lmax+3;
    tsb[n-1]=vsb1;tsb[n]=vsb0;tsb[n+1]=vsbi;
    tmp=tsb[n-1];tsb[n-1]/=tmp;tsb[n]/=tmp;
    for(n=lmax+2;0<n;n--) {
      tsb[n-1]=(2.0*n+1.0)*invx*tsb[n]-tsb[n+1];
      if(1.0e+250<tsb[n-1]){tmp=tsb[n-1];for(m=n-1;m<=lmax+1;m++)tsb[m]/=tmp;}
    }
    si=std::sin(x);co=std::cos(x);ix=1.0/x;
    j0=si*ix;j1=si*ix*ix-co*ix;
    if(std::fabs(tsb[1])<std::fabs(tsb[0]))sf=j0/tsb[0];else sf=j1/tsb[1];
    for(n=0;n<=lmax+1;n++)sb[n]=tsb[n]*sf;
    dsb[0]=co*ix-si*ix*ix;
    for(n=1;n<=lmax;n++)dsb[n]=((double)n*sb[n-1]-(double)(n+1.0)*sb[n+1])/(2.0*(double)n+1.0);
  } else {
    for(n=0;n<=lmax+1;n++)sb[n]=0.0;
    sb[0]=1.0;
    dsb[0]=0.0;
    for(n=1;n<=lmax;n++)dsb[n]=((double)n*sb[n-1]-(double)(n+1.0)*sb[n+1])/(2.0*(double)n+1.0);
  }
}
void bessel_case() {
  std::mt19937 rng(5u);
  std::uniform_real_distribution<double> u(0.0,1.0);
  for(int lmax:{0,1,4,16,30})for(int points:{0,1,2,7,8,15,16,17,33,900}) {
    const size_t stride=size_t(points)+(points%2),tstride=size_t(lmax)+1+(lmax%3);
    std::vector<double> x(points);
    for(int i=0;i<points;++i) {
      // the Gauss mesh of k times a distance: tiny to several hundred, with exact zeros
      const int kind=int(rng()%8);
      x[i]=kind==0?0.0:kind==1?1e-9*u(rng):kind==2?u(rng):kind==3?u(rng)*900.0:u(rng)*60.0*(0.5+10.0*u(rng));
    }
    std::vector<double> esb(size_t(lmax+1)*std::max<size_t>(stride,1),3.0),edsb(esb.size(),3.0);
    for(int i=0;i<points;++i) {
      double sb[40],dsb[40];
      legacy_bessel(x[i],lmax,sb,dsb);
      for(int l=0;l<=lmax;++l){esb[size_t(l)*stride+i]=sb[l];edsb[size_t(l)*stride+i]=dsb[l];}
    }
    for(int mode:modes())for(int transposed:{0,1}) {
      std::vector<double> sb(esb.size(),3.0),dsb(esb.size(),3.0),t(size_t(points)*tstride,3.0),td(t.size(),3.0);
      OpenMXVnaBessel b={points,lmax,x.data(),sb.data(),dsb.data(),stride,
          transposed?t.data():nullptr,transposed?td.data():nullptr,tstride};
      require(openmx_core_vna_bessel(&b,mode),"bessel refused a valid request");
      if(!same(sb,esb))report("Bessel values differ from Spherical_Bessel2",mode,sb,esb);
      if(!same(dsb,edsb))report("Bessel derivatives differ from Spherical_Bessel2",mode,dsb,edsb);
      if(transposed)for(int i=0;i<points;++i)for(int l=0;l<=lmax;++l)
        require(!std::memcmp(&t[size_t(i)*tstride+l],&esb[size_t(l)*stride+i],sizeof(double)) &&
            !std::memcmp(&td[size_t(i)*tstride+l],&edsb[size_t(l)*stride+i],sizeof(double)),"transposed Bessel layout differs");
    }
  }
}

// ---- quadrature ------------------------------------------------------------
void quadrature_case() {
  std::mt19937 rng(21u);
  for(int points:{0,1,5,64,900})for(int orders:{0,1,3,5,7,8,9,16,17}) {
    const size_t stride=size_t(orders)+(orders%2);
    std::vector<double> product(size_t(points)*std::max<size_t>(stride,1)),bessel(product.size()),bd(product.size());
    std::vector<double> crude(points),w(points),k(points);
    for(auto& v:product)v=noisy(rng);for(auto& v:bessel)v=noisy(rng);for(auto& v:bd)v=noisy(rng);
    for(auto& v:crude)v=noisy(rng);for(auto& v:w)v=noisy(rng);for(auto& v:k)v=noisy(rng);
    const double dk=noisy(rng);
    std::vector<double> expected(2*size_t(orders));
    for(int l=0;l<orders;++l) {
      double sum=0.0,sumr=0.0;
      for(int i=0;i<points;++i) {
        const double tmp10=0.50*dk*crude[i]*product[size_t(i)*stride+l]*w[i]*k[i]*k[i];
        sum+=tmp10*bessel[size_t(i)*stride+l];
        sumr+=tmp10*bd[size_t(i)*stride+l]*k[i];
      }
      expected[l]=sum;expected[size_t(orders)+l]=sumr;
    }
    for(int mode:modes()) {
      std::vector<double> out(expected.size(),9.0);
      OpenMXVnaQuadrature q={points,orders,stride,product.data(),bessel.data(),bd.data(),crude.data(),w.data(),k.data(),
          0.50*dk,out.data(),out.data()+orders};
      require(openmx_core_vna_quadrature(&q,mode),"quadrature refused a valid request");
      if(!same(out,expected))report("quadrature differs from the legacy loop",mode,out,expected);
    }
  }
}

// ---- HVNA rows -------------------------------------------------------------
void hvna_case() {
  std::mt19937 rng(31u);
  std::uniform_real_distribution<float> uf(-2.0f,2.0f);
  // every tile remainder: lane chunks of 1..4 vectors (full and masked), broadcast tails 1..3
  for(int rows:{0,1,2,3,4,5,8,9,13,16,17,23,33})for(int columns:{0,1,2,3,4,7,8,9,13,16,17,19,24,33})
  for(int fp32:{0,1})for(int first:{1,0})for(int transposed:{0,1}) {
    // projector groups of lengths 2L+1, repeated as in the VNA expansion
    std::vector<int> lengths;std::vector<double> energies;
    const int lmax=int(rng()%4),mult=1+int(rng()%3);
    for(int l=0;l<=lmax;++l)for(int m=0;m<mult;++m){lengths.push_back(2*l+1);energies.push_back(noisy(rng));}
    const int empty=int(rng()%5);
    if(empty==0){lengths.clear();energies.clear();}
    else if(empty==1)std::fill(lengths.begin(),lengths.end(),0);
    int components=0;for(int v:lengths)components+=v;
    const size_t lstride=size_t(transposed?components:rows)+1,rstride=size_t(transposed?columns:components)+2;
    std::vector<float> left(size_t(transposed?rows:components)*lstride),right(size_t(transposed?components:columns)*rstride);
    for(auto& v:left)v=rng()%7==0?0.0f:uf(rng)*std::pow(10.0f,float(int(rng()%9)-4));
    for(auto& v:right)v=rng()%7==0?-0.0f:uf(rng)*std::pow(10.0f,float(int(rng()%9)-4));
    auto L=[&](int m,int l){return transposed?left[size_t(m)*lstride+l]:left[size_t(l)*lstride+m];};
    auto R=[&](int n,int l){return transposed?right[size_t(l)*rstride+n]:right[size_t(n)*rstride+l];};
    const size_t stride=size_t(columns)+3;
    std::vector<double> seed(size_t(rows)*stride);
    for(auto& v:seed)v=noisy(rng);
    std::vector<double> expected=seed;
    for(int m=0;m<rows;++m)for(int n=0;n<columns;++n) {
      double sum=0.0;int l=0;
      for(size_t g=0;g<lengths.size();++g) {
        double tmp0=0.0;
        for(int member=0;member<lengths[g];++member,++l)
          tmp0+=fp32?(double)(L(m,l)*R(n,l)):(double)L(m,l)*(double)R(n,l);
        sum+=energies[g]*tmp0;
      }
      if(first)expected[size_t(m)*stride+n]=sum;else expected[size_t(m)*stride+n]+=sum;
    }
    // the fp64 products also from operands widened beforehand (the floats are then not read)
    std::vector<double> dleft(left.begin(),left.end()),dright(right.begin(),right.end());
    for(int mode:modes())for(int doubles:{0,1}) {
      if(doubles && fp32)continue;
      std::vector<double> matrix=seed;
      OpenMXVnaHvnaBlock b={rows,columns,int(lengths.size()),fp32,first,transposed,lengths.data(),energies.data(),
          left.data(),lstride,right.data(),rstride,matrix.data(),stride};
      if(doubles){b.left=nullptr;b.right=nullptr;b.left_double=dleft.data();b.right_double=dright.data();}
      require(openmx_core_vna_hvna_block(&b,mode),"hvna block refused a valid request");
      if(!same(matrix,expected))report(doubles?"HVNA block from double operands differs from the legacy rows":
          "HVNA block differs from the legacy rows",mode,matrix,expected);
    }
  }
}

// ---- DS_VNA m sum and conversions ------------------------------------------
struct Cx { double r,i; };
Cx cmul(Cx a,Cx b) { Cx c;c.r=a.r*b.r-a.i*b.i;c.i=a.i*b.r+a.r*b.i;return c; }
Cx cadd(Cx a,Cx b) { Cx c;c.r=a.r+b.r;c.i=a.i+b.i;return c; }
Cx im_pow(int Ls) { if(Ls%2==0)return Ls%4==0?Cx{1.0,0.0}:Cx{-1.0,0.0};return (Ls+1)%4==0?Cx{0.0,1.0}:Cx{0.0,-1.0}; }
void angular_case() {
  std::mt19937 rng(41u);
  for(int l0:{0,1,2,3,4})for(int l1:{0,1,2,3,5,8,14})for(int first:{0,1,3,6,13})for(int count:{0,1,2,3,5,9}) {
    const int rows=2*l0+1,cols=2*l1+1,pad=l0+l1+1,top=first+count-1;
    const size_t cy_stride=size_t(pad+std::max(2*std::max(top,0)+1,std::max(top,0)+l0+l1+8))+3;
    const size_t cy_length=size_t(std::max(top,0)+1)*cy_stride,stride=size_t(cols)+3;
    std::vector<double> cy[6];
    for(auto& c:cy){c.assign(cy_length,0.0);for(int ll=first;ll<=top;++ll)for(int m=-ll;m<=ll;++m)c[size_t(ll)*cy_stride+size_t(pad+ll-m)]=noisy(rng);}
    const size_t gstride=size_t(cols)+2;
    std::vector<std::vector<double>> gaunt(std::max(count,0),std::vector<double>(size_t(rows)*gstride));
    for(auto& g:gaunt)for(auto& v:g)v=noisy(rng);
    std::vector<double> cpr(count),cpi(count),s0(count),s1(count);
    std::vector<const double*> grows(count);
    for(int t=0;t<count;++t){const Cx cp=im_pow(-l0+l1+first+t+4*int(rng()%2));cpr[t]=cp.r;cpi[t]=cp.i;s0[t]=noisy(rng);s1[t]=noisy(rng);grows[t]=gaunt[t].data();}
    std::vector<double> seed[8];
    for(auto& a:seed){a.resize(size_t(rows)*stride+4);for(auto& v:a)v=noisy(rng);}
    // legacy: zero block; for LL: for m: CY1 = Cmul(Cpow, Y(m)); for M0: M1 = M0-m; if |M1| <= l1: accumulate
    std::vector<double> expected[8];
    for(int c=0;c<8;++c)expected[c]=seed[c];
    for(int c=0;c<8;++c)for(int a=0;a<rows;++a)for(int j=0;j<cols;++j)expected[c][size_t(a)*stride+j]=0.0;
    for(int t=0;t<count;++t) {
      const int ll=first+t;const Cx cpow={cpr[t],cpi[t]};
      for(int m=-ll;m<=ll;++m) {
        const size_t e=size_t(ll)*cy_stride+size_t(pad+ll-m);
        const Cx Y={cy[0][e],cy[1][e]},Yt={cy[2][e],cy[3][e]},Yp={cy[4][e],cy[5][e]};
        const Cx y1=cmul(cpow,Y),t1=cmul(cpow,Yt),p1=cmul(cpow,Yp);
        for(int m0=-l0;m0<=l0;++m0) {
          const int m1=m0-m;
          if(std::abs(m1)>l1)continue;
          const size_t at=size_t(m0+l0)*stride+size_t(m1+l1);
          const double gant=gaunt[t][size_t(m0+l0)*gstride+size_t(m1+l1)];
          const double tmp2=gant*s0[t],tmp0=gant*s1[t];
          expected[0][at]+=y1.r*tmp2;expected[1][at]+=y1.i*tmp2;
          expected[2][at]+=y1.r*tmp0;expected[3][at]+=y1.i*tmp0;
          expected[4][at]+=t1.r*tmp2;expected[5][at]+=t1.i*tmp2;
          expected[6][at]+=p1.r*tmp2;expected[7][at]+=p1.i*tmp2;
        }
      }
    }
    for(int mode:modes()) {
      std::vector<double> acc[8];
      OpenMXVnaAngular a;
      a.rows=rows;a.columns=cols;a.l0=l0;a.l1=l1;a.pad=pad;a.ll_first=first;a.ll_count=count;
      a.cy_length=cy_length;a.cy_stride=cy_stride;
      for(int c=0;c<6;++c)a.cy[c]=cy[c].data();
      a.cpow_r=cpr.data();a.cpow_i=cpi.data();a.s0=s0.data();a.s1=s1.data();
      a.gaunt=grows.data();a.gaunt_stride=gstride;
      for(int c=0;c<8;++c){acc[c]=seed[c];a.acc[c]=acc[c].data();}
      a.acc_stride=stride;
      require(openmx_core_vna_angular(&a,mode),"angular refused a valid request");
      for(int c=0;c<8;++c)if(!same(acc[c],expected[c]))report("m-sum accumulation differs from the legacy loop",mode,acc[c],expected[c]);
    }
  }
}
void convert_case() {
  std::mt19937 rng(43u);
  for(int l0:{0,1,2,3,4,7})for(int l1:{0,1,2,3,5,8,14,15})for(int onsite:{0,1})for(int polar:{0,1}) {
    const int rows=2*l0+1,cols=2*l1+1,width=2*std::max(l0,l1)+3;
    const size_t stride=size_t(cols)+5,ostride=size_t(cols)+7;
    std::vector<double> acc[8];
    for(auto& a:acc){a.resize(size_t(rows)*stride);for(auto& v:a)v=noisy(rng);}
    std::vector<double> conv0(2*size_t(rows)*width),convt_r(size_t(cols)*width),convt_i(convt_r.size());
    for(auto& v:conv0)v=noisy(rng);for(auto& v:convt_r)v=noisy(rng);for(auto& v:convt_i)v=noisy(rng);
    const double c[8]={noisy(rng),noisy(rng),noisy(rng),noisy(rng),noisy(rng),noisy(rng),noisy(rng),noisy(rng)};
    // legacy transcription
    std::vector<float> expected[4];
    for(auto& e:expected)e.assign(size_t(rows)*ostride,3.5f);
    for(int m0=0;m0<rows;++m0) {
      std::vector<Cx> cmat[4];
      for(auto& m:cmat)m.resize(cols);
      for(int m1=0;m1<cols;++m1) {
        Cx sum[4]={{0,0},{0,0},{0,0},{0,0}};
        for(int k=0;k<rows;++k) {
          const Cx c1={conv0[2*(size_t(m0)*width+k)],-conv0[2*(size_t(m0)*width+k)+1]};
          for(int q=0;q<4;++q) {
            const Cx t={acc[2*q][size_t(k)*stride+m1],acc[2*q+1][size_t(k)*stride+m1]};
            sum[q]=cadd(sum[q],cmul(c1,t));
          }
        }
        for(int q=0;q<4;++q)cmat[q][m1]=sum[q];
      }
      for(int m1=0;m1<cols;++m1) {
        Cx s[4]={{0,0},{0,0},{0,0},{0,0}};
        for(int k=0;k<cols;++k)for(int q=0;q<4;++q) {
          const double vr=convt_r[size_t(k)*width+m1],vi=convt_i[size_t(k)*width+m1];
          s[q].r+=(cmat[q][k].r*vr-cmat[q][k].i*vi);
          s[q].i+=(cmat[q][k].r*vi+cmat[q][k].i*vr);
        }
        const size_t at=size_t(m0)*ostride+m1;
        expected[0][at]=8.0*(float)s[0].r;
        if(onsite){expected[1][at]=0.0;expected[2][at]=0.0;expected[3][at]=0.0;}
        else if(polar) {
          expected[1][at]=-8.0*(float)(c[0]*s[1].r+c[1]*s[2].r);
          expected[2][at]=-8.0*(float)(c[3]*s[1].r+c[4]*s[2].r);
          expected[3][at]=-8.0*(float)(c[6]*s[1].r-c[7]*s[2].r);
        } else {
          expected[1][at]=-8.0*(float)(c[0]*s[1].r+c[1]*s[2].r-c[2]*s[3].r);
          expected[2][at]=-8.0*(float)(c[3]*s[1].r+c[4]*s[2].r+c[5]*s[3].r);
          expected[3][at]=-8.0*(float)(c[6]*s[1].r-c[7]*s[2].r);
        }
      }
    }
    for(int mode:modes()) {
      std::vector<float> out[4];
      OpenMXVnaConvert v;
      v.rows0=rows;v.columns=cols;v.onsite=onsite;v.polar=polar;
      for(int q=0;q<8;++q)v.acc[q]=acc[q].data();
      v.acc_stride=stride;v.conv0=conv0.data();v.width=size_t(width);
      v.convt_r=convt_r.data();v.convt_i=convt_i.data();v.ld=size_t(width);
      for(int q=0;q<4;++q){out[q].assign(size_t(rows)*ostride,3.5f);v.out[q]=out[q].data();}
      v.out_stride=ostride;
      v.c1=c[0];v.c2=c[1];v.c3=c[2];v.c4=c[3];v.c5=c[4];v.c6=c[5];v.c7=c[6];v.c8=c[7];
      require(openmx_core_vna_convert(&v,mode),"convert refused a valid request");
      for(int q=0;q<4;++q)
        require(out[q].size()==expected[q].size() && !std::memcmp(out[q].data(),expected[q].data(),out[q].size()*sizeof(float)),
            "conversion output differs from the legacy loop");
    }
  }
}

// ---- Set_OLP_Kin radial quadrature --------------------------------------------
void overlap_case() {
  std::mt19937 rng(51u);
  for(int points:{0,1,2,3,9,901})for(int orders:{0,1,2,3,5,7,9})for(int combos:{0,1,3,8,9,16,25,33})for(int onsite:{0,1}) {
    const size_t tstride=size_t(combos)+3,sstride=size_t(points)+2,ostride=size_t(combos)+1;
    std::vector<double> k(points),r0(size_t(points)*tstride),r1(r0.size()),sb(size_t(orders)*sstride),sbp(sb.size());
    for(auto& v:k)v=noisy(rng);for(auto& v:r0)v=noisy(rng);for(auto& v:r1)v=noisy(rng);
    for(auto& v:sb)v=noisy(rng);for(auto& v:sbp)v=noisy(rng);
    const double h=noisy(rng);
    // legacy: for l, for combo: for i (Set_OLP_Kin.c CPU branch)
    std::vector<double> expected[4];
    for(auto& e:expected)e.assign(size_t(orders)*ostride,5.5);
    const int last=points-1;
    for(int l=0;l<orders;l++)for(int c=0;c<combos;c++) {
      double s0=0,k0=0,sr=0,kr=0;
      for(int i=0;i<points;i++) {
        const double coe0=(i==0 || i==last)?0.50:1.00;
        const double Normk=k[i],Normk2=Normk*Normk,sj=sb[size_t(l)*sstride+i],sjp=sbp[size_t(l)*sstride+i];
        const double tmp0=coe0*h*Normk2*r0[size_t(i)*tstride+c];
        const double tmp1=tmp0*sj,tmp2=tmp0*Normk*sjp;
        const double tmp3=tmp1*r1[size_t(i)*tstride+c],tmp4=tmp2*r1[size_t(i)*tstride+c];
        s0+=tmp3;k0+=tmp3*Normk2;sr+=tmp4;kr+=tmp4*Normk2;
      }
      if(onsite){sr=0.0;kr=0.0;}
      const size_t at=size_t(l)*ostride+c;
      expected[0][at]=s0;expected[1][at]=k0;expected[2][at]=sr;expected[3][at]=kr;
    }
    for(int mode:modes()) {
      std::vector<double> out[4];
      for(auto& o:out)o.assign(size_t(orders)*ostride,5.5);
      OpenMXVnaOverlapRadial q;
      q.points=points;q.orders=orders;q.combos=combos;q.onsite=onsite;q.h=h;q.k=k.data();
      q.rf0t=r0.data();q.rf1t=r1.data();q.table_stride=tstride;q.sb=sb.data();q.sbp=sbp.data();q.sb_stride=sstride;
      for(int c=0;c<4;++c)q.out[c]=out[c].data();
      q.out_stride=ostride;
      require(openmx_core_vna_overlap_radial(&q,mode),"overlap radial refused a valid request");
      for(int c=0;c<4;++c)if(!same(out[c],expected[c]))report("overlap radial differs from the legacy loop",mode,out[c],expected[c]);
    }
  }
}

// ---- Set_Nonlocal radial quadrature -------------------------------------------
void nonlocal_case() {
  std::mt19937 rng(53u);
  for(int points:{0,1,2,3,9,901})for(int orders:{0,1,2,3,5,7,9})for(int combos:{0,1,3,8,9,16,30,33})for(int onsite:{0,1}) {
    const size_t tstride=size_t(combos)+3,sstride=size_t(points)+2,ostride=size_t(combos)+1;
    std::vector<double> k(points),rf(size_t(points)*tstride),nl(rf.size()),sb(size_t(orders)*sstride),sbp(sb.size());
    for(auto& v:k)v=noisy(rng);for(auto& v:rf)v=noisy(rng);for(auto& v:nl)v=noisy(rng);
    for(auto& v:sb)v=noisy(rng);for(auto& v:sbp)v=noisy(rng);
    const double h=noisy(rng);
    // legacy: for LL, zero the sums, then for i, (L0,Mul0), L (openmx_nonlocal_cpu.h CPU loop)
    std::vector<double> expected[2];
    for(auto& e:expected)e.assign(size_t(orders)*ostride,5.5);
    for(int LL=0;LL<orders;LL++) {
      std::vector<double> SumNL0(combos,0.0),SumNLr0(combos,0.0);
      for(int i=0;i<points;i++) {
        double coe0;
        if(i==0 || i==(points-1))coe0=0.50;
        else coe0=1.00;
        const double Normk=k[i],sj=sb[size_t(LL)*sstride+i],sjp=sbp[size_t(LL)*sstride+i];
        for(int c=0;c<combos;c++) {
          const double Bessel_Pro0=rf[size_t(i)*tstride+c];
          const double tmp0=coe0*h*Normk*Normk*Bessel_Pro0;
          const double tmp1=tmp0*sj;
          const double tmp2=tmp0*Normk*sjp;
          const double Bessel_Pro1=nl[size_t(i)*tstride+c];
          const double tmp3=tmp1*Bessel_Pro1;
          const double tmp4=tmp2*Bessel_Pro1;
          SumNL0[c]+=tmp3;
          SumNLr0[c]+=tmp4;
        }
      }
      if(onsite)for(int c=0;c<combos;c++)SumNLr0[c]=0.0;
      for(int c=0;c<combos;c++){expected[0][size_t(LL)*ostride+c]=SumNL0[c];expected[1][size_t(LL)*ostride+c]=SumNLr0[c];}
    }
    for(int mode:modes()) {
      std::vector<double> out[2];
      for(auto& o:out)o.assign(size_t(orders)*ostride,5.5);
      OpenMXVnaNonlocalRadial q;
      q.points=points;q.orders=orders;q.combos=combos;q.onsite=onsite;q.h=h;q.k=k.data();
      q.rft=rf.data();q.nlrft=nl.data();q.table_stride=tstride;q.sb=sb.data();q.sbp=sbp.data();q.sb_stride=sstride;
      q.out[0]=out[0].data();q.out[1]=out[1].data();
      q.out_stride=ostride;
      require(openmx_core_vna_nonlocal_radial(&q,mode),"nonlocal radial refused a valid request");
      for(int c=0;c<2;++c)if(!same(out[c],expected[c]))report("nonlocal radial differs from the legacy loop",mode,out[c],expected[c]);
    }
  }
}

void invalid_case() {
  double d[16]={0};const double* rows[1]={d};int lengths[1]={2};float f[8]={0};
  OpenMXVnaContract c={4,2,1,2,2,d,rows,rows,d+4,d+8};
  require(!openmx_core_vna_contract(&c,99) && !openmx_core_vna_contract(&c,-1),"contract accepted an unknown mode");
  require(openmx_core_vna_contract(&c,OPENMX_GRID_SCALAR),"contract refused a valid request");
  c.table_stride=1;require(!openmx_core_vna_contract(&c,OPENMX_GRID_SCALAR),"contract accepted a short table stride");
  c.table_stride=2;c.out0=nullptr;require(!openmx_core_vna_contract(&c,OPENMX_GRID_SCALAR),"contract accepted a missing output");
  OpenMXVnaBessel b={3,2,d,d+4,d+8,3,nullptr,nullptr,0};
  require(!openmx_core_vna_bessel(&b,99),"bessel accepted an unknown mode");
  b.lmax=31;require(!openmx_core_vna_bessel(&b,OPENMX_GRID_SCALAR),"bessel accepted lmax above the table");
  b.lmax=2;b.stride=2;require(!openmx_core_vna_bessel(&b,OPENMX_GRID_SCALAR),"bessel accepted a short stride");
  b.stride=3;b.transposed=d;require(!openmx_core_vna_bessel(&b,OPENMX_GRID_SCALAR),"bessel accepted a lone transposed output");
  OpenMXVnaQuadrature q={2,2,2,d,d,d,d,d,d,1.0,d+8,d+10};
  require(!openmx_core_vna_quadrature(&q,99),"quadrature accepted an unknown mode");
  q.table_stride=1;require(!openmx_core_vna_quadrature(&q,OPENMX_GRID_SCALAR),"quadrature accepted a short stride");
  q.table_stride=2;q.sum=nullptr;require(!openmx_core_vna_quadrature(&q,OPENMX_GRID_SCALAR),"quadrature accepted a missing output");
  OpenMXVnaHvnaBlock h={2,2,1,0,1,0,lengths,d,f,2,f,2,d,2};
  require(!openmx_core_vna_hvna_block(&h,99),"hvna accepted an unknown mode");
  require(openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna refused a valid request");
  h.right_stride=1;require(!openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna accepted a short component stride");
  h.right_stride=2;h.stride=1;require(!openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna accepted a short matrix stride");
  h.stride=2;lengths[0]=-1;require(!openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna accepted a negative group length");
  lengths[0]=2;double dd[8]={0};
  h.left=nullptr;h.right=nullptr;h.left_double=dd;h.right_double=dd;
  require(openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna refused fp64 products from double operands");
  h.fp32=1;require(!openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna accepted fp32 products without float operands");
  h.fp32=0;h.right_double=nullptr;require(!openmx_core_vna_hvna_block(&h,OPENMX_GRID_SCALAR),"hvna accepted a lone double operand without floats");
  double cy[40]={0},planes[8*8]={0},one[1]={1.0};
  const double* grows[1]={cy};
  OpenMXVnaAngular a;
  a.rows=1;a.columns=3;a.l0=0;a.l1=1;a.pad=1;a.ll_first=1;a.ll_count=1;a.cy_length=40;a.cy_stride=20;
  for(int c=0;c<6;++c)a.cy[c]=cy;
  a.cpow_r=one;a.cpow_i=one;a.s0=one;a.s1=one;a.gaunt=grows;a.gaunt_stride=3;
  for(int c=0;c<8;++c)a.acc[c]=planes+8*c;
  a.acc_stride=3;
  require(openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"angular refused a valid request");
  require(!openmx_core_vna_angular(&a,99),"angular accepted an unknown mode");
  a.pad=0;require(!openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"angular accepted a short pad");
  a.pad=1;a.cy_length=5;require(!openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"angular accepted short harmonic tables");
  a.cy_length=40;a.rows=2;require(!openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"angular accepted rows that do not match l0");
  a.rows=1;a.ll_first=6;require(!openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"angular accepted tables shorter than the m range");
  a.ll_first=1;a.ll_count=0;a.gaunt=nullptr;require(openmx_core_vna_angular(&a,OPENMX_GRID_SCALAR),"an empty LL range must still zero the block");
  for(int c=0;c<8;++c)for(int j=0;j<3;++j)require(planes[8*c+j]==0.0,"an empty LL range left the block unwritten");
  float fo[16]={0};
  OpenMXVnaConvert v;
  v.rows0=1;v.columns=3;v.onsite=0;v.polar=0;
  for(int q=0;q<8;++q)v.acc[q]=planes+8*q;
  v.acc_stride=3;v.conv0=cy;v.width=1;v.convt_r=cy;v.convt_i=cy;v.ld=3;
  for(int q=0;q<4;++q)v.out[q]=fo+4*q;
  v.out_stride=4;v.c1=v.c2=v.c3=v.c4=v.c5=v.c6=v.c7=v.c8=1.0;
  require(openmx_core_vna_convert(&v,OPENMX_GRID_SCALAR),"convert refused a valid request");
  require(!openmx_core_vna_convert(&v,99),"convert accepted an unknown mode");
  v.columns=33;require(!openmx_core_vna_convert(&v,OPENMX_GRID_SCALAR),"convert accepted too many columns");
  v.columns=3;v.ld=2;require(!openmx_core_vna_convert(&v,OPENMX_GRID_SCALAR),"convert accepted a short transposed stride");
  v.ld=3;v.out[2]=nullptr;require(!openmx_core_vna_convert(&v,OPENMX_GRID_SCALAR),"convert accepted a missing output row");
  // the two radial quadratures: refusals leave every output untouched
  double ko[4]={1,2,3,4},ro[16]={0},so[16]={0},oo[4][8];
  auto untouched=[&](const char* message) {
    for(auto& row:oo)for(double v:row)require(v==7.0,message);
  };
  OpenMXVnaOverlapRadial o={};
  o.points=2;o.orders=2;o.combos=3;o.h=0.1;o.k=ko;o.rf0t=ro;o.rf1t=ro;o.table_stride=3;o.sb=so;o.sbp=so;o.sb_stride=2;
  for(int c=0;c<4;++c)o.out[c]=oo[c];
  o.out_stride=3;
  require(openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial refused a valid request");
  for(auto& row:oo)for(double& v:row)v=7.0;
  require(!openmx_core_vna_overlap_radial(&o,99) && !openmx_core_vna_overlap_radial(&o,OPENMX_HOST_SIMD_LEGACY),
          "overlap radial accepted an unknown mode");
  o.table_stride=2;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted a short table stride");
  o.table_stride=3;o.sb_stride=1;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted a short Bessel stride");
  o.sb_stride=2;o.out_stride=2;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted a short output stride");
  o.out_stride=3;o.rf1t=nullptr;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted a missing table");
  o.rf1t=ro;o.combos=-1;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted negative combos");
  o.combos=3;o.out[3]=nullptr;require(!openmx_core_vna_overlap_radial(&o,OPENMX_GRID_SCALAR),"overlap radial accepted a missing output");
  untouched("overlap radial wrote output on a refused request");
  OpenMXVnaNonlocalRadial n={};
  n.points=2;n.orders=2;n.combos=3;n.h=0.1;n.k=ko;n.rft=ro;n.nlrft=ro;n.table_stride=3;n.sb=so;n.sbp=so;n.sb_stride=2;
  n.out[0]=oo[0];n.out[1]=oo[1];n.out_stride=3;
  require(!openmx_core_vna_nonlocal_radial(&n,99) && !openmx_core_vna_nonlocal_radial(&n,OPENMX_HOST_SIMD_LEGACY),
          "nonlocal radial accepted an unknown mode");
  n.table_stride=2;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted a short table stride");
  n.table_stride=3;n.sb_stride=1;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted a short Bessel stride");
  n.sb_stride=2;n.out_stride=2;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted a short output stride");
  n.out_stride=3;n.nlrft=nullptr;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted a missing table");
  n.nlrft=ro;n.points=-1;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted negative points");
  n.points=2;n.out[1]=nullptr;require(!openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial accepted a missing output");
  untouched("nonlocal radial wrote output on a refused request");
  n.out[1]=oo[1];require(openmx_core_vna_nonlocal_radial(&n,OPENMX_GRID_SCALAR),"nonlocal radial refused a valid request");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"contract";
    if(name=="contract")contract_case();
    else if(name=="bessel")bessel_case();
    else if(name=="quadrature")quadrature_case();
    else if(name=="hvna")hvna_case();
    else if(name=="angular")angular_case();
    else if(name=="convert")convert_case();
    else if(name=="overlap")overlap_case();
    else if(name=="nonlocal")nonlocal_case();
    else if(name=="invalid")invalid_case();
    else throw std::runtime_error("unknown host vna case");
    std::cout<<"PASS host vna "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
