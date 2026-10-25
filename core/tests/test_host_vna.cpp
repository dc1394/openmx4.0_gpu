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
  for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
bool same(const std::vector<double>& a,const std::vector<double>& b) {
  return a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(double));
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
  for(int rows:{0,1,4,8,9,13,16,17})for(int columns:{0,1,4,8,9,13,19})for(int fp32:{0,1})for(int first:{1,0})for(int transposed:{0,1}) {
    // projector groups of lengths 2L+1, repeated as in the VNA expansion
    std::vector<int> lengths;std::vector<double> energies;
    const int lmax=int(rng()%4),mult=1+int(rng()%3);
    for(int l=0;l<=lmax;++l)for(int m=0;m<mult;++m){lengths.push_back(2*l+1);energies.push_back(noisy(rng));}
    if(rng()%5==0){lengths.clear();energies.clear();}
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
    for(int mode:modes()) {
      std::vector<double> matrix=seed;
      OpenMXVnaHvnaBlock b={rows,columns,int(lengths.size()),fp32,first,transposed,lengths.data(),energies.data(),
          left.data(),lstride,right.data(),rstride,matrix.data(),stride};
      require(openmx_core_vna_hvna_block(&b,mode),"hvna block refused a valid request");
      if(!same(matrix,expected))report("HVNA block differs from the legacy rows",mode,matrix,expected);
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
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"contract";
    if(name=="contract")contract_case();
    else if(name=="bessel")bessel_case();
    else if(name=="quadrature")quadrature_case();
    else if(name=="hvna")hvna_case();
    else if(name=="invalid")invalid_case();
    else throw std::runtime_error("unknown host vna case");
    std::cout<<"PASS host vna "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
