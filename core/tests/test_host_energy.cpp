#include "openmx_core_energy_simd.h"
#include "openmx_core_host_simd.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The energy family must reproduce the legacy loops bit for bit: the
// transcriptions below are VH_AtomF.c, Dr_VH_AtomF.c, the point loop of
// OpenMX_Neutral_Pair and the (i, j, k) loops of the band density matrix,
// compiled with the Core's contraction-free flags.
namespace {
void require(bool ok,const char* what) { if(!ok)throw std::runtime_error(what); }
std::vector<int> modes() {
  std::vector<int> m;
  for(int mode=OPENMX_GRID_SCALAR;mode<=OPENMX_GRID_NEON;++mode)if(openmx_core_host_simd_usable(mode))m.push_back(mode);
  require(!m.empty() && m[0]==OPENMX_GRID_SCALAR,"scalar host kernels must always be usable");
  return m;
}
bool same_bits(double a,double b) { return std::memcmp(&a,&b,sizeof(double))==0; }
bool same(const std::vector<double>& a,const std::vector<double>& b) {
  return a.size()==b.size() && (a.empty() || !std::memcmp(a.data(),b.data(),a.size()*sizeof(double)));
}

/* ---- legacy transcriptions ---- */
double VH_AtomF(double core_charge,int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  int i;
  double t,dt;
  double xmin,xmax;
  xmin = xv[0];
  xmax = xv[N-1];
  if (xmax<=x){
    return core_charge/r;
  }
  else if (r<rv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;
    m = 4;
    rm = rv[m];
    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];
    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];
    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);
    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;
    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;
    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return a*r*r + b;
  }
  else{
    x = ((x)>(xmin))?  (x):(xmin);
    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    i = std::floor(t);
    dt = t - (double)i;
    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt
                 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1];
  }
}
double Dr_VH_AtomF(double core_charge,int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  int i;
  double t,dt;
  double xmin,xmax,tmp;
  xmin = xv[0];
  xmax = xv[N-1];
  if (xmax<=x){
    return -core_charge/r/r;
  }
  else if (r<rv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,a,b;
    double g1,g2,x1,x2,y1,y2,y12,y22,f,df;
    m = 4;
    rm = rv[m];
    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];
    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];
    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);
    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;
    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;
    a = 0.5*df/rm;
    b = f - a*rm*rm;
    (void)b;
    return 2.0*a*r;
  }
  else{
    x = ((x)>(xmin))?  (x):(xmin);
    tmp = ((double)N-1.0)/(xmax-xmin);
    t = (x-xmin)*tmp;
    i = std::floor(t);
    dt = t - (double)i;
    return 0.5*(( 3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  +2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt
                +(yv[i+2]-yv[i]))*tmp/r;
  }
}
struct Mesh { int N; double charge; std::vector<double> xv,rv,yv; };
Mesh make_mesh(std::mt19937& rng,int N) {
  Mesh m;m.N=N;
  std::uniform_real_distribution<double> u(0.0,1.0);
  m.charge=1.0+13.0*u(rng);
  const double xmin=std::log(1.0e-5*(1.0+u(rng))),xmax=std::log(8.0+4.0*u(rng));
  for(int i=0;i<N;++i){m.xv.push_back(xmin+(xmax-xmin)*i/(N-1.0));m.rv.push_back(std::exp(m.xv.back()));}
  for(int i=0;i<N+2;++i) {
    const double r=i<N?m.rv[i]:m.rv[N-1]*(1.0+0.1*(i-N+1));
    double v=m.charge/r*(1.0-std::exp(-r))*(1.0+0.05*(u(rng)-0.5));
    if(i%17==5)v=-0.0;else if(i%17==11)v=1e8*(u(rng)-0.5);
    m.yv.push_back(v);
  }
  return m;
}
struct Points { std::vector<double> x,y,z,rho,wt; };
Points make_points(std::mt19937& rng,int count,const Mesh& m,double separation) {
  Points p;
  std::uniform_real_distribution<double> u(0.0,1.0);
  for(int n=0;n<count;++n) {
    // every branch of the interpolation, both guards, and exact zeros
    const int kind=n%13;
    double rr;
    if(kind==0)rr=0.0;
    else if(kind==1)rr=1.0e-15*u(rng);
    else if(kind==2)rr=1.0e-11*u(rng);
    else if(kind==3)rr=m.rv[0]*u(rng);
    else if(kind==4)rr=m.rv[m.N-1]*(1.0+3.0*u(rng));
    // the band around exp(xmax) where the branch needs the logarithm, its
    // edges, and the first mesh point
    else if(kind==5)rr=std::exp(m.xv[m.N-1])*(1.0+1.0e-12*(u(rng)-0.5));
    else if(kind==6)rr=std::exp(m.xv[m.N-1])*(1.0+5.0e-10);
    else if(kind==7)rr=std::exp(m.xv[m.N-1])*(1.0-5.0e-10);
    else if(kind==8)rr=std::exp(m.xv[m.N-1])*(1.0+2.0e-9*(u(rng)-0.5));
    else if(kind==9)rr=m.rv[0]*(1.0+1.0e-12*(u(rng)-0.5));
    else rr=std::exp(m.xv[0]+(m.xv[m.N-1]-m.xv[0])*u(rng));
    const double theta=std::acos(2.0*u(rng)-1.0),phi=6.283185307179586*u(rng);
    p.x.push_back(rr*std::sin(theta)*std::cos(phi));
    p.y.push_back(rr*std::sin(theta)*std::sin(phi));
    p.z.push_back(rr*std::cos(theta)+separation);
    p.rho.push_back(n%7==3?-0.0:(u(rng)-0.3)*(n%11==4?1e8:1.0));
    p.wt.push_back(n%13==6?0.0:u(rng));
  }
  return p;
}
// OpenMX_Neutral_Pair's point loop (guard) / OpenMX_Neutral_CutoffAccumulate (no guard).
void legacy_points(const Points& p,const Mesh& m,int derivative,int guard,double separation,
    std::vector<double>& values,std::vector<double>& derivatives,double& sum,double& sumr) {
  sum=0.0;sumr=0.0;
  for(size_t n1=0;n1<p.x.size();++n1) {
    double x = p.x[n1];
    double y = p.y[n1];
    double z = p.z[n1];
    double rho0 = p.rho[n1];
    double wt = p.wt[n1];
    double z2 = z - separation;
    double r2 = x*x + y*y + z2*z2;
    double r = std::sqrt(r2);
    double xx = 0.5*std::log(r2);
    if (guard && r<1.0e-10) r = 1.0e-10;
    double va0 = VH_AtomF(m.charge,m.N,xx,r,m.xv.data(),m.rv.data(),m.yv.data());
    values[n1]=wt*va0*rho0;
    sum += wt*va0*rho0;
    derivatives[n1]=0.0;
    if (derivative!=0 && 1.0e-14<r){
      double dr_va0 = Dr_VH_AtomF(m.charge,m.N,xx,r,m.xv.data(),m.rv.data(),m.yv.data());
      derivatives[n1]=wt*rho0*dr_va0*z2/r;
      sumr -= wt*rho0*dr_va0*z2/r;
    }
  }
}
void neutral() {
  std::mt19937 rng(2026u);
  for(int N:{6,7,20,61})for(int count:{0,1,2,7,8,9,15,16,17,33,100,1001})for(int derivative:{0,1})
    for(int guard:{1,0})for(double separation:{0.0,0.37,3.3}) {
      if(!guard && derivative)continue;
      const Mesh m=make_mesh(rng,N);
      const Points p=make_points(rng,count,m,separation);
      std::vector<double> values(count),derivatives(count);
      double sum,sumr;
      legacy_points(p,m,derivative,guard,separation,values,derivatives,sum,sumr);
      for(int mode:modes()) {
        std::vector<double> v(count+2,-7.5),d(count+2,-7.5);
        OpenMXEnergyNeutralPoints s{count,derivative,guard,p.x.data(),p.y.data(),p.z.data(),p.rho.data(),p.wt.data(),
            separation,m.charge,N,m.xv.data(),m.rv.data(),m.yv.data(),v.data()+1,d.data()+1};
        require(openmx_core_energy_neutral_points(&s,mode),"neutral points refused a valid request");
        require(v.front()==-7.5 && v.back()==-7.5 && d.front()==-7.5 && d.back()==-7.5,"neutral points wrote outside the output");
        std::vector<double> got(v.begin()+1,v.end()-1),gotd(d.begin()+1,d.end()-1);
        if(!derivative)gotd=derivatives;
        if(!same(got,values) || !same(gotd,derivatives)) {
          for(int n=0;n<count;++n)if(got[n]!=values[n] || gotd[n]!=derivatives[n])
            std::cerr<<"mode "<<openmx_core_host_simd_name(mode)<<" N="<<N<<" point "<<n<<": "<<got[n]<<" vs "<<values[n]
                <<", derivative "<<gotd[n]<<" vs "<<derivatives[n]<<'\n';
          throw std::runtime_error("neutral-atom integrand differs from the legacy loop");
        }
        // the caller's sums over the stored values reproduce the legacy sums
        double s2=0.0,r2=0.0;
        for(int n=0;n<count;++n)s2+=got[n];
        if(derivative)for(int n=0;n<count;++n)r2-=gotd[n];
        require(std::memcmp(&s2,&sum,sizeof(double))==0 && std::memcmp(&r2,&sumr,sizeof(double))==0,"summing the stored values changed the legacy sum");
      }
    }
}

/* ---- the atomic exchange-correlation shell ---- */
double KumoF(int N,double x,const double* xv,const double* rv,const double* yv) {
  if (x<xv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,f,df,r;
    double g1,g2,x1,x2,y1,y2,y12,y22,a,b;
    r = std::exp(x);
    m = 4;
    rm = rv[m];
    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];
    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];
    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);
    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;
    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;
    a = 0.5*df/rm;
    b = f - a*rm*rm;
    return a*r*r + b;
  }
  else{
    int i;
    double t,dt;
    double xmin,xmax;
    xmin = xv[0];
    xmax = xv[N-1];
    x = (((x)<(xmax))?  (x):(xmax));
    x = (((x)>(xmin))?  (x):(xmin));
    t = ((double)N-1.0)*(x-xmin)/(xmax-xmin);
    i = std::floor(t);
    dt = t - (double)i;
    return 0.5*( ((yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  -yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i])*dt
                 +(yv[i+2]-yv[i]))*dt
                 +yv[i+1];
  }
}
double Dr_KumoF(int N,double x,double r,const double* xv,const double* rv,const double* yv) {
  if (x<xv[0]){
    int m;
    double rm,h1,h2,h3,f1,f2,f3,f4,a,b;
    double g1,g2,x1,x2,y1,y2,y12,y22,f,df;
    r = std::exp(x);
    m = 4;
    rm = rv[m];
    h1 = rv[m-1] - rv[m-2];
    h2 = rv[m]   - rv[m-1];
    h3 = rv[m+1] - rv[m];
    f1 = yv[m-2];
    f2 = yv[m-1];
    f3 = yv[m];
    f4 = yv[m+1];
    g1 = ((f3-f2)*h1/h2 + (f2-f1)*h2/h1)/(h1+h2);
    g2 = ((f4-f3)*h2/h3 + (f3-f2)*h3/h2)/(h2+h3);
    x1 = rm - rv[m-1];
    x2 = rm - rv[m];
    y1 = x1/h2;
    y2 = x2/h2;
    y12 = y1*y1;
    y22 = y2*y2;
    f =  y22*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y12*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1);
    df = 2.0*y2/h2*(3.0*f2 + h2*g1 + (2.0*f2 + h2*g1)*y2)
       + y22*(2.0*f2 + h2*g1)/h2
       + 2.0*y1/h2*(3.0*f3 - h2*g2 - (2.0*f3 - h2*g2)*y1)
       - y12*(2.0*f3 - h2*g2)/h2;
    a = 0.5*df/rm;
    b = f - a*rm*rm;
    (void)b;
    return 2.0*a*r;
  }
  else{
    int i;
    double t,dt;
    double xmin,xmax,tmp;
    xmin = xv[0];
    xmax = xv[N-1];
    x = ((x)<(xmax))?  (x):(xmax);
    x = ((x)>(xmin))?  (x):(xmin);
    tmp = ((double)N-1.0)/(xmax-xmin);
    t = (x-xmin)*tmp;
    i = std::floor(t);
    dt = t - (double)i;
    return 0.5*(( 3.0*(yv[i+3]-yv[i]-3.0*(yv[i+2]-yv[i+1]))*dt
                  +2.0*(-yv[i+3]+4.0*yv[i+2]-5.0*yv[i+1]+2.0*yv[i]))*dt
                  +(yv[i+2]-yv[i]))*tmp/r;
  }
}
double XC_Ceperly_Alder(double den,int P_switch) {
  double dum,rs,coe;
  double Ex,Ec,dEx,dEc;
  double tmp0,tmp1;
  double result;
  if (den<=1.0e-15){
    result = 0.0;
  }
  else{
    coe = 0.6203504908994;
    rs = coe*std::pow(den,-0.3333333333333333333);
    tmp0 = 0.458165293632163/rs;
    Ex = -tmp0;
    dEx = tmp0/rs;
    if (1.0<=rs){
      tmp0 = std::sqrt(rs);
      dum = (1.0 + 1.0529*tmp0 + 0.3334*rs);
      tmp1 = 0.1423/dum;
      Ec = -tmp1;
      dEc = tmp1/dum*(0.52645/tmp0 + 0.3334);
    }
    else{
      tmp0 = std::log(rs);
      Ec = -0.0480 + 0.0311*tmp0 + rs*(0.0020*tmp0 - 0.0116);
      dEc = 0.0311/rs + 0.0020*tmp0 - 0.0096;
    }
    if      (P_switch==0)
      result = Ex + Ec;
    else if (P_switch==1)
      result = Ex + Ec - 0.33333333333333333333*rs*(dEx + dEc);
    else if (P_switch==2)
      result = 0.3333333333333333333*rs*(dEx + dEc);
    else
      result = -0.3333333333333333333/(coe*coe*coe)*rs*rs*rs*rs*(dEx + dEc);
  }
  return result;
}
struct Shell {
  int points,neighbors;size_t stride;
  std::vector<double> lebedev,nx,ny,nz,cutoff2;
  std::vector<int> mesh;
  std::vector<Mesh> tables;
  std::vector<const double*> xv,rv,yv;
  double radius,center[3],vxc;
};
// A shell with neighbours inside and outside the Lebedev sphere, points on a
// neighbour (r2 = 0), a centre whose test can fail, both XC branches.
Shell make_shell(std::mt19937& rng,int points,int neighbors,size_t stride,bool weak_center) {
  Shell sh;sh.points=points;sh.neighbors=neighbors;sh.stride=stride;
  std::uniform_real_distribution<double> u(0.0,1.0);
  for(int k=0;k<3;++k)sh.center[k]=10.0*(u(rng)-0.5);
  const double rcut=4.0+4.0*u(rng);
  sh.radius=rcut*u(rng);
  sh.vxc=u(rng)<0.1?0.0:1.0;
  for(int ia=0;ia<points;++ia) {
    const double theta=std::acos(2.0*u(rng)-1.0),phi=6.283185307179586*u(rng);
    std::vector<double> row(stride,-3.0);
    row[0]=std::sin(theta)*std::cos(phi);row[1]=std::sin(theta)*std::sin(phi);row[2]=std::cos(theta);row[3]=u(rng);
    sh.lebedev.insert(sh.lebedev.end(),row.begin(),row.end());
  }
  for(int h=0;h<neighbors;++h) {
    const int N=20+int(41*u(rng));
    Mesh m;m.N=N;m.charge=1.0;
    const double xmin=std::log(1.0e-4*(1.0+u(rng))),xmax=std::log(rcut*(1.2+u(rng)));
    for(int i=0;i<N;++i){m.xv.push_back(xmin+(xmax-xmin)*i/(N-1.0));m.rv.push_back(std::exp(m.xv.back()));}
    for(int i=0;i<N+2;++i){const double r=i<N?m.rv[i]:m.rv[N-1]*(1.0+0.1*(i-N+1));
      double v=(0.2+u(rng))*std::exp(-r)*(1.0+0.1*(u(rng)-0.5));if(i%23==7)v=1.0e-16;if(i%23==15)v=-0.0;m.yv.push_back(v);}
    sh.tables.push_back(m);
    double x,y,z,c2;
    if(h==0){x=sh.center[0];y=sh.center[1];z=sh.center[2];c2=weak_center?0.25*sh.radius*sh.radius:rcut*rcut;}
    else if(h%7==3 && points) {  // exactly on a Lebedev point: r2 = 0 there
      const size_t ia=size_t(h)%size_t(points);
      x=sh.radius*sh.lebedev[ia*stride]+sh.center[0];y=sh.radius*sh.lebedev[ia*stride+1]+sh.center[1];z=sh.radius*sh.lebedev[ia*stride+2]+sh.center[2];
      c2=rcut*rcut;
    }
    else {
      const double d=2.2*rcut*u(rng),theta=std::acos(2.0*u(rng)-1.0),phi=6.283185307179586*u(rng);
      x=sh.center[0]+d*std::sin(theta)*std::cos(phi);y=sh.center[1]+d*std::sin(theta)*std::sin(phi);z=sh.center[2]+d*std::cos(theta);
      c2=(0.5+u(rng))*rcut*rcut;
    }
    sh.nx.push_back(x);sh.ny.push_back(y);sh.nz.push_back(z);sh.cutoff2.push_back(c2);sh.mesh.push_back(N);
  }
  for(const auto& m:sh.tables){sh.xv.push_back(m.xv.data());sh.rv.push_back(m.rv.data());sh.yv.push_back(m.yv.data());}
  return sh;
}
// Total_Energy.c, the Lebedev loop of Calc_EXC_EH1 for one radial shell.
void legacy_shell(const Shell& sh,double& den0,double& sumt,std::vector<double>& sum_gx,std::vector<double>& sum_gy,std::vector<double>& sum_gz) {
  const int FNAN=sh.neighbors-1;
  std::vector<double> gx(sh.neighbors),gy(sh.neighbors),gz(sh.neighbors);
  const double r=sh.radius;
  for (int ia=0; ia<sh.points; ia++){
    const double* leb=sh.lebedev.data()+size_t(ia)*sh.stride;
    double x0 = r*leb[0] + sh.center[0];
    double y0 = r*leb[1] + sh.center[1];
    double z0 = r*leb[2] + sh.center[2];
    double den = 0.0;
    for (int h_AN=0; h_AN<=FNAN; h_AN++){
      double rcutH2 = sh.cutoff2[h_AN];
      double x1 = sh.nx[h_AN];
      double y1 = sh.ny[h_AN];
      double z1 = sh.nz[h_AN];
      double dx = x1 - x0;
      double dy = y1 - y0;
      double dz = z1 - z0;
      double r2 = dx*dx+dy*dy+dz*dz;
      gx[h_AN] = 0.0;
      gy[h_AN] = 0.0;
      gz[h_AN] = 0.0;
      if (r2<rcutH2){
        double x = 0.5*std::log(r2);
        den += KumoF(sh.mesh[h_AN],x,sh.xv[h_AN],sh.rv[h_AN],sh.yv[h_AN])*sh.vxc;
        if (h_AN==0) den0 = den;
        if (h_AN!=0){
          double r1 = std::sqrt(dx*dx + dy*dy + dz*dz);
          double gden0 = Dr_KumoF(sh.mesh[h_AN],x,r1,sh.xv[h_AN],sh.rv[h_AN],sh.yv[h_AN])*sh.vxc;
          gx[h_AN] = gden0/r1*dx;
          gy[h_AN] = gden0/r1*dy;
          gz[h_AN] = gden0/r1*dz;
        }
      }
    }
    double exc0 = XC_Ceperly_Alder(den,0);
    double dexc0 = XC_Ceperly_Alder(den,3);
    double w = leb[3];
    sumt += w*den0*exc0;
    for (int h_AN=1; h_AN<=FNAN; h_AN++){
      sum_gx[h_AN] += w*den0*dexc0*gx[h_AN];
      sum_gy[h_AN] += w*den0*dexc0*gy[h_AN];
      sum_gz[h_AN] += w*den0*dexc0*gz[h_AN];
    }
  }
}
void xc() {
  std::mt19937 rng(314u);
  for(int points:{0,1,2,7,8,9,16,17,50,110})for(int neighbors:{1,2,5,13,40})for(size_t stride:{size_t(4),size_t(6)})
    for(bool weak:{false,true}) {
      if(points>50 && neighbors>13)continue;
      // three consecutive shells share the carried centre density
      std::vector<Shell> shells;
      for(int k=0;k<3;++k)shells.push_back(make_shell(rng,points,neighbors,stride,weak));
      double den0=0.0;
      std::vector<double> sumt(3),gx(3*neighbors+3),gy(gx.size()),gz(gx.size());
      std::vector<double> carries;
      for(int k=0;k<3;++k) {
        std::vector<double> sx(neighbors,0.0),sy(neighbors,0.0),sz(neighbors,0.0);
        double st=0.0;
        legacy_shell(shells[k],den0,st,sx,sy,sz);
        sumt[k]=st;carries.push_back(den0);
        std::copy(sx.begin(),sx.end(),gx.begin()+k*neighbors);std::copy(sy.begin(),sy.end(),gy.begin()+k*neighbors);std::copy(sz.begin(),sz.end(),gz.begin()+k*neighbors);
      }
      for(int mode:modes()) {
        std::vector<double> scratch(openmx_core_energy_xc_shell_scratch(neighbors)+1,-9.0);
        double carry=0.0;
        for(int k=0;k<3;++k) {
          const Shell& sh=shells[k];
          std::vector<double> sx(neighbors+2,0.0),sy(neighbors+2,0.0),sz(neighbors+2,0.0);
          sx.front()=sx.back()=sy.front()=sy.back()=sz.front()=sz.back()=4.5;
          double st=0.0;
          OpenMXEnergyXcShell shell{points,sh.lebedev.data(),stride,sh.radius,{sh.center[0],sh.center[1],sh.center[2]},sh.vxc,
              neighbors,sh.nx.data(),sh.ny.data(),sh.nz.data(),sh.cutoff2.data(),sh.mesh.data(),sh.xv.data(),sh.rv.data(),sh.yv.data(),
              scratch.data(),scratch.size()-1,&carry,&st,sx.data()+1,sy.data()+1,sz.data()+1};
          require(openmx_core_energy_xc_shell(&shell,mode),"xc shell refused a valid request");
          require(scratch.back()==-9.0 && sx.front()==4.5 && sx.back()==4.5 && sy.back()==4.5 && sz.back()==4.5,"xc shell wrote outside its buffers");
          require(sx[1]==0.0 && sy[1]==0.0 && sz[1]==0.0,"xc shell touched the centre's gradient slot");
          // bitwise: a neighbour sitting exactly on a Lebedev point yields 0/0 in
          // both the legacy loop and the kernel (the same NaN)
          bool ok=same_bits(st,sumt[k]) && same_bits(carry,carries[k]);
          for(int h=1;h<neighbors;++h)ok=ok && same_bits(sx[h+1],gx[k*neighbors+h]) && same_bits(sy[h+1],gy[k*neighbors+h]) && same_bits(sz[h+1],gz[k*neighbors+h]);
          if(!ok) {
            std::cerr<<std::hexfloat<<"mode "<<openmx_core_host_simd_name(mode)<<" points="<<points<<" neighbors="<<neighbors<<" shell "<<k
                <<" vxc="<<sh.vxc<<" radius="<<sh.radius<<": sumt "<<st<<" vs "<<sumt[k]<<", den0 "<<carry<<" vs "<<carries[k]<<'\n';
            for(int h=1;h<neighbors;++h)if(!same_bits(sx[h+1],gx[k*neighbors+h]) || !same_bits(sy[h+1],gy[k*neighbors+h]) || !same_bits(sz[h+1],gz[k*neighbors+h]))
              std::cerr<<"  h="<<h<<" cutoff2="<<sh.cutoff2[h]<<" gx "<<sx[h+1]<<" vs "<<gx[k*neighbors+h]<<" gy "<<sy[h+1]<<" vs "<<gy[k*neighbors+h]<<" gz "<<sz[h+1]<<" vs "<<gz[k*neighbors+h]<<'\n';
            throw std::runtime_error("xc shell differs from the legacy loop");
          }
        }
      }
    }
}

/* ---- band density-matrix block ---- */
struct Band {
  int rows,columns,bands;size_t stride;
  std::vector<double> re0,im0,re1,im1,re1T,im1T,eigen,cdm,edm;
  std::vector<const double*> re0_rows,im0_rows;
  double co,si;
};
Band make_band(std::mt19937& rng,int rows,int columns,int bands) {
  Band b;b.rows=rows;b.columns=columns;b.bands=bands;b.stride=(size_t(columns)+7)/8*8;
  std::uniform_real_distribution<double> u(-1.0,1.0);
  auto fill=[&](std::vector<double>& v,size_t n){v.resize(n);for(size_t i=0;i<n;++i)v[i]=i%23==7?-0.0:i%29==3?1e8*u(rng):u(rng);};
  fill(b.re0,size_t(rows)*bands);fill(b.im0,size_t(rows)*bands);
  fill(b.re1,size_t(columns)*bands);fill(b.im1,size_t(columns)*bands);
  fill(b.eigen,bands);fill(b.cdm,size_t(rows)*columns);fill(b.edm,size_t(rows)*columns);
  b.re1T.assign(size_t(bands)*b.stride,0.0);b.im1T.assign(size_t(bands)*b.stride,0.0);
  for(int k=0;k<bands;++k)for(size_t j=0;j<b.stride;++j) {
    b.re1T[k*b.stride+j]=j<size_t(columns)?b.re1[j*bands+k]:0.5*u(rng);  // padding: any finite value
    b.im1T[k*b.stride+j]=j<size_t(columns)?b.im1[j*bands+k]:0.5*u(rng);
  }
  for(int i=0;i<rows;++i){b.re0_rows.push_back(b.re0.data()+size_t(i)*bands);b.im0_rows.push_back(b.im0.data()+size_t(i)*bands);}
  b.co=std::cos(2.1*u(rng));b.si=std::sin(2.1*u(rng));
  return b;
}
// Band_DFT_Col.c, the (i, j, k) loops of the dense DM accumulation.
void legacy_band(const Band& b,std::vector<double>& CDM1,std::vector<double>& EDM1) {
  const int nk=b.bands;size_t p=0;
  for (int i = 0; i < b.rows; ++i) {
    const double * r0  = b.re0.data()+size_t(i)*nk;
    const double * im0 = b.im0.data()+size_t(i)*nk;
    for (int j = 0; j < b.columns; ++j, ++p) {
      const double * r1  = b.re1.data()+size_t(j)*nk;
      const double * im1 = b.im1.data()+size_t(j)*nk;
      double d1 = 0.0, d2 = 0.0, d3 = 0.0, d4 = 0.0;
      int k = 0;
      for (; k + 3 < nk; k += 4) {
#define KSTEP(idx) do { double reA = r0[idx] * r1[idx] + im0[idx] * im1[idx]; double imA = r0[idx] * im1[idx] - im0[idx] * r1[idx]; \
        d1 += reA; d2 += imA; d3 += reA * b.eigen[idx]; d4 += imA * b.eigen[idx]; } while (0)
        KSTEP(k);KSTEP(k + 1);KSTEP(k + 2);KSTEP(k + 3);
#undef KSTEP
      }
      for (; k < nk; ++k) {
        double reA = r0[k] * r1[k] + im0[k] * im1[k];
        double imA = r0[k] * im1[k] - im0[k] * r1[k];
        d1 += reA;d2 += imA;d3 += reA * b.eigen[k];d4 += imA * b.eigen[k];
      }
      CDM1[p] += b.co * d1 - b.si * d2;
      EDM1[p] += b.co * d3 - b.si * d4;
    }
  }
}
void band() {
  std::mt19937 rng(77u);
  for(int rows:{0,1,5,13,16})for(int columns:{0,1,3,7,8,13,16,17,29})for(int bands:{0,1,2,3,4,17,100,441}) {
    if(bands>100 && rows>5)continue;
    const Band b=make_band(rng,rows,columns,bands);
    std::vector<double> cdm=b.cdm,edm=b.edm;
    legacy_band(b,cdm,edm);
    for(int mode:modes()) {
      std::vector<double> c(b.cdm.size()+2,3.25),e(b.edm.size()+2,3.25);
      std::copy(b.cdm.begin(),b.cdm.end(),c.begin()+1);std::copy(b.edm.begin(),b.edm.end(),e.begin()+1);
      OpenMXEnergyBandBlock block{rows,columns,bands,b.stride,b.re0_rows.data(),b.im0_rows.data(),
          b.re1T.data(),b.im1T.data(),b.eigen.data(),b.co,b.si,c.data()+1,e.data()+1};
      require(openmx_core_energy_band_block(&block,mode),"band block refused a valid request");
      require(c.front()==3.25 && c.back()==3.25 && e.front()==3.25 && e.back()==3.25,"band block wrote outside the output");
      std::vector<double> gc(c.begin()+1,c.end()-1),ge(e.begin()+1,e.end()-1);
      if(!same(gc,cdm) || !same(ge,edm)) {
        std::cerr<<"mode "<<openmx_core_host_simd_name(mode)<<" rows="<<rows<<" columns="<<columns<<" bands="<<bands<<'\n';
        throw std::runtime_error("band density-matrix block differs from the legacy loops");
      }
    }
  }
}
void invalid() {
  std::mt19937 rng(5u);
  const Mesh m=make_mesh(rng,20);
  const Points p=make_points(rng,10,m,0.2);
  std::vector<double> v(10,4.0),d(10,4.0);
  OpenMXEnergyNeutralPoints s{10,1,1,p.x.data(),p.y.data(),p.z.data(),p.rho.data(),p.wt.data(),0.2,m.charge,20,
      m.xv.data(),m.rv.data(),m.yv.data(),v.data(),d.data()};
  require(!openmx_core_energy_neutral_points(&s,99),"unknown mode accepted");
  require(!openmx_core_energy_neutral_points(&s,OPENMX_HOST_SIMD_LEGACY),"legacy mode accepted as a kernel");
  require(!openmx_core_energy_neutral_points(nullptr,OPENMX_GRID_SCALAR),"missing request accepted");
  auto broken=s;broken.derivatives=nullptr;
  require(!openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"missing derivative output accepted");
  broken=s;broken.mesh=5;
  require(!openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"short mesh accepted");
  broken=s;broken.count=-1;
  require(!openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"negative count accepted");
  broken=s;broken.hartree=nullptr;
  require(!openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"missing table accepted");
  broken=s;broken.weight=nullptr;
  require(!openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"missing weights accepted");
  for(double x:v)require(x==4.0,"a refused call wrote values");
  for(double x:d)require(x==4.0,"a refused call wrote derivatives");
  broken=s;broken.count=0;broken.hartree=nullptr;
  require(openmx_core_energy_neutral_points(&broken,OPENMX_GRID_SCALAR),"an empty request needs no tables");
  {
    const Shell sh=make_shell(rng,9,4,4,false);
    std::vector<double> scratch(openmx_core_energy_xc_shell_scratch(4),0.0),sx(4,1.0),sy(4,1.0),sz(4,1.0);
    double carry=0.0,st=1.0;
    OpenMXEnergyXcShell shell{9,sh.lebedev.data(),4,sh.radius,{sh.center[0],sh.center[1],sh.center[2]},sh.vxc,4,
        sh.nx.data(),sh.ny.data(),sh.nz.data(),sh.cutoff2.data(),sh.mesh.data(),sh.xv.data(),sh.rv.data(),sh.yv.data(),
        scratch.data(),scratch.size(),&carry,&st,sx.data(),sy.data(),sz.data()};
    require(!openmx_core_energy_xc_shell(&shell,99),"unknown xc mode accepted");
    auto bad=shell;bad.scratch_count=scratch.size()-1;
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"short xc scratch accepted");
    bad=shell;bad.lebedev_stride=3;
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"narrow lebedev rows accepted");
    bad=shell;bad.neighbors=0;
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"a shell without its centre accepted");
    bad=shell;bad.den0=nullptr;
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"missing centre-density carry accepted");
    bad=shell;bad.sum_gy=nullptr;
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"missing gradient sums accepted");
    std::vector<int> short_mesh(sh.mesh);short_mesh[2]=5;
    bad=shell;bad.mesh=short_mesh.data();
    require(!openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR),"short density mesh accepted");
    require(st==1.0 && carry==0.0,"a refused xc call wrote its sums");
    for(double v:sx)require(v==1.0,"a refused xc call wrote gradient sums");
    bad=shell;bad.points=0;bad.lebedev=nullptr;
    require(openmx_core_energy_xc_shell(&bad,OPENMX_GRID_SCALAR) && st==1.0,"an empty shell needs no points");
  }
  const Band b=make_band(rng,3,5,7);
  std::vector<double> c(15,2.0),e(15,2.0);
  OpenMXEnergyBandBlock block{3,5,7,8,b.re0_rows.data(),b.im0_rows.data(),b.re1T.data(),b.im1T.data(),b.eigen.data(),b.co,b.si,c.data(),e.data()};
  require(!openmx_core_energy_band_block(&block,99),"unknown band mode accepted");
  auto bad=block;bad.stride=12;
  require(!openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"stride not a multiple of 8 accepted");
  bad=block;bad.stride=0;
  require(!openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"stride below the columns accepted");
  bad=block;bad.re1T=nullptr;
  require(!openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"missing transposed rows accepted");
  bad=block;bad.cdm=nullptr;
  require(!openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"missing output accepted");
  bad=block;bad.rows=-1;
  require(!openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"negative rows accepted");
  for(double x:c)require(x==2.0,"a refused band call wrote output");
  bad=block;bad.rows=0;bad.re1T=nullptr;
  require(openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"an empty block needs no inputs");
  bad=block;bad.bands=0;bad.re1T=nullptr;bad.eigen=nullptr;
  require(openmx_core_energy_band_block(&bad,OPENMX_GRID_SCALAR),"a block without bands needs no inputs");
  for(double x:c)require(x==2.0,"a block without bands changed the output");
}
}
int main(int argc,char** argv) {
  try {
    const std::string name=argc>1?argv[1]:"neutral";
    if(name=="neutral")neutral();
    else if(name=="xc")xc();
    else if(name=="band")band();
    else if(name=="invalid")invalid();
    else throw std::runtime_error("unknown host energy case");
    std::cout<<"PASS host energy "<<name<<" (modes:";
    for(int mode:modes())std::cout<<' '<<openmx_core_host_simd_name(mode);
    std::cout<<")\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
