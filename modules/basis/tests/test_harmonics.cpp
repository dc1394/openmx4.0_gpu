#include <openmx/basis/harmonics.hpp>
#include <openmx/basis/quadrature.hpp>
#include "allocation_gate.hpp"

#include <array>
#include <climits>
#include <cmath>
#include <complex>
#include <future>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::basis;
using Complex=std::complex<double>;
constexpr double pi=std::numbers::pi;
void require(bool value,const char* reason) { if(!value)throw std::runtime_error(reason); }
void near(Complex value,Complex reference,double tolerance=3e-11) {
  require(std::isfinite(value.real()) && std::isfinite(value.imag()) &&
          std::abs(value-reference)<=tolerance*(1+std::abs(reference)),"harmonic value differs");
}
template<class F> void rejects(F&& f) {
  bool caught=false;
  try { f(); } catch(const std::invalid_argument&) { caught=true; }
  require(caught,"invalid harmonic input accepted");
}
long double factorial(int n) { long double result=1;for(int i=2;i<=n;++i)result*=i;return result; }

// Independent Rodrigues polynomial coefficients, not the implementation's
// three-term recurrence. d is the derivative order with respect to x.
long double polynomial_derivative(int l,int d,long double x) {
  long double result=0;
  for(int k=0;k<=l/2;++k) {
    const int power=l-2*k;
    if(power<d)continue;
    const long double coefficient=(k%2 ? -1.L : 1.L)*factorial(2*l-2*k)/
        (std::pow(2.L,l)*factorial(k)*factorial(l-k)*factorial(power-d));
    result+=coefficient*std::pow(x,power-d);
  }
  return result;
}
Complex oracle(int l,int m,double theta,double phi) {
  const int a=std::abs(m);
  const long double p=(a%2 ? -1.L : 1.L)*std::pow(std::sin((long double)theta),a)*
      polynomial_derivative(l,a,std::cos((long double)theta));
  const long double normalization=std::sqrt((2*l+1)/(4*std::numbers::pi_v<long double>)*
      factorial(l-a)/factorial(l+a));
  const double amplitude=static_cast<double>(normalization*p)*(m<0 && a%2 ? -1 : 1);
  return amplitude*std::exp(Complex(0,m*phi));
}
void values() {
  near(integral_harmonic(0,0,.71,-.4).value,1/std::sqrt(4*pi));
  near(integral_harmonic(1,0,.71,-.4).value,std::sqrt(3/(4*pi))*std::cos(.71));
  near(integral_harmonic(1,1,.71,-.4).value,
       -std::sqrt(3/(8*pi))*std::sin(.71)*std::exp(Complex(0,-.4)));
  for(int l=0;l<=10;++l)for(int m=-l;m<=l;++m)
    for(double theta:{.03,.4,1.2,pi/2,2.4,pi-.07})for(double phi:{-2.3,0.,.8,4.7}) {
      const auto integral=integral_harmonic(l,m,theta,phi);
      const auto orbital=orbital_harmonic(l,m,std::sin(theta),std::cos(theta),phi);
      near(integral.value,oracle(l,m,theta,phi));near(orbital.value,oracle(l,m,theta,phi));
      near(integral.polar_derivative,orbital.polar_derivative);
      near(integral.azimuthal_derivative,std::sin(theta)*orbital.azimuthal_over_sine);
      near(integral_harmonic(l,-m,theta,phi).value,(m%2 ? -1. : 1.)*std::conj(integral.value));
      if(m>=0) {
        const double x=std::cos(theta),st=std::sqrt((1-x)*(1+x));
        const auto leg=associated_legendre(l,m,x);
        const long double t=polynomial_derivative(l,m,x),dt=polynomial_derivative(l,m+1,x);
        near(leg.value,(m%2 ? -1. : 1.)*std::pow(st,m)*static_cast<double>(t),3e-10);
        const long double derivative=m ? m*std::pow((long double)st,m-1)*x*t-
            std::pow((long double)st,m+1)*dt : -st*dt;
        near(leg.polar_derivative,(m%2 ? -1. : 1.)*static_cast<double>(derivative),3e-9);
      }
    }
}
void derivatives() {
  const double h=1e-4;
  for(int l=0;l<=8;++l)for(int m=-l;m<=l;++m)
    for(double theta:{.19,.8,1.7,2.9})for(double phi:{-.9,.3,2.1}) {
      auto y=[&](double t,double p) {return integral_harmonic(l,m,t,p).value;};
      const auto sample=integral_harmonic(l,m,theta,phi);
      const auto dt=(-y(theta+2*h,phi)+8.*y(theta+h,phi)-8.*y(theta-h,phi)+y(theta-2*h,phi))/(12*h);
      const auto dp=(-y(theta,phi+2*h)+8.*y(theta,phi+h)-8.*y(theta,phi-h)+y(theta,phi-2*h))/(12*h);
      near(sample.polar_derivative,dt,2e-9);near(sample.azimuthal_derivative,dp,2e-9);
      near(sample.azimuthal_derivative,Complex(0,m)*sample.value);
      const auto orbital=orbital_harmonic(l,m,std::sin(theta),std::cos(theta),phi);
      near(orbital.polar_derivative,dt,2e-9);
      near(orbital.azimuthal_over_sine,dp/std::sin(theta),2e-9);
    }
}
void poles() {
  static_assert(!std::is_convertible_v<IntegralHarmonicSample,OrbitalHarmonicSample>);
  for(int l=1;l<=64;++l)for(double ct:{-1.,1.})for(double phi:{-.7,0.,1.9}) {
    const auto integral=integral_harmonic(l,1,ct>0 ? 0 : pi,phi);
    near(integral.polar_derivative,0);near(associated_legendre(l,1,ct).polar_derivative,0);
    const double derivative=(ct<0 && (l+1)%2 ? -1. : 1.)*l*(l+1)/2;
    const double normalization=std::sqrt((2.*l+1)/(4*pi*l*(l+1)));
    const auto phase=std::exp(Complex(0,phi));
    const auto expected=-normalization*ct*derivative*phase;
    const auto azimuthal=Complex(0,-normalization*derivative)*phase;
    const auto orbital=orbital_harmonic(l,1,0,ct,phi);
    near(orbital.value,0);near(orbital.polar_derivative,expected);
    near(orbital.azimuthal_over_sine,azimuthal);
    const auto negative=orbital_harmonic(l,-1,0,ct,phi);
    near(negative.polar_derivative,-std::conj(expected));
    near(negative.azimuthal_over_sine,-std::conj(azimuthal));
    // A transverse component whose squared size is lost by cos/acos still
    // contributes to Y and keeps the finite g-orbital derivatives.
    const auto almost=orbital_harmonic(l,1,1e-20,ct,phi);
    near(almost.value/1e-20,-normalization*derivative*phase);
    near(almost.polar_derivative,expected);near(almost.azimuthal_over_sine,azimuthal);
    for(int m:{0,l,-l}) {
      const auto limit=orbital_harmonic(l,m,0,ct,phi);
      require(std::isfinite(std::abs(limit.value)) && std::isfinite(std::abs(limit.polar_derivative)) &&
          std::isfinite(std::abs(limit.azimuthal_over_sine)),"nonfinite pole limit");
    }
  }
}
void orthogonality() {
  const GaussLegendreRule rule(18);const auto grid=rule.view();
  for(auto orders:std::array<std::array<int,4>,7>{{{0,0,0,0},{2,1,2,1},{4,-3,4,-3},
      {8,7,8,7},{4,2,4,-2},{3,1,5,1},{6,0,4,0}}}) {
    Complex integral=0;
    for(std::size_t i=0;i<grid.nodes().size();++i)for(int j=0;j<40;++j) {
      const double theta=std::acos(grid.nodes()[i]),phi=2*pi*j/40;
      integral+=grid.weights()[i]*2*pi/40*
          std::conj(integral_harmonic(orders[0],orders[1],theta,phi).value)*
          integral_harmonic(orders[2],orders[3],theta,phi).value;
    }
    near(integral,orders[0]==orders[2] && orders[1]==orders[3] ? 1. : 0.);
  }
  for(int l:{0,1,4,8,16,32,64})for(double theta:{0.,.7,1.4,pi}) {
    double sum=0;
    for(int m=-l;m<=l;++m)sum+=std::norm(integral_harmonic(l,m,theta,-.3).value);
    near(sum,(2.*l+1)/(4*pi));
  }
}
void invalid() {
  const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
  for(int l:{INT_MIN,-1,65,INT_MAX}) {
    rejects([&]{(void)associated_legendre(l,0,.1);});
    rejects([&]{(void)integral_harmonic(l,0,.1,0);});
    rejects([&]{(void)orbital_harmonic(l,0,0,1,0);});
  }
  for(int m:{INT_MIN,-5,5,INT_MAX}) {
    rejects([&]{(void)associated_legendre(4,m,.1);});
    rejects([&]{(void)integral_harmonic(4,m,.1,0);});
    rejects([&]{(void)orbital_harmonic(4,m,0,1,0);});
  }
  rejects([]{(void)associated_legendre(4,-1,.1);});
  for(double x:{nan,inf,-inf,-1.01,1.01})rejects([&]{(void)associated_legendre(4,1,x);});
  for(double t:{nan,inf,-inf,-.1,pi+.1})rejects([&]{(void)integral_harmonic(4,1,t,0);});
  for(double p:{nan,inf,-inf,std::numeric_limits<double>::max()}) {
    rejects([&]{(void)integral_harmonic(4,4,.1,p);});
    rejects([&]{(void)orbital_harmonic(4,4,0,1,p);});
  }
  for(double x:{nan,inf,-inf,-1.01,1.01})rejects([&]{(void)orbital_harmonic(4,1,0,x,0);});
  for(double s:{nan,inf,-inf,-.01,1.01})rejects([&]{(void)orbital_harmonic(4,1,s,1,0);});
  rejects([]{(void)orbital_harmonic(4,1,.5,.5,0);});
  rejects([]{(void)orbital_harmonic(4,1,0,0,0);});
}
void allocation() {
  allocations_before_failure=0;
  double sum=0;
  for(int l:{0,4,16,64})for(int m=-l;m<=l;++m) {
    auto sample=integral_harmonic(l,m,.7,-.3);auto copy=sample;sample.value=0;
    sum+=std::norm(copy.value)+std::norm(orbital_harmonic(l,m,std::sin(.7),std::cos(.7),-.3).value);
    if(m>=0)sum+=std::abs(associated_legendre(l,m,.4).value);
  }
  allocations_before_failure=-1;
  require(sum>0 && std::isfinite(sum),"allocation-free values invalid");
}
void concurrent() {
  std::array<std::future<void>,4> workers;
  for(int i=0;i<4;++i)workers[i]=std::async(std::launch::async,[i]{
    for(int repeat=0;repeat<10;++repeat)for(int l=0;l<=10;++l)for(int m=-l;m<=l;++m) {
      const double theta=.4+i*.4,phi=-.3+repeat*.2;
      near(integral_harmonic(l,m,theta,phi).value,oracle(l,m,theta,phi));
      near(orbital_harmonic(l,m,std::sin(theta),std::cos(theta),phi).value,oracle(l,m,theta,phi));
    }
  });
  for(auto& worker:workers)worker.get();
}
} // namespace
int main(int argc,char** argv) {
  try {
    if(argc!=2)throw std::invalid_argument("expected harmonics test case");
    const std::string name=argv[1];
    if(name=="values")values();else if(name=="derivatives")derivatives();else if(name=="poles")poles();
    else if(name=="orthogonality")orthogonality();else if(name=="invalid")invalid();
    else if(name=="allocation")allocation();else if(name=="concurrent")concurrent();
    else throw std::invalid_argument("unknown harmonics test case");
    std::cout<<"PASS harmonics "<<name<<'\n';return 0;
  } catch(const std::exception& error) {
    allocations_before_failure=-1;std::cerr<<error.what()<<'\n';return 1;
  }
}
