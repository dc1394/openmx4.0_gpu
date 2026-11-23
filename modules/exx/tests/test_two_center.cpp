// exx-two-center-gaussian and exx-two-center-symmetry: equations (11) and (12)
// against analytic values for Gaussian functions, and the symmetries of the
// two-centre blocks.
#include <openmx/exx/two_center.hpp>

#include <openmx/basis/quadrature.hpp>

#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <numbers>
#include <string>
#include <vector>

namespace exx_test {
namespace {
using namespace openmx::exx;
using Vector = std::array<double,3>;

const double pi = std::numbers::pi;

double length(const Vector& v) { return std::hypot(v[0],v[1],v[2]); }

// r^l exp(-exponent r^2) on a grid.
RadialFunction gaussian(const RadialGrid& grid, int l, double exponent)
{
  RadialFunction f{l,std::vector<double>(grid.size())};
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.radius(i);
    f.values[i] = std::pow(r,l)*std::exp(-exponent*r*r);
  }
  return f;
}

// Real spherical harmonics up to l = 2 from their Cartesian formulas, in the
// OpenMX order.
std::vector<double> cartesian_harmonics(const Vector& v)
{
  const double x = v[0], y = v[1], z = v[2], r2 = x*x+y*y+z*z, r = std::sqrt(r2);
  const double c1 = std::sqrt(3.0/(4.0*pi)), c2 = std::sqrt(15.0/(4.0*pi));
  return {0.5/std::sqrt(pi),
          c1*x/r, c1*y/r, c1*z/r,
          std::sqrt(5.0/(16.0*pi))*(3.0*z*z-r2)/r2, 0.5*c2*(x*x-y*y)/r2, c2*x*y/r2, c2*x*z/r2, c2*y*z/r2};
}

// A quadrature on the unit sphere that integrates products of harmonics up to
// a total degree of 2*points-1 exactly: Gauss-Legendre in cos(theta), uniform
// in phi.
struct Sphere {
  std::vector<Vector> directions;
  std::vector<double> weights;
};

Sphere sphere(int points)
{
  const openmx::basis::GaussLegendreRule rule(points);
  const auto view = rule.view();
  Sphere s;
  const int azimuthal = 2*points;
  for (int i = 0; i < points; ++i) {
    const double c = view.nodes()[static_cast<std::size_t>(i)], sine = std::sqrt(1.0-c*c);
    for (int j = 0; j < azimuthal; ++j) {
      const double phi = 2.0*pi*j/azimuthal;
      s.directions.push_back({sine*std::cos(phi),sine*std::sin(phi),c});
      s.weights.push_back(view.weights()[static_cast<std::size_t>(i)]*2.0*pi/azimuthal);
    }
  }
  return s;
}

// The matrix D with Y_l(Q u) = D Y_l(u) for a rotation Q, by quadrature.
std::vector<double> harmonic_rotation(const RealHarmonics& harmonics, int l, const std::array<Vector,3>& q,
                                      const Sphere& s)
{
  const int count = 2*l+1, total = (harmonics.maximum_degree()+1)*(harmonics.maximum_degree()+1);
  std::vector<double> d(static_cast<std::size_t>(count*count),0.0), y(static_cast<std::size_t>(total)),
      rotated(static_cast<std::size_t>(total));
  for (std::size_t k = 0; k < s.directions.size(); ++k) {
    const Vector& u = s.directions[k];
    const Vector v = {q[0][0]*u[0]+q[0][1]*u[1]+q[0][2]*u[2], q[1][0]*u[0]+q[1][1]*u[1]+q[1][2]*u[2],
                      q[2][0]*u[0]+q[2][1]*u[1]+q[2][2]*u[2]};
    harmonics.evaluate(u,y);
    harmonics.evaluate(v,rotated);
    for (int a = 0; a < count; ++a)
      for (int b = 0; b < count; ++b)
        d[static_cast<std::size_t>(a*count+b)] += s.weights[k]*rotated[static_cast<std::size_t>(l*l+a)]*
                                                  y[static_cast<std::size_t>(l*l+b)];
  }
  return d;
}

// Rotation by "angle" about a unit axis.
std::array<Vector,3> rotation(Vector axis, double angle)
{
  const double n = length(axis);
  for (double& x : axis) x /= n;
  const double c = std::cos(angle), s = std::sin(angle), t = 1.0-c;
  const double x = axis[0], y = axis[1], z = axis[2];
  return {{{t*x*x+c, t*x*y-s*z, t*x*z+s*y}, {t*x*y+s*z, t*y*y+c, t*y*z-s*x}, {t*x*z-s*y, t*y*z+s*x, t*z*z+c}}};
}

// A smooth normalized function that vanishes at the cutoff, with the behaviour
// r^l at the origin; "shape" selects one of several radial dependences.
RadialFunction model(const RadialGrid& grid, int l, double cutoff, int shape)
{
  RadialFunction f{l,std::vector<double>(grid.size(),0.0)};
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.radius(i);
    if (r >= cutoff) continue;
    const double x = r/cutoff, bump = (1.0-x*x)*(1.0-x*x)*(1.0-x*x);
    f.values[i] = std::pow(r,l)*bump*(1.0+0.7*shape*std::cos((1.3+shape)*x))*std::exp(-0.4*(1+shape)*x*x);
  }
  const double scale = 1.0/std::sqrt(radial_integral(grid,f.values,f.values));
  for (double& value : f.values) value *= scale;
  return f;
}

// F(d) of two s Gaussians for the three long-range kernels, with its first and
// second derivative: (g_a|v|g_b) = (pi^2/(a b))^(3/2) F(d).
struct Radial { double f, d1, d2; };

Radial pair_function(const Kernel& kernel, double mu, double d)
{
  // N(d) with F = N/d.
  const auto g = [](double c, double x) { return 2.0*c/std::sqrt(pi)*std::exp(-c*c*x*x); };   // d erf(c x)/dx
  const auto g1 = [&g](double c, double x) { return -2.0*c*c*x*g(c,x); };
  const double c = std::sqrt(mu);
  double n = 0.0, n1 = 0.0, n2 = 0.0;
  switch (kernel.kind) {
    case KernelKind::coulomb:
      n = std::erf(c*d); n1 = g(c,d); n2 = g1(c,d);
      break;
    case KernelKind::short_range: {
      const double omega = kernel.parameter, ce = std::sqrt(mu*omega*omega/(mu+omega*omega));
      n = std::erf(c*d)-std::erf(ce*d); n1 = g(c,d)-g(ce,d); n2 = g1(c,d)-g1(ce,d);
      break;
    }
    case KernelKind::truncated_coulomb: {
      const double rc = kernel.parameter;
      n = std::erf(c*d)+0.5*std::erf(c*(rc-d))-0.5*std::erf(c*(rc+d));
      n1 = g(c,d)-0.5*g(c,rc-d)-0.5*g(c,rc+d);
      n2 = g1(c,d)+0.5*g1(c,rc-d)-0.5*g1(c,rc+d);
      break;
    }
    default:
      throw std::runtime_error("no analytic pair function for this kernel");
  }
  return {n/d, n1/d-n/(d*d), n2/d-2.0*n1/(d*d)+2.0*n/(d*d*d)};
}

} // namespace

void two_center_gaussian()
{
  const RadialGrid grid{0.0,0.02,500};
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(16.0,800);
  const double a = 0.9, b = 0.6;

  // Equation (11): int r^(l+2) exp(-a r^2) j_l(kappa r) dr = sqrt(pi) kappa^l exp(-kappa^2/(4a)) / (2^(l+2) a^(l+3/2)).
  Worst transform("spherical Bessel transform of r^l exp(-a r^2)",1e-13);
  for (int l = 0; l <= 4; ++l) {
    const ReciprocalFunction t = bessel_transform(grid,gaussian(grid,l,a),reciprocal);
    for (std::size_t k = 0; k < reciprocal.size(); k += 7) {
      const double kappa = reciprocal.nodes()[k];
      transform.absolute(t.values[k],std::sqrt(pi)*std::pow(kappa,l)*std::exp(-kappa*kappa/(4.0*a))/
                                         (std::pow(2.0,l+2)*std::pow(a,l+1.5)));
    }
  }
  transform.finish();

  const std::vector<RadialFunction> radial = {gaussian(grid,0,a),gaussian(grid,1,a),gaussian(grid,0,b),gaussian(grid,1,b)};
  const std::vector<ReciprocalFunction> f = bessel_transform(grid,radial,reciprocal);
  const ReciprocalFunction &sa = f[0], &pa = f[1], &sb = f[2], &pb = f[3];
  const double mu = a*b/(a+b), prefactor = std::pow(pi*pi/(a*b),1.5);

  // The overlap of two Gaussians is elementary: (pi/(a+b))^(3/2) exp(-mu d^2).
  Worst overlap_s("overlap, s-s",1e-10), overlap_p("overlap, pairs with p",1e-10);
  Worst ss("Coulomb, short-range and truncated kernel, s-s",1e-10);
  Worst withp("Coulomb, short-range and truncated kernel, pairs with p",1e-10);
  for (const Vector& d : {Vector{0.7,-0.4,1.3},Vector{-1.1,0.3,0.2},Vector{0.0,0.0,2.4},Vector{1.9,0.0,0.0}}) {
    const double dist = length(d);
    {
      const TwoCenterIntegrator integrator(reciprocal,overlap_kernel(),1,1);
      const auto pair = integrator.pair(d);
      const double s0 = std::pow(pi/(a+b),1.5)*std::exp(-mu*dist*dist);
      double block[9];
      pair.block(sa,sb,{block,1});
      overlap_s.absolute(block[0],s0/(4.0*pi));
      // p on the first centre: z exp(-a r^2) = (1/(2a)) d/dA_z; d S/d A_i = 2 mu d_i S.
      pair.block(pa,sb,{block,3});
      for (int i = 0; i < 3; ++i) overlap_p.absolute(block[i],std::sqrt(3.0)/(4.0*pi)*2.0*mu*d[i]*s0/(2.0*a));
      pair.block(sa,pb,{block,3});
      for (int i = 0; i < 3; ++i) overlap_p.absolute(block[i],-std::sqrt(3.0)/(4.0*pi)*2.0*mu*d[i]*s0/(2.0*b));
      // d^2 S/(dA_i dB_j) = (2 mu delta_ij - 4 mu^2 d_i d_j) S.
      pair.block(pa,pb,{block,9});
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          overlap_p.absolute(block[3*i+j],3.0/(4.0*pi)*(2.0*mu*(i == j)-4.0*mu*mu*d[i]*d[j])*s0/(4.0*a*b));
    }
    for (const Kernel& kernel : {coulomb_kernel(),short_range_kernel(0.11),truncated_coulomb_kernel(3.0)}) {
      const TwoCenterIntegrator integrator(reciprocal,kernel,1,1);
      const auto pair = integrator.pair(d);
      const Radial F = pair_function(kernel,mu,dist);
      double block[9];
      pair.block(sa,sb,{block,1});
      ss.absolute(block[0],prefactor*F.f/(4.0*pi));
      // d/dA_i F(|d-A|) = -F' d_i/d and d/dB_i F = +F' d_i/d.
      pair.block(pa,sb,{block,3});
      for (int i = 0; i < 3; ++i)
        withp.absolute(block[i],std::sqrt(3.0)/(4.0*pi)*prefactor/(2.0*a)*(-d[i]/dist)*F.d1);
      pair.block(sa,pb,{block,3});
      for (int i = 0; i < 3; ++i)
        withp.absolute(block[i],std::sqrt(3.0)/(4.0*pi)*prefactor/(2.0*b)*(d[i]/dist)*F.d1);
      // d^2/(dA_i dB_j) F = -[(F'' - F'/d) d_i d_j/d^2 + delta_ij F'/d].
      pair.block(pa,pb,{block,9});
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          const double hessian = (F.d2-F.d1/dist)*d[i]*d[j]/(dist*dist)+(i == j ? F.d1/dist : 0.0);
          withp.absolute(block[3*i+j],3.0/(4.0*pi)*prefactor/(4.0*a*b)*(-hessian));
        }
    }
  }
  overlap_s.finish();
  overlap_p.finish();
  ss.finish();
  withp.finish();

  // Zero displacement: only L = 0 remains, and the block is diagonal.
  // int r^2 exp(-(a+b) r^2) dr = sqrt(pi)/(4 (a+b)^(3/2)); for p the integrand has r^4.
  Worst onsite("overlap at zero displacement",1e-12);
  {
    const TwoCenterIntegrator integrator(reciprocal,overlap_kernel(),1,1);
    const auto pair = integrator.pair({0.0,0.0,0.0});
    double block[9];
    pair.block(sa,sb,{block,1});
    onsite.absolute(block[0],std::sqrt(pi)/(4.0*std::pow(a+b,1.5)));
    pair.block(pa,pb,{block,9});
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        onsite.absolute(block[3*i+j],i == j ? 3.0*std::sqrt(pi)/(8.0*std::pow(a+b,2.5)) : 0.0);
    pair.block(pa,sb,{block,3});
    for (int i = 0; i < 3; ++i) onsite.absolute(block[i],0.0);
  }
  onsite.finish();
}

void two_center_symmetry()
{
  // Real harmonics: the Cartesian formulas up to l = 2 fix the order and the
  // signs; orthonormality on the sphere for all degrees.
  const int maximum = 8;
  const RealHarmonics harmonics(maximum);
  const Sphere s = sphere(12);
  const int total = (maximum+1)*(maximum+1);
  std::vector<double> y(static_cast<std::size_t>(total));
  Worst cartesian("real harmonics against Cartesian formulas, l <= 2",1e-14);
  for (const Vector& v : {Vector{0.3,-0.5,0.8},Vector{-1.2,0.4,-0.7},Vector{0.0,0.0,2.0},Vector{0.0,0.0,-1.0},
                          Vector{1.0,0.0,0.0},Vector{0.0,-3.0,0.0},Vector{2.0,1.0,0.0}}) {
    harmonics.evaluate(v,y);
    const std::vector<double> reference = cartesian_harmonics(v);
    for (std::size_t i = 0; i < reference.size(); ++i) cartesian.absolute(y[i],reference[i]);
  }
  cartesian.finish();
  // For l >= 3 the rows are m = 0, cos(phi), sin(phi), cos(2 phi), sin(2 phi), ...
  // A turn about the z axis mixes the two rows of one m like a plane rotation.
  // Near the pole both rows carry a positive factor, which fixes the signs.
  Worst azimuth("azimuthal dependence of the rows, l >= 3",1e-13);
  {
    const double phi = 0.37, shift = 0.61;
    std::vector<double> y0(static_cast<std::size_t>(total));
    harmonics.evaluate({0.6*std::cos(phi),0.6*std::sin(phi),0.8},y0);
    harmonics.evaluate({0.6*std::cos(phi+shift),0.6*std::sin(phi+shift),0.8},y);
    for (int l = 3; l <= maximum; ++l) {
      azimuth.absolute(y[static_cast<std::size_t>(l*l)],y0[static_cast<std::size_t>(l*l)]);
      for (int m = 1; m <= l; ++m) {
        const double c0 = y0[static_cast<std::size_t>(l*l+2*m-1)], s0 = y0[static_cast<std::size_t>(l*l+2*m)];
        azimuth.absolute(y[static_cast<std::size_t>(l*l+2*m-1)],c0*std::cos(m*shift)-s0*std::sin(m*shift));
        azimuth.absolute(y[static_cast<std::size_t>(l*l+2*m)],s0*std::cos(m*shift)+c0*std::sin(m*shift));
      }
    }
    const double polar = 0.1;
    harmonics.evaluate({std::sin(polar)*std::cos(phi),std::sin(polar)*std::sin(phi),std::cos(polar)},y0);
    for (int l = 3; l <= maximum; ++l) {
      require(y0[static_cast<std::size_t>(l*l)] > 0.0,"the m = 0 row must be positive near the pole");
      for (int m = 1; m <= l; ++m)
        azimuth.absolute(std::atan2(y0[static_cast<std::size_t>(l*l+2*m)],y0[static_cast<std::size_t>(l*l+2*m-1)]),
                         std::remainder(m*phi,2.0*pi));
    }
  }
  azimuth.finish();
  Worst orthonormal("orthonormality of the real harmonics",1e-12);
  {
    std::vector<double> gram(static_cast<std::size_t>(total*total),0.0);
    for (std::size_t k = 0; k < s.directions.size(); ++k) {
      harmonics.evaluate(s.directions[k],y);
      for (int i = 0; i < total; ++i)
        for (int j = 0; j < total; ++j)
          gram[static_cast<std::size_t>(i*total+j)] += s.weights[k]*y[static_cast<std::size_t>(i)]*y[static_cast<std::size_t>(j)];
    }
    for (int i = 0; i < total; ++i)
      for (int j = 0; j < total; ++j) orthonormal.absolute(gram[static_cast<std::size_t>(i*total+j)],i == j ? 1.0 : 0.0);
  }
  orthonormal.finish();

  // Real Gaunt coefficients against the quadrature of three real harmonics.
  const int left = 4, right = 4;
  const RealGauntTable gaunt(left,right);
  Worst theta("real Gaunt coefficients against quadrature",1e-12);
  {
    std::vector<std::vector<double>> samples(s.directions.size(),std::vector<double>(static_cast<std::size_t>(total)));
    for (std::size_t k = 0; k < s.directions.size(); ++k) harmonics.evaluate(s.directions[k],samples[k]);
    for (int l1 = 0; l1 <= left; ++l1)
      for (int l2 = 0; l2 <= right; ++l2)
        for (int L = 0; L <= l1+l2; ++L) {
          const bool stored = L >= std::abs(l1-l2) && (l1+l2+L)%2 == 0;
          for (int r1 = 0; r1 <= 2*l1; ++r1)
            for (int r2 = 0; r2 <= 2*l2; ++r2)
              for (int R = 0; R <= 2*L; ++R) {
                double sum = 0.0;
                for (std::size_t k = 0; k < s.directions.size(); ++k)
                  sum += s.weights[k]*samples[k][static_cast<std::size_t>(l1*l1+r1)]*
                         samples[k][static_cast<std::size_t>(l2*l2+r2)]*samples[k][static_cast<std::size_t>(L*L+R)];
                theta.absolute(stored ? gaunt.value(l1,r1,l2,r2,L,R) : 0.0,sum);
              }
        }
  }
  theta.finish();

  // Two-centre blocks of model functions with compact support.
  const RadialGrid grid{0.0,0.01,500};
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(40.0,1000);
  std::vector<RadialFunction> radial;
  for (int l = 0; l <= 4; ++l) radial.push_back(model(grid,l,4.6,l%3));
  for (int l = 0; l <= 4; ++l) radial.push_back(model(grid,l,3.9,(l+1)%3));
  const std::vector<ReciprocalFunction> f = bessel_transform(grid,radial,reciprocal);
  const Vector d = {1.7,-2.3,0.9};
  const std::array<Vector,3> q = rotation({0.4,-1.0,0.7},1.234);
  const Vector qd = {q[0][0]*d[0]+q[0][1]*d[1]+q[0][2]*d[2], q[1][0]*d[0]+q[1][1]*d[1]+q[1][2]*d[2],
                     q[2][0]*d[0]+q[2][1]*d[1]+q[2][2]*d[2]};
  const Vector minus = {-d[0],-d[1],-d[2]};
  std::vector<std::vector<double>> wigner;
  for (int l = 0; l <= 4; ++l) wigner.push_back(harmonic_rotation(harmonics,l,q,s));
  Worst unitary("rotation matrices of the real harmonics are orthogonal",1e-12);
  for (int l = 0; l <= 4; ++l) {
    const int n = 2*l+1;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) sum += wigner[static_cast<std::size_t>(l)][static_cast<std::size_t>(i*n+k)]*
                                           wigner[static_cast<std::size_t>(l)][static_cast<std::size_t>(j*n+k)];
        unitary.absolute(sum,i == j ? 1.0 : 0.0);
      }
  }
  unitary.finish();

  Worst transpose("V_ab(R) = V_ba(-R)^T",1e-10), rotated("invariance under a rotation of the coordinate system",1e-10);
  double largest = 0.0;
  for (const Kernel& kernel : {overlap_kernel(),coulomb_kernel(),short_range_kernel(0.11),truncated_coulomb_kernel(5.0)}) {
    const TwoCenterIntegrator integrator(reciprocal,kernel,4,4);
    const auto forward = integrator.pair(d), backward = integrator.pair(minus), turned = integrator.pair(qd);
    for (int lp = 0; lp <= 4; ++lp)
      for (int lq = 0; lq <= 4; ++lq) {
        const ReciprocalFunction &p = f[static_cast<std::size_t>(lp)], &g = f[static_cast<std::size_t>(5+lq)];
        const int rows = 2*lp+1, columns = 2*lq+1;
        std::vector<double> v(static_cast<std::size_t>(rows*columns)), vt(v.size()), vr(v.size());
        forward.block(p,g,v);
        backward.block(g,p,vt);
        turned.block(p,g,vr);
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j < columns; ++j) {
            transpose.absolute(v[static_cast<std::size_t>(i*columns+j)],vt[static_cast<std::size_t>(j*rows+i)]);
            // V(Q d) = D_lp(Q) V(d) D_lq(Q)^T.
            double sum = 0.0;
            for (int k = 0; k < rows; ++k)
              for (int m = 0; m < columns; ++m)
                sum += wigner[static_cast<std::size_t>(lp)][static_cast<std::size_t>(i*rows+k)]*
                       v[static_cast<std::size_t>(k*columns+m)]*
                       wigner[static_cast<std::size_t>(lq)][static_cast<std::size_t>(j*columns+m)];
            rotated.absolute(vr[static_cast<std::size_t>(i*columns+j)],sum);
            largest = std::fmax(largest,std::fabs(v[static_cast<std::size_t>(i*columns+j)]));
          }
      }
  }
  transpose.finish();
  rotated.finish();
  std::printf("largest element of the compared blocks: %.3f\n",largest);
  require(largest > 0.1,"the blocks of the symmetry test are too small to be meaningful");

  // Invalid arguments.
  bool thrown = false;
  try { (void)ReciprocalGrid::uniform(1.0,0); } catch (const std::invalid_argument&) { thrown = true; }
  require(thrown,"a reciprocal grid without intervals must be rejected");
  thrown = false;
  try { (void)RealGauntTable(12,8); } catch (const std::invalid_argument&) { thrown = true; }
  require(thrown,"a real Gaunt table beyond the supported degrees must be rejected");
  thrown = false;
  try { (void)bessel_transform(RadialGrid{0.0,0.1,2},radial[0],reciprocal); } catch (const std::invalid_argument&) { thrown = true; }
  require(thrown,"a radial grid with fewer than three intervals must be rejected");
}

} // namespace exx_test
