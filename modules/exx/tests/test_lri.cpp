// Tests of the local RI (lri.hpp):
//   exx-three-center-gaussian  the right-hand side of equation (7) for s and p
//                              Gaussian functions against analytic values
//   exx-lri-identity           functions inside the span of the ABFs are
//                              expanded with their own coefficients
//   exx-lri-convergence        the self-interaction of the expanded products
//                              as the auxiliary basis grows (a record)
#include <openmx/exx/abf.hpp>
#include <openmx/exx/lri.hpp>

#include "overlap_fixture.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace exx_test {

using namespace openmx::exx;

namespace {

using Vec = std::array<double,3>;
constexpr double pi = std::numbers::pi;

// ---------------------------------------------------------------------------
// Analytic three-centre Coulomb integrals of Gaussian functions.

double boys0(double x)
{
  if (x < 1e-6) return 1.0-x/3.0+x*x/10.0;
  return 0.5*std::sqrt(pi/x)*std::erf(std::sqrt(x));
}

// int int exp(-a|r-A|^2) exp(-c|r-C|^2) (1/|r-r'|) exp(-p|r'-B|^2) d3r d3r'.
double coulomb3(double a, const Vec& A, double c, const Vec& C, double p, const Vec& B)
{
  const double gamma = a+c;
  double ac = 0.0, rb = 0.0;
  for (std::size_t x = 0; x < 3; ++x) {
    ac += (A[x]-C[x])*(A[x]-C[x]);
    const double r = (a*A[x]+c*C[x])/gamma-B[x];
    rb += r*r;
  }
  return 2.0*std::pow(pi,2.5)/(gamma*p*std::sqrt(gamma+p))*std::exp(-a*c/gamma*ac)*boys0(gamma*p/(gamma+p)*rb);
}

// A function r^l exp(-exponent r^2) Y_lm with l = 0 or 1.
struct Gaussian {
  int l = 0;
  double exponent = 1.0;
};

RadialFunction sample(const RadialGrid& grid, const Gaussian& g)
{
  RadialFunction f{g.l,std::vector<double>(grid.size())};
  for (std::size_t i = 0; i < grid.size(); ++i) {
    const double r = grid.radius(i);
    f.values[i] = std::pow(r,g.l)*std::exp(-g.exponent*r*r);
  }
  return f;
}

// exp(-a r^2) Y_00 = exp(-a r^2)/sqrt(4 pi), and
// r exp(-a r^2) Y_1x = sqrt(3/(4 pi)) x exp(-a r^2) = sqrt(3/(4 pi))/(2a) d/dA_x exp(-a|r-A|^2).
double prefactor(const Gaussian& g)
{
  return g.l == 0 ? 1.0/std::sqrt(4.0*pi) : std::sqrt(3.0/(4.0*pi))/(2.0*g.exponent);
}

// Applies d/d(centre[level])_direction for every level with a direction >= 0,
// by central differences of sixth order.
double differentiate(std::array<Vec,3> centres, const std::array<int,3>& direction, std::size_t level,
                     const std::function<double(const std::array<Vec,3>&)>& f)
{
  if (level == 3) return f(centres);
  if (direction[level] < 0) return differentiate(centres,direction,level+1,f);
  constexpr double h = 0.02;
  constexpr double weight[3] = {45.0,-9.0,1.0};
  const std::size_t x = static_cast<std::size_t>(direction[level]);
  const double origin = centres[level][x];
  double sum = 0.0;
  for (int k = 1; k <= 3; ++k) {
    centres[level][x] = origin+k*h;
    const double plus = differentiate(centres,direction,level+1,f);
    centres[level][x] = origin-k*h;
    const double minus = differentiate(centres,direction,level+1,f);
    sum += weight[k-1]*(plus-minus);
  }
  return sum/(60.0*h);
}

// (phi_mu phi_lambda | 1/r | P) for components of Gaussian functions; a p
// function has its components in the order x, y, z.
double expected(const Gaussian& mu, int row_mu, const Vec& A, const Gaussian& lambda, int row_lambda, const Vec& C,
                const Gaussian& P, int row_P, const Vec& B)
{
  const std::array<int,3> direction{mu.l == 1 ? row_mu : -1,lambda.l == 1 ? row_lambda : -1,P.l == 1 ? row_P : -1};
  return prefactor(mu)*prefactor(lambda)*prefactor(P)*
         differentiate({A,C,B},direction,0,[&](const std::array<Vec,3>& x) {
           return coulomb3(mu.exponent,x[0],lambda.exponent,x[1],P.exponent,x[2]);
         });
}

// The radial part of the Coulomb potential of r^l exp(-p r^2) Y_lm.
double gaussian_potential(int l, double p, double r)
{
  const double e = std::exp(-p*r*r);
  // I_l = int_0^r t^(2l+2) exp(-p t^2) dt.
  double integral = std::sqrt(pi)*std::erf(std::sqrt(p)*r)/(4.0*std::pow(p,1.5))-r*e/(2.0*p);
  for (int k = 1; k <= l; ++k) integral = ((2*k+1)*integral-std::pow(r,2*k+1)*e)/(2.0*p);
  return 4.0*pi/(2*l+1)*(integral/std::pow(r,l+1)+std::pow(r,l)*e/(2.0*p));
}

struct Component {
  std::size_t shell = 0;
  int row = 0;
};

std::vector<Component> components(const std::vector<Gaussian>& shells)
{
  std::vector<Component> list;
  for (std::size_t s = 0; s < shells.size(); ++s)
    for (int row = 0; row <= 2*shells[s].l; ++row) list.push_back({s,row});
  return list;
}

template <class Call>
void expect_throw(const std::string& what, Call call)
{
  bool thrown = false;
  try { call(); } catch (const std::invalid_argument&) { thrown = true; } catch (const std::runtime_error&) { thrown = true; }
  require(thrown,"no exception for "+what);
}

// Deterministic numbers in (-1, 1).
double pseudo(std::size_t index)
{
  const double x = std::sin(12.9898*static_cast<double>(index+1))*43758.5453;
  return 2.0*(x-std::floor(x))-1.0;
}

// The sub-matrix and the sub-right-hand side of the rows in "keep".
void restrict_system(const std::vector<double>& g, const std::vector<double>& b, std::size_t n,
                     const std::vector<std::size_t>& keep, std::vector<double>& g_out, std::vector<double>& b_out)
{
  const std::size_t m = b.size()/n, k = keep.size();
  g_out.resize(k*k);
  b_out.resize(k*m);
  for (std::size_t i = 0; i < k; ++i) {
    for (std::size_t j = 0; j < k; ++j) g_out[i*k+j] = g[keep[i]*n+keep[j]];
    std::copy(b.begin()+static_cast<std::ptrdiff_t>(keep[i]*m),b.begin()+static_cast<std::ptrdiff_t>((keep[i]+1)*m),
              b_out.begin()+static_cast<std::ptrdiff_t>(i*m));
  }
}

double dot(const std::vector<double>& a, const std::vector<double>& b)
{
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i]*b[i];
  return sum;
}

int largest_angular(std::span<const RadialFunction> functions)
{
  int maximum = 0;
  for (const RadialFunction& f : functions) maximum = std::max(maximum,f.angular);
  return maximum;
}

} // namespace

// ---------------------------------------------------------------------------

void three_center_gaussian()
{
  const RadialGrid grid{0.0,0.005,2000};
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(16.0,800);

  // Equation (13) for r^l exp(-p r^2).  The rule of coulomb_potential() is of
  // fourth order; the deviation is 2e-9 with a step of 0.005 bohr and
  // 1.3e-10 with the step used here.
  {
    const RadialGrid fine{0.0,0.0025,4000};
    Worst potential("Coulomb potential of r^l exp(-p r^2), l = 0..3",1e-9);
    Worst origin("the same at the origin",1e-9);
    for (int l = 0; l <= 3; ++l)
      for (const double p : {0.6,1.3}) {
        RadialFunction f{l,std::vector<double>(fine.size())};
        for (std::size_t i = 0; i < fine.size(); ++i) {
          const double r = fine.radius(i);
          f.values[i] = std::pow(r,l)*std::exp(-p*r*r);
        }
        const RadialFunction w = coulomb_potential(fine,f);
        require(w.angular == l && w.values.size() == fine.size(),"the potential has the wrong shape");
        // Below 0.5 bohr the closed form loses digits by cancellation.
        for (std::size_t i = 200; i < fine.size(); i += 7) potential.absolute(w.values[i],gaussian_potential(l,p,fine.radius(i)));
        origin.absolute(w.values[0],l == 0 ? 2.0*pi/p : 0.0);
      }
    potential.finish();
    origin.finish();
  }

  const std::vector<Gaussian> orbitals_a{{0,0.9},{1,0.7}}, abfs_a{{0,0.8},{1,0.6}};
  const std::vector<Gaussian> orbitals_c{{0,1.1},{1,0.5}}, abfs_c{{0,0.65},{1,0.95}};
  const auto table = [&](const std::vector<Gaussian>& orbitals, const std::vector<Gaussian>& abfs) {
    std::vector<RadialFunction> o, f;
    for (const Gaussian& g : orbitals) o.push_back(sample(grid,g));
    for (const Gaussian& g : abfs) f.push_back(sample(grid,g));
    return SpeciesTable(grid,std::move(o),std::move(f),reciprocal);
  };
  const SpeciesTable a = table(orbitals_a,abfs_a), c = table(orbitals_c,abfs_c);
  require(a.orbital_count() == 4 && a.abf_count() == 4 && c.orbital_count() == 4 && c.abf_count() == 4,
          "wrong number of components of the Gaussian species");
  const LriSolver solver(reciprocal,1,1);
  const std::vector<Component> mu_list = components(orbitals_a), lambda_list = components(orbitals_c);
  const std::vector<Component> p_list = components(abfs_a), q_list = components(abfs_c);
  const Vec A{0.0,0.0,0.0};

  Worst first("three-centre integrals, ABF on the first atom",1e-8);
  Worst second("three-centre integrals, ABF on the second atom",1e-8);
  double largest = 0.0;
  for (const Vec& d : {Vec{0.7,-0.4,1.1},Vec{-1.3,0.9,0.2},Vec{0.0,0.0,2.1}}) {
    const std::vector<double> b = solver.right_hand_side(a,c,d);
    require(b.size() == 8*4*4,"the right-hand side has the wrong size");
    for (std::size_t mu = 0; mu < 4; ++mu)
      for (std::size_t lambda = 0; lambda < 4; ++lambda) {
        const Gaussian& gm = orbitals_a[mu_list[mu].shell];
        const Gaussian& gl = orbitals_c[lambda_list[lambda].shell];
        for (std::size_t p = 0; p < 4; ++p) {
          const double reference = expected(gm,mu_list[mu].row,A,gl,lambda_list[lambda].row,d,
                                            abfs_a[p_list[p].shell],p_list[p].row,A);
          first.absolute(b[(p*4+mu)*4+lambda],reference);
          largest = std::fmax(largest,std::fabs(reference));
        }
        for (std::size_t q = 0; q < 4; ++q) {
          const double reference = expected(gm,mu_list[mu].row,A,gl,lambda_list[lambda].row,d,
                                            abfs_c[q_list[q].shell],q_list[q].row,d);
          second.absolute(b[((4+q)*4+mu)*4+lambda],reference);
          largest = std::fmax(largest,std::fabs(reference));
        }
      }
  }
  first.finish();
  second.finish();
  require(largest > 0.1,"the three-centre integrals of the test are too small to be a check");

  // All three functions on one atom.
  {
    Worst onsite("three-centre integrals on one atom",1e-8);
    const std::span<const double> b = a.onsite_right_hand_side();
    for (std::size_t mu = 0; mu < 4; ++mu)
      for (std::size_t lambda = 0; lambda < 4; ++lambda)
        for (std::size_t p = 0; p < 4; ++p)
          onsite.absolute(b[(p*4+mu)*4+lambda],
                          expected(orbitals_a[mu_list[mu].shell],mu_list[mu].row,A,
                                   orbitals_a[mu_list[lambda].shell],mu_list[lambda].row,A,
                                   abfs_a[p_list[p].shell],p_list[p].row,A));
    onsite.finish();
  }

  // The metric on one atom (radial integrals) against equation (12) at zero
  // displacement, and against the closed forms of two s and two p functions.
  {
    Worst metric("metric on one atom against equation (12)",1e-8);
    const TwoCenterIntegrator coulomb(reciprocal,coulomb_kernel(),1,1);
    const std::vector<double> k = two_center_matrix(coulomb.pair({0.0,0.0,0.0}),a.abf_transforms(),a.abf_transforms());
    const std::span<const double> r = a.onsite_metric();
    for (std::size_t i = 0; i < 16; ++i) metric.absolute(r[i],k[i]);
    metric.finish();
    Worst closed("metric on one atom against the closed forms",1e-8);
    // Two s functions exp(-p r^2) Y_00 on one centre: 2 pi^(5/2)/(p^2 sqrt(2p))/(4 pi).
    const double p = abfs_a[0].exponent;
    closed.absolute(r[0],2.0*std::pow(pi,2.5)/(p*p*std::sqrt(2.0*p))/(4.0*pi));
    // Two p_x functions with exponent q on one centre: the derivative of the
    // two-centre integral of s functions, 2 pi^(5/2)/(q^2 sqrt(2q)) F0(q|A-B|^2/2),
    // with respect to A_x and B_x.
    const double q = abfs_a[1].exponent;
    const auto ss = [q](const std::array<Vec,3>& x) {
      double d2 = 0.0;
      for (std::size_t i = 0; i < 3; ++i) d2 += (x[0][i]-x[2][i])*(x[0][i]-x[2][i]);
      return 2.0*std::pow(pi,2.5)/(q*q*std::sqrt(2.0*q))*boys0(0.5*q*d2);
    };
    const double pp = std::pow(std::sqrt(3.0/(4.0*pi))/(2.0*q),2)*differentiate({A,A,A},{0,-1,0},0,ss);
    for (std::size_t m = 1; m < 4; ++m) closed.absolute(r[m*4+m],pp);
    closed.finish();
  }
}

// ---------------------------------------------------------------------------

void lri_identity(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);

  // 1. One atom.  Model orbitals: a constant s function, and for every
  //    angular momentum L <= 4 one function sqrt(4 pi) sum_n c_n f_n over the
  //    ABFs of that L.  The product of the constant function with component M
  //    of the other one is sum_n c_n f_n Y_LM, so equation (7) must return c_n
  //    for the ABF (n, M) and zero elsewhere; T[0] holds half of that.
  {
    Worst onsite("coefficients of functions inside the span, one atom",1e-10);
    const ReciprocalGrid coarse = ReciprocalGrid::uniform(4.0,8);
    for (const FixtureSpecies& species : fixture.species) {
      const RadialGrid grid{0.0,species.cutoff/fixture.radial_intervals,fixture.radial_intervals};
      const AuxiliaryBasis basis = make_auxiliary_basis(grid,sample_shells(species,grid),1e-4);
      const int top = std::min(4,largest_angular(basis.functions));
      std::vector<RadialFunction> model{RadialFunction{0,std::vector<double>(grid.size(),1.0)}};
      std::vector<double> coefficient(basis.functions.size());
      for (std::size_t n = 0; n < coefficient.size(); ++n) coefficient[n] = 0.3+0.7*pseudo(n);
      for (int angular = 0; angular <= top; ++angular) {
        RadialFunction f{angular,std::vector<double>(grid.size(),0.0)};
        for (std::size_t n = 0; n < basis.functions.size(); ++n) {
          if (basis.functions[n].angular != angular) continue;
          for (std::size_t i = 0; i < grid.size(); ++i)
            f.values[i] += std::sqrt(4.0*pi)*coefficient[n]*basis.functions[n].values[i];
        }
        model.push_back(std::move(f));
      }
      const SpeciesTable table(grid,model,basis.functions,coarse);
      const LriSolver solver(coarse,4,table.maximum_abf_angular());
      LriReport report;
      const std::vector<double> t = solver.onsite(table,&report);
      require(report.systems == 1 && report.eigen_decompositions == 0,"the metric of one atom needed the eigen-decomposition");
      const std::size_t u = table.orbital_count(), np = table.abf_count();
      require(t.size() == np*u*u,"the T block of one atom has the wrong size");
      for (std::size_t shell = 1; shell < model.size(); ++shell) {
        const int angular = model[shell].angular;
        for (int m = 0; m <= 2*angular; ++m) {
          const std::size_t lambda = table.orbital_offsets()[shell]+static_cast<std::size_t>(m);
          for (std::size_t n = 0; n < basis.functions.size(); ++n)
            for (int row = 0; row <= 2*basis.functions[n].angular; ++row) {
              const std::size_t p = table.abf_offsets()[n]+static_cast<std::size_t>(row);
              const double reference = basis.functions[n].angular == angular && row == m ? coefficient[n] : 0.0;
              onsite.absolute(2.0*t[(p*u+0)*u+lambda],reference);
              onsite.absolute(2.0*t[(p*u+lambda)*u+0],reference);
            }
        }
      }
    }
    onsite.finish();
  }

  // 2. Two atoms, with the PAOs and the ABFs of the fixture on coarser grids
  //    (the checks below do not depend on the accuracy of the quadrature).
  {
    const int intervals = 300;
    const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(24.0,240);
    std::vector<SpeciesTable> tables;
    int maximum_orbital = 0, maximum_abf = 0;
    for (const FixtureSpecies& species : fixture.species) {
      const RadialGrid grid{0.0,species.cutoff/intervals,intervals};
      const std::vector<RadialFunction> orbitals = sample_shells(species,grid);
      const AuxiliaryBasis basis = make_auxiliary_basis(grid,orbitals,1e-4);
      tables.emplace_back(grid,orbitals,basis.functions,reciprocal);
      maximum_orbital = std::max(maximum_orbital,tables.back().maximum_orbital_angular());
      maximum_abf = std::max(maximum_abf,tables.back().maximum_abf_angular());
    }
    const LriSolver solver(reciprocal,maximum_orbital,maximum_abf);

    Worst recovered("density recovered from G c, two atoms (Coulomb norm)",1e-10);
    Worst elementwise("coefficients recovered from G c, two atoms",1e-8);
    Worst separated("coefficients recovered from G c, distant atoms",1e-10);
    Worst swapped("the pair solved from the other atom",1e-10);
    Worst rotated("self-interaction of the fit after a rotation (relative)",1e-10);
    Worst symmetric("symmetry of the metric of a pair",1e-14);
    // Four pairs at bond distances and two pairs whose functions barely
    // overlap.  The metric of the distant pairs is as well conditioned as that
    // of one atom, so their coefficients must come back to 1e-10 one by one.
    struct Case { std::size_t a, c; Vec d; bool distant; };
    const std::vector<Case> cases{{0,1,{1.9,-1.2,2.6},false},{1,2,{-0.9,1.5,1.0},false},
                                  {0,0,{2.5,2.6,-2.4},false},{2,2,{0.3,-1.2,0.6},false},
                                  {0,1,{5.9,-4.2,5.6},true},{1,2,{-4.9,5.5,3.0},true}};
    // A rotation by 0.83 rad about (1,2,-2)/3.
    const Vec axis{1.0/3.0,2.0/3.0,-2.0/3.0};
    const double angle = 0.83, cosine = std::cos(angle), sine = std::sin(angle);
    const auto rotate = [&](const Vec& v) {
      const double along = axis[0]*v[0]+axis[1]*v[1]+axis[2]*v[2];
      const Vec cross{axis[1]*v[2]-axis[2]*v[1],axis[2]*v[0]-axis[0]*v[2],axis[0]*v[1]-axis[1]*v[0]};
      Vec out{};
      for (std::size_t x = 0; x < 3; ++x) out[x] = cosine*v[x]+sine*cross[x]+(1.0-cosine)*along*axis[x];
      return out;
    };
    for (const Case& item : cases) {
      const SpeciesTable& a = tables[item.a];
      const SpeciesTable& c = tables[item.c];
      const std::size_t na = a.abf_count(), nc = c.abf_count(), n = na+nc;
      const std::size_t ua = a.orbital_count(), uc = c.orbital_count();
      const std::vector<double> g = solver.metric(a,c,item.d);
      require(g.size() == n*n,"the metric of a pair has the wrong size");
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) symmetric.absolute(g[i*n+j],g[j*n+i]);

      // b = G c for three coefficient vectors.  The scaled metric of a pair
      // has eigenvalues down to 2e-6 (the ABFs of two bonded atoms are nearly
      // dependent), so single coefficients come back with nine to ten digits;
      // the density they describe comes back to rounding.  Its error is
      // measured in the Coulomb norm.  Coefficients are in units of the norm
      // of their ABF, 1/sqrt(G_PP), because these norms differ by orders of
      // magnitude.
      const std::size_t m = 3;
      std::vector<double> coefficient(n*m), b(n*m,0.0);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = 0; k < m; ++k) coefficient[i*m+k] = pseudo(i*m+k+17*n)/std::sqrt(g[i*n+i]);
      for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t k = 0; k < m; ++k) b[i*m+k] += g[i*n+j]*coefficient[j*m+k];
      LriReport report;
      solve_metric(g,n,b,solver.options(),&report);
      require(report.systems == 1 && report.eigen_decompositions == 0 && report.dropped == 0,
              "the metric of a pair needed the eigen-decomposition");
      require(report.smallest_pivot > 1e-4 && report.smallest_pivot <= 1.0,"unexpected pivot of the metric of a pair");
      for (std::size_t k = 0; k < m; ++k) {
        double norm = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          const double di = b[i*m+k]-coefficient[i*m+k];
          (item.distant ? separated : elementwise).add(std::fabs(di)*std::sqrt(g[i*n+i]));
          for (std::size_t j = 0; j < n; ++j) norm += di*g[i*n+j]*(b[j*m+k]-coefficient[j*m+k]);
        }
        recovered.add(std::sqrt(std::fabs(norm)));
      }

      // The same pair seen from the second atom.
      const PairCoefficients forward = solver.pair(a,c,item.d);
      const PairCoefficients backward = solver.pair(c,a,{-item.d[0],-item.d[1],-item.d[2]});
      require(forward.first.size() == na*ua*uc && forward.second.size() == nc*uc*ua,"a T block has the wrong size");
      for (std::size_t i = 0; i < forward.first.size(); ++i) swapped.absolute(forward.first[i],backward.second[i]);
      for (std::size_t i = 0; i < forward.second.size(); ++i) swapped.absolute(forward.second[i],backward.first[i]);

      // sum over all products of (fit|g|fit) = b^T G^-1 b does not depend on
      // the orientation of the pair.
      const auto energy = [&](const Vec& d) {
        const std::vector<double> rhs = solver.right_hand_side(a,c,d);
        std::vector<double> x = rhs;
        solve_metric(solver.metric(a,c,d),n,x,solver.options());
        return dot(rhs,x);
      };
      const double reference = energy(item.d);
      require(reference > 1e-3,"the self-interaction of a pair is too small to be a check");
      rotated.relative(energy(rotate(item.d)),reference);
      rotated.relative(energy({-item.d[0],-item.d[1],-item.d[2]}),reference);
    }
    recovered.finish();
    elementwise.finish();
    separated.finish();
    swapped.finish();
    rotated.finish();
    symmetric.finish();

    expect_throw("coinciding sites",[&] { solver.pair(tables[0],tables[1],{0.0,0.0,0.0}); });
    const ReciprocalGrid other = ReciprocalGrid::uniform(24.0,120);
    const LriSolver mismatched(other,maximum_orbital,maximum_abf);
    expect_throw("a table of another reciprocal grid",[&] { mismatched.metric(tables[0],tables[1],{1.0,0.0,0.0}); });
    const LriSolver small(reciprocal,0,maximum_abf);
    expect_throw("a table beyond the angular momenta of the solver",[&] { small.onsite(tables[0]); });
    expect_throw("angular momenta beyond the Gaunt table",[&] { LriSolver(reciprocal,5,10); });
  }

  // 3. The solver of equation (7).
  {
    // A well conditioned matrix: Cholesky.
    const std::size_t n = 12, m = 3;
    std::vector<double> root(n*n), g(n*n,0.0), x(n*m), b(n*m,0.0);
    for (std::size_t i = 0; i < root.size(); ++i) root[i] = pseudo(i+5);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t k = 0; k < n; ++k) g[i*n+j] += root[i*n+k]*root[j*n+k];
        if (i == j) g[i*n+j] += 1.0+static_cast<double>(i);
      }
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = pseudo(i+900);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < m; ++k) b[i*m+k] += g[i*n+j]*x[j*m+k];
    LriReport report;
    solve_metric(g,n,b,{},&report);
    Worst cholesky("solution of a positive definite system",1e-12);
    for (std::size_t i = 0; i < x.size(); ++i) cholesky.absolute(b[i],x[i]);
    cholesky.finish();
    require(report.systems == 1 && report.eigen_decompositions == 0 && report.dropped == 0 &&
            report.smallest_pivot > 0.0 && report.smallest_pivot <= 1.0,"wrong report of a Cholesky solution");

    // A matrix of rank 8 out of 12: the eigen-decomposition leaves four
    // eigenvectors out, and the solution still satisfies the equations.
    const std::size_t rank = 8;
    std::vector<double> singular(n*n,0.0), rhs(n*m,0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < rank; ++k) singular[i*n+j] += root[i*n+k]*root[j*n+k];
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < m; ++k) rhs[i*m+k] += singular[i*n+j]*x[j*m+k];
    std::vector<double> solution = rhs;
    solve_metric(singular,n,solution,{},&report);
    require(report.systems == 2 && report.eigen_decompositions == 1 && report.dropped == static_cast<long>(n-rank),
            "wrong report of a solution by eigen-decomposition");
    Worst residual("equations of a singular system after the eigen-decomposition",1e-10);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t k = 0; k < m; ++k) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += singular[i*n+j]*solution[j*m+k];
        residual.absolute(sum,rhs[i*m+k]);
      }
    residual.finish();

    // With both tolerances at zero the Cholesky factorization of the first
    // matrix is used whatever its pivots are; a threshold of 0.999... on the
    // pivot forces the eigen-decomposition, which gives the same solution.
    std::vector<double> again(n*m,0.0);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t k = 0; k < m; ++k) again[i*m+k] += g[i*n+j]*x[j*m+k];
    LriReport forced;
    solve_metric(g,n,again,LriOptions{0.999999,0.0},&forced);
    require(forced.eigen_decompositions == 1 && forced.dropped == 0,"the eigen-decomposition was not used");
    Worst eigen("solution of the positive definite system by eigen-decomposition",1e-12);
    for (std::size_t i = 0; i < x.size(); ++i) eigen.absolute(again[i],x[i]);
    eigen.finish();

    LriReport sum;
    sum.merge(report);
    sum.merge(forced);
    require(sum.systems == 3 && sum.eigen_decompositions == 2 && sum.dropped == static_cast<long>(n-rank) &&
            sum.smallest_pivot == report.smallest_pivot,"LriReport::merge is wrong");

    std::vector<double> bad = g;
    bad[3*n+3] = 0.0;
    std::vector<double> scratch(n*m,1.0);
    expect_throw("a diagonal element of zero",[&] { solve_metric(bad,n,scratch); });
    expect_throw("a matrix of the wrong size",[&] { solve_metric(std::span<const double>(g).first(n*n-1),n,scratch); });
    expect_throw("right-hand sides of the wrong size",[&] { solve_metric(g,n,std::span<double>(scratch).first(n*m-1)); });
    expect_throw("a pivot tolerance of one",[&] { solve_metric(g,n,scratch,LriOptions{1.0,0.0}); });
    expect_throw("a negative eigenvalue threshold",[&] { solve_metric(g,n,scratch,LriOptions{0.0,-1.0}); });
  }
}

// ---------------------------------------------------------------------------

void lri_convergence(const std::string& path)
{
  const OverlapFixture fixture = read_overlap_fixture(path);
  const ReciprocalGrid reciprocal = ReciprocalGrid::uniform(fixture.kmax,fixture.radial_intervals);
  const std::vector<double> thresholds{1e-2,1e-3,1e-4,1e-5,1e-6};
  const double smallest = thresholds.back();

  // Tables with the largest auxiliary basis; the ABFs of a larger threshold
  // are a subset of it, so every smaller system is a part of the largest one.
  struct Species {
    std::string name;
    AuxiliaryBasis basis;
    std::vector<RadialFunction> orbitals;
    RadialGrid grid;
  };
  std::vector<Species> species;
  std::vector<SpeciesTable> tables;
  int maximum_orbital = 0, maximum_abf = 0;
  for (const FixtureSpecies& item : fixture.species) {
    const RadialGrid grid{0.0,item.cutoff/fixture.radial_intervals,fixture.radial_intervals};
    std::vector<RadialFunction> orbitals = sample_shells(item,grid);
    AuxiliaryBasis basis = make_auxiliary_basis(grid,orbitals,smallest);
    tables.emplace_back(grid,orbitals,basis.functions,reciprocal);
    maximum_orbital = std::max(maximum_orbital,tables.back().maximum_orbital_angular());
    maximum_abf = std::max(maximum_abf,tables.back().maximum_abf_angular());
    species.push_back({item.name,std::move(basis),std::move(orbitals),grid});
  }
  const LriSolver solver(reciprocal,maximum_orbital,maximum_abf);

  // Components of the ABFs that a threshold keeps.
  const auto kept = [&](std::size_t s, double threshold, std::size_t shift) {
    std::vector<std::size_t> list;
    std::size_t function = 0;
    for (const AbfChannel& channel : species[s].basis.channels)
      for (std::size_t k = 0; k < channel.kept; ++k, ++function) {
        if (!(channel.eigenvalues[k] > threshold*channel.eigenvalues[0])) continue;
        for (std::size_t i = tables[s].abf_offsets()[function]; i < tables[s].abf_offsets()[function+1]; ++i)
          list.push_back(shift+i);
      }
    return list;
  };

  // One atom: the sum over all products of the self-interaction of the fit,
  // against the same sum for the products themselves, which radial integrals
  // give exactly.
  std::printf("self-interaction of the expanded products, summed over all pairs of PAOs\n");
  std::printf("one atom:   threshold   ABFs   sum (fit|g|fit)        1 - fit/exact\n");
  Worst complete("one atom, all components kept: fit against exact (relative)",1e-9);
  for (std::size_t s = 0; s < species.size(); ++s) {
    const SpeciesTable& table = tables[s];
    const RealGauntTable gaunt(table.maximum_orbital_angular(),table.maximum_orbital_angular());
    double exact = 0.0;
    for (std::size_t k = 0; k < species[s].orbitals.size(); ++k)
      for (std::size_t j = 0; j <= k; ++j) {
        const RadialFunction& fk = species[s].orbitals[k];
        const RadialFunction& fj = species[s].orbitals[j];
        RadialFunction product{0,std::vector<double>(fk.values.size())};
        for (std::size_t i = 0; i < product.values.size(); ++i) product.values[i] = fk.values[i]*fj.values[i];
        for (int order = std::abs(fk.angular-fj.angular); order <= fk.angular+fj.angular; order += 2) {
          product.angular = order;
          const RadialFunction potential = coulomb_potential(species[s].grid,product);
          double weight = 0.0;
          for (const double theta : gaunt.block(fk.angular,fj.angular,order)) weight += theta*theta;
          exact += (k == j ? 1.0 : 2.0)*weight*radial_integral(species[s].grid,product.values,potential.values);
        }
      }
    const std::size_t n = table.abf_count();
    const std::vector<double> g(table.onsite_metric().begin(),table.onsite_metric().end());
    const std::vector<double> b(table.onsite_right_hand_side().begin(),table.onsite_right_hand_side().end());
    double previous = 0.0;
    bool all_kept = true;
    for (const AbfChannel& channel : species[s].basis.channels) all_kept = all_kept && channel.kept == channel.eigenvalues.size();
    for (const double threshold : thresholds) {
      const std::vector<std::size_t> keep = kept(s,threshold,0);
      std::vector<double> gs, bs;
      restrict_system(g,b,n,keep,gs,bs);
      std::vector<double> x = bs;
      solve_metric(gs,keep.size(),x,solver.options());
      const double fit = dot(bs,x);
      std::printf("  %-8s  %.0e   %5zu   %.12f   %10.3e\n",species[s].name.c_str(),threshold,keep.size(),fit,1.0-fit/exact);
      require(fit >= previous*(1.0-1e-10),"the self-interaction of one atom decreased with more ABFs");
      require(fit <= exact*(1.0+1e-9),"the self-interaction of the fit exceeds the exact value");
      previous = fit;
      if (threshold == smallest && all_kept) complete.relative(fit,exact);
    }
  }
  complete.finish();

  // Two atoms at bond lengths (bohr).
  struct Pair { std::size_t a, c; double distance; };
  const std::vector<Pair> pairs{{0,0,4.44},{1,1,2.90},{1,2,2.06},{2,2,1.40},{0,1,3.56},{0,2,2.80},{0,0,7.26}};
  std::printf("two atoms:            d    threshold   ABFs   sum (fit|g|fit)      increase to the next   smallest pivot\n");
  long increases = 0;
  for (const Pair& pair : pairs) {
    const SpeciesTable& a = tables[pair.a];
    const SpeciesTable& c = tables[pair.c];
    const std::size_t n = a.abf_count()+c.abf_count();
    const Vec d{0.36*pair.distance,-0.48*pair.distance,0.8*pair.distance};
    const std::vector<double> g = solver.metric(a,c,d), b = solver.right_hand_side(a,c,d);
    std::vector<double> fits, pivots;
    std::vector<std::size_t> counts;
    for (const double threshold : thresholds) {
      std::vector<std::size_t> keep = kept(pair.a,threshold,0);
      const std::vector<std::size_t> other = kept(pair.c,threshold,a.abf_count());
      keep.insert(keep.end(),other.begin(),other.end());
      std::vector<double> gs, bs;
      restrict_system(g,b,n,keep,gs,bs);
      std::vector<double> x = bs;
      LriReport report;
      solve_metric(gs,keep.size(),x,solver.options(),&report);
      require(report.eigen_decompositions == 0,"a pair of the record needed the eigen-decomposition");
      fits.push_back(dot(bs,x));
      pivots.push_back(report.smallest_pivot);
      counts.push_back(keep.size());
    }
    for (std::size_t t = 0; t < thresholds.size(); ++t) {
      const double increase = t+1 < fits.size() ? (fits[t+1]-fits[t])/fits.back() : 0.0;
      std::printf("  %-6s %-6s  %5.2f    %.0e   %5zu   %.12f   %10.3e             %.2e\n",
                  species[pair.a].name.c_str(),species[pair.c].name.c_str(),pair.distance,thresholds[t],counts[t],
                  fits[t],increase,pivots[t]);
      if (t+1 < fits.size()) {
        require(fits[t+1] >= fits[t]*(1.0-1e-10),"the self-interaction of a pair decreased with more ABFs");
        ++increases;
      }
    }
    // The steps shrink: the last one is smaller than the first one.
    require(fits.back()-fits[fits.size()-2] <= fits[1]-fits[0],"the self-interaction of a pair does not converge");
  }
  require(increases >= 20,"too few comparisons in the record");
}

} // namespace exx_test
