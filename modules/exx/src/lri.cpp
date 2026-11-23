#include <openmx/exx/lri.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace openmx::exx {
namespace {

constexpr int orbital_angular_limit = 4;
constexpr int abf_angular_limit = 8;

std::vector<std::size_t> component_offsets(std::span<const RadialFunction> functions)
{
  std::vector<std::size_t> offsets(functions.size()+1,0);
  for (std::size_t i = 0; i < functions.size(); ++i)
    offsets[i+1] = offsets[i]+static_cast<std::size_t>(2*functions[i].angular+1);
  return offsets;
}

int validate(const RadialGrid& grid, std::span<const RadialFunction> functions, int maximum, const char* what)
{
  if (functions.empty()) throw std::invalid_argument(std::string("a species needs at least one ")+what);
  int largest = 0;
  for (const RadialFunction& f : functions) {
    if (f.angular < 0 || f.angular > maximum)
      throw std::invalid_argument(std::string("the angular momentum of an ")+what+" is out of range");
    if (f.values.size() != grid.size())
      throw std::invalid_argument(std::string("an ")+what+" does not match the radial grid");
    for (const double value : f.values)
      if (!std::isfinite(value)) throw std::invalid_argument(std::string("an ")+what+" has a value that is not finite");
    largest = std::max(largest,f.angular);
  }
  return largest;
}

// Cyclic Jacobi diagonalization of a symmetric matrix (n x n, row-major, both
// triangles).  The matrix is overwritten; values[k] is an eigenvalue and
// vectors[i*n + k] component i of its eigenvector.
void diagonalize(std::vector<double>& a, std::size_t n, std::vector<double>& values, std::vector<double>& vectors)
{
  vectors.assign(n*n,0.0);
  double scale = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    vectors[i*n+i] = 1.0;
    scale = std::max(scale,std::fabs(a[i*n+i]));
  }
  const double tolerance = 4.0*std::numeric_limits<double>::epsilon();
  const double floor = std::numeric_limits<double>::epsilon()*scale;
  constexpr int maximum_sweeps = 100;
  for (int sweep = 0; sweep < maximum_sweeps; ++sweep) {
    bool rotated = false;
    for (std::size_t p = 0; p+1 < n; ++p)
      for (std::size_t q = p+1; q < n; ++q) {
        const double apq = a[p*n+q], app = a[p*n+p], aqq = a[q*n+q];
        if (std::fabs(apq) <= tolerance*std::max(std::sqrt(std::fabs(app*aqq)),floor)) continue;
        rotated = true;
        const double zeta = (aqq-app)/(2.0*apq);
        const double t = std::copysign(1.0,zeta)/(std::fabs(zeta)+std::sqrt(1.0+zeta*zeta));
        const double c = 1.0/std::sqrt(1.0+t*t), s = c*t;
        for (std::size_t k = 0; k < n; ++k) {
          const double akp = a[k*n+p], akq = a[k*n+q];
          a[k*n+p] = c*akp-s*akq;
          a[k*n+q] = s*akp+c*akq;
        }
        double* row_p = a.data()+p*n;
        double* row_q = a.data()+q*n;
        for (std::size_t k = 0; k < n; ++k) {
          const double apk = row_p[k], aqk = row_q[k];
          row_p[k] = c*apk-s*aqk;
          row_q[k] = s*apk+c*aqk;
        }
        for (std::size_t k = 0; k < n; ++k) {
          const double vkp = vectors[k*n+p], vkq = vectors[k*n+q];
          vectors[k*n+p] = c*vkp-s*vkq;
          vectors[k*n+q] = s*vkp+c*vkq;
        }
      }
    if (!rotated) {
      values.resize(n);
      for (std::size_t i = 0; i < n; ++i) values[i] = a[i*n+i];
      return;
    }
  }
  throw std::runtime_error("the diagonalization of the metric of the local RI did not converge");
}

} // namespace

SpeciesTable::SpeciesTable(const RadialGrid& grid, std::vector<RadialFunction> orbitals,
    std::vector<RadialFunction> abfs, const ReciprocalGrid& reciprocal)
  : grid_(grid), reciprocal_size_(reciprocal.size()), orbitals_(std::move(orbitals)), abfs_(std::move(abfs))
{
  check(grid_);
  maximum_orbital_ = validate(grid_,orbitals_,orbital_angular_limit,"orbital");
  maximum_abf_ = validate(grid_,abfs_,abf_angular_limit,"auxiliary function");
  orbital_offsets_ = component_offsets(orbitals_);
  abf_offsets_ = component_offsets(abfs_);
  const std::size_t u = orbital_count(), np = abf_count();

  potentials_.reserve(abfs_.size());
  for (const RadialFunction& f : abfs_) potentials_.push_back(coulomb_potential(grid_,f));

  // G_PQ on one atom: int r^2 f_P w_Q dr for equal l and m.  The two orders
  // of P and Q differ by the error of the quadrature; their mean is stored.
  onsite_metric_.assign(np*np,0.0);
  for (std::size_t p = 0; p < abfs_.size(); ++p)
    for (std::size_t q = 0; q <= p; ++q) {
      if (abfs_[p].angular != abfs_[q].angular) continue;
      const double value = 0.5*(radial_integral(grid_,abfs_[p].values,potentials_[q].values)+
                                radial_integral(grid_,abfs_[q].values,potentials_[p].values));
      for (std::size_t m = 0; m < static_cast<std::size_t>(2*abfs_[p].angular+1); ++m) {
        const std::size_t row = abf_offsets_[p]+m, column = abf_offsets_[q]+m;
        onsite_metric_[row*np+column] = value;
        onsite_metric_[column*np+row] = value;
      }
    }

  orbital_transforms_ = bessel_transform(grid_,orbitals_,reciprocal);
  abf_transforms_ = bessel_transform(grid_,abfs_,reciprocal);

  // The products phi_k w_P with every order L1 of their expansion in
  // spherical harmonics: |l_k - L_P| <= L1 <= l_k + L_P in steps of two.
  {
    std::vector<RadialFunction> radial;
    for (std::size_t k = 0; k < orbitals_.size(); ++k)
      for (std::size_t p = 0; p < abfs_.size(); ++p) {
        const int lk = orbitals_[k].angular, lp = abfs_[p].angular;
        std::vector<double> product(grid_.size());
        for (std::size_t i = 0; i < grid_.size(); ++i) product[i] = orbitals_[k].values[i]*potentials_[p].values[i];
        for (int order = std::abs(lk-lp); order <= lk+lp; order += 2) {
          radial.push_back({order,product});
          products_.push_back({k,p,{}});
        }
      }
    std::vector<ReciprocalFunction> transforms = bessel_transform(grid_,radial,reciprocal);
    for (std::size_t i = 0; i < products_.size(); ++i) products_[i].transform = std::move(transforms[i]);
  }

  // b^P_{mu lambda} on one atom: the Coulomb integral of the radial product
  // g = phi_k phi_j with f_P, times the Gaunt coefficient of the three real
  // spherical harmonics.  As for the metric, the radial integral is the mean
  // of int r^2 g w_P dr and int r^2 f_P w_g dr; a product inside the span of
  // the ABFs is then expanded exactly.
  onsite_rhs_.assign(np*u*u,0.0);
  const RealGauntTable gaunt(maximum_orbital_,maximum_orbital_);
  for (std::size_t k = 0; k < orbitals_.size(); ++k)
    for (std::size_t j = 0; j <= k; ++j) {
      const int lk = orbitals_[k].angular, lj = orbitals_[j].angular;
      RadialFunction product{0,std::vector<double>(grid_.size())};
      for (std::size_t i = 0; i < grid_.size(); ++i) product.values[i] = orbitals_[k].values[i]*orbitals_[j].values[i];
      for (int order = std::abs(lk-lj); order <= lk+lj && order <= maximum_abf_; order += 2) {
        bool present = false;
        for (const RadialFunction& f : abfs_) present = present || f.angular == order;
        if (!present) continue;
        product.angular = order;
        const RadialFunction potential = coulomb_potential(grid_,product);
        const std::span<const double> theta = gaunt.block(lk,lj,order);
        for (std::size_t p = 0; p < abfs_.size(); ++p) {
          if (abfs_[p].angular != order) continue;
          const double radial = 0.5*(radial_integral(grid_,product.values,potentials_[p].values)+
                                     radial_integral(grid_,abfs_[p].values,potential.values));
          for (int rk = 0; rk <= 2*lk; ++rk)
            for (int rj = 0; rj <= 2*lj; ++rj)
              for (int rp = 0; rp <= 2*order; ++rp) {
                const std::size_t row = abf_offsets_[p]+static_cast<std::size_t>(rp);
                const std::size_t mu = orbital_offsets_[k]+static_cast<std::size_t>(rk);
                const std::size_t lambda = orbital_offsets_[j]+static_cast<std::size_t>(rj);
                const double value = radial*theta[static_cast<std::size_t>((rk*(2*lj+1)+rj)*(2*order+1)+rp)];
                onsite_rhs_[(row*u+mu)*u+lambda] = value;
                onsite_rhs_[(row*u+lambda)*u+mu] = value;
              }
        }
      }
    }
}

void LriReport::merge(const LriReport& other) noexcept
{
  systems += other.systems;
  eigen_decompositions += other.eigen_decompositions;
  dropped += other.dropped;
  smallest_pivot = std::min(smallest_pivot,other.smallest_pivot);
}

void solve_metric(std::span<const double> matrix, std::size_t n, std::span<double> right_hand_sides,
                  const LriOptions& options, LriReport* report)
{
  if (n == 0 || matrix.size() != n*n || right_hand_sides.size()%n != 0)
    throw std::invalid_argument("the metric and the right-hand sides of the local RI do not match");
  if (!(options.pivot_tolerance >= 0.0 && options.pivot_tolerance < 1.0) ||
      !(options.eigenvalue_threshold >= 0.0 && options.eigenvalue_threshold < 1.0))
    throw std::invalid_argument("the tolerances of the local RI must be in [0,1)");
  const std::size_t m = right_hand_sides.size()/n;
  double* b = right_hand_sides.data();

  // Scale to a unit diagonal.
  std::vector<double> scale(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double diagonal = matrix[i*n+i];
    if (!(diagonal > 0.0) || !std::isfinite(diagonal))
      throw std::runtime_error("the metric of the local RI has a diagonal element that is not positive");
    scale[i] = 1.0/std::sqrt(diagonal);
  }
  std::vector<double> g(n*n);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      const double value = 0.5*(matrix[i*n+j]+matrix[j*n+i])*scale[i]*scale[j];
      g[i*n+j] = value;
      g[j*n+i] = value;
    }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < m; ++k) b[i*m+k] *= scale[i];

  // Cholesky factorization row by row: g = L L^T.
  std::vector<double> factor(n*n,0.0);
  bool positive = true;
  double smallest = 1.0;
  for (std::size_t i = 0; i < n && positive; ++i) {
    const double* row_i = factor.data()+i*n;
    for (std::size_t j = 0; j <= i; ++j) {
      const double* row_j = factor.data()+j*n;
      double sum = g[i*n+j];
      for (std::size_t k = 0; k < j; ++k) sum -= row_i[k]*row_j[k];
      if (j < i) {
        factor[i*n+j] = sum/row_j[j];
      } else {
        if (!(sum > options.pivot_tolerance)) { positive = false; break; }
        smallest = std::min(smallest,sum);
        factor[i*n+i] = std::sqrt(sum);
      }
    }
  }

  LriReport local;
  local.systems = 1;
  if (positive) {
    local.smallest_pivot = smallest;
    // L Y = B, then L^T X = Y; a row of B holds all right-hand sides.
    for (std::size_t i = 0; i < n; ++i) {
      double* row = b+i*m;
      for (std::size_t k = 0; k < i; ++k) {
        const double l = factor[i*n+k];
        if (l == 0.0) continue;
        const double* other = b+k*m;
        for (std::size_t r = 0; r < m; ++r) row[r] -= l*other[r];
      }
      const double inverse = 1.0/factor[i*n+i];
      for (std::size_t r = 0; r < m; ++r) row[r] *= inverse;
    }
    for (std::size_t i = n; i-- > 0;) {
      double* row = b+i*m;
      for (std::size_t k = i+1; k < n; ++k) {
        const double l = factor[k*n+i];
        if (l == 0.0) continue;
        const double* other = b+k*m;
        for (std::size_t r = 0; r < m; ++r) row[r] -= l*other[r];
      }
      const double inverse = 1.0/factor[i*n+i];
      for (std::size_t r = 0; r < m; ++r) row[r] *= inverse;
    }
  } else {
    // X = sum_k v_k (v_k^T B)/lambda_k over the eigenvectors that are kept.
    local.eigen_decompositions = 1;
    std::vector<double> values, vectors;
    diagonalize(g,n,values,vectors);
    std::vector<double> projected(n*m,0.0);
    for (std::size_t i = 0; i < n; ++i) {
      const double* row = b+i*m;
      for (std::size_t k = 0; k < n; ++k) {
        if (!(values[k] > options.eigenvalue_threshold)) continue;
        const double v = vectors[i*n+k];
        double* target = projected.data()+k*m;
        for (std::size_t r = 0; r < m; ++r) target[r] += v*row[r];
      }
    }
    for (std::size_t k = 0; k < n; ++k) {
      if (!(values[k] > options.eigenvalue_threshold)) { ++local.dropped; continue; }
      const double inverse = 1.0/values[k];
      double* target = projected.data()+k*m;
      for (std::size_t r = 0; r < m; ++r) target[r] *= inverse;
    }
    for (std::size_t i = 0; i < n; ++i) {
      double* row = b+i*m;
      std::fill(row,row+m,0.0);
      for (std::size_t k = 0; k < n; ++k) {
        if (!(values[k] > options.eigenvalue_threshold)) continue;
        const double v = vectors[i*n+k];
        const double* source = projected.data()+k*m;
        for (std::size_t r = 0; r < m; ++r) row[r] += v*source[r];
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < m; ++k) b[i*m+k] *= scale[i];
  if (report) report->merge(local);
}

LriSolver::LriSolver(const ReciprocalGrid& reciprocal, int maximum_orbital, int maximum_abf,
    const LriOptions& options)
  : options_(options),
    maximum_orbital_(maximum_orbital < 0 || maximum_abf < 0
                     ? throw std::invalid_argument("the angular momenta of the local RI must not be negative")
                     : maximum_orbital),
    maximum_abf_(maximum_abf),
    overlap_(reciprocal,overlap_kernel(),maximum_orbital+maximum_abf,maximum_orbital),
    coulomb_(reciprocal,coulomb_kernel(),maximum_abf,maximum_abf),
    gaunt_(maximum_orbital,maximum_abf)
{
}

void LriSolver::check(const SpeciesTable& table) const
{
  if (table.reciprocal_size() != overlap_.grid().size())
    throw std::invalid_argument("a species table was made on another reciprocal grid");
  if (table.maximum_orbital_angular() > maximum_orbital_ || table.maximum_abf_angular() > maximum_abf_)
    throw std::invalid_argument("a species table exceeds the angular momenta of the solver of the local RI");
}

std::vector<double> LriSolver::onsite(const SpeciesTable& a, LriReport* report) const
{
  check(a);
  const std::span<const double> b = a.onsite_right_hand_side();
  std::vector<double> x(b.begin(),b.end());
  solve_metric(a.onsite_metric(),a.abf_count(),x,options_,report);
  for (double& value : x) value *= 0.5;
  return x;
}

std::vector<double> LriSolver::metric(const SpeciesTable& a, const SpeciesTable& c,
    const std::array<double,3>& displacement) const
{
  check(a);
  check(c);
  const std::size_t na = a.abf_count(), nc = c.abf_count(), n = na+nc;
  std::vector<double> g(n*n);
  const std::span<const double> ga = a.onsite_metric(), gc = c.onsite_metric();
  for (std::size_t i = 0; i < na; ++i)
    for (std::size_t j = 0; j < na; ++j) g[i*n+j] = ga[i*na+j];
  for (std::size_t i = 0; i < nc; ++i)
    for (std::size_t j = 0; j < nc; ++j) g[(na+i)*n+na+j] = gc[i*nc+j];
  const std::vector<double> off =
      two_center_matrix(coulomb_.pair(displacement),a.abf_transforms(),c.abf_transforms());
  for (std::size_t i = 0; i < na; ++i)
    for (std::size_t j = 0; j < nc; ++j) {
      g[i*n+na+j] = off[i*nc+j];
      g[(na+j)*n+i] = off[i*nc+j];
    }
  return g;
}

std::vector<double> LriSolver::right_hand_side(const SpeciesTable& a, const SpeciesTable& c,
    const std::array<double,3>& displacement) const
{
  check(a);
  check(c);
  const std::size_t na = a.abf_count(), nc = c.abf_count(), ua = a.orbital_count(), uc = c.orbital_count();
  std::vector<double> b((na+nc)*ua*uc,0.0);
  std::vector<double> block;

  // P on the first site: the overlap of (phi_mu W_P), centred there, with
  // phi_lambda on the second site.
  const TwoCenterIntegrator::Pair forward = overlap_.pair(displacement);
  for (const SpeciesTable::Product& product : a.products()) {
    const int lk = a.orbitals()[product.orbital].angular, lp = a.abfs()[product.abf].angular;
    const int order = product.transform.angular;
    const std::span<const double> theta = gaunt_.block(lk,lp,order);
    const std::size_t mu0 = a.orbital_offsets()[product.orbital], p0 = a.abf_offsets()[product.abf];
    for (std::size_t q = 0; q < c.orbitals().size(); ++q) {
      const int lq = c.orbitals()[q].angular;
      const std::size_t lambda0 = c.orbital_offsets()[q];
      block.resize(static_cast<std::size_t>((2*order+1)*(2*lq+1)));
      forward.block(product.transform,c.orbital_transforms()[q],block);
      for (int rk = 0; rk <= 2*lk; ++rk)
        for (int rp = 0; rp <= 2*lp; ++rp) {
          double* target = b.data()+((p0+static_cast<std::size_t>(rp))*ua+mu0+static_cast<std::size_t>(rk))*uc+lambda0;
          const double* t = theta.data()+static_cast<std::size_t>((rk*(2*lp+1)+rp)*(2*order+1));
          for (int r1 = 0; r1 <= 2*order; ++r1) {
            if (t[r1] == 0.0) continue;
            const double* source = block.data()+static_cast<std::size_t>(r1*(2*lq+1));
            for (int rq = 0; rq <= 2*lq; ++rq) target[rq] += t[r1]*source[rq];
          }
        }
    }
  }

  // Q on the second site: the overlap of (phi_lambda W_Q), centred there,
  // with phi_mu on the first site; the displacement is reversed.
  const TwoCenterIntegrator::Pair backward =
      overlap_.pair({-displacement[0],-displacement[1],-displacement[2]});
  for (const SpeciesTable::Product& product : c.products()) {
    const int lk = c.orbitals()[product.orbital].angular, lq = c.abfs()[product.abf].angular;
    const int order = product.transform.angular;
    const std::span<const double> theta = gaunt_.block(lk,lq,order);
    const std::size_t lambda0 = c.orbital_offsets()[product.orbital], q0 = na+c.abf_offsets()[product.abf];
    for (std::size_t p = 0; p < a.orbitals().size(); ++p) {
      const int lp = a.orbitals()[p].angular;
      const std::size_t mu0 = a.orbital_offsets()[p];
      block.resize(static_cast<std::size_t>((2*order+1)*(2*lp+1)));
      backward.block(product.transform,a.orbital_transforms()[p],block);
      for (int rk = 0; rk <= 2*lk; ++rk)
        for (int rq = 0; rq <= 2*lq; ++rq) {
          const double* t = theta.data()+static_cast<std::size_t>((rk*(2*lq+1)+rq)*(2*order+1));
          for (int r1 = 0; r1 <= 2*order; ++r1) {
            if (t[r1] == 0.0) continue;
            const double* source = block.data()+static_cast<std::size_t>(r1*(2*lp+1));
            for (int rp = 0; rp <= 2*lp; ++rp)
              b[((q0+static_cast<std::size_t>(rq))*ua+mu0+static_cast<std::size_t>(rp))*uc+lambda0+
                static_cast<std::size_t>(rk)] += t[r1]*source[rp];
          }
        }
    }
  }
  return b;
}

PairCoefficients LriSolver::pair(const SpeciesTable& a, const SpeciesTable& c,
    const std::array<double,3>& displacement, LriReport* report) const
{
  if (displacement[0] == 0.0 && displacement[1] == 0.0 && displacement[2] == 0.0)
    throw std::invalid_argument("the two sites of a pair of the local RI coincide");
  const std::size_t na = a.abf_count(), nc = c.abf_count(), ua = a.orbital_count(), uc = c.orbital_count();
  const std::vector<double> g = metric(a,c,displacement);
  std::vector<double> b = right_hand_side(a,c,displacement);
  solve_metric(g,na+nc,b,options_,report);
  PairCoefficients result;
  result.first.assign(b.begin(),b.begin()+static_cast<std::ptrdiff_t>(na*ua*uc));
  result.second.resize(nc*uc*ua);
  for (std::size_t q = 0; q < nc; ++q)
    for (std::size_t mu = 0; mu < ua; ++mu)
      for (std::size_t lambda = 0; lambda < uc; ++lambda)
        result.second[(q*uc+lambda)*ua+mu] = b[((na+q)*ua+mu)*uc+lambda];
  return result;
}

} // namespace openmx::exx
