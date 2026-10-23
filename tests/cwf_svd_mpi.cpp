// Real ScaLAPACK integration: full-rank rectangular real/complex projections.
// This checks numerical invariants independently of the SVD implementation.
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

using Complex = std::complex<double>;
extern "C" {
void Cblacs_get(int, int, int*);
void Cblacs_gridinit(int*, const char*, int, int);
void Cblacs_gridinfo(int, int*, int*, int*, int*);
void Cblacs_gridexit(int);
int numroc_(const int*, const int*, const int*, const int*, const int*);
void descinit_(int*, const int*, const int*, const int*, const int*, const int*,
               const int*, const int*, const int*, int*);
void pdgesvd_(const char*, const char*, const int*, const int*, double*, const int*,
    const int*, const int*, double*, double*, const int*, const int*, const int*,
    double*, const int*, const int*, const int*, double*, const int*, int*);
void pzgesvd_(const char*, const char*, const int*, const int*, Complex*, const int*,
    const int*, const int*, double*, Complex*, const int*, const int*, const int*,
    Complex*, const int*, const int*, const int*, Complex*, const int*, double*, int*);
void pdgemm_(const char*, const char*, const int*, const int*, const int*, const double*,
    const double*, const int*, const int*, const int*, const double*, const int*,
    const int*, const int*, const double*, double*, const int*, const int*, const int*);
void pzgemm_(const char*, const char*, const int*, const int*, const int*, const Complex*,
    const Complex*, const int*, const int*, const int*, const Complex*, const int*,
    const int*, const int*, const Complex*, Complex*, const int*, const int*, const int*);
}

namespace {
constexpr int zero = 0, one = 1, block = 2;
struct Grid {
  int context = 0, rows, columns, row, column;
  explicit Grid(int size) {
    columns = static_cast<int>(std::sqrt(size));
    while (size % columns) --columns;
    rows = size / columns;
    Cblacs_get(-1, 0, &context);
    Cblacs_gridinit(&context, "Row", rows, columns);
    Cblacs_gridinfo(context, &rows, &columns, &row, &column);
  }
  Grid(const Grid&) = delete;
  Grid& operator=(const Grid&) = delete;
  ~Grid() { Cblacs_gridexit(context); }
};
int global_index(int local, int process, int processes) {
  return ((local / block) * processes + process) * block + local % block;
}
template<class T> struct Matrix {
  int m, n, rows, columns, leading;
  std::array<int,9> descriptor{};
  std::vector<T> values;
  Matrix(const Grid& grid, int m_, int n_, int padding = 0) : m(m_), n(n_) {
    rows = numroc_(&m, &block, &grid.row, &zero, &grid.rows);
    columns = numroc_(&n, &block, &grid.column, &zero, &grid.columns);
    leading = std::max({1, rows, padding});
    values.resize(static_cast<std::size_t>(leading) * std::max(1, columns));
    int info = 0;
    descinit_(descriptor.data(), &m, &n, &block, &block, &zero, &zero,
              &grid.context, &leading, &info);
    if (info) throw std::runtime_error("invalid test matrix descriptor");
  }
  std::vector<T> gather(const Grid& grid) const {
    std::vector<T> full(static_cast<std::size_t>(m) * n);
    for (int j = 0; j < columns; ++j)
      for (int i = 0; i < rows; ++i)
        full[global_index(j, grid.column, grid.columns) * m +
             global_index(i, grid.row, grid.rows)] = values[j * leading + i];
    static_assert(sizeof(T) == sizeof(double) || sizeof(T) == 2 * sizeof(double),
                  "unexpected scalar representation");
    MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<double*>(full.data()),
                  m * n * static_cast<int>(sizeof(T) / sizeof(double)),
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return full;
  }
};
void svd(Matrix<double>& a, std::vector<double>& s, Matrix<double>& u,
         Matrix<double>& vt, double* work, int length, double*, int& info) {
  pdgesvd_("V", "V", &a.m, &a.n, a.values.data(), &one, &one, a.descriptor.data(),
      s.data(), u.values.data(), &one, &one, u.descriptor.data(),
      vt.values.data(), &one, &one, vt.descriptor.data(), work, &length, &info);
}
void svd(Matrix<Complex>& a, std::vector<double>& s, Matrix<Complex>& u,
         Matrix<Complex>& vt, Complex* work, int length, double* real_work, int& info) {
  pzgesvd_("V", "V", &a.m, &a.n, a.values.data(), &one, &one, a.descriptor.data(),
      s.data(), u.values.data(), &one, &one, u.descriptor.data(),
      vt.values.data(), &one, &one, vt.descriptor.data(), work, &length, real_work, &info);
}
void product(const Matrix<double>& u, const Matrix<double>& vt, Matrix<double>& q) {
  const double alpha = 1, beta = 0;
  pdgemm_("N", "N", &q.m, &q.n, &q.n, &alpha, u.values.data(), &one, &one,
      u.descriptor.data(), vt.values.data(), &one, &one, vt.descriptor.data(),
      &beta, q.values.data(), &one, &one, q.descriptor.data());
}
void product(const Matrix<Complex>& u, const Matrix<Complex>& vt, Matrix<Complex>& q) {
  const Complex alpha(1,0), beta(0,0);
  pzgemm_("N", "N", &q.m, &q.n, &q.n, &alpha, u.values.data(), &one, &one,
      u.descriptor.data(), vt.values.data(), &one, &one, vt.descriptor.data(),
      &beta, q.values.data(), &one, &one, q.descriptor.data());
}
double entry(double real, double) { return real; }
Complex entry(Complex real, double imaginary) { return {real.real(), imaginary}; }
void include(double& error, Complex value) {
  const double magnitude = std::abs(value);
  error = std::isfinite(magnitude) ? std::max(error, magnitude) :
          std::numeric_limits<double>::infinity();
}
double printable(double error) {
  return std::isfinite(error) ? error : std::numeric_limits<double>::max();
}
template<class T> bool check(const Grid& grid, int m, int n, bool padding, int rank, int ranks) {
  Matrix<T> a(grid,m,n), u(grid,m,n), vt(grid,n,n,padding ? a.leading : 0), q(grid,m,n);
  for (int j = 0; j < a.columns; ++j) for (int i = 0; i < a.rows; ++i) {
    const int row = global_index(i,grid.row,grid.rows);
    const int column = global_index(j,grid.column,grid.columns);
    const double real = std::sin(.33*(row+1)*(column+1)) + (row == column ? 2 : 0);
    const double imaginary = .17*std::cos(.21*(row+2)*(column+1));
    a.values[j*a.leading+i] = entry(T(real),imaginary);
  }
  const auto original = a.gather(grid);
  std::vector<double> singular(n), real_work(8192);
  T query{};
  int info = 0;
  svd(a,singular,u,vt,&query,-1,real_work.data(),info);
  const double requested = std::real(query);
  if (info || !std::isfinite(requested) || requested < 1 ||
      requested >= std::numeric_limits<int>::max())
    throw std::runtime_error("invalid SVD workspace query");
  std::vector<T> work(static_cast<int>(requested)+1);
  svd(a,singular,u,vt,work.data(),static_cast<int>(work.size()),real_work.data(),info);
  if (info) throw std::runtime_error("SVD execution failed");
  product(u,vt,q);
  const auto left = u.gather(grid), right = vt.gather(grid), polar = q.gather(grid);
  double u_error = 0, vt_error = 0, q_error = 0, reconstruction = 0, product_error = 0;
  for (double value : singular)
    if (!std::isfinite(value) || value <= 1e-8) throw std::runtime_error("test projection lost full rank");
  for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
    Complex sum_u{}, sum_vt{}, sum_q{};
    for (int k = 0; k < m; ++k) {
      sum_u += std::conj(Complex(left[i*m+k]))*Complex(left[j*m+k]);
      sum_q += std::conj(Complex(polar[i*m+k]))*Complex(polar[j*m+k]);
    }
    for (int k = 0; k < n; ++k)
      sum_vt += Complex(right[k*n+i])*std::conj(Complex(right[k*n+j]));
    include(u_error,sum_u-Complex(i==j));
    include(vt_error,sum_vt-Complex(i==j));
    include(q_error,sum_q-Complex(i==j));
  }
  for (int j = 0; j < n; ++j) for (int i = 0; i < m; ++i) {
    Complex recovered{}, expected_polar{};
    for (int k = 0; k < n; ++k) {
      const Complex value = Complex(left[k*m+i])*Complex(right[j*n+k]);
      recovered += value*singular[k]; expected_polar += value;
    }
    include(reconstruction,recovered-Complex(original[j*m+i]));
    include(product_error,expected_polar-Complex(polar[j*m+i]));
  }
  const double error = std::max({u_error,vt_error,q_error,reconstruction,product_error});
  if (!rank)
    std::printf("{\"type\":\"%s\",\"ranks\":%d,\"m\":%d,\"n\":%d,\"padded_VT\":%d,"
      "\"U_error\":%.17g,\"VT_error\":%.17g,\"Q_error\":%.17g,"
      "\"reconstruction\":%.17g,\"product_error\":%.17g,\"passed\":%s}\n",
      sizeof(T)==sizeof(double) ? "real":"complex",ranks,m,n,int(padding),
      printable(u_error),printable(vt_error),printable(q_error),printable(reconstruction),
      printable(product_error),error<1e-11 ? "true":"false");
  return error < 1e-11;
}
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  int rank = 0, ranks = 0, failed = 0;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  try {
    Grid grid(ranks);
    for (const auto shape : {std::array<int,2>{8,6},std::array<int,2>{7,5}})
      for (bool padded : {false,true}) {
        failed |= !check<double>(grid,shape[0],shape[1],padded,rank,ranks);
        failed |= !check<Complex>(grid,shape[0],shape[1],padded,rank,ranks);
      }
  } catch (const std::exception& failure) {
    std::fprintf(stderr,"SVD regression: %s\n",failure.what()); MPI_Abort(MPI_COMM_WORLD,2);
  }
  MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  MPI_Finalize(); return failed;
}
