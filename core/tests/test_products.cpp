#include <openmx/core/product_storage.hpp>
#include <openmx/core/radial_table.hpp>
#include "openmx_core_pao.h"
#include "openmx_core_vps.h"
#include "openmx_core_product.h"

#include <array>
#include <climits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>

namespace {
using namespace openmx::core;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(double value, double expected, double tolerance = 3e-12) {
  if (!std::isfinite(value) || std::abs(value-expected) > tolerance*(1+std::abs(expected))) {
    std::ostringstream message;
    message << std::setprecision(17) << "product value " << value << ", expected " << expected;
    throw std::runtime_error(message.str());
  }
}
template<class F> void rejects(F&& function) {
  bool caught = false;
  try { function(); } catch (const std::exception&) { caught = true; }
  require(caught,"invalid product input accepted");
}

void storage_test() {
  const int a[]{2,0,1}, b[]{1}, c[]{0,0,0}, d[]{0,1};
  const ProductSpecies basis[]{{a},{b},{c},{d}};
  ProductStorage storage(basis,5);
  const auto& layout = storage.layout(0);
  require(layout.channels().size() == 19 && layout.pair_count() == 7 && layout.orbital_count() == 3,
          "incorrect ragged product layout");
  require(layout.row(0,0,0,0) == 0 && layout.row(0,0,0,1) == 1 && layout.row(0,0,2,0) == 2 &&
          layout.row(0,1,0,0) == 7 && layout.row(0,1,0,1) == 8 && layout.row(0,1,2,0) == 9 &&
          layout.row(2,0,2,0) == 14,"product restart row order changed");
  require(storage.values().size() == 23*7,"inactive product blocks allocated");
  for (std::size_t i = 0; i < storage.values().size(); ++i)
    require(storage.values().data_handle()[i] == 0,"uninitialized product value or padding");
  require(storage.species(1).data_handle() == storage.values().data_handle()+19*7,"species offset mismatch");
  require(storage.species(2).size() == 0,"empty species has rows");
  auto block = storage.block(3,1,0,1,0);
  require(block.extent(0) == 3 && block.extent(1) == 7,"incorrect block mdspan");
  element(block,2,6) = 57;
  near(storage.values().data_handle()[23*7-1],57);
  const auto& fixed = storage;
  static_assert(std::is_const_v<std::remove_reference_t<decltype(element(fixed.block(3,1,0,1,0),0,0))>>);
  rejects([&] { (void)storage.block(0,2,0,0,0); });
  rejects([&] { (void)storage.block(0,0,0,1,0); });
  rejects([&] { (void)storage.block(0,0,2,2,0); });
  rejects([&] { (void)storage.species(4); });
  rejects([] { ProductLayout invalid({}); });
  rejects([] { const int counts[]{-1}; ProductLayout invalid(counts); });
  rejects([] { const int counts[]{INT_MAX,1}; ProductLayout invalid(counts); });
  rejects([] { const int counts[]{46341}; ProductLayout invalid(counts); });
  rejects([] { const int counts[16]{}; ProductLayout invalid(counts); });
  rejects([&] { ProductStorage invalid(basis,std::numeric_limits<std::size_t>::max()); });
  rejects([&] { ProductStorage invalid(basis,std::numeric_limits<std::size_t>::max()-2); });
  rejects([&] { ProductStorage invalid(basis,0); });
  const ProductSpecies empty[]{{c}};
  ProductStorage zero(empty,5);
  require(zero.species(0).size() == 0,"all-empty storage rejected");
}

void check_c_rows(const OpenMXProductShape& shape, const OpenMXProductViews& views) {
  double* beginning = nullptr;
  std::size_t offset = 0;
  require(views.rows != nullptr,"missing species pointers");
  for (int s = 0; s < shape.species; ++s)
    for (int l1 = 0; l1 <= shape.lmax[s]; ++l1) {
      if (!shape.multiplicities[s][l1]) require(!views.rows[s][l1],"empty angular group allocated");
      for (int m1 = 0; m1 < shape.multiplicities[s][l1]; ++m1)
        for (int l2 = 0; l2 <= shape.lmax[s]; ++l2) {
          if (l1 > l2 || !shape.multiplicities[s][l2]) {
            require(!views.rows[s][l1][m1][l2],"inactive product group allocated");
            continue;
          }
          for (int m2 = 0; m2 < shape.multiplicities[s][l2]; ++m2)
            for (int l = 0; l <= 2*l2; ++l) {
              double* row = views.rows[s][l1][m1][l2][m2][l];
              if (!beginning) beginning = row;
              require(row == beginning+offset,"product rows are not contiguous in restart order");
              for (int k = 0; k < shape.wavevectors+2; ++k)
                require(row[k] == 0,"uninitialized C product row or padding");
              offset += shape.wavevectors+2;
            }
        }
    }
}

int independent_entry(int, char**, OpenMXCoreContext* context) {
  int counts[]{1,0,2}; int* rows[]{counts}; const int lmax[]{2};
  const OpenMXProductShape shape{1,7,lmax,rows};
  OpenMXProductViews views{}; char error[256];
  require(openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),error);
  check_c_rows(shape,views);
  views.rows[0][0][0][2][1][4][6] = 123;
  return 0; // Context destruction owns cleanup, without a separate release.
}

int lifecycle_entry(int argc, char** argv, OpenMXCoreContext* context) {
  char error[256];
  const OpenMXPaoShape pao_shape{1,1,1,3,3}; OpenMXPaoViews pao{};
  const OpenMXVpsShape vps_shape{1,1,1,3,3}; OpenMXVpsViews vps{};
  require(openmx_core_pao_allocate(context,&pao_shape,&pao,error,sizeof(error)),error);
  require(openmx_core_vps_allocate(context,&vps_shape,nullptr,&vps,error,sizeof(error)),error);
  pao.orbitals[0][0][0][0] = 73; vps.projectors[0][0][0][0] = 19;
  for (int iteration = 0; iteration < 12; ++iteration) {
    int a[]{1+iteration%3,iteration%2,1}, b[]{0,0,0}, c[]{2,1,0};
    int* counts[]{a,b,c}; const int lmax[]{2,iteration%3,1};
    const OpenMXProductShape shape{3,1+iteration,lmax,counts};
    OpenMXProductViews views{};
    require(openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),error);
    check_c_rows(shape,views);
    views.rows[0][0][0][0][0][0][0] = 91;
    const auto saved = views.rows;
    require(!openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),"live products replaced");
    require(views.rows == saved,"failed allocation changed product views");
    require(openmx_core_run(argc,argv,independent_entry) == 0,"independent products failed");
    near(views.rows[0][0][0][0][0][0][0],91);
    openmx_core_product_release(context); openmx_core_product_release(context);
    near(pao.orbitals[0][0][0][0],73); near(vps.projectors[0][0][0][0],19);
    // Invalid second species after the first layout was constructed.
    b[0] = -1;
    views = {};
    require(!openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),"invalid late species accepted");
    require(!views.rows && error[0],"partial products were published");
  }
  require(!openmx_core_product_allocate(nullptr,nullptr,nullptr,error,1),"null allocation accepted");
  require(error[0] == '\0',"error termination failed");
  int count[]{1}; int* counts[]{count}; const int lmax[]{0};
  const OpenMXProductShape shape{1,1,lmax,counts}; OpenMXProductViews views{};
  require(openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),error);
  openmx_core_pao_release(context); openmx_core_vps_release(context);
  views.rows[0][0][0][0][0][0][0] = 42;
  near(views.rows[0][0][0][0][0][0][0],42);
  openmx_core_finish(context); openmx_core_finish(context);
  require(!openmx_core_product_allocate(context,&shape,&views,error,sizeof(error)),"finished context reused");
  openmx_core_product_release(context); openmx_core_product_release(nullptr);
  return 0;
}

// Independent power series, not the production downward Bessel recurrence.
long double reference_bessel(int l, long double x) {
  long double term = 1;
  for (int j = 1; j <= l; ++j) term *= x/(2*j+1);
  long double sum = term;
  for (int j = 1; j < 80; ++j) {
    term *= -x*x/(2*j*(2*l+2*j+1));
    sum += term;
  }
  return sum;
}
constexpr double nodes[]{-.9602898564975363,-.7966664774136267,-.5255324099163290,-.1834346424956498,
                         .1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
constexpr double weights[]{.1012285362903763,.2223810344533745,.3137066458778873,.3626837833783620,
                           .3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
long double polynomial(int row, long double r) { return (row+1)+(.2-.1*row)*r+.03*(row+1)*r*r; }
double reference(int left, int right, int angular, double k) {
  long double sum = 0;
  for (int i = 0; i < 8; ++i) {
    const long double r = 1+static_cast<long double>(nodes[i]);
    sum += r*r*weights[i]*reference_bessel(angular,k*r)*polynomial(left,r)*polynomial(right,r);
  }
  return static_cast<double>(sum);
}

void quadrature_test() {
  const double zero[]{0}, two[]{2}, duplicate[]{0,0};
  auto grid = gauss_quadrature(0,0,2,0,0,zero,two,zero);
  RadialSamples samples{grid,{{0,0}},Tensor<double,2>({1,1}),{{0,0,0}}};
  samples.values.data()[0] = 2;
  near(transform_host(samples).data()[0],8); // No trapezoidal halving of a Gauss node.
  samples.products[0].right = 1;
  rejects([&] { (void)transform_host(samples); });
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,zero,zero,zero); });
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,duplicate,two,zero); });
  const double repeated_weights[]{1,1};
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,duplicate,repeated_weights,zero); });
  rejects([&] { (void)gauss_quadrature(0,2,0,0,1,zero,two,zero); });
  const double outside[]{1.1}, nan[]{std::numeric_limits<double>::quiet_NaN()};
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,outside,two,zero); });
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,zero,nan,zero); });
  rejects([&] { (void)gauss_quadrature(0,0,2,0,1,zero,two,nan); });
}

int transform_entry(OpenMXCoreContext* context, int device) {
  int counts[]{2,1,1}; int* count_rows[]{counts}; const int lmax[]{2};
  const OpenMXProductShape shape{1,35,lmax,count_rows}; OpenMXProductViews products{};
  const OpenMXPaoShape pao_shape{1,3,2,8,3}; OpenMXPaoViews pao{};
  char error[256];
  require(openmx_core_pao_allocate(context,&pao_shape,&pao,error,sizeof(error)),error);
  const double radii[]{0,.2,.5,.9,1.5,2.1,2.7,3};
  for (int i = 0; i < 8; ++i) pao.radii[0][i] = radii[i];
  int row = 0;
  for (int l = 0; l <= 2; ++l)
    for (int m = 0; m < counts[l]; ++m,++row)
      for (int i = 0; i < 8; ++i) pao.orbitals[0][l][m][i] = polynomial(row,radii[i]);
  std::array<double,35> wavevector_nodes{};
  for (int k = 0; k < 35; ++k) wavevector_nodes[k] = -1+2.*k/34;
  OpenMXGaussRadialGrid grid{8,35,0,2,0,3,nodes,weights,wavevector_nodes.data()};
  for (int iteration = 0; iteration < 2; ++iteration) {
    require(openmx_core_product_allocate(context,&shape,&products,error,sizeof(error)),error);
    OpenMXProductTable table{2,8,counts,pao.radii[0],pao.orbitals[0],products.rows[0]};
    require(openmx_core_transform_products(context,device,&table,&grid,error,sizeof(error)),error);
    // At k=0 the first channel is the exact polynomial integral, independent
    // of both the Bessel reference and the supplied Gauss nodes/weights.
    const long double coefficients[]{1,.2L,.03L};
    long double analytic = 0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        analytic += coefficients[i]*coefficients[j]*std::pow(2.L,i+j+3)/(i+j+3);
    near(table.output[0][0][0][0][0][0],analytic);
    const int first[]{0,2,3};
    for (int l1 = 0; l1 <= 2; ++l1)
      for (int m1 = 0; m1 < counts[l1]; ++m1)
        for (int l2 = l1; l2 <= 2; ++l2)
          for (int m2 = 0; m2 < counts[l2]; ++m2)
            for (int l = 0; l <= 2*l2; ++l) {
              const auto output = table.output[l1][m1][l2][m2][l];
              for (int k = 0; k < 35; ++k)
                near(output[k],reference(first[l1]+m1,first[l2]+m2,l,1.5*(wavevector_nodes[k]+1)));
              near(output[35],0,0); near(output[36],0,0);
            }
    double* last = table.output[2][0][2][0][4];
    table.output[2][0][2][0][4] = nullptr;
    const double before = table.output[0][0][0][0][0][0];
    require(!openmx_core_transform_products(context,device,&table,&grid,error,sizeof(error)),"late missing output accepted");
    near(table.output[0][0][0][0][0][0],before,0);
    table.output[2][0][2][0][4] = last;
    pao.orbitals[0][2][0][7] = std::numeric_limits<double>::quiet_NaN();
    require(!openmx_core_transform_products(context,device,&table,&grid,error,sizeof(error)),"nonfinite last input accepted");
    near(table.output[0][0][0][0][0][0],before,0);
    pao.orbitals[0][2][0][7] = polynomial(3,3);
    // The null-context library path uses the same checked host computation.
    if (device < 0) require(openmx_core_transform_products(nullptr,-1,&table,&grid,error,sizeof(error)),error);
    openmx_core_product_release(context);
  }
  return 0;
}
int host_entry(int, char**, OpenMXCoreContext* context) { return transform_entry(context,-1); }
int execution_entry(int, char**, OpenMXCoreContext* context) { return transform_entry(context,openmx_core_has_accelerator() ? 0 : -1); }
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("specify a product test");
    const std::string name = argv[1];
    if (name == "storage") storage_test();
    else if (name == "quadrature") quadrature_test();
    else if (name == "lifecycle") require(openmx_core_run(argc,argv,lifecycle_entry) == 0,"product lifecycle failed");
    else if (name == "host") require(openmx_core_run(argc,argv,host_entry) == 0,"host product transform failed");
    else if (name == "execution") require(openmx_core_run(argc,argv,execution_entry) == 0,"execution product transform failed");
    else throw std::invalid_argument("unknown product test");
    std::cout << name << ": passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAILED: " << error.what() << '\n';
    return 1;
  }
}
