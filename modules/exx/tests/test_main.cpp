// Entry point of the exx unit tests: "openmx_exx_tests <case> [fixture]".
#include "test_support.hpp"

#include <cstdio>
#include <exception>
#include <string>

int main(int argc, char** argv)
{
  try {
    const std::string name = argc > 1 ? argv[1] : "";
    const std::string fixture = argc > 2 ? argv[2] : "";
    if (name == "kernel-analytic") exx_test::kernel_analytic(fixture);
    else if (name == "two-center-gaussian") exx_test::two_center_gaussian();
    else if (name == "two-center-symmetry") exx_test::two_center_symmetry();
    else if (name == "two-center-gradient") exx_test::two_center_gradient();
    else if (name == "two-center-overlap") exx_test::two_center_overlap(fixture);
    else if (name == "abf-pca") exx_test::abf_pca(fixture);
    else if (name == "three-center-gaussian") exx_test::three_center_gaussian();
    else if (name == "lri-identity") exx_test::lri_identity(fixture);
    else if (name == "lri-convergence") exx_test::lri_convergence(fixture);
    else if (name == "system-neighbours") exx_test::system_neighbours(fixture);
    else if (name == "system-blocks") exx_test::system_blocks(fixture);
    else if (name == "dense-backend") exx_test::dense_backend();
    else if (name == "exchange-fixture") exx_test::exchange_fixture(fixture);
    else if (name == "exchange-layout") exx_test::exchange_layout(fixture);
    else if (name == "exchange-gamma") exx_test::exchange_gamma(fixture);
    else if (name == "exchange-gradient") exx_test::exchange_gradient(fixture);
    else if (name == "exchange-backend") exx_test::exchange_backend(fixture);
    else if (name == "exchange-system") exx_test::exchange_system(fixture);
    else throw std::runtime_error("unknown exx test case: "+name);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what());
    return 1;
  }
}
