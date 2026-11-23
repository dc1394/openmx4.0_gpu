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
    else throw std::runtime_error("unknown exx test case: "+name);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr,"%s\n",e.what());
    return 1;
  }
}
